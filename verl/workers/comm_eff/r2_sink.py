# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cloudflare R2 artifact sink for the comm-eff weight / gradient dumps.

The heavy diagnostic tensors (full per-step weight snapshots, raw per-tick
gradient dumps) are far too large to keep on the box or rsync to the laptop —
an 80-step bf16 weight trajectory is ~246 GB, the per-TICK variant ~492 GB.
This module makes local disk a STAGING area only: a ``.pt`` is written locally
by the observer / capture writer, uploaded to R2, verified, recorded in a small
local manifest, and then the local ``.pt`` is deleted. Steady-state local
footprint is one in-flight snapshot, not the whole trajectory.

Design contract:

1. **Creds from env only.** ``R2_BUCKET`` / ``R2_ENDPOINT`` (or ``R2_ACCOUNT_ID``)
   / ``R2_ACCESS_KEY_ID`` / ``R2_SECRET_ACCESS_KEY`` are read from the process
   env (the launcher sources ``~/.config/verl-research/secrets.env``). Values are
   NEVER logged — only the bucket name, object key and verified size.

2. **Hard bucket guard.** The sink refuses to construct unless the bucket is
   exactly ``shamane-pluralis``. A misconfigured ``R2_BUCKET`` fails loud at
   build, never silently writes to the wrong place.

3. **Upload -> verify -> manifest -> delete-local.** Each upload shells out to
   ``aws s3 cp`` against the R2 S3-compatible endpoint, verifies the remote
   object size (and optionally a sha256 round-trip), appends one row to a local
   ``r2_manifest.jsonl``, then deletes the local ``.pt`` IFF verification passed.
   On ANY failure the local file is KEPT so the run can retry / the operator can
   recover it; the small manifest stays local regardless.

We shell out to the ``aws`` CLI rather than add a ``boto3`` dependency: the repo
has zero boto3/s3 imports, the harness already shells out to ``vastai`` / ``gh``
/ ``rsync``, and ``aws s3 cp`` handles multipart upload of multi-GB ``.pt`` files
to a Cloudflare R2 endpoint out of the box. All subprocess calls go through this
module's ``subprocess`` reference so tests can monkeypatch it with no network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "R2_REQUIRED_BUCKET",
    "R2ArtifactSink",
    "build_r2_sink_from_env",
    "maybe_build_r2_sink",
]

# The ONLY bucket this sink will ever write to. A hard guard, not a default.
R2_REQUIRED_BUCKET = "shamane-pluralis"


def _sha256(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class R2ArtifactSink:
    """Upload-then-delete sink for heavy ``.pt`` artifacts, backed by R2.

    Constructed once per writer (rank 0) when the relevant ``r2_enabled`` flag is
    set. :meth:`upload` is the single entry point used by both the weight-traj
    observer and the gradient capture writer.
    """

    def __init__(
        self,
        *,
        bucket: str,
        endpoint: str,
        access_key_id: str,
        secret_access_key: str,
        key_prefix: str,
        manifest_path: str,
        delete_local: bool = True,
        region: str = "auto",
        verify: str = "size",
        aws_bin: str = "aws",
    ):
        # Hard bucket guard — fail loud, never write to the wrong bucket.
        if bucket != R2_REQUIRED_BUCKET:
            raise RuntimeError(
                f"R2 sink refuses bucket {bucket!r}; the only allowed bucket is {R2_REQUIRED_BUCKET!r}. "
                "Set R2_BUCKET=shamane-pluralis in the secrets file."
            )
        if not endpoint:
            raise RuntimeError(
                "R2 sink requires R2_ENDPOINT (or R2_ACCOUNT_ID to derive it); none found in the env."
            )
        if not access_key_id or not secret_access_key:
            # Do NOT echo the (empty) values — just name the missing keys.
            raise RuntimeError(
                "R2 sink requires R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY in the env (from the secrets file)."
            )
        if verify not in ("size", "sha256"):
            raise ValueError(f"R2 sink verify must be one of (size, sha256); got {verify!r}")

        self.bucket = bucket
        self.endpoint = endpoint
        self.key_prefix = key_prefix.strip("/")
        self.manifest_path = manifest_path
        self.delete_local = bool(delete_local)
        self.verify = verify
        self.aws_bin = aws_bin
        self._n_uploaded = 0
        # Subprocess env carries the R2 creds as AWS_* (S3-compatible). Never logged.
        self._env = {
            **os.environ,
            "AWS_ACCESS_KEY_ID": access_key_id,
            "AWS_SECRET_ACCESS_KEY": secret_access_key,
            "AWS_DEFAULT_REGION": region,
        }
        os.makedirs(os.path.dirname(self.manifest_path) or ".", exist_ok=True)
        # Bucket name + prefix are safe to print; creds are not.
        print(
            f"[comm_eff][r2] sink -> s3://{self.bucket}/{self.key_prefix}/ "
            f"(delete_local={self.delete_local} verify={self.verify} manifest={self.manifest_path})",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    def _run(self, cmd: list) -> subprocess.CompletedProcess:
        """Run an ``aws`` subprocess with the R2 creds in the env. Mockable."""
        return subprocess.run(cmd, env=self._env, capture_output=True, text=True)

    def upload(self, *, local_path: str, key_suffix: str, meta: Optional[dict] = None) -> dict:
        """Upload ``local_path`` to ``<key_prefix>/<key_suffix>``, verify, record, delete-local.

        Returns the manifest row dict. Raises on cp/verify failure and KEEPS the
        local file in that case (so a failed upload never loses the tensor).
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"R2 upload: local artifact missing: {local_path}")
        key = f"{self.key_prefix}/{key_suffix.lstrip('/')}"
        uri = f"s3://{self.bucket}/{key}"
        local_bytes = os.path.getsize(local_path)

        # Optionally tag the object with a sha256 so verification is a true
        # round-trip, not just a size check.
        local_sha = _sha256(local_path) if self.verify == "sha256" else None
        cp_cmd = [self.aws_bin, "s3", "cp", local_path, uri, "--endpoint-url", self.endpoint]
        if local_sha is not None:
            cp_cmd += ["--metadata", f"sha256={local_sha}"]
        cp = self._run(cp_cmd)
        if cp.returncode != 0:
            raise RuntimeError(
                f"R2 upload failed (aws s3 cp rc={cp.returncode}) for {uri}; local file KEPT at {local_path}. "
                f"stderr: {cp.stderr.strip()[:500]}"
            )

        # Verify via head-object: ContentLength must match the local size.
        head = self._run(
            [self.aws_bin, "s3api", "head-object", "--bucket", self.bucket, "--key", key,
             "--endpoint-url", self.endpoint]
        )
        if head.returncode != 0:
            raise RuntimeError(
                f"R2 verify failed (head-object rc={head.returncode}) for {uri}; local file KEPT at {local_path}. "
                f"stderr: {head.stderr.strip()[:500]}"
            )
        try:
            head_obj = json.loads(head.stdout)
        except (ValueError, json.JSONDecodeError) as e:
            raise RuntimeError(f"R2 verify: could not parse head-object for {uri}: {e}; local file KEPT.")
        remote_bytes = int(head_obj.get("ContentLength", -1))
        if remote_bytes != local_bytes:
            raise RuntimeError(
                f"R2 verify size mismatch for {uri}: local={local_bytes} remote={remote_bytes}; "
                f"local file KEPT at {local_path}."
            )
        if local_sha is not None:
            remote_sha = (head_obj.get("Metadata") or {}).get("sha256")
            if remote_sha != local_sha:
                raise RuntimeError(
                    f"R2 verify sha256 mismatch for {uri}: local={local_sha} remote={remote_sha}; "
                    f"local file KEPT at {local_path}."
                )

        row = {
            "key": key,
            "bucket": self.bucket,
            "uri": uri,
            "local_bytes": local_bytes,
            "remote_bytes": remote_bytes,
            "sha256": local_sha,
            "verified": True,
        }
        if meta:
            row.update({k: v for k, v in meta.items()})
        with open(self.manifest_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Delete the local staging file ONLY after a verified upload.
        if self.delete_local:
            try:
                os.remove(local_path)
            except OSError as e:
                logger.warning("comm_eff.r2: verified upload of %s but could not delete local file: %s", uri, e)
        self._n_uploaded += 1
        return row

    @property
    def n_uploaded(self) -> int:
        return self._n_uploaded


def build_r2_sink_from_env(
    *,
    artifact_kind: str,
    manifest_dir: str,
    delete_local: bool = True,
    verify: str = "size",
) -> R2ArtifactSink:
    """Build a :class:`R2ArtifactSink` from the R2 env vars (fail-loud).

    Key prefix is ``verl-research/<experiment>/<regime>/<artifact_kind>`` where
    ``<experiment>``/``<regime>`` come from ``R2_EXPERIMENT`` / ``R2_REGIME``
    (set by the launcher). The role + step/tick + filename are appended by the
    caller as the ``key_suffix`` at upload time.
    """
    if artifact_kind not in ("weights", "grads"):
        raise ValueError(f"artifact_kind must be one of (weights, grads); got {artifact_kind!r}")
    bucket = os.environ.get("R2_BUCKET", "")
    endpoint = os.environ.get("R2_ENDPOINT", "")
    account_id = os.environ.get("R2_ACCOUNT_ID", "")
    if not endpoint and account_id:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    experiment = os.environ.get("R2_EXPERIMENT", "EXP-unknown")
    regime = os.environ.get("R2_REGIME", "regime")
    key_prefix = f"verl-research/{experiment}/{regime}/{artifact_kind}"
    manifest_path = os.path.join(manifest_dir or ".", "r2_manifest.jsonl")
    return R2ArtifactSink(
        bucket=bucket,
        endpoint=endpoint,
        access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
        secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        key_prefix=key_prefix,
        manifest_path=manifest_path,
        delete_local=delete_local,
        verify=verify,
    )


def maybe_build_r2_sink(
    *,
    enabled: bool,
    artifact_kind: str,
    manifest_dir: str,
    delete_local: bool = True,
    verify: str = "size",
) -> Optional[R2ArtifactSink]:
    """Return an :class:`R2ArtifactSink` iff ``enabled``, else ``None`` (strict no-op).

    When disabled the caller keeps writing local ``.pt`` files exactly as before
    (byte-identical behavior). When enabled, creds are read from the env and the
    bucket guard fails loud on a misconfiguration.
    """
    if not enabled:
        return None
    return build_r2_sink_from_env(
        artifact_kind=artifact_kind,
        manifest_dir=manifest_dir,
        delete_local=delete_local,
        verify=verify,
    )
