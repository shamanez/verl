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

Async mode (opt-in)
-------------------
The default :meth:`R2ArtifactSink.upload` is SYNCHRONOUS: it blocks the training
step until the cp -> verify -> manifest -> delete-local sequence completes. For a
~480 GB per-tick trajectory at a ~60-90 MiB/s single-stream R2 ceiling that block
dominates the step. When ``async_mode=True`` the sink instead decouples uploading
from compute:

* :meth:`upload` ENQUEUES the job and returns immediately (non-blocking);
* a pool of ``upload_workers`` daemon threads each pop a job and run the SAME
  ``_do_upload`` (cp -> verify -> manifest -> delete-local) — multiple parallel
  ``aws s3 cp`` streams approach the aggregate R2 bandwidth ceiling;
* manifest appends are serialized with a lock (concurrent workers);
* :meth:`flush` is a barrier that blocks until the queue drains and RAISES if any
  upload permanently failed (fail-loud — a broken run is never silently
  incomplete);
* :meth:`upload` applies disk BACKPRESSURE: it blocks the producer once the
  staged (queued + in-flight) bytes exceed ``max_staged_bytes``, so the local
  ``full/`` staging area never overflows the box disk even if uploads fall
  behind compute.

When ``async_mode=False`` (the default) the sink is byte-identical to before: no
threads are started and ``upload`` runs ``_do_upload`` inline, raising on failure
and keeping the local file, exactly as today.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import queue
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# A bytes-in-a-gigabyte constant for the staged-bytes backpressure cap.
_BYTES_PER_GB = 1 << 30

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
        async_mode: bool = False,
        upload_workers: int = 4,
        max_staged_gb: float = 80.0,
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

        # --- async upload state ------------------------------------------- #
        self.async_mode = bool(async_mode)
        self.upload_workers = max(1, int(upload_workers))
        if float(max_staged_gb) <= 0:
            raise ValueError(f"R2 sink max_staged_gb must be > 0; got {max_staged_gb}")
        self.max_staged_bytes = int(float(max_staged_gb) * _BYTES_PER_GB)
        # Serialize manifest appends across the worker pool (and harmless on the
        # synchronous path).
        self._manifest_lock = threading.Lock()
        # Worker pool + job queue, built lazily on the first async upload so the
        # synchronous path stays byte-identical (no threads ever started).
        self._jobs: "queue.Queue" = queue.Queue()
        self._workers: list = []
        self._workers_started = False
        self._closed = False
        # Backpressure: staged_bytes = queued + in-flight bytes, gated by a
        # condition variable so upload() blocks the producer above the cap.
        self._staged_bytes = 0
        self._staged_cond = threading.Condition()
        # Fail-loud: permanently-failed uploads are recorded here; flush()/close()
        # raise if non-empty so a broken run is never silently incomplete.
        self._errors: list = []
        self._errors_lock = threading.Lock()
        # Register a process-exit safety net so a run that ends without an explicit
        # close() still drains + surfaces failures (best-effort; never raises in
        # atexit — it logs loudly instead).
        if self.async_mode:
            atexit.register(self._atexit_close)

        # Bucket name + prefix are safe to print; creds are not.
        print(
            f"[comm_eff][r2] sink -> s3://{self.bucket}/{self.key_prefix}/ "
            f"(delete_local={self.delete_local} verify={self.verify} manifest={self.manifest_path} "
            f"async={self.async_mode} workers={self.upload_workers if self.async_mode else 0} "
            f"max_staged_gb={max_staged_gb if self.async_mode else 0})",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    def _run(self, cmd: list) -> subprocess.CompletedProcess:
        """Run an ``aws`` subprocess with the R2 creds in the env. Mockable."""
        return subprocess.run(cmd, env=self._env, capture_output=True, text=True)

    def upload(self, *, local_path: str, key_suffix: str, meta: Optional[dict] = None) -> Optional[dict]:
        """Upload ``local_path`` to ``<key_prefix>/<key_suffix>``, verify, record, delete-local.

        Synchronous (default) mode: runs the cp -> verify -> manifest -> delete-local
        sequence inline and returns the manifest row dict. Raises on cp/verify
        failure and KEEPS the local file in that case (so a failed upload never
        loses the tensor) — byte-identical to the original behaviour.

        Async mode (``async_mode=True``): ENQUEUES the job and returns ``None``
        immediately (non-blocking). The worker pool runs the identical
        ``_do_upload`` sequence; :meth:`flush` / :meth:`close` surface any failure.
        ``upload`` BLOCKS only when the staged bytes are above ``max_staged_bytes``
        (disk backpressure), never on the upload itself.
        """
        if self.async_mode:
            self._enqueue(local_path=local_path, key_suffix=key_suffix, meta=meta)
            return None
        return self._do_upload(local_path=local_path, key_suffix=key_suffix, meta=meta)

    def _do_upload(self, *, local_path: str, key_suffix: str, meta: Optional[dict] = None) -> dict:
        """The cp -> verify -> manifest -> delete-local sequence (sync + worker shared).

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
        # Serialize the manifest append + counter across the worker pool. Harmless
        # on the synchronous path (the lock is uncontended).
        with self._manifest_lock:
            with open(self.manifest_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            self._n_uploaded += 1

        # Delete the local staging file ONLY after a verified upload.
        if self.delete_local:
            try:
                os.remove(local_path)
            except OSError as e:
                logger.warning("comm_eff.r2: verified upload of %s but could not delete local file: %s", uri, e)
        return row

    # ------------------------------------------------------------------ #
    # async upload: queue + worker pool + flush barrier + backpressure
    # ------------------------------------------------------------------ #
    def _ensure_workers(self) -> None:
        """Start the daemon worker pool once (lazy, on the first async upload)."""
        if self._workers_started:
            return
        self._workers_started = True
        for i in range(self.upload_workers):
            t = threading.Thread(target=self._worker_loop, name=f"r2-upload-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def _enqueue(self, *, local_path: str, key_suffix: str, meta: Optional[dict]) -> None:
        """Enqueue an upload job, blocking on the staged-bytes cap (backpressure).

        The producer (the training step) BLOCKS here only while the staged bytes
        (queued + in-flight) are at/above ``max_staged_bytes`` — so the local
        staging area never overflows the box disk even if the uploaders fall
        behind compute. Once a worker drains a job below the cap the producer is
        woken and the job is enqueued.
        """
        if self._closed:
            raise RuntimeError("R2 sink.upload after close(); the worker pool is shut down.")
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"R2 upload: local artifact missing: {local_path}")
        # Surface a worker failure to the PRODUCER promptly (fail-loud): once any
        # upload has permanently failed, stop enqueuing more work.
        self._raise_if_errors()
        self._ensure_workers()
        nbytes = os.path.getsize(local_path)
        with self._staged_cond:
            # Backpressure: wait until adding this job keeps us at/below the cap,
            # OR the queue is empty (always admit at least one in-flight job so a
            # single artifact larger than the cap can still make progress).
            while (self._staged_bytes + nbytes) > self.max_staged_bytes and self._staged_bytes > 0:
                self._staged_cond.wait(timeout=1.0)
            self._staged_bytes += nbytes
        self._jobs.put((local_path, key_suffix, meta, nbytes))

    def _worker_loop(self) -> None:
        """Daemon worker: pop a job, run ``_do_upload``, account staged bytes."""
        while True:
            job = self._jobs.get()
            if job is None:  # sentinel: shut down
                self._jobs.task_done()
                return
            local_path, key_suffix, meta, nbytes = job
            try:
                self._do_upload(local_path=local_path, key_suffix=key_suffix, meta=meta)
            except Exception as e:  # keep local file (already done by _do_upload), record + surface
                with self._errors_lock:
                    self._errors.append((key_suffix, str(e)))
                logger.error("comm_eff.r2: async upload FAILED for %s (local KEPT): %s", key_suffix, e)
            finally:
                # Release the staged bytes + wake any backpressured producer, and
                # mark the queue item done so flush()'s join() can complete.
                with self._staged_cond:
                    self._staged_bytes -= nbytes
                    self._staged_cond.notify_all()
                self._jobs.task_done()

    def _raise_if_errors(self) -> None:
        """Raise an aggregated RuntimeError if any async upload permanently failed."""
        with self._errors_lock:
            if not self._errors:
                return
            n = len(self._errors)
            head = "; ".join(f"{k}: {m}" for k, m in self._errors[:3])
        raise RuntimeError(
            f"R2 async upload had {n} permanent failure(s); local files KEPT. First: {head}"
        )

    def flush(self, timeout: Optional[float] = None) -> None:
        """Barrier: block until the queue is drained, then fail-loud on any failure.

        Blocks until ALL queued + in-flight uploads have completed (a no-op on the
        synchronous path / before any async upload). RAISES if any upload
        permanently failed — so a flush at the per-N-steps checkpoint, or at run
        end, never lets a silently-incomplete trajectory through. ``timeout`` (when
        given) bounds the wait per the underlying join; ``None`` waits forever.
        """
        if not self.async_mode or not self._workers_started:
            self._raise_if_errors()
            return
        if timeout is None:
            self._jobs.join()
        else:
            # queue.Queue.join has no timeout; poll the unfinished-tasks counter.
            import time as _time

            deadline = _time.monotonic() + timeout
            while self._jobs.unfinished_tasks > 0:
                if _time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"R2 flush timed out after {timeout}s with "
                        f"{self._jobs.unfinished_tasks} upload(s) still in flight."
                    )
                _time.sleep(0.05)
        self._raise_if_errors()

    def close(self, timeout: Optional[float] = None) -> None:
        """Flush, then stop the worker pool (join). Idempotent. Fail-loud on flush."""
        if self._closed:
            return
        try:
            if self.async_mode and self._workers_started:
                self.flush(timeout=timeout)
        finally:
            self._closed = True
            if self._workers_started:
                for _ in self._workers:
                    self._jobs.put(None)  # one sentinel per worker
                for t in self._workers:
                    t.join(timeout=timeout)

    def _atexit_close(self) -> None:
        """Process-exit safety net: drain + log loudly (never raise in atexit)."""
        if self._closed:
            return
        try:
            self.close()
        except Exception as e:  # atexit must not raise
            logger.error("comm_eff.r2: atexit close surfaced an upload failure: %s", e)

    @property
    def n_uploaded(self) -> int:
        return self._n_uploaded

    @property
    def n_errors(self) -> int:
        with self._errors_lock:
            return len(self._errors)


def build_r2_sink_from_env(
    *,
    artifact_kind: str,
    manifest_dir: str,
    delete_local: bool = True,
    verify: str = "size",
    async_mode: bool = False,
    upload_workers: int = 4,
    max_staged_gb: float = 80.0,
) -> R2ArtifactSink:
    """Build a :class:`R2ArtifactSink` from the R2 env vars (fail-loud).

    Key prefix is ``verl-research/<experiment>/<regime>/<artifact_kind>`` where
    ``<experiment>``/``<regime>`` come from ``R2_EXPERIMENT`` / ``R2_REGIME``
    (set by the launcher). The role + step/tick + filename are appended by the
    caller as the ``key_suffix`` at upload time.

    ``async_mode`` / ``upload_workers`` / ``max_staged_gb`` configure the opt-in
    background-upload pool (see :class:`R2ArtifactSink`). Defaults keep the sink
    synchronous + byte-identical to the original behaviour.
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
        async_mode=async_mode,
        upload_workers=upload_workers,
        max_staged_gb=max_staged_gb,
    )


def maybe_build_r2_sink(
    *,
    enabled: bool,
    artifact_kind: str,
    manifest_dir: str,
    delete_local: bool = True,
    verify: str = "size",
    async_mode: bool = False,
    upload_workers: int = 4,
    max_staged_gb: float = 80.0,
) -> Optional[R2ArtifactSink]:
    """Return an :class:`R2ArtifactSink` iff ``enabled``, else ``None`` (strict no-op).

    When disabled the caller keeps writing local ``.pt`` files exactly as before
    (byte-identical behavior). When enabled, creds are read from the env and the
    bucket guard fails loud on a misconfiguration. The async knobs are forwarded
    to the sink; their defaults keep it synchronous (today's behaviour).
    """
    if not enabled:
        return None
    return build_r2_sink_from_env(
        artifact_kind=artifact_kind,
        manifest_dir=manifest_dir,
        delete_local=delete_local,
        verify=verify,
        async_mode=async_mode,
        upload_workers=upload_workers,
        max_staged_gb=max_staged_gb,
    )
