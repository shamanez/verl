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

"""CPU unit tests for the Cloudflare R2 artifact sink (no network).

All ``aws`` subprocess calls are monkeypatched, so these tests exercise the
bucket guard, the upload -> verify -> manifest -> delete-local flow, and the
keep-local-on-failure contract without touching R2.
"""

import json
import os

import pytest

from verl.workers.comm_eff import r2_sink as r2mod
from verl.workers.comm_eff.r2_sink import (
    R2_REQUIRED_BUCKET,
    R2ArtifactSink,
    build_r2_sink_from_env,
    maybe_build_r2_sink,
)


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _install_fake_aws(monkeypatch, *, cp_rc=0, head_rc=0, content_length=None, remote_meta=None):
    """Patch r2_sink.subprocess.run with a fake that records calls.

    ``content_length`` defaults to the real on-disk size of the cp source (so the
    size check passes). Pass an int to force a mismatch.
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "cp" in cmd:
            return _FakeProc(returncode=cp_rc, stderr="cp boom" if cp_rc else "")
        if "head-object" in cmd:
            if head_rc:
                return _FakeProc(returncode=head_rc, stderr="head boom")
            src = calls[0][3]  # aws s3 cp <src> <uri> ...
            size = content_length if content_length is not None else os.path.getsize(src)
            body = {"ContentLength": size}
            if remote_meta is not None:
                body["Metadata"] = remote_meta
            return _FakeProc(returncode=0, stdout=json.dumps(body))
        raise AssertionError(f"unexpected aws cmd: {cmd}")

    monkeypatch.setattr(r2mod.subprocess, "run", fake_run)
    return calls


def _mk_sink(tmp_path, **kw):
    defaults = dict(
        bucket=R2_REQUIRED_BUCKET,
        endpoint="https://acct.r2.cloudflarestorage.com",
        access_key_id="AKIA_test",
        secret_access_key="secret_test",
        key_prefix="verl-research/EXP-43/regimeA/weights",
        manifest_path=str(tmp_path / "r2_manifest.jsonl"),
    )
    defaults.update(kw)
    return R2ArtifactSink(**defaults)


def _write_pt(tmp_path, name="step_5.pt", payload=b"0123456789"):
    p = tmp_path / name
    p.write_bytes(payload)
    return str(p)


def test_bucket_guard_rejects_other_bucket(tmp_path):
    with pytest.raises(RuntimeError, match="shamane-pluralis"):
        _mk_sink(tmp_path, bucket="some-other-bucket")
    # The allowed bucket constructs fine.
    sink = _mk_sink(tmp_path)
    assert sink.bucket == R2_REQUIRED_BUCKET


def test_missing_endpoint_or_creds_fail_loud(tmp_path):
    with pytest.raises(RuntimeError, match="R2_ENDPOINT"):
        _mk_sink(tmp_path, endpoint="")
    with pytest.raises(RuntimeError, match="R2_ACCESS_KEY_ID"):
        _mk_sink(tmp_path, access_key_id="")


def test_upload_verify_delete_success(tmp_path, monkeypatch):
    calls = _install_fake_aws(monkeypatch)
    local = _write_pt(tmp_path)
    sink = _mk_sink(tmp_path)
    row = sink.upload(local_path=local, key_suffix="full/step_5/step_5.pt", meta={"role": "weights", "tick": 9})

    # local staging file deleted after a verified upload
    assert not os.path.exists(local)
    # exactly one verified manifest row with the correct key + merged meta
    rows = [json.loads(l) for l in open(sink.manifest_path)]
    assert len(rows) == 1
    assert rows[0]["verified"] is True
    assert rows[0]["key"] == "verl-research/EXP-43/regimeA/weights/full/step_5/step_5.pt"
    assert rows[0]["role"] == "weights" and rows[0]["tick"] == 9
    assert rows[0]["local_bytes"] == rows[0]["remote_bytes"] == 10
    assert sink.n_uploaded == 1
    assert row == rows[0]
    # cp then head-object were issued, both with the endpoint
    assert any("cp" in c for c in calls) and any("head-object" in c for c in calls)
    assert all("--endpoint-url" in c for c in calls)


def test_keep_local_on_verify_size_mismatch(tmp_path, monkeypatch):
    _install_fake_aws(monkeypatch, content_length=999)  # remote size != local 10
    local = _write_pt(tmp_path)
    sink = _mk_sink(tmp_path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        sink.upload(local_path=local, key_suffix="full/step_5/step_5.pt")
    assert os.path.exists(local)  # local file survives a failed verify
    assert not os.path.exists(sink.manifest_path)  # no manifest row written
    assert sink.n_uploaded == 0


def test_keep_local_on_cp_failure(tmp_path, monkeypatch):
    _install_fake_aws(monkeypatch, cp_rc=1)
    local = _write_pt(tmp_path)
    sink = _mk_sink(tmp_path)
    with pytest.raises(RuntimeError, match="aws s3 cp"):
        sink.upload(local_path=local, key_suffix="full/step_5/step_5.pt")
    assert os.path.exists(local)
    assert sink.n_uploaded == 0


def test_keep_local_on_head_failure(tmp_path, monkeypatch):
    _install_fake_aws(monkeypatch, head_rc=1)
    local = _write_pt(tmp_path)
    sink = _mk_sink(tmp_path)
    with pytest.raises(RuntimeError, match="head-object"):
        sink.upload(local_path=local, key_suffix="full/step_5/step_5.pt")
    assert os.path.exists(local)


def test_no_delete_when_delete_local_false(tmp_path, monkeypatch):
    _install_fake_aws(monkeypatch)
    local = _write_pt(tmp_path)
    sink = _mk_sink(tmp_path, delete_local=False)
    sink.upload(local_path=local, key_suffix="full/step_5/step_5.pt")
    assert os.path.exists(local)  # kept a local copy alongside the R2 object
    assert sink.n_uploaded == 1


def test_sha256_verify_roundtrip(tmp_path, monkeypatch):
    import hashlib

    payload = b"weights-bytes"
    local = _write_pt(tmp_path, payload=payload)
    sha = hashlib.sha256(payload).hexdigest()
    calls = _install_fake_aws(monkeypatch, remote_meta={"sha256": sha})
    sink = _mk_sink(tmp_path, verify="sha256")
    sink.upload(local_path=local, key_suffix="full/step_5/step_5.pt")
    # cp tagged the object with the sha256 metadata
    cp_cmd = next(c for c in calls if "cp" in c)
    assert "--metadata" in cp_cmd and f"sha256={sha}" in cp_cmd
    assert not os.path.exists(local)


def test_sha256_verify_mismatch_keeps_local(tmp_path, monkeypatch):
    local = _write_pt(tmp_path, payload=b"abc")
    _install_fake_aws(monkeypatch, remote_meta={"sha256": "deadbeef"})
    sink = _mk_sink(tmp_path, verify="sha256")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        sink.upload(local_path=local, key_suffix="full/step_5/step_5.pt")
    assert os.path.exists(local)


def test_maybe_build_disabled_returns_none(tmp_path):
    assert maybe_build_r2_sink(enabled=False, artifact_kind="weights", manifest_dir=str(tmp_path)) is None


def test_build_from_env_key_prefix_and_creds(tmp_path, monkeypatch):
    monkeypatch.setenv("R2_BUCKET", R2_REQUIRED_BUCKET)
    monkeypatch.setenv("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKIA_env")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret_env")
    monkeypatch.setenv("R2_EXPERIMENT", "EXP-43")
    monkeypatch.setenv("R2_REGIME", "regimeA")
    sink = build_r2_sink_from_env(artifact_kind="grads", manifest_dir=str(tmp_path))
    assert sink.key_prefix == "verl-research/EXP-43/regimeA/grads"
    # creds reach the subprocess env (as AWS_*), never logged
    assert sink._env["AWS_ACCESS_KEY_ID"] == "AKIA_env"
    assert sink._env["AWS_SECRET_ACCESS_KEY"] == "secret_env"


def test_build_from_env_derives_endpoint_from_account(tmp_path, monkeypatch):
    monkeypatch.setenv("R2_BUCKET", R2_REQUIRED_BUCKET)
    monkeypatch.delenv("R2_ENDPOINT", raising=False)
    monkeypatch.setenv("R2_ACCOUNT_ID", "abc123")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "s")
    sink = build_r2_sink_from_env(artifact_kind="weights", manifest_dir=str(tmp_path))
    assert sink.endpoint == "https://abc123.r2.cloudflarestorage.com"


def test_build_from_env_wrong_bucket_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("R2_BUCKET", "not-allowed")
    monkeypatch.setenv("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "s")
    with pytest.raises(RuntimeError, match="shamane-pluralis"):
        build_r2_sink_from_env(artifact_kind="weights", manifest_dir=str(tmp_path))
