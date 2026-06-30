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
bucket guard, the upload -> verify -> manifest -> delete-local flow, the
keep-local-on-failure contract, AND the opt-in async (queue + worker-pool +
flush-barrier + backpressure + fail-loud) mode without touching R2.
"""

import json
import os
import threading
import time

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

    Concurrency-safe: cp records the uploaded source size keyed by destination
    ``--key`` (the head-object verify then looks size up by key), so a pool of
    workers running interleaved cp/head pairs each verify against their OWN object
    rather than racing on a shared "first call" reference.
    """
    calls = []
    sizes = {}  # key -> uploaded source size, recorded at cp time
    lock = threading.Lock()

    def _key_of(cmd):
        # aws s3 cp <src> s3://bucket/<key> ...  OR  s3api head-object --key <key>
        if "cp" in cmd:
            return cmd[4].split("/", 3)[-1]  # cmd[4] = s3://bucket/<key>
        i = cmd.index("--key")
        return cmd[i + 1]

    def fake_run(cmd, **kwargs):
        with lock:
            calls.append(cmd)
        if "cp" in cmd:
            if cp_rc == 0:
                with lock:
                    sizes[_key_of(cmd)] = os.path.getsize(cmd[3])  # cmd[3] = local source
            return _FakeProc(returncode=cp_rc, stderr="cp boom" if cp_rc else "")
        if "head-object" in cmd:
            if head_rc:
                return _FakeProc(returncode=head_rc, stderr="head boom")
            if content_length is not None:
                size = content_length
            else:
                with lock:
                    size = sizes[_key_of(cmd)]
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


# ====================================================================== #
# Async mode (queue + worker pool + flush barrier + backpressure + fail-loud)
# ====================================================================== #


def _install_slow_fake_aws(monkeypatch, *, cp_rc=0, cp_sleep=0.0, head_rc=0, content_length=None):
    """Like _install_fake_aws but with a per-cp sleep (to exercise backpressure).

    Concurrency-safe (cp records source size keyed by destination key; head-object
    verifies against that key). The cp branch optionally sleeps ``cp_sleep`` seconds
    to simulate a slow upload.
    """
    calls = []
    sizes = {}
    lock = threading.Lock()

    def _key_of(cmd):
        if "cp" in cmd:
            return cmd[4].split("/", 3)[-1]  # cmd[4] = s3://bucket/<key>
        i = cmd.index("--key")
        return cmd[i + 1]

    def fake_run(cmd, **kwargs):
        with lock:
            calls.append(cmd)
        if "cp" in cmd:
            if cp_sleep:
                time.sleep(cp_sleep)
            if cp_rc == 0:
                with lock:
                    sizes[_key_of(cmd)] = os.path.getsize(cmd[3])  # cmd[3] = local source
            return _FakeProc(returncode=cp_rc, stderr="cp boom" if cp_rc else "")
        if "head-object" in cmd:
            if head_rc:
                return _FakeProc(returncode=head_rc, stderr="head boom")
            if content_length is not None:
                size = content_length
            else:
                with lock:
                    size = sizes[_key_of(cmd)]
            return _FakeProc(returncode=0, stdout=json.dumps({"ContentLength": size}))
        raise AssertionError(f"unexpected aws cmd: {cmd}")

    monkeypatch.setattr(r2mod.subprocess, "run", fake_run)
    return calls


def test_async_enqueue_flush_uploads_all(tmp_path, monkeypatch):
    """N enqueued jobs -> flush() -> all uploaded, N verified rows, all .pt gone."""
    _install_fake_aws(monkeypatch)
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=4)
    n = 12
    locals_ = []
    for i in range(n):
        p = _write_pt(tmp_path, name=f"step_{i}.pt", payload=f"payload-{i}".encode())
        locals_.append(p)
        # upload() is non-blocking in async mode and returns None.
        assert sink.upload(local_path=p, key_suffix=f"full/step_{i}/step_{i}.pt", meta={"i": i}) is None

    sink.flush()  # barrier: drains the queue
    # every local staging file deleted after a verified upload
    assert all(not os.path.exists(p) for p in locals_)
    rows = [json.loads(l) for l in open(sink.manifest_path)]
    assert len(rows) == n
    assert all(r["verified"] is True for r in rows)
    assert {r["i"] for r in rows} == set(range(n))
    assert sink.n_uploaded == n
    assert sink.n_errors == 0
    sink.close()


def test_async_default_off_is_synchronous(tmp_path, monkeypatch):
    """async_mode=False (default) keeps the synchronous path: no workers, inline upload."""
    _install_fake_aws(monkeypatch)
    sink = _mk_sink(tmp_path)  # async_mode defaults False
    assert sink.async_mode is False
    n_threads_before = threading.active_count()
    local = _write_pt(tmp_path)
    row = sink.upload(local_path=local, key_suffix="full/step_5/step_5.pt")
    # synchronous: a row is RETURNED inline (async returns None), file already gone
    assert row is not None and row["verified"] is True
    assert not os.path.exists(local)
    assert sink._workers_started is False
    assert threading.active_count() == n_threads_before  # no worker threads spawned


def test_async_backpressure_blocks_above_cap(tmp_path, monkeypatch):
    """With a tiny staged cap + a slow uploader, the producer BLOCKS at the cap.

    Each artifact is ~1000 bytes; the cap is set just under 2 artifacts, with a
    single slow worker. The 3rd enqueue must not return until a worker drains one,
    so the staged bytes never exceed the cap by more than one in-flight job.
    """
    _install_slow_fake_aws(monkeypatch, cp_sleep=0.25)
    payload = b"x" * 1000
    # cap = 1.5 artifacts worth of bytes (so at most 1 queued + 1 in-flight).
    cap_gb = (1500) / float(r2mod._BYTES_PER_GB)
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=1, max_staged_gb=cap_gb)

    enqueued = []
    peak_staged = {"v": 0}

    def producer():
        for i in range(5):
            p = _write_pt(tmp_path, name=f"bp_{i}.pt", payload=payload)
            sink.upload(local_path=p, key_suffix=f"full/bp_{i}/bp_{i}.pt")
            enqueued.append(i)
            peak_staged["v"] = max(peak_staged["v"], sink._staged_bytes)

    t = threading.Thread(target=producer)
    t.start()
    # Shortly after start, the producer should be BLOCKED on backpressure: with a
    # 0.25s/upload single worker it cannot have enqueued all 5 yet.
    time.sleep(0.15)
    assert len(enqueued) < 5, "producer should be backpressured, not racing ahead"

    t.join(timeout=10)
    assert not t.is_alive()
    sink.flush()
    # The staged bytes never blew far past the cap: at most 1 queued + 1 in-flight.
    assert peak_staged["v"] <= 2 * 1000
    rows = [json.loads(l) for l in open(sink.manifest_path)]
    assert len(rows) == 5 and sink.n_uploaded == 5
    sink.close()


def test_async_fail_loud_on_cp_failure(tmp_path, monkeypatch):
    """A failed upload keeps the local file, writes NO verified row, and flush() RAISES.

    Fail-loud surfaces on EITHER a subsequent ``upload()`` (the producer is told
    promptly) OR the ``flush()`` barrier; the test tolerates both by collecting
    every enqueue and asserting flush ultimately raises.
    """
    _install_slow_fake_aws(monkeypatch, cp_rc=1)
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=2)
    locals_ = []
    raised_early = False
    for i in range(4):
        p = _write_pt(tmp_path, name=f"bad_{i}.pt", payload=b"data")
        locals_.append(p)
        try:
            sink.upload(local_path=p, key_suffix=f"full/bad_{i}/bad_{i}.pt")
        except RuntimeError as e:
            assert "permanent failure" in str(e)
            raised_early = True
            break

    if not raised_early:
        with pytest.raises(RuntimeError, match="permanent failure"):
            sink.flush()
    # local files KEPT (no data loss) and NO verified manifest row exists
    assert all(os.path.exists(p) for p in locals_)
    assert not os.path.exists(sink.manifest_path) or open(sink.manifest_path).read() == ""
    assert sink.n_uploaded == 0
    assert sink.n_errors >= 1
    # close() must also surface the failure (run-end fail-loud).
    with pytest.raises(RuntimeError, match="permanent failure"):
        sink.close()


def test_async_close_surfaces_failure(tmp_path, monkeypatch):
    """close() (run-end barrier) also RAISES if any async upload permanently failed."""
    _install_slow_fake_aws(monkeypatch, cp_rc=1)
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=2)
    p = _write_pt(tmp_path, name="bad.pt", payload=b"data")
    sink.upload(local_path=p, key_suffix="full/bad/bad.pt")
    with pytest.raises(RuntimeError, match="permanent failure"):
        sink.close()
    assert os.path.exists(p)  # kept


def test_async_flush_is_noop_before_any_upload(tmp_path, monkeypatch):
    """flush()/close() are safe no-ops on an async sink that never uploaded."""
    _install_fake_aws(monkeypatch)
    sink = _mk_sink(tmp_path, async_mode=True)
    sink.flush()  # nothing queued -> returns immediately, no error
    sink.close()
    assert sink.n_uploaded == 0 and sink.n_errors == 0


def test_async_partial_failure_keeps_good_rows(tmp_path, monkeypatch):
    """Good uploads still land + delete; the bad one keeps its file; flush RAISES."""
    sizes = {}
    lock = threading.Lock()

    def _key_of(cmd):
        if "cp" in cmd:
            return cmd[4].split("/", 3)[-1]
        i = cmd.index("--key")
        return cmd[i + 1]

    def fake_run(cmd, **kwargs):
        if "cp" in cmd:
            # fail only the artifact whose key contains 'bad'
            rc = 1 if any("bad" in str(a) for a in cmd) else 0
            if rc == 0:
                with lock:
                    sizes[_key_of(cmd)] = os.path.getsize(cmd[3])
            return _FakeProc(returncode=rc, stderr="cp boom" if rc else "")
        if "head-object" in cmd:
            with lock:
                size = sizes[_key_of(cmd)]
            return _FakeProc(returncode=0, stdout=json.dumps({"ContentLength": size}))
        raise AssertionError(cmd)

    monkeypatch.setattr(r2mod.subprocess, "run", fake_run)
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=3)
    # Enqueue the 3 good ones FIRST and barrier them, so they are all verified +
    # deleted before the bad one is enqueued (deterministic, not timing-raced).
    good = [_write_pt(tmp_path, name=f"good_{i}.pt", payload=b"ok") for i in range(3)]
    for i, p in enumerate(good):
        sink.upload(local_path=p, key_suffix=f"full/good_{i}/good_{i}.pt")
    sink.flush()  # the 3 good uploads all land
    assert all(not os.path.exists(p) for p in good)
    rows = [json.loads(l) for l in open(sink.manifest_path)]
    assert len(rows) == 3 and all(r["verified"] for r in rows)
    assert sink.n_uploaded == 3 and sink.n_errors == 0

    bad = _write_pt(tmp_path, name="bad.pt", payload=b"no")
    sink.upload(local_path=bad, key_suffix="full/bad/bad.pt")
    with pytest.raises(RuntimeError, match="permanent failure"):
        sink.flush()
    # the bad one is KEPT, never gets a verified row, and the good rows are intact
    assert os.path.exists(bad)
    rows = [json.loads(l) for l in open(sink.manifest_path)]
    assert len(rows) == 3 and sink.n_uploaded == 3 and sink.n_errors == 1
    with pytest.raises(RuntimeError, match="permanent failure"):
        sink.close()


# ====================================================================== #
# Regression tests for the adversarial-review findings (async-r2-upload)
# ====================================================================== #


def test_r2sink001_no_staged_bytes_leak_when_put_fails(tmp_path, monkeypatch):
    """R2SINK-001: a failure between the bytes-increment and the enqueue must NOT
    leak ``_staged_bytes`` and wedge all future producers on phantom backpressure.

    We simulate the failure by making ``_jobs.put`` raise (stands in for a
    KeyboardInterrupt/SystemExit/signal landing in that window). The fix puts the
    increment AFTER a successful put inside the same lock, so a failed put leaves
    ``_staged_bytes`` at 0 and the next producer is not blocked.
    """
    _install_fake_aws(monkeypatch)
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=1)
    # Start the worker pool (so _ensure_workers is past) and then break put().
    p0 = _write_pt(tmp_path, name="warm.pt", payload=b"warm")
    sink.upload(local_path=p0, key_suffix="full/warm/warm.pt")
    sink.flush()
    assert sink._staged_bytes == 0

    boom = _write_pt(tmp_path, name="boom.pt", payload=b"x" * 500)
    orig_put = sink._jobs.put

    def exploding_put(item, *a, **kw):
        raise RuntimeError("simulated interrupt between increment and enqueue")

    monkeypatch.setattr(sink._jobs, "put", exploding_put)
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        sink.upload(local_path=boom, key_suffix="full/boom/boom.pt")
    # CRITICAL: the counter did NOT leak — it is still 0, not 500.
    assert sink._staged_bytes == 0, "staged-bytes leaked on a failed enqueue (backpressure deadlock)"

    # Restore put: a subsequent producer must NOT be blocked by phantom backpressure.
    monkeypatch.setattr(sink._jobs, "put", orig_put)
    good = _write_pt(tmp_path, name="after.pt", payload=b"ok")
    sink.upload(local_path=good, key_suffix="full/after/after.pt")
    sink.flush()  # would hang/raise if the counter were still inflated
    assert sink.n_uploaded == 2 and sink.n_errors == 0
    sink.close()


class _BlockingProc:
    """A cp that blocks on an Event so the worker is 'stuck' until released.

    After release the cp completes normally and a subsequent head-object verifies
    against the uploaded source size (so a released worker can finish cleanly).
    """

    def __init__(self, release_evt, started_evt):
        self._release = release_evt
        self._started = started_evt
        self._sizes = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key_of(cmd):
        if "cp" in cmd:
            return cmd[4].split("/", 3)[-1]
        i = cmd.index("--key")
        return cmd[i + 1]

    def __call__(self, cmd, **kwargs):
        if "cp" in cmd:
            self._started.set()
            self._release.wait(timeout=30)  # bounded so a leaked test can't hang CI
            with self._lock:
                self._sizes[self._key_of(cmd)] = os.path.getsize(cmd[3])
            return _FakeProc(returncode=0, stderr="")
        if "head-object" in cmd:
            with self._lock:
                size = self._sizes.get(self._key_of(cmd), 0)
            return _FakeProc(returncode=0, stdout=json.dumps({"ContentLength": size}))
        raise AssertionError(f"unexpected aws cmd while blocked: {cmd}")


def test_close_with_timeout_does_not_hang_on_stuck_worker(tmp_path, monkeypatch):
    """critical-queue-join-deadlock + R2SINK-004 + missing-atexit-timeout:
    close(timeout=T) must RETURN within ~T (never block forever on queue.join())
    when a worker is stuck mid-upload, and it must NOT report clean success —
    it RAISES because a worker is still alive.
    """
    release = threading.Event()
    started = threading.Event()
    monkeypatch.setattr(r2mod.subprocess, "run", _BlockingProc(release, started))
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=1)
    p = _write_pt(tmp_path, name="stuck.pt", payload=b"data")
    sink.upload(local_path=p, key_suffix="full/stuck/stuck.pt")
    assert started.wait(timeout=5), "worker never started the (blocking) cp"

    t0 = time.monotonic()
    with pytest.raises((RuntimeError, TimeoutError)):
        sink.close(timeout=0.5)  # bounded — must not hang
    elapsed = time.monotonic() - t0
    # Bounded: well under the 30s _BlockingProc wait, proves no unbounded join().
    assert elapsed < 10.0, f"close() did not return within a bounded time ({elapsed:.1f}s)"
    # Release the worker so the daemon thread exits cleanly for the rest of the suite.
    release.set()


def test_flush_with_timeout_is_bounded(tmp_path, monkeypatch):
    """per-step-flush-timeout-handling: flush(timeout=T) raises TimeoutError within
    ~T when a worker is stuck, never blocking forever on queue.join()."""
    release = threading.Event()
    started = threading.Event()
    monkeypatch.setattr(r2mod.subprocess, "run", _BlockingProc(release, started))
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=1)
    p = _write_pt(tmp_path, name="slow.pt", payload=b"data")
    sink.upload(local_path=p, key_suffix="full/slow/slow.pt")
    assert started.wait(timeout=5)

    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="still in flight"):
        sink.flush(timeout=0.5)
    assert time.monotonic() - t0 < 10.0
    release.set()
    # After release the in-flight upload completes; a bounded flush now succeeds.
    sink.flush(timeout=10.0)
    assert sink.n_uploaded == 1
    sink.close(timeout=5.0)


def test_atexit_close_uses_finite_timeout_and_never_raises(tmp_path, monkeypatch):
    """missing-atexit-timeout: the atexit handler passes a FINITE timeout to close()
    and never raises (atexit must not raise), even with a stuck worker — it logs.
    """
    release = threading.Event()
    started = threading.Event()
    monkeypatch.setattr(r2mod.subprocess, "run", _BlockingProc(release, started))
    # Force the atexit handler's bounded timeout tiny so the test is fast.
    monkeypatch.setattr(r2mod, "_DEFAULT_ATEXIT_CLOSE_TIMEOUT_S", 0.3)
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=1)
    p = _write_pt(tmp_path, name="ax.pt", payload=b"data")
    sink.upload(local_path=p, key_suffix="full/ax/ax.pt")
    assert started.wait(timeout=5)

    t0 = time.monotonic()
    # Must NOT raise (atexit-safe) and must be bounded by the finite timeout.
    sink._atexit_close()
    assert time.monotonic() - t0 < 10.0
    release.set()


def test_ensure_workers_guarded_during_finalizing(tmp_path, monkeypatch):
    """thread-creation-during-shutdown: _ensure_workers refuses to spawn threads
    when the interpreter is finalizing (would raise an opaque RuntimeError)."""
    _install_fake_aws(monkeypatch)
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=2)
    monkeypatch.setattr(r2mod.sys, "is_finalizing", lambda: True)
    p = _write_pt(tmp_path, name="fin.pt", payload=b"data")
    with pytest.raises(RuntimeError, match="interpreter shutdown"):
        sink.upload(local_path=p, key_suffix="full/fin/fin.pt")
    assert sink._workers_started is False  # no threads spawned
    assert os.path.exists(p)  # artifact kept


def test_close_after_clean_flush_does_not_raise(tmp_path, monkeypatch):
    """A clean async run: close(timeout=T) drains, joins, and returns WITHOUT raising
    (no spurious is_alive failure when workers exit promptly)."""
    _install_fake_aws(monkeypatch)
    sink = _mk_sink(tmp_path, async_mode=True, upload_workers=3)
    for i in range(5):
        p = _write_pt(tmp_path, name=f"clean_{i}.pt", payload=f"p{i}".encode())
        sink.upload(local_path=p, key_suffix=f"full/clean_{i}/clean_{i}.pt")
    sink.close(timeout=10.0)  # must not raise
    assert sink.n_uploaded == 5 and sink.n_errors == 0
    assert not any(t.is_alive() for t in sink._workers)


class _ProbingLock:
    """Wraps a real lock and counts acquire() calls (a _thread.lock's acquire is a
    read-only C attribute, so we substitute the whole lock object instead)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.acquires = 0

    def acquire(self, *a, **kw):
        self.acquires += 1
        return self._lock.acquire(*a, **kw)

    def release(self, *a, **kw):
        return self._lock.release(*a, **kw)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def test_sync_path_takes_no_manifest_lock_and_no_threads(tmp_path, monkeypatch):
    """FINDING-001: the synchronous (default) path stays byte-identical — it never
    acquires _manifest_lock and never starts a worker thread."""
    _install_fake_aws(monkeypatch)
    sink = _mk_sink(tmp_path)  # async_mode defaults False
    n_threads_before = threading.active_count()

    probe = _ProbingLock()
    sink._manifest_lock = probe  # the sync path must NOT touch this lock at all
    local = _write_pt(tmp_path)
    row = sink.upload(local_path=local, key_suffix="full/step_5/step_5.pt")

    assert probe.acquires == 0, "sync path must NOT acquire _manifest_lock (FINDING-001)"
    assert row is not None and row["verified"] is True
    assert sink._workers_started is False
    assert threading.active_count() == n_threads_before
    rows = [json.loads(l) for l in open(sink.manifest_path)]
    assert len(rows) == 1 and rows[0]["verified"] is True


def test_async_path_does_take_manifest_lock(tmp_path):
    """Complement to FINDING-001: the async path DOES serialize the manifest append
    under _manifest_lock (the lock exists to make _do_upload thread-safe)."""
    # Use a real fake-aws via monkeypatch-free direct subprocess swap.
    import json as _json

    sizes = {}

    def fake_run(cmd, **kwargs):
        if "cp" in cmd:
            sizes[cmd[4].split("/", 3)[-1]] = os.path.getsize(cmd[3])
            return _FakeProc(returncode=0)
        if "head-object" in cmd:
            i = cmd.index("--key")
            return _FakeProc(returncode=0, stdout=_json.dumps({"ContentLength": sizes[cmd[i + 1]]}))
        raise AssertionError(cmd)

    import pytest as _pytest  # local to avoid touching module imports

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(r2mod.subprocess, "run", fake_run)
        sink = _mk_sink(tmp_path, async_mode=True, upload_workers=1)
        probe = _ProbingLock()
        sink._manifest_lock = probe
        p = _write_pt(tmp_path, name="async.pt", payload=b"data")
        sink.upload(local_path=p, key_suffix="full/async/async.pt")
        sink.flush()
        assert probe.acquires == 1  # the worker serialized its manifest append
        sink.close()
