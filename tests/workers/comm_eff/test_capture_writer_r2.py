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

"""CPU tests for routing CaptureWriter (gradient) dumps through the R2 sink."""

import glob
import json
import os

import pytest
import torch

from verl.workers.comm_eff.capture import CaptureWriter


class _MockSink:
    def __init__(self, delete_local=True):
        self.calls = []
        self.delete_local = delete_local
        self.closed = 0

    def upload(self, *, local_path, key_suffix, meta=None):
        self.calls.append({"local_path": local_path, "key_suffix": key_suffix, "meta": meta})
        if self.delete_local:
            os.remove(local_path)
        return {"key": key_suffix, "verified": True}

    def close(self, timeout=None):
        self.closed += 1


def test_capture_writer_routes_grad_dump_through_sink(tmp_path):
    sink = _MockSink(delete_local=True)
    w = CaptureWriter(capture_dir=str(tmp_path), max_ticks=0, stratified_targets=0, rank=0, r2_sink=sink)
    ok = w.dump(
        role="G_dense",
        target_name="model.layers.0.self_attn.q_proj.weight",
        tensor=torch.randn(4, 4),
        global_step=2,
        optimizer_tick=5,
    )
    assert ok is True
    assert len(sink.calls) == 1
    c = sink.calls[0]
    assert c["key_suffix"] == "G_dense/tick_2_5/model_layers_0_self_attn_q_proj_weight.pt"
    assert c["meta"]["role"] == "G_dense" and c["meta"]["global_step"] == 2 and c["meta"]["optimizer_tick"] == 5
    # local .pt deleted by the (mock) sink; the manifest row persists locally
    assert glob.glob(str(tmp_path / "rank0" / "tick_2_5" / "G_dense" / "*.pt")) == []
    rows = [json.loads(l) for l in open(tmp_path / "rank0" / "manifest.jsonl")]
    assert len(rows) == 1 and rows[0]["role"] == "G_dense"


def test_capture_writer_no_sink_keeps_local(tmp_path):
    w = CaptureWriter(capture_dir=str(tmp_path), max_ticks=0, stratified_targets=0, rank=0)
    assert w.r2_sink is None
    w.dump(role="G_comp", target_name="q_proj", tensor=torch.randn(3, 3), global_step=0, optimizer_tick=0)
    files = glob.glob(str(tmp_path / "rank0" / "tick_0_0" / "G_comp" / "*.pt"))
    assert len(files) == 1  # byte-identical to pre-R2 behavior


def test_capture_writer_close_drains_sink(tmp_path):
    sink = _MockSink(delete_local=True)
    w = CaptureWriter(capture_dir=str(tmp_path), max_ticks=0, stratified_targets=0, rank=0, r2_sink=sink)
    w.dump(role="G_comp", target_name="q_proj", tensor=torch.randn(2, 2), global_step=0, optimizer_tick=0)
    w.close()
    assert sink.closed == 1


def test_capture_writer_close_no_sink_is_safe(tmp_path):
    w = CaptureWriter(capture_dir=str(tmp_path), max_ticks=0, stratified_targets=0, rank=0)
    w.close()  # must not raise with no sink attached


def test_capture_writer_upload_runs_outside_lock(tmp_path):
    """M5 (dump-raises-under-lock-in-sync-mode): the slow r2_sink.upload() call runs
    OUTSIDE the writer's _lock, so a slow / failing upload neither serializes other
    dumps behind the network nor can leave the lock held."""

    class _LockProbingSink:
        def __init__(self, writer):
            self.writer = writer
            self.lock_was_free_during_upload = None
            self.closed = 0

        def upload(self, *, local_path, key_suffix, meta=None):
            # Try to acquire the writer lock non-blockingly: if dump() still held it
            # across the upload, acquire() would FAIL (return False).
            got = self.writer._lock.acquire(blocking=False)
            self.lock_was_free_during_upload = got
            if got:
                self.writer._lock.release()
            os.remove(local_path)
            return {"key": key_suffix, "verified": True}

        def close(self, timeout=None):
            self.closed += 1

    w = CaptureWriter(capture_dir=str(tmp_path), max_ticks=0, stratified_targets=0, rank=0)
    sink = _LockProbingSink(w)
    w.r2_sink = sink
    ok = w.dump(role="G_comp", target_name="q_proj", tensor=torch.randn(2, 2), global_step=0, optimizer_tick=0)
    assert ok is True
    assert sink.lock_was_free_during_upload is True, "upload() ran while dump() still held _lock"


def test_capture_writer_upload_failure_releases_lock(tmp_path):
    """A synchronous upload failure (raise) propagates out of dump() but must leave
    the writer's _lock released (so subsequent dumps are not deadlocked)."""

    class _RaisingSink:
        def upload(self, *, local_path, key_suffix, meta=None):
            raise RuntimeError("R2 upload failed (aws s3 cp rc=1); local file KEPT")

        def close(self, timeout=None):
            pass

    w = CaptureWriter(capture_dir=str(tmp_path), max_ticks=0, stratified_targets=0, rank=0, r2_sink=_RaisingSink())
    with pytest.raises(RuntimeError, match="aws s3 cp"):
        w.dump(role="G_comp", target_name="q_proj", tensor=torch.randn(2, 2), global_step=0, optimizer_tick=0)
    # Lock is free: a fresh non-blocking acquire succeeds.
    assert w._lock.acquire(blocking=False) is True
    w._lock.release()


def test_capture_writer_close_surfaces_failure(tmp_path):
    """CaptureWriter.close() RAISES when the sink reports a permanent upload failure
    (run-end fail-loud, wired into the engine teardown)."""

    class _RaisingCloseSink:
        flush_timeout_s = 1800.0

        def __init__(self):
            self.close_timeouts = []

        def upload(self, *, local_path, key_suffix, meta=None):
            os.remove(local_path)
            return None

        def close(self, timeout=None):
            self.close_timeouts.append(timeout)
            raise RuntimeError("R2 async upload had 1 permanent failure(s)")

    sink = _RaisingCloseSink()
    w = CaptureWriter(capture_dir=str(tmp_path), max_ticks=0, stratified_targets=0, rank=0, r2_sink=sink)
    w.dump(role="G_comp", target_name="q_proj", tensor=torch.randn(2, 2), global_step=0, optimizer_tick=0)
    with pytest.raises(RuntimeError, match="permanent failure"):
        w.close()
    assert sink.close_timeouts == [1800.0]  # finite timeout threaded, not None
