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
