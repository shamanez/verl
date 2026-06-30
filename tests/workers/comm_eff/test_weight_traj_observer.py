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

"""CPU unit tests for the FULL-weight trajectory observer + selector.

Covers: the always-full selector (every floating param, no subset/no select_all
toggle), per-step vs per-tick dump cadence, dedup + every_steps, bf16 fidelity,
inactive-rank no-op, and R2 routing (with a mocked sink).
"""

import glob
import json
import os

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.capture import WeightTrajObserver, select_weight_traj_targets


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=True)  # weight (2-D) + bias (1-D)
        self.embed = nn.Embedding(6, 4)  # weight (2-D), the projector EXCLUDES this
        self.norm = nn.LayerNorm(4)  # weight + bias (1-D), excluded by the projector
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))


def test_selector_returns_all_floating_params():
    m = _Tiny()
    sel = select_weight_traj_targets(m.named_parameters())
    names = {n for n, _ in sel}
    # EVERY floating param, including the projector-EXCLUDED embedding + norm + biases
    assert names == {"q_proj.weight", "q_proj.bias", "embed.weight", "norm.weight", "norm.bias"}


def test_selector_skips_non_floating_params():
    pairs = [("w", torch.randn(2, 3)), ("idx", torch.arange(3))]  # idx is int64
    sel = select_weight_traj_targets(pairs)
    assert [n for n, _ in sel] == ["w"]


def test_selector_is_single_arg():
    # The subset/select_all/target_substrs knobs are gone — extra kwargs are an error.
    with pytest.raises(TypeError):
        select_weight_traj_targets([("w", torch.randn(2, 2))], select_all=True)


def _weights():
    return {"down_proj": torch.randn(8, 5), "norm": torch.randn(8), "embed": torch.randn(6, 4)}


def test_per_step_full_dump_real_shapes_bf16(tmp_path):
    obs = WeightTrajObserver(out_dir=str(tmp_path), dump_dtype="bf16", per_tick=False, rank=0)
    w = _weights()
    tick = obs.observe(w, global_step=0)
    assert tick == 0
    files = glob.glob(str(tmp_path / "full" / "step_*.pt"))
    assert len(files) == 1 and files[0].endswith("step_0.pt")

    sd = torch.load(files[0], map_location="cpu")
    assert set(sd.keys()) == {"down_proj", "norm", "embed"}
    assert sd["down_proj"].shape == (8, 5) and sd["down_proj"].dtype == torch.bfloat16

    rows = [json.loads(l) for l in open(tmp_path / "full_manifest.jsonl")]
    assert len(rows) == 1
    r = rows[0]
    assert r["global_step"] == 0 and r["tick"] == 0 and r["n_matrices"] == 3 and r["dump_dtype"] == "bf16"
    # the manifest fro_norm is the EXACT fp32 norm; the bf16 dump matches within rounding
    by_name = {m["name"]: m for m in r["matrices"]}
    exact = float(torch.linalg.norm(w["down_proj"]).item())
    assert abs(by_name["down_proj"]["fro_norm"] - exact) < 1e-4
    bf16_norm = float(torch.linalg.norm(sd["down_proj"].float()).item())
    assert abs(bf16_norm - exact) / exact < 0.02
    assert obs.n_dumped == 1


def test_per_step_dedup_and_every_steps(tmp_path):
    obs = WeightTrajObserver(out_dir=str(tmp_path), per_tick=False, every_steps=2, rank=0)
    w = _weights()
    obs.observe(w, global_step=0)
    obs.observe(w, global_step=0)  # same step -> deduped
    obs.observe(w, global_step=1)  # 1 % 2 != 0 -> skipped
    obs.observe(w, global_step=2)  # dumped
    steps = sorted(int(os.path.basename(f)[5:-3]) for f in glob.glob(str(tmp_path / "full" / "step_*.pt")))
    assert steps == [0, 2]


def test_per_tick_dumps_every_tick(tmp_path):
    obs = WeightTrajObserver(out_dir=str(tmp_path), per_tick=True, rank=0)
    w = _weights()
    obs.observe(w, global_step=0)  # tick 0
    obs.observe(w, global_step=0)  # tick 1 (same step, NOT deduped in per-tick mode)
    obs.observe(w, global_step=1)  # tick 2
    files = sorted(glob.glob(str(tmp_path / "full" / "tick_*.pt")))
    assert [os.path.basename(f) for f in files] == ["tick_0.pt", "tick_1.pt", "tick_2.pt"]

    rows = [json.loads(l) for l in open(tmp_path / "full_manifest.jsonl")]
    assert [(r["global_step"], r["tick"]) for r in rows] == [(0, 0), (0, 1), (1, 2)]
    # the per-tick set subsamples to the per-step trajectory (first tick of each step)
    first_tick_per_step = {}
    for r in rows:
        first_tick_per_step.setdefault(r["global_step"], r["tick"])
    assert first_tick_per_step == {0: 0, 1: 2}


def test_inactive_rank_is_noop(tmp_path):
    obs = WeightTrajObserver(out_dir=str(tmp_path), per_tick=True, rank=1, rank0_only=True)
    assert obs.observe(_weights(), global_step=0) == -1
    assert glob.glob(str(tmp_path / "full" / "*.pt")) == []
    assert obs.n_dumped == 0


class _MockSink:
    def __init__(self):
        self.calls = []
        self.flushes = 0
        self.closed = 0

    def upload(self, *, local_path, key_suffix, meta=None):
        self.calls.append({"local_path": local_path, "key_suffix": key_suffix, "meta": meta})
        return {"key": key_suffix, "verified": True}

    def flush(self, timeout=None):
        self.flushes += 1

    def close(self, timeout=None):
        self.closed += 1


def test_r2_routes_each_snapshot(tmp_path):
    obs = WeightTrajObserver(out_dir=str(tmp_path), per_tick=False, rank=0)
    sink = _MockSink()
    obs.r2_sink = sink  # inject mock (r2 build path is exercised in test_r2_sink.py)
    obs.observe(_weights(), global_step=3)
    assert len(sink.calls) == 1
    c = sink.calls[0]
    assert c["key_suffix"] == "full/step_3/step_3.pt"
    assert c["meta"]["role"] == "weights" and c["meta"]["global_step"] == 3 and c["meta"]["n_matrices"] == 3


def test_async_flush_cadence_and_close(tmp_path):
    """The observer flushes every r2_flush_every_steps steps and drains on close()."""
    obs = WeightTrajObserver(out_dir=str(tmp_path), per_tick=False, rank=0, r2_flush_every_steps=3)
    sink = _MockSink()
    obs.r2_sink = sink
    # global_step 0,3,6 trigger a flush (0 % 3 == 0); 1,2,4,5 do not.
    for gs in range(7):
        obs.observe(_weights(), global_step=gs)
    assert sink.flushes == 3  # steps 0, 3, 6 (each deduped on global_step)
    # the flush at a given step fires once even if observe() is called again same step
    obs.observe(_weights(), global_step=6)
    assert sink.flushes == 3  # step 6 already flushed -> no extra flush

    obs.close()
    assert sink.closed == 1
    obs.close()  # idempotent
    assert sink.closed == 1


def test_async_per_tick_flush_dedup_on_step(tmp_path):
    """In per_tick mode the per-step flush still fires once per matching global_step."""
    obs = WeightTrajObserver(out_dir=str(tmp_path), per_tick=True, rank=0, r2_flush_every_steps=2)
    sink = _MockSink()
    obs.r2_sink = sink
    # two ticks at step 0 (flush once), two at step 1 (no flush), two at step 2 (flush once)
    for gs in (0, 0, 1, 1, 2, 2):
        obs.observe(_weights(), global_step=gs)
    assert sink.flushes == 2  # step 0 and step 2, each once despite 2 ticks


def test_inactive_rank_close_is_safe(tmp_path):
    """close() on an inactive (non-writer) rank is a no-op (no sink attached)."""
    obs = WeightTrajObserver(out_dir=str(tmp_path), per_tick=True, rank=1, rank0_only=True)
    obs.close()  # must not raise (r2_sink is None on the inactive rank)
