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

"""CPU contracts for the bounded causal rank-1 weight probe."""

import importlib.util
import json
import math
import pathlib
import sys

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_probe = _load(
    "verl_workers_comm_eff_rank1_projection_probe_testonly",
    "verl/workers/comm_eff/rank1_probe.py",
)


def _qwen_snapshot(offset: float = 0.0, *, wrapped: bool = False):
    infix = "._fsdp_wrapped_module" if wrapped else ""
    return {
        "model.embed_tokens.weight": (torch.arange(20, dtype=torch.float32).reshape(5, 4) + offset),
        f"model.layers.14{infix}.self_attn.q_proj.weight": (
            torch.arange(24, dtype=torch.float32).reshape(6, 4) + offset
        ),
        f"model.layers.14{infix}.input_layernorm.weight": torch.arange(6, dtype=torch.float32) + offset,
        "model.norm.weight": torch.arange(6, dtype=torch.float32) + offset,
    }


def _record(probe, projected, latest, *, target_tick=79, source_tick=59, fire_step=80):
    return probe.record_prediction(
        fire_step=fire_step,
        target_tick=target_tick,
        source_tick=source_tick,
        history_ticks=(1, 19, 39, source_tick),
        projected=projected,
        latest_exact=latest,
    )


def _float_metrics(record):
    yield from (value for value in record["aggregate"].values() if isinstance(value, float))
    for tensor in record["tensors"]:
        yield from (value for value in tensor["metrics"].values() if isinstance(value, float))


def test_qwen_selector_is_order_independent_and_canonicalizes_fsdp_infixes():
    names = [
        "model.layers.27._fsdp_wrapped_module.self_attn.q_proj.weight",
        "_fsdp_wrapped_module.model.norm.weight",
        "model.layers.14._fsdp_wrapped_module.input_layernorm.weight",
        "model.layers.0._fsdp_wrapped_module.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "model.layers.14._fsdp_wrapped_module.self_attn.q_proj.weight",
        "model.layers.27._fsdp_wrapped_module.input_layernorm.weight",
        "model.layers.0._fsdp_wrapped_module.input_layernorm.weight",
    ]

    selected = _probe.select_representative_qwen_tensors(reversed(names))

    assert list(selected) == list(_probe.PROBE_ROLES)
    assert selected["embedding"] == "model.embed_tokens.weight"
    assert selected["decoder"] == "model.layers.14._fsdp_wrapped_module.self_attn.q_proj.weight"
    assert selected["layer_norm"] == "model.layers.14._fsdp_wrapped_module.input_layernorm.weight"
    assert selected["final_norm"] == "_fsdp_wrapped_module.model.norm.weight"
    assert _probe.canonical_parameter_name(selected["decoder"]) == "model.layers.14.self_attn.q_proj.weight"


def test_qwen_selector_rejects_missing_category():
    incomplete = _qwen_snapshot()
    incomplete.pop("model.norm.weight")
    with pytest.raises(_probe.Rank1ProjectionProbeError, match="final_norm"):
        _probe.select_representative_qwen_tensors(incomplete)


def test_sample_indices_are_unique_stable_and_do_not_touch_torch_rng():
    torch.manual_seed(123)
    state_before = torch.random.get_rng_state().clone()

    first = _probe.deterministic_sample_indices("model.embed_tokens.weight", (151936, 1536), 16)
    second = _probe.deterministic_sample_indices("model.embed_tokens.weight", (151936, 1536), 16)

    assert first == second
    assert len(first) == len(set(first)) == 16
    assert min(first) >= 0
    assert max(first) < 151936 * 1536
    assert torch.equal(torch.random.get_rng_state(), state_before)
    assert _probe.deterministic_sample_indices("model.norm.weight", (3,), 16) == tuple(
        _probe.deterministic_sample_indices("model.norm.weight", (3,), 16)
    )
    assert len(_probe.deterministic_sample_indices("model.norm.weight", (3,), 16)) == 3


def test_tensor_sampler_refuses_noncontiguous_model_sized_copy():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    assert not tensor.is_contiguous()
    with pytest.raises(_probe.Rank1ProjectionProbeError, match="contiguous"):
        _probe.sample_tensor_values(tensor, (0, 1))


def test_record_keeps_only_bounded_scalar_copies_not_tensor_aliases():
    latest = _qwen_snapshot(0.0)
    projected = _qwen_snapshot(2.0)
    actual = _qwen_snapshot(2.0)
    probe = _probe.Rank1ProjectionProbe(samples_per_tensor=4, max_pending=1)

    summary = _record(probe, projected, latest)
    for tensor in projected.values():
        tensor.add_(1000.0)
    for tensor in latest.values():
        tensor.sub_(1000.0)

    record = probe.resolve_exact(resolve_step=100, exact_tick=79, exact_snapshot=actual)

    assert summary["pending"] == 1
    assert probe.retained_scalar_count == 0
    assert record["aggregate"]["projected_rmse"] == pytest.approx(0.0)
    assert all(tensor["projected"] == tensor["actual"] for tensor in record["tensors"])
    assert probe.status() == {
        "predictions_recorded": 1,
        "resolutions_completed": 1,
        "pending": 0,
        "retained_scalars": 0,
    }


def test_perfect_projection_beats_stale_and_has_correct_update_direction():
    latest = _qwen_snapshot(0.0)
    projected = _qwen_snapshot(2.0)
    exact = _qwen_snapshot(2.0)
    probe = _probe.Rank1ProjectionProbe(samples_per_tensor=4)
    _record(probe, projected, latest)

    record = probe.resolve_exact(resolve_step=100, exact_tick=79, exact_snapshot=exact)

    assert record["aggregate"]["projected_rmse"] == pytest.approx(0.0)
    assert record["aggregate"]["stale_rmse"] > 0.0
    assert record["aggregate"]["skill"] == pytest.approx(1.0)
    assert record["aggregate"]["direction_cos"] == pytest.approx(1.0)
    assert record["aggregate"]["projection_beats_stale"] is True
    assert all(math.isfinite(value) for value in _float_metrics(record))


def test_wrong_way_projection_is_worse_than_stale():
    latest = _qwen_snapshot(0.0)
    projected = _qwen_snapshot(-2.0)
    exact = _qwen_snapshot(2.0)
    probe = _probe.Rank1ProjectionProbe(samples_per_tensor=4)
    _record(probe, projected, latest)

    record = probe.resolve_exact(resolve_step=100, exact_tick=79, exact_snapshot=exact)

    assert record["aggregate"]["projected_rmse"] > record["aggregate"]["stale_rmse"]
    assert record["aggregate"]["skill"] < 0.0
    assert record["aggregate"]["direction_cos"] == pytest.approx(-1.0)
    assert record["aggregate"]["projection_beats_stale"] is False


def test_zero_motion_metrics_are_finite_and_tie_the_stale_baseline():
    values = (1.0, -2.0, 3.0)
    metrics = _probe.projection_sample_metrics(values, values, values)

    assert metrics["projected_rmse"] == 0.0
    assert metrics["stale_rmse"] == 0.0
    assert metrics["skill"] == 0.0
    assert metrics["direction_cos"] == 1.0
    assert metrics["projection_beats_stale"] is False
    assert all(math.isfinite(value) for value in metrics.values() if isinstance(value, float))


def test_only_matching_delayed_exact_tick_can_resolve_forecast():
    latest = _qwen_snapshot(0.0)
    projected = _qwen_snapshot(2.0)
    probe = _probe.Rank1ProjectionProbe(samples_per_tensor=2)
    _record(probe, projected, latest)

    assert probe.resolve_exact(resolve_step=90, exact_tick=59, exact_snapshot=latest) is None
    assert probe.pending_ticks == (79,)
    with pytest.raises(_probe.Rank1ProjectionProbeError, match="skipped unresolved"):
        probe.resolve_exact(resolve_step=100, exact_tick=80, exact_snapshot=projected)
    assert probe.pending_ticks == (79,)
    assert probe.resolve_exact(resolve_step=100, exact_tick=79, exact_snapshot=projected) is not None


def test_matching_newer_target_cannot_skip_an_older_pending_forecast():
    latest = _qwen_snapshot(0.0)
    projected = _qwen_snapshot(2.0)
    probe = _probe.Rank1ProjectionProbe(samples_per_tensor=2, max_pending=2)
    _record(probe, projected, latest)
    _record(
        probe,
        projected,
        latest,
        target_tick=99,
        source_tick=79,
        fire_step=100,
    )

    with pytest.raises(_probe.Rank1ProjectionProbeError, match=r"skipped unresolved.*79"):
        probe.resolve_exact(resolve_step=120, exact_tick=99, exact_snapshot=projected)
    assert probe.pending_ticks == (79, 99)


def test_pending_bound_and_duplicate_target_fail_closed():
    latest = _qwen_snapshot(0.0)
    projected = _qwen_snapshot(2.0)
    probe = _probe.Rank1ProjectionProbe(samples_per_tensor=2, max_pending=1)
    _record(probe, projected, latest)

    with pytest.raises(_probe.Rank1ProjectionProbeError, match="already has a pending"):
        _record(probe, projected, latest)
    with pytest.raises(_probe.Rank1ProjectionProbeError, match="pending forecast bound"):
        probe.record_prediction(
            fire_step=100,
            target_tick=99,
            source_tick=79,
            history_ticks=(19, 39, 59, 79),
            projected=projected,
            latest_exact=latest,
        )
    assert probe.pending_ticks == (79,)
    assert probe.retained_scalar_count == len(_probe.PROBE_ROLES) * 2 * 2


def test_prediction_timeline_must_be_pinned_to_newest_exact_tick():
    snapshot = _qwen_snapshot()
    probe = _probe.Rank1ProjectionProbe(samples_per_tensor=2)
    with pytest.raises(_probe.Rank1ProjectionProbeError, match="must equal newest exact"):
        probe.record_prediction(
            fire_step=80,
            target_tick=79,
            source_tick=58,
            history_ticks=(1, 19, 39, 59),
            projected=snapshot,
            latest_exact=snapshot,
        )
    with pytest.raises(_probe.Rank1ProjectionProbeError, match="must be newer"):
        probe.record_prediction(
            fire_step=80,
            target_tick=59,
            source_tick=59,
            history_ticks=(1, 19, 39, 59),
            projected=snapshot,
            latest_exact=snapshot,
        )


def test_resolution_writes_one_finite_jsonl_record(tmp_path):
    path = tmp_path / "nested" / "rank1_projection_samples.jsonl"
    latest = _qwen_snapshot(0.0, wrapped=True)
    projected = _qwen_snapshot(2.0, wrapped=True)
    probe = _probe.Rank1ProjectionProbe(samples_per_tensor=3, out_path=path)
    _record(probe, projected, latest)

    returned = probe.resolve_exact(resolve_step=100, exact_tick=79, exact_snapshot=projected)
    lines = path.read_text().splitlines()
    persisted = json.loads(lines[0])

    assert len(lines) == 1
    assert persisted == returned
    assert persisted["schema"] == "rank1_projection_probe/v1"
    assert persisted["prediction_fire_step"] == 80
    assert persisted["resolution_step"] == 100
    assert persisted["target_tick"] == persisted["exact_transfer_tick"] == 79
    assert persisted["source_tick"] == 59
    assert persisted["history_ticks"] == [1, 19, 39, 59]
    assert persisted["tensor_count"] == len(_probe.PROBE_ROLES)
    assert [tensor["role"] for tensor in persisted["tensors"]] == list(_probe.PROBE_ROLES)
    assert all(len(tensor["flat_indices"]) == 3 for tensor in persisted["tensors"])
    assert all(math.isfinite(value) for value in _float_metrics(persisted))


def test_nonwriter_computes_result_without_creating_jsonl(tmp_path):
    path = tmp_path / "must_not_exist.jsonl"
    latest = _qwen_snapshot(0.0)
    projected = _qwen_snapshot(2.0)
    probe = _probe.Rank1ProjectionProbe(samples_per_tensor=2, out_path=path, writer=False)
    _record(probe, projected, latest)

    record = probe.resolve_exact(resolve_step=100, exact_tick=79, exact_snapshot=projected)

    assert record["aggregate"]["projected_rmse"] == 0.0
    assert not path.exists()
