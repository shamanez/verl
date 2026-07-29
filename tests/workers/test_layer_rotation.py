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
"""The nine laptop CPU gates for issue #95 (layer-rotation GRPO).

ALL nine must pass before any GPU is provisioned. They are numbered exactly as the
plan's "Laptop CPU gates" section:

1. Schedule grammar.
2. Static freeze parity on a 4-layer toy model.
3. FSDP mechanism gate -- the P1-vs-P2 decision (gloo, CPU, 4-layer toy Qwen2,
   ``use_orig_params=true``, transformer auto-wrap, 5 numeric bars across 3
   consecutive rotations). Writes the verdict to
   ``research/runs/95-layer-rotation-grpo/mechanism.txt``.
4. Rotation determinism and visit accounting.
5. State accounting (``persist_park``).
6. Gate visibility: every money-gate line is ``print(..., flush=True)``, never
   ``logger.info`` (the issue #64 lesson).
7. Tied-tensor identity (rider), by ``data_ptr`` and never by name.
8. Rider anti-clobber on the 4-layer toy.
9. Knob-off regression: ``LAYER_OTHER=freeze`` / unset reproduces the pre-rider path.

Nothing here needs a GPU, a real model download, or a network call.
"""

from __future__ import annotations

import ast
import functools
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from verl.workers.layer_rotation import (
    GATE_PREFIX,
    LayerRotationController,
    RotationSchedule,
    apply_active_set,
    decoder_layer_params,
    grad_bytes,
    one_layer_opt_bytes,
    optimizer_state_bytes_split,
    park_optimizer_state,
    parse_layer_schedule,
    root_params,
    schedule_from_env,
)

NUM_TOY_LAYERS = 4
NUM_REAL_LAYERS = 28  # Qwen2.5-Math-1.5B
REPO_ROOT = Path(__file__).resolve().parents[2]
# The durable copy lives in the PRIMARY checkout's research/ tree (the harness's
# state root). ``VERL_RESEARCH_DIR`` points there when the runner sets it; otherwise
# this worktree's own research/ dir is used and the runner copies the file across.
_RESEARCH_DIR = Path(os.environ.get("VERL_RESEARCH_DIR", REPO_ROOT / "research"))
MECHANISM_FILE = _RESEARCH_DIR / "runs" / "95-layer-rotation-grpo" / "mechanism.txt"


# ---------------------------------------------------------------------------#
# Shared toy model.                                                          #
# ---------------------------------------------------------------------------#
def toy_config(tie: bool = True):
    from transformers.models.qwen2 import Qwen2Config

    return Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_TOY_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
        max_position_embeddings=64,
        attn_implementation="eager",
    )


def toy_model(seed: int = 0, tie: bool = True):
    from transformers.models.qwen2 import Qwen2ForCausalLM

    torch.manual_seed(seed)
    return Qwen2ForCausalLM(toy_config(tie=tie))


def toy_batch(seed: int = 1234, bsz: int = 2, seqlen: int = 8):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, (bsz, seqlen), generator=g)


# ===========================================================================#
# Gate 1: schedule grammar.                                                   #
# ===========================================================================#
def test_gate1_schedule_grammar_accepts_the_four_specs_and_dense():
    for empty in (None, "", "   "):
        sched = parse_layer_schedule(empty, NUM_REAL_LAYERS)
        assert sched.mode == "dense"
        assert sched.indices == ()
        assert sched.is_dense

    assert parse_layer_schedule("static:14", NUM_REAL_LAYERS).mode == "static"
    assert parse_layer_schedule("static:14", NUM_REAL_LAYERS).indices == (14,)
    assert parse_layer_schedule("static:11-15", NUM_REAL_LAYERS).indices == (11, 12, 13, 14, 15)
    rot5 = parse_layer_schedule("rotate:11-15", NUM_REAL_LAYERS)
    assert rot5.mode == "rotate" and rot5.is_rotating
    assert rot5.indices == (11, 12, 13, 14, 15)
    rot28 = parse_layer_schedule("rotate:0-27", NUM_REAL_LAYERS)
    assert rot28.indices == tuple(range(28))
    assert len(rot28.indices) == NUM_REAL_LAYERS


@pytest.mark.parametrize("spec", ["static:28", "static:15-11", "rotate:x", "rotate:-1-5"])
def test_gate1_schedule_grammar_raises_on_malformed_or_out_of_range(spec):
    with pytest.raises(ValueError):
        parse_layer_schedule(spec, NUM_REAL_LAYERS)


def test_gate1_schedule_grammar_raises_on_bad_support_knobs_and_missing_mode():
    # A bare index with no mode prefix is ambiguous: refuse it rather than guess.
    with pytest.raises(ValueError):
        parse_layer_schedule("14", NUM_REAL_LAYERS)
    with pytest.raises(ValueError):
        parse_layer_schedule("shuffle:11-15", NUM_REAL_LAYERS)
    with pytest.raises(ValueError):
        parse_layer_schedule("rotate:11-15", NUM_REAL_LAYERS, rotate_every=0)
    with pytest.raises(ValueError):
        parse_layer_schedule("rotate:11-15", NUM_REAL_LAYERS, rotate_every="every")
    with pytest.raises(ValueError):
        parse_layer_schedule("rotate:11-15", NUM_REAL_LAYERS, adam_policy="keep")
    with pytest.raises(ValueError):
        parse_layer_schedule("rotate:11-15", NUM_REAL_LAYERS, layer_other="thaw")
    with pytest.raises(ValueError):
        parse_layer_schedule("rotate:11-15", NUM_REAL_LAYERS, state_device="")


def test_gate1_schedule_from_env_reads_the_documented_knob_names():
    env = {
        "LAYER_SCHEDULE": "rotate:11-15",
        "ROTATE_EVERY": "2",
        "ROTATE_ADAM": "persist_park",
        "ROTATE_STATE_DEVICE": "cpu",
        "LAYER_OTHER": "train",
    }
    sched = schedule_from_env(NUM_REAL_LAYERS, env=env)
    assert (sched.mode, sched.indices) == ("rotate", (11, 12, 13, 14, 15))
    assert sched.rotate_every == 2
    assert sched.adam_policy == "persist_park"
    assert sched.state_device == "cpu"
    assert sched.layer_other == "train"
    # Defaults when only LAYER_SCHEDULE is set.
    bare = schedule_from_env(NUM_REAL_LAYERS, env={"LAYER_SCHEDULE": "static:14"})
    assert (bare.rotate_every, bare.adam_policy, bare.state_device, bare.layer_other) == (
        1,
        "persist_park",
        "cpu",
        "freeze",
    )


# ===========================================================================#
# Gate 2: static freeze parity on a 4-layer toy model.                        #
# ===========================================================================#
def _analytic_trainable(model, active, root_on):
    by_index = decoder_layer_params(model)
    total = 0
    for i in active:
        total += sum(p.numel() for p in by_index[i])
    if root_on:
        total += sum(p.numel() for p in root_params(model))
    return total


def test_gate2_static_freeze_leaves_exactly_the_requested_layers_trainable():
    """Explicit ``root_requires_grad=False`` reproduces the literal issue #64 freeze."""
    model = toy_model()
    by_index = decoder_layer_params(model)
    total_numel = sum(p.numel() for p in model.parameters())

    for active in [(2,), (1, 2), (0, 1, 2, 3)]:
        report = apply_active_set(model, active, layer_other="freeze", root_requires_grad=False)
        assert report["active_layers"] == tuple(sorted(active))
        assert report["num_decoder_layers"] == NUM_TOY_LAYERS
        # exactly the requested layers are trainable, and nothing else is
        for i in range(NUM_TOY_LAYERS):
            expect = i in active
            assert all(p.requires_grad is expect for p in by_index[i]), i
        assert all(not p.requires_grad for p in root_params(model))
        analytic = _analytic_trainable(model, active, root_on=False)
        assert report["trainable_params"] == analytic
        assert report["optimized_params"] == analytic
        assert report["trainable_frac"] == pytest.approx(analytic / total_numel, rel=1e-12)
        assert report["optimized_frac"] == pytest.approx(analytic / total_numel, rel=1e-12)


def test_gate2_tied_root_is_frozen_by_masking_not_by_requires_grad():
    """AUTO mode: a TIED root group keeps requires_grad=True but is never optimized.

    FSDP1 + ``use_orig_params=True`` cannot run a frozen flat_param that owns a tied
    tensor (CPU gate 3 measured the crash), so the root group is frozen the other
    way: excluded from the optimizer and grad-masked before the clip. The number that
    matters -- ``optimized_params`` -- is unchanged, and the decoder-layer freeze is
    exactly as before.
    """
    model = toy_model(tie=True)
    by_index = decoder_layer_params(model)
    total_numel = sum(p.numel() for p in model.parameters())
    report = apply_active_set(model, (2,), layer_other="freeze")  # AUTO
    assert report["root_tied"] is True
    assert report["root_requires_grad"] is True
    assert report["root_grad_masked"] is True
    assert report["root_optimized"] is False
    analytic = _analytic_trainable(model, (2,), root_on=False)
    assert report["optimized_params"] == analytic
    assert report["optimized_frac"] == pytest.approx(analytic / total_numel, rel=1e-12)
    # the decoder-layer freeze is unaffected
    for i in range(NUM_TOY_LAYERS):
        assert all(p.requires_grad is (i == 2) for p in by_index[i]), i
    # an UNTIED model needs no such workaround, so AUTO leaves the root truly frozen
    untied = apply_active_set(toy_model(tie=False), (2,), layer_other="freeze")
    assert untied["root_tied"] is False
    assert untied["root_requires_grad"] is False
    assert untied["root_grad_masked"] is False


def test_gate2_empty_spec_leaves_every_param_trainable_off_path_parity():
    """The issue #64 off-path parity assertion: dense means dense."""
    from verl.workers.layer_rotation import build_controller

    model = toy_model()
    ctrl = build_controller(model, rank=0, env={}, forward_only=False)
    assert ctrl is None
    assert all(p.requires_grad for p in model.parameters())

    # ... and a forward_only engine is never frozen, even with a schedule set.
    model2 = toy_model()
    assert build_controller(model2, rank=0, env={"LAYER_SCHEDULE": "static:2"}, forward_only=True) is None
    assert all(p.requires_grad for p in model2.parameters())


def test_gate2_out_of_range_and_empty_active_set_raise():
    model = toy_model()
    with pytest.raises(ValueError):
        apply_active_set(model, [NUM_TOY_LAYERS], layer_other="freeze")
    with pytest.raises(ValueError):
        apply_active_set(model, [], layer_other="freeze")


# ===========================================================================#
# Gate 3: FSDP mechanism gate -- the P1-vs-P2 decision.                       #
# ===========================================================================#
def _init_gloo(port: int):
    import torch.distributed as dist

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group("gloo", rank=0, world_size=1)


def _wrap_fsdp(model):
    """FSDP1, ``use_orig_params=True``, transformer auto-wrap: one unit per decoder layer.

    ``device_id=cpu`` keeps the whole gate on CPU (world size 1 => NO_SHARD, which is
    also what a 1-GPU cell gets, so the gate is representative of the real box).
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer})
    return FSDP(model, auto_wrap_policy=policy, use_orig_params=True, device_id=torch.device("cpu"))


def _run_mechanism_trial(mechanism: str, port: int, layer_other: str = "freeze"):
    """Three consecutive rotations over the toy band, returning the five bars."""
    from verl.workers.layer_rotation import build_controller

    _init_gloo(port)
    env = {
        "LAYER_SCHEDULE": "rotate:1-3",
        "ROTATE_EVERY": "1",
        "ROTATE_ADAM": "persist_park",
        "ROTATE_STATE_DEVICE": "cpu",
        "LAYER_OTHER": layer_other,
        "ROTATE_MECHANISM": mechanism,
    }
    model = toy_model(seed=7)
    ctrl = build_controller(model, rank=0, env=env, forward_only=False)
    assert ctrl is not None and ctrl.mechanism == mechanism

    wrapped = _wrap_fsdp(model)
    optimizer = torch.optim.AdamW(ctrl.optimizer_input(wrapped), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    ctrl.bind(wrapped, optimizer, compute_device="cpu")

    bars = {
        "active_grads_finite_nonzero": True,
        "inactive_grads_none_or_zero": True,
        "inactive_params_bit_identical": True,
        "opt_state_two_tensors_per_active_tensor": True,
        "no_fsdp_exception": True,
    }
    detail = []
    try:
        for step in (1, 2, 3):
            ctrl.advance(step)
            active = ctrl.active[0]
            keep_ids = {id(p) for p in ctrl.optimized_params()}
            before = {n: p.detach().clone() for n, p in wrapped.named_parameters() if id(p) not in keep_ids}

            optimizer.zero_grad(set_to_none=True)
            x = toy_batch(seed=step)
            wrapped(input_ids=x, labels=x).loss.backward()

            # pre_step runs the masking (P2) + asserts, BEFORE the clip, exactly as
            # the engine calls it.
            ctrl.pre_step()

            # (a) the active layer's grads are finite and nonzero
            active_grads = [p.grad for p in ctrl.params_of([active])]
            if not active_grads or any(g is None for g in active_grads):
                bars["active_grads_finite_nonzero"] = False
                detail.append(f"step{step}: an active param has no grad")
            else:
                finite = all(bool(torch.isfinite(g).all()) for g in active_grads)
                nonzero = any(float(g.abs().sum()) > 0.0 for g in active_grads)
                if not (finite and nonzero):
                    bars["active_grads_finite_nonzero"] = False
                    detail.append(f"step{step}: active grads finite={finite} nonzero={nonzero}")

            # (b) every other layer's grad is None or exactly 0
            for name, p in wrapped.named_parameters():
                if id(p) in keep_ids or p.grad is None:
                    continue
                if float(p.grad.abs().sum()) != 0.0:
                    bars["inactive_grads_none_or_zero"] = False
                    detail.append(f"step{step}: inactive grad nonzero on {name}")
                    break

            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
            optimizer.step()
            ctrl.post_step()

            # (c) max abs delta of every non-active param across the step is exactly 0
            for name, p in wrapped.named_parameters():
                if id(p) in keep_ids:
                    continue
                ref = before.get(name)
                if ref is None:
                    continue
                delta = float((p.detach() - ref).abs().max())
                if delta != 0.0:
                    bars["inactive_params_bit_identical"] = False
                    detail.append(f"step{step}: non-active param moved by {delta:.3e} on {name}")
                    break

            # (d) optimizer state holds exactly 2 tensors per active-layer param tensor
            active_params = ctrl.params_of([active])
            counts = [
                len([v for v in optimizer.state.get(p, {}).values() if torch.is_tensor(v)]) for p in active_params
            ]
            moment_counts = []
            for p in active_params:
                st = optimizer.state.get(p, {})
                moment_counts.append(sum(1 for k in ("exp_avg", "exp_avg_sq") if torch.is_tensor(st.get(k))))
            if any(c != 2 for c in moment_counts):
                bars["opt_state_two_tensors_per_active_tensor"] = False
                detail.append(f"step{step}: moment tensor counts {moment_counts} (all-tensor counts {counts})")
    except Exception as exc:  # noqa: BLE001 - the bar is "no FSDP exception at all"
        bars["no_fsdp_exception"] = False
        detail.append(f"{type(exc).__name__}: {exc}")
    return bars, detail


def test_gate3_fsdp_mechanism_gate_decides_p1_vs_p2():
    p1_bars, p1_detail = _run_mechanism_trial("p1", port=29711)
    p2_bars, p2_detail = _run_mechanism_trial("p2", port=29711)

    from verl.workers import layer_rotation as lr_mod

    verdict = "p1" if all(p1_bars.values()) else "p2"
    lines = [
        "issue 95 layer-rotation mechanism gate (CPU gate 3)",
        "toy: 4-layer Qwen2, gloo, CPU, use_orig_params=true, transformer auto-wrap,",
        "rotate:1-3, three consecutive rotations (steps 1,2,3).",
        "",
        "P1 = post-wrap requires_grad toggle on orig params AND the owning flat_param",
        "P2 = grad masking before the clip (p.grad = None outside the active set)",
        "",
        "five numeric bars (a) active grads finite+nonzero, (b) inactive grads None/0,",
        "(c) non-active param max abs delta exactly 0, (d) exactly 2 optimizer state",
        "tensors per active param tensor, (e) no FSDP / use_orig_params exception:",
        "",
        "PREREQUISITE FOUND BY THIS GATE (torch 2.12, FSDP1, use_orig_params=true):",
        "a FROZEN root flat_param that owns a TIED tensor breaks on the SECOND",
        "forward -- AssertionError 'as_params=True type(prim_param)=<class Tensor>'",
        "from FlatParamHandle._use_unsharded_views, then NotImplementedError",
        "'Changing shared parameters is not supported yet'. It is triggered by the",
        "frozen root unit alone, with NO rotation logic involved, and it hit BOTH",
        "mechanisms identically before the fix. Fix: under LAYER_OTHER=freeze the root",
        "group (tied embedding + final norm) keeps requires_grad=True and is frozen by",
        "exclusion from the optimizer plus a grad mask before the clip. Weights stay",
        "bit-identical and it holds zero optimizer state; the cost is a dense root",
        "gradient buffer, so the memory verdict is read as success-criteria item 7b",
        "(optimizer term only) with the grad term reported honestly.",
        "",
    ]
    for label, bars, detail in (("P1", p1_bars, p1_detail), ("P2", p2_bars, p2_detail)):
        lines.append(f"{label}:")
        for k, v in bars.items():
            lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")
        if detail:
            lines.append(f"  detail: {detail[:6]}")
        lines.append("")
    lines.append(f"SHIPPED MECHANISM: {verdict}")
    lines.append(f"module DEFAULT_MECHANISM: {lr_mod.DEFAULT_MECHANISM}")
    MECHANISM_FILE.parent.mkdir(parents=True, exist_ok=True)
    MECHANISM_FILE.write_text("\n".join(lines) + "\n")

    # P1 ships if and only if it passes all five bars; otherwise P2 ships. Either
    # way the mechanism the module defaults to MUST be the one this gate chose --
    # a mismatch means the box would run something the gate never validated.
    assert lr_mod.DEFAULT_MECHANISM == verdict, (
        f"module default {lr_mod.DEFAULT_MECHANISM!r} != gate verdict {verdict!r}; P1 bars={p1_bars} detail={p1_detail}"
    )
    # The FALLBACK must itself be sound, otherwise there is nothing to fall back to.
    assert all(p2_bars.values()), f"P2 fallback failed its own bars: {p2_bars} detail={p2_detail}"


# ===========================================================================#
# Gate 4: rotation determinism and visit accounting.                          #
# ===========================================================================#
def test_gate4_rotation_determinism_and_visit_accounting():
    band = RotationSchedule(range(11, 16), rotate_every=1)
    assert band.active_layer(1) == 11
    assert band.active_layer(2) == 12
    assert band.active_layer(6) == 11  # cyclic
    assert band.visit_index(1) == 0 and band.visits_of_active_layer(1) == 1
    assert band.visits_of_active_layer(6) == 2

    counts = band.visit_counts(300)
    assert set(counts) == {11, 12, 13, 14, 15}
    assert all(v == 60 for v in counts.values()), counts
    assert sum(counts.values()) == 300

    allsched = RotationSchedule(range(28), rotate_every=1)
    counts28 = allsched.visit_counts(300)
    assert set(counts28) == set(range(28))
    assert sum(counts28.values()) == 300
    assert all(v in (10, 11) for v in counts28.values()), counts28
    # cyclic order, not a random permutation
    assert [allsched.active_layer(s) for s in range(1, 29)] == list(range(28))
    assert allsched.active_layer(29) == 0

    # determinism: the same step always yields the same layer, and a replayed step
    # is a no-op (advance is pure in the step).
    seq_a = [band.active_layer(s) for s in range(1, 51)]
    seq_b = [band.active_layer(s) for s in range(1, 51)]
    assert seq_a == seq_b

    # ROTATE_EVERY keeps the ticks-per-visit reading separable.
    every2 = RotationSchedule(range(11, 16), rotate_every=2)
    assert [every2.active_layer(s) for s in (1, 2, 3, 4)] == [11, 11, 12, 12]
    assert all(v == 30 for v in every2.visit_counts(300).values())


def test_gate4_controller_advance_is_idempotent_and_matches_the_calendar():
    from verl.workers.layer_rotation import build_controller

    model = toy_model()
    ctrl = build_controller(
        model,
        rank=0,
        env={"LAYER_SCHEDULE": "rotate:1-3", "ROTATE_EVERY": "1"},
        forward_only=False,
    )
    ctrl.bind(model, optimizer=None, compute_device="cpu")
    seen = []
    for step in range(1, 13):
        ctrl.advance(step)
        ctrl.advance(step)  # replay: must not move the schedule
        seen.append(ctrl.active[0])
    assert seen == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert ctrl.rotations == 11  # step 1 is the seeded state, 11 transitions after it


# ===========================================================================#
# Gate 5: state accounting with persist_park.                                 #
# ===========================================================================#
def test_gate5_state_accounting_persist_park():
    from verl.workers.layer_rotation import build_controller

    model = toy_model(seed=3)
    ctrl = build_controller(
        model,
        rank=0,
        env={
            "LAYER_SCHEDULE": "rotate:1-3",
            "ROTATE_EVERY": "1",
            "ROTATE_ADAM": "persist_park",
            "ROTATE_STATE_DEVICE": "cpu",
            "ROTATE_MECHANISM": "p1",
        },
        forward_only=False,
    )
    optimizer = torch.optim.AdamW(ctrl.optimizer_input(model), lr=1e-3)
    ctrl.bind(model, optimizer, compute_device="cpu")

    by_index = decoder_layer_params(model)
    per_layer_bytes = {i: 2 * 4 * sum(p.numel() for p in ps) for i, ps in by_index.items()}

    visited = []
    for step in (1, 2, 3, 4):
        ctrl.advance(step)
        active = ctrl.active[0]
        visited.append(active)
        optimizer.zero_grad(set_to_none=True)
        x = toy_batch(seed=step)
        model(input_ids=x, labels=x).loss.backward()
        ctrl.pre_step()
        optimizer.step()
        ctrl.post_step()

        resident, parked = optimizer_state_bytes_split(
            optimizer, ctrl.parked_params(), park_device="cpu", compute_device="cpu"
        )
        # one_layer_opt_bytes is the analytic 2 x 4 x param-count of the active layer
        analytic_active = one_layer_opt_bytes(model, active)
        assert analytic_active == per_layer_bytes[active]

        # GPU-resident optimizer bytes equal ONE layer's worth (the AdamW `step`
        # scalar per param is a few bytes of bookkeeping, so allow it explicitly
        # rather than pretending it is zero).
        n_active_tensors = len(by_index[active])
        step_scalar_slack = n_active_tensors * 8
        assert analytic_active <= resident <= analytic_active + step_scalar_slack, (active, resident, analytic_active)

        # CPU-parked bytes equal the VISITED-but-not-active set's worth.
        parked_layers = sorted(set(visited) - {active})
        analytic_parked = sum(per_layer_bytes[i] for i in parked_layers)
        parked_slack = sum(len(by_index[i]) for i in parked_layers) * 8
        assert analytic_parked <= parked <= analytic_parked + parked_slack, (parked_layers, parked, analytic_parked)

        # never-visited layers hold nothing at all (Adam state is lazy in torch)
        for i in set(by_index) - set(visited):
            for p in by_index[i]:
                assert not optimizer.state.get(p), i

    telemetry = ctrl.telemetry()
    assert telemetry["layer_rotation/one_layer_opt_bytes"] == float(per_layer_bytes[ctrl.active[0]])
    assert telemetry["layer_rotation/opt_state_bytes_cpu"] > 0.0
    assert set(telemetry) >= {
        "layer_rotation/active_layer",
        "layer_rotation/visit_index",
        "layer_rotation/visits_of_active_layer",
        "layer_rotation/opt_state_bytes_gpu",
        "layer_rotation/opt_state_bytes_cpu",
        "layer_rotation/grad_bytes_gpu",
        "layer_rotation/one_layer_opt_bytes",
        "layer_rotation/trainable_params",
        "layer_rotation/trainable_frac",
    }


def test_gate5_park_helper_moves_moments_and_reports_bytes():
    model = toy_model(seed=5)
    params = list(decoder_layer_params(model)[1])
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    x = toy_batch()
    model(input_ids=x, labels=x).loss.backward()
    optimizer.step()
    analytic = 2 * 4 * sum(p.numel() for p in params)
    # already on cpu, so nothing to move
    assert park_optimizer_state(optimizer, params, "cpu") == 0
    resident, parked = optimizer_state_bytes_split(optimizer, params, park_device="cpu", compute_device="cpu")
    assert parked >= analytic and resident == 0
    assert grad_bytes(model) > 0


# ===========================================================================#
# Gate 6: money-gate visibility (the issue #64 lesson).                       #
# ===========================================================================#
GATE_SOURCES = (
    "verl/workers/layer_rotation.py",
    "verl/workers/engine/fsdp/transformer_impl.py",
    "verl/workers/engine_workers.py",
)


def _literal_text(node):
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out.append(sub.value)
    return "".join(out)


def test_gate6_every_money_gate_line_is_print_flush_never_logger():
    """No ``[layer_rotation]`` message may leave the process through a logger.

    The vast launchers tee stdout into ``train.log`` and drop ``logger.info``. Issue
    #64 lost a full investigation to exactly this, so the gate is mechanical: every
    call whose arguments contain the literal ``[layer_rotation]`` must be ``print``
    with ``flush=True``.
    """
    checked_calls = 0
    for rel in GATE_SOURCES:
        tree = ast.parse((REPO_ROOT / rel).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            text = _literal_text(ast.Module(body=[ast.Expr(value=a) for a in node.args], type_ignores=[]))
            if GATE_PREFIX not in text:
                continue
            checked_calls += 1
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            assert name == "print", f"{rel}: {GATE_PREFIX} message passed to {name!r}, not print()"
            flush = [kw for kw in node.keywords if kw.arg == "flush"]
            assert flush and getattr(flush[0].value, "value", None) is True, (
                f"{rel}: print() of a {GATE_PREFIX} message without flush=True"
            )
    assert checked_calls >= 1, "no money-gate print found at all"

    # The layer_rotation module must not even have a logger to misuse. Checked by
    # AST, not by substring, so prose in a docstring cannot trip or hide it.
    src = (REPO_ROOT / "verl/workers/layer_rotation.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "logging" for a in node.names), "must not import logging"
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "logging", "must not import from logging"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id not in ("logger", "logging"), f"must not use logger.{node.attr}"

    # Every gate message in that module goes through the single print helper.
    prints = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
    ]
    assert len(prints) == 1, f"layer_rotation.py should funnel every gate line through gate_print, found {len(prints)}"
    assert any(kw.arg == "flush" and kw.value.value is True for kw in prints[0].keywords)

    # And there must be a real number of gate call sites, all of them gate_print:
    # the resolved schedule, the active set, trainable=X/Y, the post-wrap bind, the
    # optimizer surface, each rotation event, the step-1 grad-flow assert and the
    # frozen-param immutability check.
    gate_calls = 0
    for rel in GATE_SOURCES:
        tree = ast.parse((REPO_ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "gate_print":
                gate_calls += 1
    assert gate_calls >= 8, f"expected at least 8 money-gate call sites, found {gate_calls}"

    # And the layer-rotation regions of the two engine files log through gate_print
    # too, i.e. they never route a rotation message into `logger`.
    for rel in GATE_SOURCES[1:]:
        text = (REPO_ROOT / rel).read_text()
        for line in text.splitlines():
            if "layer_rotation" in line and "logger." in line:
                raise AssertionError(f"{rel}: layer-rotation line routed through logger: {line.strip()}")


def test_gate6_gate_print_emits_prefixed_flushed_line(capsys):
    from verl.workers.layer_rotation import gate_print

    gate_print("hello gate")
    captured = capsys.readouterr().out
    assert captured.startswith(f"{GATE_PREFIX} hello gate")


def test_gate6_money_gate_lines_are_emitted_on_the_real_build_path(capsys):
    """The six named gate lines must actually appear, not merely be greppable."""
    from verl.workers.layer_rotation import build_controller

    model = toy_model(seed=11)
    ctrl = build_controller(
        model,
        rank=0,
        env={"LAYER_SCHEDULE": "rotate:1-3", "ROTATE_EVERY": "1", "ROTATE_MECHANISM": "p1"},
        forward_only=False,
    )
    optimizer = torch.optim.AdamW(ctrl.optimizer_input(model), lr=1e-3)
    ctrl.bind(model, optimizer, compute_device="cpu")
    ctrl.advance(2)
    x = toy_batch()
    model(input_ids=x, labels=x).loss.backward()
    ctrl.pre_step()
    optimizer.step()
    ctrl.post_step()
    out = capsys.readouterr().out
    for needle in (
        "resolved schedule: rotate:1-3",
        "active set: layers=",
        "trainable=",
        "rotation #1 at step 2",
        "grad-flow OK",
        "immutability OK",
    ):
        assert needle in out, f"missing money-gate line: {needle!r}\n--- captured ---\n{out}"


# ===========================================================================#
# Gate 7: tied-tensor identity (rider), by data_ptr and never by name.         #
# ===========================================================================#
def test_gate7_tied_tensor_identity_post_wrap_by_data_ptr():
    _init_gloo(29711)
    model = toy_model(seed=13, tie=True)
    wrapped = _wrap_fsdp(model)

    in_ptr = wrapped.get_input_embeddings().weight.data_ptr()
    out_ptr = wrapped.get_output_embeddings().weight.data_ptr()
    assert in_ptr == out_ptr, "tie_word_embeddings=true must give ONE tensor for embedding and head"

    def _hits(layer_other):
        ctrl = LayerRotationController(
            parse_layer_schedule("rotate:1-3", NUM_TOY_LAYERS, layer_other=layer_other), NUM_TOY_LAYERS
        )
        optimizer = torch.optim.AdamW(ctrl.optimizer_input(wrapped), lr=1e-3)
        return sum(1 for g in optimizer.param_groups for p in g["params"] if p.data_ptr() == in_ptr)

    # RIDER (LAYER_OTHER=train): the tied matrix appears exactly ONCE across all
    # optimizer param groups -- one tensor, one entry, no double update.
    assert _hits("train") == 1, f"tied matrix appears {_hits('train')} times in the rider optimizer, expected 1"
    # Non-rider cells exclude it from the optimizer entirely, so it cannot be
    # stepped or weight-decayed even if a grad mask were ever skipped.
    assert _hits("freeze") == 0

    # a `lm_head` NAME filter matches ZERO params under default named_parameters()
    # dedup, so a P2 exclude-list must key on `embed_tokens` or the data_ptr.
    by_name = [n for n, _ in wrapped.named_parameters() if "lm_head" in n]
    assert by_name == [], f"expected no lm_head params under dedup, got {by_name}"
    by_embed = [n for n, _ in wrapped.named_parameters() if "embed_tokens" in n]
    assert len(by_embed) == 1, by_embed
    # ... and the NON-deduplicated list does expose both names for the same object,
    # which is exactly what puts the tensor in an FSDP1 handle's _shared_param_infos.
    dup = [n for n, p in wrapped.named_parameters(remove_duplicate=False) if p.data_ptr() == in_ptr]
    assert len(dup) == 2, dup
    from verl.workers.layer_rotation import tied_root_params

    assert [p.data_ptr() for p in tied_root_params(wrapped)] == [in_ptr]

    # and the harness's own root-group discovery finds it by object identity.
    roots = root_params(wrapped)
    assert any(p.data_ptr() == in_ptr for p in roots)


def test_gate7_root_group_is_the_tied_matrix_plus_final_norm():
    model = toy_model(seed=13, tie=True)
    roots = root_params(model)
    names = {n for n, p in model.named_parameters() if any(p is r for r in roots)}
    assert names == {"model.embed_tokens.weight", "model.norm.weight"}, names


# ===========================================================================#
# Gate 8: rider anti-clobber on the 4-layer toy.                              #
# ===========================================================================#
def _build_rider(layer_other: str, seed: int = 21, mechanism: str = "p1"):
    from verl.workers.layer_rotation import build_controller

    model = toy_model(seed=seed)
    ctrl = build_controller(
        model,
        rank=0,
        env={
            "LAYER_SCHEDULE": "rotate:1-3",
            "ROTATE_EVERY": "1",
            "LAYER_OTHER": layer_other,
            "ROTATE_MECHANISM": mechanism,
        },
        forward_only=False,
    )
    optimizer = torch.optim.AdamW(ctrl.optimizer_input(model), lr=1e-3, weight_decay=0.01)
    ctrl.bind(model, optimizer, compute_device="cpu")
    return model, ctrl, optimizer


def test_gate8_rider_root_stays_trainable_and_the_tied_matrix_moves():
    model, ctrl, optimizer = _build_rider("train")
    emb = model.get_input_embeddings().weight
    emb0 = emb.detach().clone()
    assert emb.requires_grad

    for step in (1, 2, 3):
        ctrl.advance(step)
        # root params stay trainable at EVERY tick (the fake-null trap)
        assert all(p.requires_grad for p in root_params(model)), step
        optimizer.zero_grad(set_to_none=True)
        x = toy_batch(seed=step)
        model(input_ids=x, labels=x).loss.backward()
        ctrl.pre_step()
        assert emb.grad is not None and float(emb.grad.abs().sum()) > 0.0, step
        optimizer.step()
        ctrl.post_step()
        assert all(p.requires_grad for p in root_params(model)), step

    # the tied matrix accumulates nonzero updates
    assert float((emb.detach() - emb0).abs().max()) > 0.0
    # ... and carries Adam state (the extra optimizer-checkpoint fingerprint)
    assert optimizer.state.get(emb), "tied matrix has no Adam state"


def test_gate8_rider_step0_trainable_numel_matches_the_analytic_total():
    model, ctrl, _ = _build_rider("train")
    by_index = decoder_layer_params(model)
    active = ctrl.active[0]
    analytic = sum(p.numel() for p in by_index[active]) + sum(p.numel() for p in root_params(model))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == analytic, (trainable, analytic)
    # the same arithmetic on the real model is 46.8M + 233.4M + 1536 ~= 280.2M; the
    # toy reproduces the STRUCTURE (one active layer + tied embedding + final norm).
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert "model.embed_tokens.weight" in names and "model.norm.weight" in names
    assert all(f"model.layers.{i}." not in n for n in names for i in range(NUM_TOY_LAYERS) if i != active)


def test_gate8_masking_independence_train_vs_freeze_give_identical_active_updates():
    """Same seed: LAYER_OTHER=train must not perturb the active layer's step-1 update."""
    deltas = {}
    for layer_other in ("train", "freeze"):
        model, ctrl, optimizer = _build_rider(layer_other, seed=31)
        active = ctrl.active[0]
        before = [p.detach().clone() for p in decoder_layer_params(model)[active]]
        ctrl.advance(1)
        optimizer.zero_grad(set_to_none=True)
        x = toy_batch(seed=99)
        model(input_ids=x, labels=x).loss.backward()
        ctrl.pre_step()
        optimizer.step()
        ctrl.post_step()
        after = decoder_layer_params(model)[active]
        deltas[layer_other] = [(a.detach() - b) for a, b in zip(after, before, strict=True)]

    assert len(deltas["train"]) == len(deltas["freeze"])
    for dt, df in zip(deltas["train"], deltas["freeze"], strict=True):
        assert torch.equal(dt, df), float((dt - df).abs().max())


def test_gate8_rider_asserts_raise_when_the_root_group_is_clobbered():
    model, ctrl, optimizer = _build_rider("train")
    # simulate a rotation implementation that wrongly froze the root group
    for p in root_params(model):
        p.requires_grad_(False)
    with pytest.raises(RuntimeError, match="anti-clobber"):
        ctrl.rider_assert()


# ===========================================================================#
# Gate 9: knob-off regression.                                                #
# ===========================================================================#
def _surface(model, ctrl, optimizer):
    named = list(model.named_parameters())
    trainable = tuple(sorted(n for n, p in named if p.requires_grad))
    groups = tuple(
        (g.get("layer_rotation_group"), tuple(sorted(id(p) for p in g["params"]))) for g in optimizer.param_groups
    )
    numel = sum(p.numel() for _, p in named if p.requires_grad)
    return trainable, groups, numel


def test_gate9_layer_other_freeze_matches_the_pre_rider_path():
    """``LAYER_OTHER=freeze`` and an absent ``LAYER_OTHER`` are the SAME surface.

    The pre-rider code path had no ``LAYER_OTHER`` knob at all: the embedding, the
    tied head and the final norm were permanently frozen. Both the trainable set and
    the optimizer param-group structure must be identical, so the four non-rider
    cells and any rerun are protected.
    """
    from verl.workers.layer_rotation import build_controller

    surfaces = []
    for env in (
        {"LAYER_SCHEDULE": "rotate:1-3", "LAYER_OTHER": "freeze"},
        {"LAYER_SCHEDULE": "rotate:1-3"},  # knob absent entirely
        {"LAYER_SCHEDULE": "rotate:1-3", "LAYER_OTHER": ""},  # knob present but empty
    ):
        model = toy_model(seed=41)
        ctrl = build_controller(model, rank=0, env=env, forward_only=False)
        optimizer = torch.optim.AdamW(ctrl.optimizer_input(model), lr=1e-3)
        ctrl.bind(model, optimizer, compute_device="cpu")
        assert ctrl.schedule.layer_other == "freeze"
        assert not ctrl.root_trainable
        surfaces.append(_surface(model, ctrl, optimizer))

    trainables = {s[0] for s in surfaces}
    assert len(trainables) == 1, trainables
    group_shapes = {tuple((name, len(ids)) for name, ids in s[1]) for s in surfaces}
    assert len(group_shapes) == 1, group_shapes
    assert len({s[2] for s in surfaces}) == 1

    # the shipped group structure is one group per decoder layer plus one `other`
    names = [name for name, _ in surfaces[0][1]]
    assert names == [f"layer{i}" for i in range(NUM_TOY_LAYERS)] + ["other"], names

    # The root group is never optimized and never moves: that is the invariant the
    # pre-rider path guaranteed, and it survives the tied-tensor workaround (which
    # keeps requires_grad=True purely so FSDP1 can run).
    model = toy_model(seed=41)
    ctrl = build_controller(model, rank=0, env={"LAYER_SCHEDULE": "rotate:1-3"}, forward_only=False)
    optimizer = torch.optim.AdamW(ctrl.optimizer_input(model), lr=1e-3)
    ctrl.bind(model, optimizer, compute_device="cpu")
    root_before = [p.detach().clone() for p in root_params(model)]
    for step in (1, 2, 3):
        ctrl.advance(step)
        optimizer.zero_grad(set_to_none=True)
        x = toy_batch(seed=step)
        model(input_ids=x, labels=x).loss.backward()
        ctrl.pre_step()
        # the grad mask ran BEFORE the clip, so the root grads are gone
        assert all(p.grad is None for p in root_params(model)), step
        optimizer.step()
        ctrl.post_step()
    for p, ref in zip(root_params(model), root_before, strict=True):
        assert not optimizer.state.get(p), "root group must hold ZERO optimizer state"
        assert torch.equal(p.detach(), ref), "root group must be bit-identical"
    # and it is never in any optimizer param group in the first place
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert not (opt_ids & {id(p) for p in root_params(model)})


def test_gate9_static_arms_keep_the_issue64_trainable_only_optimizer_surface():
    """Static arms must reproduce the #64 mechanism: only trainable params reach Adam."""
    from verl.workers.layer_rotation import build_controller

    model = toy_model(seed=43)
    ctrl = build_controller(model, rank=0, env={"LAYER_SCHEDULE": "static:1-2"}, forward_only=False)
    assert ctrl is not None and not ctrl.is_rotating
    assert ctrl.active == (1, 2)
    by_index = decoder_layer_params(model)
    expected = {id(p) for i in (1, 2) for p in by_index[i]}
    # the optimizer receives EXACTLY the active block: a flat list (not groups), and
    # nothing that must stay frozen, including the grad-masked tied root group
    payload = ctrl.optimizer_input(model)
    assert isinstance(payload, list) and all(isinstance(p, nn.Parameter) for p in payload)
    assert {id(p) for p in payload} == expected
    optimizer = torch.optim.AdamW(payload, lr=1e-3)
    ctrl.bind(model, optimizer, compute_device="cpu")
    assert ctrl.advance(5) is None  # static schedules never rotate
    assert ctrl.active == (1, 2)
    # ... and after a step the frozen surface holds no state and has not moved
    root_before = [p.detach().clone() for p in root_params(model)]
    x = toy_batch()
    model(input_ids=x, labels=x).loss.backward()
    ctrl.pre_step()
    optimizer.step()
    ctrl.post_step()
    for p, ref in zip(root_params(model), root_before, strict=True):
        assert not optimizer.state.get(p) and torch.equal(p.detach(), ref)
    for i in (0, 3):
        for p in by_index[i]:
            assert not optimizer.state.get(p)


def test_gate9_dense_path_adds_no_metric_keys():
    """A dense engine has no controller, so no ``layer_rotation/*`` key is emitted."""
    from verl.workers.layer_rotation import build_controller

    model = toy_model(seed=47)
    assert build_controller(model, rank=0, env={}, forward_only=False) is None


# ===========================================================================#
# Cross-cutting: the plan's per-cell knob table must resolve as written.       #
# ===========================================================================#
CELL_ENVS = {
    "static-layer14": {"LAYER_SCHEDULE": "static:14"},
    "static-mid5": {"LAYER_SCHEDULE": "static:11-15"},
    "rotate-band5": {"LAYER_SCHEDULE": "rotate:11-15", "ROTATE_EVERY": "1"},
    "rotate-all28": {"LAYER_SCHEDULE": "rotate:0-27", "ROTATE_EVERY": "1"},
    "rotate-band5-embhead": {"LAYER_SCHEDULE": "rotate:11-15", "ROTATE_EVERY": "1", "LAYER_OTHER": "train"},
}


@pytest.mark.parametrize("cell", sorted(CELL_ENVS))
def test_every_cell_env_resolves_on_the_real_28_layer_geometry(cell):
    env = dict(CELL_ENVS[cell])
    sched = schedule_from_env(NUM_REAL_LAYERS, env=env)
    assert sched.spec == env["LAYER_SCHEDULE"]
    assert sched.state_device == "cpu"
    assert sched.adam_policy == "persist_park"
    assert sched.layer_other == ("train" if cell.endswith("embhead") else "freeze")
    if cell.startswith("rotate"):
        rot = RotationSchedule(sched.indices, sched.rotate_every)
        counts = rot.visit_counts(300)
        if cell == "rotate-all28":
            assert all(v in (10, 11) for v in counts.values())
        else:
            assert all(v == 60 for v in counts.values())


def test_real_model_arithmetic_matches_the_plan_numbers():
    """One Qwen2.5-Math-1.5B decoder layer is ~46.8M params => ~0.37 GB of Adam state."""
    hidden, inter, heads, kv_heads, vocab = 1536, 8960, 12, 2, 151936
    head_dim = hidden // heads
    q = hidden * hidden + hidden
    k = hidden * (kv_heads * head_dim) + kv_heads * head_dim
    v = k
    o = hidden * hidden
    mlp = 3 * hidden * inter
    norms = 2 * hidden
    per_layer = q + k + v + o + mlp + norms
    assert 46.0e6 < per_layer < 47.5e6, per_layer
    one_layer_adam_gb = 2 * 4 * per_layer / 1024**3
    assert 0.33 < one_layer_adam_gb < 0.40, one_layer_adam_gb
    # five visited layers => the ~1.87 GB cumulative optimizer checkpoint
    assert 1.70 < 5 * one_layer_adam_gb < 2.05
    # the rider's step-0 trainable numel: active layer + tied matrix + final norm
    tied = vocab * hidden
    rider = per_layer + tied + hidden
    assert 279.0e6 < rider < 281.5e6, rider
    assert 0.17 < rider / 1.54e9 < 0.19
