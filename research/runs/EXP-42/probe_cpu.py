#!/usr/bin/env python3
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

"""EXP-42 Phase-1 CPU hard-gate probe (GPU-free).

Runs the CPU-testable correctness invariants from research/.claude/plans/42.md
for the weight-trajectory sketch instrument. Exit code 0 iff every gate passes.

Gates:
  1. off-path parity      — disabled ⇒ no observer; enabled ⇒ observe() never
                            mutates the weights it measures (dump-only) and the
                            disabled path writes nothing.
  2. decoder-only / 196   — exactly the 196 decoder 2-D matrices are selected;
                            LayerNorm / embed / lm_head / bias excluded (same
                            selector as the projector).
  3. predictor parity     — the offline NumPy compute_theta_hat / learned update
                            reproduce the on-device lookahead.py outputs
                            bit-for-bit; limiting cases α=0 ⇒ ratio==1 and
                            learned-first-fire == fixed.
  4. (bonus) sketch fidelity end-to-end — a synthetic linear-ish trajectory run
                            through the real WeightTrajObserver + the offline
                            sweep agrees with the on-box EXACT calib within 5%
                            (the local de-risk of the GPU-phase fidelity gate).

Usage:  python research/runs/EXP-42/probe_cpu.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)  # so `verl` is importable when run as a script
sys.path.insert(0, os.path.join(_REPO, "research", "scripts"))

import weight_proj_sweep as W  # noqa: E402
from verl.workers.comm_eff import lookahead as LA  # noqa: E402
from verl.workers.comm_eff.capture import (  # noqa: E402
    WEIGHT_TRAJ_DEFAULT_SUBSTRS,
    WeightTrajObserver,
    maybe_build_weight_traj_observer,
    select_weight_traj_targets,
)
from verl.workers.config.comm_eff import (  # noqa: E402
    CommEffConfig,
    CommEffProbeConfig,
    CommEffWeightTrajConfig,
)

SUBSTRS = WEIGHT_TRAJ_DEFAULT_SUBSTRS
_FAILS = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _FAILS.append(name)


# --------------------------------------------------------------------- #
# Gate 1 — off-path parity
# --------------------------------------------------------------------- #
def gate_off_path_parity():
    print("Gate 1: off-path parity")
    # disabled config ⇒ no observer, no filesystem touch
    cfg_off = CommEffConfig()
    obs_off = maybe_build_weight_traj_observer(cfg_off)
    check("disabled ⇒ observer is None", obs_off is None)

    # disabled even when comm_eff.enabled=true but weight_traj.enabled=false
    cfg_codec_only = CommEffConfig(enabled=True, compression_type="powersgd")
    check("codec-on + weight_traj-off ⇒ observer None", maybe_build_weight_traj_observer(cfg_codec_only) is None)

    with tempfile.TemporaryDirectory() as d:
        cfg_on = CommEffConfig(
            probe=CommEffProbeConfig(weight_traj=CommEffWeightTrajConfig(enabled=True, out_dir=d, k=512))
        )
        obs = maybe_build_weight_traj_observer(cfg_on)
        check("enabled ⇒ observer built", obs is not None and obs.enabled)
        # observe() must NOT mutate the tensors it is handed (dump-only)
        rng = torch.Generator().manual_seed(1)
        w = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16, generator=rng),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(24, 16, generator=rng),
        }
        before = {k: v.clone() for k, v in w.items()}
        obs.observe(w, global_step=1)
        unchanged = all(torch.equal(w[k], before[k]) for k in w)
        check("observe() does not mutate inputs", unchanged)
        wrote = os.path.exists(os.path.join(d, "manifest.jsonl"))
        check("enabled ⇒ sketch written", wrote)


# --------------------------------------------------------------------- #
# Gate 2 — decoder-only / 196-matrix selection
# --------------------------------------------------------------------- #
def _synthetic_qwen_named_params(n_layers=28, hidden=1536, kv=256, inter=8960):
    """A faithful Qwen2.5-1.5B parameter name/shape set (decoder + the exclusions)."""
    params = []
    # nn.Linear weight is [out_features, in_features]
    shapes = {
        "q_proj": (hidden, hidden),
        "k_proj": (kv, hidden),
        "v_proj": (kv, hidden),
        "o_proj": (hidden, hidden),
        "gate_proj": (inter, hidden),
        "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }
    for L in range(n_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            params.append((f"model.layers.{L}.self_attn.{proj}.weight", torch.zeros(*shapes[proj])))
            # attention projection biases (q/k/v have bias in Qwen2) — 1-D, must be EXCLUDED
            if proj in ("q_proj", "k_proj", "v_proj"):
                params.append((f"model.layers.{L}.self_attn.{proj}.bias", torch.zeros(shapes[proj][0])))
        for proj in ("gate_proj", "up_proj", "down_proj"):
            params.append((f"model.layers.{L}.mlp.{proj}.weight", torch.zeros(*shapes[proj])))
        # LayerNorms — 1-D, EXCLUDED
        params.append((f"model.layers.{L}.input_layernorm.weight", torch.zeros(hidden)))
        params.append((f"model.layers.{L}.post_attention_layernorm.weight", torch.zeros(hidden)))
    # embeddings / final norm / lm_head — EXCLUDED (embed/lm_head 2-D but no substr match)
    params.append(("model.embed_tokens.weight", torch.zeros(151936, hidden)))
    params.append(("model.norm.weight", torch.zeros(hidden)))
    params.append(("lm_head.weight", torch.zeros(151936, hidden)))
    return params


def gate_decoder_selection():
    print("Gate 2: decoder-only / 196-matrix selection")
    params = _synthetic_qwen_named_params()
    selected = select_weight_traj_targets(params, SUBSTRS)
    names = [n for n, _ in selected]
    check("exactly 196 matrices selected", len(selected) == 196, f"got {len(selected)}")
    # every selected is a decoder projection weight, 2-D, substr match
    all_targets = all(any(s in n for s in SUBSTRS) for n in names)
    check("all selected match a decoder substr", all_targets)
    all_2d = all(t.dim() == 2 for _, t in selected)
    check("all selected are 2-D", all_2d)
    # exclusions: no decoder *weight* matrix should be excluded
    excluded = [n for n, _ in params if n not in set(names)]
    bad = [n for n in excluded if any(s in n for s in SUBSTRS) and not n.endswith(".bias")]
    check("no decoder weight matrix excluded", not bad, f"unexpected exclusions: {bad[:3]}")
    norms_excluded = all(
        ("norm" in n) or ("embed" in n) or ("lm_head" in n) or n.endswith(".bias") for n in excluded
    )
    check("norms/embed/lm_head/bias all excluded", norms_excluded)
    # per-type count: 28 each of the 7 types
    by_type = {s: sum(1 for n in names if s in n) for s in SUBSTRS}
    check("28 of each of the 7 matrix types", all(v == 28 for v in by_type.values()), str(by_type))
    # FSDP wrap-infix is canonicalised away
    infixed = [("model.layers.0.self_attn._fsdp_wrapped_module.q_proj.weight", torch.zeros(8, 8))]
    sel_inf = select_weight_traj_targets(infixed, SUBSTRS)
    check("FSDP wrap-infix canonicalised", bool(sel_inf) and "_fsdp_wrapped_module" not in sel_inf[0][0])


# --------------------------------------------------------------------- #
# Gate 3 — predictor parity (NumPy ref vs on-device lookahead.py)
# --------------------------------------------------------------------- #
class _AnchorCfgStub:
    """Minimal anchor-cfg stub enabling the learned look-ahead projector."""

    lookahead_anchor = True
    lookahead_mode = "learned_linear_with_fixed_linear_cold_start"
    lookahead_strength = 1.0


def gate_predictor_parity():
    print("Gate 3: predictor parity (numpy ref vs lookahead.py)")
    rng = torch.Generator().manual_seed(123)
    # fixture: 2 targets (2-D), one excluded norm (1-D), one excluded 2-D (no substr)
    s0 = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(12, 8, generator=rng),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 20, generator=rng),
        "model.layers.0.input_layernorm.weight": torch.randn(12, generator=rng),
        "model.embed_tokens.weight": torch.randn(30, 8, generator=rng),
    }
    s1 = {k: v + 0.01 * torch.randn(v.shape, generator=rng) for k, v in s0.items()}
    sources_t = [s0, s1]
    sources_np = [{k: v.numpy().astype(np.float32) for k, v in s.items()} for s in sources_t]

    def parity_at(alpha, label=""):
        coeffs = W.coeffs_for_alpha(alpha)
        th_t, exc_t = LA.compute_theta_hat(sources_t, coeffs, target_substrs=SUBSTRS, residual=None)
        th_np, exc_np = W.compute_theta_hat_ref(sources_np, coeffs, target_substrs=SUBSTRS, residual=None)
        same_keys = exc_t == exc_np
        maxdiff = 0.0
        for k in th_t:
            a = th_t[k].numpy()
            b = th_np[k]
            maxdiff = max(maxdiff, float(np.max(np.abs(a - b))) if a.size else 0.0)
        check(f"theta_hat parity {label} (α={alpha})", maxdiff <= 1e-6 and same_keys, f"maxdiff={maxdiff:.2e}")
        return th_t

    parity_at(1.0, label="fixed")
    parity_at(0.5, label="under-shoot")

    # limiting case α=0 ⇒ theta_hat == theta_stale ⇒ weight_proj_ratio == 1 exactly
    th_t0 = parity_at(0.0, label="alpha0")
    target = {k: v - 0.02 for k, v in s0.items()}  # arbitrary future point
    ratios = []
    for k, p0 in s0.items():
        if not (any(s in k for s in SUBSTRS) and p0.dim() == 2):
            continue
        num = torch.linalg.norm(th_t0[k] - target[k])
        den = torch.linalg.norm(s0[k] - target[k])
        ratios.append(float(num / den))
    check("α=0 ⇒ weight_proj_ratio == 1", all(abs(r - 1.0) < 1e-6 for r in ratios), f"ratios={[round(r, 6) for r in ratios]}")

    # learned first-fire (residual == {}) == fixed
    th_fix_t, _ = LA.compute_theta_hat(sources_t, W.coeffs_for_alpha(1.0), target_substrs=SUBSTRS, residual=None)
    proj = LA.LookaheadProjector(_AnchorCfgStub(), SUBSTRS)
    th_learn0, _ = proj.project(sources_t)  # residual dict empty on first fire
    maxd = max(float(torch.max(torch.abs(th_fix_t[k] - th_learn0[k]))) for k in th_fix_t)
    check("learned first-fire == fixed", maxd <= 1e-6, f"maxdiff={maxd:.2e}")

    # learned update parity: one retrospective step, numpy ref vs on-device
    th_true_prev = {k: v + 0.05 for k, v in s0.items()}
    r_dev = proj.update_from_retrospective(th_true_prev, th_fix_t)
    r_np = W.learned_update_ref(
        {},
        {k: v.numpy() for k, v in th_true_prev.items()},
        {k: v.numpy() for k, v in th_fix_t.items()},
        target_substrs=SUBSTRS,
    )
    keys = set(r_dev) | set(r_np)
    maxr = max((abs(float(r_dev.get(k, 0.0)) - float(r_np.get(k, 0.0))) for k in keys), default=0.0)
    check("learned residual update parity", maxr <= 1e-7, f"maxdiff={maxr:.2e}")


# --------------------------------------------------------------------- #
# Gate 4 (bonus) — end-to-end sketch fidelity vs exact calib (synthetic)
# --------------------------------------------------------------------- #
def gate_sketch_fidelity_synthetic():
    print("Gate 4 (bonus): end-to-end sketch fidelity vs exact calib (synthetic trajectory)")
    rng = np.random.default_rng(0)
    # small synthetic model: 4 decoder matrices, a near-linear trajectory with noise
    shapes = {
        "model.layers.0.self_attn.q_proj.weight": (48, 32),
        "model.layers.0.self_attn.k_proj.weight": (16, 32),
        "model.layers.0.mlp.gate_proj.weight": (64, 32),
        "model.layers.0.mlp.down_proj.weight": (32, 64),
    }
    theta0 = {n: rng.standard_normal(s).astype(np.float32) * 0.02 for n, s in shapes.items()}
    vel = {n: rng.standard_normal(s).astype(np.float32) * 1e-4 for n, s in shapes.items()}  # linear drift
    n_ticks = 40
    delta, h = 10, 10
    with tempfile.TemporaryDirectory() as d:
        obs = WeightTrajObserver(
            out_dir=d, k=4096, dump_dtype="fp32",
            target_substrs=SUBSTRS, calib_deltas=(delta,), calib_horizons=(h,),
            calib_stride=delta + h, calib_max_snapshots=6, rank=0, rank0_only=True,
        )
        for t in range(n_ticks):
            w = {
                n: torch.from_numpy(
                    theta0[n] + vel[n] * t + (rng.standard_normal(shapes[n]).astype(np.float32) * 2e-5)
                )
                for n in shapes
            }
            obs.observe(w, global_step=t // 2)
        sweep = W.sweep_regime(d, deltas=(delta,), horizons=(h,), methods=("fixed_linear", "learned_linear"))
        calib = W.validate_against_calib(sweep, d, tol=0.05)
        check("calib.jsonl produced on-box", calib.get("available"), str(calib.get("available")))
        if calib.get("available"):
            for c in calib["checks"]:
                check(
                    f"sketch≈exact within 5% (Δ={c['delta']},h={c['h']})",
                    c["pass"],
                    f"sketch={c['sketch_w1_p50']:.4f} calib={c['calib_w1_p50']:.4f} rel={c['rel_err']:.2%}",
                )
        cell = sweep["results"].get(("fixed_linear", delta, h))
        check("sweep produced a headline cell", cell is not None and cell["n"] > 0, f"n={cell['n'] if cell else 0}")


def main():
    print("=" * 72)
    print("EXP-42 Phase-1 CPU hard-gate probe")
    print("=" * 72)
    gate_off_path_parity()
    gate_decoder_selection()
    gate_predictor_parity()
    gate_sketch_fidelity_synthetic()
    print("=" * 72)
    if _FAILS:
        print(f"RESULT: FAIL ({len(_FAILS)} gate check(s) failed): {_FAILS}")
        sys.exit(1)
    print("RESULT: ALL GATES PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
