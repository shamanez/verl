#!/usr/bin/env python
"""EXP-42 pre-run probe — CPU/GPU-free hard-gate invariants.

Covers the invariants that do NOT need a live FSDP/distributed runtime:
  inv2  limiting-case identity   (strength=0 => theta_hat==theta[t-K]; coeffs)
  inv3  learned cold-start       (learned first fire == fixed_linear)
  inv6  cross-rank (single-proc) (cross_rank_max_rel_dev on identical => 0.0)
  cfg   config validation        (grad_proj requires replay; does NOT force merger=none)
  cfg   Hydra/OmegaConf override merge accepts comm_eff.probe.grad_proj_enabled

The FSDP-integration invariants (1 off-path parity, 4 same-batch, 5 no-leak in
the engine, 7 backend/HBM) are exercised by the GPU smoke (probe_smoke.sh).

Exit 0 iff every check PASSES. Any failure prints FAIL and exits 1.
"""
import sys
import types

import torch

FAILS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


from verl.workers.comm_eff import lookahead as la

FIXED = la.FIXED_LINEAR_COEFFS  # (2,-1,0)


def cfg(mode, strength, anchor_on=True):
    return types.SimpleNamespace(
        lookahead_anchor=anchor_on, lookahead_mode=mode, lookahead_strength=strength
    )


TARGET_SUBSTRS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def make_sources(seed=0):
    g = torch.Generator().manual_seed(seed)
    # two target 2D matrices + one non-target (norm, 1-D) + lm_head (2D but non-target)
    def snap(scale):
        return {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, generator=g) * scale,
            "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 8, generator=g) * scale,
            "model.layers.0.input_layernorm.weight": torch.randn(8, generator=g) * scale,
            "lm_head.weight": torch.randn(8, 8, generator=g) * scale,
        }
    s0 = snap(1.0)   # theta[t-K]
    s1 = snap(1.3)   # theta[t-2K]
    s2 = snap(1.7)   # theta[t-3K]
    return [s0, s1, s2]


# ---- inv2: coeffs ---------------------------------------------------------
p05 = la.LookaheadProjector(cfg("fixed_linear", 0.5), TARGET_SUBSTRS)
check("inv2.coeffs strength=0.5 == (1.5,-0.5,0)", p05.coeffs == [1.5, -0.5, 0.0], str(p05.coeffs))
p10 = la.LookaheadProjector(cfg("fixed_linear", 1.0), TARGET_SUBSTRS)
check("inv2.coeffs strength=1.0 == AsyncPP (2,-1,0)", p10.coeffs == [2.0, -1.0, 0.0] and tuple(FIXED) == (2.0, -1.0, 0.0), str(p10.coeffs))
p00 = la.LookaheadProjector(cfg("fixed_linear", 0.0), TARGET_SUBSTRS)
check("inv2.coeffs strength=0.0 == (1,0,0)", p00.coeffs == [1.0, 0.0, 0.0], str(p00.coeffs))

# ---- inv2: limiting-case identity theta_hat==theta[t-K] at strength 0 -----
src = make_sources()
th0, excl0 = p00.project(src)
tk = src[0]
ident = all(torch.equal(th0[k], tk[k]) for k in tk)
check("inv2.theta_hat==theta[t-K] EXACT at strength=0 (all keys)", ident)
# non-targets are passthrough at ANY strength
th05, excl05 = p05.project(src)
nt_ok = torch.equal(th05["model.layers.0.input_layernorm.weight"], tk["model.layers.0.input_layernorm.weight"]) and \
        torch.equal(th05["lm_head.weight"], tk["lm_head.weight"])
check("inv2.non-targets (norm, lm_head) passthrough theta[t-K]", nt_ok, f"excluded={excl05}")
# target IS extrapolated at strength 0.5: theta_hat = 1.5*tk - 0.5*t2k
q = "model.layers.0.self_attn.q_proj.weight"
expect_q = (1.5 * src[0][q].float() - 0.5 * src[1][q].float()).to(src[0][q].dtype)
check("inv2.target extrapolated theta_hat=1.5*tK-0.5*t2K", torch.allclose(th05[q], expect_q, atol=1e-6))
# weight_proj_ratio==1 at strength 0 (theta_hat==theta[t-K]) for targets
wr = (th0[q].float() - src[2][q].float()).norm() / ((tk[q].float() - src[2][q].float()).norm() + 1e-12)
check("inv2.weight_proj_ratio==1.0 at strength=0 (target q_proj)", abs(float(wr) - 1.0) < 1e-6, f"ratio={float(wr):.8f}")

# ---- inv3: learned cold-start == fixed_linear (residual empty) ------------
pf = la.LookaheadProjector(cfg("fixed_linear", 0.5), TARGET_SUBSTRS)
pl = la.LookaheadProjector(cfg("learned_linear_with_fixed_linear_cold_start", 0.5), TARGET_SUBSTRS)
thf, _ = pf.project(src)
thl, _ = pl.project(src)
cold = all(torch.equal(thf[k], thl[k]) for k in thf)
check("inv3.learned first-fire == fixed_linear (residual=0)", cold and pl.learns and not pf.learns)

# ---- inv6: cross_rank_max_rel_dev single-proc == 0.0 ----------------------
v = torch.tensor([0.1, -0.2, 0.3])
dev = la.cross_rank_max_rel_dev(v)
check("inv6.cross_rank_max_rel_dev single-proc == 0.0", dev == 0.0, f"dev={dev}")

# ---- cfg: validation — grad_proj requires replay, NOT merger=none ----------
import dataclasses as dc

from verl.workers.config.comm_eff import CommEffConfig

# CommEffConfig is a FROZEN dataclass; build via dataclasses.replace so the
# top-level __post_init__ (where the grad_proj validation lives) re-runs with the
# overridden nested values.
def build(replay=True):
    base = CommEffConfig()
    return dc.replace(
        base,
        enabled=True,
        anchor=dc.replace(base.anchor, enabled=True, replay_paired_batch=replay),
        spectral=dc.replace(base.spectral, enabled=True, correction_mode="signed_ema"),
        probe=dc.replace(base.probe, grad_proj_enabled=True),
    )

try:
    build(replay=True)  # signed_ema active + replay + grad_proj => MUST pass
    check("cfg.grad_proj OK with signed_ema active (NOT forced to merger=none)", True)
except Exception as e:
    check("cfg.grad_proj OK with signed_ema active (NOT forced to merger=none)", False, repr(e))

try:
    build(replay=False)
    check("cfg.grad_proj REQUIRES replay_paired_batch (raises when off)", False, "no error raised")
except ValueError as e:
    check("cfg.grad_proj REQUIRES replay_paired_batch (raises when off)", "replay_paired_batch" in str(e))
except Exception as e:
    check("cfg.grad_proj REQUIRES replay_paired_batch (raises when off)", False, repr(e))

# ---- cfg: OmegaConf structured-config accepts the override key -------------
try:
    from omegaconf import OmegaConf
    base = OmegaConf.structured(CommEffConfig())
    merged = OmegaConf.merge(base, OmegaConf.from_dotlist(["probe.grad_proj_enabled=true", "probe.grad_proj_out_dir=/tmp/gp"]))
    check("cfg.OmegaConf merge accepts probe.grad_proj_enabled override",
          bool(merged.probe.grad_proj_enabled) is True and str(merged.probe.grad_proj_out_dir) == "/tmp/gp")
except Exception as e:
    check("cfg.OmegaConf merge accepts probe.grad_proj_enabled override", False, repr(e))

print("\n" + ("PROBE_CPU_RESULT: ALL PASS" if not FAILS else f"PROBE_CPU_RESULT: {len(FAILS)} FAIL -> {FAILS}"))
sys.exit(0 if not FAILS else 1)
