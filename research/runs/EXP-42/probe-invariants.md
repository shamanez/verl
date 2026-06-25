# EXP-42 — fire-forcing probe verdict (NEW code paths)

Box: private 4×H200 i_42488295. Branch exp/42-lookahead-horizon @ 08bc6d96.
Two probes at cadence=delay_K=1, diagnostics on, both `exit_rc=0`.

## P1 — fixed-α horizon knob (lookahead_strength=0.5) — PASS
- **`strength=0.5000`** surfaced in every per-fire diagnostic ⇒ the NEW knob plumbs end-to-end
  (dataclass + actor.yaml + Hydra passthrough); coeffs = (1.5, −0.5, 0).
- `source_ticks=[t−1, t−2]` every fire, newest < t ⇒ NO leakage.
- `excluded=142`, `targets_extrapolated=196` (LayerNorm/embed exclusion holds); `ring_retained=2 peak=2` (bounded).
- `anchor_align_cos` finite (0.0389, 0.0165, …); anchor isolation counters all 0
  (optimizer_steps/rollouts_generated/rewards_recomputed/mask_applications).

## P2 — learned path (learned_linear_with_fixed_linear_cold_start, never run before) — PASS
- `mode=learned_…`, `source_ticks=[t−1, t−2, t−3]` (3 points), newest < t ⇒ NO leakage.
- **`comm_eff/lookahead_coeff_cross_rank_max_rel_dev = 0.0`** ⇒ learned coeffs are EXACTLY cross-rank
  identical (the determinism invariant the plan demanded for the learned path — PROVEN, not prose).
- `ring_retained=4 peak=4` (3 source snapshots + 1 prior θ̂ for the retrospective residual) — bounded.
- cold-start: residual inits 0 ⇒ first learned fire == fixed prediction (by construction; coeffs (2,−1,0) at strength 1.0).
- `anchor_align_cos` finite (0.027–0.055); anchor isolation counters all 0; max_mem 126 GB < 143 GB (no OOM); no NaN/Traceback; rc=0.

**Decision:** both NEW paths correct (α knob + learned projector). Proceed to scored cells
A25/A50/A75 (fixed_linear α=0.25/0.5/0.75) + L (learned), all delay_K=cadence=20, 100 steps.
