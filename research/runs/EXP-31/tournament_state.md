# EXP-31 anchor-usage tournament — live state (orchestrator tracking)

**Box:** 46.243.55.155:40276 (i_41048644, 4×H200, operator's separate Vast acct ⇒ operator teardown).
**Branches:** `vast-ai-workload` (Cell A, L4 — perturb already wired) + `exp/31-anchor-usage-levers @1d9077d` (L2, L3 — verified, 262 tests pass, on origin).
**Goal:** greedy GSM8K val `mean@1` → 0.80 (surpass; dense band ~0.75–0.78). Single draws ±0.024.

## Reference — B2_live (this box/config, seed 0, disable_custom_all_reduce)
| metric | value | source |
|---|---|---|
| val@0 (untrained) | 0.0910 | A_b2_reproduce |
| **B2_live@25** | **0.7202** | WandB fy920fty |
| **B2_live@50** | **0.7354** | WandB fy920fty (Cell A DONE) |

Dense-this-box = 0.7506 (prior verdict) ⇒ B2 at parity (gap 0.015 < ±0.024). SURPASS target 0.80.
val@50 gate: EXTEND>0.7594 · BANK 0.7114–0.7594 · KILL<0.7114. val@25 gate: KILL<0.690.

bytes_ratio 0.0505 ✓ · recon_rel_error 0.0278 ✓ · anchor fires ✓ · no ignition.

## L3 calibration (from Cell A B2 telemetry)
- `‖δ‖/‖G_comp_ring‖` (ratio-mode c̄): **≈ 1.025** steady-state (warmup fires 1.15–1.17 excluded).
- `cos(δ, G_comp_ring)`: NOT logged in B2 → L3 **cos-mode** self-calibrates (first tick = B2 by construction).
- rel_change_mean ≈ 0.716.

## Early-decision gate (vs B2_live)
- **val@25:** KILL if `< 0.690` (B2_live@25−0.03) AND reward slope ≤ 0; else CONTINUE.
- **val@50:** EXTEND_TO_100 if `> B2_live@50 + 0.024`; BANK if within ±0.024; KILL if `< B2_live@50 − 0.024`.
- **ignition trip-wires** (dose/buffer cells L2/L3): P1 ≥2 consec cap-pins; P2 len-mean slope>0 sustained; P3 len-mean>2× early; E1 len/max>4k @steps 10–30.

## Cells (handoff 2026-06-16 — L2/L3/L1 remaining; code MERGED to vast-ai-workload)
| cell | config | val@25 | val@50 | decision |
|---|---|---|---|---|
| A_b2_reproduce | bitwise B2 | 0.7202 | **0.7354** | ✅ REFERENCE (WandB fy920fty) |
| L4_perturb_s001 | σ=0.01 | 0.7157 | — (stopped@34) | ✅ BANKED NULL — parity within noise; isotropic = regularization control, not anchor-usage (WandB cvu8jw1n partial) |
| L4_perturb_s003 | σ=0.03 | — | — | SKIPPED (more isotropic noise ⇒ ≤ σ=0.01) |
| L2_mom09 | μ=0.9 age_decay | **0.5701** | — (killed@26) | ❌ KILL — −0.15 below B2_live@25 (0.7202), far outside ±0.024. Code verified GO ⇒ TRUSTWORTHY null: heavy μ=0.9 over-smooths the held correction ⇒ lags the fast-changing early correction, slows convergence. Healthy (reward→0.51, len 276→150↓, no ignition). WandB ybemd5ux |
| L2_mom05 | μ=0.5 age_decay | | | 🟢 RUNNING (launched 2026-06-16) — lighter momentum; clean start, knob active |
| L3_ratio_k10 / k05 | ratio κ=1.0/0.5 cap2 | | | pending |
| L3_cos_k10 / k05 | cos κ=1.0/0.5 cap2 | | | pending |
| L1 | recenter/svrg (gated) | | | deferred — needs code (transformer_impl.py) |

**Handoff (2026-06-16):** current run STOPPED + box CLEANED per operator. L2/L3 code merged into
`vast-ai-workload` @ 36d7f60c1 (exp/31 branch deleted, worktree removed). Box 46.243.55.155:40276 idle +
clean on that HEAD, GPU 0%. A fresh `/goal` session (same prompt) resumes: launch L2 → L3 → L1 per
§HANDOFF STATE in `.claude/plans/31.md`. WandB current (Cell A + L4 synced). Final analyst verdict +
promotion pending after L2/L3/L1.

## Code verification (2026-06-16, adversarial 8-agent workflow vs plan math) — GO/GO
- **L2 GO:** gain-1 normalized EMA `m←μm+(1−μ)δ` (fixed pt m*=δ, NOT 10×), accumulate ONLY at refresh
  ticks (`if refreshed:` ⟺ anchor-fire t%5==0 via cadence/delay_K arithmetic), age-decay→G_comp,
  cross-rank deterministic, off-path parity bitwise. `spectral_filter.py:753-831`.
- **L3 GO:** mean-1 centered gate `λ_t=λ+κ(c̄−c_t)`, c̄=running median (NOT forbidden `1+κ(1−cos)`),
  ratio=‖δ‖/‖gm‖ from `delta_raw` captured before L2 transform, clamp[0,cap], off-path parity.
  `spectral_filter.py:836-899`.
- **No critical defect; nothing invalidates a null.** A NULL on L2/L3 is a TRUSTWORTHY null (the lever
  faithfully did what the plan intends). ⇒ interpret tournament results at face value.

## Launch commands (each = B2 wrapper + one env override)
- L4: `COMM_EFF_SPECTRAL_PERTURB_SIGMA=0.01 EXPERIMENT_NAME=L4_perturb_s001 ... bash <b2_sota>` (vast-ai-workload)
- L2: `COMM_EFF_SPECTRAL_DELTA_MOMENTUM_MU=0.9 COMM_EFF_SPECTRAL_DELTA_MOMENTUM_AGE_DECAY=true EXPERIMENT_NAME=L2_mom09 ...` (exp/31 checkout)
- L3: `COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_MODE=ratio COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_KAPPA=0.5 EXPERIMENT_NAME=L3_ratio_k05 ...` (exp/31 checkout)
