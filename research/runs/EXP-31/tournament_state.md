# EXP-31 anchor-usage tournament — live state (orchestrator tracking)

## ⛳ FINAL VERDICT (2026-06-16): **STOP — all 4 anchor-usage levers NULL for surpass**
None cleared B2_live@25=0.7202 at val@25: L4 perturb 0.7157 · L2 μ0.9 **0.5701** (regress) / μ0.5 0.7089 ·
L3 ratio κ1.0 0.7119 / cos κ1.0 0.7134 — all parity-or-below (within/below ±0.024). L1 SKIPPED (gate
fails: cov(G_comp,M)≈0 from F1 + no L2/L3 signal). L3 κ0.5 SKIPPED (monotone-by-cap: bounded between
κ1.0-parity and B2). Code verified GO ⇒ trustworthy nulls. Process criteria PASS (off-path parity,
bytes_ratio==B2, B2 reproduced); HEADLINE surpass FAILS. **Mechanism: B2 caps at parity because δ
reconstructs dense-on-stale-data — you can't exceed dense by reweighting (L3) / accumulating (L2) /
perturbing (L4) / de-noising (L1) a stale estimate of dense.** Full: `runs/EXP-31/verdict.md`.
GPU killed, box clean+idle (operator teardown). All WandB synced.

**Box:** 46.243.55.155:40276 (i_41048644, 4×H200, operator's separate Vast acct ⇒ operator teardown).
**Branches (FINAL 2026-06-16):** ALL code consolidated into `vast-ai-workload` (L4 perturb + L2 δ-momentum
+ L3 adaptive-dose all merged @ origin tip `5d23179a5`; verified, 262 tests pass). **`exp/31` branch DELETED
(local + origin) — only `main` + `vast-ai-workload` remain.** Box i_41048644 → operator may stop/destroy
(all WandB synced incl. recovered val@25 for L2_mom05/L3_ratio_k10). Next session resumes from `vast-ai-workload`.
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
| L2_mom05 | μ=0.5 age_decay | **0.7089** | — (killed@25) | ⏸ PARITY/KILL — −0.011 vs B2_live@25 (within ±0.024), NOT an improvement. Healthy (reward→0.69, len flat ~215). Lighter μ tracks B2 ⇒ **L2 lever CLOSED: NULL for surpass** (μ=0.9 regresses −0.15, μ=0.5 = parity). mem 30.72 (=ceiling) |
| L3_ratio_k10 | ratio κ=1.0 cap2 | **0.7119** | — (killed@25) | ⏸ PARITY/KILL — −0.008 vs B2_live@25 (within ±0.024). Healthy (reward→0.695, len flat ~212, no ignition). Max modulation = neutral. mem 45.97 (adaptive-λ temporaries, no OOM) |
| L3_cos_k10 | cos κ=1.0 cap2 | **0.7134** | — (killed@25) | ⏸ PARITY/KILL — −0.007 vs B2_live@25 (within ±0.024). Healthy (reward→0.756, len↓~181, no ignition). 2nd agreement metric also parity ⇒ **L3 lever CLOSED: NULL**. WandB wmpmmdj1 |
| L3 κ=0.5 (ratio/cos) | — | — | — | SKIP (low-info: κ=1.0 max modulation already = parity ⇒ milder κ lands between that and B2 = parity; κ=0 IS B2) |
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
