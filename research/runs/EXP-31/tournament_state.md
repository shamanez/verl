# EXP-31 anchor-usage tournament — live state (orchestrator tracking)

**Box:** 46.243.55.155:40276 (i_41048644, 4×H200, operator's separate Vast acct ⇒ operator teardown).
**Branches:** `vast-ai-workload` (Cell A, L4 — perturb already wired) + `exp/31-anchor-usage-levers @1d9077d` (L2, L3 — verified, 262 tests pass, on origin).
**Goal:** greedy GSM8K val `mean@1` → 0.80 (surpass; dense band ~0.75–0.78). Single draws ±0.024.

## Reference — B2_live (this box/config, seed 0, disable_custom_all_reduce)
| metric | value | source |
|---|---|---|
| val@0 (untrained) | 0.0910 | A_b2_reproduce |
| **B2_live@25** | **0.7202** | WandB fy920fty |
| B2_live@50 | _TBD (Cell A → ~45min)_ | |

bytes_ratio 0.0505 ✓ · recon_rel_error 0.0278 ✓ · anchor fires ✓ · no ignition.

## L3 calibration (from Cell A B2 telemetry)
- `‖δ‖/‖G_comp_ring‖` (ratio-mode c̄): **≈ 1.025** steady-state (warmup fires 1.15–1.17 excluded).
- `cos(δ, G_comp_ring)`: NOT logged in B2 → L3 **cos-mode** self-calibrates (first tick = B2 by construction).
- rel_change_mean ≈ 0.716.

## Early-decision gate (vs B2_live)
- **val@25:** KILL if `< 0.690` (B2_live@25−0.03) AND reward slope ≤ 0; else CONTINUE.
- **val@50:** EXTEND_TO_100 if `> B2_live@50 + 0.024`; BANK if within ±0.024; KILL if `< B2_live@50 − 0.024`.
- **ignition trip-wires** (dose/buffer cells L2/L3): P1 ≥2 consec cap-pins; P2 len-mean slope>0 sustained; P3 len-mean>2× early; E1 len/max>4k @steps 10–30.

## Cells
| cell | config | val@25 | val@50 | decision |
|---|---|---|---|---|
| A_b2_reproduce | bitwise B2 | **0.7202** | running | REFERENCE |
| L4_perturb_s001 | σ=0.01 | | | queued (chain) |
| L4_perturb_s003 | σ=0.03 | | | queued (chain) |
| L2_mom05 | μ=0.5 age_decay | | | pending (exp/31) |
| L2_mom09 | μ=0.9 age_decay | | | pending (exp/31) |
| L3_cos_k05 / k10 | cos κ=0.5/1.0 cap2 | | | pending (exp/31) |
| L3_ratio_k05 / k10 | ratio κ=0.5/1.0 cap2 | | | pending (exp/31) |
| L1 | recenter/svrg (gated) | | | deferred |

## Launch commands (each = B2 wrapper + one env override)
- L4: `COMM_EFF_SPECTRAL_PERTURB_SIGMA=0.01 EXPERIMENT_NAME=L4_perturb_s001 ... bash <b2_sota>` (vast-ai-workload)
- L2: `COMM_EFF_SPECTRAL_DELTA_MOMENTUM_MU=0.9 COMM_EFF_SPECTRAL_DELTA_MOMENTUM_AGE_DECAY=true EXPERIMENT_NAME=L2_mom09 ...` (exp/31 checkout)
- L3: `COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_MODE=ratio COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_KAPPA=0.5 EXPERIMENT_NAME=L3_ratio_k05 ...` (exp/31 checkout)
