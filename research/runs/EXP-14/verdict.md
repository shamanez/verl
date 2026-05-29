# EXP-14 Verdict — 2026-05-29

VERDICT: PASS on DIAGNOSIS only. The explosion is localized + its grad_norm is fixable (rescale), BUT the masked method does NOT yet TRAIN — see the Learning-check correction below. grad_norm-bounded ≠ learning.

Operator-capped at 10 trainer steps/cell. Full report posted to GitHub issue #14 (CLOSED, status:pass) + a follow-up CORRECTION comment; follow-ups tracked in #15.

## ⚠️ Learning check (operator follow-up) — CORRECTION to the "rescale = FIXED" claim
A tamed grad_norm is necessary but NOT sufficient. Val GSM8K acc@1 (step0→step10, same window for all cells):
- test1_cellA (dense): 0.083 → **0.721** ✅ learns
- test2_cellA (mask, no rescale): 0.083 → 0.079 ❌ no
- **test2_cellD (mask + rescale): 0.080 → 0.084 ❌ NO** (grad_norm tamed to 1.5, but entropy frozen ~5.9, score flat ~0.13)
- test2_cellF (clean_cadence=2): 0.080 → **0.672** ✅ learns (via the clean/unmasked steps)
=> rescale converts a LOUD failure (explosion) into a QUIET one. Masking 90% of boundary activations makes the forward a near-random surrogate that doesn't transfer to the unmasked eval policy. Only clean_cadence learns — via full-bandwidth clean steps, which undercuts the comm-efficiency premise. REAL open question (#15): can masked GRPO learn at all (mask-rate sweep), judged on val/score, not grad_norm.

## Outcome
- **Gate (test1):** comm-eff OFF reproduces dense (test1_cellA gn 0.35 ≈ dense 0.36, score 0.14→0.73); scaffold backend-clean (test1_cellB gn 0.34 ≤ 1.0). The explosion is in the method, not the scaffold.
- **Peel (test2):** pure masked GRPO explodes (test2_cellA gn 771→838, anchor/spectral OFF, no learning) → the blow-up is the mask itself.
- **ROOT CAUSE:** the mask `h*mask` (p=0.9, no rescale) collapses boundary-block RMS to √(1-p)≈0.32× → out-of-distribution magnitude shift → ~771 grad_norm. Not an IS/RNG artifact.
- **grad_norm fix (NOT a training fix):** inverted-dropout rescale `h*mask/(1-p)` → test2_cellD gn **1.49**, ppo_kl≈0 — but it does NOT recover learning (val flat, see correction above). `mask.rescale=true` is a precondition, not the answer.
- **Refuted:** `consistent_across_forwards` (test2_cellC gn 882) — the positional mask + differing phase packing means equal seed ≠ equal mask. Cross-pass consistency is instead structural via **per-channel masking** (`granularity=channel`, default; `exp/14-mask-per-channel` @ a3590173; unvalidated on a run → #15).
- **clean_cadence (test3_cellB c=10, test2_cellF c=2):** masked steps still explode from step 1 (640–1480); the clean step only resets the optimizer state — a safety net, not a fix. `rescale` fixes every step.

## FSDP-no-errors: PASS — zero FSDP/DTensor/NaN/OOM across all 8 cells.

## Cells (wandb): exp14-test1_cellA, exp14-test1_cellB, exp14-test2_cellA, exp14-test2_cellB, exp14-test2_cellC, exp14-test2_cellD, exp14-test2_cellF, exp14-test3_cellB. Metrics: runs/EXP-14/metrics/*.jsonl.

## Recommended default: mask.rescale=true + mask.granularity=channel + mask.seed settable + clean_cadence=0; anchor+spectral OFF until #15 validates longer-horizon mask-only.

## Box 38370788 (4×H200) torn down (reason: session-complete-exp14).
