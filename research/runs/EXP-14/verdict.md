# EXP-14 Verdict — 2026-05-29

VERDICT: PASS on DIAGNOSIS only. The explosion is localized + its grad_norm is fixable (rescale), BUT the masked method does NOT yet TRAIN — see the Learning-check correction below. grad_norm-bounded ≠ learning.

Operator-capped at 10 trainer steps/cell. Full report posted to GitHub issue #14 (CLOSED, status:pass) + a follow-up CORRECTION comment; follow-ups tracked in #15.

## ⚠️ Learning check (operator follow-up) — CORRECTION to the "rescale = FIXED" claim
A tamed grad_norm is necessary but NOT sufficient. Val GSM8K acc@1 (step0→step10, same window for all cells):
- test1_cellA (dense): 0.083 → **0.721** ✅ learns
- test2_cellA (mask, no rescale): 0.083 → 0.079 ❌ no
- **test2_cellD (mask + rescale): 0.080 → 0.084 ❌ NO** (grad_norm tamed to 1.5, but entropy frozen ~5.9, score flat ~0.13)
- test2_cellF (clean_cadence=2): 0.080 → 0.672 ❌ NOT SUSTAINABLE — the rise is the clean steps only; masked-step pg_clipfrac climbs 0.26→0.44 (still rising) ⇒ PPO clip saturation ⇒ learning will die (see Clip-saturation correction below)
=> rescale converts a LOUD failure (explosion) into a QUIET one. Masking 90% of boundary activations makes the forward a near-random surrogate that doesn't transfer to the unmasked eval policy. clean_cadence's apparent learning is the clean (full-bandwidth) steps alone and is NOT sustainable (clip-fraction saturation, below) — it also undercuts the comm-efficiency premise. REAL open question (#15): can masked GRPO learn at all (mask-rate sweep), judged on stable low pg_clipfrac + sustained val/score, not grad_norm.

## ⚠️ Clip-saturation correction — cellF (clean_cadence=2) does NOT work
GRPO uses PPO ratio clipping. On masked steps the ~90% activation drop makes π_new/π_old blow up, so a large + GROWING fraction of tokens is clipped (gradient → 0):
- cellF masked steps (s1,3,5,7,9): grad_norm 897→804→951→1145→**1480** (rising); pg_clipfrac 0.255→0.277→0.359→0.397→**0.439** (rising); clean steps (s2,4,6,8,10): grad_norm ~0.4, pg_clipfrac ~0.
- dense (test1_cellA): pg_clipfrac ≈ 0.001 throughout.
Mechanism: masked-step clip fraction rises toward saturation → masked steps contribute ~zero gradient → only the clean steps learn while the masked steps thrash the policy (zig-zag grad_norm) → at saturation, no learning. The early score bump is the clean steps doing the work, not the masked method. clean_cadence (c=10 AND c=2) is therefore NOT a viable fix.

## Outcome
- **Gate (test1):** comm-eff OFF reproduces dense (test1_cellA gn 0.35 ≈ dense 0.36, score 0.14→0.73); scaffold backend-clean (test1_cellB gn 0.34 ≤ 1.0). The explosion is in the method, not the scaffold.
- **Peel (test2):** pure masked GRPO explodes (test2_cellA gn 771→838, anchor/spectral OFF, no learning) → the blow-up is the mask itself.
- **ROOT CAUSE:** the mask `h*mask` (p=0.9, no rescale) collapses boundary-block RMS to √(1-p)≈0.32× → out-of-distribution magnitude shift → ~771 grad_norm. Not an IS/RNG artifact.
- **grad_norm fix (NOT a training fix):** inverted-dropout rescale `h*mask/(1-p)` → test2_cellD gn **1.49**, ppo_kl≈0 — but it does NOT recover learning (val flat, see correction above). `mask.rescale=true` is a precondition, not the answer.
- **Refuted:** `consistent_across_forwards` (test2_cellC gn 882) — the positional mask + differing phase packing means equal seed ≠ equal mask. Cross-pass consistency is instead structural via **per-channel masking** (`granularity=channel`, default; `exp/14-mask-per-channel` @ a3590173; unvalidated on a run → #15).
- **clean_cadence (test3_cellB c=10, test2_cellF c=2) — NOT a viable fix:** masked steps explode from step 1 (640–1480) AND their PPO `pg_clipfrac` climbs monotonically (cellF 0.26→0.44 over 5 masked steps, still rising) → as clipping saturates, masked-step tokens contribute zero gradient → learning stalls. The clean step only resets the optimizer; the apparent score rise is the clean steps thrashing against the masked ones (zig-zag grad_norm), not the method learning.

## FSDP-no-errors: PASS — zero FSDP/DTensor/NaN/OOM across all 8 cells.

## Cells (wandb): exp14-test1_cellA, exp14-test1_cellB, exp14-test2_cellA, exp14-test2_cellB, exp14-test2_cellC, exp14-test2_cellD, exp14-test2_cellF, exp14-test3_cellB. Metrics: runs/EXP-14/metrics/*.jsonl.

## Recommended default: mask.rescale=true + mask.granularity=channel + mask.seed settable + clean_cadence=0; anchor+spectral OFF until #15 validates longer-horizon mask-only.

## Box 38370788 (4×H200) torn down (reason: session-complete-exp14).
