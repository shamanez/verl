# Research Status — 2026-05-28 (EXP-7 verdict logged)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 7 | M2 spectral correction + FSDP grad-point discovery | DONE | — | PASS | findings/M2/EXP-7.md; PR #4 draft pending; FSDP discovery: full 2D Tensor via use_orig_params=true, correction AFTER reduction/BEFORE clipping, world_size=4 |
| 8 | M2 anchor circuit K-stale refresh | NO_STATUS | — | — | no status label — not approved; skip |
| 9 | M2 full M95+AP two-step smoke | NO_STATUS | — | — | depends on spectral+anchor; not approved; skip |
| 10 | M3 DP gradient compression scope | NO_STATUS | — | — | after M95+AP smoke; not approved; skip |
| 11 | M3 100-step M95+AP vs dense | NO_STATUS | — | — | not approved; skip |
| 6 | M2 mask contamination guard | DONE | — | PASS | PR #3 merged; findings/M2/EXP-6.md |
| 5 | M2 actor-only mask smoke | DONE | — | PASS | PR #2 merged; findings/M2/EXP-5.md |
| 3 | M1 dense GRPO baseline | DONE | — | PASS | milestone:M1, val 0.087→0.789 |

## Last tick
2026-05-28 · verify=[] · running=[] · analyzing=[] · logging=[7 verdict-logged] · blocked=[8,9,10,11 not-approved]

## Budget
EXP-7 completed: lifetime_spent_usd 9.6085 / monthly_cap 1500 (not exhausted); instance torn down.
