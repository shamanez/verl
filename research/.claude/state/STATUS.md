# Research Status — 2026-05-28 (EXP-7 launch tick)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 7 | M2 spectral correction + FSDP grad-point discovery | RUNNING | 1×4H200 (i_38122735, $14.74/hr) | — | verify=CONCERNS→VERIFIED; exp/7-spectral-fsdp-discovery pushed; tmux exp-7-156_19_254_2; weights loaded, entering 2-step smoke; FSDP grad-repr discovery = AFTER reduction/BEFORE clipping |
| 8 | M2 anchor circuit K-stale refresh | NO_STATUS | — | — | no status label — not approved; skip |
| 9 | M2 full M95+AP two-step smoke | NO_STATUS | — | — | depends on spectral+anchor; not approved; skip |
| 10 | M3 DP gradient compression scope | NO_STATUS | — | — | after M95+AP smoke; not approved; skip |
| 11 | M3 100-step M95+AP vs dense | NO_STATUS | — | — | not approved; skip |
| 6 | M2 mask contamination guard | DONE | — | PASS | PR #3 merged; findings/M2/EXP-6.md |
| 5 | M2 actor-only mask smoke | DONE | — | PASS | PR #2 merged; findings/M2/EXP-5.md |
| 3 | M1 dense GRPO baseline | DONE | — | PASS | milestone:M1, val 0.087→0.789 |

## Last tick
2026-05-28 · verify=[] · running=[7] · analyzing=[] · logging=[] · blocked=[8,9,10,11 not-approved]

## Budget
$/hr now: $14.74 (1 box, i_38122735 4×H200) · EXP-7 cap: 8 GPU-hr / 3 wall-clock-hr / max_dph $24
