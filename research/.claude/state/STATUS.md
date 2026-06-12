# Research Status — 2026-06-12T17:10:59+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 29 | Anchor on-policy replay (paired batch + CPU snapshots) | RUNNING | 1×4H200 (i_40676027, operator box) | — | 25-step smoke exp29_replay_smoke; branch exp/29-anchor-onpolicy-replay@d311904; CPU+GPU test suites 187 passed each; monitor dispatched |
| 27 | Damped ef_powersgd merger | DONE | — | STOP | lineage closed (tangential-carrier mechanism); box NOT reused (new instance for 29) |
| 26 | EF PowerSGD + Q families | DONE | — | REVISE | ef 0.7210 best realistic, M6 record stands |

## Last tick
2026-06-12T17:10:59+10:00 · running=[29] · analyzing=[] · logging=[] · blocked=[]

## Budget
operator-provided box i_40676027 (~$13.5/hr class); plan caps: max_dph 24, max_gpu_hr 16, wall_clock 4h
