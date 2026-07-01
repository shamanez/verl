# Research Status — 2026-07-01T15:11+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 44 | Weight-proj: extend offline sweep engine (orders≥2, damped-α, learnable-at-every-order, regression, EMA; new GPU-free metrics) | ANALYZING | — (kind:analysis, GPU-free) | — | analyst re-dispatched (async). Engine built (`weight_proj/` pkg + CLI); invariants gate PASS 5/5 hard + 1 soft; manifests resolved 160/160. Prior bg analyst died mid-noise-floor gate → finishing gate2 (noise-floor) + gate3 (full-sweep self-test) BOUNDED → HTML + verdict.md. |
| 43 | Collect dense regime-A full-weight per-tick trace → R2 | DONE | 1×H200 (i_43197578, TEAM) TORN_DOWN | PASS | M4 spine trace, 160/160 snapshots R2-verified; sole input to #44. |

Other M4 weight-proj issues (#45-#56) are open at kind:analysis but NOT status:approved (awaiting human plan review). Only #44 is approved.

## Last tick
2026-07-01T15:11+10:00 · running=[] · analyzing=[44] · logging=[] · blocked=[]

## Budget
No live Vast.ai box. Teardown sweep rc=0 (no-op). EXP-44 is GPU-free (R2 egress only). Ledger: EXP-43 rows ABORTED + TORN_DOWN.
