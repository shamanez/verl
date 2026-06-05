# Research Status — 2026-06-06T02:15+10:00

## Active
**EXP-25** anchor-circuit default (stale-M + anchor-owns-Q + signed_ema merger). Both hard probe gates PASS; the id-2 α-sweep is LIVE on warm box **39613656** (operator-pinned `-p 40872 root@46.243.55.155`). Sweep chains α∈{0.0,0.3,0.5} back-to-back in tmux `exp25-sweep`; training-log-monitor watching in the background. DO-NOT-PROVISION.

## Issue pipeline
| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 25 | Anchor-circuit default | RUNNING | 1×4H200 (i_39613656, pinned -p40872) | — | id-0 + id-1 probes PASS; α=0.0 @step20/50, auto-chains α=0.3→0.5; winner extends 50→100. |
| 24 | Error-feedback + basis-aligned anchor | BLOCKED | — | — | `depends_on:#25` (needs PASS). |

## EXP-25 sequence
- id-0 (anchor M / R1): **PASS** — coverage 196/196 set-equal, M DP-reduce MEAN (ratio~0.71-0.79 not 4×), anchor-load 338/338 canon-matched, ‖dM‖>0, anchor clean (ratio≡1/no-opt/no-mask/clone-iso), staleness realized_delay from step2.
- id-1 (anchor-owns-Q R2 + signed_ema merger R3): **PASS** — Q anchor-owned + broadcast every refresh (cross_rank_dev=0), fast net never updates Q, M bcast to 196 ranks, merger fires, cold-M fallback wired.
- id-2 (α sweep {0.0,0.3,0.5} ×50 steps, cadence=5/delay_K=5): **α=0.0 RUNNING @step20/50** (train-reward ~0.63-0.72, grad_norm 3.49 finite, spectral_corrections firing, anchor counters clean, no errors); α=0.3 + α=0.5 auto-chained; winner extends 50→100.
- analyst verdict + log-writer + teardown: blocked on id-2.

## Last tick
2026-06-06T02:15+10:00 · running=[25] · analyzing=[] · logging=[] · blocked=[24 dep]

## Budget
1 live box 39613656 (4×H200, $15.23/hr) · max_gpu_hr=48 → ~12h wall-clock headroom · started 00:44:45 (~1.5h elapsed). Tear down at verdict / SSH-unreachable. NEVER reprovision (operator mandate).
