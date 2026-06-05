# Research Status — 2026-06-06T00:45+10:00

## Active
**EXP-25** anchor-circuit default (stale-M + anchor-owns-Q + signed_ema merger). Code complete on `vast-ai-workload@107ca01` (R1+R2+R3). Resumed on the OPERATOR-PROVIDED warm box **39613656** (replaced nuked 39602487 at same IP). experiment-runner driving the probe→sweep sequence in the background.

## Issue pipeline
| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 25 | Anchor-circuit default | RUNNING | 1×4H200 (i_39613656, reused) | — | DO-NOT-PROVISION (operator mandate: ONLY 46.243.55.155:40872). Box prepped: code synced, secrets pushed, RUNNING ledger row registered. |
| 24 | Error-feedback + basis-aligned anchor | BLOCKED | — | — | `depends_on:#25` (needs PASS). |

## EXP-25 sequence
- id-0 (anchor M / R1): IN PROGRESS (≤3-step probe, cadence=1/delay_K=1) — gates id-1
- id-1 (anchor-owns-Q R2 + signed_ema merger R3): blocked on id-0
- id-2 (α sweep {0.0,0.3,0.5} ×50 steps, cadence=5/delay_K=5): blocked on id-1
- analyst verdict + log-writer: blocked on id-2

## Last tick
2026-06-06T00:45+10:00 · running=[25] · analyzing=[] · logging=[] · blocked=[24 dep]

## Budget
1 live box 39613656 (4×H200, $15.23/hr) · max_gpu_hr=48 → ~12h wall-clock headroom · started 00:44:45. Tear down at verdict / SSH-unreachable. NEVER reprovision (operator mandate).
