# Research Status — 2026-07-01T13:33:47+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 44 | Weight-proj: extend offline sweep engine | ANALYZING (analyst building engine) | — (kind:analysis, GPU-free) | — | analyst dispatched: builds+verifies research/scripts/weight_proj_sweep.py, streams EXP-43 R2 trace, writes verdict.md |

Other M4 weight-proj issues (#45-#56) are open at kind:analysis but NOT status:approved (awaiting human plan review); #46 is the GPU-gated tier. Only #44 is approved.

## Last tick
2026-07-01T13:33:47+10:00 · running=[] · analyzing=[44] · logging=[] · blocked=[]

## Budget
No Vast box live (runs.jsonl empty). #44 is GPU-free/local — only cost is R2 egress from streaming the EXP-43 trace (one streaming pass by contract).
