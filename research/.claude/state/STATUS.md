# Research Status — 2026-06-16T17:30+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 33 | β_anc EMA sweep {0,.25,.5,.75,1} on B2 delayed_ef | **RUNNING** | operator 4×H200 (84.8.116.228, op-managed) | — | C0 (b0p00) **val@25=0.7142 — reproduces B2** (ref 0.7202, Δ−0.006); reward 0.13→0.69, resp_len↓277→207, no ignition, all controls GREEN; step 25/55, val@50 ~10:00Z; C1-C4 skip val@0 (reuse 0.0819), C4=30st; full sweep ~17-18Z; bg monitor mon-exp33-c |
| 32 | signed_ema α=0.5 on valid-M | DONE | operator (op-managed) | (closed status:done) | result 0.7271 < B2 0.7528; stale RUNNING row flipped COMPLETE |
| 31 | anchor-usage 4-lever tournament | DONE | — | STOP | all-null for surpass; B2=SOTA |

## EXP-33 cell plan (sequential, ONE box)
- C0 b0p00 β=0.00 → 55 steps (control; val@50 must reproduce B2 band [0.716,0.774]; gross-failure gated <0.55)
- C1 b0p25 β=0.25 → 55 steps (curve point)
- C2 b0p50 β=0.50 → 55 steps (curve point)
- C3 b0p75 β=0.75 → 55 steps (curve point)
- C4 b1p00 β=1.00 → 30 steps (degenerate bracket: frozen-zero M → no-merger floor 0.6300±0.024; val@25 read)
- Hypothesis: β=0 weakly optimal (freshness > averaging). Falsified iff any C1/C2/C3 val@50 ≥ C0 + 0.024.

## Last tick
2026-06-16T19:23:35+10:00 · running=[33 C0@38/55 healthy] · analyzing=[] · logging=[] · blocked=[] · awaiting C0 val@50 ~09:50Z

## Budget
EXP-33 max_gpu_hr=96 (hard cap); ~31.5 GPU-hr projected (4 full cells + C4@30); operator box (op pays/destroys).
