# Research Status — 2026-06-16T17:30+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 33 | β_anc EMA sweep {0,.25,.5,.75,1} on B2 delayed_ef | **RUNNING** | operator 4×H200 (84.8.116.228, op-managed) | — | **C0 DONE: val@25=0.7142, val@50=0.73844 IN B2 band — control PASSES (READ a ✓)**. Bar=0.738; falsified iff a C1/C2/C3 ≥0.7624. C1 (b0p25) starting; substrate clean (coldM=0, recon 0.021, bytes 0.0505, no ignition); C1-C4 skip val@0, C4=30st; bg monitor mon-exp33-e |
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
2026-06-16T22:52:08+10:00 · running=[33 ENV-FAILURE box down ~12:43Z, recovering; C0=0.738,C1=0.740 banked, C2-C4 pending relaunch] · analyzing=[] · logging=[] · blocked=[box-infra]

## Budget
EXP-33 max_gpu_hr=96 (hard cap); ~31.5 GPU-hr projected (4 full cells + C4@30); operator box (op pays/destroys).
