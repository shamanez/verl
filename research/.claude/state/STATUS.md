# Research Status — 2026-06-16T17:30+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 33 | β_anc EMA sweep {0,.25,.5,.75,1} on B2 delayed_ef | **RUNNING** | operator 4×H200 (84.8.116.228, op-managed) | — | C0 (b0p00) launched; off-axis parity confirmed (beta_anc=0.00 passthrough last-wins); 5 cells chain sequentially in tmux exp-33-84_8_116_228; bg monitor mon-exp33-a active |
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
2026-06-16T17:30+10:00 · running=[33] · analyzing=[] · logging=[] · blocked=[]

## Budget
EXP-33 max_gpu_hr=96 (hard cap); ~31.5 GPU-hr projected (4 full cells + C4@30); operator box (op pays/destroys).
