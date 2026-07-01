# Research Status

Two priorities (detail: `PROGRESS.md`, `reports/`, `.claude/GOAL.md`):

1. **M4 — beat the anchor k-collapse by projecting the WEIGHTS** (not the gradient). The dense
   GRPO weight trajectory is collected to R2 in both precisions; **#45–#56 use the fp32 master
   weights (EXP-57)**. **#44 PASS** — offline sweep engine accepted. Analysis is GPU-free and
   **downloads the whole trace first** to a cheap big-disk box (`weight_proj_sweep.py --trace-root`;
   streaming kept for few-snapshot passes). Access: `reports/r2-access-pattern-for-analysis.md`;
   scope: `reports/m4-weight-proj-status-and-architecture.md`.
2. **M6 — shrink the ~0.04 compression train–inference mismatch.**
   `reports/priority-2-compression-train-inference-mismatch.html`.

Base: `signed_ema` (α=0.25, β_anc=0.50) at 20/20 anchor latency (k-collapse), PowerSGD r=77. No live boxes.
