# Research Status

Two priorities (detail: `PROGRESS.md`, `reports/`, `.claude/GOAL.md`):

1. **M4 — beat the anchor k-collapse by projecting the WEIGHTS** (not the gradient). The dense
   GRPO weight trajectory (raw full weights, bf16, per-tick) is collected and published to R2.
   Current phase: GPU-free offline analysis (#44–#56) measuring weight-projectability on that
   trace; entry point **#44** (the offline sweep engine). Scope:
   `reports/m4-weight-proj-status-and-architecture.md`.
2. **M6 — shrink the ~0.04 compression train–inference mismatch.**
   `reports/priority-2-compression-train-inference-mismatch.html`.

Base: `signed_ema` (α=0.25, β_anc=0.50) at 20/20 anchor latency (k-collapse), PowerSGD r=77. No live boxes.
