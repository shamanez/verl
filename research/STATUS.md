# Research Status

Focused on **two priorities**. Detail + evidence: `PROGRESS.md`. North-star: `.claude/GOAL.md`.

1. **Solve the k-collapse by projecting the weights** (milestone M4). The stale anchor gradient
   rotates orthogonal by k≈10–20; fix via weight-space projection (learned + error-feedback), gated by
   a GPU-free offline cosine-lift kill-test. `reports/priority-1-anchor-staleness-k-collapse.html`.
   **Current phase (M4 analysis):** the shared dense GRPO weight trajectory has been collected — the
   RAW full weights of every floating param (bf16, ~338 params, every optimizer tick over 80 steps)
   are published to R2. The analysis spine (offline weight-projection sweep engine + predictor
   families + per-layer/block reports) is GPU-free and reads that trace layer/block-wise; it MEASURES
   how accurately `θ̂=(1+α)θ[t−K]−αθ[t−2K]` (and richer predictors) project the WEIGHTS vs raw-stale as
   a function of steps-ahead. Architecture + access contract:
   `reports/m4-weight-proj-status-and-architecture.md` + `reports/r2-access-pattern-for-analysis.md`.
2. **Reduce the compression-induced train–inference mismatch (Gap A)** (milestone M6). Bounded ~0.04
   tax; shrink it / switch on truncated-IS. `reports/priority-2-compression-train-inference-mismatch.html`.

**Basic setup:** EMA merger (`signed_ema` α=0.25, β_anc=0.50) on the fast 1K surface at HIGH anchor
latency (cadence/delay_K = 20/20, the k-collapse regime), PowerSGD r=77 anchor substrate. Values:
`runs/FIXED_CONTROL_SURFACE.md`.

_(The orchestrator loop rewrites this file each tick when running.)_
