# Research Status

Focused on **two priorities**. Detail + evidence: `PROGRESS.md`. North-star: `.claude/GOAL.md`.

1. **Solve the k-collapse by projecting the weights** — issue #39 (M4). The stale anchor gradient
   rotates orthogonal by k≈10–20; fix via weight-space projection (learned + error-feedback), gated by
   a GPU-free offline cosine-lift kill-test. `reports/priority-1-anchor-staleness-k-collapse.html`.
2. **Reduce the compression-induced train–inference mismatch (Gap A)** — issue #40 (M6). Bounded ~0.04
   tax; shrink it / switch on truncated-IS. `reports/priority-2-compression-train-inference-mismatch.html`.

**Basic setup:** EMA merger (`signed_ema` α=0.25, β_anc=0.50) on the 2K accel surface, PowerSGD r=77
anchor substrate. Values: `runs/FIXED_CONTROL_SURFACE.md`.

_(The orchestrator loop rewrites this file each tick when running.)_
