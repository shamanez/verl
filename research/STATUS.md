# Research Status

Focused on **two priorities**. Detail + evidence: `PROGRESS.md`. North-star: `.claude/GOAL.md`.

1. **Solve the k-collapse by projecting the weights** (milestone M4). The stale anchor gradient
   rotates orthogonal by k≈10–20; fix via weight-space projection (learned + error-feedback), gated by
   a GPU-free offline cosine-lift kill-test. `reports/priority-1-anchor-staleness-k-collapse.html`.
   **Current step (EXP-42, reframed 2026-06-29):** before any gradient claim, MEASURE how accurately
   `θ̂=(1+α)θ[t−K]−αθ[t−2K]` projects the WEIGHTS vs raw-stale, as a function of steps-ahead, in 2
   regimes (plain GRPO / +activation-compression), fixed vs learned. Cheap by design — ordinary
   training on **one 1×H200** emits a tiny per-tick weight sketch; the look-ahead is replayed
   **offline on the MacBook**. This gates EXP-43 (gradient accuracy). Plans: `.claude/plans/{42,43}.md`.
2. **Reduce the compression-induced train–inference mismatch (Gap A)** (milestone M6). Bounded ~0.04
   tax; shrink it / switch on truncated-IS. `reports/priority-2-compression-train-inference-mismatch.html`.

**Basic setup:** EMA merger (`signed_ema` α=0.25, β_anc=0.50) on the fast 1K surface at HIGH anchor
latency (cadence/delay_K = 20/20, the k-collapse regime), PowerSGD r=77 anchor substrate. Values:
`runs/FIXED_CONTROL_SURFACE.md`.

_(The orchestrator loop rewrites this file each tick when running.)_
