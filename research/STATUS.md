# Research Status

Priorities (detail: `.claude/GOAL.md`, `LOG.md`):

1. **M4 — beat anchor k-collapse by projecting WEIGHTS.** fp32 trace (EXP-57) + scorecard contract (#45) done. **#47 PASS:** OOS damped-linear beats hold-stale (0.940 at Δ=10,h=10; λ*=0.3; h_safe=30 steps; best_delta=5); linearity R² 0.535/0.335; R²-vs-ratio ρ=−0.75 — steer projection by per-group R². Remaining lanes #48/#49 → #56 MOAT verdict.
2. **M6 — shrink ~0.04 compression train–inference mismatch.**

Base: `signed_ema` α=0.25, 20/20 anchor latency, PowerSGD r=77. Live: EXTERNAL analysis box 43511290 only (keep; never register RUNNING).
