# Progress

Durable record: `runs/SUMMARY.md` · `runs/FIXED_CONTROL_SURFACE.md` · `reports/*.html` · W&B · git.
North-star: `.claude/GOAL.md`. Per-run verdicts: `LOG.md`. Tick history pruned (recoverable from
git + LOG.md); the harness appends new ticks below.

**Base:** `signed_ema` (α=0.25, β_anc=0.50), fast 1K surface, HIGH anchor latency
(cadence/delay_K=20/20, the k-collapse regime), locked PowerSGD r=77 anchor. Values:
`runs/FIXED_CONTROL_SURFACE.md`.

**Two priorities:** (1) M4 — solve the k-collapse by projecting WEIGHTS; the dense weight trajectory
is collected → R2; GPU-free analysis spine #44–#56, entry **#44**. (2) M6 — shrink the ~0.04
compression train–inference mismatch.
[2026-07-01T12:58:54+10:00] [research-planner #44] plan written
[2026-07-01T12:59:34+10:00] [triage] dispatched 1 planner (#44), 0 issues already planned; now planned=1 unplanned=0 ALL_PLANNED
