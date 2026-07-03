# Progress

Durable: `runs/SUMMARY.md` · `LOG.md` · `runs/<ID>/report.html` (published to the cloud-fare site) · git. Tick history pruned 2026-07-03 (recoverable from git). Harness appends ticks below.

**Where we are (2026-07-03):** EXP-58 PASS (1000-step checkpoint+weight collection → R2; box torn down). EXP-47 PASS (ANCHOR damped-linear lane; issue closed; verdict + self-explaining report in `runs/MOAT-47-ANALYSIS/`). Band-80 per-tick + band-60 per-step stats caches persist on EXTERNAL box 43511290 for #48/#49/#56 — do not delete; operator-managed, never register RUNNING (no-heartbeat reaper).

Next: plan + run lanes #48/#49, then #56 MOAT verdict rollup.
[2026-07-03T20:47:11+10:00] [research-planner #48] plan written
[2026-07-03T22:37:52+10:00] [analyst #48] verdict=PASS carry_forward=FALSE
PROMOTE_SKIPPED: EXP-48 reason="no promote_launcher_as (kind:analysis)"
MILESTONE_PASS: M4
[2026-07-03T22:43:22+10:00] [log-writer #48] logged verdict=PASS milestone=M4
[2026-07-04T01:15:10+10:00] [research-planner #49] plan written
[2026-07-04T04:32:47+10:00] [analyst #49] verdict=PASS carry_forward=F best=armA_rolling_ls_k_K5 op_ratio=0.9351 Δratio_vs_bar=-0.0045
