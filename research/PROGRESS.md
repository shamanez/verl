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
[2026-07-04T04:39:32+10:00] [log-writer #49] logged verdict=PASS carry_forward=FALSE best=armA_rolling_ls_k_K5 milestone=M4
[2026-07-04T10:52:13+10:00] [research-planner #60] plan written
[2026-07-04T17:57:29+10:00] [analyst #60] verdict=PASS carry_forward=F best=hold_stale(do-nothing) op_ratio=1.0000 d_vs_bar=-0.0001 generalizes=dataset-specific (consec_delta_cos=0.15)
[2026-07-04T18:00:54+10:00] [log-writer #60] logged verdict=PASS carry_forward=FALSE generalizes=dataset-specific milestone=M4
[2026-07-04T18:15:14+10:00] [analyst #56] verdict=PASS (integration: GSM8K winner=fixed damped-linear 0.9396; cross-dataset=dataset-specific, Big-Math bar 1.0000/lam0.0/consec0.15)
[2026-07-04T18:32:34+10:00] [log-writer #56] logged verdict=PASS MOAT-rollup conditional-winner=fixed-damped-linear cross-dataset=dataset-specific milestone=M4; report runs/MOAT-56-ANALYSIS/report.html
[2026-07-04T22:26:58+10:00] [research-planner #61] plan written
[2026-07-05T03:15:18+10:00] [analyst #61] verdict=PASS — best Math projector = adaptive_linear rolling_ls_k K=3 @ Delta*=1,h=1 (ratio 0.98879 CI[0.98763,0.99043], evr +0.02203; deploy fixed damped_linear by prefer-simplicity); PROJECTABLE only at freshest anchor (20 global steps), do-nothing optimal at op Delta=5,h=10 — confirms #60 dataset-specific, consec_delta_cos~0.15
[2026-07-05T03:25:15+10:00] [log-writer #61] LOG.md + runs/SUMMARY.md (M4) updated — EXP-61 PASS: best Math projector adaptive_linear rolling_ls_k K=3 @ Δ*=1,h=1; deploy fixed damped_linear; do-nothing optimal at op, confirms #60 dataset-specific
[2026-07-05T10:03:23+10:00] [operator] teardown EXP-61 analysis box 43511290 (202.122.49.242, RTX A4000, private) — destroyed=1, private account now empty; final MOAT lane done. FLAG: team account has 2 UNTRACKED live boxes ~$143/day (5090 43810572 + B200 43847853), not part of any MOAT lane — awaiting operator direction.
[2026-07-05T11:07:58+10:00] [operator] team-account boxes 43810572 (5090) + 43847853 (B200) belong to ANOTHER user — operator says DO NOT tear down; vast-cost 'possible LEAK' flag is a false positive for these. STATUS.md updated.
