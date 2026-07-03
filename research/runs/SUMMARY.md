# Research Runs Summary

North-star: `../.claude/GOAL.md`. Detail: `LOG.md`, `runs/<ID>/verdict.md`, `runs/<ID>/report.html` (published to the cloud-fare site), git.

**M4 spine (all PASS):** dense GRPO weight trajectory — bf16 EXP-43, fp32 EXP-57 (canonical), 160 ticks × 338 matrices, in R2. EXP-44 sweep engine; EXP-45 MOAT scorecard contract; EXP-58 1000-step checkpoint+weight collection; EXP-47 ANCHOR damped-linear lane — R² 0.535/0.335 (between Wang SFT/RL anchors), ρ=−0.75, OOS-damped 0.940 beats naive 1.158 and hold-stale, best_delta=5, h_safe 30 steps (`runs/MOAT-47-ANALYSIS/`).

Next: lanes #48/#49 → #56 MOAT verdict. Stats caches persist on box 43511290.
