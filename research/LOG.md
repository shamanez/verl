# Research log — one line per issue (index only)

The per-issue verdict SSOT is the **issue close comment** (+ `runs/<ID>/verdict.md` while
the run dir exists). Detail: `runs/SUMMARY.md`, `runs/<ID>/report.html`, W&B, git history.
Format: `- **<id>** · <date> · <milestone> · <VERDICT> — <one-line result> · #<issue> · PR <url|—>`.

- **EXP-61** · 2026-07-04 · M4 · PASS — Math-only projector ablation: best = adaptive_linear rolling_ls_k K=3 @ Δ*=1,h=1 (ratio 0.9888); deploy fixed damped_linear at freshest anchor; do-nothing optimal at op — confirms dataset-specific · #61 · PR —
- **EXP-56** · 2026-07-04 · M4 · PASS — MOAT rollup: conditional winner = fixed #47 damped-linear (GSM8K op 0.9396, pred_evr +0.1171) but DATASET-SPECIFIC (Big-Math collapses to do-nothing) — NOT a universal ANCHOR default; gate on trajectory coherence · #56 · PR —
- **EXP-60** · 2026-07-04 · M4 · PASS — Big-Math cross-dataset validation: completed negative — NO projector beats do-nothing (GSM8K winner λ*=0.0, ratio 1.0000; consec_delta_cos≈0.15 vs 0.86 GSM8K) · #60 · PR —
- **EXP-49** · 2026-07-04 · M4 · PASS — self-correcting ANCHOR projector: completed negative — armA rolling_ls_k K=5 (op 0.9351) beats fixed bar 0.9396 but sub-threshold (Δ −0.0045 < 0.01) → keep fixed #47 rule · #49 · PR —
- **EXP-48** · 2026-07-03 · M4 · PASS — fixed 2nd-order kill-gate: completed negative — +0.2086 worse than damped-linear (1.1482 vs 0.9396), h_safe −28 → curvature dropped · #48 · PR —
- **EXP-47** · 2026-07-03 · M4 · PASS — ANCHOR damped-linear lane: projection HELPS on GSM8K — OOS-damped 0.940 beats naive 1.158 + hold-stale; λ*=0.3, best_δ=5, h_safe 30 steps; R²=0.535, ρ=−0.75 · #47 · PR —
- **EXP-58** · 2026-07-03 · M4 · PASS — Big-Math 1000-step GRPO collection: 50/50 ckpts + 50/50 fp32 weights verified:true in R2, dry_restore@1000; method-OFF byte-identical · #58 · PR shamanez/verl#20
- **EXP-45** · 2026-07-02 · M4 · PASS — MOAT scorecard CONTRACT GO: SELFTEST/EMIT/SCHEMA GO (26796 rows box+laptop); hold-stale identity exact; structure partition 338 exact · #45 · PR —
- **EXP-44** · 2026-07-01 · M4 · PASS — offline weight-projection sweep engine ACCEPTED 8/8: 15/15 families reconstruct (rel-err 0.0); prior bf16-floor STOP overturned (category error) → differenced floor + directedness (p≈1.05, R²≈0.99) · #44 · PR —
- **EXP-43** · 2026-06-30 · M4 · PASS — dense GRPO FULL bf16 weight collection: 160/160 ticks × 338 matrices in R2, verify 5/5 max_rel_norm_err 1e-4; canonical M4 spine (~492 GB, R2-only) · #43 · PR —
