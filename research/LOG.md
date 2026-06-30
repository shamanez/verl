# Research Log

Terse per-experiment verdicts only. Detail lives in `runs/SUMMARY.md`, the `reports/*.html`
summaries, W&B, and git history. Use method names + settings, not old run labels. Current operating
base + the two active priorities are in `PROGRESS.md`.

## EXP-42 · 2026-06-30 · M4 · PASS (measurement)
Weight-projection accuracy of the look-ahead anchor vs horizon, 2 regimes, single 1xH200.
- Regime A (dense GRPO, val@80 0.7695): crossover h*=10 ticks (~5 steps); at K=10 projection helps
  (weight_proj_ratio 0.972, dir_cos 0.549). dir_cos>0 at every horizon, so the overshoot is
  MAGNITUDE, not a weight-space sign flip. RLVR-linear is PARTIALLY true and scale-dependent
  (per-matrix R^2 ~0.80 at 1 tick decaying to ~0.32 at K=10).
- Regime B (PowerSGD r=77, codec-only, val@80 0.0788): h*=5. INVALID as a compressed regime: it ran a
  FROZEN random basis (anchor off + owns_q=true, basis_updates=0, recon ~0.97), a fixed random
  projection, so it collapsed. A config guard now fatals on that combo; the corrected re-run
  (owns_q=false, adaptive Q) is set up in `.claude/plans/42-corrected-rerun-prompt.md`.
- Deliverables: `reports/exp42-*.html` · verdict `.claude/plans/42-verdict.md` · tooling
  `research/scripts/{weight_proj_sweep,build_report,build_dense_report,build_dense_report_v2}.py` ·
  code `exp/42-weight-accuracy`. WandB er0syc3n (A) / 0tpez2fz (B). GitHub issue #42 closed.
- SUPERSEDED 2026-06-30: the count-sketch instrument, the listed sketch tooling, and the
  3 sketch-derived `reports/exp42-{weight-projection-accuracy,dense-weight-behavior,dense-deep-analysis}.html`
  were removed (the raw weights are kept instead). The conclusions above derive from the lossy
  k=4096 sketch — see EXP-43 (raw full-weight collection on R2) for the ground-truth replacement.

## EXP-41 · 2026-06-25 · M4 · STOP
Fixed-linear look-ahead anchor (theta_hat = 2*theta[t-20] - theta[t-40]) at cadence/delay_K=20/20.
Falsified: cos-lift present (+0.027) but cell B collapsed (val@100=0.048, length explosion);
implementation correct (10 pre-run invariants passed). Detail in git + W&B (cell A 7tbzm9kl, B g6dt6bza).
