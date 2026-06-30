# Research Log

Terse per-experiment verdicts only. Detail lives in `runs/SUMMARY.md`, the `reports/*.html`
summaries, W&B, and git history. Use method names + settings, not old run labels. Current operating
base + the two active priorities are in `PROGRESS.md`.

## EXP-43 · 2026-06-30T22:36+10:00 · M4 · PASS
Weight-proj: Collect the shared dense GRPO FULL weight matrices (all params, bf16, every optimizer TICK, regime A) to R2.
- hypothesis: collection unit (not a hypothesis test); ACCEPTED iff all five acceptance gates on the captured artifact hold.
- result: PASS, all gates hold. 160/160 bf16 full-model snapshots (n_matrices=338, real shapes, NOT a sketch/subset) published + R2-verified; the canonical M4 weight-proj spine root every issue #44-#56 cites.
- deliverable: canonical R2 trace `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/`, key form `tick_<N>/tick_<N>.pt` (~492 GB; per-step trajectory = first tick of each global_step). Heavy .pt live in R2 ONLY; never pulled to laptop.
- integrity: `verify_full_weight_dump.py --r2` PASS (5/5 sampled, max_rel_norm_err=0.0001 <= 0.01 tol); r2_manifest 160/160 verified:true. comm_eff counters all 0 on every step (codec OFF confirmed). No NaN/Inf. done.flag rc=1 = benign atexit DataLoader teardown after step 80, NOT a failure.
- run: dense regime-A (codec OFF), Qwen2.5-1.5B-Instruct + GSM8K, vanilla GRPO, 80/80 steps. WandB run a51waqza backfilled to step 80; final GSM8K val acc 0.7809 (dense-control band 0.75-0.78, provenance only — this is a collection unit, not a comparison arm).
- box: external 1xH200 inst 43197578 (TEAM account) TORN_DOWN; run cost ~$18.60 (5.08 h x $3.6635/hr).
- promote: none (collection probe; the deliverable is the trace, not a config) — no launcher-promotion PR.
- run dir: runs/EXP-43/
- verdict: runs/EXP-43/verdict.md

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
