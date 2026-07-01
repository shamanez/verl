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
