# Research Log

Terse per-experiment verdicts only. Detail lives in `runs/SUMMARY.md`, the `reports/*.html`
summaries, W&B, and git history. Use method names + settings, not old run labels. Current operating
base + the two active priorities are in `PROGRESS.md`.

## EXP-44 · 2026-07-01T18:40+10:00 · M4 · PASS
Weight-proj: Offline weight-projection sweep engine (`research/scripts/weight_proj_sweep.py` + `weight_proj/` pkg) — orders>=2, damped-alpha, learnable-at-every-order, general regression, EMA; full GPU-free metric hierarchy.
- hypothesis: infra-unit acceptance — engine ACCEPTED iff all predictor families reconstruct from the raw R2 trace AND the bf16-noise-floor gate passes (operational falsifier for the shared #45-#56 substrate).
- result: PASS, 8/8 success criteria. Engine ACCEPTED as the shared substrate for #45-#56. 15/15 families reconstruct (recon rel-err 0.0); grouping 338 matrices / 11 blocks / 28 layers exact; one streaming pass, bounded footprint (max 0 staged .pt, no recursive cp); deliverable HTML populates all 15 family curves + corrected gate table + new PuLSE table.
- STOP OVERTURNED: the prior STOP (`BF16_FLOOR_BLOCKS: down_proj,q_proj`, commit 04edbe11) rested on a bf16 noise-floor CATEGORY ERROR — it compared the differenced residual ||e||~1e-2 against the ||theta||-scaled STORAGE floor ~0.4-0.5 (597x down_proj / 2167x q_proj over-estimate; the ||theta||/||e|| factor that MUST cancel in a difference of two correlated bf16 snapshots). Independently refuted by a READ-ONLY R2 per-element probe.
- fix: corrected to a DIFFERENCED-noise floor on the CHANGED support (unchanging tensor differences to EXACTLY 0.0 = empirical null) + a DIRECTEDNESS discriminator (fixed-origin cumulative displacement ~h^p; q_proj p=1.05, down_proj p=1.04, R^2~0.99 => DIRECTED signal, which bf16 rounding noise at p~0.5 cannot produce; matches the verification probe's p~1.07) + first-class sparse-subset (PuLSE) metrics. Both moving core blocks CLEAR the floor at h>=5.
- finding (VALID, not an engine failure): weight_proj_ratio>1 / h*=0 for the polynomial extrapolators (order1/2/3, EMA) = bf16 RLVR weight motion is DIRECTED but NOT polynomially extrapolable at these horizons; surfaced through the sparsity/PuLSE lens (1.7-2.2% elems move/step, 15-16% by >=3 ULP) for #52-#56. Do NOT STOP on "no dense skill".
- regime: bf16 stays the FIXED regime — NO fp32 re-collection; fp32 deferred, tracked as issue #57. Fix in the METRIC only, committed 71500bd5 on vast-ai-workload (research/scripts/ only; kind:analysis, GPU-free, no Vast run).
- run dir: runs/EXP-44/
- verdict: runs/EXP-44/verdict.md

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
