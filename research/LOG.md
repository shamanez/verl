# Research Log

Terse per-experiment verdicts only. Detail lives in `runs/SUMMARY.md`, the `reports/*.html`
summaries, W&B, and git history. Use method names + settings, not old run labels. Current operating
base + the two active priorities are in `PROGRESS.md`.

## EXP-45 · 2026-07-02T16:39:37+10:00 · M4 · PASS
MOAT: Minimal projection scorecard plus block/layer structure for EXP-57 — the shared scorecard CONTRACT for lanes #47/#48/#49 and verdict #56 (NOT a science claim).
- hypothesis: contract/correctness, not performance — the shared replay harness over the EXP-57 fp32 trace (160 optimizer-tick snapshots, 338 matrices) is correct + complete + machine-consumable by hard numeric gates (metric-contract pin, hold-stale identity, exact structure partition, full schema/aggregates/visuals, finite naive-linear). NO algorithm-performance threshold asserted; whether projection helps is #47's science.
- result: PASS — MOAT scorecard contract GO. SELFTEST GO (all hard invariants; determinism soft-gate PASS), EMIT GO (7 completeness gates), SCHEMA GO on BOTH box AND laptop (26796 rows, schema portable). Contract ready for #47/#48/#49 to register methods against `moat_scorecard.py`'s plugin interface and for #56 to render.
- key numbers: hold-stale identity worst |ratio−1| = 0.000e+00 and worst |skill| = 0.0 over 13377 in-bounds rows across all 21 (Δ∈{5,10,20}, h∈{1,2,5,10,20,30,40}) cells; off-path metric parity worst rel diff 0.00e+00 (surrogate path == direct predictors.Order1 + metrics.full_metric_row); structure partition EXACT 338 (block_type sum 338, super_block sum 338, other=0); 42 tied lm_head rows (all tied_to=embed, not silently dropped); 26796 atomic rows emitted; 0 denom-guard NaN over 13398 naive-linear rows; 0 bounds-honesty violations, min n_windows=100 (worst cell Δ=20,h=40); 7 visual-data arrays all present + non-empty.
- science readout (NON-gating, recorded for #56/#47): naive_linear global (Δ=20,h=40) ratio_median = 1.404, h_star = 5, best_delta = 5 — projection does not beat hold-stale at these horizons; that reading belongs to #47, not #45. A ratio>1 / h_star=0 is a VALID contract output.
- compute: GPU-free weight-geometry replay; full-trace streaming pass 99.8 min (bounded footprint — 22 chunks, RAM cap 40 GB, peak ~31 GB RSS, no OOM) on the EXTERNAL operator-managed analysis box 43511290 (RTX A4000, shared with #47/#48/#49/#56 — NOT torn down by #45). 0 of 3 allowed harness-fix cycles used (clean first run).
- code: additive-only harness landed DIRECTLY on `vast-ai-workload` per the plan's `## Code change` (research/ is writable there — NO exp/* branch, NO PR, no head branch to PR from): `research/scripts/moat_scorecard.py` + `research/scripts/weight_proj/structure.py`. Protected `weight_proj/{metrics,predictors,sweep,report}.py` + `weight_proj_sweep.py` unmodified (imported, not edited); no verl/ source touched; box harness md5-identical to laptop.
- run dir: runs/MOAT-45-ANALYSIS/
- verdict: runs/MOAT-45-ANALYSIS/verdict.md

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
