# Research Runs Summary

Durable record (run dirs de-bloated). North-star: `../.claude/GOAL.md`. Detail: `../reports/*.html`, `LOG.md`, git.

**Baseline** (`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`):
`signed_ema` (α=0.25, β_anc=0.50), PowerSGD r=77 anchor (owns Q), at 20/20 anchor latency — the
**k-collapse regime** where the method fails (parity holds only at 5/5). Values: `FIXED_CONTROL_SURFACE.md`.

**M4 (PASS) — dense full-weight per-tick trajectory (160 snapshots, all 338 matrices).** The spine root
every analysis issue reads. Available in **both precisions** (160/160 R2-verified, `verify_full_weight_dump.py --r2` PASS):
- **bf16** (EXP-43, ~3.1 GB/snap): `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/tick_<N>/tick_<N>.pt`
- **fp32** (EXP-57, ~6.17 GB/snap — the true fp32 master weights; **use this for #44–#56**): `s3://shamane-pluralis/verl-research/EXP-57/regimeA/weights/full/tick_<N>/tick_<N>.pt`  (N = 0..159)

**#44 PASS** — offline sweep engine accepted (`research/scripts/weight_proj/`, self-test `reports/infra-b-sweep-engine-selftest.html`).
Next: **#45–#56** (GPU-free projection science) on the **fp32 EXP-57 trace**. Analysis runs
**download-the-whole-trace-first** on a cheap big-disk, GPU-free box (`weight_proj_sweep.py --trace-root`;
streaming kept for few-snapshot passes like #46). On fp32 the bf16 noise-floor gate is off ⇒ reliability =
projection accuracy + linearity. Fetch/manifests: `weight_proj_fetch_trace.py`, `synth_exp57_manifests.py`.
Access: `reports/r2-access-pattern-for-analysis.md`.

## Milestone M4

Dense weight-trajectory collection + the GPU-free projection-analysis spine (#43 -> #44 -> #45).
>=2 PASS reached; roll-up for operator review. Per-experiment detail: `LOG.md`, `runs/<ID>/verdict.md`.

- **EXP-43 (PASS)** — collected the canonical dense GRPO full-weight per-tick trajectory (160 bf16
  snapshots, all 338 matrices) to R2. Gates: all 5 acceptance gates hold; `verify_full_weight_dump.py
  --r2` PASS (max_rel_norm_err=0.0001 <= 0.01), 160/160 verified, comm_eff counters 0 (codec OFF), no
  NaN/Inf. Deliverable: `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/`.
- **EXP-44 (PASS)** — offline weight-projection sweep engine accepted as the shared #44-#56 substrate.
  Gates: 8/8 success criteria; 15/15 predictor families reconstruct (recon rel-err 0.0); grouping 338
  matrices / 11 blocks / 28 layers exact; one bounded-footprint streaming pass. A mid-run STOP (bf16
  noise-floor category error) was overturned -> corrected differenced floor + directedness p~1.05; core
  blocks clear the floor at h>=5.
- **EXP-45 (PASS)** — MOAT scorecard CONTRACT for lanes #47/#48/#49/#56 (harness correct + complete +
  machine-consumable, not a science claim). Gates: SELFTEST GO, EMIT GO (7 completeness gates), SCHEMA GO
  on box AND laptop. Key metrics: hold-stale identity worst |ratio-1| = 0 over 13377 rows across all 21
  (Delta,h) cells; off-path metric parity worst rel diff 0.00e+00; structure partition exact 338 (other=0);
  42 tied lm_head rows; 26796 atomic rows; 0 denom-guard NaN; 7 visual arrays non-empty. Additive harness
  on vast-ai-workload (`research/scripts/moat_scorecard.py` + `research/scripts/weight_proj/structure.py`).
  Non-gating science readout for #56: naive_linear global (Delta=20,h=40) ratio_median=1.404, h_star=5.
  Run: `runs/MOAT-45-ANALYSIS/`.
