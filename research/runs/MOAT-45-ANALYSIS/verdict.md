# Verdict EXP-45 — 2026-07-02T16:39:37+10:00

## Result
VERDICT: PASS

`kind: analysis` contract verdict. This is a CONTRACT check ("correct + complete +
machine-consumable"), NOT a science claim. Every box in the plan's `## Success criteria`
is satisfied, so `moat_scorecard.py` + `weight_proj/structure.py` are ready for lanes
#47/#48/#49 to plug methods into and for #56 to render from. Per the plan's
`## Notes for analyst`, naive_linear ratio > 1 or h_star != 0 is a VALID output and does
NOT gate #45 (observed: naive_linear global (Δ=20,h=40) ratio_median = 1.404, h_star = 5
— that is #47's science, read through #56).

## Success criteria
**Correctness gates (hard):**
- [x] harness asserts `METRIC_CONTRACT == "weight-proj-metrics-v1"` and reuses those functions (observed: `INVARIANT metric_contract_pin: PASS`; meta.json `metric_contract="weight-proj-metrics-v1"`; off-path parity worst rel diff 0.00e+00 proves surrogate path == direct `predictors.Order1`+`metrics.full_metric_row`)
- [x] hold-stale identity: median `weight_proj_ratio == 1.0 ± 1e-6` AND `skill == 0 ± 1e-6` for every (Δ∈{5,10,20}, h∈{1,2,5,10,20,30,40}) cell (observed: `GATE hold_stale_identity: PASS — 13377 rows, worst |ratio-1| = 0.000e+00, violations = 0`; independent laptop recompute over scorecard.jsonl: max |ratio−1| = 0.0, max |skill| = 0.0 across all 21 cells)
- [x] structure partition exact + `count(other)==0` (observed: meta.json `structure_block_type_counts` = {q_proj:28,k_proj:28,v_proj:28,o_proj:28,gate_proj:28,up_proj:28,down_proj:28,norm:57,bias:84,embed:1,lm_head:0,other:0} sum 338; `structure_super_block_counts` = {attention:112,mlp:84,norm:57,bias:84,embed:1,lm_head:0,other:0} sum 338; `GATE structure_partition: PASS — 338-partition exact (other=0)`)
- [x] explicit special rows for embed, norm, bias, AND lm_head (lm_head tied) (observed: schema `special` kind = 4 groups {embed,norm,bias,lm_head}; 42 lm_head rows, all `tied=true`/`tied_to=embed`; `GATE aggregates_present: PASS — lm_head rows=42 (tied)`)
- [x] causality/leakage: θ̂ uses only ticks ≤ t, scoring point strictly later, fit-leakage assert wired (observed: `INVARIANT causality_leakage_guard: PASS — valid split fit_idx=[4,5,6]<score=7; overlap-guard-fires=True; fit_window_positions wired (max fit idx 99 < anchor 100)`)
- [x] finite + bounds honesty: naive_linear finite (only denom-guard NaN allowed), every in-bounds cell `n_windows>0`, worst cell Δ=20,h=40 ≥100, OOB cells flagged (observed: `GATE naive_linear_finite: PASS — 13398 rows, denom-guard NaN windows = 0`; `GATE bounds_honesty: PASS — violations=0, zero=0`; laptop recompute: min n_windows over in-bounds rows = 100, worst cell (Δ=20,h=40) n_windows = 100, out-of-bounds rows = 0)

**Completeness gates (hard):**
- [x] atomic table (JSONL) one row per (method,Δ,h,group_kind,group_key) with all required keys (observed: `SCHEMA: GO`; 26796 rows, all 29 `REQUIRED_ROW_KEYS` present per-row incl structure axes, minimal metrics median+p10+p90, diagnostics, bookkeeping n_windows/in_bounds/n_nan_windows, derived h_star/best_delta, tied)
- [x] required aggregates all present (observed: schema kinds = {global:1, matrix:338, block_type:10, super_block:5, layer:28, layer_block:252, special:4}; `SCHEMA: GO` validates need_bt⊆block_type, need_sb⊆super_block, all 28 layers, 252≥7×28 layer_block for decoder (layer_idx,block_type), embed/norm/bias/lm_head specials, 338 matrix rows)
- [x] coverage safety: every group emits delta_norm + coverage, none dropped for near-static (observed: `GATE coverage_safety: PASS — rows missing delta_norm/coverage = 0`)
- [x] 7 visual-data arrays all emitted + non-empty (observed: visuals.json present+non-empty for all of a_accuracy_vs_horizon, b_delta_sensitivity, c_target_horizon_sweep, d_traj_r2, e_ratio_heatmap, f_hstar_heatmap, g_special_groups; `SCHEMA: GO` re-checks non-emptiness)
- [x] operating point Δ=20,h=20 AND fast example Δ=10,h=10 emitted (observed: meta.json `operating_point=[20,20]`, `also_points=[[10,10]]`; `GATE operating_points: PASS — missing=[]`; schema op-row presence check passes for both points × both methods)
- [x] `--verify-schema` round-trips + `SCHEMA: GO` (observed: box `SCHEMA: GO` (SCHEMA_BOX_EXIT=0) AND laptop `SCHEMA: GO` (LAPTOP_SCHEMA_EXIT=0) on the rsynced dir — schema portability confirmed, identical rows=26796/kinds/NaN=0)
- [x] harness code ADDITIVE under `research/scripts/` only; metrics.py/predictors.py/sweep.py unmodified; no verl/ touched (observed: `git status --porcelain` + `git diff --stat HEAD` on metrics.py/predictors.py/sweep.py/report.py/weight_proj_sweep.py both EMPTY; moat_scorecard.py + structure.py committed/clean; box harness md5-identical to laptop)

## Metrics summary
- hold_stale identity: worst |ratio−1| = 0.000e+00 over 13377 in-bounds rows (target ≤ 1e-6); worst |skill| = 0.0 (target ≤ 1e-6)
- off-path parity (surrogate vs direct predictors+metrics): worst rel diff 0.00e+00 synthetic + 0.00e+00 real (target ≤ 1e-6)
- structure partition: block_type sum = 338, super_block sum = 338, other = 0 (target: exact counts, other=0)
- naive_linear finite: 0 denom-guard NaN windows over 13398 rows (target: 0 non-denom-guard NaN)
- bounds honesty: 0 n_windows-formula violations, 0 zero-window in-bounds rows, 0 out-of-bounds rows in the required grid; min n_windows = 100 (worst cell Δ=20,h=40; target ≥ 100)
- atomic rows emitted: 26796 (338 matrix + 28 layer + 252 layer_block + 10 block_type + 5 super_block + 4 special + 1 global, × 2 methods × 21 cells, + 42 tied lm_head)
- lm_head tied rows: 42, all tied_to=embed
- 7 visual arrays: all present + non-empty
- schema round-trip: GO on box AND laptop (26796 rows both)
- self-test: SELFTEST: GO — all 10 hard invariants PASS + determinism (soft) PASS
- naive_linear science readout (NON-gating, for #56): global (Δ=20,h=20) and (Δ=20,h=40) ratio_median = 1.404, h_star = 5, best_delta = 5 — projection does not beat hold-stale at these horizons; that reading is #47's, not #45's

## Comparisons to baseline_run: none
No baseline. Per the plan, EXP-45 is the parent scorecard CONTRACT (a weight-geometry
replay over the EXP-57 fp32 trace), not a dense-vs-method comparison; there is no
baseline arm. `diff_against_baseline.py` is n/a for a `kind: analysis` contract verdict.

## Resolved parameters (ground truth)
No Vast training run / no `train.log` `main_ppo` trace exists for this GPU-free analysis,
so `capture_resolved_config.py` (which parses a launcher's `set -x` trace) does not apply.
The ground-truth "what actually ran" is the emit CLI, recorded in `scorecard/meta.json`
(the harness serializes its fully-resolved config) and in `analysis.log`:

```
methods            = ["hold_stale", "naive_linear"]
delta_ticks        = [5, 10, 20]
h_ticks            = [1, 2, 5, 10, 20, 30, 40]
operating_point    = [20, 20]     also_points = [[10, 10]]
n_ticks            = 160          band = 60
trace_root         = /workspace/trace/EXP-57
manifest           = runs/EXP-57/regimeA/weights/full_manifest.jsonl  (338 matrices, fp32)
metric_contract    = weight-proj-metrics-v1
stats_cache_fingerprint = 55defd52e25fa70d
```

Divergence check vs the plan's `## Verification commands`: NONE. Every launched knob
(method set, Δ grid, h grid, --operating-point 20,20, --also 10,10, --trace-root) matches
the plan's step-4 command verbatim. Box harness md5 == laptop harness md5 for
moat_scorecard.py / structure.py / metrics.py, so the code that produced the tables is the
reviewed code.

## Notes
- The full streaming pass over the ~987 GB fp32 all-160 trace took 99.8 min (bounded
  footprint: 22 chunks, ram cap 40 GB, peak ~31 GB RSS observed, 92 GB free throughout —
  no OOM). Stats cache `stats_cache.npz` (18 MB) was written; a re-emit on the same
  --trace-root would reuse it.
- Results are on the laptop at `runs/MOAT-45-ANALYSIS/scorecard/` (scorecard.jsonl 22 MB,
  visuals.json, meta.json, selftest.json, stats_cache.npz) + `analysis.log`, rsynced from
  box 43511290 (`/workspace/trace/EXP-57`). Box is EXTERNAL/operator-managed and shared
  with #47/#48/#49/#56 — NOT torn down by #45.
- No harness fix cycles were needed (0 of the allowed 3 iterations): the self-test and
  full emit both passed clean on the first run.
- Determinism (soft gate) PASSED: re-streamed delta-Gram byte-identical, recomputed rows
  byte-identical.
- For the lanes: the plugin interface (`MoatMethod.predict(history, delta, h)`) and the
  leakage-guarded `fit_window_positions()` (for #49's fit method) are wired and self-tested.
  Out-of-scope families (OrderPoly/EMA/Learnable*/GeneralRegression) are intentionally NOT
  registered, per the plan's dependency boundary.
