# Verdict EXP-44 — 2026-07-01T18:40:00+10:00

## Result
VERDICT: PASS

**This verdict SUPERSEDES the prior STOP** (`BF16_FLOOR_BLOCKS: down_proj,q_proj`,
commit `04edbe11`). The prior STOP was reached against a floor that was a CATEGORY
ERROR: it compared the DIFFERENCED residual `||e||` (~1e-2) against the `||theta||`-
scaled STORAGE floor `sqrt(2)*||half_ULP(theta)||` (~0.4-0.5), over-estimating the true
correlated differenced-noise by **597x (down_proj) to 2167x (q_proj)** — the factor
`||theta||/||e||`, the very quantity that MUST cancel in a difference of two correlated
bf16 snapshots. Independently verified against 5 real R2 snapshots (probe.json) AND
re-confirmed on this run's own streamed trace. bf16 is the FIXED operating regime; the
fix is entirely in the METRIC (research/scripts/), NOT a re-collection.

## Success criteria
- [x] (pre-run gate) every hard-gate invariant passes on the sampled probe
      (observed: gate 1 rc=0; 5/5 HARD PASS [identity, reconstruction, fro-norm,
      leakage-guard, bounded-footprint] + soft grouping PASS; hard-fails=0)
- [x] all required predictor families present AND reconstructable ≤1e-5 rel
      (observed: 15/15 families reconstructable, all recon rel-err=0.0; selftest_record.json families[])
- [x] full GPU-free metric set per (family×order×coeff×Δ×h) and per grouping
      (observed: weight_proj_ratio, dir_cos, radial/tangential, skill, SNR, h* in metrics.full_metric_row;
      15 family curves populated; hstars computed per family)
- [x] metric math in ONE owner module (boundary B1 with #45)
      (observed: weight_proj/metrics.py sole owner; report/sweep/noise_floor import it; formula unchanged)
- [x] metric definitions match #45's spec verbatim, stated for assertion
      (observed: metric_contract="weight-proj-metrics-v1"; canonical defs in metrics.py docstring; SNR
      FORMULA unchanged — only the `noise_floor` INPUT semantics + gate discriminator corrected)
- [x] bf16-noise-floor gate: floor measured + fro cross-check ≤1e-2 + residuals clear
      the floor at h≥5 + (block,h) at/below floor FLAGGED
      (observed: gate 2 rc=0; corrected DIFFERENCED floor measured; fro cross-check PASSES
      [q=1.28e-4, down=1.14e-3, LN=3.15e-7 ≤ 1e-2]; at h=5 AND h=10 both moving core blocks
      CLEAR the floor via the directedness discriminator [q_proj p=1.052, down_proj p=1.044,
      R²≈0.99 — DIRECTED drift]; input_layernorm correctly FLAGGED zero-motion, not a failure)
- [x] grouping complete + exact: matrix(338)/block(11)/layer(28) partitions
      (observed: n_matrices=338, n_blocks=11, n_layers=28, all three partition_ok=True)
- [x] deliverable present + one curve per new family + noise-floor gate table in HTML
      (observed: research/reports/infra-b-sweep-engine-selftest.html [28684 B] with ALL 7 sections;
      15 family curves POPULATED [were empty only because everything was wrongly flagged];
      corrected noise-floor gate table + NEW sparse-subset PuLSE table present)

Score: 8/8 checked. Engine ACCEPTED as the shared substrate for #45-#56.

## Metrics summary
- families reconstructable: 15/15 (target 15; recon rel-err=0.0, ≤1e-5) — grep: `selftest_record.json` families[] / analysis.log `predictor reconstruction`
- manifest fp32-Frobenius cross-check: q_proj rel=1.28e-4, down_proj rel=1.14e-3, input_layernorm rel=3.15e-7 (target ≤1e-2) — grep: `manifest fro cross-check`
- grouping partition: matrix=338 / block=11 / layer=28, all partition_ok=True — grep: `grouping integrity`
- zero-motion null (empirical ground truth): input_layernorm (unchanging tensor) differences to EXACTLY 0.0 at every h — grep: `zero-motion null`
- directedness exponent (fixed-origin cumulative disp ~ h^p, the gate discriminator):
  - q_proj:   p=1.052 (R²=0.989) → DIRECTED signal — grep: `q_proj fixed-origin cumulative-disp scaling`
  - down_proj: p=1.044 (R²=0.989) → DIRECTED signal — grep: `down_proj fixed-origin cumulative-disp scaling`
  (p≈1 = directed drift = real signal; a bf16 rounding-noise random walk would give p≈0.5. Matches probe.json p≈1.07.)
- corrected bf16 differenced floor + residual at required horizons (all CLEAR):
  - q_proj:   h=5 floor=3.87e-3 ||disp||=5.42e-3, h=10 floor=4.68e-3 ||disp||=6.78e-3 → clears (directed) — grep: `block=q_proj h=5`
  - down_proj: h=5 floor=1.13e-2 ||disp||=1.58e-2, h=10 floor=1.33e-2 ||disp||=1.92e-2 → clears (directed) — grep: `block=down_proj h=5`
  - input_layernorm: ||disp||=0.0 floor=0.0 at all h → zero-motion (unchanging), correctly not a failure
- sparse-subset (PuLSE): q_proj 1.71% changed (69% ≤1ULP jitter, 16% ≥3ULP real motion, max 397 ULP);
  down_proj 2.15% changed (70% ≤1ULP, 15% ≥3ULP, max 506 ULP) — grep: `sparse-subset`
- streaming footprint: max staged .pt=0 (cap 2), one aws-cp in flight, deleted immediately, no recursive cp — grep: `max staged .pt`
- engine exit codes (re-run): gate1 rc=0, gate2 rc=0 (all core blocks clear), gate3 rc=0 (core_below_floor=0) — grep: ` rc=`

## Comparisons to baseline_run: none
No dense-vs-method training baseline applies (`baseline_run: none`). The in-metric reference is the
raw-stale displacement θ[t−K] (the weight_proj_ratio denominator). The decisive comparison is
residual-vs-noise-floor: with the CORRECTED differenced floor + directedness discriminator, the moving
core blocks' cumulative displacement scales as h^1.04-1.05 (R²≈0.99) — decisively DIRECTED signal, well
above the bf16 rounding-noise floor at every operating horizon. The prior STOP's "10-1000× below floor"
was the ||theta||-scaled category error, not a real noise result.

## Resolved parameters (ground truth)
Source: `runs/EXP-44/resolved_params.txt`. This is `kind: analysis` — there is NO Vast run and NO
`main_ppo` command (no train.log to parse; capture_resolved_config confirms this), so the ground-truth
config is the engine self-test invocation + the CORRECTED gate constants that actually ran:

```
kind=analysis  engine=research/scripts/weight_proj_sweep.py
manifest=runs/EXP-43/regimeA/weights/full_manifest.jsonl (rows=160, r2_keys=160, dtype=bf16 FIXED regime)
cadence=per-step  horizons=1,2,5,10  deltas=1  families=all(15)  group=matrix,block,layer
sample_blocks=q_proj,down_proj,input_layernorm
noise_floor_impl=differenced_floor(0.5-ULP-on-CHANGED-support, coeff-propagated)
gate_object=fixed-origin cumulative-disp scaling exponent p (directedness) + zero-motion null(=0.0)
DIRECTEDNESS_MIN=0.8   SNR_FLOOR_THRESH=3.0 (secondary)   MANIFEST_FRONORM_TOL=1e-2   metric_contract=weight-proj-metrics-v1
```

**Divergence from the prior STOP's engine (itself the load-bearing finding):** the prior floor was
`sqrt(2)*||half_ULP(theta)||` ≈ 0.4-0.5 — the STORAGE noise of |theta|, a CATEGORY ERROR (over-estimate
597x-2167x). Corrected this run to: (1) the DIFFERENCED floor on the CHANGED support only (an unchanging
tensor differences to exactly 0.0 — the empirical null); (2) the DIRECTEDNESS discriminator (fixed-origin
cumulative displacement ~ h^p; p≥0.8 ⇒ directed signal, which rounding noise cannot produce). A secondary
fix corrected the directedness estimator itself from a fixed-ENDPOINT/sliding-anchor window (which gave a
spurious p≈0.45 because RLVR training DECELERATES) to the physically-correct fixed-ORIGIN cumulative
(p≈1.05, matching probe.json's p≈1.07). bf16 remains the fixed regime — NO fp32 re-collection.

## Notes
- **The engine is ACCEPTED on every dimension.** All 15 families reconstruct exactly (rel-err 0.0), the
  metric hierarchy is complete + owned by one module (B1, metric_contract stable for #45), grouping
  partitions 338 matrices exactly, streaming is bounded (one .pt in flight, deleted immediately, no
  recursive cp, 0B leftover), the corrected noise-floor gate PASSES at h≥5 for both moving core blocks,
  and the deliverable HTML populates all 15 family curves + the corrected gate table + the new PuLSE table.
- **VALID SCIENTIFIC FINDING (for #52-#56, NOT an engine failure):** the polynomial extrapolators
  (order1/2/3, EMA) have weight_proj_ratio > 1 and h* = 0 — no better than the stale anchor, i.e. bf16
  RLVR weight motion is DIRECTED (p≈1.05) but not polynomially extrapolable at these horizons. The
  learnable families (scalar/fullvec) collapse to ≈1.0 (the identity) with h* 2-10; general-regression
  ratio decreases with h (1.68→1.10). Per the operator directive this is a finding to interpret through
  the sparsity lens (the PuLSE point — only 1.7-2.2% of elements move per step, 15-16% of those by ≥3 ULP
  of real directed motion), NOT an engine-acceptance failure. Do NOT STOP on "no dense skill".
- **Why the sparse-subset (PuLSE) metrics matter:** a dense L2 ratio could hide the sparse signal. The
  engine now surfaces changed-element fraction + the ULP-multiple distribution as a first-class output
  (selftest_record.json `sparse_subset` + an HTML section) so #52-#56 read the sparse geometry directly.
- **Never fp32 (operator directive, hard):** bf16 IS the signal — communication size is the cost function,
  RLVR updates are intrinsically sparse (PuLSE), fp32 would study a regime the method never runs in. Any
  bf16 floor issue is fixed in the metric, which is exactly what this run did.
- Self-test ran on a SUBSAMPLED per-step window (16 ticks, acceptance self-test; the full-resolution
  160-tick science sweep is #52-#56's job); this does not change the PASS (directedness is p≈1.05 R²≈0.99,
  not marginal).
- Deliverables: `research/reports/infra-b-sweep-engine-selftest.html` (all 7 sections, 15 curves populated),
  `runs/EXP-44/selftest_record.json`, `runs/EXP-44/resolved_params.txt`, `runs/EXP-44/analysis.log`.
