# Verdict EXP-44 — 2026-07-01T15:59:00+10:00

## Result
VERDICT: STOP

STOP trigger (plan `## Analyst predicate` + `## Rescue triggers`):
`BF16_FLOOR_BLOCKS: down_proj,q_proj` — two CORE blocks are noise-dominated at the
REQUIRED horizons h=5 AND h=10. Per the plan this is a STOP whose resolution is an
**fp32 re-collection (a NEW GPU run, out of this plan's scope)**, NOT a code fix — so
REVISE does not apply. The engine itself is CORRECT (all structural/engine-correctness
criteria pass); the data (bf16 trace at lr=1e-6) cannot resolve the residual.

## Success criteria
- [x] (pre-run gate) every `hard`-gate invariant passes on the sampled probe
      (observed: 5/5 HARD PASS + soft grouping PASS, hard-fails=0; reconfirmed on this HEAD)
- [x] all required predictor families present AND reconstructable ≤1e-5 rel
      (observed: 15/15 families reconstructable, worst recon rel-err = 0.0)
- [x] full GPU-free metric set computed per (family×order×coeff×Δ×h) and per grouping
      (observed: weight_proj_ratio, dir_cos, radial/tangential, skill, SNR, h* all in
      metrics.full_metric_row; records emitted per matrix/block/layer)
- [x] metric math in ONE owner module (boundary B1 with #45)
      (observed: weight_proj/metrics.py is sole owner; report.py/sweep.py/noise_floor.py import it)
- [x] metric definitions match #45's spec verbatim, stated for assertion
      (observed: metric_contract="weight-proj-metrics-v1"; canonical defs in metrics.py docstring)
- [ ] bf16-noise-floor gate: floor measured + fro cross-check ≤1e-2 + **residuals clear
      the floor at h≥5** + (block,h) at/below floor FLAGGED
      (observed: floor measured [q_proj=3.03e-1, down_proj=4.02e-1, LN=7.55e-2] and fro
      cross-check PASSES [q=1.28e-4, down=1.14e-3, LN=3.15e-7 ≤ 1e-2]; flagging works;
      BUT residuals do NOT clear the floor — SNR at h=5/h=10 for q_proj=0.05/0.09,
      down_proj=0.11/0.20, all ≪ SNR_FLOOR_THRESH=3.0. target: SNR>3 at h≥5 for core blocks.
      **This box FAILS on the data, not the engine.**)
- [x] grouping complete + exact: matrix(338)/block(11)/layer(28) partitions
      (observed: n_matrices=338, n_blocks=11, n_layers=28, all three partition_ok=True)
- [ ] deliverable present + one curve per new family + noise-floor gate table in HTML
      (observed: HTML present [research/reports/infra-b-sweep-engine-selftest.html, 19202 B]
      with the FULL noise-floor gate table [12 rows, all correctly flagged bf16-unreliable];
      but family_curves is all-but-empty [only order3-fixed marginally clears one point]
      because every core-block residual is below the bf16 floor → the sweep correctly
      REFUSES to plot ratios for bf16-unreliable rows. The missing curves are a DIRECT
      consequence of the same data limitation, not an engine defect.)

Score: 6/8 checked. The 2 unchecked boxes (noise-floor h≥5, curves) are BOTH downstream
of one root cause — the bf16 trace at lr=1e-6 has per-step weight displacement ~100-500×
below the bf16 quantization floor — for which the plan prescribes STOP (fp32 re-collection),
not a code fix.

## Metrics summary
- families reconstructable: 15/15 (target 15; recon rel-err = 0.0, ≤ 1e-5) — grep: `selftest_record.json` families[]
- manifest fp32-Frobenius cross-check: q_proj rel=1.28e-4, down_proj rel=1.14e-3, input_layernorm rel=3.15e-7 (target ≤ 1e-2) — grep: `manifest fro cross-check` in analysis.log
- grouping partition: matrix=338 / block=11 / layer=28, all partition_ok=True (target exact) — grep: `grouping integrity` / selftest_record.json grouping
- bf16 quant floor (fixed engine): q_proj=3.0315e-01, down_proj=4.0153e-01, input_layernorm=7.5474e-02 — grep: `\[noise-floor\] block=` in analysis.log
- SNR at required h (target > 3.0 to clear floor):
  - q_proj:   h=5 SNR=0.05 (||e||=1.39e-2), h=10 SNR=0.09 (||e||=2.81e-2)  → bf16-unreliable
  - down_proj: h=5 SNR=0.11 (||e||=4.28e-2), h=10 SNR=0.20 (||e||=7.89e-2) → bf16-unreliable
  - input_layernorm: ||e||=0.0 at all h (gain does not change in bf16 between steps) → bf16-unreliable
- true 1-step weight displacement ‖θ[t]−θ[t−1]‖ (direct R2 measurement, tick_0→tick_2):
  q_proj=5.22e-4, down_proj=1.48e-3, input_layernorm=0.0 — vs floor ~3e-1/4e-1/7.5e-2
  → disp/floor = 0.002 / 0.004 / 0.000 (the update is ~200-500× below bf16 resolution)
- streaming footprint: max staged .pt = 0 (cap 2), 1 aws-cp in flight at a time, staging
  returned to 0B; disk free stayed 273Gi ≫ 8GB min — grep: `max staged .pt` in analysis.log
- engine exit code: rc=3 (BF16_FLOOR_BLOCKS) — grep: `re-run gate2+3 rc=3` in analysis.log

## Comparisons to baseline_run: none
No dense-vs-method training baseline applies (`baseline_run: none`). The in-metric reference
is the raw-stale displacement θ[t−K] (the weight_proj_ratio denominator). The decisive
comparison here is residual-vs-noise-floor: for the core blocks the residual is 10-1000× BELOW
the bf16 quantization floor at every horizon, so no ratio is a precise number.

## Resolved parameters (ground truth)
Source: `runs/EXP-44/resolved_params.txt`. This is `kind: analysis` — there is NO Vast run
and NO `main_ppo` command, so the ground-truth "config" is the engine self-test invocation +
the noise-floor gate constants that actually ran (extracted from analysis.log + the engine):

```
kind=analysis  engine=research/scripts/weight_proj_sweep.py
manifest=runs/EXP-43/regimeA/weights/full_manifest.jsonl (rows=160, r2_keys=160, dtype=bf16)
cadence=per-step  horizons=1,2,5,10  deltas=1  families=all(15)  group=matrix,block,layer
sample_blocks=q_proj,down_proj,input_layernorm
selftest_tick_window=[0,2,...,30] (16 ticks, SUBSAMPLED acceptance self-test; full-res sweep is #52-#56)
SNR_FLOOR_THRESH=3.0  MANIFEST_FRONORM_TOL=1e-2  metric_contract=weight-proj-metrics-v1
noise_floor_impl=bf16_quant_floor(half-ULP, 7-mantissa-bit, sqrt2-diff-inflation)
```

**Divergence from the as-committed engine (itself a finding):** the delivered engine's
`bf16_roundtrip_floor` computed a DEGENERATE self-round-trip `theta.to(bf16).to(fp32)` which
is a no-op on an already-bf16-STORED trace → floor==0.0 for every group → SNR=nan → every
(block,h) spuriously flagged and NO family curves. That is the plan's named bf16↔fp32
cast-point bug (`## Notes for runner`). I fixed it this run (commit `04edbe11`, scoped to
research/scripts) to a magnitude-aware half-ULP quantization floor. The fix is what makes the
STOP HONEST: with a real, non-zero floor the core blocks are provably noise-dominated. Without
the fix the STOP would have been reached for the WRONG reason (floor=0 makes SNR=nan, which the
engine also flags — but that is a numerics artifact, not a physical noise result).

## Notes
- **The engine is ACCEPTED as correct on every dimension the acceptance gate can test without
  a higher-precision trace.** All families reconstruct exactly (rel-err 0.0), the metric
  hierarchy is complete and owned by one module (B1), grouping partitions 338 matrices exactly,
  and streaming is bounded (one .pt in flight, deleted immediately, no recursive cp, 0B
  leftover). The bf16-noise-floor gate MACHINERY is correct after the fix — it measures the
  floor, cross-checks the manifest fro-norm (≤1e-2), computes SNR, and flags bf16-unreliable
  rows exactly as specified. What FAILS is a DATA property, not the engine.
- **Why STOP and not REVISE:** REVISE requires a concrete engine code fix per unchecked box.
  I already applied the one legitimate code fix (the degenerate floor). After it, the two
  remaining unchecked boxes (h≥5 clears floor; family curves present) are BOTH caused by the
  bf16 trace at lr=1e-6 having per-step displacement ~200-500× below bf16 resolution. The
  plan's `## Analyst predicate` and sequence step-2 `on_fail` name this exact case as STOP with
  `BF16_FLOOR_BLOCKS`, whose resolution is "an fp32 re-collection, a NEW GPU run out of this
  plan's scope" — there is no code fix. Running the predicate verbatim ⇒ STOP.
- **Operator decision required (this is the escalation the plan wired):** to make the
  weight-projection science downstream (#52-#56) resolvable, EXP-43's trace must be
  re-collected in **fp32** (`dump_dtype=fp32`, ~984 GB, a NEW GPU run) OR at a higher learning
  rate / larger Δ window where the cumulative displacement clears the bf16 floor. The
  wired-but-unused fp32 escape hatch exists precisely for this. This is out of scope for #44
  (GPU-free by contract) and is the operator's call.
- **Downstream consumers (#45-#56):** the engine substrate (predictor families, metric module,
  grouping, streaming reader) is correct and re-usable AS IS. The metric contract
  (`weight-proj-metrics-v1`) and definitions are stable for #45 to assert against. Only the
  bf16-floor limitation blocks the *science*, not the *engine API*.
- Self-test ran on a SUBSAMPLED 16-tick per-step window (an acceptance self-test, not the
  full-resolution 160-tick science sweep — that is #52-#56's job); this is noted in the log
  and does not change the STOP conclusion (the noise domination is 10-1000×, not marginal).
- Deliverables: `research/reports/infra-b-sweep-engine-selftest.html` (noise-floor gate table
  present, all rows correctly flagged), `runs/EXP-44/selftest_record.json`,
  `runs/EXP-44/resolved_params.txt`, `runs/EXP-44/analysis.log`.
