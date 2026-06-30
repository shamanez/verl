# M4 weight-projection track: status + architecture review

> Principal-architect status + architecture review of the M4 weight-projection track.
> GPU-free, read-only assessment. Audience: the planner who will scope issue #44
> (the offline sweep engine) and the rest of the #45-#56 analysis spine.
> Date: 2026-06-30. Branch: `vast-ai-workload`.

## 0. Why this track exists (one paragraph)

The comm-eff method's anchor holds STALE weights `theta[t-K]`. At the realistic
high anchor latency (cadence/delay_K = 20/20) the stale anchor gradient rotates to
orthogonal by `k ~ 10-20` (the k-collapse). Priority 1 of the project goal is to
fix this by extrapolating the anchor's WEIGHTS forward (Nesterov-style: compute the
gradient at the look-ahead weights `theta_hat`, not patch a stale gradient). The M4
weight-projection track is the GPU-free kill-test for that idea: given the dense
GRPO weight trajectory `theta(t)`, can a predictor extrapolate `theta_hat(h)` close
enough to the true `theta_now` (in weight space) to be worth doing, and how far
ahead (horizon `h`) before it overshoots? The metric hierarchy below answers
"close enough"; the crossover `h*` answers "how far". EXP-43 collected the ground-
truth trajectory. The analysis spine (#44-#56) consumes it.

---

## 1. CURRENT STATUS - the data, what is solid, what is analyzable now

### 1.1 What was collected (EXP-43, PASS, issue closed)

The shared dense regime-A (codec OFF) full-weight per-tick trajectory. Verified
this review directly against the in-repo manifests:

| Property | Value (verified) |
|---|---|
| R2 prefix | `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/` |
| Key form | `tick_<N>/tick_<N>.pt` (one full bf16 model state_dict per object) |
| Snapshots | **160** (`full_manifest.jsonl` = 160 rows; first `tick=0 gs=1`, last `tick=159 gs=80`) |
| Cadence | per-tick, **exactly 2 ticks/step** for all 80 steps (batch128/mini64) |
| Step coverage | global_step 1..80, dense + contiguous |
| Per-snapshot matrices | **338** floating params (real shapes, NOT a sketch, NOT a 196-subset) |
| Precision | bf16 (`dump_dtype=bf16`) |
| R2 durability | `r2_manifest.jsonl` 160/160 rows `verified:true`; total **494.0 GB** (`tick_0` = 3.087 GB) |
| Box-side verify | `verify_full_weight_dump.py --r2 --r2-sample 5` PASS, max_rel_norm_err = 0.0001 (<= 0.01 tol) |
| Final val acc | 0.7809 (dense-control band 0.75-0.78; provenance only, no acceptance threshold) |

The 338 matrices decompose (verified from the first snapshot's manifest row) as:
28 decoder layers (0..27), each with q/k/v/o_proj (q has weight+bias, k/v have
weight+bias, o weight only -> 56 q + 56 k + 56 v + 28 o), gate/up/down_proj (28
each), input_layernorm + post_attention_layernorm gains (28 each), plus
embed_tokens (1) and the final norm (1). So the trace covers the WHOLE model:
decoder linears + biases + RMSNorm gains + token embeddings. (lm_head is tied to
embed_tokens in Qwen2.5-1.5B, hence one embedding tensor, not two.)

This is the canonical artifact every downstream M4 issue (#44-#56) cites. It is
self-describing: the two index manifests live both in R2 and tracked in-repo at
`runs/EXP-43/regimeA/weights/{full_manifest,r2_manifest}.jsonl`. The heavy `.pt`
live in R2 ONLY.

**Solid / verified:** snapshot count, per-tick-to-per-step alignment (subsample
the first tick of each global_step: ticks 0,2,...,158 -> steps 1..80), matrix
coverage and real shapes, R2 durability (160/160 verified), and the dense-control
healthiness (val 0.7809, no NaN/Inf, codec counters all 0).

### 1.2 The precision / noise-floor caveat (load-bearing for the analysis)

The trace is **bf16** (~0.4% per-element relative error). The whole track is built
on DIFFERENCING consecutive snapshots (`Delta-theta = theta[t] - theta[t-1]`, and
displacement vectors `theta_now - theta_stale`). Differencing two nearly-equal bf16
tensors triggers both quantization noise and catastrophic cancellation: the finest
single-tick deltas are the noisiest, because the true per-step update (~1e-3
relative, at lr=1e-6) is the same order as the bf16 rounding (~4e-3).

Consequences for #44-#56:
- The metric hierarchy is **reliable for cumulative displacement and horizons
  `h >= ~5`** (the displacement vectors are large relative to the noise floor).
- Single-tick / very-short-horizon deltas carry a real bf16 noise floor that any
  honest SNR / skill metric must report against, not hide.
- `fp32` is a **wired-but-unused escape hatch**: `dump_dtype=fp32` is validated and
  plumbed end to end (`CommEffWeightTrajConfig`, observer, launcher
  `WEIGHT_TRAJ_FULL_DTYPE`), it just doubles the storage to ~984 GB. If a study is
  blocked specifically by the single-tick noise floor, the fix is a fresh fp32
  collection run, NOT a code change. Do not silently treat bf16 deltas as exact.

### 1.3 What is analyzable NOW (GPU-free, today)

Everything the metric hierarchy needs is a function of weight-difference vectors,
and those reconstruct exactly (modulo bf16) from the raw snapshots. So with only
the laptop + R2 streaming access, an analyst can compute, per (predictor x order x
coefficient-source x Delta x h) and per matrix / layer / block:

- `weight_proj_ratio = ||theta_hat(h) - theta_now|| / ||theta_stale - theta_now||`
  (below 1 = projection helps)
- `dir_cos = cos(theta_stale - theta_old, theta_now - theta_stale)`
- radial/tangential split of the error `e = theta_hat - theta_now` (radial along
  `theta_now - theta_stale` = damping-fixable overshoot; tangential = the rotation
  ceiling no scalar damping can fix)
- skill `= 1 - weight_proj_ratio^2`
- SNR (residual norm vs the per-step + bf16 noise floor)
- crossover `h* = largest h with median ratio < 1`

What is NOT computable from this trace (be explicit in any report): anything
needing a gradient or the loss (Fisher/K-FAC KL, true gradient-recovery cos at the
look-ahead weights, output-space KL on a held batch). Those are the **GPU-gated
functional-validation tier (#46)**, by design separate from the GPU-free metric
tier. The weight-space metrics are PROXIES; #45 owns ranking which proxy best
predicts the #46 functional truth.

### 1.4 Provenance note on the superseded EXP-42 numbers

The SUMMARY's EXP-42 section (crossover `h*=10`, `weight_proj_ratio 0.972`,
`dir_cos 0.549` at K=10, damped-alpha ~0.5 lever) is explicitly marked SUPERSEDED:
those came from the lossy k=4096 count-sketch of only 196 decoder matrices. They
are a useful PRIOR (they motivate damped-alpha and the order-1 baseline) but must
be RE-DERIVED on the EXP-43 raw trace before they are cited as ground truth. The
new engine should reproduce the EXP-42 order-1 / fixed-linear result as a sanity
check, not assume it.

---

## 2. CODE STATE - implemented, merged, deleted, debt, risk

### 2.1 Implemented + merged (on `vast-ai-workload`)

The weight-traj observer + R2 sink are merged and were exercised end-to-end by the
EXP-43 collection run. Component map:

- **`verl/workers/comm_eff/capture.py` :: `WeightTrajObserver`** - dump-only full-
  weight recorder. `per_tick=True` dumps every optimizer tick (`full/tick_<N>.pt`);
  `select_weight_traj_targets` selects EVERY floating param (no subset, no
  `select_all` toggle). Pure I/O: reads live weights, feeds nothing to
  optimizer/EMA/Q. Built independently of `comm_eff.enabled` so the codec-OFF dense
  regime is instrumented. Each snapshot is a `torch.save` state dict
  `{canon_name -> tensor}` + a `full_manifest.jsonl` row carrying both global_step
  and tick and per-matrix names/shapes/fp32-norms.
- **`verl/workers/comm_eff/r2_sink.py` :: `R2ArtifactSink`** - upload -> verify
  (head-object size, optional sha256) -> manifest -> delete-local. Hard bucket
  guard (`shamane-pluralis` only, fail-loud at construction). Creds read from env
  as `R2_* -> AWS_*` (region=auto); shells to `aws` (no boto3). Synchronous mode is
  byte-identical to the original.
- **`verl/workers/config/comm_eff.py` :: `CommEffWeightTrajConfig`** /
  `CommEffCaptureConfig` - the Hydra schema, including all the r2_async knobs;
  validated in `__post_init__`. Mirrored in `verl/trainer/config/actor/actor.yaml`
  under `comm_eff.probe.weight_traj` and `comm_eff.capture`.
- **`verl/workers/engine_workers.py`** - `_maybe_weight_traj_observer()` builds the
  observer once and attaches it to the actor-train engine; the train-batch hook
  mirrors the trainer global_step onto the engine so per-tick dumps are keyed
  correctly on the dense path. `comm_eff_close(timeout)` is the run-end barrier that
  drains + fail-loud-closes both the observer's and the capture writer's R2 pools.
- **`verl/trainer/ppo/ray_trainer.py`** - calls `comm_eff_close()` at `is_last_step`
  (line ~1799) AND again as an unconditional loop-end safety net (line ~1816) so the
  async pool is drained and fails loud even if the loop exits without reaching the
  last step. (This is the C1 fix from the review below; verified present.)
- **`research/scripts/verify_full_weight_dump.py`** - the dump-integrity gate.
  Asserts each snapshot loads to a real state dict (2D matrices keep their shape =
  not a flattened sketch), n_matrices matches the manifest, and per-matrix
  Frobenius norms match within `--tol` relative. Has a `--r2` sample-download mode.

### 2.2 NEW: async / parallel-worker upload mode (merged, DEFAULT-ON, box-UNVALIDATED)

`R2ArtifactSink(async_mode=True)` adds a background worker pool (default 4 threads),
a job queue, disk backpressure (`max_staged_gb`, default 80), a per-N-steps flush
barrier, and a fail-loud `close()`. Merged to `vast-ai-workload` (commit 70c63a86,
was PR #19) and made **default-on for collection runs** (`WEIGHT_TRAJ_R2_ASYNC=true`
in `exp42_run_cell.sh`, commit 38ea06fd).

The four critical + three high-severity defects flagged in the pre-merge review
(`reports/async-r2-upload-review.md`) appear to be fixed in the merged code (verified
by reading r2_sink.py + capture.py + ray_trainer.py):
- C1 (close never called) -> `comm_eff_close()` now wired into ray_trainer at both
  `is_last_step` and loop-end.
- C2/C3 (unbounded `queue.join()` at atexit) -> `_atexit_close` passes a finite
  120 s timeout; `flush(timeout)` polls `unfinished_tasks` instead of unbounded
  join.
- H1 (staged-bytes leak) -> `put()` is now INSIDE the condition lock, increment only
  after a successful put (the R2SINK-001 comment).
- H2 (phantom manifest row) -> `full_manifest.jsonl` carries an explicit
  `r2: "pending"|"off"` field; durability is attested only by `r2_manifest.jsonl`
  `verified:true`. Docstring documents the dumped-vs-verified split.
- H3 (per-step flush no timeout) -> the observer passes a finite `flush_timeout_s`.
- L1 (clean-success-while-hung) -> `close()` checks `is_alive()` and raises.

Unit coverage is thorough: `tests/workers/comm_eff/test_r2_sink.py` (28 tests,
including async enqueue/flush, backpressure, fail-loud, the R2SINK-001 leak, bounded
timeouts, atexit, the finalizing guard) and `test_weight_traj_observer.py` (15
tests, including async flush cadence, close-surfaces-late-failure, finite-timeout,
r2_status pending-vs-off).

**THE RISK (architectural, must be flagged):** async mode is unit-tested but NOT
box-validated, and it is default-on for collection. The review's Section 3 lists
exactly what only a training box can settle and which is therefore still open:

1. **Real R2 throughput under N concurrent `aws s3 cp` streams** - does
   `upload_workers=4` saturate the box NIC / keep up with a ~3 GB-every-few-seconds
   dump cadence, or does it chronically sit at the backpressure cap?
2. **Real disk backpressure under the full ~494 GB trajectory** - does the
   staged-bytes cap actually bound local staging below box disk end to end, does the
   producer block (rather than ENOSPC) when uploaders fall behind, and does the
   staged-bytes counter return to ~0 with no leak over a long run?
3. **`close()` / atexit firing under real Ray teardown** - does the engine-level
   `close()` actually run on every rank, before Ray kills daemon threads, on both
   normal and exception exit, and does an injected last-step failure propagate as a
   non-zero run exit?

Mitigating fact: EXP-43 itself succeeded and certified 160/160 verified uploads.
But the EXP-43 verdict does NOT state whether that run used the async path or the
synchronous fallback, and the async default landed in the same session window. So
"async is proven by EXP-43" is NOT established. **Recommendation: any future
collection run (e.g. an fp32 re-collection) that relies on async must either (a) be
treated as the box-validation run with the Section-3 checklist actively watched, or
(b) set `WEIGHT_TRAJ_R2_ASYNC=false` to take the proven synchronous path.** This is
the one piece of live architectural debt in the collection half of the track.

### 2.3 Deleted (confirmed gone)

The old count-sketch world was purged (commit 2d7f3c63, operator directive
2026-06-30 - "the study needs the raw weights, not a sketch"):
- `research/scripts/weight_proj_sweep.py` (531-line offline count-sketch sweep
  engine) - **GONE** (verified: file absent).
- The 3 sketch report builders `build_report.py`, `build_dense_report.py`,
  `build_dense_report_v2.py` - **GONE** (verified).
- The 3 sketch-derived HTML reports (`exp42-weight-projection-accuracy.html`,
  `exp42-dense-weight-behavior.html`, `exp42-dense-deep-analysis.html`) - deleted;
  SUMMARY marks their conclusions SUPERSEDED.
- In `capture.py`: the k-bucket count-sketch, the exact-calibration ring, the
  `select_all` toggle, the 196-subset, and the `k` / `calib_*` / fp16 knobs - all
  removed (only historical NOTE comments remain, pointing at git history).
- `exp42_drive_all.sh` (the `select_all=true` driver) - deleted.

There is no residual sketch code in the live path. `select_weight_traj_targets` is
single-arg and selects all floating params; the config has no sketch knobs.

---

## 3. THE GAP - the offline sweep engine no longer exists

When the count-sketch was purged, the engine that ANALYZED it
(`weight_proj_sweep.py`, 531 lines) was deleted with it. The downstream issues
#44-#56 are written assuming that file still exists and that the input is a sketch:

- **#44's body still says** "Today `research/scripts/weight_proj_sweep.py`
  (531 lines) replays only an order-1 `fixed_linear`..." and "EXTENDS
  `research/scripts/weight_proj_sweep.py`". That file is gone. #44 is therefore NOT
  an "extend the engine" task; it is a **build-the-engine-from-scratch** task.
- **The parity gate is MOOT.** #44/#45 specify a hard gate: "every sketch-derived
  metric matches the saved on-box EXACT calib scalar within 5%". There is no sketch
  and no calib ring anymore - the raw weights ARE the ground truth (exact modulo
  bf16). The "sketch-vs-calib parity" gate cannot and should not be implemented. It
  must be REPLACED by a bf16-noise-floor sanity check (see Section 4.6).
- The input path in #44 ("the dense `select_all` trace under
  `research/runs/EXP-<COLLECT>/weights/`", "read via `weight_proj_sweep.py
  <trace_dir>`") is wrong twice over: the trace is the R2 raw-weight trace (not a
  local select_all sketch), and the reader does not exist.

So the gap is: **#44 must deliver a NEW offline engine that streams RAW full-weight
snapshots from R2 (never bulk-downloads ~494 GB), computes the metric hierarchy on
real weight-difference vectors, and runs the full predictor x order x coeff-source x
Delta x h sweep with per-matrix / per-layer / per-block grouping.** Issues #45-#56
then each just select a predictor family + a grouping and consume the engine's
intermediates - exactly the "thin downstream issues" design #44 intends, except the
shared substrate has to be written first.

The #44 (and #45) issue bodies should be updated to: (a) drop the "extend
weight_proj_sweep.py" framing for "build a new engine", (b) replace the
sketch-vs-calib parity gate with the bf16-noise-floor sanity check, (c) fix the
input path to the R2 raw-weight trace + the mandatory streaming access pattern.

---

## 4. RECOMMENDED ARCHITECTURE for the #44 engine (raw-weight world)

All of this is `research/scripts/` only (GPU-free, `code_change` confined to the
research tree, never verl source). The mandatory access contract is
`reports/r2-access-pattern-for-analysis.md`: stream layer/block-wise, bound the
local working set to a few GB, reduce to small per-layer intermediates, combine,
THEN render HTML. NEVER `aws s3 cp --recursive` the prefix.

### 4.1 Module structure (suggested)

```
research/scripts/weight_proj/                 # new package (or flat files)
  r2_stream.py        # streaming R2 reader (one .pt at a time, delete after use)
  predictors.py       # the predictor-family API (order-1/2/3, damped, learnable, regression, EMA)
  metrics.py          # the GPU-free metric hierarchy (ratio, dir_cos, radial/tangential, skill, SNR, h*)
  sweep.py            # the (family x order x coeff-source x Delta x h) driver + grouping
  noise_floor.py      # the bf16-noise-floor sanity check (replaces the parity gate)
  report.py           # per-layer/block intermediates -> combine -> HTML
  __main__.py         # CLI: python -m weight_proj <r2_prefix-or-manifest> [--group ...] [--families ...]
```

Keep `metrics.py` as the SOLE owner of the metric math (the #44/#45 boundary B1:
#44 owns the engine metric code, #45 owns only the proxy-ranking layer + its own
report). Keep `predictors.py` family-pluggable so #47-#51 add a family without
touching the driver.

### 4.2 The streaming R2 reader (the hard constraint)

The engine must operate block-wise on the OUTSIDE, ticks on the INSIDE, so the
local footprint is one (or a few) in-flight snapshot(s):

- **Keys + creds.** Object key form is `tick_<N>/tick_<N>.pt` under
  `verl-research/EXP-43/regimeA/weights/full/`. For a manual `aws` call, map
  `AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY`,
  `AWS_DEFAULT_REGION=auto`, and pass `--endpoint-url "$R2_ENDPOINT"`. Bucket =
  `shamane-pluralis` ONLY. (The existing repo scripts - `r2_sink.py`,
  `verify_full_weight_dump.py` - already do this mapping internally and can be
  cribbed.)
- **Tick order from the manifest.** Read `full_manifest.jsonl` (160 rows, in-repo)
  for the (global_step, tick, path, per-matrix names/shapes/fp32-norms). Read
  `r2_manifest.jsonl` for the verified keys. Drive the download from these; do not
  list the bucket.
- **Two streaming idioms** (per the access doc):
  - *Need one block across all ticks:* loop ticks, `torch.load(map_location="cpu")`
    one snapshot, extract ONLY that block's tensors, accumulate the partial result,
    DELETE the local `.pt`, next tick.
  - *Need all blocks independently:* load each snapshot ONCE, fan its per-block
    slices into per-block accumulators, delete, next tick. This is the efficient
    pass because each 3 GB object is downloaded exactly once.
- **Per-step vs per-tick.** Subsample the per-step trajectory (80 points) by taking
  the first tick of each global_step (ticks 0,2,...,158). Most predictor families
  operate on the per-step trajectory; per-tick is available for finer-Delta studies
  but is the noisier (cancellation) regime.
- **The predictors only need DIFFERENCE vectors**, so the accumulator state per
  block is small: a running buffer of the last `order+1` (or window-size) snapshots
  of that block, plus the per-(family, h) metric scalars. Never hold the whole
  trajectory of a block in RAM if the window is short - a ring of the last few
  snapshots suffices for fixed/damped/EMA; the general regression (#50) needs the
  full past window for that block, which for one block (e.g. a 1536x8960 down_proj
  = ~13.7M params) across 80 steps is ~4.4 GB fp32 - still streamable one block at
  a time, NOT the whole model.

### 4.3 The predictor-family API

A predictor maps a window of past snapshots of one block + a horizon `h` to a
prediction `theta_hat(h)`. Reference families (each is a linear-or-mildly-nonlinear
combination of past snapshots, so all reconstruct from the raw weights):

| Family | Form | Coeff source(s) | Owning issue |
|---|---|---|---|
| order-1 fixed | `theta_hat = theta_stale + alpha*(theta_stale - theta_old)`, `alpha = h/Delta` | fixed | #47 |
| order-1 damped | same, `alpha` shrunk by an in-sample factor (~0.5 per EXP-42 prior) | offline-damped | #47 |
| order-2 / order-3 | Newton forward-difference (3-pt) / Lagrange-Newton (>=4-pt) polynomial | fixed + offline-damped | #48 |
| learnable-at-order | weak (per-matrix scalar residual) AND strong (full coefficient-vector least-squares) | learnable | #49 |
| general regression | least-squares fit of the past window onto retrospective truth (+ optional small nonlinear) | learned | #50 |
| EMA / momentum | smoothed-velocity, fixed-decay and learnable-decay beta | fixed + learnable | #51 |

API sketch: `predict(history: list[Tensor], h: int, coeff_source: str) -> Tensor`,
with a `fit(history, truth)` step for the learnable/regression families (fit on a
retrospective split of the trajectory, evaluate forward - guard against fitting on
the same point you score, the classic leakage trap). "Coefficient source" (fixed /
offline-damped / learnable / learned) is an orthogonal axis to "order", which is
why the sweep is a cross-product not a list.

### 4.4 The metric hierarchy (defs are the contract; from #45 / Section 1.3)

All operate on weight-difference vectors of one block (then aggregate):

- `weight_proj_ratio(h) = ||theta_hat(h) - theta_now|| / ||theta_stale - theta_now||`
- `dir_cos = cos(theta_stale - theta_old, theta_now - theta_stale)`
- error `e = theta_hat - theta_now`; let `u = (theta_now - theta_stale)/||.||`.
  radial `= <e, u>` (along the displacement = damping-fixable overshoot),
  tangential `= ||e - <e,u>u||` (orthogonal = the rotation ceiling).
- skill `= 1 - weight_proj_ratio^2`
- SNR `= ||e|| / noise_floor` where noise_floor is the bf16 + per-step floor
  (Section 4.6) - report it, do not hide it.
- crossover `h* = largest h with median ratio < 1` (the actionable horizon).

These are medians/percentiles over the grouping (per matrix, then aggregated per
block / per layer). Keep the per-matrix percentile machinery so #52/#53/#54 consume
it unchanged.

### 4.5 The sweep + grouping

- Sweep axes: predictor-family x order x coefficient-source x Delta (snapshot
  spacing, e.g. per-step vs per-2-step) x horizon `h`. Per (cell, block) produce the
  metric vector.
- Grouping: per-matrix (338), per-block (the 11 families above: q/k/v/o_proj,
  gate/up/down_proj, the two layernorm gains, embed, norm), per-layer (28 decoder
  layers, 0..27, available directly from the manifest names). #52 = per-block, #53 =
  per-layer/depth, #54 = per-matrix small-multiples + depth x block heatmaps.
- Intermediates: write small per-layer / per-block `.npz` (or parquet) of the
  metric-vs-horizon arrays. Combine those into the final arrays, THEN build HTML
  (#54). Never hold the full trajectory in RAM or on disk at once.

### 4.6 Replace the dead parity gate with a bf16-noise-floor sanity check

The old gate ("sketch metric within 5% of the on-box exact calib") is MOOT - there
is no sketch and no calib ring; the weights are exact modulo bf16. Replace it with
a bf16-noise-floor sanity check that an engine self-test (#44's
`infra-b-sweep-engine-selftest.html`) must pass:

1. **Round-trip floor.** For a sample of blocks, cast the loaded bf16 tensor to
   fp32, perturb by a fresh bf16 round-trip, and measure the induced relative error
   on each difference vector. This gives the empirical noise floor per block.
2. **Metric-above-floor gate.** Assert that at the operating horizons (`h >= 5`) the
   residual norms the metrics report are comfortably above that floor (e.g. SNR > a
   few). FLAG any (block, h) where the metric is at/below the floor as
   "bf16-unreliable - re-collect in fp32 to resolve", rather than reporting a
   precise-looking but noise-dominated ratio.
3. **Manifest cross-check.** Recompute each loaded matrix's fp32 Frobenius norm and
   assert it matches the manifest `fro_norm` within tol (this is exactly what
   `verify_full_weight_dump.py` does - reuse it, do not reinvent).

This keeps the spirit of the old gate (don't trust a metric the instrument can't
support) while matching the raw-weight reality.

---

## 5. RISKS / OPEN QUESTIONS for the analysis phase

1. **bf16 single-tick noise floor may block the finest-Delta studies.** If a
   predictor family (e.g. high-order polynomial over closely-spaced ticks, or EMA at
   short horizon) lives at/below the floor, the answer is a fp32 re-collection
   (`dump_dtype=fp32`, ~984 GB), which is a NEW GPU run, not a code fix. Decide the
   trigger threshold (Section 4.6 SNR gate) up front so this is a clean go/no-go, not
   a mid-analysis surprise. This is the single biggest scientific risk.
2. **Async-upload box-validation is still owed** (Section 2.2). It does not block the
   #44-#56 ANALYSIS (those read the already-collected, already-verified EXP-43
   trace). It only matters for any FUTURE collection run (notably an fp32
   re-collection per risk 1). If/when that run happens, treat it as the
   box-validation run or fall back to synchronous.
3. **Streaming I/O cost.** Each full pass over the trace downloads 494 GB from R2.
   At a single-stream ~60-90 MiB/s ceiling that is hours per pass; parallel `aws s3
   cp` helps but costs egress. Design the engine so ONE streaming pass fans out to
   ALL per-block accumulators and ALL families/horizons at once (Section 4.2 second
   idiom), so the analysis is not N separate 494 GB downloads. This is an
   architectural must, not a nice-to-have.
4. **Learnable / regression leakage.** The learnable (#49) and general-regression
   (#50) families fit coefficients; they MUST fit on a retrospective split and score
   forward, or they will report illusory skill. EXP-42's prior already found the weak
   scalar-residual learned predictor INERT on the dense run (max |resid| below the
   sketch floor); re-confirm on raw weights and beware the strong-regression
   overfitting mirror image.
5. **Weight-space proxy != functional truth.** The whole GPU-free tier measures
   weight-space closeness. The thing the method actually needs is that `g(theta_hat)`
   recovers the gradient direction (the goal's "cos@k5 >= 0.40, off-diagonal"
   kill-test). #45 must rank which weight-space proxy best predicts the #46 GPU-gated
   functional metric; until #46 runs, a low `weight_proj_ratio` is necessary-not-
   sufficient evidence. Do not let a strong weight-space result be read as a method
   win on its own.
6. **`h*` is a median; per-block dispersion matters.** EXP-42's sketch prior put
   per-matrix `h*` in a 9-14 band with attention v/o_proj projecting furthest and
   MLP / k/q least. If that holds on raw weights, the verdict (#56) is per-block /
   per-layer, not one global horizon - the engine must preserve that granularity
   (it does, via the grouping in 4.5), and the recommendation should be "project
   these blocks to here, exclude those", not a single number.
7. **Issue-body drift.** #44 and #45 still describe the deleted sketch engine and
   the moot parity gate (Section 3). The planner should reconcile the issue bodies
   with this raw-weight reality before writing the #44 plan, so the analyst is not
   sent to extend a file that no longer exists.

---

## Pointers

- Trace + access contract: `research/reports/r2-access-pattern-for-analysis.md`
- Collection verdict: `research/runs/EXP-43/verdict.md`
- Roll-up + superseded EXP-42 numbers: `research/runs/SUMMARY.md` (Milestone M4)
- Async-upload pre-merge review (the box-validation checklist):
  `research/reports/async-r2-upload-review.md`
- Method wiring: `CODE_WALKTHROUGH.md`; project north-star: `research/.claude/GOAL.md`
- Observer / sink / config / verifier: `verl/workers/comm_eff/capture.py`,
  `verl/workers/comm_eff/r2_sink.py`, `verl/workers/config/comm_eff.py`,
  `research/scripts/verify_full_weight_dump.py`, `verl/workers/engine_workers.py`
  (`comm_eff_close`), `verl/trainer/ppo/ray_trainer.py` (close call sites).
