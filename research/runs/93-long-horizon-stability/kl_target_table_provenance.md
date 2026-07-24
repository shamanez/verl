# COMM_EFF_PROBE_KL_TARGET_TABLE provenance (corrected pass)

## Correction notice

A previous pass computed this table from WandB and stopped at step 400
because the WandB run for `90-dense-600` (entity `shamanework-pl`, project
`90-dense-600`) shows `state: crashed` with history truncated at step 408.
**That WandB state is a sync artifact, not a training crash.** The run
continued training on-box well past step 408 and reached the full 600 steps.
This pass re-derives the table from the on-box `train.log`, which is the
source of truth for this project (see WandB-final-step-drop pattern noted
for this workload).

## Source

- Box: `45621340` (dense reference run `90-dense-600`, read-only access via
  `ssh -i ~/.ssh/vast_ai -p 40222 root@145.241.108.98`)
- File: `/workspace/runs/90-dense-600/train.log`
- Method: parsed every line containing both `actor/kl_loss:<float>` and
  `training/global_step:<int>` with a regex extracting both fields from the
  same line, keyed on the explicit `training/global_step` field (not on line
  order or tqdm progress-bar text). tqdm `N/600 [` progress lines were used
  only as a secondary liveness signal, never as the step source for a metric.

## Coverage

The parsed series is contiguous and clean:

- min step 1, max step 600
- 600 (step, actor/kl_loss) pairs total
- 0 duplicate steps, 0 missing steps in [1, 600]

The run reached its target of 600/600 steps on-box (final tqdm line:
`Training Progress: 100%|##########| 600/600 [11:31:02<00:00, 72.18s/it]`).
All 12 setpoint-table steps (50, 100, ..., 600) were exact logged hits, so
**no interpolation was needed anywhere in this table.**

## Formula

`setpoint(step) = round(max(0.005, 2 * actor_kl_loss(step)), 6)`

The 0.005 floor and the 2x factor are applied as two separate steps, per the
task spec.

## Raw actor/kl_loss values used and derived setpoints

| step | actor/kl_loss (raw)     | 2x raw                  | setpoint (floor 0.005 applied) |
|------|--------------------------|--------------------------|----------------------------------|
| 50   | 0.0024895993228710722   | 0.0049791986457421444   | 0.005 (floor applied) |
| 100  | 0.0038317571925290395   | 0.007663514385058079    | 0.007664 |
| 150  | 0.004434461101482157    | 0.008868922202964314    | 0.008869 |
| 200  | 0.0056325753685086966   | 0.011265150737017393    | 0.011265 |
| 250  | 0.006949810282094404    | 0.013899620564188808    | 0.0139 |
| 300  | 0.007384579017525539    | 0.014769158035051078    | 0.014769 |
| 350  | 0.009041579440236092    | 0.018083158880472184    | 0.018083 |
| 400  | 0.01032971365202684     | 0.02065942730405368     | 0.020659 |
| 450  | 0.010299785120878369    | 0.020599570241756738    | 0.0206 |
| 500  | 0.012322535185376182    | 0.024645070370752364    | 0.024645 |
| 550  | 0.013930435496149585    | 0.02786087099229917     | 0.027861 |
| 600  | 0.015604312589857727    | 0.031208625179715454    | 0.031209 |

Only step 50 hits the 0.005 floor (raw 2x value 0.004979 is just under it).
All later steps are past the floor and reflect 2x the dense control's
actor/kl_loss directly.

## Resulting table string

```
50:0.005,100:0.007664,150:0.008869,200:0.011265,250:0.0139,300:0.014769,350:0.018083,400:0.020659,450:0.0206,500:0.024645,550:0.027861,600:0.031209
```

## Cross-check against issue #93 (0.0049 @ step 173)

Parsed value at step 173: `actor/kl_loss = 0.004936533041473012`. This is
within 3.65e-5 of the issue body's recorded reference value of 0.0049
(about 0.7% relative difference, consistent with the issue value being
reported to 2 significant figures). This matches and validates the parse.

## Monotonicity and shape check

- **Monotone increasing**: holds for all practical purposes. Across the full
  600-step raw series there is exactly one small non-monotonic step in the
  12 sampled setpoint steps: step 450 (0.0102998) is very slightly below
  step 400 (0.0103297), a decrease of about 3.0e-5 (0.29% relative), which is
  within the noise band of a single-step, unsmoothed metric. A 21-step
  centered rolling average of the full 600-point series is strictly monotone
  increasing at every one of the 12 sampled points, confirming the dip is
  point noise, not a reversal of the underlying trend.
- **Concavity: FLAGGED, does not hold as cleanly as expected.** The task's
  cross-check expects the series to be "roughly monotone increasing and
  concave" (i.e., decelerating growth). The raw and smoothed data instead
  show mild acceleration in the back half of training: average per-50-step
  increment is about 0.00109 over steps 50-300 versus about 0.00131 over
  steps 300-600 (roughly 20% higher growth rate in the second half). A
  quadratic least-squares fit over the full 600-point series has a small but
  consistently positive curvature coefficient (convex), not negative
  (concave). This is a real property of this run's KL trajectory, not a
  parsing artifact: the parse is independently verified by (a) the step-173
  cross-check matching the issue's recorded 0.0049, and (b) complete, gapless,
  duplicate-free step coverage from 1 to 600. Flagging this loudly per
  instructions rather than silently asserting concavity: **the dense
  reference run's KL growth is monotone increasing but mildly convex
  (accelerating), not concave (decelerating), over the 600-step horizon.**

## WandB truncation note

The WandB run for this experiment (entity `shamanework-pl`, project
`90-dense-600`) shows `state: crashed` with history ending at step 408. This
is a known WandB sync-drop pattern for this workload (the trainer's atexit
teardown can race the final WandB flush) and does **not** reflect the actual
training run, which continued uninterrupted on-box to step 600 as shown
above. Any consumer of this table should treat the on-box train.log, not
WandB, as the source of truth for this run's full history.
