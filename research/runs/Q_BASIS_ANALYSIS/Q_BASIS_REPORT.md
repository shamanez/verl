# Q-Basis Analysis: K=20 vs K=5

Scope: only the two W&B runs requested.

- K=20: `fxo8chsv`, `exp-37-cad20-delay20`
- K=5: `rsvo7y1p`, `exp-36-c2eff-055-diag`

The production runs did not capture full Q tensors (`capture.enabled=false`), so this analysis
cannot measure principal angles or per-column Q motion directly. It uses the scalar W&B telemetry
that is available: anchor Q update counters, fast PowerSGD update counters, Q reconstruction error,
per-boundary reconstruction error, and Q condition number.

## Pipeline Finding

With `anchor.owns_q=true`, the fast path is a read-only consumer of Q. The fast PowerSGD update
cadence is configured as `powersgd.update_cadence=1`, but it does not write Q in these runs:
`actor/comm_eff/powersgd_basis_updates` stays 0 in both histories.

The anchor is the sole Q writer. It refreshes Q when the anchor fires, using the slow clean anchor
pass activation sketch and then broadcasting Q. Therefore the effective Q update cadence is:

- K=20 run: `anchor.cadence=20`, `delay_K=20`
- K=5 run: `anchor.cadence=5`, `delay_K=5`

There are two optimizer ticks per logged global training step on this surface.

## What Happens Before The First Q Update

Before the first anchor fire, Q is the deterministic random bootstrap basis. Both runs show nearly
the same behavior:

- anchor Q updates: 0
- anchor backwards: 0
- fast PowerSGD basis updates: 0
- Q reconstruction error: about 0.975
- Q condition number: about 1.0000003
- cold-M fallbacks: 196

So the bootstrap Q is numerically well-conditioned but very poor at reconstructing boundary
activations. This is expected: it is orthonormal, but not yet fitted to the activation distribution.

The difference is how long each run remains in that bootstrap-Q state:

- K=5 first anchor Q update appears in W&B at global step 3. The exact fire is optimizer tick 5;
  W&B reports the row after the two-tick global step, so `spectral_step=6`.
- K=20 first anchor Q update appears at global step 10, optimizer tick 20.

## After The First Q Update

At the first anchor Q update, reconstruction error collapses in both runs:

- K=5: about 0.975 -> 0.042
- K=20: about 0.975 -> 0.040

After that, both runs usually sit near 0.03-0.04 reconstruction error. The Q basis itself does not
collapse: `q_cond - 1` remains on the order of 2.5e-7 to 3.6e-7.

The notable late difference is K=20 after step 90: Q reconstruction error rises from about 0.032 to
0.055 by step 100, while response length also spirals. That is a real deterioration in the Q error
metric, but it occurs late and after many successful anchor-owned Q refreshes; it is not a bootstrap
or orthogonality failure.

## Summary Table

See `q_basis_summary.csv`.

| run | first learned-Q row | final anchor Q updates | fast Q updates | pre-update Q error mean | post-update Q error mean | final Q error |
|---|---:|---:|---:|---:|---:|---:|
| K=20 `fxo8chsv` | step 10 | 10 | 0 | 0.9750 | 0.0342 | 0.0554 |
| K=5 `rsvo7y1p` | step 3 | 20 | 0 | 0.9752 | 0.0339 | 0.0335 |

## Plots

- `figs/q_error_and_update_counters.png`
- `figs/pre_first_q_update_zoom.png`
- `figs/q_error_by_update_window.png`
- `figs/per_boundary_q_error_heatmaps.png`
- `figs/q_error_with_clipfrac_and_length.png`

