# Communication-Efficient Path Walkthrough

This document is a compact map of the current communication-efficient code path.
Treat the code and config as the source of truth; run summaries live under
`research/runs/SUMMARY.md`.

## Mental Model

The path has two circuits:

1. Fast circuit: the normal actor train forward/backward, optionally compressed
   at pipeline boundaries by PRF masking or PowerSGD activation projection.
2. Anchor circuit: an uncompressed no-optimizer clone pass at a configured
   cadence. It produces anchor gradients and, when configured, owns the
   PowerSGD basis `Q`.

The merger, if enabled, rewrites selected fast gradients after backward and
before `optimizer_step()`.

## Main Files

| File | Role |
| --- | --- |
| `verl/workers/config/comm_eff.py` | Hydra dataclass schema and validation for mask, anchor, spectral merger, PowerSGD, capture, and probe options. |
| `verl/workers/comm_eff/state.py` | Per-worker state, counters, path tags, replay/ring buffers, compressor/filter construction, and metric export. |
| `verl/workers/comm_eff/powersgd_activation.py` | PowerSGD boundary projection hooks, basis bootstrap/update, anchor-owned Q path, basis family sketches, and byte counters. |
| `verl/workers/comm_eff/spectral_filter.py` | Per-target anchor EMA and merger modes: `delayed_ef`, `ef_powersgd`, `signed_ema`, `inject`, `blend`, and `none`. |
| `verl/workers/comm_eff/anchor.py` | Anchor staleness queue, paired replay ring, snapshot/canary helpers, gradient extraction, and geometry probe math. |
| `verl/workers/comm_eff/capture.py` | Optional fp32 tensor dumps keyed by `(global_step, optimizer_tick, role, target)`. |
| `verl/workers/engine/base.py` | Backend-neutral hook points in `train_batch()`. |
| `verl/workers/engine/fsdp/transformer_impl.py` | FSDP implementation of anchor refresh, G_dense capture, geometry probe, and gradient correction writeback. |
| `verl/workers/engine_workers.py` | Actor update wrapper, path tags, PowerSGD lifecycle, metrics, and end-of-step basis update. |

## Train Step Flow

1. `BaseEngine.train_batch()` zeroes grads.
2. `_maybe_comm_eff_anchor_refresh()` may run first. It loads a stale or paired
   replay snapshot into an isolated clone, runs the clean anchor backward, reads
   raw anchor gradients, DP-reduces them, updates `M`, and optionally updates and
   broadcasts anchor-owned `Q`.
3. `forward_backward_batch()` runs the fast path. It registers PRF or PowerSGD
   hooks only for the train path and unregisters them in `finally`.
4. Optional G_dense capture runs on a dedicated no-hook clone.
5. `_maybe_comm_eff_grad_correction()` may apply the spectral merger to selected
   full logical 2D gradients.
6. Optional geometry probe stages fast gradients and emits telemetry.
7. `optimizer_step()` consumes the final gradients.

## Correction Modes

`delayed_ef` uses paired replay plus the retained fast gradient from the matching
tick:

```text
delta(t)  = M_rep - G_comp_ring(t - K)
G_corr(t) = G_comp(t) + lambda * delta(t)
```

`ef_powersgd` adds a clipped off-subspace residual without replacing signs.
`signed_ema` replaces or mixes signs from the anchor EMA. `inject` and `blend`
are alternate anchor combiners. `none` leaves fast gradients unchanged while
allowing telemetry paths to run.

## Q Ownership

With `anchor.owns_q=true`, the fast PowerSGD path is a read-only consumer of
`Q`. The anchor pass harvests uncompressed activations into the sketch, computes
`Q <- orth(V)`, and broadcasts the result to DP ranks. The fast basis update is
fail-closed in this mode.

With `anchor.owns_q=false`, the fast path updates `Q` from its own compressed
train forwards at `powersgd.update_cadence`.

## Invariants

- Disabled `comm_eff` constructs no state, registers no hooks, draws no RNG, and
  issues no extra collectives.
- Anchor passes are unmasked, never take optimizer steps, and never generate
  rollouts or recompute rewards.
- Anchor gradients are read raw, DP-reduced with mean semantics, then fed into
  `M`; the merger never rewrites anchor gradients.
- Paired replay aligns the anchor gradient with the batch and generator weights
  that produced the matching fast circuit rollout.
- `max_targets=-1` means full target coverage. Non-negative caps are diagnostic
  throttles and cap both anchor extraction and merger writeback.
- Capture and probe code is dump-only/telemetry-only and must not feed the
  optimizer, EMA, sketch, `Q`, or loss.

## Debug Signals

Important train-log prefixes include:

- `[comm_eff][stale-replay]`: paired replay snapshot, batch, and canary checks.
- `[comm_eff][anchor-load]`: stale snapshot load coverage into the clone.
- `[comm_eff][dp-reduce]`: anchor gradient DP mean-reduction scale check.
- `[comm_eff][bcast]`: anchor-owned Q/M broadcast receipts.
- `[comm_eff][spectral]`: per-target correction writeback diagnostics.
- `[comm_eff][delayed_ef]`: delayed residual refresh/hold counters.
- `[comm_eff][capture]`: capture writer setup and dump metadata.
- `[geometry-probe]`: paired replay geometry telemetry.

Keep future notes anchored to current behavior and code contracts. Historical
run verdicts should stay in run summaries, not in source comments.
