# Communication-Efficient GRPO Configuration

This is the operator reference for the comm-eff launcher:

```bash
examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

The default path is PowerSGD activation compression plus an anchor-owned basis
and delayed error-feedback merger. Compression applies to training
pipeline-boundary activations; rollouts still come from the ordinary unmasked
vLLM policy.

All knobs are env overrides read by the launcher and forwarded to
`actor_rollout_ref.actor.comm_eff.*`. The Hydra dataclass defaults remain
all-off so `comm_eff.enabled=false` is a dense no-op.

## Master Switch

| env | hydra | launcher default | meaning |
|---|---|---:|---|
| `COMM_EFF_ENABLED` | `comm_eff.enabled` | `true` | Turn the comm-eff path on. Set `false` for dense verl behavior. |
| `COMM_EFF_COMPRESSION_TYPE` | `comm_eff.compression_type` | `powersgd` | Select `powersgd`, `prf_mask`, or `dense`. |
| `COMM_EFF_CLEAN_CADENCE` | `comm_eff.clean_cadence` | `0` | Optional periodic dense step for diagnostics. Keep `0` for the baseline path. |

## PowerSGD Codec

| env | hydra | launcher default | meaning |
|---|---|---:|---|
| `COMM_EFF_POWERSGD_RANK` | `powersgd.rank` | `77` | Low-rank width for boundary activation sketches. |
| `COMM_EFF_POWERSGD_UPDATE_CADENCE` | `powersgd.update_cadence` | `1` | Fast-path basis update cadence. Gated off when `anchor.owns_q=true`. |
| `COMM_EFF_POWERSGD_WARM_START` | `powersgd.warm_start` | `true` | Reuse the prior basis as the next update seed. |
| `COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE` | `powersgd.compress_recompute` | `true` | Apply the same compression to the old-log-prob recompute path. |
| `COMM_EFF_POWERSGD_SYNC_BASIS` | `powersgd.sync_basis` | `true` | Share a consensus basis across data-parallel ranks. |
| `COMM_EFF_POWERSGD_QR_DTYPE` | `powersgd.qr_dtype` | `fp32` | QR precision used to orthogonalize the basis. |
| `COMM_EFF_POWERSGD_Q_BASIS` | `powersgd.q_basis` | `act` | Live basis family. |
| `COMM_EFF_POWERSGD_Q_BASIS_PASSIVE` | `powersgd.q_basis_passive` | `[]` | Optional passive basis families accumulated by the anchor only. |

## Anchor

The anchor periodically replays a delayed snapshot to obtain a full-gradient
reference `M` and, when `anchor.owns_q=true`, refreshes the PowerSGD basis `Q`.

| env | hydra | launcher default | meaning |
|---|---|---:|---|
| `COMM_EFF_ANCHOR_ENABLED` | `anchor.enabled` | `true` | Enable the anchor pass. |
| `COMM_EFF_ANCHOR_CADENCE` | `anchor.cadence` | `5` | Refresh cadence in optimizer ticks. |
| `COMM_EFF_ANCHOR_DELAY_K` | `anchor.delay_K` | `5` | Weight-snapshot delay in optimizer ticks. |
| `COMM_EFF_ANCHOR_OWNS_Q` | `anchor.owns_q` | `true` | Let the anchor be the only writer of the live PowerSGD basis. |
| `COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH` | `anchor.replay_paired_batch` | `true` | Replay the same batch/weights seen by the fast circuit. |
| `COMM_EFF_ANCHOR_SNAPSHOT_DEVICE` | `anchor.snapshot_device` | `cpu` | Store delayed snapshots on CPU or GPU. |

## Merger

The default merger is delayed error feedback:

```text
G_corr = G_comp + lambda * (M_rep - G_comp_ring)
```

Here `G_comp` is the fast compressed gradient, `M_rep` is the replayed anchor
gradient, and `G_comp_ring` is the fast compressed gradient saved for the same
delayed tick.

| env | hydra | launcher default | meaning |
|---|---|---:|---|
| `COMM_EFF_SPECTRAL_ENABLED` | `spectral.enabled` | `true` | Enable anchor-guided gradient correction. |
| `COMM_EFF_SPECTRAL_CORRECTION_MODE` | `spectral.correction_mode` | `delayed_ef` | Choose `delayed_ef`, `inject`, `blend`, or `ef_powersgd`. |
| `COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA` | `spectral.delayed_ef_lambda` | `1.0` | Dose for delayed-EF correction. `0.0` recovers plain PowerSGD. |
| `COMM_EFF_SPECTRAL_BETA_ANC` | `spectral.beta_anc` | `0.0` | EMA decay for anchor gradients. |
| `COMM_EFF_SPECTRAL_CADENCE` | `spectral.cadence` | `1` | Correction cadence in optimizer ticks. |
| `COMM_EFF_SPECTRAL_EMA_DEVICE` | `spectral.ema_device` | `cpu` | Store correction state on CPU or GPU. |
| `COMM_EFF_SPECTRAL_MAX_TARGETS` | `spectral.max_targets` | `-1` | Optional cap on corrected target matrices. |

Optional merger extensions:

| env prefix | purpose |
|---|---|
| `COMM_EFF_SPECTRAL_EF_*` | Error-feedback residual controls for `ef_powersgd`. |
| `COMM_EFF_SPECTRAL_DELTA_SUBBASIS_*` | Add a low-rank sub-basis to the delayed-EF correction. |
| `COMM_EFF_SPECTRAL_PERTURB_*` | Add deterministic cross-rank perturbation after correction. |
| `COMM_EFF_SPECTRAL_DELTA_MOMENTUM_*` | Keep momentum over correction deltas. |
| `COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_*` | Adjust delayed-EF dose from per-target agreement signals. |

## Legacy Mask Codec

`prf_mask` is kept for reference ablations and should not be mixed with
anchor-owned Q.

```bash
COMM_EFF_COMPRESSION_TYPE=prf_mask \
COMM_EFF_MASK_ENABLED=true \
COMM_EFF_ANCHOR_OWNS_Q=false \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

| env | hydra | default | meaning |
|---|---|---:|---|
| `COMM_EFF_MASK_ENABLED` | `mask.enabled` | `false` | Enable per-token/channel activation masking. |
| `COMM_EFF_MASK_P` | `mask.p` | `0.9` | Fraction of entries masked. |
| `COMM_EFF_MASK_RESCALE` | `mask.rescale` | `true` | Apply inverted-dropout rescale. |
| `COMM_EFF_MASK_RECOMPUTE` | `mask.mask_recompute` | `true` | Reuse the mask on old-log-prob recompute. |
| `COMM_EFF_MASK_SEED` | `mask.seed` | `0` | PRF seed. |
| `COMM_EFF_MASK_PP_SIZE` | `mask.pp_size` | `8` | Simulated pipeline boundary count. |

## Capture Probes

Capture is dump-only and should be off for normal runs.

| env | hydra | default | meaning |
|---|---|---:|---|
| `COMM_EFF_CAPTURE_ENABLED` | `capture.enabled` | `false` | Enable tensor dumps. |
| `COMM_EFF_CAPTURE_DIR` | `capture.capture_dir` | `/workspace/captures` | Output directory on the training box. |
| `COMM_EFF_CAPTURE_MAX_TICKS` | `capture.max_ticks` | `10` | Maximum captured optimizer ticks. |
| `COMM_EFF_CAPTURE_MIN_TICK` | `capture.min_tick` | `0` | Skip early ticks before capture starts. |
| `COMM_EFF_CAPTURE_G_DENSE` | `capture.capture_g_dense` | `false` | Also compute a dense-gradient probe. |
| `COMM_EFF_CAPTURE_FRESH_ANCHOR` | `capture.capture_fresh_anchor` | `false` | Also compute a delay-zero anchor probe. |

## Common Invocations

Dense reference:

```bash
COMM_EFF_ENABLED=false EXPERIMENT_NAME=dense_ref \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

Default comm-eff baseline:

```bash
EXPERIMENT_NAME=delayed_ef_comm_eff \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

Plain PowerSGD limiting case:

```bash
COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA=0.0 EXPERIMENT_NAME=powersgd_plain \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```
