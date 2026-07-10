# Communication-Efficient GRPO Configuration

Operator reference for the comm-eff launcher. The canonical entry point — and
the current **baseline** — is:

```bash
examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
```

It pins the whole surface + substrate + merger and execs the generic engine
(`vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`). A bare run reproduces the
baseline; override only run length / name.

The method is PowerSGD activation compression at the training pipeline
boundaries, with an **anchor circuit** that owns the projection basis and feeds
a **signed_ema** gradient merger. Rollouts still come from the ordinary unmasked
vLLM policy. The Hydra dataclass defaults remain all-off, so
`comm_eff.enabled=false` is a dense no-op (byte-identical to upstream verl).

All knobs are env overrides read by the launcher and forwarded to
`actor_rollout_ref.actor.comm_eff.*`.

## Current baseline (the problem state)

The baseline deliberately sits in the **k-collapse regime**: high anchor latency
(`cadence`/`delay_K` = 20/20), where the stale anchor gradient has rotated
~orthogonal to the live gradient. This is the failure Priority 1 targets — see
`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh` (the
authoritative locked-surface value sheet) and
`research/reports/priority-1-anchor-staleness-k-collapse.html`.

| knob | baseline value |
|---|---|
| compression | `powersgd`, rank 77, anchor owns `Q` |
| anchor latency | `cadence` 20 / `delay_K` 20 (k-collapse regime) |
| merger | `signed_ema`, α=0.25, β_anc=0.50 |
| surface | resp 1024, dynamic-bsz, rollout TP=1, gpu_mem 0.55, 50 steps, val@25/50 |

## Master switch

| env | hydra | launcher default | meaning |
|---|---|---:|---|
| `COMM_EFF_ENABLED` | `comm_eff.enabled` | `true` | Turn the comm-eff path on. Set `false` for dense verl behavior. |
| `COMM_EFF_COMPRESSION_TYPE` | `comm_eff.compression_type` | `powersgd` | Compression codec. |
| `COMM_EFF_CLEAN_CADENCE` | `comm_eff.clean_cadence` | `0` | Periodic uncompressed optimizer steps — disabled; the anchor replaces them. |

## PowerSGD codec

| env | hydra | launcher default | meaning |
|---|---|---:|---|
| `COMM_EFF_POWERSGD_RANK` | `powersgd.rank` | `77` | Low-rank width for boundary activation sketches. |
| `COMM_EFF_POWERSGD_UPDATE_CADENCE` | `powersgd.update_cadence` | `1` | Fast-path basis update cadence. Gated off when `anchor.owns_q=true`. |
| `COMM_EFF_POWERSGD_WARM_START` | `powersgd.warm_start` | `true` | Reuse the prior basis as the next update seed. |
| `COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE` | `powersgd.compress_recompute` | `true` | Apply the same compression to the old-log-prob recompute path. |
| `COMM_EFF_POWERSGD_SYNC_BASIS` | `powersgd.sync_basis` | `true` | Share a consensus basis across data-parallel ranks. |
| `COMM_EFF_POWERSGD_QR_DTYPE` | `powersgd.qr_dtype` | `fp32` | QR precision used to orthogonalize the basis. |
| `COMM_EFF_POWERSGD_Q_BASIS` | `powersgd.q_basis` | `act` | Live basis family. |

## Anchor

The anchor periodically replays a delayed snapshot to obtain a full-gradient
reference `M` and, when `anchor.owns_q=true`, refreshes the PowerSGD basis `Q`.

| env | hydra | launcher default | meaning |
|---|---|---:|---|
| `COMM_EFF_ANCHOR_ENABLED` | `anchor.enabled` | `true` | Enable the anchor pass (mandatory). |
| `COMM_EFF_ANCHOR_CADENCE` | `anchor.cadence` | `20`* | Refresh cadence in optimizer ticks. |
| `COMM_EFF_ANCHOR_DELAY_K` | `anchor.delay_K` | `20`* | Weight-snapshot delay in optimizer ticks. |
| `COMM_EFF_ANCHOR_OWNS_Q` | `anchor.owns_q` | `true` | Let the anchor be the only writer of the live PowerSGD basis. |
| `COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH` | `anchor.replay_paired_batch` | `true` | Replay the same batch/weights seen by the fast circuit. |
| `COMM_EFF_ANCHOR_SNAPSHOT_DEVICE` | `anchor.snapshot_device` | `cpu` | Store delayed snapshots on CPU or GPU. |

\* The baseline launcher pins `cadence`/`delay_K` = 20/20 (the k-collapse regime).
The generic engine's bare default is 5/5; the baseline wrapper overrides it.

## Merger — signed_ema

The merger folds the anchor `M` into the fast gradient via a signed EMA.

| env | hydra | launcher default | meaning |
|---|---|---:|---|
| `COMM_EFF_SPECTRAL_ENABLED` | `spectral.enabled` | `true` | Enable anchor-guided gradient correction. |
| `COMM_EFF_SPECTRAL_CORRECTION_MODE` | `spectral.correction_mode` | `signed_ema` | The merger mode. |
| `COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA` | `spectral.signed_ema_alpha` | `0.25` | signed_ema mixing weight. |
| `COMM_EFF_SPECTRAL_BETA_ANC` | `spectral.beta_anc` | `0.50` | Anchor-gradient EMA decay. |
| `COMM_EFF_SPECTRAL_CADENCE` | `spectral.cadence` | `1` | Correction cadence in optimizer ticks. |
| `COMM_EFF_SPECTRAL_EMA_DEVICE` | `spectral.ema_device` | `cpu` | Store correction state on CPU (OOM guard). |
| `COMM_EFF_SPECTRAL_MAX_TARGETS` | `spectral.max_targets` | `-1` | `-1` = full coverage (all 196 matrices). |

## Capture probes

Capture is dump-only and off for normal runs (OOM hazard).

| env | hydra | default | meaning |
|---|---|---:|---|
| `COMM_EFF_CAPTURE_ENABLED` | `capture.enabled` | `false` | Enable tensor dumps. |
| `COMM_EFF_CAPTURE_DIR` | `capture.capture_dir` | `/workspace/captures` | Output directory on the training box. |
| `COMM_EFF_CAPTURE_MAX_TICKS` | `capture.max_ticks` | `10` | Maximum captured optimizer ticks. |
| `COMM_EFF_CAPTURE_MIN_TICK` | `capture.min_tick` | `0` | Skip early ticks before capture starts. |

## Common invocations

Baseline (reproduces the collapse-regime config):

```bash
EXPERIMENT_NAME=ce_baseline \
bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
```

Push to 100 steps to actually manifest the collapse (~step 61):

```bash
TOTAL_TRAINING_STEPS=100 EXPERIMENT_NAME=ce_baseline_100 \
bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
```

Dense control (comm-eff OFF) on the same surface:

```bash
COMM_EFF_ENABLED=false USE_DYNAMIC_BSZ=True MAX_RESPONSE_LENGTH=1024 ROLLOUT_TP=1 \
  ROLLOUT_GPU_MEM_UTIL=0.55 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
  TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 VAL_BEFORE_TRAIN=False EXPERIMENT_NAME=dense_ref \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```
