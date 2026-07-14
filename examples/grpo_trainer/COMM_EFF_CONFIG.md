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

## Qwen2.5-Math/MATH compression default (pending)

The exact `Qwen/Qwen2.5-Math-1.5B` + MATH train/test comparison is a separate,
surface-scoped benchmark from the GSM8K operating baseline below. Its default
selection is intentionally pending. Corrected W2, corrected strict-readiness
W4, and the qboot-v2 no-projected-weight-increment control are complete. The
control was valid but destabilized late (67.31% at step 75 to 61.90% at step
100); the matched qboot-v2 projection system is now active. The old progressive-W4,
W2-no-increment, and `fixed_linear` placeholders are superseded/not queued;
`fixed_linear` remains an optional legacy follow-up. The authoritative
placeholder and preregistered selection rule live at
`research/.claude/project.yaml` → `compression_defaults.math_qwen25_math_1p5b`.

The current W=4 values in
`run_qwen25_math_1p5b_rank1_relex_fsdp.sh` are provisional experiment values,
not a selected champion. After the matrix closes, the best valid compressed arm
will be promoted to the neutral
`run_qwen25_math_1p5b_comm_eff_default_fsdp.sh` launcher so omitting method
knobs selects it; explicit overrides will remain available. The claim will be
“best observed on the locked single-seed matrix,” not literature/global SOTA
without broader replication and comparison.

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
| `COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP` | `powersgd.fast_q_bootstrap` | `false` | Opt in to one discarded dense fast-actor observation on the first rollout batch, then DP-sync and atomically activate Q before the first compressed old/current PPO pair. Requires anchor-owned Q, recompute compression, synchronized `act` Q, and PowerSGD. |
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
| `COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR` | `anchor.lookahead_anchor` | `false` | Enable an opt-in anchor weight projector. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_MODE` | `anchor.lookahead_mode` | `disabled` | `disabled`, unchanged `fixed_linear`, or sliding `rank1_relex`. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH` | `anchor.lookahead_strength` | `1.0` | Scale the projected one-delay-horizon increment. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE` | `anchor.lookahead_rollout_source` | `auto` | Use current trajectories on projected fires when `auto`. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS` | `anchor.lookahead_window_snapshots` | `4` | Rank-1 checkpoint window including the oldest base; `W=2` is explicitly the per-tensor two-checkpoint secant/naive-linear fallback, while `W>=3` uses rank-1 OLS. Every unique floating named parameter tensor is projected independently. |
| `COMM_EFF_ANCHOR_WARMUP_MODE` | `anchor.warmup_mode` | `stale_correct` | `stale_correct` runs the paired exact anchor backward and populates M before rank-1 readiness; `q_only` refreshes only anchor-owned Q; `no_correct` is the fast-owned-Q ablation. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS` | `anchor.lookahead_min_snapshots` | `-1` | `-1` waits for the complete configured window. For rank-1, a concrete value in `[2,W]` enables progressive readiness while retaining at most W checkpoints; `2` starts with the earliest legal secant. |

\* The baseline launcher pins `cadence`/`delay_K` = 20/20 (the k-collapse regime).
The generic engine's bare default is 5/5; the baseline wrapper overrides it.

The anchor backward is dense and uncompressed, but it is intentionally not the
fast path's clipped PPO loss: compressed-policy `old_log_probs` are not a valid
importance-ratio denominator for a stale or projected uncompressed anchor. It
therefore uses a ratio-one advantage policy gradient and retains the fast
configuration's reference-policy KL penalty, type, coefficient, mask, and loss
normalization. On the locked MATH surface this is `low_var_kl` with coefficient
`0.001`. A KL-enabled anchor fails closed when `ref_log_prob` is absent.

### Strict versus progressive W4 readiness

The retained legacy comparison launcher keeps the original strict W4 arm
unchanged with `window_snapshots=4` and `min_snapshots=-1`. Its preregistered
but unrun `w4_progressive` arm used `window_snapshots=4` and
`min_snapshots=2`: it did not shrink the target window, but would have started
projecting and populating M as soon as two exact checkpoints existed. The
qboot-v2 composite supersedes that queued placeholder while retaining its
W2/W3/W4 history-growth idea. On this fixed C=K=20 surface with two optimizer
ticks per global step, the original intended schedule was:

| global step | optimizer tick | available checkpoints | action |
|---:|---:|---:|---|
| 10 | 20 | 1 | Q-only; M and correction remain disabled |
| 20 | 40 | 2 | all-tensor W2 secant; first dense anchor backward/M update |
| 30 | 60 | 3 | all-tensor W3 rank-1 fit |
| 40 onward | 80 onward | sliding W4 | full all-tensor W4 rank-1 OLS |

That old progressive-versus-strict comparison would have moved earlier
projection and earlier M activation together. qboot-v2 instead makes M ready
at the first fire for both new arms and uses the same W2/W3/W4 history schedule
in both. W3 has two cumulative deltas, so its direction is useful but its
two-point temporal OLS R² is tautologically one and must not be described as a
strong denoising diagnostic.

### qboot-v2 first-fire and progressive schedule

The run-ready qboot-v2 launcher is
`run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh`. At global step 1,
after the initial rollouts exist but before their first old-log-probability
calculation, both arms run one discarded dense no-grad observation through the
exact fast actor on that rollout batch, build the consensus activation basis,
and activate it before the real old-log-probability forward. This is a Q-only
bootstrap: it does not compute gradients or M, and the first real PPO old and
current-policy forwards therefore see the same Q.

With C=K=20 optimizer ticks and two optimizer ticks per global step, both arms
then follow this same anchor schedule:

| global step | retained exact checkpoints | shared projector/anchor action |
|---:|---:|---|
| 1, pre-PPO | 0 | After initial rollout generation, dense fast observation on that batch; atomically install Q1 before old/current-policy forwards. No gradient or M. |
| 10 | 1 | `stale_correct`: paired exact initial/base checkpoint and batch; dense backward; first all-floating M; stage the next anchor Q. |
| 20 | 2 | Per-tensor W2 secant; dense backward/M/Q. |
| 30 | 3 | Per-tensor W3 rank-1 OLS; dense backward/M/Q. |
| 40 onward | sliding 4 | Per-tensor W4 rank-1 OLS; dense backward/M/Q. |

The `no_weight_increment` arm retains the same W2/W3/W4 history, fires, M/Q
work, and projection telemetry as the composite, but sets
`lookahead_strength=0.0`. Its applied anchor weights are therefore exactly the
newest transferred checkpoint, and `lookahead_rollout_source=stale_paired`
routes that checkpoint's exact paired trajectories. The composite changes to
`strength=1.0` and `rollout_source=auto`, which resolves to current trajectories
on projected fires. Those trajectories are target-tick aligned but were
generated by the live fast actor, not by the projected weights; they reduce
temporal batch staleness without forming a literal on-policy projected pair.
This is the requested systems control: the
projector still computes diagnostics, but it cannot alter the control arm's
weights. Because trajectory routing also differs, it is not a pure scalar-dose
ablation.

## Merger — signed_ema

The merger folds the anchor `M` into the fast gradient via a signed EMA.

| env | hydra | launcher default | meaning |
|---|---|---:|---|
| `COMM_EFF_SPECTRAL_ENABLED` | `spectral.enabled` | `true` | Enable anchor-guided gradient correction. |
| `COMM_EFF_SPECTRAL_TARGET_SCOPE` | `spectral.target_scope` | `decoder_matrices` | `decoder_matrices` preserves the 196 substring-selected 2-D tensors. `all_floating` covers every de-duplicated floating parameter with a gradient, including embeddings, norms, biases, and an untied LM head. |
| `COMM_EFF_SPECTRAL_DIAGNOSTICS` | `spectral.diagnostics` | `true` | Per-target diagnostic metrics. qboot-v2 pins this `false`; coverage/counter/integrity guards remain active without emitting hundreds of per-tensor values. |
| `COMM_EFF_SPECTRAL_CORRECTION_MODE` | `spectral.correction_mode` | `signed_ema` | The merger mode. |
| `COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA` | `spectral.signed_ema_alpha` | `0.25` | signed_ema mixing weight. |
| `COMM_EFF_SPECTRAL_BETA_ANC` | `spectral.beta_anc` | `0.50` | Anchor-gradient EMA decay. |
| `COMM_EFF_SPECTRAL_CADENCE` | `spectral.cadence` | `1` | Correction cadence in optimizer ticks. |
| `COMM_EFF_SPECTRAL_EMA_DEVICE` | `spectral.ema_device` | `cpu` | Store correction state on CPU (OOM guard). |
| `COMM_EFF_SPECTRAL_MAX_TARGETS` | `spectral.max_targets` | `-1` | `-1` = full coverage under the selected target scope. |

For Qwen2.5-Math-1.5B, `all_floating` is 338 unique tensors: the existing 196
decoder weights plus 84 q/k/v biases, 57 norms, and one tied embedding/LM-head
tensor. Its fp32 M is 5.751 GiB per rank (4.881 GiB for the prior 196 plus
0.870 GiB added). Rank-local copies scale linearly with actor DP size (about
11.5 GiB total across this experiment's two actor ranks); each anchor fire also
adds roughly 0.87 GiB of logical fp32 payload to both the anchor-gradient
reduction and M broadcast.
The qboot-v2 arms therefore pin `ema_device=cpu`, `max_targets=-1`, and
`diagnostics=false`. Runtime coverage counters, not an architecture-hard-coded
count, remain authoritative. The geometry/SVD probe stays restricted to its
historical decoder-matrix subset.

## Capture probes

Capture is dump-only and off for normal runs (OOM hazard).

| env | hydra | default | meaning |
|---|---|---:|---|
| `COMM_EFF_CAPTURE_ENABLED` | `capture.enabled` | `false` | Enable tensor dumps. |
| `COMM_EFF_CAPTURE_DIR` | `capture.capture_dir` | `/workspace/captures` | Output directory on the training box. |
| `COMM_EFF_CAPTURE_MAX_TICKS` | `capture.max_ticks` | `10` | Maximum captured optimizer ticks. |
| `COMM_EFF_CAPTURE_MIN_TICK` | `capture.min_tick` | `0` | Skip early ticks before capture starts. |

The causal sampled-weight probe is much smaller than tensor capture. It stores
only deterministic scalar samples from one embedding, one middle decoder
matrix, one middle layer norm, and the final norm. A forecast is scored only
when its target checkpoint later arrives through the normal delayed-transfer
path; it never peeks at the current fast network. Positive `skill` means the
projection beat the newest exact/stale baseline on those samples.

| env | hydra | default | meaning |
|---|---|---:|---|
| `COMM_EFF_RANK1_PROJECTION_PROBE_ENABLED` | `probe.rank1_projection_enabled` | `false` | Enable delayed sampled-weight verification for `rank1_relex`. |
| `COMM_EFF_RANK1_PROJECTION_PROBE_SAMPLES` | `probe.rank1_projection_samples` | `16` | Deterministic scalar samples per representative tensor (`1..64`). |
| `COMM_EFF_PROBE_OUT_DIR` | `probe.out_dir` | `<run>/rank1_projection_probe` | Directory for `rank1_projection_samples.jsonl`. |
| `COMM_EFF_PROBE_RANK0_ONLY` | `probe.rank0_only` | `true` | Write JSON/stdout on rank 0 only. |

## Common invocations

Run the qboot-v2 matrix in its preregistered order (two-circuit with zero applied
projected weight increment, then the complete composite), or pass one arm
explicitly:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh composite
```

The earlier comparison launcher is retained for provenance and explicit
legacy arms; it is not the active post-W4 queue:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_comparison_fsdp.sh fixed_linear
```

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
