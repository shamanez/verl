# Communication-Efficient GRPO Configuration

Operator reference for the current Qwen2.5-Math/MATH communication-efficient
surface. The latest completed single-arm reference command is:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh composite
```

The explicit `composite` argument matters: a bare invocation runs the two-arm
matrix. This command is the latest completed diagnostic reference, not a
scientifically promoted compression champion.

The method is rank-77 PowerSGD activation projection at the training boundaries,
with an **anchor circuit** that owns `Q` and feeds a **signed_ema** gradient
merger. RELEX separately projects delayed anchor weights. Rollouts still come
from the ordinary unmasked vLLM policy. Raw Hydra defaults remain all-off, so
`comm_eff.enabled=false` is the dense compatibility path.

All knobs are env overrides read by the launcher and forwarded to
`actor_rollout_ref.actor.comm_eff.*`. Tables below give the qboot-v2 composite
reference values unless they explicitly say generic or optional.

## Current Qwen2.5-Math/MATH reference (no selected champion)

The active answer surface is `Qwen/Qwen2.5-Math-1.5B` + native MATH train/test.
Corrected W2, corrected strict-readiness W4, and both qboot-v2 arms are complete
as pre-anchor-KL diagnostics. The no-increment control was valid but destabilized
late (67.31% at step 75 to 61.90% at step 100); the matched
projection/current-trajectory composite ended at 66.85%. Corrected W2 won the
diagnostic primary at 67.89%, 1.04 points above the composite.

The objective-parity-complete W2 validation was stopped by the operator at
global step 12, after its first transactional Q-only fire but before the first
dense anchor backward at global step 20. It therefore supplies no GPU evidence
for the corrected M path. No neutral implicit-default launcher is created by
this experiment handoff. The generic Hydra switch remains all-off, and a future
promotion requires a completed objective-parity validation plus broader
replication. The evidence supports continued investigation of the combined
projection/current-trajectory package, not a causal claim for projection alone
or a literature/global SOTA claim. The old progressive-W4, W2-no-increment, and
`fixed_linear` placeholders are superseded/not queued; `fixed_linear` remains an
optional legacy follow-up.

### Operator knobs and the v9sfxnaz reference run

Treat these controls as a coupled experimental contract:

| Question | Knob / invariant | Operational caution |
|---|---|---|
| How many prompts feed Q and M? | `COMM_EFF_ANCHOR_BATCH_SCOPE=ppo_minibatch` selects 256 prompts/2,048 responses on this surface; `rollout_batch` selects all 512 prompts/4,096 responses. | This is a shared Q+M scope. Full scope roughly doubles anchor compute/replay payload and is not yet GPU-validated. |
| Does anchor M match the fast objective? | Fast and anchor must share policy-loss mode, rollout weights, entropy, reference-KL type/coefficient, response mask, and aggregation. | The anchor intentionally uses ratio one and no PPO clipping. Unsupported or missing objective inputs fail closed. Never change fast KL without verifying the resolved anchor contract and `parity=PASS`. |
| When does projection become ready? | `COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS=W`; `COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS=-1` is strict readiness, while `2` is progressive. | W=2 is a per-tensor secant. W>=3 uses rank-1 OLS. Window size does not change `cadence` or `delay_K`; record all three. |
| How far is the anchor behind? | `COMM_EFF_ANCHOR_CADENCE` and `COMM_EFF_ANCHOR_DELAY_K`, expressed in optimizer ticks. | With two optimizer ticks per trainer step, the locked 20/20 values fire every 10 global steps. Do not describe them as 20 trainer steps. |

The completed [W&B `v9sfxnaz`](https://wandb.ai/shamanework-pl/verl_compression_research/runs/v9sfxnaz)
reference used commit `8bad0656`, 2x H200 NVL,
`Qwen/Qwen2.5-Math-1.5B`, MATH train/test, train batch 512, PPO mini-batch
256, micro-batch 1/GPU with dynamic batching, `shuffle=false`, rollout `n=8`,
prompt/response 1024/3072, train temperature 1, greedy validation, AdamW
`lr=1e-6`, weight decay 0.01, gradient clip 1, 100 steps, and validation at
0/25/50/75/100. Its fast objective used GRPO reference KL
(`low_var_kl`, coefficient 0.001), no reward-side KL, and entropy coefficient 0.

Its communication path used PowerSGD rank 77 with synchronized warm-started
activation Q, one-time fast-Q bootstrap, anchor cadence/delay 20/20, paired CPU
replay, `stale_correct`, progressive W2/W3/W4 readiness (`window=4`,
`min_snapshots=2`), strength 1, auto/current trajectories, and all-floating
signed-EMA M over 338 tensors (`alpha=0.25`, `beta_anc=0.5`). Because the
explicit batch-scope knob did not yet exist, its effective anchor scope was one
256-prompt PPO mini-batch; its legacy `anchor_batch_fraction=1` did not mean all
512 prompts.

The run scored 44.678/63.505/65.426/67.467/66.847% at steps
0/25/50/75/100, with communication ratio 0.05013569, final sampled mismatch KL
10.73, and aggregate projection-probe skill -0.058 (3/8 wins). Its Q and M
transaction counters were valid. However, the fast PPO included reference KL
while dense anchor M omitted it on that historical commit. `v9sfxnaz` is
therefore a useful pre-anchor-KL diagnostic, never evidence that the current
objective-parity implementation is GPU-validated.

## Historical GSM8K control

The original Qwen2.5-1.5B-Instruct / GSM8K k-collapse launchers have been removed
(2026-07-15). Their comm-eff engine survives, renamed to the model/dataset-neutral
`vast_comm_eff_engine_grpo.sh`, which every MATH launcher now `exec`s. GSM8K is no
longer a model/data default and must not be presented as the current baseline; the
current surface is Qwen2.5-Math-1.5B / MATH (`project.yaml`
`compression_defaults.math_qwen25_math_1p5b`).

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
| `COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP` | `powersgd.fast_q_bootstrap` | `true` | Run one discarded dense fast-actor observation on the first rollout batch, then DP-sync and atomically activate Q before the first compressed old/current PPO pair. Requires anchor-owned Q, recompute compression, synchronized `act` Q, and PowerSGD. |
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
| `COMM_EFF_ANCHOR_BATCH_SCOPE` | `anchor.batch_scope` | `ppo_minibatch` | `ppo_minibatch` uses one complete PPO mini-batch; `rollout_batch` uses the complete pre-split actor update. The selected scope is shared by Q and M. |
| `COMM_EFF_ANCHOR_SNAPSHOT_DEVICE` | `anchor.snapshot_device` | `cpu` | Store delayed snapshots on CPU or GPU. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR` | `anchor.lookahead_anchor` | `true` | Enable the anchor weight projector. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_MODE` | `anchor.lookahead_mode` | `rank1_relex` | `disabled`, unchanged `fixed_linear`, or sliding `rank1_relex`. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH` | `anchor.lookahead_strength` | `1.0` | Scale the projected one-delay-horizon increment. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE` | `anchor.lookahead_rollout_source` | `auto` | Use current trajectories on projected fires when `auto`. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS` | `anchor.lookahead_window_snapshots` | `4` | Rank-1 checkpoint window including the oldest base; `W=2` is explicitly the per-tensor two-checkpoint secant/naive-linear fallback, while `W>=3` uses rank-1 OLS. Every unique floating named parameter tensor is projected independently. |
| `COMM_EFF_ANCHOR_WARMUP_MODE` | `anchor.warmup_mode` | `stale_correct` | `stale_correct` runs the paired exact anchor backward and populates M before rank-1 readiness; `q_only` refreshes only anchor-owned Q; `no_correct` is the fast-owned-Q ablation. |
| `COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS` | `anchor.lookahead_min_snapshots` | `2` | `-1` waits for the complete configured window. For rank-1, a concrete value in `[2,W]` enables progressive readiness while retaining at most W checkpoints; `2` starts with the earliest legal secant. |

\* The qboot-v2 reference pins `cadence`/`delay_K` = 20/20 optimizer ticks.

### Anchor batch scope

`anchor.batch_scope` is an explicit signal-quality/cost knob:

- `ppo_minibatch` is the historical default. On the locked MATH surface, one
  anchor fire consumes 256 prompt groups × 8 responses = 2,048 response
  sequences, or exactly half of the 512-prompt actor update.
- `rollout_batch` consumes all 512 prompt groups × 8 responses = 4,096 response
  sequences. The private full batch is still processed through dynamic
  microbatches, and `token-mean` is normalized once by the DP-global valid-token
  count rather than averaging two separately normalized halves.

The scope is shared by the anchor-owned Q observation and dense M backward
because both are harvested from one anchor forward. It is therefore a combined
Q+M experiment, not a pure larger-M ablation. A pure `M512/Q256` comparison
would require a separate forward or separately gated Q harvest. Moving from 256
to 512 prompt groups should roughly halve prompt-sampling variance and reduce
standard error by `1/sqrt(2)` (about 29%), but it does not increase the expected
gradient magnitude and is not guaranteed to improve optimization. Anchor
compute and CPU replay storage are approximately doubled per retained/fire
batch; peak activation memory remains microbatch-bounded, and Q/M communication
volume is unchanged.

For that reason, `rollout_batch` is opt-in until a matched multi-seed ablation
justifies promotion:

```bash
COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
  bash examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh
```

Runtime telemetry reports response sequences, response-count/`rollout_n`
prompt-equivalents, `rollout_n`, the actual Q/M signal role, and
`anchor_batch_fraction` relative to the complete actor update. A shuffled PPO
mini-batch can split groups even when its row count is divisible by `rollout_n`,
so only `rollout_batch` intrinsically guarantees complete groups. The locked
surface uses `actor.shuffle=false`, making its historical 256-prompt count
group-preserving; its expected fractions are 0.5 and 1.0 respectively. Older
runs logged `anchor_batch_fraction=1.0` as “100% of the selected PPO
mini-batch.” The corrected metric denominator is the complete actor update, so
those legacy values must not be interpreted as 512-prompt anchor passes.

The anchor backward is dense and uncompressed, but it is intentionally not the
fast path's clipped PPO loss: compressed-policy `old_log_probs` are not a valid
importance-ratio denominator for a stale or projected uncompressed anchor. It
therefore uses a ratio-one advantage policy gradient and retains the fast
configuration's rollout importance weights, entropy regularizer,
reference-policy KL penalty, mask, and loss normalization. On the locked MATH
surface this is vanilla policy loss, `low_var_kl` with coefficient `0.001`,
entropy coefficient `0`, no rollout importance weights, and `token-mean`
aggregation. A KL-enabled anchor fails closed when `ref_log_prob` is absent; a
nonzero entropy coefficient fails closed when the forward did not return
entropy.

### Objective-parity contract

`M_anchor` controls the sign of the fast update, so “dense and uncompressed” is
not sufficient: it must also differentiate the same configured objective. A
silently omitted loss term changes the vector used as the correction signal.
The permanent invariant is:

> `M_anchor` is the dense, uncompressed gradient of the same resolved actor
> objective as the fast circuit, evaluated under the anchor's explicitly
> declared ratio-one surrogate. Unsupported terms fail closed; they are never
> silently dropped.

| Objective component | Fast actor | Dense anchor M |
|---|---|---|
| Advantage policy gradient | Vanilla PPO ratio and clipping against compressed-policy `old_log_probs` | Same advantages, response mask, and normalization, but ratio is fixed to one and clipping is removed |
| Rollout importance weights | Multiply the per-token policy-gradient term when `rollout_is_weights` is present | The identical weights multiply the ratio-one policy-gradient term |
| Entropy | Subtract `entropy_coeff * entropy_loss` | Same entropy tensor, coefficient, mask, and aggregation; missing entropy with a nonzero coefficient is an error |
| Reference-policy KL | Add configured `kl_loss_coef * KL(logpi, logpi_ref)` | Same `ref_log_prob`, KL type, coefficient, mask, and aggregation; missing reference log-probabilities are an error |
| Reward-side KL | Already folded into rewards and therefore advantages | Inherited through the exact same advantages |

The only standard exceptions are `old_log_probs`, PPO importance ratio, and PPO
clipping: they compare different compressed/uncompressed policies and are not a
valid anchor denominator. Active comm-eff anchors currently accept
`actor.policy_loss.loss_mode=vanilla` only; another loss mode requires an
explicitly implemented and tested ratio-one mapping before it can run.
`distillation_ppo_loss` is also rejected before training because its additional
terms do not yet have an anchor mapping; the runtime loss binder independently
rejects any fast callable other than plain `ppo_loss`. On the
first anchor loss each worker prints a resolved contract line containing the
loss mode, KL type/coefficient, entropy coefficient, rollout-weight presence,
aggregation, the intentional exceptions, and `parity=PASS`.

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

For an immutable corrected rerun, either comparison launcher accepts
`EXPERIMENT_NAME_OVERRIDE` only when exactly one arm is selected. This changes
the run directory/W&B name, not the arm configuration; multi-arm use is rejected
to prevent two cells from colliding.

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
| `COMM_EFF_SPECTRAL_TARGET_SCOPE` | `spectral.target_scope` | `all_floating` | `decoder_matrices` preserves the 196 substring-selected 2-D tensors. `all_floating` covers every de-duplicated floating parameter with a gradient, including embeddings, norms, biases, and an untied LM head. |
| `COMM_EFF_SPECTRAL_DIAGNOSTICS` | `spectral.diagnostics` | `false` | Per-target diagnostic metrics are disabled; coverage/counter/integrity guards remain active without emitting hundreds of per-tensor values. |
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
| `COMM_EFF_RANK1_PROJECTION_PROBE_ENABLED` | `probe.rank1_projection_enabled` | `true` | Enable delayed sampled-weight verification for `rank1_relex`. |
| `COMM_EFF_RANK1_PROJECTION_PROBE_SAMPLES` | `probe.rank1_projection_samples` | `16` | Deterministic scalar samples per representative tensor (`1..64`). |
| `COMM_EFF_PROBE_OUT_DIR` | `probe.out_dir` | `<run>/rank1_projection_probe` | Directory for `rank1_projection_samples.jsonl`. |
| `COMM_EFF_PROBE_RANK0_ONLY` | `probe.rank0_only` | `true` | Write JSON/stdout on rank 0 only. |

## Common invocations

Run the latest completed reference explicitly:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh composite
```

Run both qboot arms only when a matrix is intended:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh
```

Dense control on the same Qwen-Math/MATH surface:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_comparison_fsdp.sh dense
```

The earlier (legacy) comparison launcher is retained for explicit historical
reproduction, not as a current default. For example:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_comparison_fsdp.sh fixed_linear
```
