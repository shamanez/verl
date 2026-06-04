# Fixed Control Surface — GSM8K comm-eff experiments

**Status: LOCKED (operator directive, 2026-06-04).** Every experiment in this
project holds these constant. The **only** axis that may vary between arms is the
**codec** (the line marked ☆). Changing anything else requires a separate,
explicit justification — same bar as the model/loss/hardware controls in
`CLAUDE.md §1`. This file extends those three with the *training* hyperparameters.

Why this exists: small RL deltas (the EXP-20 arms differ by ±0.005 val-acc) are
only interpretable if the rest of the run is byte-for-byte comparable. Silent
drift in batch size / cadence / steps between experiments destroys that. Pin it
once, here, and read it off for every launch.

## The control surface (held constant)

| Dimension | Value | Notes |
|---|---|---|
| **Model** | `Qwen/Qwen2.5-1.5B-Instruct` | dense control anchored to it; **H (hidden) = 1536** |
| **Data** | GSM8K (`openai/gsm8k`) | `train.parquet` / `test.parquet`; test set = 1319 |
| **RL objective** | vanilla GRPO, **no-KL, no-entropy** | `use_kl_loss=False`, `use_kl_in_reward=False`, `entropy_coeff=0`, `adv_estimator=grpo` |
| **Learning rate** | `1e-6` | AdamW (verl default betas) |
| **train_batch_size** | **128** prompts | ×`rollout.n`=8 ⇒ 1024 sequences/step |
| **ppo_mini_batch_size** | **64** | |
| **ppo_micro_batch_size_per_gpu** | **1** | static batching (`use_dynamic_bsz=False`) for trackability |
| **rollout.n** | **8** | rollouts per prompt |
| **rollout.tensor_model_parallel_size** | **2** | vLLM TP |
| **rollout.gpu_memory_utilization** | **0.4** | |
| **max_prompt_length** | **1024** | |
| **max_response_length** | **16384** | the 16K-response headroom that forces multi-GPU |
| **ppo_max_token_len_per_gpu** | **36864** | (+ log_prob / ref variants) |
| **total_training_steps** | **50** | the standard short-run horizon |
| **total_epochs** | **2** | a ceiling; 50 steps is reached inside epoch 1 (≈59 steps/epoch) |
| **clean_cadence** (comm-eff arms only) | **5** | ⇒ 10 clean/dense steps {5,10,…,50} + 40 compressed; inert when comm-eff OFF |
| **save_freq** | **50** | |
| **val_before_train** | **True** | the step-0 val point |
| **calculate_log_probs** | **True** | train-inference mismatch diagnostic; rollout CORRECTION stays OFF (old_log_prob always recomputed) |
| **Hardware** | 4×H200 (pref) or 8×H100 | Vast `verl-research-vllm020`; `max_dph=24` |
| **seeds** | comm_eff `seed=0` (mask + powersgd) | |

## ☆ The ONLY axis that varies — the codec

| arm | `comm_eff.enabled` | `compression_type` | knob | logical bytes/tok |
|---|---|---|---|---|
| **dense control** | `false` | (n/a) | — | H (uncompressed) |
| **PRF mask** | `true` | `prf_mask` | `mask.p` (e.g. 0.95) | (1−p)·H |
| **PowerSGD** | `true` | `powersgd` | `powersgd.rank` r (e.g. 77, 102) | r |

Budget note (H=1536): the mask at p=0.95 keeps `0.05·1536 = 76.8` coords/token, so
**r=77 is the byte-matched PowerSGD arm** (the equal-budget comparison); r=102 is
+33% budget. The dense control sends the full activation (no compression) and is
the learning ceiling / reference trajectory.

## Measurement knobs (NOT control variables — may vary freely)

These don't touch the trained model (validation is a separate, read-only pass), so
they can differ between runs without breaking comparability:

- **`test_freq`** — validation cadence. **Canonical = 10** going forward (val at
  0/10/20/30/40/50 on a 50-step run) so the early trajectory — in particular the
  *post-step-10* region the central research question hinges on — is resolved.
  The three EXP-20 arms predate this and ran at `test_freq=25` (val@0/25/50);
  their finer-grained signal is the per-step **train** reward (`critic/score/mean`,
  logged every step regardless of `test_freq`). The dense control `ce_dense_50s_gsm8k`
  runs at `test_freq=10`.

## How to launch on this surface

One canonical launcher, override only the codec axis (+ experiment_name):

```bash
# dense control (this run):
COMM_EFF_ENABLED=false TOTAL_TRAINING_STEPS=50 TEST_FREQ=10 \
  EXPERIMENT_NAME=ce_dense_50s_gsm8k \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# PowerSGD r=77 (byte-matched):
COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_POWERSGD_RANK=77 \
  COMM_EFF_CLEAN_CADENCE=5 TOTAL_TRAINING_STEPS=50 TEST_FREQ=10 \
  EXPERIMENT_NAME=ce_powersgd_r77_clean5_50s_gsm8k \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# PRF mask p=0.95:
COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=prf_mask COMM_EFF_MASK_P=0.95 \
  COMM_EFF_CLEAN_CADENCE=5 TOTAL_TRAINING_STEPS=50 TEST_FREQ=10 \
  EXPERIMENT_NAME=ce_mask_p95_clean5_50s_gsm8k \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

The launcher's `${VAR:-default}` defaults already encode the rest of this surface
(batch sizes, lr, rollout shape, contexts, objective). The TWO defaults that do
NOT match (and so must be passed every time) are `TOTAL_TRAINING_STEPS` (default
100 → 50) and `TEST_FREQ` (default 25 → 10).

See also: `CLAUDE.md §1` (model/loss/hardware controls), `examples/grpo_trainer/VAST_README.md`
(launcher stability contract), `research/.claude/project.yaml` (`default_compute`, provisioning).
