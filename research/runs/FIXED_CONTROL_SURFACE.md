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
| **total_training_steps** | **50 → 100** | start at 50; extend to 100 once 50 trains cleanly |
| **total_epochs** | **2+** | a ceiling sized to reach the step target (≈59 steps/epoch) |
| **anchor refresh cadence** (realistic setting) | **5** | the anchor circuit refreshes `M` (and `Q`) every 5 steps from stale, delayed weights — this REPLACES any periodic dense clean step; do **not** assume a `clean_cadence` |
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

- **`test_freq`** — validation cadence. **= 25** (val at 0/25/50 on a 50-step run;
  0/25/50/75/100 at 100 steps). The per-step **train** reward (`critic/score/mean`,
  logged every step regardless of `test_freq`) is the fine-grained signal between
  validations.

## How to launch on this surface

One canonical launcher (`examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`);
override only the codec axis + run length:

```bash
# PowerSGD r=77 (byte-matched), 50 steps, validation every 25:
COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_POWERSGD_RANK=77 \
  TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 \
  EXPERIMENT_NAME=ce_powersgd_r77_50s_gsm8k \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# dense reference: same launch with COMM_EFF_ENABLED=false.
```

The launcher's `${VAR:-default}` defaults encode the core surface (batch sizes, lr,
rollout shape, contexts, objective). Pass `TOTAL_TRAINING_STEPS` (50, then 100) and
`TEST_FREQ=25` per launch. The realistic anchor setting (issue #25) adds the anchor
flags — stale-weight gradient refresh, anchor-owned `Q`, signed-EMA merger — and runs
**without** a periodic dense clean step.

See also: `CLAUDE.md §1` (model/loss/hardware controls), `examples/grpo_trainer/VAST_README.md`
(launcher stability contract), `research/.claude/project.yaml` (`default_compute`, provisioning).
