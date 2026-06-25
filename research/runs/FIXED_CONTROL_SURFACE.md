# Fixed Control Surface — GSM8K comm-eff baseline

**Status: LOCKED.** The baseline is the fast 1K comm-eff loop via
`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`:
resp **1024**, dynamic-bsz, rollout TP=1, gpu_mem 0.55, 50 steps, val@25/50,
`signed_ema` (α=0.25, β_anc=0.50) merger, diagnostics off, on the locked
**PowerSGD r=77 anchor substrate** — run at **high anchor latency
(cadence/delay_K = 20/20)**, which is the **k-collapse regime** the baseline
deliberately sits in (the problem Priority 1 targets).

A bare run of the launcher reproduces this exactly. The only axis a test may vary
is the **merger** (the ☆ section); anything else needs separate justification
(same bar as `CLAUDE.md §1`). Values live in the launcher, not duplicated here.

Why this exists: small RL deltas differ by only a few thousandths of val-acc and
are interpretable only if the rest of the run is byte-for-byte comparable. Pin it
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
| **use_dynamic_bsz** | **True** | token-balanced dynamic batching |
| **rollout.n** | **8** | rollouts per prompt |
| **rollout.tensor_model_parallel_size** | **1** | vLLM TP |
| **rollout.gpu_memory_utilization** | **0.55** | |
| **max_prompt_length** | **1024** | |
| **max_response_length** | **1024** | fast 1K surface — short responses for quick turnaround |
| **ppo_max_token_len_per_gpu** | **24576** actor | actor budget under dynamic-bsz |
| **total_training_steps** | **50** | val at 0/25/50; push to **100** to manifest the collapse (~step 61) |
| **total_epochs** | **2+** | a ceiling sized to reach the step target |
| **save_freq** | **50** | |
| **val_before_train** | **False** | skip step-0 val (untrained base is constant) |
| **calculate_log_probs** | **True** | train-inference mismatch diagnostic; rollout CORRECTION stays OFF (old_log_prob recomputed) |
| **Hardware** | 4×H200 (pref) or 8×H100 | Vast `verl-research-vllm020`; `max_dph=24` |
| **seeds** | comm_eff `seed=0` | |

## ☆ The locked substrate + the variable axis (the merger)

The comm-eff base is the **anchor circuit on a PowerSGD codec**, locked. The
single axis a test may vary is the **merger** (how the anchor `M` corrects the
fast gradient). Exact values are the launcher `${VAR:-default}`; the ground
truth of any run is its `resolved_params.txt`.

| knob | value | note |
|---|---|---|
| `compression_type` | `powersgd` | the locked codec (only one compatible with anchor-owns-`Q`) |
| `powersgd.rank` | `77` | byte-matched to mask p=0.95 (H=1536: `0.05·1536 ≈ 77`) |
| `anchor.enabled` | `true` | MANDATORY — the stale full-grad reference `M` |
| `anchor.owns_q` | `true` | the anchor is the ONLY thing that updates `Q` |
| `anchor.cadence` / `delay_K` | `20` / `20` | refresh + staleness, in optimizer ticks — **the k-collapse regime** |
| `clean_cadence` | `0` | DEAD — the anchor replaced the periodic dense step |
| `spectral.correction_mode` | `signed_ema` | core merger — α=0.25, `β_anc=0.50` |
| `spectral.diagnostics` | `false` | math-neutral speed knob — skips per-step rel_change syncs + diag prints |
| `replay_paired_batch` / `snapshot_device` | `true` / `cpu` | valid on-policy anchor `M` — part of the substrate |
| vLLM `disable_custom_all_reduce` | `true` | **required** for the box to init (CUDA-IPC under the mp executor); greedy-val-neutral |

**The variable axis — how the anchor `M` is USED.** The baseline merger is
`signed_ema` (α=0.25, β_anc=0.50). A test varies the merger and nothing else.

**Reference codec (NOT the base; ablation only):** the dense control
(`comm_eff.enabled=false`, the learning ceiling). Always compare a comm-eff cell
to a dense run sharing its code + surface. The dense ceiling is
run-variance-dominated; report it as a **band ≈ 0.75–0.78** (rollout
nondeterminism ≈ ±0.024/draw even at seed 0), not a single point.

## Measurement knobs (NOT control variables — may vary freely)

These don't touch the trained model (validation is a separate read-only pass):

- **`test_freq`** = 25 (val at 0/25/50 on a 50-step run; 0/25/50/75/100 at 100).
  The per-step **train** reward (`critic/score/mean`, logged every step) is the
  fine-grained signal between validations.

## Diagnostics policy — production runs ship with capture OFF

Diagnostic instrumentation that **holds tensors in memory** is an OOM hazard. Tiers:

1. **Scalar telemetry — always-on safe.** Norms, cosines, byte counters: log on every run.
2. **Bounded tensor captures — only when the plan's success criteria require them.**
   `capture.enabled=true` MUST come with bounds: `max_ticks` (≈10–12), `min_tick`,
   `rank0_only=true`, fp32-dump-then-free. Name the consuming criterion or keep it OFF.
3. **Extra-backward probes — NEVER on a production arm.** `capture_g_dense` /
   `capture_fresh_anchor` each add a parallel full backward + held fp32 tensors.
   Dedicated short diagnostic runs only.

Standing OOM guards on EVERY run: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`spectral.ema_device=cpu` (keeps the ~6 GB fp32 M state off-GPU), and the actor
token budget above while the anchor is on.

## How to launch

**THE canonical launcher is
`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`.** It
pins the surface + substrate + signed_ema + diagnostics off explicitly, then
execs the generic engine, so a bare run reproduces the baseline. Override only
run length / name / the one merger axis you're varying.

```bash
# comm-eff baseline — bare run, 50 steps:
EXPERIMENT_NAME=ce_baseline \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh

# dense reference on the same surface — one-knob OFF on the GENERIC launcher:
COMM_EFF_ENABLED=false USE_DYNAMIC_BSZ=True MAX_RESPONSE_LENGTH=1024 ROLLOUT_TP=1 \
  ROLLOUT_GPU_MEM_UTIL=0.55 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
  TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 VAL_BEFORE_TRAIN=False EXPERIMENT_NAME=dense_ref \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

Pass `TOTAL_TRAINING_STEPS` per launch; the rest is baked in. Ground truth of any
run is its `resolved_params.txt`.

See also: `CLAUDE.md §1` (model/loss/hardware controls),
`examples/grpo_trainer/VAST_README.md` (launcher stability contract),
`research/.claude/project.yaml` (`default_compute`, provisioning).
