## EXP-3 — closed at global_step 100

**Status: PASS** on the headline criterion. Run stopped at step 100 (of ~116 planned) to conserve budget — step_100 checkpoint preserved on HF, training otherwise complete in epoch 2.

### Headline result

| Metric | step 0 | step 100 | Δ | criterion |
|---|---:|---:|---:|---|
| `val-core/openai/gsm8k/acc/mean@1` | **0.0872** | **0.7892** | **+0.7020** | ≥ +0.05 ✅ (14× the threshold) |
| `critic/score/mean` (rollout) | 0.126 | 0.874 | +0.748 | rising ✅ |
| `actor/grad_norm` (latest) | 0.365 | 0.262 | — | finite, < 5.0 ✅ |
| NaN/Inf in loss/grad/reward | — | none | — | none ✅ |

Validation ran over **all 1,319 GSM8K test samples** at steps 0, 25, 50, 75, 100 (full sweep — `val_batch_size = len(val_dataset)`, `drop_last=False`, no subsampling).

Per-eval val progression: `0.0872 → 0.7400 → 0.7672 → 0.7945 → 0.7892` (peak at step 75 then flat — diminishing returns confirmed).

### Links

- **WandB run**: https://wandb.ai/shamanework-pl/verl_compression_research/runs/wybop525
- **HF checkpoint @ step 50** (private): https://huggingface.co/gshasiri/qwen25-1p5b-grpo-gsm8k-baseline-step50
- **HF checkpoint @ step 100** (private, headline): https://huggingface.co/gshasiri/qwen25-1p5b-grpo-gsm8k-baseline-step100
- **Reproducibility manifest** (laptop ↔ remote ↔ box hashes, re-run recipe): `runs/EXP-3/REPRODUCIBILITY.md`
- **Verbatim launcher snapshot** (sha256-attested): `runs/EXP-3/launcher.snapshot.sh`

### Training cost & time (training-only, excluding container onstart)

| | value |
|---|---|
| Training start (`wandb.init()`) | 2026-05-26 07:35:21 UTC |
| Step 100 finished | 2026-05-26 08:13:48 UTC |
| **Training wall-clock** | **2,307 s = 38.45 min** (from `_runtime` at step 100) |
| Mean s/step | ~20 s (range 19.2 – 27.6) |
| MFU (actor) | 0.28–0.30 steady |
| Throughput | ~4,500–4,750 tok/s |
| Hourly rate | $16.054/hr (4×H200 tier) |
| **Training-only spend** | **~$10.29** (38.45 min × $16.054/hr) |
| Total instance spend (incl. ~9 min onstart/setup + checkpoint saves) | ~$12.76 |

### Hyperparameters — full settings used by this run (THE source of truth)

Verbatim from `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` at commit `81e4ab32` (sha256 `b6d4d594…`).

**Model**
```yaml
MODEL_PATH:           Qwen/Qwen2.5-1.5B-Instruct      # Apache-2.0, ~3 GB bf16, no HF gating
```

**Rollout (vLLM)**
```yaml
ROLLOUT_TP:                 2                          # tensor-parallel within vLLM
ROLLOUT_N:                  8                          # GRPO group size — 8 rollouts/prompt
ROLLOUT_GPU_MEM_UTIL:       0.4                        # vLLM KV-cache budget per GPU
```

**Batch shape**
```yaml
TRAIN_BATCH_SIZE:           128                        # prompts per training step (× 8 = 1024 rollouts/step)
PPO_MINI_BATCH_SIZE:        64                         # → 2 PPO inner passes per step
PPO_MICRO_BATCH_SIZE_PER_GPU:    1                     # fallback; dynamic_bsz handles real packing
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU: 1
use_dynamic_bsz:            True                       # token-budget packing instead of fixed micro-batch
PPO_MAX_TOKEN_LEN_PER_GPU:        36864                # token budget / GPU / micro
LOG_PROB_MAX_TOKEN_LEN_PER_GPU:   36864
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU: 36864
```

**Context**
```yaml
MAX_PROMPT_LENGTH:          1024
MAX_RESPONSE_LENGTH:        16384                      # user mandate — do NOT shrink
```

**GRPO objective**
```yaml
ACTOR_LR:                   1.0e-6
KL_LOSS_COEF:               0.001
ENTROPY_COEFF:              0
```

**Memory / FSDP**
```yaml
actor.fsdp_config.param_offload:     false              # actor weights stay on GPU
actor.fsdp_config.optimizer_offload: false              # Adam state stays on GPU
ref.fsdp_config.param_offload:       true               # ref model offloads (we don't backprop)
model.enable_gradient_checkpointing: true
model.use_remove_padding:            true
```

**Run schedule**
```yaml
TOTAL_EPOCHS:               2                          # planned (run stopped at step 100 ≈ epoch 1.7)
SAVE_FREQ:                  50                         # checkpoint every 50 global_steps
TEST_FREQ:                  25                         # validate every 25 global_steps
```

**Logging**
```yaml
trainer.project_name:       verl_compression_research
trainer.experiment_name:    qwen25_1p5b_grpo_gsm8k_baseline
trainer.logger:             [console, wandb]
```

**Compute (provisioned tier — chosen_tier_idx=1)**
```yaml
GPU:                        4 × H200 (141 GB HBM)
container image:            verlai/verl:vllm020.dev1
torch:                      2.11.0+cu130
vllm:                       0.20.2
vast template:              verl-research-vllm020 (hash 6485b9625ddd6d25a5f2f09b9f7fde17)
dph:                        $16.054/hr
```

**Data**
```yaml
data_source:                openai/gsm8k (HF "main" config)
train split:                7,473 prompts
test split:                 1,319 prompts (every val pass evaluates ALL of them)
instruction template:       "Let's think step by step and output the final answer after \"####\"."
reward model:               rule-based exact-match on extracted `####` answer
```

### Reproducibility certificate

The launcher used on the Vast box is **bit-identical** to the file at `origin/vast-ai-workload@81e4ab32` AND to the laptop's working tree (same blob `b600dfae…`, same sha256 `b6d4d594…`). Re-running the recipe in `runs/EXP-3/REPRODUCIBILITY.md` should reproduce the curve.

Closing this issue. Next experiment will branch from `vast-ai-workload` for the first compression cell (single-process activation masking with `MASK_RATIO=0.95`).
