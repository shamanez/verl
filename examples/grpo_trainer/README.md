# Communication-efficient GRPO launchers (Qwen2.5-Math-1.5B / MATH)

This directory holds the launchers for **this fork's** communication-efficient
GRPO baseline. The fixed surface is **`Qwen/Qwen2.5-Math-1.5B` on MATH**
(last-`\boxed{}` + `is_equiv` reward). With the method disabled
(`COMM_EFF_ENABLED=false`) training is byte-identical to upstream verl.

> This is not the upstream verl example set. The generic per-model demo scripts
> (`run_qwen3_*`, `run_deepseek_*`, vision/VL, etc.) and the old
> Qwen2.5-1.5B-Instruct / GSM8K launchers were removed on 2026-07-15. The
> project-level source of truth is
> [`research/.claude/project.yaml`](../../research/.claude/project.yaml)
> (`compression_defaults.math_qwen25_math_1p5b`); what "done" means lives in
> [`research/.claude/GOAL.md`](../../research/.claude/GOAL.md).

## What GRPO is (in one paragraph)

GRPO drops PPO's critic: for each prompt it samples a group of `n` completions,
scores each against the reward, and uses the group mean as the baseline, so
better-than-average completions are reinforced and worse-than-average ones are
discouraged. This fork runs vanilla GRPO (no DAPO / GSPO), no-KL no-entropy on
the method path. Reference: [DeepSeekMath](https://arxiv.org/pdf/2402.03300).

## Files

| File | Role |
|---|---|
| `vast_comm_eff_engine_grpo.sh` | The shared, **model/dataset-agnostic** comm-eff engine. Every launcher below `exec`s it; `MODEL_PATH` / `DATA_DIR` / `EXPERIMENT_NAME` are supplied by the caller. Dense control = run it with `COMM_EFF_ENABLED=false`. |
| `run_qwen25_math_1p5b_rank1_relex_fsdp.sh` | The MATH **method/base launcher**: pins the Qwen2.5-Math-1.5B / MATH surface (prompt/response 1024/3072, batch 512, mini 256, `n=8`, AdamW `1e-6`, `kl=0.001`) and the RELEX chat template, then execs the engine. |
| `run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh` | The **comparison matrix** launcher (`matrix_launcher` in project.yaml). Runs named arms; a bare invocation runs the matrix. |
| `run_qwen25_math_1p5b_relex_comparison_fsdp.sh` | The **legacy matrix** launcher and the **dense control** arm. |
| `relex_qwen_chat_template.jinja` | RELEX Qwen chat template, loaded by the base launcher for rollout + validation parity. |
| `COMM_EFF_CONFIG.md` | Every `COMM_EFF_*` env var and its hydra mapping. |
| `VAST_README.md` | Vast.ai launch conventions + the launch-script stability contract. |

## Running it

The MATH launchers require a prepared MATH dataset in `DATA_DIR` (they FATAL if
`train.parquet` / `test.parquet` are absent). The canonical prep is
[`research/scripts/prepare_rlvr_math.py`](../../research/scripts/prepare_rlvr_math.py);
a bare engine run falls back to `examples/data_preprocess/math_dataset.py`.

```bash
# Comm-eff method run (the reference surface):
bash examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh

# Dense control (comm-eff OFF, same surface, the parity bar):
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_comparison_fsdp.sh dense

# Latest completed comm-eff reference (explicit arm required):
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh composite
```

Common knobs are exposed as environment variables at the top of each launcher
(`MODEL_PATH`, `DATA_DIR`, `ROLLOUT_N`, `TRAIN_BATCH_SIZE`, `MAX_RESPONSE_LENGTH`,
the `COMM_EFF_*` family, …). Hardware provisioning and the launch flow are driven
by the harness under `research/`; see
[`research/researcher_steps.md`](../../research/researcher_steps.md).

## Reference

- Method implementation map: [`CODE_WALKTHROUGH.md`](../../CODE_WALKTHROUGH.md).
- verl baselines: <https://verl.readthedocs.io/en/latest/algo/baseline.html>.
