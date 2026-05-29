# Project north-star — the big goal

> The single authoritative statement of what this project is trying to achieve
> and what "done" means. Every plan, verdict, and PR is checked against it.
> Agents may read it freely; the operator keeps it current.

## The goal

**A communication-efficient, pipeline-parallel verl GRPO trainer.**

Train Qwen2.5-1.5B-Instruct on GSM8K where the **training** path (the
forward/backward activation + gradient traffic across pipeline-parallel stage
boundaries) runs under a communication-efficient method: activation masking at
the stage boundaries, with optional anchor + spectral correction. Communication
efficiency is about the **inter-stage traffic during training** — **rollouts
(generation) may come from ordinary, non-pipeline-parallel verl + vLLM**, that
is fine and out of scope for compression.

With the method switched off, training is byte-identical to unmodified verl.

## "Done" means

1. **Stable** — the method ENABLED trains end-to-end with no grad_norm
   explosion, NaN, or divergence.
2. **Parity** — final GSM8K reward/accuracy ≥ the dense control (= method OFF),
   within noise.
3. **Savings** — inter-stage communication volume is measured and is materially
   lower than dense, reported as a concrete number.
4. **Reproducible** — one canonical launcher under `examples/grpo_trainer/`
   reproduces it.

## Where we are

- **Dense control (method OFF) — proven.** Byte-identical to unmodified verl;
  learns cleanly on GSM8K in a short run. This is the bar to match.
- **Method implementation — correct.** Masking, anchor, spectral and the FSDP
  integration are wired and unit-tested; OFF ⇒ dense parity.
- **Masking — under test.** Plain masked GRPO does not yet learn at high mask
  rates. The open question is whether, and at what mask rate, it learns; the
  next step is a mask-rate sweep. Anchor + spectral correction are layered
  fixes brought in only if masking alone is not enough — **default OFF, kept
  OFF to start.**

## Why code changes are in scope

The method lives **in the verl source of this fork** (mask / anchor / spectral /
FSDP integration — see `CODE_WALKTHROUGH.md`). Reaching a stable run requires
patching that source, so code-change experiments on `exp/<N>-<slug>` branches
are expected; diagnostic-only issues stay `code_change:false`.

## Fixed control variables (do not change without separate justification)

- **Model** — Qwen2.5-1.5B-Instruct.
- **RL loss** — vanilla GRPO (not DAPO / GSPO), no-KL no-entropy.
- **Dataset** — GSM8K.
- **Hardware** — multi-GPU only, 4 ≤ num_gpus ≤ 8, Vast.ai H100/H200 via the
  locked `verl-research-vllm020` template.

## Pointers

- Engineering map of the method → `CODE_WALKTHROUGH.md`
- Authoritative operating config → `.claude/project.yaml`
- Comm-eff launcher (baseline = run it with `COMM_EFF_ENABLED=false`) →
  `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- Dense control launcher → `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
