# Project north-star — the big goal

> The single authoritative statement of what this project is trying to achieve
> and what "done" means. Every plan, verdict, and PR is checked against it.
> Agents may read it freely; the operator keeps it current.

## The goal

**A communication-efficient, pipeline-parallel verl GRPO trainer.**

Train Qwen2.5-1.5B-Instruct on GSM8K where the **training** path (the
forward/backward activation + gradient traffic across pipeline-parallel stage
boundaries) runs under a communication-efficient method: per-element (per-token,
per-dimension) activation masking at the stage boundaries, with optional anchor +
spectral correction. Communication
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

The comm-eff base is **settled and realistic**: the **anchor circuit on a PowerSGD
codec**. The full result + why + what's next, and all the numbers, live in
`research/runs/SUMMARY.md` (the single source of truth — not restated here). In brief:

- **Dense control (method OFF)** — proven, byte-identical to verl; the bar to match.
- **Settled comm-eff base** — PowerSGD r=77 + a mandatory **anchor**: a
  continuously-maintained, `delay_K=5`-stale, full-coverage, DP-reduced gradient EMA
  `M` that is **the only thing that updates the projection basis `Q`**
  (`anchor.owns_q`; the fast compressed circuit is a read-only `Q` consumer). This
  **replaces** the old unrealistic `clean_cadence` periodic-dense-step. The substrate
  is mechanically **proven** (EXP-25 R1+R2 probe gates green). Judge on **val/score,
  not grad_norm**. Do not relitigate the substrate.
- **Honest result** — the substrate is realistic + correct, but the current
  `signed_ema` **merger** (how `M` corrects the fast gradient) is **falsified**: it
  does not match plain PowerSGD, so we do **not** beat dense (EXP-25, STOP). This is a
  good base to *start from*, not a finished result.
- **Frontier** — the single open axis is the **gradient-correction merger primitive**.
  `signed_ema` (sign-replacement) is the wrong primitive; the next candidate is
  **error-feedback on the PowerSGD residual** (issue #24). This is RL — no recipe to
  copy — so the correction is found **empirically**: propose a merger, run it, compare
  the training curve to dense, refine. Target: match the dense curve within ≤50 steps.

## Why the anchor (the motivating logic)

A compressed/masked gradient is **biased + noisy**; the decisive earlier finding was
that **periodically passing a full dense gradient re-anchors training and recovers
dense-comparable results** — so the signal is recoverable, not lost. But a periodic
full-rank clean step is **not communication-efficient** (full-H transfer) and, on a
real decentralized-PP link, would itself be stale. The anchor circuit is the realistic
realization of that idea: a **low-frequency, stale, full-gradient reference**
maintained continuously and folded into the fast compressed gradient — and it also
owns the projection basis `Q`. The open question is no longer *whether* a correction
helps but **which merger** converts the anchor into a dense-matching update;
sign-replacement (`signed_ema`) does not (see `SUMMARY.md`).

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

- Durable run record + result + why → `research/runs/SUMMARY.md`
- Engineering map of the method → `CODE_WALKTHROUGH.md`
- Authoritative operating config → `.claude/project.yaml`
- Comm-eff launcher (dense control = run it with `COMM_EFF_ENABLED=false`) →
  `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- Dense control launcher → `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
