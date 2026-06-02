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

The base is **settled**. Operational detail, the proven result + why, and knobs →
`research/runs/SUMMARY.md`; next-cycle plan → `research/findings/NEXT_RESEARCH.md`.
In brief:

- **Dense control (method OFF)** — proven, byte-identical to verl; the bar to match.
- **Settled comm-eff base** = mask (p=0.9, per-(token,channel), 7 boundaries) **+
  rescale (ON, permanent — its job is unbias, not a learning fix) + a true dense
  clean gradient every K steps (`clean_cadence`)**. Mask cross-pass consistency is
  solved; judge on **val/score, not grad_norm**. Do not relitigate these.
- **Proven result** — masked+clean@K is **stable** (clean-resettable sawtooth, no
  ratchet) and reaches **GSM8K dense parity** (EXP-17: 0.735 vs 0.741) — but that is
  **elicitation** (base already 0.715). On **Big-Math** (base 0.48) it **stalls flat
  ~0.55** while dense reaches ~0.61: a gradient-fidelity limit, not a missing ceiling.
- **Anchor + spectral as implemented did NOT work** (GSM8K 0.080, inert) — fails by
  **orthogonality** (reweights the masked gradient in a subspace instead of applying
  the true gradient). The clean step is the only lever that worked.
- **Frontier** — redesign anchor + spectral as a **cheap, continuous surrogate** for
  the periodic clean step, grounded in the delta-method curvature bias (not
  anchor-gradient-SVD). Gated by a **p-sweep** and a **clean-only ablation**.

## The moment of truth

Masking alone makes a **biased, noisy** gradient that does not learn at high mask
rates. The decisive finding: **periodically passing a full dense gradient every K
steps re-anchors training and recovers results comparable to dense** (GSM8K parity).
A sparse periodic correction pulls the drifting masked trajectory back — so **the
signal is recoverable, not lost.** The opening: if a *sparse* full gradient works, a
**cheaper, continuous spectral correction** (driven by an anchor — a low-frequency
true-gradient reference) should recover the same at a fraction of the communication.
That is the direction worth pursuing.

Honest bar: GSM8K parity is *elicitation* (the base model is already strong there) —
on genuinely harder data (Big-Math) even the periodic full gradient stalls, so the
correction's real job is to recover *learning*, not just match the clean step; and
the first anchor+spectral attempt failed by **orthogonality** (it reweighted the
masked gradient instead of supplying the missing true-gradient component), so the
redesign must target the bias directly. The core logic still stands: periodic true
gradients work → a continuous correction should too.

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
- Next-cycle plan → `research/findings/NEXT_RESEARCH.md`
- Engineering map of the method → `CODE_WALKTHROUGH.md`
- Authoritative operating config → `.claude/project.yaml`
- Comm-eff launcher (dense control = run it with `COMM_EFF_ENABLED=false`) →
  `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- Dense control launcher → `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
