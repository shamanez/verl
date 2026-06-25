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
  is mechanically **proven by the paired-replay path** (EXP-29 infra +
  EXP-30 canary/relevance/geometry gates). Only post-paired-replay valid-M evidence should
  be used for anchor-circuit claims. Judge on **val/score, not grad_norm**. Do
  not relitigate the substrate.
- **Settled result (the current SOTA)** — the merger question is answered: the **`delayed_ef`
  merger (B2)** — error-feedback on the PowerSGD codec residual, `G_corr = G_comp + λ·δ`, λ=1, β_anc=0 —
  reaches **val@50 ≈ 0.74–0.75 = PARITY with dense at ~5% gradient-comm cost** (EXP-30, PASS). This is
  the comm-eff SOTA; its exact settings are `runs/EXP-31/B2_baseline/resolved_params_B2.txt`. **Goals 1–3 (stable /
  parity / savings) are met; Goal 4 (one canonical launcher) is pending a surpass.**
- **Closed frontier axes** — the anchor-usage and β_anc
  sweeps are closed. Perturbation, δ-momentum, adaptive dose, control-variate
  gating, sub-basis amplification, and mild β averaging all failed to beat B2
  beyond eval noise. B2 remains the reference; see `research/runs/SUMMARY.md`.
- **Async-realism constraint (drives the levers)** — the substrate's fixed `delay_K=5` lock-step is a
  *simulation*; the real target is a single **SLOW** anchor node serving a fast **SWARM** over the
  network ⇒ the anchor is **always lagging, never leads**. Admissible levers use it as a *lagging*
  reference, tolerate **variable staleness**, and stay **cross-rank-identical**. (⇒ no delay-compensation
  / anchor-lead.) The **two-circuit** structure is mandatory — it is the practical-future-use point.

### Current priorities (2026-06-25) — the only things in active scope

The base is a working comm-eff trainer at parity; the two open fronts are both about the
**anchor ↔ fast-circuit coupling**:

1. **Solve the k-collapse by projecting the weights** (milestone M4). The stale anchor gradient
   rotates to orthogonal by k≈10–20 (GSM8K cos 0.51→0.18@k5→0.02@k10→−0.01@k20; norm ratio ≈1.0 ⇒
   *pure rotation*, magnitude intact). Fix = **extrapolate the anchor's _weights_ forward** (Nesterov-style — the gradient is computed at the
   look-ahead weights θ̂≈θ_t, *not* a patched gradient), via a **learned per-block weight-projection**
   **supervised by the fast circuit's synced weights** (the residual θ_t−θ̂ trains the projector online,
   beating AsyncPP's fixed-linear rule, arXiv:2505.01099). Gated by a GPU-free offline kill-test
   (weight-prediction → does g(θ̂) recover cos@k5 ≥0.40, off-diagonal). Summary:
   `reports/priority-1-anchor-staleness-k-collapse.html`.
2. **Reduce the compression-induced train–inference mismatch** (milestone M6). The codec's
   forward-pass distortion ("Gap A") is a bounded ~0.04 tax GRPO absorbs; shrink it (the truncated-IS
   corrector is available but unused). Summary:
   `reports/priority-2-compression-train-inference-mismatch.html`.

**Basic setup / operating base for both:** the **EMA merger** — `signed_ema` (α=0.25, β_anc=0.50) —
on the **2K accel surface** (resp 2048, dynamic-bsz, rollout TP=1, gpu_mem 0.55, 50 steps), on the
locked PowerSGD r=77 anchor substrate. Exact values: `runs/FIXED_CONTROL_SURFACE.md`.

## Why the anchor (the motivating logic)

A compressed/masked gradient is **biased + noisy**; the decisive earlier finding was
that **periodically passing a full dense gradient re-anchors training and recovers
dense-comparable results** — so the signal is recoverable, not lost. But a periodic
full-rank clean step is **not communication-efficient** (full-H transfer) and, on a
real decentralized-PP link, would itself be stale. The anchor circuit is the realistic
realization of that idea: a **low-frequency, stale, full-gradient reference**
maintained continuously and folded into the fast compressed gradient — and it also
owns the projection basis `Q`. The merger that converts the anchor into a **dense-matching** update is
settled — `delayed_ef` (error-feedback on the codec residual), and the current operating merger is the
EMA-family `signed_ema`. The open questions are now the **two priorities above**: projecting the stale
anchor forward to fix the k-collapse and reducing the compression-induced mismatch. See
`SUMMARY.md`.

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
