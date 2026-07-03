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
codec**. The full result + why + what's next live in `research/runs/SUMMARY.md`
(the single source of truth — not restated here). In brief:

- **Dense control (method OFF)** — proven, byte-identical to verl; the bar to match.
- **Settled substrate** — PowerSGD r=77 + a mandatory **anchor**: a
  continuously-maintained, stale, full-coverage, DP-reduced gradient `M` that is
  **the only thing that updates the projection basis `Q`** (`anchor.owns_q`; the
  fast compressed circuit is a read-only `Q` consumer). This **replaces** the old
  unrealistic `clean_cadence` periodic-dense-step. The substrate is mechanically
  **proven by the paired-replay path**. Judge on **val/score, not grad_norm**. Do
  not relitigate the substrate.
- **The merger is the EMA family** — `signed_ema` (α=0.25, β_anc=0.50). At LOW
  anchor latency (cadence/delay_K = 5/5) the comm-eff run reaches **parity with
  dense at ~5% gradient-comm cost** ⇒ Goals 1–3 (stable / parity / savings) are
  met at low latency. Goal 4 (one canonical launcher) is open.
- **The baseline is the PROBLEM STATE.** It runs at HIGH anchor latency
  (cadence/delay_K = 20/20), where the method **collapses** (the k-collapse). That
  is intentional: the baseline sits in the regime the two priorities must fix.
- **Async-realism constraint (drives the levers)** — the real target is a single
  **SLOW** anchor node serving a fast **SWARM** over the network ⇒ the anchor is
  **always lagging, never leads**. Admissible levers use it as a *lagging*
  reference, tolerate **variable staleness**, and stay **cross-rank-identical**.
  (⇒ no delay-compensation / anchor-lead.) The **two-circuit** structure is
  mandatory — it is the practical-future-use point.

### Current priorities (2026-06-25) — the only things in active scope

The base is a working comm-eff trainer at parity; the two open fronts are both about the
**anchor ↔ fast-circuit coupling**:

1. **Solve the anchor-staleness failure at high latency** (milestone M4). At high anchor
   latency the method loses parity with dense (the observed "k-collapse"). *Why* the stale
   anchor degrades the update, and *what* restores parity, are open questions to settle
   empirically — this north-star does **not** prescribe a mechanism or a fix. Weight projection
   is the current candidate direction under investigation, gated by a GPU-free offline test on
   the shared dense weight trajectory before any GPU commitment. Summary published on the
   cloud-fare site (`github.com/shamanez/cloud-fare`): `anchor-delay/`.
2. **Reduce the compression-induced train–inference mismatch** (milestone M6). The codec's
   forward-pass distortion ("Gap A") is a bounded ~0.04 tax GRPO absorbs; shrink it (the truncated-IS
   corrector is available but unused). Summary published on the cloud-fare site
   (`github.com/shamanez/cloud-fare`): `train-inference-mis-match/`.

**Basic setup / operating base for both:** the **EMA merger** — `signed_ema` (α=0.25, β_anc=0.50) —
on the **fast 1K surface** (resp 1024, dynamic-bsz, rollout TP=1, gpu_mem 0.55, 50 steps) at HIGH
anchor latency (cadence/delay_K = 20/20, the k-collapse regime), on the locked PowerSGD r=77 anchor
substrate. Exact values: `runs/FIXED_CONTROL_SURFACE.md`.

## Why the anchor (the motivating logic)

A compressed/masked gradient is **biased + noisy**; the decisive earlier finding was
that **periodically passing a full dense gradient re-anchors training and recovers
dense-comparable results** — so the signal is recoverable, not lost. But a periodic
full-rank clean step is **not communication-efficient** (full-H transfer) and, on a
real decentralized-PP link, would itself be stale. The anchor circuit is the realistic
realization of that idea: a **low-frequency, stale, full-gradient reference**
maintained continuously and folded into the fast compressed gradient — and it also
owns the projection basis `Q`. The operating merger is the EMA-family `signed_ema`.
The open questions are now the **two priorities above**: correcting for the stale
anchor to restore parity at high latency, and reducing the compression-induced mismatch.
See `SUMMARY.md`.

## Why code changes are in scope

The method lives **in the verl source of this fork** (mask / anchor / spectral /
FSDP integration — see `CODE_WALKTHROUGH.md`). Reaching a stable run requires
patching that source, so code-change experiments on `exp/<N>-<slug>` branches
are expected; diagnostic-only issues stay `code_change:false`.

## Fixed control variables (do not change without separate justification)

- **Model** — Qwen2.5-1.5B-Instruct.
- **RL loss** — vanilla GRPO (not DAPO / GSPO), no-KL no-entropy.
- **Dataset** — EASY = GSM8K (the default); HARD = Big-Math
  (`gshasiri/Big-Math-RL-Verified-filtered`) at `MAX_RESPONSE_LENGTH=4096`.
  Registry: `.claude/project.yaml` `datasets:`.
- **Hardware** — default 1×H200 (ladder 1×H200 → 1×B200 → 2×H200, machine
  reliability >0.99) on Vast.ai via the locked `verl-research-vllm020`
  template; 1–8 GPUs supported; legacy 4×H200/8×H100 for explicit operator
  request only. See `.claude/project.yaml` `default_compute`.

## Pointers

- Durable run record + result + why → `research/runs/SUMMARY.md`
- Engineering map of the method → `CODE_WALKTHROUGH.md`
- Authoritative operating config → `.claude/project.yaml`
- Comm-eff launcher (dense control = run it with `COMM_EFF_ENABLED=false`) →
  `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- Dense control launcher → `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
