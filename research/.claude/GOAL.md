# Project north-star — communication-efficient GRPO

> The authoritative statement of what this project is trying to achieve and what
> “done” means. Agents may read it freely; the operator keeps it current.

## The goal

Build a communication-efficient, pipeline-parallel verl GRPO trainer on the
current fixed surface:

- **Model** — `Qwen/Qwen2.5-Math-1.5B`.
- **Data** — native MATH train/test from `EleutherAI/hendrycks_math`, with
  `DigitalLearningGmbH/MATH-lighteval` reward routing (last `\boxed{}` answer plus
  `is_equiv`).
- **Training traffic** — project activations at logical pipeline boundaries and
  use a delayed dense anchor to stabilize the fast compressed circuit. Rollout
  generation may remain ordinary non-pipeline-parallel verl + vLLM.

With communication efficiency disabled, the actor path remains the dense verl
control.

## “Done” means

1. **Stable** — the enabled method trains end to end without NaN, divergence, or
   transaction-integrity failures.
2. **Parity** — final MATH-test accuracy matches the dense control within noise on
   the exact fixed surface.
3. **Savings** — inter-stage communication is measured and materially below
   dense, with the ratio reported.
4. **Objective-correct** — the dense anchor differentiates the same resolved actor
   objective as the fast circuit under its declared ratio-one surrogate.
5. **Reproducible** — a single-arm launcher reproduces the selected method; no arm
   is promoted from incomplete or diagnostic-only evidence.

## Current default answer/reference surface

When asked for “the default setting,” answer with this qboot-v2 surface:

- GRPO train batch 512, PPO mini-batch 256, rollout `n=8`, dynamic micro-batch
  1/GPU, actor shuffle off, prompt/response 1024/3072.
- AdamW learning rate `1e-6`, weight decay `0.01`, gradient clip `1`, one PPO
  epoch, token-mean loss aggregation.
- Actor reference KL enabled (`low_var_kl`, coefficient `0.001`); reward-side KL
  off; entropy coefficient `0`.
- 100 trainer steps; greedy validation at 0/25/50/75/100; the completed reference
  ran on 2×H200 NVL.

The exact machine-readable values and evidence gate are in
`.claude/project.yaml` under `compression_defaults.math_qwen25_math_1p5b`.

## Communication-efficient method

### Activation projection

Boundary activation matrices are projected with PowerSGD rank 77 using an
activation-derived basis `Q`: conceptually `X_hat = (X Q) Q^T`. The basis is
synchronized across data-parallel ranks and warm-started. On Qwen2.5-Math-1.5B
(`hidden_size=1536`), rank 77 is a 77/1536 = 5.013% sketch; the completed
reference measured communication ratio `0.05013569`.

The anchor owns `Q`. The fast circuit is a read-only consumer, so it cannot drift
the basis independently. qboot-v2 installs one consensus fast-Q observation
before the first compressed old/current-policy pair.

### Anchor signal

The anchor performs paired dense replay on a delayed exact checkpoint and builds
the full, DP-reduced gradient signal `M`. It runs at cadence/delay 20/20 optimizer
ticks; with two optimizer ticks per trainer step, an anchor transaction occurs
every 10 global steps. The reference scope is one 256-prompt PPO mini-batch
(2,048 response sequences at `n=8`). Replay state and signed-EMA state live on
CPU.

`M` both refreshes anchor-owned `Q` and corrects the fast gradient through
`signed_ema` (`alpha=0.25`, `beta_anc=0.50`). qboot-v2 uses `all_floating`
coverage: 338 unique tensors, diagnostics off, `max_targets=-1`.

The anchor objective uses a ratio-one, unclipped advantage policy gradient while
matching the fast actor’s advantages, masks, normalization, rollout weights,
entropy term, and reference-KL type/coefficient. Compressed-policy
`old_log_probs`, PPO ratio, and PPO clipping are the intentional exceptions;
unsupported or missing terms fail closed.

### RELEX weight projection

RELEX is separate from activation projection: it forecasts the delayed anchor’s
weights per tensor. The latest completed qboot-v2 composite retains four exact
checkpoints with `min_snapshots=2`, applies a W2 secant at step 20, W3 rank-1 fit
at step 30, and sliding W4 rank-1 OLS from step 40. It uses strength 1,
`stale_correct` first-fire behavior, and current trajectories on projected fires.

## Scientific status

The qboot-v2 composite (`v9sfxnaz`, commit `8bad0656`) is the **latest completed
reference**, ending at 66.85% MATH with ~5.01% communication. It predates the
anchor-KL objective-parity fix and is diagnostic evidence, not a promoted
champion. The current status is:

- the default **model/data/run surface** is Qwen2.5-Math-1.5B + MATH train/test;
- the latest completed **method reference** is qboot-v2 composite;
- the qboot-v2 settings above are the active single-arm launcher defaults;
- the reference remains diagnostic until an objective-parity-complete run
  finishes, so it must not be described as a statistically established champion.

Generic actor/Hydra communication defaults remain all-off for dense
compatibility; the MATH method launcher enables the qboot-v2 surface explicitly.

## Current priority

Complete an objective-parity run through all anchor/RELEX phases, then compare it
with the dense control and replicate before promotion. Score, full trajectory,
mismatch KL, communication ratio, Q transaction counters, M coverage, and RELEX
readiness/correction metrics all belong in the verdict.

## Pointers

- Operating configuration: `.claude/project.yaml`
- Latest completed reference: `bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh composite`
- Dense MATH control: `bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_comparison_fsdp.sh dense`
