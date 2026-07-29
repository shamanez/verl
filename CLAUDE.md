# Agent Instructions for this verl fork

This fork is only for the communication-efficient GRPO pipeline. Ignore model,
dataset, and launcher defaults from unrelated upstream examples.

## Deployment context (why this pipeline exists)

- Goal: RL fine-tuning (RLVR/GRPO) of models too large for one GPU. As models
  grow the model must be split with PIPELINE PARALLELISM into stages, and the
  stages live on small community GPUs connected over the ordinary internet:
  no InfiniBand, no co-located mesh.
- Because stage boundaries cross the internet, inter-stage activations
  (forward) and boundary gradients (backward) MUST be aggressively compressed.
  That compression, and keeping RL stable under it, is the research object.
- Rollout generation is not the constrained path: rollouts can come from
  quantized serving copies of the policy. The compressed path is the training
  forward/backward through the pipeline stages.
- A periodic slow sync is allowed: from time to time the system can do a dense
  weight sync or dense passes (for example in a central GPU mesh). Cadence and
  latency of that slow path are set by the network, not tunable per step. The
  anchor circuit models this.
- Cardinal rule of post-training: do not damage the base model's weights or
  capabilities. Stability and capability preservation outrank raw reward.
- Out of scope: data-parallel replica merging / federated averaging. One
  model, split into stages, is the setting.

## Current default

- Model: `Qwen/Qwen2.5-Math-1.5B`.
- Data: prepared MATH `train.parquet` and `test.parquet` from
  `EleutherAI/hendrycks_math`; default directory `$HOME/data/math`.
- GRPO: train batch 512, PPO mini-batch 256, rollout `n=8`, prompt/response
  1024/3072, AdamW `1e-6`, reference `low_var_kl=0.001`, 100 steps.
- Activations: **PRF exact-k** (`compression_type=prf_mask`, `p=0.95`,
  `exact_k=true`, `rescale_mode=constant`, masking the train forward, the
  old-logprob recompute and the reference forward). Exactly 77 of 1536
  coordinates per token, 1232 bits/token/boundary. Unbiased, and the mask is a
  PRF of seed/step/layer so nothing is transmitted and there is no side channel.
  **Changed from PowerSGD rank 77 on 2026-07-29 by issue #93:** twelve arms were
  run to beat it on stability and none did; it is the only codec with 600 steps
  of evidence that the optimizer stays in a steady state (gap slope
  +0.000848/step over 100-599, grad-norm block median flat 1.50-1.82, block max
  never above 4.645). PowerSGD rank 77 is unchanged and still tested; reach it
  with `COMM_EFF_COMPRESSION_TYPE=powersgd`.
- Anchor: paired dense replay, cadence/delay 20/20 optimizer ticks,
  `rollout_batch` (full 512p/4096s replay; `ppo_minibatch` halves it at ~no
  accuracy cost per #84), CPU replay/snapshots, **`owns_q=false`** (the plain PRF
  mask has no basis `Q` for the anchor to own, and the config validator rejects
  `owns_q=true` with `prf_mask` unless `mask.frlr=true`).
- Weights: rank-1 RELEX, fixed W2 (window 2, two-checkpoint secant; #83 val@60
  winner over progressive/W4), strength 1, `auto`, `stale_correct`.
- Gradient signal: signed-EMA `M` over all floating parameters,
  `beta_anc=0.25` (#84 best; `alpha=0.5` hurt + destabilised), `alpha=0.25`,
  CPU state.

Run it with:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh
```

Generic Hydra keeps `comm_eff.enabled=false`; the launcher enables the method.

## Scope

Keep implementation and documentation limited to PowerSGD activations, the
paired anchor and its `Q`/`M` transaction, rank-1 RELEX, signed EMA, and the
dense control. W2 is the linear/secant case; do not add a separate projector.

## Sources of truth

- Machine-readable research defaults: `research/.claude/project.yaml`.
- Run reference: `examples/grpo_trainer/COMM_EFF_CONFIG.md`.
- Runtime: `verl/workers/comm_eff/` and
  `verl/workers/engine/fsdp/transformer_impl.py`.

## Repository rules

- Keep `AGENTS.md` unchanged and follow it.
- Work on `exp/*` branches; never commit directly to `main`.
- Experiment code targets the base branch configured in
  `research/.claude/project.yaml`.
- Do not expose values from `~/.config/verl-research/secrets.env`.
- Preserve unrelated changes; do not push or open a PR unless requested.
