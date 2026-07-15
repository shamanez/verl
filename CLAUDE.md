# Agent Instructions for this verl fork

This fork is only for the communication-efficient GRPO pipeline. Ignore model,
dataset, and launcher defaults from unrelated upstream examples.

## Current default

- Model: `Qwen/Qwen2.5-Math-1.5B`.
- Data: prepared MATH `train.parquet` and `test.parquet` from
  `EleutherAI/hendrycks_math`; default directory `$HOME/data/math`.
- GRPO: train batch 512, PPO mini-batch 256, rollout `n=8`, prompt/response
  1024/3072, AdamW `1e-6`, reference `low_var_kl=0.001`, 100 steps.
- Activations: PowerSGD rank 77 with synchronized, warm-started,
  activation-derived `Q` and fast-Q bootstrap.
- Anchor: paired dense replay, cadence/delay 20/20 optimizer ticks,
  `ppo_minibatch`, CPU replay/snapshots, anchor-owned `Q`.
- Weights: rank-1 RELEX, progressive W2 → W3 → W4, window 4, minimum 2,
  strength 1, `auto`, `stale_correct`.
- Gradient signal: signed-EMA `M` over all floating parameters,
  `beta_anc=0.50`, `alpha=0.25`, CPU state.

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
