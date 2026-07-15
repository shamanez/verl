# Communication-Efficient GRPO Default

## Run

Prepare MATH `train.parquet` and `test.parquet` under `$HOME/data/math`, then run:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh
```

Dense control:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_comparison_fsdp.sh dense
```

## Settings

| Area | Default |
| --- | --- |
| Model | `Qwen/Qwen2.5-Math-1.5B` |
| Data | MATH train/test, last `\boxed{}` answer with `is_equiv` |
| GRPO | batch 512, mini-batch 256, `n=8`, prompt/response 1024/3072 |
| Optimizer | AdamW `1e-6`, weight decay `0.01`, gradient clip `1` |
| Objective | reference `low_var_kl=0.001`, reward KL off, entropy 0 |
| Schedule | 100 steps, validation every 25 steps |
| Activations | PowerSGD rank 77, synchronized warm-start `Q`, fast-Q bootstrap |
| Anchor | paired dense replay, cadence/delay 20/20, `ppo_minibatch`, CPU |
| Weight projection | rank-1 RELEX, W2 → W3 → W4, window 4/minimum 2, strength 1 |
| Anchor warmup/routing | `stale_correct`, `auto`/current trajectories |
| Gradient signal | signed-EMA all-floating `M`, beta 0.50, alpha 0.25, CPU |

## Q lifecycle

`Q` is seeded once as a deterministic random orthonormal basis, then (with
`fast_q_bootstrap`) overwritten once by an activation-derived `Q1` from a
discarded dense prepass, before the first compressed forward. After that the
anchor owns `Q` and refreshes it only when it fires (cadence 20); the fast
circuit is a read-only consumer. `Q` is always warm: refined across fires,
never re-randomized.

## Weight projection (`anchor.lookahead_mode`)

- `rank1_relex` (default): forecast each anchor tensor forward with a rank-1 fit
  over the last `W` anchor snapshots (`window 4`, `min 2`, `strength 1`); the
  window ramps W2 → W3 → W4 as snapshots accumulate.
- `disabled`: no forecast; the anchor uses its stale weights as-is.

## Anchor batch scope (`anchor.batch_scope`)

- `ppo_minibatch` (default): anchor replays one PPO mini-batch, 256 prompts
  (2048 sequences), half the 512-prompt actor update.
- `rollout_batch`: anchor replays the full update, all 512 prompts (4096
  sequences) per fire, i.e. the entire prompt set.

Generic Hydra keeps `comm_eff.enabled=false`. The method launcher sets it to
`true`. Exact shell defaults live in
`run_qwen25_math_1p5b_rank1_relex_fsdp.sh`; schema defaults live in
`verl/workers/config/comm_eff.py`.
