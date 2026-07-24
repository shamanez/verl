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
  over the retained anchor snapshots (`min 2`, `strength 1`); the fit ramps
  W2 → W3 → W4 as snapshots accumulate.
- `disabled`: no forecast; the anchor uses its stale weights as-is.

## Delta base / history (`anchor.lookahead_history_mode`)

Both modes compute cumulative deltas against a single base (`snapshot − base`,
never consecutive diffs), run the same per-tensor rank-1 Gram SVD and OLS fit
over the checkpoints' actual ticks, and pin the prediction to the newest exact
tensor (`latest + strength · slope · horizon · v1`, preserving its off-subspace
residual). They differ only in what plays the role of the base and how history
is retained.

- `sliding_window` (default): keep the last `lookahead_window_snapshots`
  checkpoints (`window 4`). The base is the oldest snapshot still in the window,
  so it advances as the window slides and the deltas track *local* drift.
  Bounded memory. `lookahead_max_snapshots` must stay `-1` here.
- `growing_fixed_base`: pin the first (seeded) anchor snapshot as a fixed base
  for the whole run and never evict it; every later exact checkpoint is appended,
  so the base-relative delta history keeps growing. This is the RELEX-faithful
  regime (deltas measured from a fixed origin over a long prefix), which gives a
  better-conditioned rank-1 direction and a more noise-robust extrapolation slope
  the longer training runs, at the cost of retaining one full-model CPU snapshot
  per checkpoint. `lookahead_max_snapshots` caps retention (`-1` = unbounded, the
  default; a positive value must be `>= lookahead_window_snapshots` and evicts the
  oldest **non-base** entry so the fixed base always survives). Warmup is
  unchanged: the projector still waits for `lookahead_min_snapshots` checkpoints
  before it engages.

Rule of thumb: prefer `growing_fixed_base` when the concern is anchor-weight
*staleness / divergence* (a long, denoised lever arm projects the stale
checkpoint forward more reliably); prefer `sliding_window` when the trajectory is
strongly non-stationary (a local base adapts to bends) or CPU memory is tight.

## sr_quant boundary codec (`compression_type=sr_quant`)

Dense low-bit stochastic-rounding quantization of the pipeline-boundary
activations, plus the same quantization of the boundary backward gradient
(modeling the compressed backward wire). The dense-but-low-precision
counterfactual to the PRF mask (sparse but full-precision). Unbiased both
ways: `E[q] = h` on the forward, `E[g_hat] = g` on the backward. The PRF draw
is keyed on `(seed, layer, step, sample_id, position_id, channel, direction)`
with no path component, so the old-logprob / train / reference forwards of one
step (and any gradient-checkpoint recompute) are bit-identical and the PPO
ratio identity is preserved; each new step draws fresh.

Knobs:

- `quant.bits` (`COMM_EFF_QUANT_BITS`, default 1): bits per channel;
  `2**bits` uniform levels span `[-s, +s]`. At `bits=1` this is a sign-like
  codec: `q in {-s, +s}` with `P(+s) = (h/s + 1)/2`.
- `quant.block_size` (`COMM_EFF_QUANT_BLOCK_SIZE`, default 32): channels per
  fp16 absmax-scale block within a token (QSGD-style bucketing; cuts noise vs
  one whole-token scale). `0` means one whole-token scale.
- `quant.rounding` (`COMM_EFF_QUANT_ROUNDING`, default `sr`): `sr` = unbiased
  PRF stochastic rounding; `rn` = deterministic round-to-nearest on the same
  grid (biased; the ablation control).
- Reused mask knobs: `mask.mask_recompute` / `mask.mask_reference` widen the
  eligible forwards exactly as for prf_mask; `mask.seed` / `mask.pp_size` key
  and place the codec. `mask.p` / `rescale*` / `exact_k` / `antithetic` /
  `frlr*` are ignored.
- Like prf_mask, sr_quant cannot anchor-own-Q: set
  `COMM_EFF_ANCHOR_OWNS_Q=false` (`COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP` is
  inert under this codec).

Budget math per token per boundary (H = 1536, bf16 wire = 16-bit values):

| Codec | Payload | Bits | Bytes |
| --- | --- | --- | --- |
| dense | 1536 ch x 16 b | 24576 | 3072 B |
| prf_mask p=0.95 | 77 kept ch x 16 b | 1232 | 154 B |
| sr_quant bits=1, block 32 (default) | 1536x1 + 48 fp16 scales | 2304 | 288 B |
| sr_quant bits=1, whole-token scale | 1536x1 + 1 fp16 scale | 1552 | 194 B |

`comm_eff/logical_pp_bits_sr_quant` (and the `_bytes_` variant) reports
`H*bits + n_blocks*16` at runtime, the sr_quant analogue of
`comm_eff/logical_pp_bytes_prf`.

## Anchor batch scope (`anchor.batch_scope`)

- `ppo_minibatch` (default): anchor replays one PPO mini-batch, 256 prompts
  (2048 sequences), half the 512-prompt actor update.
- `rollout_batch`: anchor replays the full update, all 512 prompts (4096
  sequences) per fire, i.e. the entire prompt set.

Generic Hydra keeps `comm_eff.enabled=false`. The method launcher sets it to
`true`. Exact shell defaults live in
`run_qwen25_math_1p5b_rank1_relex_fsdp.sh`; schema defaults live in
`verl/workers/config/comm_eff.py`.
