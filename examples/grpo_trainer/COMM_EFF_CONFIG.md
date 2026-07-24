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
- `quant.subset_k` (`COMM_EFF_QUANT_SUBSET_K`, default 0 = full width): the
  issue #93 I5 byte-parity hybrid. `> 0`: per token quantize only a PRF-fresh
  EXACT-`subset_k` channel subset J (drawn with the mask codec's exact-k order
  statistic, keyed identically, so J is bit-identical across the
  old/train/ref passes of one step and shared by the backward wire), zero
  elsewhere, rescale by `H/subset_k`. Unbiased through BOTH the subset draw
  and the stochastic rounding: `E[q] = h`. Blocks then span `subset_k`
  consecutive KEPT channels; J costs no index bits (PRF-derivable at the
  receiver).
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
| sr_quant bits=2, block 32, subset_k=493 | 493x2 + 493/32 fp16 scales | 1232.5 (ceil 1233) | 154.1 B |

The subset row is the #93 4.3 byte-parity arm: 1233 bits vs the prf exact-k
incumbent's 1232 (77 kept x fp16), the same-budget fair fight. The ragged
final scale block is counted pro-rata (16/block bits per kept channel).

`comm_eff/logical_pp_bits_sr_quant` (and the `_bytes_` variant) reports
`H*bits + n_blocks*16` at runtime (`subset_k*bits + subset_k*16/block` in
subset mode), the sr_quant analogue of `comm_eff/logical_pp_bytes_prf`.

## Dense-view probe + adaptive KL coefficient (`comm_eff.probe`, issue #93 I3)

Every `probe.probe_every` trainer steps the trainer reruns the current step's
actor and reference log-prob computations ONCE with every codec silent (the
worker stamps path tag `None`, the anchor's dense precedent; no codec
eligibility set contains `None`): measurement only, no backward, no weight
change. Cost: two extra forward passes per probe (~1 percent overhead at
`probe_every=25`). The probe runs after the step's actor update, so its
reading is one step offset from the `actor/kl_loss` it is compared to.

Logged each probe step:

- `probe/kl_dense`: token-mean `kl_loss_type` (default low_var_kl / k3)
  estimate of KL(pi_theta_dense || pi_ref_dense) on response tokens, same
  estimator + aggregation as `actor/kl_loss`.
- `probe/kl_gain`: `actor/kl_loss / probe/kl_dense`, the measured G(t)
  (how much of the codec-view KL is real dense drift).
- `probe/gap_dense`: token-mean `rollout_log_probs - dense actor log probs`,
  the dense-view train-inference gap (needs `calculate_log_probs=true`).
- `probe/lr_brake_triggered`: 1.0 when the dormant LR brake WOULD fire
  (kl_dense doubled across consecutive probes while beta is pinned at
  `ctrl_beta_max`, or the gap_dense slope over the last 4 probes exceeds 3x
  the previous 4). Detection only; this build never mutates the LR.
- `probe/kl_setpoint`, `probe/kl_coef` (controller on): the setpoint `c_k`
  and the post-update beta.

Controller (`probe.ctrl_enabled`): projected dual ascent in log space with
proportional damping, updated once per probe:
`beta <- clip(beta * exp(ki*e + kp*(e - e_prev)), beta_min, beta_max)` with
`e = (kl_dense - c)/c` and
`c = max(kl_target_floor, kl_target_gain * table(step))`; anti-windup by
conditional integration (integral term freezes while pinned at a bound).
`beta_0` = the actor's `kl_loss_coef`; the trainer stamps the live beta onto
every `update_actor` batch and `ppo_loss` applies it (`actor/kl_coef` reflects
the value actually used). The anchor's replay loss keeps the static
coefficient (the anchor is unchanged in phase 1 per #93 4.8).

Knobs (defaults off / bit-identical):

- `probe.probe_every` (`COMM_EFF_PROBE_EVERY`, default 0 = off): probe
  cadence in trainer global steps.
- `probe.ctrl_enabled` (`COMM_EFF_PROBE_CTRL_ENABLED`, default false):
  requires `probe_every >= 1`.
- `probe.kl_target_table` (`COMM_EFF_PROBE_KL_TARGET_TABLE`, default empty):
  `"step:value,step:value"` dense-control reference-KL curve, linear
  interpolation with edge clamping; empty = the floor alone.
- `probe.kl_target_floor` (`COMM_EFF_PROBE_KL_TARGET_FLOOR`, default 0.005).
- `probe.kl_target_gain` (`COMM_EFF_PROBE_KL_TARGET_GAIN`, default 2.0).
- `probe.ctrl_ki` / `probe.ctrl_kp` (`COMM_EFF_PROBE_CTRL_KI/KP`, defaults
  0.3 / 0.1).
- `probe.ctrl_beta_min` / `probe.ctrl_beta_max`
  (`COMM_EFF_PROBE_CTRL_BETA_MIN/MAX`, defaults 2e-4 / 0.05).

## CVC: train the disagreement down (`actor.cvc_*`, `comm_eff.dc`, issue #93 I4)

Two independent modes, both default off (bit-identical paths), both zero
extra forward passes and zero wire cost:

CE mode (`actor.cvc_lambda`, `COMM_EFF_CVC_LAMBDA`, default 0.0 = off): adds
`lambda_eff * CE_codec` to the actor loss, where
`CE_codec = -mean_t log pi_theta(a_t)` over response tokens. Under an active
codec that log-prob IS the codec view, so the term is the training-view gap
and its gradient pulls the codec view toward the sampler's choices.
`lambda_eff` ramps linearly from 0 over `actor.cvc_warmup_steps`
(`COMM_EFF_CVC_WARMUP_STEPS`, default 20) trainer steps because the codec
view starts below uniform (the earliest CE gradient points toward
uniformizing). Kill-guard is observational: kill the arm if rollout ppl,
reward slope, or the val proxy degrade. Logs `actor/cvc_ce` and
`actor/cvc_lambda` (the coefficient actually applied at this step's warmup
clock). The anchor's replay loss does not mirror this term: its dense forward
has no codec disagreement to train down (#93 4.8, anchor unchanged in
phase 1).

DC mode (`comm_eff.dc`, DC-GRPO, arXiv 2606.08779): driver-side advantage
shaping once per step, after advantages exist and before `update_actor`:
`A_t <- A_t - lambda * delta_t` on response tokens only, with
`delta_t = |exp(old_log_probs) - exp(rollout_log_probs)|` (the codec-view
trainer probability vs the sampler's; bounded, ratio-free, so it stays alive
at E[rho] ~ 1e-3 where importance sampling dies). Then one projected dual
ascent step: `lambda <- clip(lambda + eta * (delta_bar - target), 0,
lambda_max)` with `delta_bar` the response-masked mean of `delta_t`, so
lambda regulates the GROWTH of the gap without fighting its static part.
Requires `rollout.calculate_log_probs=true`. Logs `dc/lambda` (applied this
step) and `dc/delta_bar`.

Knobs:

- `actor.cvc_lambda` (`COMM_EFF_CVC_LAMBDA`, default 0.0 = off).
- `actor.cvc_warmup_steps` (`COMM_EFF_CVC_WARMUP_STEPS`, default 20).
- `comm_eff.dc.enabled` (`COMM_EFF_DC_ENABLED`, default false).
- `comm_eff.dc.eta` (`COMM_EFF_DC_ETA`, default 1.0).
- `comm_eff.dc.target` (`COMM_EFF_DC_TARGET`, NO default: the measured
  step-1 static per-token discrepancy floor plus slack; the -1.0 sentinel is
  rejected at config time when DC is enabled).
- `comm_eff.dc.lambda0` (`COMM_EFF_DC_LAMBDA0`, default 0.05).
- `comm_eff.dc.lambda_max` (`COMM_EFF_DC_LAMBDA_MAX`, default 1.0).

## Issue #93 run matrix (`run_93_cell.sh`)

One launcher covers the whole #93 long-horizon stability matrix on the #90
protocol (batch 128 / mini 128, 1024/2048, pp 8, LR 1e-6, one H200). It
resolves the arm and echoes the full config BEFORE any bring-up, fails loud on
an unknown `ARM`, and `DRY_RUN=1` stops right after the echo:

```bash
ARM=a1 bash examples/grpo_trainer/run_93_cell.sh   # sr_quant b1/32 sr, 120 steps
ARM=a2 bash examples/grpo_trainer/run_93_cell.sh   # rounding=rn bias control
ARM=a3 bash examples/grpo_trainer/run_93_cell.sh   # byte-parity subset_k=493
ARM=a4 bash examples/grpo_trainer/run_93_cell.sh   # prf exact-k + CVC CE 0.003
ARM=a5 bash examples/grpo_trainer/run_93_cell.sh   # frlr r48 k28 + token-IS 2.0
# rounds B/C reuse a codec arm and add the control plane (table REQUIRED):
ARM=b1 CODEC_ARM=a3 COMM_EFF_PROBE_KL_TARGET_TABLE="0:5e-4,..." bash examples/grpo_trainer/run_93_cell.sh
ARM=c  CODEC_ARM=a3 COMM_EFF_PROBE_KL_TARGET_TABLE="0:5e-4,..." bash examples/grpo_trainer/run_93_cell.sh
```

Rounds A (120 steps) and B (200 steps) run validation OFF
(`VAL_BEFORE_TRAIN=False`, `TEST_FREQ=-1`) and save nothing (`SAVE_FREQ=-1`);
cell `c` runs 600 steps with val at 0/150/300/450/600, `SAVE_FREQ=100` and the
R2 checkpoint sink on (`CKPT_R2_ENABLED=true`; `R2_BUCKET` stays hard-guarded
to `shamane-pluralis`). `b1`/`c` turn on `COMM_EFF_PROBE_EVERY=25` +
`COMM_EFF_PROBE_CTRL_ENABLED=true` and REQUIRE the setpoint table from the
env. WandB: project `93-long-horizon-stability` (via `WANDB_RUN_GROUP`), run
`<arm>-<slug>`. Arm a5 enables decoupled token importance weighting via the
engine's `ROLLOUT_IS=token` / `ROLLOUT_IS_THRESHOLD=2.0` knobs (default
`null` = correction strictly off, unchanged behavior).

## Anchor batch scope (`anchor.batch_scope`)

- `ppo_minibatch` (default): anchor replays one PPO mini-batch, 256 prompts
  (2048 sequences), half the 512-prompt actor update.
- `rollout_batch`: anchor replays the full update, all 512 prompts (4096
  sequences) per fire, i.e. the entire prompt set.

Generic Hydra keeps `comm_eff.enabled=false`. The method launcher sets it to
`true`. Exact shell defaults live in
`run_qwen25_math_1p5b_rank1_relex_fsdp.sh`; schema defaults live in
`verl/workers/config/comm_eff.py`.
