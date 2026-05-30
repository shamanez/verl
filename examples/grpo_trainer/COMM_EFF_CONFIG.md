# Communication-efficient masked-GRPO — configuration & how to change it

This is the operator reference for the comm-eff method wired into
`vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh` (Qwen2.5-1.5B / GSM8K / vanilla
GRPO). It documents **what the method does, every knob, and how to change it**.

The method masks per-(token, channel) activations at the pipeline-stage boundary
layers during the *training* forward/backward only. **Rollouts come from ordinary,
unmasked vLLM** — compression is about inter-stage *training* traffic. With the
method disabled the path is **byte-identical to upstream verl** (proven: dense
reference cell, `grad_norm 0.387`, GSM8K val acc 74.1%).

> All knobs are leading `KEY=value` env overrides in front of the launcher
> (`launcher reads them as ${VAR:-default}`). They map to
> `actor_rollout_ref.actor.comm_eff.*` Hydra fields.

## The master switch

| env | hydra | default | meaning |
|---|---|---|---|
| `COMM_EFF_ENABLED` | `comm_eff.enabled` | `false` | **`false` ⇒ strict no-op = unmodified dense verl.** Set `true` to turn the method on. Everything below is inert when this is `false`. |

## 1. Activation mask (the core)

| env | hydra | default | how to change |
|---|---|---|---|
| `COMM_EFF_MASK_ENABLED` | `mask.enabled` | `false` | `true` to mask boundary activations. |
| `COMM_EFF_MASK_P` | `mask.p` | `0.9` | Fraction of (token,channel) entries zeroed at each boundary. **Lower `p` ⇒ less compression but smaller train↔inference gap.** Try `0.7`/`0.5` to recover signal if `0.9` is too aggressive. |
| `COMM_EFF_MASK_RESCALE` | `mask.rescale` | `true` | Inverted-dropout `1/(1-p)` gain. **Keep `true`.** `false` collapses activation RMS and blows up `grad_norm` (~2700 vs ~4.5) via the pre-norm `1/RMS` backward — a closed finding; do not run no-rescale. |
| `COMM_EFF_MASK_RECOMPUTE` | `mask.mask_recompute` | `true` | Recompute the mask in the `old_log_prob` pass so it is **bit-identical** to the train forward (keeps the importance ratio valid). Keep `true`. |
| (n/a — structural) | `mask.rescale_mode` | `auto` | Magnitude-restoration scheme — see below. The legacy `rescale` bool maps through `auto`. |

### `mask.rescale_mode ∈ {none, constant, rms_match, auto}` (added EXP-16)

How the masked activation `h⊙m` is re-scaled before it crosses the wire:

| mode | formula | use it when |
|---|---|---|
| `none` | `h⊙m` | never (RMS collapse → grad blow-up). |
| `constant` | `h⊙m / (1-p)` | **default / recommended.** Inverted dropout; the grad-norm stabilizer. Its RMS *overshoot* damps the downstream RMSNorm backward — a feature. |
| `rms_match` | `h⊙m · detach(rms_true / rms_masked)` | only for exact forward-activation stats / low-bit quant. Per-token EXACT pre-mask RMS, but **worse** grad-norm than `constant`. |
| `auto` | `constant` if `rescale=true` else `none` | back-compat; what existing configs resolve to. |

To select explicitly: `actor_rollout_ref.actor.comm_eff.mask.rescale_mode=rms_match`.

## 2. Clean cadence — *the stabilizer that works*

| env | hydra | default | how to change |
|---|---|---|---|
| `COMM_EFF_CLEAN_CADENCE` | `comm_eff.clean_cadence` | `0` | Every Nth step runs **unmasked** (a true dense step). `0` = off. |

**This is the lever that recovers dense parity.** EXP-16: pure mask stalls
(reward 0.13→0.15); `clean_cadence=4` → 0.62 in 20 steps; `clean_cadence=5` over
50 steps → reward **0.778 / val acc 72.9%** ≈ dense (0.779 / 74.1%). The clean
step periodically re-correlates the masked actor with the on-policy rollout
(pearson→0.999 on clean steps vs ~0.005 on masked steps). **To trade compression
for convergence, lower the cadence (more frequent clean steps).**

## 3. Anchor EMA + spectral correction (advanced; default OFF)

| env | hydra | default | how to change |
|---|---|---|---|
| `COMM_EFF_ANCHOR_ENABLED` | `anchor.enabled` | `false` | turn on the anchor EMA. |
| `COMM_EFF_ANCHOR_CADENCE` | `anchor.cadence` | `1` | refresh the anchor every Nth step (fires on multiples only — verified: `step=2,4,6,…` for cadence 2). |
| `COMM_EFF_ANCHOR_DELAY_K` | `anchor.delay_K` | `20` | use a K-step-stale snapshot. **For short runs set it to the cadence (e.g. `2`)** — a `delay_K=20` snapshot never materializes in a ≤20-step run. |
| `COMM_EFF_SPECTRAL_ENABLED` | `spectral.enabled` | `false` | turn on spectral grad correction. |
| `COMM_EFF_SPECTRAL_CADENCE` | `spectral.cadence` | `1` | **(added EXP-16)** apply correction only every Nth step via `state.should_run_spectral_correction()`. `1` = every step = prior behavior = strict no-op when disabled. Set **equal to `anchor.cadence`** so corrections use a fresh basis. |
| `COMM_EFF_SPECTRAL_ALPHA` | `spectral.alpha` | `0.5` | projection strength (less `G_mask`). Lower (`0.3`) ⇒ stronger projection toward the dense direction. |
| `COMM_EFF_SPECTRAL_TAU` | `spectral.tau` | `0.01` | singular-value threshold. |
| `COMM_EFF_SPECTRAL_BETA_ANC` | `spectral.beta_anc` | `0.9` | anchor EMA decay. |
| `COMM_EFF_SPECTRAL_EMA_DEVICE` | `spectral.ema_device` | `gpu` | `cpu` to offload the EMA (saves GPU memory; slower). |
| `COMM_EFF_SPECTRAL_SVD_MODE` | `spectral.svd_mode` | `full` | `lowrank` for cheaper SVD / less memory. |
| `COMM_EFF_SPECTRAL_BASIS_CACHE` | `spectral.basis_cache` | `cache` | basis-cache mode (note: `cache` grows GPU memory over steps). |
| `COMM_EFF_SPECTRAL_MAX_TARGETS` | `spectral.max_targets` | `4` | how many 2D weight targets to correct per firing. |

> **EXP-16 result:** anchor+spectral as configured (`alpha0.5/tau0.01/beta_anc0.9`)
> runs **stably under FSDP1** (grad_norm 4.56, no DTensor/FSDP error, cadence gate
> verified) but does **not** close the train↔inference gap (pearson stays ~0.0045,
> val acc ~0.08). It is a layered fix kept **OFF by default**; the clean step is the
> working lever today. To make spectral matter, push `alpha` lower and/or combine
> with clean steps.

## 4. Throughput / memory (not comm_eff fields, but you will need these)

| env | default | how to change |
|---|---|---|
| `USE_DYNAMIC_BSZ` | `False` | **set `True`** — token-balanced packing. With static `micro_batch=1` a 1.5B model gets ~0.75% MFU; dynamic bsz → ~14% MFU, step time 129s→37s. |
| `PPO_MAX_TOKEN_LEN_PER_GPU` / `LOG_PROB_MAX_TOKEN_LEN_PER_GPU` / `REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU` | `36864` | tokens/GPU per micro-batch under dynamic bsz. On a 183 GB B200, `98304` peaks ~62 GB (no mask) / packs the full batch in ~1 micro-batch. **`free_cache_engine=True` frees vLLM KV during the actor update, so training owns ~full GPU.** |
| (anchor on) | — | **anchor runs a 2nd full forward-backward** → roughly doubles activation memory. With anchor+spectral, *halve* the token budget (EXP-16 used `32768`, peaked 162 GB) or you will OOM at `98304`. |

## How to turn it OFF
`COMM_EFF_ENABLED=false` ⇒ every field above is inert and the run is the dense
control (byte-identical to upstream). This is the parity baseline.

## Recipes (copy-paste env in front of the launcher)

```bash
# Dense reference (the bar to match)
COMM_EFF_ENABLED=false TOTAL_TRAINING_STEPS=25 \
EXPERIMENT_NAME=dense_ref bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# Recommended comm-eff config (mask + rescale + clean cadence — recovers parity)
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 \
COMM_EFF_MASK_RESCALE=true COMM_EFF_CLEAN_CADENCE=5 \
USE_DYNAMIC_BSZ=True PPO_MAX_TOKEN_LEN_PER_GPU=98304 \
TOTAL_TRAINING_STEPS=50 EXPERIMENT_NAME=mask_p0p9_rescale_clean5 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# Spectral switch-on (advanced; halve token budget for anchor's 2nd pass)
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 \
COMM_EFF_MASK_RESCALE=true COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=2 \
COMM_EFF_ANCHOR_DELAY_K=2 COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CADENCE=2 \
USE_DYNAMIC_BSZ=True PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
TOTAL_TRAINING_STEPS=20 EXPERIMENT_NAME=mask_anchor2_spectral2 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

## What to watch (numeric, in `train.log` / WandB)
- `actor/grad_norm` — finite, ~4–8 masked / ~0.4 dense. NaN/Inf ⇒ stop.
- `actor/comm_eff/mask_ratio` ≈ `p`; `mask_applications/{train,old_logprob}` equal & nonzero, `{rollout,ref,val}=0` (proves cross-pass consistency).
- `comm_eff/clean_steps` increments on clean-cadence steps; `spectral_corrections`/`anchor_backwards` are rank-summed (read the per-fire log lines for true cadence).
- `training/rollout_actor_probs_pearson_corr` — the train↔inference correlation (the gap to close).
