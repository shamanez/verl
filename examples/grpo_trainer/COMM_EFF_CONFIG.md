# Communication-efficient masked-GRPO — configuration & how to change it

This is the operator reference for the comm-eff method wired into
`vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh` (Qwen2.5-1.5B / GSM8K / vanilla
GRPO). It documents **what the method does, every knob, and how to change it**.

The comm-eff base (issue #25) is the **anchor circuit on a PowerSGD codec**: PowerSGD
projects each pipeline-boundary activation onto a low-rank basis `Q`, a mandatory
**anchor** maintains a stale full-gradient reference `M` and is the **only** thing that
updates `Q`, and a **merger** folds `M` into the fast gradient. **Rollouts come from
ordinary, unmasked vLLM** — compression is about inter-stage *training* traffic. With the
method disabled (`comm_eff.enabled=false`) the path is **byte-identical to upstream verl**.
The legacy per-(token, channel) activation **mask** (`prf_mask`) is retained as a
reference-only codec. Result + why the merger is still open: `research/runs/SUMMARY.md`.

> All knobs are leading `KEY=value` env overrides in front of the launcher
> (`launcher reads them as ${VAR:-default}`). They map to
> `actor_rollout_ref.actor.comm_eff.*` Hydra fields. **The `default` column below is the
> Hydra dataclass default (the all-OFF state, so `enabled=false` is byte-identical dense);
> the BASE values that actually run are the launcher `${VAR:-default}` — see the launcher
> header + `FIXED_CONTROL_SURFACE.md`, not duplicated here.**

## The master switch

| env | hydra | default | meaning |
|---|---|---|---|
| `COMM_EFF_ENABLED` | `comm_eff.enabled` | `false` | **`false` ⇒ strict no-op = unmodified dense verl.** Set `true` to turn the method on. Everything below is inert when this is `false`. |

## 1. Activation mask (`prf_mask`) — reference-only codec (NOT the base)

> The base codec is PowerSGD (§5). The mask below is the legacy codec, kept for
> reference/ablation; to run it set `COMM_EFF_COMPRESSION_TYPE=prf_mask
> COMM_EFF_MASK_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=false`.


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

## 2. Clean cadence — DEAD (superseded by the anchor)

| env | hydra | default | how to change |
|---|---|---|---|
| `COMM_EFF_CLEAN_CADENCE` | `comm_eff.clean_cadence` | `0` | Every Nth step runs a full **unmasked** dense step. **Leave at `0`.** |

A periodic full-rank dense step recovered parity in early masked experiments, but it is
**not communication-efficient** (full-H transfer) and, on a real decentralized-PP link,
would itself be stale. The mandatory **anchor circuit** (§3) is its realistic replacement.
`clean_cadence` is kept only as a historical/diagnostic knob, is OFF in the base, and
should not be re-enabled.

## 3. Anchor circuit + merger — the MANDATORY base (issue #25)

> These default OFF in the Hydra dataclass (byte-identity) but are **ON in the launcher
> base**: the anchor is mandatory and is the only thing that updates `Q`
> (`anchor.owns_q=true`); the `signed_ema` merger folds `M` into the fast gradient. The
> merger is the open **research axis** (`signed_ema` is falsified — see `SUMMARY.md`).


| env | hydra | default | how to change |
|---|---|---|---|
| `COMM_EFF_ANCHOR_ENABLED` | `anchor.enabled` | `false` | turn on the anchor EMA. |
| `COMM_EFF_ANCHOR_CADENCE` | `anchor.cadence` | `1` | refresh the anchor every Nth step (fires on multiples only — verified: `step=2,4,6,…` for cadence 2). |
| `COMM_EFF_ANCHOR_DELAY_K` | `anchor.delay_K` | `20` | use a K-step-stale snapshot. **For short runs set it to the cadence (e.g. `2`)** — a `delay_K=20` snapshot never materializes in a ≤20-step run. |
| `COMM_EFF_SPECTRAL_ENABLED` | `spectral.enabled` | `false` | turn on anchor-guided grad correction. |
| `COMM_EFF_SPECTRAL_CADENCE` | `spectral.cadence` | `1` | apply correction only every Nth step via `state.should_run_spectral_correction()`. `1` = every step = strict no-op when disabled. Set **equal to `anchor.cadence`** so corrections use a freshly-refreshed anchor EMA. |
| `COMM_EFF_SPECTRAL_BETA_ANC` | `spectral.beta_anc` | `0.9` | anchor EMA decay. |
| `COMM_EFF_SPECTRAL_EMA_DEVICE` | `spectral.ema_device` | `gpu` | `cpu` to offload the EMA (saves GPU memory; slower). |
| `COMM_EFF_SPECTRAL_MAX_TARGETS` | `spectral.max_targets` | `-1` | how many 2D weight targets to correct per firing. `-1` = no cap = full coverage of all 196 projection matrices the merger corrects; caps BOTH anchor extraction AND the merger. |
| `COMM_EFF_SPECTRAL_CORRECTION_MODE` | `spectral.correction_mode` | `signed_ema` | anchor combiner. `signed_ema` (EXP-25/R3): `α·G_noisy + (1−α)·\|G_noisy\|·sign(M)`. `inject`: add the scale-matched anchor complement. `blend`: convex blend toward the scale-matched anchor. |
| `COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA` | `spectral.signed_ema_alpha` | `0.0` | the signed_ema merger weight α. `0` = pure `\|G_noisy\|·sign(M)`; `1` = `G_noisy` unchanged. THE swept axis. Active iff `correction_mode=signed_ema`. |
| `COMM_EFF_SPECTRAL_INJECT_GAMMA` | `spectral.inject_gamma` | `1.0` | injection strength for `correction_mode=inject`; unused otherwise. |
| `COMM_EFF_SPECTRAL_BLEND_ETA` | `spectral.blend_eta` | `0.5` | convex-blend weight for `correction_mode=blend`; unused otherwise. |

> **Note (EXP-25):** the old SVD/Tikhonov/two-sided-projection "reweight" correction +
> its seeded-anchor cache were **removed** (EXP-21 proved that projection operator inert
> here, `G_filt`≈0). The live merger is `signed_ema` — magnitude from the fast compressed
> grad, sign from the stale anchor EMA `M`. It is **ON in the base** but **falsified** as a
> learning improvement (net-harmful vs plain PowerSGD; see `SUMMARY.md`) — the merger
> primitive is the open research axis, not a settled win.

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

# The anchor base — nothing to set, the launcher defaults ARE the base
TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 EXPERIMENT_NAME=ce_anchor_base \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# Sweep the merger axis (the research axis)
COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.7 EXPERIMENT_NAME=ce_a0p7 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# Legacy prf_mask codec (reference only; cannot anchor-own-Q)
COMM_EFF_COMPRESSION_TYPE=prf_mask COMM_EFF_MASK_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=false \
COMM_EFF_MASK_P=0.9 EXPERIMENT_NAME=ce_mask_ref \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
```

## What to watch (numeric, in `train.log` / WandB)
- `actor/grad_norm` — finite, ~4–8 masked / ~0.4 dense. NaN/Inf ⇒ stop.
- `actor/comm_eff/mask_ratio` ≈ `p`; `mask_applications/{train,old_logprob}` equal & nonzero, `{rollout,ref,val}=0` (proves cross-pass consistency).
- `comm_eff/clean_steps` increments on clean-cadence steps; `spectral_corrections`/`anchor_backwards` are rank-summed (read the per-fire log lines for true cadence).
- `training/rollout_actor_probs_pearson_corr` — the train↔inference correlation (the gap to close).
