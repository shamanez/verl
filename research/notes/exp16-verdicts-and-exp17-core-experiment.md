# EXP-16 claim verdicts, the no-rescale/pre-norm grad-norm mechanism, and the EXP-17 core experiment

**2026-06-01. Qwen2.5-1.5B-Instruct / GSM8K / vanilla GRPO no-KL no-entropy.**
Verdicts on the operator's standing claims about EXP-16 (#16, PASS, merged PR #10),
verified against the *current* code on `vast-ai-workload` and the logged runs in
`research/runs/EXP-16/`. Feeds the core experiment **issue #17**.

This doc is the analysis; the deep evidence lives in sibling notes — see Pointers.

---

## Summary table

| # | Claim | Verdict |
|---|---|---|
| 0 | Mask identical per-step / boundary / token / prompt → clean importance sampling | ✅ Correct & proven |
| 0b | Are we 100% sure the gradient update happens in the comm-eff case? | ✅ Yes — updates happen; the question is whether they're *useful* |
| 1 | Rescale is a must / default; reduces grad-norm; residual a few× larger, OK for now | 🟡 Right conclusion, wrong reason (it's unbias, residual is variance) |
| 2a | Spectral correction "doesn't work" | ✅ Correct (val 0.080 ≈ random) |
| 2b | Full anchor grad is NOT added to autograd graph — only into M, then used to correct | ✅ Correct & code-verified |
| 2c | Reason: RL doesn't update principal components / sparse high-rank | 🟡 Right instinct; sharper reason = orthogonality + grad never applied |
| 3 | Identical masks ⇒ no clipping; the every-K drop is suspicious — "find what's wrong" | ✅ Nothing wrong — the drop IS the clean step; ~0.04 is normal PPO inner-loop drift |
| 4 | Without rescale grad-norm is huge because pre-norm + the backward gets something big | ✅ Correct — RMSNorm's 1/RMS backward, compounded over 7 boundaries |

---

## Claim 0 — mask consistency → clean IS ✅

The mask key is `(base_seed, layer_idx, global_step, sample_id, position_id, channel)` —
**no per-call term**, and `global_step` is constant across the `old_log_prob` recompute
and the train forward. With `mask_recompute=true` (launcher default) both gradient-feeding
forwards get the **bit-identical** mask. Proven, not assumed:

- cell0 preflight: `cross_pass_mask_bit_identical: True`; 44/44 mask + 8/8 rescale unit tests pass.
- cell 7 runtime (all 50 steps): `mask_applications/{rollout,ref_logprob,val}=0` every step;
  `train` and `old_logprob` masked **together** or clean **together**; all 7 boundaries report
  `mask_ratio=0.899902`, within-step spread `0.0`.
- `actor/ppo_kl ≈ 1e-3` throughout ⇒ `exp(logp − old_logp) ≈ 1` at the first inner step.

This is what keeps the IS ratio ≈ 1 at the first inner mini-batch.

## Claim 0b — is the gradient update actually happening? ✅ Yes, 100%

`engine/base.py:155-182` — `train_batch` is `zero_grad → anchor_refresh → forward_backward_batch
(masked) → grad_correction → optimizer_step()`, **exactly one** `optimizer.step()` per call.
Masked steps show finite nonzero `grad_norm ≈ 5–8`, mask counters increment, weights move.

The real distinction is *useful* vs *not*:
- pure masked, no clean step (cell 2): reward `0.126 → 0.147` — updates happen, **don't learn**;
- masked + clean step (cell 4/7): reward `0.13 → 0.62 / 0.78` — **learns**.

## Claim 1 — rescale 🟡 right conclusion, reframe the reason

Rescale is the launcher default (`COMM_EFF_MASK_RESCALE=true`) and is necessary — but its
**primary role is *unbiasedness*, not grad-norm taming**:

- Without rescale, `E[h·m] = (1−p)·h` → the forward sits at ~10% magnitude, **off-distribution**
  → biased gradient. Adam cannot fix a biased *direction*. The grad-norm 2700→~5 collapse is a
  *side effect* of removing that distortion.
- With rescale `h·m/(1−p)`, `E[h̃]=h` → **unbiased**.
- The residual (`grad_norm ~5–8` masked vs `~0.38` dense, ≈ 13–20×) is the **variance** penalty
  `≈ p/(1−p) = 9×`, *not* leftover bias — bounded by Adam scale-invariance + grad-clip = 1.0.

So "OK for now" is fair, but the residual large grad-norm is *variance*, and it's exactly that
variance which prevents pure-masked learning and which anchor/spectral was supposed (but fails) to tame.

## Claim 2 — spectral correction

### 2a "doesn't work" ✅
cell 5 `…anchor2_spectral2_20steps` (α=0.5, τ=0.01, β=0.9): reward `0.140 → 0.131` (flat),
**GSM8K val 0.080 ≈ random**, masked pearson(actor,rollout) `~0.0045`, spectral `rel_change ~0.24`.
It did not close the train-inference gap and did not learn.

### 2b "full grad not in autograd graph — only into M, used only to correct" ✅ code-verified
`anchor.py` + `engine/base.py:172-178`:
- the anchor runs its unmasked fwd/bwd on a **deep-copied no-hook clone** (`build_anchor_module`),
  reads **RAW** `p.grad` off the *clone* (`extract_target_grads`), and feeds it to
  `SpectralFilter.update_anchor` (the EMA `M ← β·M + (1−β)·G_anchor`) via `feed_anchor_grads_into_ema`
  — which calls `update_anchor` and **never** `correct_matrix`;
- the clone shares **no** param `id()` with the live optimizer/FSDP module
  (`assert_anchor_module_isolated`), takes **no** `optimizer.step()`, and the invariants
  `anchor_optimizer_steps == anchor_grad_corrected == 0` enforce it;
- `M`'s SVD basis `(U,S,V)` then projects **only** the *masked* gradient:
  `G_filt = U diag(d)(Uᵀ G_mask V)diag(d) Vᵀ`, `G_proj = α·G_mask + (1−α)·G_filt` (`spectral_filter.py`).

So **`G_anchor` influences the update purely as geometry, never as an additive gradient term** —
exactly as claimed, and within this design adding it directly would be wrong.

### 2c the reason — right instinct, sharper statement
"RL doesn't update the principal components / sparse high-rank" is in the right spirit, but the
decisive fact is **orthogonality + the clean gradient is never applied**:

1. **Spectral is a *linear reweighting of `G_mask`* in the anchor basis** (`two_sided_projection`
   is linear in `g_mask`). Masking makes `G_mask` nearly **orthogonal** to the dense/true gradient
   (cos ≈ 0, pearson ≈ 0.004). **No linear projection of `G_mask` onto a clean basis can
   manufacture a direction `G_mask` doesn't contain.** The filter denoises a vector that no longer
   points the right way — hence `rel_change ≈ 0.24` (it changes the gradient, just not toward dense).
2. **Non-stationarity / staleness (secondary):** GRPO advantages shift every step ⇒ the dominant
   subspace is non-stationary and higher-rank than SFT; the anchor is also `delay_K`-stale and
   EMA-lagged (β=0.9), so the basis itself is weak.

**Load-bearing lesson:** `clean_cadence` (which *applies* the true dense gradient) → val **0.729**;
anchor+spectral (which *only uses* the dense gradient as projection geometry and **never applies it**)
→ val **0.080**. The clean signal must be **applied**, not used as geometry. → motivates EXP-17.

## Claim 3 — the clipfrac drop every K steps — **nothing is wrong**

From `…clean_every5_50steps` (clean_cadence=5). The drop the operator flagged (step 24 → 25):

| step | type | grad_norm | entropy | ppo_kl | pg_clipfrac | clipfrac_lower | train-mask | old-mask |
|---|---|---|---|---|---|---|---|---|
| 23 | masked | 7.23 | 5.91 | 8.0e-4 | 0.0435 | 5.9e-4 | 266 | 175 |
| 24 | masked | 7.59 | 5.93 | 1.2e-3 | 0.0428 | 6.1e-4 | 280 | 182 |
| **25** | **clean** | **0.379** | **0.365** | **2.3e-5** | **3.7e-4** | **0.0** | **280** (frozen) | **182** (frozen) |
| 26 | masked | 5.86 | 5.93 | 6.0e-4 | 0.0398 | 5.4e-4 | 294 | 189 |

**(a) Why clipfrac collapses at step 25 — it's the clean step.** Mask counters **freeze**
(train 280→280, old 182→182), entropy 5.93→0.365, grad 7.59→0.379, `rollout_probs_diff` 0.85→0.0035.
On a clean step **both** the `old_log_prob` recompute and the train forward are unmasked
(`clean_cadence` replaces the masked step and unmasks both — `comm_eff.py` docstring), so the IS
ratio is **identically 1** at the first inner step → `clipfrac → 4e-4`, `clipfrac_lower → exactly 0`.
**This is a *positive* correctness signal** — if you were using vLLM log-probs or had a mask
inconsistency, the clean step would *spike*, not collapse.

**(b) Why clipfrac is ~0.04 on masked steps despite identical masks — standard PPO inner-loop drift.**
Mask consistency only guarantees ratio ≡ 1 at *sub-step 1*. The config is `TRAIN_BATCH=128`,
`PPO_MINI_BATCH=64`, `ppo_epochs=1` (`actor.py:180`) ⇒ **2 sequential mini-batch optimizer
sub-steps per global step**. `old_log_prob` is computed once, before the update:
- sub-step 1: weights = old → ratio ≡ 1 (mask consistent) → ~0 clip;
- sub-step 2: optimizer already stepped once → weights moved → ratio `= exp(logπ(w₁) − logπ(w₀)) ≠ 1`
  → contributes to clipfrac.

This is **standard PPO** and happens in dense GRPO too. It is **not** mask inconsistency and **not**
vLLM log-probs (`mask_recompute=true`; `rollout_correction.{rollout_is,rollout_rs}=null, bypass_mode=false`).

**Why masked clipfrac (~0.04) ≈ 100× the clean value (~0.0004):** the masked forward is a
near-uniform, high-entropy (5.92) function — flattens the logits ~15×, so it is **high-curvature /
fat-tailed**. After one optimizer sub-step, a small fraction (~4%) of tokens' log-probs swing
outside the clip band **even though the mean ratio drift (`ppo_kl ~1e-3`) is tiny**. The smooth
true policy on the clean step (entropy 0.37) has almost no tail → 0.04%. clipfrac reads out the
between-sub-step curvature of whatever function is differentiated: distorted masked vs smooth clean.

**The only thing to *watch* (and EXP-17 measures it): at a sparser clean cadence (K=20), does
masked-step clipfrac climb toward saturation?** At K=5 it's flat at ~0.04 (no-rescale was a
saturating ~0.15). A climb between clean steps = the masked policy drifting away from the rollout
faster than the clean step can re-anchor it.

## Claim 4 — without rescale the grad-norm is huge: pre-norm RMSNorm's 1/RMS backward ✅

**Yes — exactly right, and it's the dominant mechanism.** Full triangulated evidence (single-GPU
probes, full-model CPU run, elementary autograd, p-sweep) is in the sibling note
[`grad_norm_blowup_norescale_rmsnorm.md`](grad_norm_blowup_norescale_rmsnorm.md); the chain:

**1. Masking shrinks the residual-stream RMS.** At a boundary `h̃ = h ⊙ m`, `m∈{0,1}`, keep prob
`(1−p)=0.1`. The zeroed channels stay in RMSNorm's normalizer, so
`RMS(h̃) = √((1/H)Σ_kept h_i²) ≈ √(1−p)·RMS(h) ≈ 0.316·RMS(h)` (i.i.d. floor). On the *real* model
the boundary RMS collapses **~48×** (`grad_diag2.py`: 52.6 → 1.09), much more than √(1−p), because
masking usually removes the few **massive-activation / outlier channels** that carry most of the RMS energy.

**2. Pre-norm is the coupling.** Qwen2.5 is pre-norm: the masked residual stream feeds **directly**
into the next block's `RMSNorm(x) = x/RMS(x)·γ`. The forward is **scale-invariant**
(`RMSNorm(c·x)=RMSNorm(x)`) — which is exactly why masking is *invisible* as a forward magnitude
problem. But scale-invariance of the forward **forces** the backward Jacobian to scale as **1/RMS**:
if scaling the input by `c` leaves the output unchanged, the Jacobian must scale by `1/c`. Exactly,
`∂y_i/∂x_k = (γ_i/RMS(x))·[δ_ik − x_i x_k/(H·RMS(x)²)]` — leading factor **`1/RMS(x)`**. So the
gradient flowing **back into the masked residual** is multiplied by `1/RMS(h̃)` — *larger* when the
masked RMS is *smaller*. (If the architecture were post-norm, the norm would sit upstream of the mask
point and this coupling would change — pre-norm is what guarantees a norm is always immediately
downstream of the masked residual.)

**3. It compounds over the 7 boundaries** (layers `3,7,11,15,18,21,24`). A gradient descending to the
**early** layers crosses every downstream boundary-norm, each contributing a `1/RMS` amplification:
```
no-rescale  ≈ (1/√(1−p))^7 ≈ 3.16^7 ≈ 3900×   (measured 5620× isolated, ~7100× in the FSDP harness)
rescale     ≈ ~1×                              (measured 2.3× isolated, ~12× harness)
```
The fingerprint is **where** the gradient lives: the per-layer breakdown for no-rescale is dominated
by **layers 0–4** (the earliest, whose backward path crosses the most amplifying norms). Dense/rescale
show no such pile-up. (The clean geometric `3.16^7` matches better than the naive `48^7` because the
quantity that actually compounds along the backprop chain is closer to the i.i.d. estimate, not the
full single-boundary forward RMS collapse — the projection term in the Jacobian and re-normalization
interact. The order-of-magnitude and the layer signature are the real proof.)

**4. Why rescale fixes it.** `1/(1−p)` restores — actually *overshoots* — the RMS
(`RMS(h̃) = √(1/(1−p))·RMS(h) > RMS(h)`), so the `1/RMS` factor returns to ~1 (even slightly below) →
no per-boundary amplification → nothing to compound. The forward was always fine (norm is
scale-invariant); **rescale's real job is keeping the *backward* well-conditioned.** The residual ~12×
(harness) is the mask-backward gain `√(1/(1−p))≈3.2×` + stochastic-mask variance `~p/(1−p)≈9×`,
**not** the RMS collapse.

This is **architecture-general** (any pre-norm RMSNorm/LayerNorm LLM — Llama, Mistral, Gemma, Phi);
the law `masking a later-normalized activation ⇒ backward gain ∝ 1/scale` and its inverted-dropout
fix do not depend on Qwen. **Not FSDP, not gradient-checkpointing, not in-place/fused kernels** —
all ruled out by single-GPU + elementary-autograd probes (see the sibling note).

---

## Synthesis for the new PP-comm-efficient algorithm

1. **Rescale (unbias) is mandatory** but only buys correctness at a variance cost; pure-masked p=0.9
   does not learn.
2. **The clean gradient must be APPLIED, not used as geometry.** clean_cadence (applies) learns to
   near-dense (val 0.729 vs 0.741); anchor+spectral (geometry only) fails (0.080). Spectral can't help
   because masking rotates `G_mask` ~orthogonal to dense and the filter is a *linear* reweighting of it.
3. **The train-inference gap is mask-caused, large, but stationary at K=5.** The open scaling question:
   does it stay stationary — and does masked-step clipfrac stay bounded — at a *sparse* clean cadence?

## EXP-17 — the core experiment (issue #17)

Single run, `code_change:false`. Mask p=0.9 + rescale, per-(token,channel) at the 7 boundaries,
**`clean_cadence=20`** (apply the true dense gradient every 20 steps), **anchor+spectral OFF**,
**2 epochs (~116 steps)**, **validate every 10**, no train-inference correction (rollout correction
strictly off). Clean steps fire at 20/40/60/80/100; ~85.5% of boundary-activation traffic saved.

**Question:** does it keep learning / diverge / entropy-collapse / saturate, and does the
train-inference mismatch (pearson, `rollout_corr/kl`, `rollout_probs_diff`) stay flat or grow at K=20?
PASS = learns (val ≥ step0+0.05) + gap stationary + grad finite; parity with dense is a bonus, not
required (characterization run). Supersedes the falsified M95+AP anchor+spectral plan in #11.

## Pointers (sibling notes)

- [`grad_norm_blowup_norescale_rmsnorm.md`](grad_norm_blowup_norescale_rmsnorm.md) — full evidence for Claim 4 (probes, p-sweep, ruled-out hypotheses).
- [`grpo_update_forward_count.md`](grpo_update_forward_count.md) — the per-step forward/optimizer-sub-step count behind Claim 3.
- [`pipeline_parallel_dense_matching_strategy.md`](pipeline_parallel_dense_matching_strategy.md) — the low-rank/spectral direction analysis behind Claim 2c.
- [`grpo_mask_cross_pass_consistency.md`](grpo_mask_cross_pass_consistency.md) — the mask-key cross-pass proof behind Claim 0.
- [`fast-circuit-vs-anchor-pass.md`](fast-circuit-vs-anchor-pass.md) / [`anchor-memory-cost.md`](anchor-memory-cost.md) — the anchor circuit behind Claim 2b.
