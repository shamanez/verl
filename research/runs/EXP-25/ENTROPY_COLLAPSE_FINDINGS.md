# EXP-25 α=0 Entropy Collapse — Scientific Findings

**Run:** `exp25_alpha_0p0` (WandB `shamanework-pl/verl_compression_research/uyrpaftw`, state=running at time of analysis, 49 train steps logged)
**Arm:** signed-EMA merger, **α = 0.0** ⇒ `G_correct = |G_noisy| · sign(M_anchor)`
**Source data:** `research/runs/EXP-25/logs/train.log` (re-extracted in full, steps 1–49) + WandB.
**Config (effective, from the launch line in `train.log`):** GRPO, `use_kl_loss=False`, `use_kl_in_reward=False`, `entropy_coeff=0` (no-KL / no-entropy), `algorithm.rollout_correction.rollout_is=null rollout_rs=null bypass_mode=false` (rollout correction OFF — `old_log_prob` recomputed by the training policy), `lr=1e-6`, `train_batch_size=128`, `ppo_mini_batch_size=64` (⇒ 2 optimizer ticks/global step), `n=8`, `max_response_length=16384`. comm_eff: `compression_type=powersgd rank=77`, `spectral.correction_mode=signed_ema signed_ema_alpha=0.0 beta_anc=0.95 cadence=1`, `anchor.cadence=5 delay_K=5 owns_q=true`, `clean_cadence=0`.

---

## 1. One-line verdict

**The α=0 signed-EMA merger turns the optimizer into magnitude-preserving sign-SGD whose sign vector is a slow (β=0.95) EMA of a K-stale anchor gradient. With no sign cancellation across the batch and no regularizer, this drives a monotonic entropy collapse (5.69 → 0.06), a response-length explosion to the 16 384-token cap, and a reward peak-then-degrade (0.79 → 0.32). H1 is the root cause; H3 is the amplifier; H4 is corroborated; H2 is a necessary-but-insufficient permissive condition; the IS metrics are *symptoms* of collapse and the rollout-correction-OFF choice removes a brake but is shared with the non-collapsing references, so it is an amplifier, not the cause.**

---

## 2. Timeline (full re-extraction, steps 1–49)

Entropy decays **monotonically and continuously** — there is no sudden "step-37 event." The acceleration phase is ~step 25–36, exactly when response length explodes.

| step | entropy | ppo_kl | clipfrac | grad_norm | IS gap (probs_diff_mean) | pearson(train,rollout) | rollout_corr/kl | reward (critic/score/mean) | resp_len mean | resp clip_ratio | coldM | rel_change (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.69 | -0.03 | 0.143 | 414 | 0.843 | 0.021 | 13.2 | 0.136 | 278 | 0.00 | **196** | 0.000 (cold) |
| 2 | 5.86 | 0.088 | 0.136 | 197 | 0.826 | 0.016 | — | 0.148 | 285 | 0.00 | **196** | 0.000 (cold) |
| 3 | 5.75 | -7.03 | 0.136 | 0.70 | 0.822 | 0.007 | — | 0.115 | 287 | 0.00 | 0 | warming |
| 4 | **3.52** | ~0 | 0.0005 | 1.64 | 0.777 | 0.112 | — | 0.149 | 282 | 0.00 | 0 | — |
| 5 | 3.54 | -1.01 | 0.107 | 1.52 | 0.772 | 0.117 | 6.28 | 0.172 | 272 | 0.00 | 0 | **anchor refresh #1; rel→~1.4** |
| 10 | 2.08 | -0.008 | 0.151 | 3.35 | 0.612 | 0.225 | 4.25 | 0.568 | 230 | 0.00 | 0 | ~1.41 |
| 20 | 1.35 | -0.064 | 0.097 | 3.49 | 0.601 | 0.213 | 4.54 | 0.722 | 175 | 0.00 | 0 | ~1.41 |
| 25 | 1.39 | 0.174 | 0.104 | 3.51 | 0.592 | 0.229 | — | 0.733 | 229 | 0.00 | 0 | ~1.41 |
| 28 | 0.75 | -0.115 | 0.174 | 1.38 | 0.432 | 0.264 | 3.18 | **0.787 (PEAK)** | 298 | 0.008 | 0 | ~1.41 |
| 30 | 0.47 | 0.036 | 0.098 | 16.0 | 0.350 | 0.256 | — | 0.728 | **593** | 0.022 | 0 | ~1.41 |
| 34 | 0.27 | 0.006 | 0.005 | 11.8 | 0.186 | 0.303 | 1.19 | 0.725 | 1406 | 0.056 | 0 | ~1.41 |
| 36 | 0.16 | 0.006 | 0.005 | 8.75 | 0.095 | 0.351 | — | 0.700 | 1896 | 0.093 | 0 | ~1.41 |
| 38 | 0.15 | -0.066 | 0.226 | 2.05 | 0.114 | 0.301 | — | 0.582 | 3408 | 0.165 | 0 | ~1.41 |
| 40 | 0.094 | 0.344 | 0.083 | 2.51 | 0.066 | 0.335 | 0.53 | 0.504 | 4867 | 0.250 | 0 | ~1.41 |
| 42 | 0.066 | 0.002 | 0.003 | 2.46 | 0.052 | 0.353 | — | 0.407 | 6864 | 0.358 | 0 | ~1.41 |
| 45 | 0.065 | 0.016 | 0.060 | 1.03 | 0.045 | 0.346 | 0.34 | **0.318 (trough)** | 8611 | 0.459 | 0 | ~1.41 |
| 48 | 0.087 | 0.141 | 0.122 | 1.73 | 0.080 | 0.297 | — | 0.412 | 7390 | 0.390 | 0 | ~1.41 |
| 49 | 0.096 | 0.004 | 0.007 | 8.16 | 0.079 | 0.352 | 0.76 | 0.451 | 6383 | 0.324 | 0 | ~1.41 |

Validation (WandB): **α=0 val@25 = 0.7180** (only val point so far; the run was still inside the reward-degradation phase, with the reward peak at step 28 *after* this val tick).

---

## 3. The mechanism, in plain RL terms (H1 — SUPPORTED, root cause)

### 3a. What the merger computes at α=0
`verl/workers/comm_eff/spectral_filter.py:268-308` (`signed_ema_matrix`):

```python
g_corr = alpha * gm + (1.0 - alpha) * gm.abs() * torch.sign(anc)   # α=0 ⇒ |G_noisy|·sign(M)
```

`M_anchor` is a β=0.95 EMA of the **K-stale** anchor gradient (`update_anchor`, line 181; fed from a frozen clone of the policy at `θ_{t−delay_K}`). At α=0 the live (compressed) gradient `G_noisy` contributes **only its per-coordinate magnitude**; the entire **direction** of every parameter update is set by `sign(M_anchor)`.

### 3b. Why this is far more sharpening-prone than the true gradient
The true minibatch policy-gradient at a coordinate is a **sum of signed per-sample contributions** `g_i = Σ_b a_b · ∂log π / ∂θ_i`. Across a GRPO group the advantages `a_b` carry both signs and the score-function terms have mixed signs, so **per-coordinate signs partially cancel**: the true gradient's magnitude on most coordinates is small relative to the sum of absolute contributions, and many coordinates sit near zero (sign ill-defined, effective step ≈ 0). That partial cancellation is the implicit regularizer that keeps each step modest.

The merger **destroys that cancellation**: it replaces the small, partially-cancelled signed gradient with `|G_noisy|·sign(M)` — a vector that has the **full activation-rescaled magnitude of the compressed gradient on every coordinate** but a **fixed sign** taken from the slow anchor EMA. Two consequences:

1. **Effective step-size inflation.** For a coordinate whose true gradient is near zero (signs nearly cancel, `|g_i| ≈ small`), `|G_noisy|_i` is *not* near zero (it is the absolute compressed magnitude, and `mask.rescale=true` further inflates magnitudes ~9×, per `inject_matrix` docstring line 211). So coordinates that the true optimizer would barely touch now take a **full-magnitude, fixed-sign step every tick**. This is exactly the regime that sharpens the policy: the logits of the currently-favored tokens get pushed monotonically in one direction, driving probabilities to 0/1 and entropy → 0.
2. **No averaging-out over steps.** Because the sign comes from a β=0.95 EMA (effective memory ≈ 1/(1−β) ≈ 20 ticks), `sign(M)` is **persistent** — it does not re-randomize step to step the way the noisy true-gradient sign would. So the fixed-direction pressure on each coordinate accumulates coherently across many updates instead of self-cancelling. Momentum-like persistence + full magnitude + no regularizer = aggressive, coherent sharpening.

### 3c. Direct quantitative evidence: rel_change ≈ √2
The hook logs `rel_change = ||G_corr − G_noisy|| / ||G_noisy||` per matrix (`spectral_filter.py:310`). Over **1 260 warm samples (step ≥ 5)** the distribution is sharply peaked at the median **1.414 = √2** (min 0.57, max 1.89, mean 1.404; mode bucket 1.4 holds 729/1260 = 58 %).

This is the signature of the sign operation. Decompose by coordinate: where `sign(M)=sign(G_noisy)` the coordinate is unchanged; where they disagree, `G_corr_i − G_noisy_i = −2·G_noisy_i`. If a magnitude-weighted fraction `f` of coordinates disagree, then `||G_corr − G_noisy||² = 4f·||G_noisy||²`, i.e. **rel_change = 2√f**. rel_change = 1.414 ⇒ **f ≈ 0.5**: the stale anchor sign disagrees with the live compressed-gradient sign on **~half the (magnitude-weighted) coordinates, every step**. So roughly half of every parameter matrix's gradient has its sign flipped to the stale-anchor sign and its magnitude preserved — a maximally disruptive, near-orthogonal rewrite of the update direction, applied every tick. (Cold steps 1–2 log rel_change = 0.000000 — the cold-M guard at `spectral_filter.py:296` correctly no-ops before M warms.)

### 3d. The isolation argument (the "we never saw this before")
- **coldM fallback ⇒ merger ON only from step 3.** `merger_coldM_fallbacks` is **196 (= all targeted matrices) at steps 1–2**, then **0 from step 3 onward** (`[comm_eff][merger]` lines + the per-step table column `coldM`). Steps 1–2 are therefore byte-equivalent to plain PowerSGD-compressed GRPO (`G_corr = G_noisy`), and entropy is flat ~5.7. **The monotonic entropy descent begins exactly when M warms** (step 4: 5.75→3.52; first anchor refresh fires at step 5 per `[comm_eff][EXP-12] anchor refresh step=5`). This temporally pins the collapse to the merger turning on.
- **The 4 finished reference runs with the anchor/merger OFF do not collapse.** All four have `actor/comm_eff/anchor_backwards = 0` and converge monotonically on the *identical* model/data/no-KL/no-entropy surface:

  | run | anchor_backwards | val trajectory | val@50 |
  |---|---|---|---|
  | `5e2jpho9` (dense) | 0 | 0.732→0.738→0.742→0.748→**0.754** | 0.7536 |
  | `kqozxfr0` | 0 | 0.732 (s25) → **0.744** (s50) | 0.7437 |
  | `oquyeic3` | 0 | 0.710 (s25) → **0.742** (s50) | 0.7415 |
  | `3yxzzwn3` | 0 | 0.720 (s25) → **0.738** (s50) | 0.7384 |

  Every reference's val **rises** from step 25 to step 50. The α=0 arm's *training* reward instead **peaks at step 28 (0.787) and falls to 0.318 by step 45**. The only experimental difference is the active α=0 merger. **This isolates the α=0 `|G|·sign(M)` merger as the differentiator.**

---

## 4. Per-hypothesis verdicts

### H1 — α=0 `|G|·sign(M)` is magnitude-preserving sign-SGD with persistent EMA signs. **SUPPORTED (root cause).**
See §3. Quantified: rel_change ≈ √2 ⇒ ~50 % of coordinates sign-flipped every step at full magnitude; β=0.95 ⇒ ~20-tick sign persistence (coherent, non-cancelling pressure); rescale inflates magnitudes ~9×. Entropy onset pinned to M-warming (step 3–4). This is the mechanism.

### H2 — no regularizer to arrest it. **SUPPORTED but INSUFFICIENT alone (permissive condition).**
no-KL/no-entropy GRPO normally relies on (a) the group-relative advantage zeroing-out and (b) the true-gradient geometry (sign cancellation, §3b) for *implicit* regularization. The merger breaks (b). But the 4 references share the no-KL/no-entropy surface and do **not** collapse, so the missing regularizer is only the *permission* for collapse — it is **H1 that supplies the driving force**. With an explicit entropy floor or KL the same merger would likely be held back (see §6 prediction). Verdict: necessary background condition, not the differentiator.

### H3 — length-degeneration feedback loop. **SUPPORTED (amplifier, second-order).**
The timing locks: entropy crosses below ~0.5 at step 30, and **`response_length/mean` explodes precisely there** — 298 (s28) → 593 (s30) → 1406 (s34) → 3408 (s38) → 6864 (s42) → 8611 (s45), with `response_length/clip_ratio` rising 0.00 → **0.46** (nearly half of all rollouts now hit the 16 384 cap). Low entropy ⇒ the policy locks onto repetitive/degenerate continuations that never emit EOS ⇒ very long generations ⇒ wrong/garbled answers ⇒ **reward falls** (0.787 → 0.318) ⇒ within-group reward variance shrinks ⇒ GRPO advantages degrade ⇒ even weaker, noisier learning signal ⇒ further collapse. The reward peak-then-degrade and the length explosion are the same event viewed two ways. This is a consequence of H1 that then feeds back, not an independent cause.

### H4 — stale-sign mismatch. **SUPPORTED (corroborating).**
`sign(M)` is built from `θ_{t−delay_K}` (delay_K=5 ticks) through a β=0.95 EMA, so its sign lags the live policy by ~5–25 ticks. As the policy moves *fast* (the steep entropy descent, steps 25–36), the stale sign increasingly disagrees with the live gradient — consistent with the observed acceleration of collapse exactly in that fast-moving window, and with rel_change holding at √2 (≈50 % disagreement) rather than decaying. Caveat: at α=0 the merger discards the live direction entirely, so even a *perfectly fresh* sign would still be sign-SGD; staleness makes the wrong-direction full-magnitude steps worse but is not separable from H1 in this arm. The `[comm_eff][EXP-25][stale] realized_delay=4 warmup_fallback=True` line at step 5 confirms the stale path is live.

---

## 5. Importance-sampling / rollout-correction analysis (explicitly requested)

The IS-family metrics all move in the **collapse-symptom** direction and are *consistent with* a policy going deterministic — they are diagnostics of the collapse, not its cause:

- **`training/rollout_probs_diff_mean` (the IS gap) SHRINKS 0.843 → 0.045.** This is the mean per-token gap between the **training-recomputed** policy prob and the **rollout (vLLM)** prob on sampled tokens. As the policy sharpens, the training policy assigns ≈1.0 to the tokens it now deterministically prefers, and the rollout policy (a recent snapshot, also sharpening) concentrates on the same tokens, so the gap collapses. **`rollout_probs_diff_max` stays pinned at 1.0** throughout — there are always a few maximally-disagreeing tokens — confirming the *mean* shrink is a concentration effect, not genuine train≈rollout agreement everywhere.
- **`rollout_actor_probs_pearson_corr` RISES 0.021 → 0.35.** Early on, training and rollout per-token prob vectors are nearly uncorrelated (diffuse 5.7-nat distributions); as both sharpen onto the same peaks, their correlation rises. Higher pearson here = *both distributions are collapsing together*, not = healthier sampling.
- **`rollout_corr/kl` (rollout→training KL) DROPS 13.2 → 0.34** and `rollout_corr/rollout_ppl` falls toward ~1: the rollout distribution itself becomes near-deterministic. (`chi2_token`/`ppl_ratio` are noisy/heavy-tailed and not load-bearing here.)
- **`actor/ppo_kl` stays ≈0 and `pg_clipfrac` stays small** for the whole run. This is the key brake observation: with **rollout correction OFF** (`rollout_is=null`, `bypass_mode=false`), `old_log_prob` is **recomputed by the current training policy**, so the PPO ratio is ≈1 by construction at the start of each mini-update — `ppo_kl≈0`, clipfrac small. **PPO clipping therefore provides essentially no brake on the drift**: it can only clip *within* a mini-batch's inner steps relative to a same-policy reference, not against the behaviour/rollout policy that actually generated the data. A true behaviour-policy IS ratio (rollout correction ON) would have produced large, off-1 ratios as the policy diverged from the rollout snapshot, triggering clipping and damping the move.

**Framing (critical):** this rollout-correction-OFF configuration is **shared with all 4 non-collapsing reference runs**, so it cannot be the root cause. It is an **amplifier**: it removes a clipping brake that would otherwise have partially resisted the merger-driven drift. The root cause remains the α=0 merger (§3).

---

## 6. Prediction for the α=0.3 and α=0.5 arms

The merger is `G_corr = α·G_noisy + (1−α)·|G_noisy|·sign(M)`. Increasing α **blends the true (compressed) gradient direction back in**, which (a) restores per-coordinate sign cancellation on the α-weighted part and (b) shrinks the effective magnitude of the fixed-sign term. Concretely, on a coordinate where `sign(M) = −sign(G_noisy)` (a flip), `G_corr_i = α·G_noisy_i − (1−α)·|G_noisy_i| = (2α−1)·|G_noisy_i|·sign(G_noisy_i)`. So:

- **α = 0.5 is the sign-cancellation knee.** At α=0.5 a flipped coordinate contributes `(2·0.5−1)=0` — i.e. coordinates where the stale sign disagrees with the live gradient are **driven to ~0** rather than pushed full-magnitude in the wrong direction. The disruptive term changes sign at α=0.5: for **α > 0.5 the live gradient direction wins** every coordinate (the merger only modulates magnitudes); for **α < 0.5 the stale sign still wins** on flipped coordinates (just with reduced magnitude). I therefore predict **α=0.5 substantially arrests the collapse** — entropy should decay much more slowly and the reward should track the dense reference far more closely; response length should stay bounded (clip_ratio low). It may still under-perform the anchor-OFF references slightly because magnitudes are still distorted, but it should **not** exhibit the catastrophic 0.79→0.32 reward crash.
- **α = 0.3 is intermediate and still on the collapse side of the knee** (`2·0.3−1 = −0.4`): flipped coordinates still take a 40 %-magnitude wrong-direction step. I predict α=0.3 **slows the collapse but does not fully arrest it** — entropy still trends toward 0 but more gradually (reaching the danger zone later than step ~30, perhaps not within 50 steps), reward peaks higher/later and degrades less steeply than α=0. It is a softer version of the same pathology.
- **Monotonic ordering prediction:** collapse severity (entropy decay rate, reward-crash depth, resp-len explosion) should be **α=0 ≫ α=0.3 > α=0.5**, with the qualitative phase transition at α≈0.5. If even α=0.5 collapses, that would implicate the *magnitude* distortion (rescale ~9× + abs) independent of sign, and point to the rescale path rather than just the sign merger.

---

## 7. Recommended next diagnostics / mitigations

**Diagnostics to add (cheap, log-only):**
1. Per-step **entropy slope** and an **absolute entropy floor alarm** (see the standing watch doc). The collapse was *visible by step 10* (entropy 5.7→2.1, −65 %) — we did not need to burn 49 steps.
2. **`response_length/clip_ratio` alarm** — it is a clean, early, monotone collapse proxy (0→0.46) and a direct cost signal (rollouts pinned at the 16 384 cap waste compute).
3. **Reward peak-tracker**: flag when `critic/score/mean` falls > ~10 % below its running max over a window — caught the 0.787→0.32 crash.
4. Log the **per-step rel_change median** as a first-class metric; √2 is the sign-flip fingerprint and tells you instantly the merger is doing maximal-disruption sign rewriting.

**Mitigations (in rough order of preference):**
1. **Raise α (sweep already planned).** Per §6, α≥0.5 should arrest it. This is the cleanest fix because it is the existing swept axis.
2. **Re-enable a regularizer:** a small KL-to-reference (`use_kl_in_reward` or `use_kl_loss` with a small coef) or an **entropy floor / small positive `entropy_coeff`** to resist the sharpening pressure directly. Cheapest insurance for *any* merger arm.
3. **Cap response length** lower (e.g. 4 096 in the diagnostic phase) to break the H3 length-degeneration loop and bound cost while studying the sign mechanism. Pair with a length/repetition penalty.
4. **Clip the merger output**: bound `|G_corr|` per coordinate to the live `|G_noisy|` (or clip the *sign-flipped* term) so the disruptive full-magnitude wrong-direction steps are damped without abandoning the merger.
5. **Re-enable rollout correction (IS ratio against the behaviour policy)** to restore the clipping brake (§5). This will not fix the root cause but will resist the runaway and surface the divergence as a large IS ratio early.

---

## 8. Evidence index (file:line + metric provenance)

- Merger formula & cold-M guard: `verl/workers/comm_eff/spectral_filter.py:268-308`; rel_change definition `:310-321`; merger/coldM logging `:417-433`; anchor EMA `:181-202`.
- Per-step metrics: `research/runs/EXP-25/logs/train.log` (re-extracted; full 49-step table in §2).
- coldM=196→0 at step 3, anchor refresh at step 5, rel_change median √2 over 1260 warm samples: grepped from `train.log` (`[comm_eff][merger]`, `[comm_eff][EXP-12] anchor refresh`, `[comm_eff][EXP-7][spectral] ... rel_change=`).
- α=0 val@25=0.7180; 4 references anchor_backwards=0 with rising val to 0.738–0.754: WandB `shamanework-pl/verl_compression_research` runs `uyrpaftw`, `5e2jpho9`, `kqozxfr0`, `oquyeic3`, `3yxzzwn3`.
- Effective config (no-KL/no-entropy, rollout-correction OFF, delay_K=5, cadence=5, β=0.95, α=0): launch line in `train.log`.
