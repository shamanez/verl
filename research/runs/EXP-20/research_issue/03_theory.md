# EXP-20 — Theory: Why the Projected (Biased) RL Gradient Learns, and What the Clean Step Is Actually For

**Compiled by:** theory-analyst · **Date:** 2026-06-04
**Inputs:** `01_wandb_metrics.md` (raw per-step data, 3 arms) + `02_math_interpretation.md` (the
quantitative decomposition — **its headline is the starting point of this document**) +
`POWERSGD_IMPLEMENTATION.md` + the codec source `verl/workers/comm_eff/powersgd_activation.py`
(verified line-by-line) + `CODE_WALKTHROUGH.md` (FSDP/engine integration + gap list).
**Scope:** mechanism. *Why* the data looks the way it does, grounded in the **actual** implementation
(an exact projector with **no error feedback**, not idealized PowerSGD), plus the improvement levers and
the open research questions a downstream agent will act on. No re-runs; no GitHub posting (that is task #4).

> **The center of gravity (operator's must-answer question):**
> *"Is the ~0.74 just because of the 10 full-grad (clean) steps, or something else?"*
> The interpreter's data already **falsifies** "the 10 clean steps do all the work" (§0/§1 of `02`):
> compressed steps book **57–95 %** of the train-reward gain, reward climbs steeply *between* clean steps,
> and the compressed-step OLS slope (**+0.0148–0.0151/step**) is *steeper* than the clean-step slope
> (**+0.0130–0.0133/step**). This document is the **theory of why the compressed (projected) gradient is a
> useful descent direction** — and what the clean step is actually doing if it is not where the reward is made.

---

## 0. The one-paragraph answer

At each pipeline boundary the codec replaces the activation `M (N×H)` with its projection onto a frozen
orthonormal rank-`r` basis, `M̂ = M Q Qᵀ`, `Q` detached and `M` in-graph. Because `Q` is detached and `QᵀQ=I`,
PyTorch's own backward of those two matmuls is the **exact self-adjoint projector**

```
        dL/dM = (dL/dM̂) · Q Qᵀ ,     P := Q Qᵀ ,   P² = P = Pᵀ .
```

So the gradient that flows upstream is the dense boundary gradient **projected onto the same rank-`r` subspace
`Q`** (no straight-through, no surrogate — verified at `powersgd_activation.py:338-340`). The boundary
activations, and hence that boundary gradient, are empirically **low-rank**: reconstruction error falls to
~2 % at `r=77` and `r=102` and is **flat across `[77,102]`** (`01 §2.2/§2.5`, `02 §2/§2a`). A projection onto
the top-`r` activation subspace therefore keeps ~98 % of the energy, and the **off-subspace component it
discards — `(I−P)g` — is small precisely because `g` is low-rank.** The projected gradient is thus a
**structured biased estimator**: biased (it drops `(I−P)g`), but the bias is tiny and the estimator is
**low-variance** (an energy-preserving projection, not a stochastic zeroing). That is why the compressed
steps carry the learning. The clean step is **not** where reward is made; it is a periodic **full-rank flush**
of whatever off-subspace bias accumulated under the no-error-feedback codec — and because that off-subspace
bias is small (low-rank gradient), the flush corrects little, which is exactly why its reward contribution is
small (clean-step share 4.8–19.6 %, `02 §1a`). **Prediction (central, testable): `clean_cadence` can likely be
relaxed or removed with little loss** — the dense run + a clean-only/clean-cadence ablation gate it.

The rest of this document makes each clause precise.

---

## 1. The projected gradient as a structured biased estimator

### 1.1 The operator and its exact gradient (grounded in the code)

The forward at every boundary block is (`powersgd_activation.py:339-340`):

```
Y   = M @ Q            # (N, r)  — the r coordinates actually transmitted
M̂  = Y @ Qᵀ           # (N, H)  — reconstruction, in-graph through M
```

`Q` is a plain fp32 buffer (`self._basis`, never an `nn.Parameter`, never `requires_grad`), and `M` carries the
autograd graph. Hence there is **no custom `autograd.Function` and no straight-through estimator** (verified;
docstring lines 31-33, code 334-340). The chain rule on the two matmuls gives, exactly,

```
M̂ = M P,   P = Q Qᵀ            ⇒     dL/dM = (dL/dM̂) Pᵀ = (dL/dM̂) P .
```

`P = QQᵀ` is the **orthogonal projector** onto `col(Q)` (idempotent and symmetric because `QᵀQ=I`). Write the
*dense* boundary gradient — the gradient that *would* flow if the codec were off — as `g := dL/dM_dense`. Under
the codec, the upstream activation gradient the rest of the network (the earlier pipeline stages / earlier
blocks) receives is

```
g_codec = g_hat · P ,      where g_hat = dL/dM̂  is the gradient w.r.t. the reconstruction.
```

To first order in the activation perturbation (`M̂ = M` on a clean step; on a compressed step `M̂` differs from
`M` only off-subspace, and that difference is small — §1.3), `g_hat ≈ g`, so

```
g_codec ≈ P g = g − (I−P) g .
```

This is the central object. **The codec replaces the boundary gradient with its projection onto `col(Q)` and
throws away the orthogonal complement `(I−P) g` every step** (no accumulator — §3).

### 1.2 The bias is exactly the off-subspace energy of the gradient

Decompose the dense gradient into its in-subspace and off-subspace parts:

```
g = P g + (I−P) g ,        ‖g‖² = ‖P g‖² + ‖(I−P) g‖²   (orthogonality of P).
```

The codec applies `P g`; the **bias** (the systematic, non-stochastic error of the descent direction) is

```
b := g_codec − g  =  −(I−P) g ,      ‖b‖ = ‖(I−P) g‖ .
```

The relative bias is therefore the **fraction of the gradient's energy that lies off the basis**:

```
‖b‖ / ‖g‖ = ‖(I−P) g‖ / ‖g‖ = sqrt( 1 − ‖P g‖²/‖g‖² ) .
```

There is no error-feedback term to cancel `b` (§3), so this is the actual per-step direction error.

### 1.3 Why the bias is small here: the boundary gradient is low-rank, and `Q` tracks its top subspace

Two facts make `‖(I−P)g‖` small in EXP-20:

1. **`Q` is fit to the activation second moment, which is where the gradient energy concentrates.** The basis
   update is block power iteration on the *activation* Gram matrix `C = Σ MᵀM`: `V += (MᵀM)Q` off-graph, then
   `Q ← orth(V)` post-backward (`powersgd_activation.py:374, 512`; `IMPLEMENTATION §4`). This drives `Q` toward
   the **top-`r` right-singular subspace of the activations** (Eckart–Young-optimal rank-`r` reconstruction of
   `M`). The *measured* surrogate for "how much of `M` lives in `col(Q)`" is
   `reconstruction_rel_error = ‖M − M̂‖/‖M‖ = ‖(I−P)M‖/‖M‖`, which converges to **~0.02** within ~9 steps and
   stays there (`01 §2.2`). So `col(Q)` captures ≈ **98 % of the activation energy**.

2. **The boundary gradient inherits the activations' low-rank structure.** The boundary gradient is the
   upstream activation gradient `dL/dM`; for a GRPO policy-gradient loss it is a sum over tokens of
   advantage-weighted per-token gradient contributions whose row space is spanned by the same hidden-state
   directions that carry the activations (the loss couples to `M` only through the downstream linear/attention
   maps acting on those directions). Empirically the per-layer reconstruction is **<4 % at every depth and the
   depth profile is near-flat** (L24/L3 ≈ 1.6–2.0×, `02 §2b`) — the activation tensor, hence the gradient that
   flows back through it, is low effective rank everywhere. The flat rank–fidelity curve across `[77,102]`
   (`02 §2a`: −25 % rank ⇒ only +10.8 % relative recon error) says the spectrum has a knee **below** 77: the
   top ~77 singular directions already hold essentially all the energy.

Putting these together: `‖(I−P)g‖/‖g‖` is on the order of the activation off-subspace fraction (~2 % in energy,
i.e. **~14 % in norm** — `sqrt(0.02)`), and almost certainly smaller for the gradient than for the raw activation,
because the *advantage-weighted* policy gradient is even more concentrated in the dominant response/answer
directions than the full activation is (§4.3). **The bias of the projected RL gradient is small because the
gradient is low-rank, and it is low-rank because the boundary activations are.**

> **Falsifiable corollary (the metric the verdict says was never logged):** the per-step **cosine between the
> dense and compressed updates** should be `≈ ‖P g‖/‖g‖ = sqrt(1 − recon²) ≳ 0.98` once the basis warms in
> (steps ≥ ~9). This is *exactly* the EXP-20 success criterion #7 that was **unmeasurable because it was never
> instrumented** (`verdict.md:21, 44, 111`). My theory predicts its value; logging it is the direct test (§7,
> Q1, and the improvement-lever instrument in §6).

### 1.4 Step-1 transient: a cold (random) basis is a near-orthogonal projector

At step 1 there is no warmed basis — `Q` is the seeded random orthonormal frame, uncorrelated with the
activation subspace — so `recon ≈ 0.97` (`01 §2.2`), i.e. `‖P M‖/‖M‖ ≈ 0.24`. A random rank-77/102 subspace in
`H=1536` captures only `r/H ≈ 5–7 %` of a *generic* vector's energy in expectation; the projector is then almost
entirely **wrong direction**, which is why the compressed-step grad-norm spikes to **166 (r=102) / 194 (r=77)**
at step 1 (`02 §3c`). This is benign: it lands on a single update at init (reward flat ~0.13 through steps 1–4),
grad-clip + lr 1e-6 absorb it, and the basis has locked in before reward begins climbing (§5). The mask has no
such spike (8.3 at step 1) because a coordinate mask needs no learned basis — a structural difference between the
two codecs, not a defect.

---

## 2. PowerSGD (biased, low-variance) vs the PRF mask (unbiased, high-variance) — the bias–variance contrast

This is the contrast that explains the data, and it is a genuine RL-gradient insight.

### 2.1 The two codecs sit at opposite corners of the bias–variance plane

**PRF mask (per-element dropout, rescaled).** The mask multiplies each activation coordinate by an independent
Bernoulli keep/drop and rescales the survivors by `1/(1−p)` (inverted dropout). It is therefore **unbiased in
expectation**: `E[h · mask/(1−p)] = h`, so `E[g_codec] = g` (no systematic direction error). But the price is
**high variance**: zeroing `p=95 %` of the coordinates and rescaling the surviving 5 % by `1/(1−p)=20×` injects a
per-coordinate variance ∝ `p/(1−p) = 19` into both the activation and its gradient. The gradient it hands the
optimizer is a **large, noisy, in-expectation-correct** vector.

**PowerSGD projection.** `g_codec = P g` is **biased** (it deterministically drops `(I−P)g`, §1.2) but
**low-variance**: a projection is energy-preserving (`‖Pg‖ ≤ ‖g‖`, never amplified) and, with `sync_basis=true`,
`Q` is a single bit-identical consensus codebook across all DP ranks (`q_cross_rank_max_rel_dev = 0.0`, `01 §2.3`,
`02 §2c`), so there is no per-rank stochasticity in the operator either. The gradient it hands the optimizer is a
**small, smooth, slightly-wrong-direction** vector.

| | systematic error (bias) | stochastic error (variance) | grad it produces |
|---|---|---|---|
| **PRF mask** | ~0 (unbiased, rescaled) | **high** (∝ p/(1−p) ≈ 19) | large & noisy |
| **PowerSGD** | small (`(I−P)g`, ~14 % norm) | **low** (energy-preserving, consensus `Q`) | small & smooth |

### 2.2 The grad-norm gap is the direct fingerprint of this contrast

`02 §3a/§3b` measured exactly this:

```
                  clean-step grad   compressed-step grad (steady)   compressed/clean
  mask p=0.95          0.399                ~11.8 (median 10.9)            ~27×
  PowerSGD r=102       0.408                ~1.7  (median 1.57)            ~3.8×
  PowerSGD r=77        0.390                ~2.1  (median 1.87)            ~4.8×
```

The clean-step grad-norm is **~0.4 in all three arms** — necessarily identical, because a clean step bypasses the
codec and computes the *same* dense gradient (modulo the arms' slightly divergent weights). The informative number
is the *compressed* grad: the **mask's is ~6–7× PowerSGD's**. That is the bias–variance contrast made visible —
the mask's `1/(1−p)=20×` rescale of a sparse survivor set produces a ~27× grad-norm inflation; PowerSGD's
energy-preserving projection produces only a modest ~4× inflation over the dense norm (the residual >1× factor is
the warm-up + the fact that the projected gradient is computed on the *compressed-forward* loss, not literally
`P·g_dense`).

### 2.3 The headline RL insight: progress is governed by gradient *direction*, not *magnitude*

**Despite the 6–7× grad-norm difference, the two codecs reach identical accuracy** (mask 0.7384, r=77 0.7415,
r=102 0.7437 — spread 0.53 pp, within RL noise; `02 §5`). Both deliver wildly different *magnitudes* to AdamW yet
the same learning. The reconciliation:

- **lr 1e-6 + grad-clip absorb the magnitude.** With a tiny fixed lr and gradient clipping, a 6–7× scale
  difference is a difference in *effective step length per update*, which the optimizer (and the over-many-steps
  averaging) absorbs. Magnitude is a nuisance variable here, not a driver.
- **What both codecs preserve is the *direction*.** The mask preserves direction *in expectation* (unbiased,
  averaged over its high variance across the 143360 per-microbatch applications); PowerSGD preserves direction
  *per step* (small bias, low variance). Two different routes to "the update points the right way on average,"
  and the policy advances at the same rate either way (the compressed-step OLS slopes are within 2 % of each
  other across arms — `02 §1d`).

**State it crisply:** *in this regime the learning signal lives in the gradient's direction / subspace alignment,
not its norm; a codec is "good" iff it preserves the dominant gradient subspace (high `‖Pg‖/‖g‖`), and a 6–7×
norm difference between two subspace-preserving codecs is immaterial to the policy that emerges.* This is why the
right success metric is the **update cosine** (§7 Q1), not the grad-norm.

---

## 3. What the clean step is actually for (and why it contributes little)

### 3.1 The mechanism: a periodic full-rank flush of accumulated off-subspace bias

This is where the **no-error-feedback** property of the implementation is load-bearing. Classic PowerSGD carries an
**error-feedback accumulator**: the part it cannot send this step, `M − M̂` (equivalently the off-subspace gradient
`(I−P)g`), is *remembered* and *re-injected* next step, so nothing is permanently lost. **This fork implements no
such accumulator** — `maybe_update_basis` calls `_reset_sketch()` and the discarded `M − M̂` is dropped every step
(verified: `powersgd_activation.py:516, 641-644`; the only state carried across steps is the *basis* `Q`, not a
residual). I confirmed there is no residual buffer anywhere in the compressor.

Consequence: on each compressed step the optimizer never sees `(I−P)g`. The component of the true gradient lying
*outside* `col(Q)` is silently set to zero. Over a run of compressed steps, a **bias can accumulate in those
off-subspace directions** — the policy is descending on the rank-`r`-restricted loss, and any curvature/signal that
lives off `col(Q)` is systematically ignored. The **clean step is the correction**: every `k`-th step both codecs
are bypassed (`mask_active = not clean_step`, `engine_workers.py:917`; `IMPLEMENTATION §7`), the actor-train forward
and the old-logprob recompute are byte-identical dense, and **AdamW refreshes its moments on the full-rank gradient**
— including the off-subspace component the compressed steps dropped. It is a periodic **full-rank flush** of the
no-EF off-subspace bias (and incidentally a grad-norm reset: grad collapses ~27× to ~0.4, the visible heartbeat).
On a clean step `Q` is **held, not updated** (`maybe_update_basis(is_clean_step=True)` returns early, line 458-462),
so the dense step does not also perturb the basis.

### 3.2 Why the flush corrects little — and the data that says so

The flush corrects the *accumulated off-subspace bias*. But §1.2–1.3 showed the *per-step* off-subspace gradient
`(I−P)g` is **small** (the gradient is low-rank). A small per-step bias accumulates into a small drift, so the
full-rank flush has little to correct — hence its **small reward contribution**:

- **Clean-step reward share = 4.8–19.6 %** (strict attribution A, `02 §1a`); the per-clean-step Δreward is small
  and frequently *negative* (e.g. mask clean Δ = {5:−0.023, 10:+0.090, …, 45:−0.023, 50:+0.047}).
- **Compressed steps book 57–95 %** of the gain; reward rises monotonically *within* almost every inter-clean
  segment, especially the steep early ones (`02 §1c`); the compressed-step slope is *steeper* than the clean-step
  slope (`02 §1d`).

This is the empirical signature of a *biased-but-aligned* compressed gradient whose bias is too small for the
periodic debiaser to matter much for reward. **The clean step is doing real work (flushing off-subspace bias + a
grad-norm reset + the step-5 re-alignment in §4.2), but that work is small because the bias it flushes is small.**

### 3.3 The central, testable prediction

If the clean step's *raison d'être* is flushing an off-subspace bias that is empirically small, then **removing or
relaxing `clean_cadence` should cost little accuracy** — the compressed steps were already carrying the learning.
This is the single most important falsifiable consequence of the theory. It is gated by:

- the in-flight **dense run** (the absolute ceiling — does pure-compressed even need to reach it?), and
- a **clean-only / clean-cadence ablation**: run the same config at `clean_cadence ∈ {5, 10, 25, ∞(never)}` and a
  pure-10-dense-step control. The theory predicts a graceful, shallow degradation as cadence lengthens, and that
  `clean_cadence=∞` (pure compressed, no EF) loses only a small amount — bounded by the accumulated off-subspace
  drift, which §1.3 argues is small but **nonzero** (so some floor of degradation is expected without EF; §6.1).

> **Contrast with the PRF mask.** For the *mask*, the clean step is far more load-bearing: prior project results
> show pure-masked GRPO **stalls** without clean steps (memory `exp16-cleanstep-convergence`: pure-masked
> 0.13→0.15, clean_cadence=4 unlocks 0.13→0.62) — because the mask's bias, though zero *in expectation*, has
> **enormous variance** that the optimizer cannot average away fast enough at this lr, and the clean step injects a
> low-variance full-rank anchor. So the *same* clean-cadence knob plays *different* roles for the two codecs:
> a **variance reset** for the mask, a **bias flush** for PowerSGD. PowerSGD's low variance is exactly why it
> should tolerate clean-step removal far better than the mask — a sharp, testable cross-codec prediction.

---

## 4. RL-specific subtleties — why a biased gradient is tolerable in *on-policy* GRPO here

### 4.1 Frozen-`Q`-within-step ⇒ the importance ratio ρ ≈ 1 (the codec does not break on-policyness)

GRPO forms `ρ = exp(logπ_new − logπ_old)`, where `logπ_old` is an old-policy recompute (`compute_log_prob`) and
`logπ_new` the actor-train forward (`update_actor`). The implementation **freezes `Q` for the entire global step**:
the basis update runs only in `update_actor`'s `finally:` block *after* all PPO micro-batch forwards/backwards
(`engine_workers.py:928-943`; `maybe_update_basis` advances `Q_t → Q_{t+1}` for the *next* step), and
`compress_recompute=true` stamps the *same* `Q_t` onto the old-logprob recompute (`engine_workers.py:697-706`;
`IMPLEMENTATION §6, §12`). Therefore **both paired forwards apply the identical operator `Q_t Q_tᵀ`**.

The consequence is subtle and important: the bias of the projector shifts **both** `logπ_old` and `logπ_new`
**the same way**. The importance ratio is a *ratio*, so a common multiplicative/additive distortion of the two
log-probs largely **cancels** — `ρ ≈ 1` at step 0 (no weight change), and `ρ` stays well-behaved thereafter because
the only thing that differs between the two forwards is the legitimate one-update weight change, not a drifting
codec. The empirical confirmation is the `rollout_actor_probs_pearson_corr`: it **snaps to 0.999 by step 5 and
stays there, identically across all three arms** (`01 §2.4`, `02 §4`). The codec does **not** corrupt the on-policy
assumption after warm-up — the compressed actor's log-probs agree with the (dense, uncompressed) vLLM rollout
log-probs to 3–4 decimals.

### 4.2 The step-5 snap is the *warm-up/policy-sync* clean step, not a codec effect

The Pearson correlation is ~0 at steps 1–2 (the freshly-loaded actor's forward is uncorrelated with the rollout
policy — also the cause of the anomalous `val@0 ≈ 0.08`) and snaps to 0.999 at **step 5, the first clean step**
(`02 §4`). Two readings combine:

- It is **codec-independent** — identical in all three arms — so it is a property of the policy/rollout warm-up,
  *not* of compression (`02 §4` rules it out as an inter-arm differentiator).
- That the snap *coincides with the first clean step* is consistent with the clean step's third role: a **full-rank
  re-alignment** that lets the actor's forward lock onto the rollout policy. (It is not exclusively the clean step —
  the compressed steps 1–4 also move the policy — but the first dense forward is where the train↔inference
  consistency completes.) This is a real, if minor, additional function of the clean step beyond the bias flush.

### 4.3 GRPO's advantage weighting *sharpens* the low-rank structure

GRPO's policy gradient is an **advantage-weighted** sum of per-token score-function gradients,
`g_pg = Σ_tokens A_token ∇ logπ(token)`, with group-normalized advantages `A`. Two implications for the projector:

- The advantage weighting concentrates gradient energy on the **high-|A| tokens** — the decisive
  answer/reasoning tokens that distinguish high- from low-reward rollouts. These are a *low-dimensional, highly
  structured* set of directions in hidden space, so the advantage-weighted gradient is plausibly **even
  lower-rank than the raw activation** — which is why §1.3's bound (`recon ~0.02`) is an *over-estimate* of the
  gradient bias, and why early reward gain happens while recon is still 17–39 % (§5): the projector already
  captures the *advantage-relevant* directions before it captures the full activation spectrum.
- Because advantages come from the **uncompressed vLLM rollouts** (`IMPLEMENTATION §6`), the *reward signal itself*
  is never compressed — only the *policy-gradient pathway* through `M̂` is. The codec biases *how the gradient is
  routed back*, not *what the policy is being rewarded for*. This is why a biased gradient is far safer here than,
  say, a biased *reward* would be.

### 4.4 What would break if `Q` drifted *within* a step

If the basis advanced *between* the old-logprob recompute and the actor-train forward, the two forwards would
apply **different** projectors `Q_old Q_oldᵀ ≠ Q_new Q_newᵀ`, injecting a spurious component into
`logπ_new − logπ_old` that is **not** a real policy change. That would make `ρ ≠ 1` for a non-policy reason and
**corrupt the GRPO objective** (the importance ratio would partly measure codec drift). The frozen-Q rule
(`IMPLEMENTATION §6`, INF-17) is precisely the guard against this; the `q_cross_rank_max_rel_dev = 0.0` invariant
(`02 §2c`) additionally guarantees the *same* operator across DP ranks, so the objective is consistent across the
data-parallel group too. **This is the load-bearing correctness property that lets an on-policy method tolerate a
biased gradient: the bias is applied *consistently* to numerator and denominator of ρ.**

---

## 5. Two timescales: basis learning vs policy learning

The data shows **two well-separated dynamical timescales**:

- **Basis learning (fast):** `recon` falls 0.97 → <0.025 in **~9 steps** and is flat thereafter (`01 §2.2`,
  `02 §2`). The warm-started block power iteration locks onto the top-`r` activation subspace in ~8 refreshes; the
  subspace is then **slowly varying**, so one power-iteration/step tracks it (residual ~0.02 for 41 more steps).
- **Policy learning (slower):** reward keeps climbing well past the basis-lock — 73–82 % of the gain is in the
  first 20 steps but it continues to ~step 30+ (`02 §1e`), and val rises 0.71→0.74 from step 25→50.

Two consequences:

1. **Accurate reconstruction is *not* a precondition for useful descent.** Reward is already climbing at steps 3–4
   while `recon = 17–39 %` (`02 §2`, end). The projector need only capture the *advantage-relevant* directions
   (§4.3), which it acquires before it nails the full activation spectrum — the basis is "good enough to learn from"
   long before it is "accurate."
2. **The basis nearly stops moving while the policy keeps learning.** Once `recon` plateaus (~step 9), `Q` is
   effectively a *fixed* low-rank codebook and the remaining ~40 steps are policy descent through a static
   projector. This decoupling is *why* the no-EF bias is well-controlled in steady state: a fixed `col(Q)` that
   captures 98 % of a slowly-varying activation subspace means the *same* small `(I−P)g` is dropped each step, so
   the drift is slow and bounded — and a single clean step suffices to flush it. (If the subspace drifted fast,
   the dropped component would rotate and the no-EF bias would be far more dangerous.)

---

## 6. Improvement levers — mechanism + EXP-20 evidence + how to test (prioritized)

### 6.1 ★ Error feedback / residual accumulation (highest leverage — currently ABSENT)

**Mechanism.** Classic PowerSGD's defining trick is an **error-feedback (EF) accumulator**: keep a residual
buffer `e`, and each step compress `g + e` instead of `g`, then set `e ← (g + e) − P(g+e)` (the part not sent).
The off-subspace component is never discarded — it is *remembered and re-injected next step*, so over time **every
direction is eventually applied** and the estimator becomes **asymptotically unbiased** (the bias telescopes to
zero in the running sum). For our activation codec the analogue is to accumulate the discarded
`R := M − M̂ = (I−P)M` (or, gradient-side, `(I−P)g`) per boundary and add it back into the next step's
pre-projection input.

**EXP-20 evidence that motivates it.** The codec **explicitly has no residual buffer** — `_reset_sketch()` drops
everything but `Q` each step (`powersgd_activation.py:516, 641-644`). The *only* mechanism currently flushing the
off-subspace bias is the **clean step** (§3). EF would make the off-subspace gradient be applied *continuously*
rather than in a periodic dense burst, which **could remove the need for clean steps entirely** — converting the
"10 dense steps" from a structural requirement into an optional accelerator.

**How to test.** Add an EF buffer to the compressor (one `(N,H)` or `(H,r)` residual per boundary), run at
`clean_cadence=∞` (no clean steps), and compare to (a) the current no-EF + clean_cadence=5 arm and (b) no-EF +
clean_cadence=∞. **Predictions:** EF + no-clean ≈ current accuracy (EF substitutes for the flush); no-EF + no-clean
degrades by the accumulated off-subspace drift (small but nonzero — the floor §3.3 anticipates). Direct internal
metric: the **off-subspace energy `‖(I−P)g‖/‖g‖`** should trend to ~0 under EF (the residual gets paid down) but
stay flat (~0.14) without it.

> **Caveats specific to this setting (flag, don't hand-wave):** (i) EF is an *activation/gradient* residual across
> a boundary, not the usual DP-gradient residual — its memory cost is `O(N·H)` per boundary per micro-batch
> (potentially large at `N` = packed 16K-token sequences), so a *gradient-side* residual `(I−P)g` (`H×r`-ish after
> projection bookkeeping, or full `N×H`) needs a memory budget check. (ii) EF interacts with the **frozen-Q /
> ρ≈1** invariant (§4.1): the residual must be applied *identically* to the old-logprob recompute and the
> actor-train forward, or it reintroduces a within-step inconsistency that breaks `ρ≈1`. (iii) EF + a *moving*
> basis needs care: the residual was computed in the *old* `col(Q)`; re-injecting it after `Q` rotates mixes
> frames. Simplest correct version: EF only in steady state (basis ~fixed, steps ≥ ~9), or rotate the residual
> into the new basis. **This is the highest-leverage idea and also the one with the most implementation
> subtlety — it deserves its own experiment.**

### 6.2 Relax / remove `clean_cadence` (directly implied by §1/§3)

**Mechanism + evidence.** Already argued (§3.3): compressed steps carry 57–95 % of the gain, the bias the clean
step flushes is small, so a longer cadence should cost little. **How to test:** sweep
`clean_cadence ∈ {5, 10, 25, ∞}` at fixed everything-else; plot val@50 vs cadence. **What each outcome means:**
flat curve ⇒ clean steps are near-free padding (compression is essentially self-sufficient — strongest possible
version of the headline); steep falloff ⇒ the bias flush *is* load-bearing and §1.3's "bias is small" is
overstated for the *cumulative* (not per-step) drift, which would in turn *motivate EF* (§6.1) as the continuous
replacement. **The dense run** sets the ceiling these are measured against. *(Note: prior project work used
`clean_cadence=20` for the mask and still reached parity — memory `m4-curvematch-issue18`,
`exp17-core-clean-cadence-diagnostic` — weak prior evidence that long cadence is tolerable, but that was the mask,
whose clean step plays the *variance-reset* role, §3.3, so it does not directly transfer.)*

### 6.3 Rank below 77 — find the true knee with a downward sweep

**Mechanism + evidence.** The rank–fidelity curve is **flat across `[77,102]`** (`02 §2a`: −25 % rank ⇒ +10.8 %
relative recon, +0.22 pp accuracy loss only), so **the knee is below 77** — r=77 is already past it, and r=102 is
wasted budget. **How to test:** sweep `r ∈ {16, 32, 48, 64, 77}` (downward), track steady `recon`, the update
cosine (§6.6), and val@50. **Prediction:** accuracy holds until some `r* < 77` where `recon` starts climbing
steeply and the update cosine drops below ~0.9; that `r*` is the true minimal byte budget. This converts EXP-20's
"r=77 matches the mask" into "the *real* floor is `r* ≪ 77`," a strictly stronger compression claim.

### 6.4 Per-layer / depth-adaptive rank at fixed total budget

**Mechanism + evidence.** The deepest layer **L24 is higher effective rank** — its recon is essentially
rank-*independent* (0.0376 at r=102 vs 0.0380 at r=77, a +1.1 % gap, `02 §2b`), meaning neither rank resolves it,
while the **extra 25 ranks of r=102 land on shallow layers where they are not needed** (L3 gap 0.0184 vs 0.0233 is
the largest). So a uniform rank **mis-allocates budget**: it over-provisions shallow layers and under-provisions
L24. **How to test:** at a *fixed total* `Σ_layers r_layer`, allocate more rank to deep boundaries (e.g. `r_layer`
∝ that layer's singular-value tail mass, or simply a higher rank for the last 1–2 boundaries) and compare val@50 +
per-layer recon to uniform-rank at the same total budget. **Prediction:** depth-adaptive rank lowers the *max*
per-layer recon (currently bottlenecked by L24) at no extra total budget — i.e. a better fidelity/byte frontier.
The codec already keys the basis per-`layer_idx` (`self._basis: dict[int, Tensor]`), so per-layer rank is a small
extension.

### 6.5 Adaptive / triggered clean step (replace the fixed cadence with a drift detector)

**Mechanism + evidence.** A *fixed* cadence fires dense steps on a clock regardless of need; §3.2 showed most fire
when there is little to flush (clean Δreward ≈ 0). **Better:** fire a clean step **only when a drift signal crosses
a threshold** — candidates: (a) `reconstruction_rel_error` spikes (basis lost the subspace, e.g. after a
distribution shift); (b) the **off-subspace energy `‖(I−P)g‖/‖g‖`** exceeds a budget; (c) the **dense-vs-compressed
update cosine** (§6.6) drops below a threshold. **How to test:** run with a triggered clean step vs the fixed
cadence at matched *average* clean-step count; compare val@50 and total clean-step budget. **Prediction:** the
adaptive policy reaches the same accuracy with *fewer* clean steps (cheaper) by spending them only when the bias is
actually large — most useful in non-stationary settings (curriculum, longer runs) where a fixed cadence is either
wasteful or too sparse. This is the natural bridge between §6.2 (relax) and §6.1 (EF): EF removes the *average*
need for clean steps; a trigger keeps a cheap safety valve for the rare large-drift event.

### 6.6 ★ Instrument the dense-vs-compressed update cosine (the success criterion that was never logged)

**Mechanism + evidence.** This is the **direct measurement of the alignment my entire §1–§2 argues for**, and it is
exactly EXP-20 success criterion #7, which was **UNMEASURABLE because it was never instrumented** (`verdict.md:21,
44, 111` — "the dense-vs-compressed update cosine was NOT instrumented in this run"). My theory *predicts* its value
(§1.3 corollary: `cos ≈ sqrt(1 − recon²) ≳ 0.98` post-warm-up). **How to instrument:** on a periodic step, compute
*both* the compressed update and the dense update on the *same* minibatch (one extra dense forward/backward, like a
clean step but without applying it) and log `cos(Δθ_compressed, Δθ_dense)` per layer and globally. **Why it is the
right metric:** §2.3 established that progress is **direction-driven**; the cosine is the scalar that *is* that
direction agreement. It would (a) confirm/falsify the projected-gradient theory directly, (b) provide the trigger
signal for §6.5, and (c) make the head-to-head machine-checkable (the verdict explicitly asks for this before any
launcher promotion). **Lowest-effort, highest-diagnostic-value lever — do this first.**

### 6.7 (Secondary) Optimizer-state / momentum handling across the clean↔compressed boundary

**Mechanism.** AdamW's first/second moments are accumulated across a *mixed* sequence of biased (compressed,
~small-norm) and unbiased (clean, the lone large-direction full-rank) gradients. The clean step injects a
full-rank gradient of *different statistics* (27× larger for the mask, ~4× for PowerSGD) into the *same* running
moment estimates, which can momentarily mis-scale the per-coordinate adaptive lr right after a clean step.
**Evidence (weak/observational):** grad-norm collapses to ~0.4 on clean steps then jumps back; the post-clean
reward Δ is noisy (`02 §1a`). **How to test:** (i) reset/rescale the second-moment estimate at clean steps, or
(ii) maintain separate moment buffers for the off-subspace directions only on clean steps. **Lower priority** —
the data shows AdamW + lr 1e-6 already absorbs the magnitude mismatch (§2.3), so this is a refinement, not a fix;
worth a quick ablation only if §6.2's cadence sweep reveals clean-step instability.

---

## 7. The dense run (in flight) — what each outcome would mean, theoretically

A same-config **DENSE** GSM8K 50-step run (`ce_dense_50s_gsm8k`, comm-eff OFF, `test_freq=10`) is launching in
parallel; numbers are **TBD**. Per `01 §4`/`02 §7`, **no usable ≥50-step dense GSM8K trajectory currently exists**
(the only dense run is Big-Math/MATH-eval `lwl9yk4y`; the GSM8K dense runs are empty 2-step probes; the prose
ceiling ≈0.741 is a *different-config* EXP-17-era figure). So the dense run supplies the **absolute ceiling** the
internal codec-vs-codec decomposition cannot. **Crucially: the dense ceiling cannot overturn §1–§4** — those are
*internal* to each arm (compressed-vs-clean within the same run) and are already settled by the EXP-20 data. The
dense run only calibrates *how close to the ceiling* compression gets.

**Comparison 1 — dense@10 vs compressed@50** (do 10 full grads alone already reach ~0.74?):
- **DENSE val@10 = `<TBD — append after dense run>`** vs compressed val@50 ≈ 0.738–0.744.
- *If DENSE@10 ≳ 0.73:* 10 dense gradients alone nearly reach the ceiling ⇒ a strong-form reading where the
  *quantity* of dense signal (not its interleaving with compression) dominates. **But note the confound the
  interpreter flagged:** EXP-20's 10 clean steps are *interleaved with and build on* 40 compressed updates, so a
  true 10-step *pure*-dense control is the honest test of this — not the EXP-20 clean steps. Even if true, it would
  **not** contradict §1's *internal* finding (compressed steps carry 57–95 % of *this run's* gain); it would say
  "fewer total dense steps could also get there," a *different* claim. **`<fill after dense run>`**
- *If DENSE@10 ≪ 0.73:* the 40 compressed steps materially advance the policy (consistent with §1–§3). **`<fill>`**

**Comparison 2 — dense@50 vs compressed@50** (the parity ceiling / is compression ~free?):
- **DENSE val@50 = `<TBD>`**; gaps `DENSE − {mask 0.7384, r77 0.7415, r102 0.7437}` = `<TBD>`.
- *If |gap| ≲ 0.5–1 pp* (the inter-arm spread is 0.53 pp): **compression is accuracy-free at this budget** — the
  projected gradient is not merely "useful" but **ceiling-matching**, the strongest version of the §1–§2 thesis,
  and it would mean the small off-subspace bias (§1.2) costs essentially nothing in final accuracy. *If gap ≫ 1 pp:*
  a real compression tax exists, which would **directly motivate EF (§6.1)** as the mechanism to recover it (the
  tax would be the un-flushed off-subspace bias EF is designed to pay down). **`<fill after dense run>`**

**Comparison 3 — dense's own post-step-10 slope** (shape, not just endpoints):
- **DENSE val@{10,20,30,40,50} = `<TBD,…>`** ⇒ dense post-10 slope = `<TBD>`/step; compare to the compressed
  arms' val@25→50 slope (≈ +0.0007/step for mask) and the compressed train-reward late slope (~flat after step 30).
- *If dense shares the steep-to-~step-15 then-flat diminishing-returns shape:* compression preserves the **learning
  *dynamics***, not just the endpoint — i.e. the projected gradient reproduces the dense *trajectory*, the
  strongest curve-match. *If dense is materially steeper late:* compression slows late-stage learning (an
  off-subspace-bias drag that grows as the easy in-subspace signal is exhausted — again an EF target). **`<fill>`**

**Settled vs awaiting-dense (be explicit):**
- **Already settled by internal codec-vs-codec data (no dense needed):** the compressed gradient is a useful,
  biased-but-aligned descent direction (§1); the bias–variance contrast and direction-not-magnitude insight (§2);
  the clean-step-as-small-bias-flush mechanism and the 4.8–19.6 % clean share (§3); ρ≈1 / on-policy safety (§4);
  the two-timescale basis-vs-policy decoupling (§5); the budget/fidelity conclusions and r=77-past-the-knee (§6.3).
- **Awaiting the dense run:** the *absolute* parity gap and whether compression is accuracy-*free* vs accuracy-*cheap*
  (Comparison 2); whether ~10 dense steps alone suffice (Comparison 1); whether the *learning dynamics* match
  (Comparison 3). None of these can overturn the internal decomposition; they calibrate the ceiling.

---

## 8. Open research questions (falsifiable; each with the metric/experiment that answers it)

The final issue is **interactive** — a downstream research agent will read the code + all WandB metrics and derive
its own insights. These are the sharp, falsifiable questions to hand it, each paired with the decisive measurement.

1. **Is the projected gradient actually aligned with the dense gradient?**
   *Predicted:* `cos(Δθ_compressed, Δθ_dense) ≈ sqrt(1 − recon²) ≳ 0.98` for steps ≥ ~9 (§1.3).
   *Measure:* instrument the dense-vs-compressed update cosine per-layer + global (§6.6) — the criterion #7 that was
   never logged (`verdict.md:21`). *Falsified if* the cosine is materially < the recon-implied bound (⇒ `g_hat ≉ g`,
   i.e. the compressed-forward loss is not a first-order-faithful proxy for the dense loss).

2. **Can error feedback remove the clean step?**
   *Predicted:* EF + `clean_cadence=∞` ≈ current accuracy; no-EF + `clean_cadence=∞` degrades by a small but
   nonzero floor (§6.1, §3.3). *Measure:* the {EF, no-EF} × {clean, no-clean} 2×2, tracking val@50 and the
   off-subspace energy `‖(I−P)g‖/‖g‖` (should →0 under EF). *Falsified if* EF+no-clean underperforms current
   (⇒ the clean step does more than flush off-subspace bias — e.g. its variance-reset / re-alignment roles, §3.3/§4.2,
   matter more than the theory assigns).

3. **How far can `clean_cadence` relax before accuracy falls?**
   *Predicted:* shallow degradation; PowerSGD tolerates removal far better than the mask (low variance, §3.3).
   *Measure:* `clean_cadence ∈ {5,10,25,∞}` for both PowerSGD and the mask; val@50 vs cadence. *Falsified if*
   PowerSGD falls off as steeply as the mask (⇒ its clean step is also a variance reset, not just a bias flush).

4. **Where is the true rank knee?**
   *Predicted:* knee at some `r* < 77`; accuracy holds until `recon` climbs / cosine drops (§6.3).
   *Measure:* `r ∈ {16,32,48,64,77}` downward; steady recon + cosine + val@50. *Falsified if* accuracy degrades
   smoothly from r=77 with no plateau (⇒ no sharp knee; every bit of rank buys accuracy — contra the flat-curve
   reading of `02 §2a`).

5. **Does depth-adaptive rank beat uniform rank at fixed total budget?**
   *Predicted:* yes — moving budget from over-provisioned shallow layers to the higher-effective-rank L24 lowers the
   max per-layer recon at equal total bytes (§6.4). *Measure:* depth-allocated vs uniform `Σ r_layer`; per-layer
   recon + val@50. *Falsified if* uniform matches or beats adaptive (⇒ deep-layer recon is not the binding
   constraint on accuracy, only on reconstruction).

6. **Is the basis subspace actually slowly varying (justifying one power-iteration/step + a fixed steady codebook)?**
   *Predicted:* yes — the principal-angle drift between `col(Q_t)` and `col(Q_{t+1})` is small after warm-up (§5).
   *Measure:* log per-step principal angles / `‖Q_{t+1}Q_{t+1}ᵀ − Q_t Q_tᵀ‖` after step 9. *Falsified if* the
   subspace rotates fast (⇒ one power-iteration/step under-tracks it, the no-EF dropped component rotates, and the
   bias is more dangerous than §5 claims — would argue for `update_cadence>1` power iterations or EF-with-rotation).

7. **Does the advantage weighting make the gradient lower-rank than the activation (RL-specific)?**
   *Predicted:* the advantage-weighted policy gradient is *more* concentrated than the raw activation (§4.3), so the
   gradient's off-subspace energy < the activation's recon error. *Measure:* compute the recon error of the
   *gradient* `‖(I−P)g‖/‖g‖` directly (the codec currently logs only the *activation* recon) and compare to
   `recon`. *Falsified if* gradient off-subspace energy ≥ activation recon (⇒ advantage weighting does not sharpen
   the low-rank structure; the projector's safety margin is smaller than §1.3 assumes).

8. **Is compression accuracy-*free* or accuracy-*cheap* against the true dense ceiling?**
   *Predicted:* |gap| ≲ ~0.5–1 pp (within inter-arm spread) ⇒ free (§7, Comparison 2). *Measure:* the in-flight
   dense run, val@50. *Falsified if* gap ≫ 1 pp (⇒ a real tax ⇒ EF becomes the lever to close it).

---

## 9. Summary for task #4 (the issue-author)

**The thesis, in one line:** *The compressed step learns because projecting the boundary gradient onto the top-`r`
activation subspace is a low-variance, small-bias descent direction (the gradient is low-rank, so `(I−P)g` is
tiny); the clean step is a periodic full-rank flush of the accumulated off-subspace bias that the no-error-feedback
codec drops, and it contributes little to reward precisely because that bias is small — so `clean_cadence` can
likely be relaxed or removed, and error feedback is the principled way to remove it.*

**Top 3 improvement levers:**
1. **Instrument the dense-vs-compressed update cosine (§6.6)** — lowest effort, directly tests the whole theory,
   was the one unmeasured EXP-20 success criterion (`verdict.md`), and is the trigger signal for adaptive cleans.
2. **Error feedback / residual accumulation (§6.1)** — highest leverage; could remove the clean step entirely by
   making the off-subspace gradient be applied continuously instead of in periodic dense bursts. Carries real
   implementation subtlety (memory at 16K tokens; must preserve the frozen-Q/ρ≈1 invariant; basis-rotation of the
   residual) — deserves its own experiment.
3. **Downward rank sweep + relaxed `clean_cadence` (§6.3, §6.2)** — the knee is *below* 77 and the clean step is
   near-free for PowerSGD; these two cheap sweeps locate the true minimal byte budget and the true clean-step need.

**The central question, answered:** *No, the ~0.74 is not "just the 10 clean steps."* The compressed (projected)
gradients carry 57–95 % of the train-reward gain and advance the policy at a slope ≥ the clean steps'; the clean
step is a small-bias flush, not the engine of learning. **Settled by the internal data; the in-flight dense run only
calibrates the absolute ceiling, which cannot overturn this decomposition.**
