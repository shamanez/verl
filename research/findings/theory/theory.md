# theory.md — what the masked GRPO update actually optimizes, and why it learns GSM8K but stalls Big-Math

**Author:** theorist (comm-eff-theory team)
**Scope:** answers CONTEXT questions A, B, C with explicit mechanism + math.
**Method under study:** per-(token, channel) Bernoulli activation mask at 7 pipeline
boundaries `[3,7,11,15,18,21,24]`, drop prob `p=0.9`, **rescale ON** (constant
inverted-dropout `1/(1-p)=10`), `mask_recompute=true`, anchor+spectral OFF,
`clean_cadence=20`. Vanilla GRPO, no-KL no-entropy. Qwen2.5-1.5B-Instruct.

Every claim is tagged **[FACT]** (provable from code/logs/standard theory),
**[INFERENCE]** (follows from FACT + a stated assumption), or **[SPECULATION]**
(plausible, not established here). Code line refs are to this checkout.

---

## 0. Notation and the three forwards (grounding)

There are three distinct conditional distributions in play. Keeping them apart is
the whole game.

- **π_θ** — the *deployed policy*: the unmasked network with weights θ. This is the
  object we actually care about; validation runs it (`path_tag` ∉ eligible ⇒ mask
  hook no-ops, `activation_mask.py:289-302`). **[FACT]**
- **π̃_θ** — the *masked training-time forward*: the SAME weights θ, but with
  `h_tilde = h ⊙ m / (1-p)` injected at the 7 boundary layers
  (`activation_mask.py:349`, `_rescale_gain=10` at `:257`). `m` is a fresh
  per-(token, channel) Bernoulli(1-p) draw keyed on `(sample_id, position_id,
  layer, global_step, channel)` (`prf_token_mask`, `:168-214`). This is a
  *different conditional distribution over the response tokens* than π_θ. **[FACT]**
- **μ** — the *vLLM rollout sampler*: a third forward (fp16/bf16 vLLM kernels, no
  mask). `calculate_log_probs=True` records `log μ` for the `rollout_corr/*`
  diagnostics only; it is NOT in the loss. **[FACT]**

The GRPO actor loss (`compute_policy_loss_vanilla`, `core_algos.py:1279-1368`) is
the clipped surrogate

  L(θ) = − E_t [ A_t · clip( r_t(θ), 1−ε, 1+ε ) ],   r_t(θ) = exp( logπ_new(a_t) − logπ_old(a_t) ).

The load-bearing implementation fact: because `mask_recompute=true`, **both**
`old_log_prob` (the `compute_log_prob` recompute, `path_tag="old_logprob"`,
mask-eligible) **and** `log_prob` (the train forward, `path_tag="train"`) are
computed under **π̃**. So

  r_t = exp( log π̃_new(a_t) − log π̃_old(a_t) )    — **masked-old vs masked-new, self-consistent.**

The mask draw is identical across the two forwards because it is keyed on the
token's stable identity, not packing position (`activation_mask.py:15-29`,
docstring; verified by the 7 per-layer `mask_ratio` counters reading 0.8999 and
`pg_clipfrac≈0.03`). Hence within one global_step, π̃_old and π̃_new differ only
through the AdamW mini-batch updates to θ — **r_t ≈ 1**, `pg_clipfrac ≈ 0.03–0.04`
(EXP-17/19 logs). **[FACT]**

This already answers half of question A: **the mask does not corrupt the PPO
ratio.** The corruption is entirely in the *gradient* that flows back through
`∇_θ log π̃`.

---

## A. What gradient does the masked forward compute, and what policy is being optimized?

### A.1 The deployed policy is exactly π_θ. There is no "masked policy" being deployed. [FACT]

The clearest way to see the (a)/(b) distinction the CONTEXT asks for:

- **(b) The actual optimized/deployed object is π_θ — the weights.** Validation
  measures π_θ (unmasked). The reward, advantages, and the *meaning* of "this run
  reached 0.735" are all about π_θ. The mask never touches a deployed forward.
- **(a) π̃_θ is a transient computational artifact** used only to manufacture a
  gradient w.r.t. θ. It is near-decorrelated from the sampler μ
  (`pearson(actor,rollout) ≈ 0.004`, `ppl_ratio ≈ 2.5×10⁷`, `kl ≈ 17`,
  EXP-19 `zejoupvf`). **[FACT]**

So "what policy is the masked actor following?" is a category error if read
literally: **no policy is *following* π̃**. π̃ is never sampled from in the loop
(the sampler is μ; the deployed policy is π_θ). π̃ exists only to define a scalar
loss whose θ-gradient we descend. The right question is the next one.

### A.2 The masked update is a biased, high-variance stochastic estimator of the true GRPO gradient. [INFERENCE]

Write the per-token surrogate (in the unclipped, |r−1|<ε region where the run
lives, `pg_clipfrac≈0.03`) as `ℓ_t(θ) = − A_t · r_t(θ)`. The true GRPO gradient
(what dense computes) is

  g_true(θ) = − E_data Σ_t A_t ∇_θ log π_θ(a_t | s_t).        (1)

The masked gradient is

  g_mask(θ) = − E_data E_m Σ_t A_t ∇_θ log π̃_θ(a_t | s_t; m),   (2)

where the inner expectation is over the Bernoulli mask draws m (one realized draw
per step in the run; (2) is its expectation). Decompose:

  g_mask = g_true + b(θ) + ξ,    b(θ) ≡ E_m[∇ log π̃] − ∇ log π_θ  (bias),
                                  ξ ≡ ∇ log π̃(m) − E_m[∇ log π̃]   (zero-mean noise, per-draw).   (3)

Two separate questions: is **b = 0** (unbiased)? and how big is **Var(ξ)**?

### A.3 Why rescale makes ACTIVATIONS unbiased but the GRADIENT remains biased. [INFERENCE — the central nonlinearity argument]

This is the subtle point the SUMMARY.md correction was circling and the CONTEXT
asks to "work through."

**Activations are unbiased.** With inverted dropout, at each masked layer
`h_tilde = h ⊙ m / (1−p)` with `m ~ Bernoulli(1−p)` i.i.d. per (token, channel).
Then `E_m[h_tilde] = h ⊙ E[m]/(1−p) = h ⊙ (1−p)/(1−p) = h`. So **E_m[h_tilde] = h
exactly** (channel-wise, every masked layer). This is what "rescale restores
E[h̃]=h" means, and it is correct. **[FACT]**

**The gradient is NOT unbiased**, for two compounding reasons:

**(i) log π̃ is a nonlinear function of h_tilde.** The boundary activation feeds
through the remaining decoder blocks (attention, MLP, RMSNorm) and a softmax to
produce `log π̃(a_t)`. For any nonlinear F,

  E_m[ F(h_tilde) ] ≠ F( E_m[h_tilde] ) = F(h).

A second-order (delta-method) expansion around h makes the gap explicit. Let
`u = h_tilde − h`, with `E[u]=0` and per-coordinate `Var(u_j) = h_j² · p/(1−p)`
(this is the inverted-dropout variance: `Var(m/(1−p)) = p/(1−p)`, so at p=0.9 it
is **9× the squared activation per kept channel** — enormous). Then

  E_m[ F(h_tilde) ] ≈ F(h) + ½ Σ_{j,k} ∂²F/∂h_j∂h_k · E[u_j u_k]
                    = F(h) + ½ Σ_j ∂²F/∂h_j² · h_j² · p/(1−p).        (4)

The correction term is the curvature of F weighted by `p/(1−p)·h_j²`. It is
**generically nonzero** and it is *exactly the dropout-as-regularizer term* (Wager
et al. 2013): dropout adds a data-dependent penalty proportional to the curvature
times the activation variance. **Unbiased activations ⇒ a biased function value
⇒ a biased gradient.** The gradient is `∇_θ` of (4); the second term is an extra,
non-vanishing, θ-dependent force that g_true does not contain. So **b(θ) ≠ 0** even
with perfect rescale. **[INFERENCE]** (This is established for the dropout
*regularizer* form; the application here — that it implies a *biased policy
gradient* — is the inference.)

**(ii) The mask is applied at the FORWARD activations, but the loss is a ratio of
log-probs, and the gradient mixes both.** Even setting curvature aside, `∇_θ log
π̃` routes through the masked sub-Jacobians: the backward pass through a masked
layer multiplies the upstream gradient by `m/(1−p)` (the same mask, since
`h_tilde = (m/(1−p)) ⊙ h` and the gain is a constant w.r.t. θ in the "constant"
mode — `_rescale_gain` is not detached, it is a scalar; `activation_mask.py:349`).
So the backward signal into the lower 3–24 layers is itself a `p/(1−p)`-variance
sparse random projection of the true backward signal. This is **DropConnect-like
on the activation path** (Wan et al. 2013): random multiplicative gating of the
forward, hence of every gradient that passes through it. **[FACT about the
mechanics; INFERENCE that it inflates gradient variance.]**

**Net:** rescale fixes the *first moment of the activations* (necessary so the
masked forward is not systematically attenuated, which would make r_t corrupt and
which is what the no-rescale runs suffered — SUMMARY.md "biased mask" row). It
does **not** fix the *gradient bias* (curvature term, eq. 4) and it **adds**
gradient variance (`p/(1−p) = 9` per kept channel, ×7 layers). This is precisely
why SUMMARY.md is right that "rescale is a knob, not a fix": it trades a forward
bias for a gradient-variance explosion while leaving a residual gradient bias.
**[INFERENCE]**

### A.4 The correct estimator class: structured-multiplicative-noise / sketched gradient with a bias term. [INFERENCE]

Mapping to known estimators (the CONTEXT's request):

| Known method | Match to our mask | Mismatch |
|---|---|---|
| **Dropout** (Srivastava 2014) / inverted dropout | exact: Bernoulli per-unit multiplicative gating, `1/(1−p)` rescale, recomputed per forward. Our p=0.9 is far beyond the usual 0.1–0.5. | dropout is applied at training to *regularize the deployed net*; here it is applied to *manufacture a cheap-to-communicate gradient*, and the deployed net is the **un-dropped** π_θ. We never deploy the dropped net. |
| **DropConnect** (Wan 2013) | the *backward* path is randomly gated (Sec A.3.ii). | DropConnect gates weights; we gate activations. |
| **Sketched / sparsified SGD** (random projection of the gradient; JL) | the boundary mask is a random ±-supported coordinate subsample of the activation, so the gradient flowing through the boundary is a **sparse random projection** of the true activation gradient. With rescale it is an *unbiased sketch of the activation*; JL says random projections preserve inner products in expectation. | the projection is on activations, not on the parameter-gradient directly, so the JL inner-product preservation does NOT transfer cleanly to the parameter gradient (the same nonlinearity in A.3 breaks it). |
| **Structured-noise SGD / perturbed SGD** | best overall fit: `g_mask = g_true + b + ξ`, with structured (low-effective-rank, channel-correlated within a token) noise ξ and a curvature bias b. | the noise is multiplicative and activation-dependent, not additive isotropic. |

**Verdict (A):** the masked update descends a **biased, structured-high-variance
stochastic estimator of the true GRPO gradient on π_θ**. Rescale removes the
*activation* bias (keeping r_t≈1, the reason GRPO's ratio stays sane) but leaves a
*gradient* bias from forward nonlinearity (eq. 4) and inflates gradient variance by
`p/(1−p)` per masked channel. It is best described as **structured-multiplicative-
noise SGD with a curvature bias**, i.e. dropout-noise repurposed as a gradient
compressor. **[INFERENCE]**

### A.5 Connection to the literature (see `literature.md` tags). [grounding]

- **C3 (Mohtashami et al., AISTATS 2022)** is the most direct theoretical analogue:
  it proves convergence for *arbitrary* (not just symmetric-random) gradient masking
  where the gradient inherits a forward perturbation — exactly our case — under
  NTK-style assumptions for shallow nets. It legitimizes the estimator class but its
  guarantee does not transfer cleanly to transformer fine-tuning (the very
  nonlinearity of A.3 is what its NTK linearization assumes away). **Use as
  framing, not proof.**
- **C10 (Gradient Routing, Cloud et al. 2024)** confirms the chain-rule mechanics of
  A.3.ii: a forward/backward mask induces a structured gradient-masking pattern that
  localizes — but does not destroy — learning.
- **Important reconciliation with C1 (EF21, Richtárik et al. 2021):** lit-scout
  notes EF21 requires a *contractive* compressor and that rescale makes our binary
  mask "contractive." Be precise: EF21's contractivity is a property of the
  **gradient compressor** `C(g)` with `E‖C(g)−g‖² ≤ (1−δ)‖g‖²`. Our rescale makes
  the **activation** an unbiased compressor, NOT the parameter gradient — the
  gradient still carries the curvature bias b (eq. 4). So EF21 is the right
  *framework* (biased compressor + periodic error correction ⇒ convergence) but our
  mask is **not** an EF21 compressor in the literal sense; the curvature bias is a
  term EF21's analysis would have to absorb into its `(1−δ)` contraction constant,
  and whether it stays inside `δ<1` is exactly the task-dependent question C asks.
  **[INFERENCE — flag this as the gap between the clean theory and our setting.]**

---

## B. Why does a ~20-million× train-inference perplexity gap still LEARN (GSM8K)?

### B.1 The 20M× gap is a red herring for learning; it is a π̃-vs-μ gap, not a π̃-old-vs-π̃-new gap. [FACT/INFERENCE]

`ppl_ratio ≈ 2.5×10⁷`, `kl ≈ 17`, `pearson ≈ 0.004` all compare **π̃ to μ (the
vLLM sampler)** — that is what `rollout_corr/*` measures (`training_log_ppl` from
the masked actor forward vs `rollout_log_ppl` from vLLM). **[FACT]** The gradient,
however, does **not** depend on this gap: GRPO's importance ratio is π̃_new/π̃_old
(both masked, A.1), so the 20M× number never enters the loss. The advantages A_t
are computed from rewards on μ-sampled responses and are mask-independent. **[FACT]**

So the gap is a *diagnostic of how off-distribution the masked forward is*, not a
term the optimizer fights. SGD does not need the training forward to match the
sampler; it needs the *update direction* to have positive correlation with an
ascent direction on the deployed objective. The 20M× gap and learning are
**orthogonal**. This is the single most important reframing in this report.
**[INFERENCE]**

### B.2 Minimal condition for ascent: positive expected inner product. [FACT — standard, applied here]

Let J(θ) be the true expected return of the deployed policy π_θ (what val
measures). One AdamW step is, to first order and ignoring preconditioning,
θ⁺ = θ − η · ĝ for an estimator ĝ. The deterministic descent lemma gives the
expected one-step change

  E[ J(θ⁺) − J(θ) ] ≈ η · ⟨ ∇J(θ), E[ĝ] ⟩ − O(η²).

Ascent on J (return *increases*; loss decreases) requires, to first order,

  **⟨ ∇J(θ),  E[ g_mask ] ⟩ > 0.**        (5)

Using g_mask = g_true + b + ξ and that g_true is the true policy gradient
(∇J = −g_true direction by construction of the surrogate), (5) becomes

  ‖g_true‖² + ⟨g_true, b⟩ > 0,    i.e.   ⟨g_true, b⟩ > −‖g_true‖².      (6)

**This is the minimal condition.** It is *much* weaker than "the masked forward
matches the sampler." It only requires the bias b not to be so adversarial that it
flips the sign of the projection onto g_true. The variance ξ does **not** appear in
(5) (zero mean) — it only slows convergence (enters the O(η²) term) and is what
Adam's per-coordinate `m̂/√v̂` normalization + verl grad-clip tame (SUMMARY.md's
correct "judge on val/score not grad_norm" point: Adam is scale-invariant, so a
9×-inflated variance/norm is bounded into an O(η) update). **[FACT/INFERENCE]**

### B.3 Three structural reasons (5)/(6) holds for this method. [INFERENCE]

1. **The bias b is curvature-aligned, not return-adversarial.** From eq. (4), b is
   a regularizer-like force (penalize high-curvature directions), not an
   anti-correlated copy of g_true. There is no mechanism making ⟨g_true, b⟩ ≈
   −‖g_true‖²; generically the projection onto g_true survives. The 7 masked layers
   leave layers 0–2, 25–27, the embeddings, and all of attention/MLP *unmasked in
   weight space* (the mask is on activations at boundaries; the gradient still
   reaches every weight). So g_mask retains a substantial component along g_true.
   **[SPECULATION→INFERENCE]** (the empiricist's confirmation that reward rises
   *within* masked windows is the direct test of (5); see coordination note.)

2. **GRPO ratio-clip is weight protection.** Even on a bad masked step, the clip
   `clip(r_t, 1−ε, 1+ε)` bounds the per-token surrogate gradient magnitude. Because
   r_t≈1 (masked-self-consistent), almost nothing is clipped (`pg_clipfrac≈0.03`),
   but the clip is a *safety rail*: it prevents any single corrupted token from
   producing an unbounded update. This is why naive no-rescale + positional-keyed
   masking failed (SUMMARY.md): there r_t was corrupt (mask differed across the two
   forwards), clipfrac climbed toward saturation, clipped tokens stopped
   contributing, and learning died. With the stable-id keying, r_t≈1 keeps the
   surrogate in its linear region where (5) is meaningful. **[FACT/INFERENCE]**

3. **Rescale keeps r_t≈1.** (Restated: the *activation* unbiasedness is what makes
   π̃_old and π̃_new comparable, hence r_t≈1, hence the surrogate gradient ≈ the
   advantage-weighted score `−A_t ∇logπ̃`.) Without it the masked forward is
   systematically attenuated and the ratio is wrong. **[FACT]**

So GSM8K learns because (6) holds with a comfortable margin: the masked gradient
keeps a positive projection onto g_true, the variance is Adam-bounded, and the clip
protects the weights. **[INFERENCE]**

### B.4 The role of clean@20: error-feedback / periodic re-anchoring that removes ACCUMULATED bias before it diverges. [INFERENCE — the sawtooth mechanism]

If condition (6) held with margin on *every* step, you would not need clean steps
at all — but it does not, for a subtle reason. The bias b(θ) is **systematic**
(same curvature direction every masked step, not zero-mean), so it **accumulates**
across the 19 masked steps in a window:

  θ drifts along −Σ b over the window — a coherent, non-averaging error.

Zero-mean variance ξ averages out over a window (∝ 1/√19), but the bias does not.
Left unchecked, accumulated bias would eventually rotate g_mask far enough that (6)
fails. **The clean step is the correction.** Every 20th step runs the **true,
unmasked dense gradient** g_true (clean grad_norm ~0.2–0.4; `pearson→0.9995`;
`kl→0.0003`; `ppl_ratio→1.0003` — EXP-17 and EXP-19 both, see empirical numbers).
That single exact step:

- injects a guaranteed-ascent direction (g_true itself), and
- **resets the drift**: the clean step re-anchors θ onto the true-gradient
  manifold before accumulated bias can flip the sign in (6).

This is **exactly the error-feedback / memory mechanism of compressed SGD**
(Karimireddy et al. 2019; Stich et al. 2018): a biased compressed gradient is
tolerable *if* the accumulated error is periodically corrected. Here the
correction is not a stored residual buffer but a **periodic exact gradient**, which
is closer to:

- **Local-SGD / periodic full-sync** (Stich 2018): K local (cheap, biased-by-
  staleness) steps then 1 synchronization. Our "sync" is a clean gradient. The
  K↔quality tradeoff in EXP-17 (K=4/5/20 all reach ≈parity on GSM8K but larger K
  slows learning-speed: steps-to-reward≥0.5 = 17/18/44) is the **classic Local-SGD
  K-vs-convergence-rate curve**. **[INFERENCE — strong analogy]**
- **SVRG-style control variate** (Johnson & Zhang 2013): the clean gradient is a
  periodically-refreshed "full" gradient that re-anchors the biased cheap
  gradients. The match is loose (SVRG subtracts a stored snapshot term every step;
  we don't subtract, we just re-anchor periodically), so I rate this the **weaker**
  analogy. **[SPECULATION]**

The **clean-resettable sawtooth** in `rollout_corr` (`pearson` 0.004 in masked
windows → 0.9996 at each clean step, then back; flat across windows, NOT a
monotone ratchet — EXP-17 verdict, the steps-since-clean binning is flat R²=0.03)
is the *direct fingerprint* of this mechanism: bias accumulates within a window
(the slow rise in absolute kl level tracks the improving true policy's perplexity),
the clean step zeroes it, repeat. The fact that it **fully resets every time and
does not ratchet** is the empirical proof that the clean step is *sufficient* to
keep (6) satisfied across the whole run. **[FACT (the sawtooth) + INFERENCE (its
interpretation)]**

**Minimal condition for ascent, complete statement:** ascent holds iff, over each
clean-to-clean window, the *windowed-average* masked gradient keeps a positive
projection onto g_true (eq. 6 averaged), AND the clean step fires often enough that
accumulated bias never rotates that projection negative before the next reset.
clean@20 satisfies this on GSM8K. **[INFERENCE]**

### B.5 Connection to the literature, and a stronger mechanism for the sawtooth's stability. [grounding + INFERENCE]

The error-feedback analogy is corroborated and, importantly, **sharpened** by
recent RLVR-specific results (`literature.md` tags):

- **The strongest mechanism — RLVR is near-rank-1 (B4 RELEX, B5 Linear-RLVR).**
  These document that RLVR's *entire* parameter-space trajectory is approximately
  rank-1 (a single dominant update direction; R²>0.7 weight/logprob linearity), and
  that **high training-signal variance acts as a low-pass filter that *enforces*
  this linearity.** This gives a cleaner, more robust account of why the sawtooth
  does not diverge than my curvature-accumulation argument alone: the 19 masked
  steps are **noisy projections of the same 1-D capability direction**, so masking
  corrupts mostly the *magnitude/phase* along that line (which rescale and Adam
  handle) rather than rotating *off* it; the clean step re-defines the 1-D
  direction every 20 steps. If the trajectory is genuinely rank-1, the windowed
  projection (6) is structurally protected — there is essentially one direction for
  the bias to be projected onto. **This is now my primary mechanism for B; the
  curvature-bias-accumulation picture (B.4) is the second-order refinement on top.**
  **[INFERENCE — contingent on the rank-1 claim transferring to our masked setting;
  the B5 prediction that *more* masking → *more* enforced linearity is directly
  testable.]**
- **EF21 / total-error-minimization (C1, C2):** the clean step as a periodic
  error-accumulator reset is exactly C2's "balance compression error across
  iterations" optimality picture; C1 supplies the convergence framework (with the
  A.5 caveat that our gradient bias must fit inside the contraction constant).
- **The clean step does ≥3 jobs, not one (A2, C9).** (i) Error-accumulator reset
  (C2). (ii) It re-anchors the rank-1 direction (B4/B5). (iii) **C9 (Mroueh 2025)**:
  GRPO's success-amplification recurrence converges to a fixed point *above the
  reference policy* as long as the reward has nonzero correlation with the true
  advantage — *independent of gradient quality* — giving a lower bound that the
  masked windows alone cannot fall below. (iv) **A2 (Chen et al., ICLR 2026)**: the
  clip bias compresses entropy toward high-prior modes *every* step (our
  `pg_clipfrac≈0.03` is active on masked steps too), so even masked windows do
  *some* directional work via clip bias — which is itself a partial explanation for
  within-window reward rise (the empiricist's P1 test). **[grounding]**

---

## C. Why GSM8K parity but Big-Math stall? Elicitation tolerates a lossy gradient; learning does not.

This is the deepest question and the place to be most careful about FACT vs
SPECULATION. The empirical asymmetry is sharp (CONTEXT findings 1–2), and the
empiricist's format-controlled **base-capability eval** (`base_capability_eval.md`,
on-box 2026-06-01, 200 test problems, greedy, identical `\boxed{}` prompt + verifier)
pins down *why*:

| dataset | **base accuracy (NO RL)** | masked clean@20 final | dense final |
|---|---|---|---|
| GSM8K | **0.715** | **0.735** (EXP-17) | 0.741 (EXP-16) |
| Big-Math | **0.480** | **0.55 flat** (EXP-19) | **0.59–0.61** (EXP-20, dense climbs) |

The base eval is the load-bearing control. On GSM8K the model is **already ~72%
capable with zero RL** — the real RL headroom is only **0.715→0.735 (~+0.02)**; the
step-0 0.085 was a pure `####`-format artifact, not a capability gap. On Big-Math
the base is **0.48**, dense finds **~0.61** (so **headroom demonstrably exists**),
but masked stays flat at **~0.55**. So the stall is **specific to the masked
gradient on the hard task** — it is a *gradient-fidelity limitation exposed by task
difficulty*, **not** a lack of headroom (dense proves the headroom is there). **[FACT]**

### C.1 The decisive control: the repair mechanism is IDENTICAL on both tasks. [FACT]

I verified directly from EXP-19 (Big-Math) clean steps 20/40/60/80: `pearson →
0.9995`, `rollout_corr/kl → 0.0003`, `ppl_ratio → 1.0003`, clean grad_norm ~0.19,
clean-step entropy 0.371→0.316→0.308→0.272 (trending **down**, the healthy-policy
signature). These are statistically indistinguishable from EXP-17 (GSM8K) clean
steps. **The sawtooth, the per-clean repair, and the true-policy health signals are
the same on both tasks.** Therefore the Big-Math stall is **NOT** a failure of the
clean-step / sawtooth machinery (B.4). The clean steps are firing and repairing
perfectly on Big-Math; the policy just isn't *going anywhere*. **[FACT]**

This rules out the easy explanations (mask broke, clip saturated, drift to random)
and forces the explanation onto the **information content of the gradient relative
to what each task demands**.

### C.2 Elicitation vs learning: a signal-to-noise argument. [INFERENCE — the core claim]

Frame both tasks through condition (6) and the structure of what RLVR does to a
**Qwen base model**. The mechanism stands on its own and is strongly corroborated by
the literature (`literature.md`): **B1 (Yue et al., NeurIPS 2025 Oral)** gives the
minimal *sufficient* condition — RLVR cannot learn if the base produces zero correct
rollouts (0/1 reward → no gradient); all RL-found solutions already live in the
base's sampling distribution (pass@1 ↑, pass@k flat). **A3 (1-shot RLVR, NeurIPS
2025)** and **A1 (Spurious Rewards)** show a single example or even *random* rewards
elicit latent Qwen-math capability — i.e. easy tasks need almost no gradient
*content*. The masked gradient is just a further-degraded signal, which tightens
B1's sufficient condition into a *necessary* one for us: the task must be elicitable
enough that even a corrupted gradient raises pass@1.

**GSM8K is an ELICITATION task for this model.** The base eval settles this
quantitatively: Qwen2.5-1.5B-Instruct **already solves 71.5% of GSM8K with zero RL**
and emits a `\boxed{}` answer on 192/200 prompts (`base_capability_eval.md`). The
GSM8K-solving mode is therefore *already the dominant mode of the base policy*. The
true RL headroom is a mere **+0.02** (0.715→0.735); the step-0 0.085 was a pure
*format* artifact — RL elicited the latent capability to 0.49 by step 30 (CONTEXT
finding 1). What RL must do is **sharpen/select an already-present, already-dominant
mode** and fix output format — up-weight a chain the base model *already samples
with high probability*. The required gradient information is **coarse and
low-dimensional**: "more of this kind of answer, in this format." A direction merely
*positively correlated* with g_true (condition 6 with any positive margin) suffices,
because you are climbing a hill you already stand on. The masked gradient's surviving
g_true-component is more than enough for a 2-point move. **[INFERENCE, grounded in
the 0.715 base FACT]**

**Big-Math is a LEARNING task for this model.** Competition math requires
*capability the base model does not reliably have* (base 0.48; dense gains only ~+0.06
to ~0.61, and slowly — it is hard even with the exact gradient). The optimization
signal is
**sparse** (few correct rollouts → most advantages ≈ 0 or noisy), the useful
gradient directions are **high-dimensional and precise** (you must move probability
toward specific multi-step reasoning chains the model rarely produces), and the
landscape is **not** a nearby mode to sharpen — it is a search for a new basin.

Now apply the estimator decomposition (3): g_mask = g_true + b + ξ. On Big-Math:

1. **The true signal ‖g_true‖ is small** (sparse reward, weak advantages — dense
   barely moves). 
2. **The bias b and variance ξ are the SAME absolute size as on GSM8K** (they are
   properties of the mask `p/(1−p)·h²` curvature, eq. 4 — task-independent). 

So the **signal-to-corruption ratio ‖g_true‖² / (‖b‖² + tr Var ξ)** is far worse on
Big-Math: a small true signal buried under a fixed-size bias+noise floor. The
windowed projection (6) can hover near zero, the masked windows make ~no progress,
and the only real signal is the handful of clean steps — but 4–5 exact gradient
steps cannot, by themselves, *learn a new capability* (dense needs hundreds of
*every-step* exact gradients to gain even +0.05). On GSM8K the same 4–5 clean steps
plus weakly-correlated masked windows suffice because the task only needed
elicitation. **[INFERENCE — this is the central thesis of C.]**

**This SNR argument is made quantitative by C4 (Kolomvaki et al. 2026):** an NTK
analysis of masked-input training shows linear convergence *to an error region whose
floor scales with the mask variance* `∝ p/(1−p) ≈ 9`. Reading C4 into our setting:
the masked error floor is a fixed, task-independent quantity; GSM8K's achievable
gain (the +0.02 headroom on top of an already-dominant 0.715 mode) sits *above* that
floor → reachable; Big-Math's achievable gain (dense extracts only ~+0.06, slowly)
plausibly sits *below* the masked error floor → unreachable, hence the flat stall.
**B6 (Unlearnability, ICML 2026)** is the complementary statement in
gradient-similarity terms: hard examples have low gradient similarity to the
training distribution and are unlearnable even with correct rollouts present; masking
lowers the already-weak SNR further. C4 and B6 are two views (variance floor vs
gradient-similarity) of the same SNR collapse. **[INFERENCE, now quantitatively
anchored]**

Compact statement: **the masked gradient destroys high-frequency, low-amplitude
gradient information and preserves low-frequency, high-amplitude information.**
Elicitation lives in the low-frequency/high-amplitude band (sharpen the dominant
mode) → tolerant. Genuine learning lives in the high-frequency/low-amplitude band
(find rare correct chains) → fatal. **[SPECULATION — the frequency framing is a
heuristic; the SNR argument above is the rigorous version.]**

### C.3 An equivalent, falsifiable restatement. [INFERENCE]

If C.2 is right, three predictions follow (hand these to empiricist / future runs):

- **(P1)** On GSM8K, reward should rise measurably *within* masked windows (not only
  at clean steps) — because (6) has positive margin. On Big-Math, within-window
  reward change should be ≈0 (flat), with any micro-progress concentrated at clean
  steps. **[testable now from EXP-17 vs EXP-19 per-step reward — requested from
  empiricist.]**
- **(P2)** Dense Big-Math (EXP-20) learns *slowly* (+0.05) → even the exact gradient
  is information-poor here; a lossy gradient must be worse. Consistent. **[FACT,
  EXP-20.]**
- **(P3)** Lowering p (less masking → smaller b, smaller ξ, larger surviving
  g_true) should *recover* Big-Math learning monotonically, and there should be a
  threshold p\* below which Big-Math starts to climb. GSM8K should be insensitive to
  p down to high values (elicitation is robust). **[predicted; the p-sweep in
  SUMMARY.md's open question is the test.]**

### C.4 Two alternative hypotheses I cannot rule out (intellectual honesty). [SPECULATION / confounders]

The elicitation story (C.2) is the best fit, but two literature-grounded
alternatives are *consistent with the same data* and must be flagged for the
synthesizer rather than waved away:

1. **Memorization-shortcut, not elicitation (A5, Spurious Rewards Paradox, Yan et
   al. 2026).** A5 argues Qwen2.5's apparent RLVR gains can be *circuit-level
   memorization retrieval* — a "Functional Anchor" at **layers L18–20** retrieving
   memorized solutions, "Structural Adapters" at **L21+** — not genuine reasoning,
   evidenced by a perplexity paradox (answer-token ppl drops while prompt coherence
   degrades). **Two things make this a live confounder for us, one of which is
   striking:** (a) our masked model has a *huge* train-inference perplexity gap
   (`training_log_ppl≈17` vs `rollout_log_ppl≈0.31`), superficially the same family
   of perplexity anomaly A5 fingerprints (though ours is provably a mask artifact,
   not weight memorization — it resets at every clean step); and (b) **two of our 7
   masked boundaries are layers 18 and 21** (`[3,7,11,15,18,21,24]`) — *exactly* the
   layers A5 identifies as the memorization-retrieval circuit. If A5's mechanism is
   real, masking at L18/L21 would specifically perturb the retrieval circuit, and
   the GSM8K-tolerance / Big-Math-stall split is *also* explainable as "memorization
   retrieval survives a corrupted gradient (GSM8K has memorizable templates) but
   competition math has no template to retrieve." **This does NOT invalidate the
   communication-efficiency result** — validation is on the true unmasked weights, so
   whatever mechanism produces 0.735 is the real deployed behavior — but it means
   "the masked gradient elicits a *reasoning* capability" is *not established*; it
   could be eliciting a retrieval shortcut. The two are observationally equivalent at
   our measurement resolution. **[SPECULATION — genuine alternative; the L18/L21
   overlap is a concrete, testable hook, not decoration.]**

2. **Skill-sharpening, not pure elicitation (B2, Wang et al. 2026).** B2 argues RLVR
   *does* teach: it sharpens atomic-step probabilities so multi-step chains stop
   decaying exponentially. Under pure elicitation, the clean step's *direction*
   barely matters (any positive push helps); under skill-sharpening, the clean
   step's direction is **load-bearing**. The historical EXP-16 evidence cuts toward
   B2 on a fine scale: **mask-only stalls (0.13→0.15) but clean@4 unlocks
   convergence (→0.62)** — i.e. the clean step provides *directional* information the
   masked windows cannot, which is more than "any push helps." So the honest
   position is a **spectrum**: GSM8K sits near the elicitation end (coarse gradient
   suffices), Big-Math near the genuine-learning end (precise gradient required), and
   the clean step's direction matters *more* as you move toward the learning end.
   This is fully compatible with the SNR argument (C.2) — it just refines "coarse
   direction suffices" into "the *required directional precision* rises with task
   difficulty, and that is exactly the axis the mask degrades." **[INFERENCE —
   strengthens rather than threatens C, but corrects an over-strong reading of "pure
   elicitation."]**

**Net for C:** the SNR/error-floor mechanism (C.2, C4/B6) is robust and
quantitative. Whether the *thing being elicited* on GSM8K is reasoning or retrieval
(A5) is unresolved at 1.5B scale and does not affect the comms-efficiency claim. The
elicitation-vs-learning framing should be stated as a **spectrum of required
gradient precision** (B2), not a binary. **[the calibrated conclusion]**

---

## D. One-paragraph synthesis (for the synthesizer to cite)

The activation mask does not produce a "masked policy" — it produces a **biased,
structured-high-variance stochastic estimator** of the true GRPO gradient on the
*unmasked* deployed policy π_θ (A). Inverted-dropout rescale makes the *activations*
unbiased (keeping the PPO ratio self-consistent, r_t≈1, which is why the 20-million×
train-inference perplexity gap never enters the loss and is a red herring for
learning), but the *gradient* stays biased through the forward nonlinearity (eq. 4)
and gains `p/(1−p)=9×` variance per masked channel (A.3). Learning needs only the
weak condition ⟨∇J, E[g_mask]⟩>0 (B.2): a positive projection onto the true
gradient, which Adam's scale-invariance and GRPO's ratio-clip protect, and which the
clean-every-20 step re-anchors before accumulated bias rotates it negative — the
clean-resettable sawtooth is the error-feedback / Local-SGD fingerprint of this
(B.4). This holds with margin on GSM8K, an **elicitation** task (base 0.715 → final 0.735,
only +0.02 of headroom: RL just sharpens an already-dominant latent mode, so a coarse
gradient suffices), and fails on Big-Math, a **learning** task (base 0.48; dense
finds ~0.61, so headroom exists, but masked stays flat ~0.55) where the true signal
‖g_true‖ is small and sparse while the mask's bias+variance floor is task-independent
and fixed — so the signal-to-corruption ratio collapses and the masked gradient
cannot extract the precise, high-dimensional information genuine learning requires
(C). The Big-Math stall is thus a gradient-fidelity limit exposed by task difficulty,
not a missing-headroom artifact — the cleanest possible separation of the two.

---

## Open coordination items
- **empiricist (still open, load-bearing for C.2):** P1 — does GSM8K reward rise
  *measurably within* masked windows (between consecutive clean steps) while Big-Math
  is flat between clean steps? Direct test of condition (6)/(SNR). [base-eval +
  clean-step-health already delivered and folded in.]
- **empiricist (new, from A5 confounder):** the A5 memorization-shortcut hypothesis
  predicts our EXP-17 masked model should have *lower pass@k at k≥128* than dense
  EXP-16 on the same val set (elicitation/retrieval, not new capability). Testable
  from existing checkpoints if available. Also: does masked GSM8K show increased
  code-reasoning frequency (A1's amplification channel)?
- **lit-scout (DELIVERED):** `literature.md` complete; citations folded into A.5,
  B.5, C.2, C.4. Key load-bearing tags now cited: C3/C10 (estimator class), C1/C2
  (error-feedback), B4/B5 (rank-1 trajectory — now my primary B mechanism), C9/A2
  (clean step does multiple jobs), B1/A1/A3 (elicitation), C4/B6 (SNR error floor),
  A5/B2 (the two alternative hypotheses in C.4).

## Caveats / what would change the conclusions
- The **rank-1-trajectory** mechanism (B.5) is contingent on B4/B5's rank-1 finding
  transferring to our *masked* setting; if masking rotates the update *off* the rank-1
  line (rather than perturbing magnitude along it), B.5 weakens and the
  curvature-accumulation picture (B.4) carries more weight. B5's prediction (more
  masking → more enforced linearity) is the test.
- The **A5 memorization-vs-reasoning** question is unresolved at 1.5B scale and the
  L18/L21 mask-boundary overlap is suggestive but not proof. It does not affect the
  *comms-efficiency* claim (validation is on true weights) but it does bound how
  strongly we may claim "elicits reasoning."
- The masked **gradient is not a literal EF21 compressor** (A.5): convergence
  theory frames but does not prove our case; the curvature bias must fit inside the
  contraction constant, which is exactly the task-dependent SNR question (C).
