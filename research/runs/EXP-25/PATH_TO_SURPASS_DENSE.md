# Path to Surpass Dense — Compression-as-Exploration in Communication-Efficient RL

**Status: DRAFT (round 1, pre-mechanist-convergence).** This document develops the
operator's thesis — *that communication-efficient training can SURPASS dense in RL
(unlike SFT, where compression is pure information loss)* — into a rigorous,
falsifiable program. It is the `strategist` deliverable for the `surpass-dense`
team (task #2). The bias/noise characterization that the central argument hinges on
is being grounded by `mechanist` (task #1); sections tagged **[PENDING-MECHANIST]**
will be tightened once those answers land. Citations: `runs/EXP-25/DEEP_FINDINGS.md`,
`runs/EXP-25/ENTROPY_COLLAPSE_FINDINGS.md`, `diagnostics/ENTROPY_COLLAPSE_WATCH.md`,
issue #24, `verl/workers/comm_eff/spectral_filter.py`, and W&B
(`shamanework-pl/verl_compression_research`).

---

## 0. The bar and the prize

| reference | val@50 | what it is | W&B |
|---|---|---|---|
| **dense** | **0.7536** | no comm-eff — the bar to BEAT | `5e2jpho9` |
| A0 PowerSGD r77 + fresh-clean@5 | 0.7415 | best comm-eff so far (−0.012 vs dense) | `oquyeic3` |
| plain PowerSGD r77 (no merger) | ~0.74 / floor 0.6914 | bare codec, nearly harmless | EXP-23 A1 |
| signed_ema α=0.5 (knee) | 0.7066 | least-harmful merger point | `1wulaelw` |
| signed_ema α=0.3 | 0.6164 | delayed collapse | `r8kc702g` |
| signed_ema α=0.0 | 0.3541 | catastrophic collapse | `uyrpaftw` |

The prize is **val@50 > 0.7536** — comm-eff RL that does not merely match dense but
exceeds it. The operator's claim is that this is *attainable in RL specifically*,
because the channel that makes compression pure loss in SFT (information bottleneck)
is, in RL, also a channel for productive exploration noise.

---

## 1. Why RL ≠ SFT: the four channels that flip compression from "pure loss" to "possibly net-positive"

In SFT the objective is a **fixed**, fully-supervised log-likelihood on a static
dataset. Every bit dropped on the gradient path is a bit of the (correct, stationary)
descent direction lost — there is no mechanism by which losing signal helps; the
minimum is fixed and compression can only slow or bias the descent toward it. This is
the SFT intuition: compression = information loss = monotone harm. (This is also why
the signed_ema merger was "SL-validated" — in SFT, the stale-sign trick is a *mild*
regularizer on a stationary target; it is RL non-stationarity that turns it
adversarial, see §3.)

RL on a policy-gradient objective is different on **four** structural axes, each of
which opens a door for compression noise to be net-positive:

**(C1) The objective is non-stationary and self-generated.** The data distribution
is the policy's own rollouts; the loss landscape moves every step as the policy moves.
There is no fixed minimum to converge to — there is a *trajectory* through
policy-space. A perturbation that would be pure error against a fixed target can
instead nudge the trajectory toward a different, better basin. The relevant question
is not "how close to the true gradient" but "does the perturbed trajectory reach a
higher-reward, better-generalizing policy."

**(C2) RL is exploration-limited, not optimization-limited.** On GSM8K with
Qwen2.5-1.5B the per-step policy movement is tiny (lr 1e-6, n=8 group-relative
advantages that partially cancel — DEEP_FINDINGS §B4). The binding constraint on final
reward is whether the policy *visits* high-reward token sequences during rollout, not
whether it descends a known gradient quickly. Anything that keeps rollouts diverse
longer (delays entropy collapse, widens the sampled support) directly buys more
exploration. Gradient noise that perturbs the policy off its greedy path is exactly
such a mechanism — *if* it is zero-mean (it jitters the policy without dragging it
in a fixed wrong direction).

**(C3) The implicit step-size regularizer is geometric, and compression can act with
or against it.** The true minibatch PG is a sum of signed score-function terms × group-
normalized advantages; across a GRPO group these *partially cancel*, so the true step
is small on most coordinates and sign-ill-defined on many (DEEP_FINDINGS §B4,
ENTROPY_COLLAPSE_FINDINGS §3b). That cancellation IS the implicit regularizer that
keeps GRPO stable with no KL/entropy term. Compression interacts with this two ways:
a **zero-mean** compression residual rides along the cancellation (adds variance,
preserves the small mean step → SGD-noise-style flat-minimum bias); a **structured**
compression that *destroys* the cancellation (signed_ema: full magnitude on every
coordinate with a fixed sign) removes the only brake and detonates (the α=0 collapse).
**This is the helps-vs-hurts knife-edge, and it is exactly the bias axis.**

**(C4) The reward is a shaping channel, not just a target.** Because reward is
computed on sampled outputs, a perturbation that changes *which* outputs get sampled
changes the *learning signal itself* on the next step (advantages re-form around the
new rollout set). Zero-mean exploration noise can surface higher-reward modes that
then get reinforced — a positive feedback that has no analogue in SFT (where the
targets are fixed regardless of model behavior). The α=0 collapse is the *negative*
version of this same feedback: the length-explosion reward-hack (DEEP_FINDINGS §A2)
is C4 running backward — structured noise pushed the policy into a degenerate mode
that the lenient reward briefly rewarded, then the feedback locked it in.

**Synthesis.** RL ≠ SFT because the objective is non-stationary (C1), exploration- not
optimization-limited (C2), implicitly regularized by gradient geometry that compression
can help or break (C3), and coupled to a self-shaping reward (C4). All four say the
*sign of compression's effect is determined by whether the compression noise is
zero-mean (rides the regularizer, explores) or structured/biased (breaks the
regularizer, collapses).* That dichotomy is the spine of this document.

---

## 2. Formalizing compression-as-exploration: the helps-vs-collapses boundary

### 2.1 The decomposition

Write the update the optimizer actually sees as the true gradient plus a compression
perturbation:

```
G_used = G_true + δ
```

where `δ = G_used − G_true` is the compression-induced deviation. Decompose δ into a
**bias** (its conditional mean given the policy) and a **zero-mean fluctuation**:

```
δ = b(θ) + ξ,    b(θ) = E[δ | θ],    E[ξ | θ] = 0
```

The whole thesis reduces to the relative size and structure of `b` vs `ξ`:

- **ξ (zero-mean fluctuation) = productive exploration.** Standard SGD-noise theory:
  zero-mean gradient noise is an implicit regularizer biasing SGD toward **flat
  minima** (better generalization) and, in RL specifically, **perturbs the policy off
  its greedy trajectory** → sustains rollout diversity → more exploration (C2). If the
  compression residual is zero-mean, compression *adds* a controllable exploration
  temperature on top of the (small, exploration-starved) RL step. This is the channel
  by which comm-eff could **beat** dense: dense has only the intrinsic PG sampling
  noise; compression adds a tunable second source.

- **b (systematic bias) = the collapse risk.** A nonzero conditional-mean deviation is
  a *persistent* wrong-direction force. Under no-KL/no-entropy GRPO it has nothing to
  oppose it, so it accumulates coherently across steps (the β=0.95 EMA in signed_ema
  makes this worse by *removing* the step-to-step re-randomization that would average a
  fluctuation away — ENTROPY_COLLAPSE_FINDINGS §3b point 2). Bias drives the policy in
  a fixed direction → the reward-hack degenerate mode (length explosion) → collapse.

**The boundary, stated precisely:**

> Compression HELPS RL when its deviation δ is dominated by a **zero-mean, step-
> decorrelated fluctuation ξ** whose magnitude is small enough not to overwhelm the
> reward signal but large enough to sustain exploration. Compression HURTS (collapses)
> when δ is dominated by a **persistent, step-correlated bias b** — especially when an
> EMA or staleness makes b *coherent* across steps and no regularizer opposes it.

### 2.2 Mapping the known operators onto the boundary

| operator | what δ looks like | bias vs fluctuation | observed | predicted regime |
|---|---|---|---|---|
| **signed_ema α=0** | `\|G\|·sign(M_stale) − G` — full magnitude, fixed stale sign on ~50% of coords | **maximal coherent bias** (β=0.95 EMA, ~20-tick persistence) | val 0.354, length→16K cap | **collapse** (confirmed) |
| signed_ema α=0.5 | disagreeing coords zeroed (projection onto sign-agreement set) | bias removed but ~half the gradient dropped (a *shrinkage*, not noise) | val 0.707, bounded length | survives but underperforms — loses signal, doesn't explore |
| **plain PowerSGD r77** | `(I−P)·G` dropped — the off-low-rank-subspace residual | **[PENDING-MECHANIST Q1]** — is `(I−P)·G` step-stable (bias) or basis-rotating (≈ξ)? | val ~0.74, nearly harmless | **near-zero-mean ⇒ candidate for the productive-noise regime** (hypothesis) |
| inject (γ=1) | adds orthogonal scale-matched M_anchor component | tiny (‖M‖≪‖G‖ after rescale) — inert | inert (EXP-23 A2 0.6967) | neither helps nor hurts |
| blend (η=0.5) | convex pull toward scale-matched M_anchor; cos(G,M)≈0.001 | shrinks step to 0.71× along ~orthogonal M | inert (EXP-23 A3 0.6861) | **[PENDING-MECHANIST Q3]** direction-correct or just magnitude? |

The single most important open cell is **plain PowerSGD's `(I−P)·G` residual**: issue
#24 measured `cos(G_powersgd, M_anchor) ≈ 0.001` — PowerSGD discards *exactly* the
directions the full-rank anchor lives in. That orthogonality is what killed inject/blend
(they tried to add back an orthogonal component), but it is also the *fingerprint of
where the dropped signal went*. Whether that dropped component is a coherent bias
(→ error-feedback #24 is the fix, removes the bias and KEEPS the noise) or a step-
decorrelated fluctuation (→ it is already the productive exploration noise and rank is
the temperature knob) is the pivotal empirical question for mechanist [Q1].

### 2.3 Why the α=0 collapse is the *negative image* of the thesis, not a refutation

The operator's framing is exactly right and the data supports it: the α=0 collapse is
"the same noise, but STRUCTURED and ADVERSARIAL." DEEP_FINDINGS §C proves the dose-
response is **monotonic** — every unit of signed correction makes RL *worse*, optimum
at α→1 (no correction). That is precisely what the bias/fluctuation decomposition
predicts: signed_ema injects a *maximally biased* δ (coherent stale sign, full
magnitude, EMA-persisted), so it sits at the far HURT end of the boundary. Its failure
does not refute compression-as-exploration; it **locates the cliff edge** — and tells
us the lever to beat dense must live at the opposite end (zero-mean, step-decorrelated,
no sign-replacement).

---

## 3. The ranked levers toward beating dense

Ranked by `P(moves us above dense) × leverage / cost`. Each: mechanism, expected
effect on the bias/fluctuation balance, and cost in GPU-hours on the fixed surface.

### Lever 1 — KL brake (CONFIRMED-good; the enabling condition, not the source of surplus)
- **Mechanism.** A small KL-to-reference penalty (`use_kl_loss=true`, coef 0.001) adds
  a restoring force toward the base policy. It does NOT remove compression noise; it
  **bounds the policy's excursion**, closing the length-degeneration reward-hack channel
  (C4 backward) that is the proximate killer of the biased arms (DEEP_FINDINGS §A2).
- **Expected effect.** Converts "structured bias → runaway collapse" into "structured
  bias → bounded drift," and — critically — makes it *safe to dial up exploration
  noise* (lower rank, see Lever 3) without detonating. KL is the **guardrail that lets
  the exploration lever be pushed past where dense's own noise sits.**
- **Status.** A0+KL(0.001) diagnostic on the α=0 collapse arm is RUNNING (team-lead
  relaying). If KL prevents even the α=0 catastrophic collapse, that is direct proof KL
  decouples "noise injection" from "collapse" — the keystone for the whole program.
- **Cost.** Already in flight; subsequent arms +1 ref-policy forward (verl auto-spins).
- **Caveat.** KL changes the FIXED control surface (no-KL/no-entropy). Run as a
  deliberate, labelled new lineage — not a silent drift (DEEP_FINDINGS §d.2).

### Lever 2 — Error-feedback on the PowerSGD residual (#24): remove the BIAS, keep the NOISE
- **Mechanism.** Accumulate `r = G − decompress(compress(G))` in FP32, fold back
  (decayed) next step. This makes the *time-average* of the compressed gradient equal
  the true gradient — i.e. it **drives the bias `b` → 0 while leaving the step-to-step
  fluctuation `ξ` intact** (PULSELoCo / arXiv 2602.03839, issue #24 top lever; issue
  #21 "NO error-feedback = top lever").
- **Expected effect.** This is the *cleanest possible instantiation of the thesis*: EF
  is precisely the operator "subtract the systematic compression bias, keep the zero-
  mean exploration noise." If PowerSGD's residual is partly biased [Q1], EF moves plain
  PowerSGD from ~0.74 toward dense; and because it preserves ξ, the resulting arm has
  dense's signal fidelity PLUS compression's exploration noise — the regime where
  beating dense is structurally possible.
- **Cost.** One FP32 buffer per boundary (memory, no extra comm). Code: #24 (blocked on
  #25's redesign, now unblocked by the STOP). 2–3 arms × 50 steps.
- **Risk.** EF re-introduces the dropped component over time; if that component is what
  was providing the exploration, EF could *reduce* exploration back toward dense. Hence
  it must be **paired with a rank/temperature lever** (Lever 3) so we can re-add noise
  deliberately rather than as an uncontrolled compression artifact.

### Lever 3 — Compression-rank-as-exploration-temperature: the lever most likely to SURPASS
- **Mechanism.** PowerSGD rank `r` controls how much of the gradient is dropped:
  `(I−P_r)·G` shrinks as `r → H`. If that residual is (or can be made, via EF, to be)
  **zero-mean**, then `r` is a direct **exploration-temperature knob** — lower `r` =
  more zero-mean perturbation = more exploration; higher `r` = closer to dense. The
  thesis predicts an **interior optimum `r* < H`** where the added exploration noise
  more than pays for the lost fidelity — i.e. `val(r*) > val(dense)`.
- **Expected effect.** This is the experiment that directly tests "comm-eff > dense."
  Dense is the `r = H` (∞-rank) endpoint with only intrinsic PG noise. A rank sweep
  with EF removing the bias isolates the *pure exploration contribution of compression*.
  If there is any `r` where val > 0.7536, the thesis is confirmed and we have a number.
- **Cost.** A rank sweep `r ∈ {38, 77, 154}` × {EF on} × 50 steps = 3 arms; cheap.
- **Why it can beat dense (the SGD-noise argument).** Flat-minima theory: zero-mean
  gradient noise of the right scale improves generalization. RL adds C2 (exploration)
  on top. Dense sits at zero injected noise. If the optimal noise level for *this*
  policy/task is > 0, then *some* compressed rank beats dense. The job is to find it —
  and to do so safely under the KL guardrail (Lever 1) so the search doesn't fall off
  the collapse cliff.
- **[PENDING-MECHANIST Q1]** This lever is only valid if PowerSGD's residual is near-
  zero-mean. If Q1 says it is a coherent bias, then rank-as-temperature only works
  *after* EF (Lever 2) sanitizes it — the two compose.

### Lever 4 — Entropy regularization (explicit exploration, the comparison control)
- **Mechanism.** Small positive `entropy_coeff` directly rewards policy entropy →
  sustains rollout diversity by construction.
- **Expected effect.** This is the **control** for compression-as-exploration: if a
  tuned entropy bonus on *dense* reaches the same val as compression's implicit
  exploration, then compression is just "entropy reg by another name" (still useful —
  it's free exploration with comm savings — but not magic). If compression+KL beats
  dense+entropy-bonus, compression's exploration is *structurally different* (gradient-
  space, not output-space) and more valuable. Either outcome is publishable.
- **Cost.** Cheap; folds into the same sweep matrix.

### Lever 5 — Direction-preserving correction primitives (replace signed_ema)
- **Mechanism.** Any anchor-driven correction must NOT override the live update
  *direction* (the falsified failure of signed_ema, DEEP_FINDINGS §C). Candidates that
  preserve direction: error-feedback (Lever 2, the principled one); trust-weighting by
  per-coordinate sign-agreement confidence; magnitude-only preconditioning.
- **Expected effect.** Lower priority than 1–3 because the verified-substrate
  experiments (blend/inject) came back **inert** (EXP-23) by orthogonality, and signed-
  replacement came back **net-harmful** (EXP-25). The anchor's role is most defensible
  as the *re-grounding flush* for error-feedback (#24's "flush against stale M every
  K"), not as a standalone correction. **[PENDING-MECHANIST Q3]** decides whether blend
  has any live directional content worth reviving.
- **Cost.** Design-heavy; defer until Levers 1–3 report.

### Lever ranking summary

| rank | lever | role | beats-dense path | cost |
|---|---|---|---|---|
| 1 | KL brake | **enabler** — safe to inject noise | indirect (unlocks 3) | in flight |
| 2 | Error-feedback (#24) | **de-bias** — keep ξ, kill b | gets to dense fidelity + keeps noise | low |
| 3 | **Rank-as-temperature** | **the surplus** — tune exploration | DIRECT (r* may exceed dense) | low |
| 4 | Entropy reg | **control** — isolate the benefit | comparison only | low |
| 5 | Direction-preserving corr. | substrate hygiene | unlikely standalone | deferred |

---

## 4. The falsifiable experiment program

**Shared surface (fixed):** Qwen2.5-1.5B-Instruct, GSM8K, GRPO, batch 128 / mini 64,
lr 1e-6, n=8, resp 16384, 50 steps, 4–8 GPU. **Metric:** `val@50`
(`val-core/openai/gsm8k/acc/mean@1`). **Bar:** dense = 0.7536. **Mandatory guardrail
on every arm:** the ENTROPY_COLLAPSE_WATCH T1–T7 triggers (length-explosion is the
discriminator — kill any arm that fires composite-RED). **Surplus claim is two-sided:**
val@50 > 0.7536 by more than run-to-run noise (estimate noise from the dense + A0
spread ≈ ±0.005–0.01; require a margin, ideally a 2-seed confirmation of the winning
arm).

### The ONE experiment to run first: **EF + rank-as-temperature sweep under a KL guardrail**

> **EXP-SURPASS-1 — "Is there a compressed rank that beats dense?"**
>
> A 2×3 grid on the verified PowerSGD substrate, KL guardrail ON (coef 0.001),
> error-feedback the only correction (signed_ema OFF, anchor as EF re-grounding flush):
>
> | | r = 38 (high noise) | r = 77 (current) | r = 154 (low noise) |
> |---|---|---|---|
> | **EF off** | exploration, biased | the floor ref | near-dense fidelity |
> | **EF on** | exploration, de-biased ← **the surplus cell** | de-biased | de-biased |
>
> Plus two controls run on the SAME surface: **dense+KL(0.001)** (the real bar — KL is
> on everywhere, so the comparison is comm-eff+KL vs dense+KL, not vs no-KL dense) and
> **dense+entropy-bonus** (Lever 4 control).

**Why this one first.**
1. It tests the thesis **directly**: the surplus cell (low rank, EF on, KL on) is the
   precise theoretical sweet spot — maximal zero-mean exploration noise (low r), bias
   removed (EF), collapse-proofed (KL). If *any* cell exceeds dense+KL, the operator's
   thesis is confirmed with a number.
2. It is **falsifiable and decisive**: monotonic "more rank = better, plateau at dense"
   falsifies compression-as-exploration (compression is then pure loss, as in SFT). A
   non-monotone curve with an interior peak above dense **confirms** it. Either way we
   learn the shape.
3. It **isolates** the exploration benefit via the two controls: comm-eff+KL vs
   dense+KL removes KL as a confound; the dense+entropy-bonus arm tells us whether
   compression's exploration is just entropy-reg-by-another-name or structurally
   different (gradient-space).
4. It **fits the budget**: 6 comm-eff arms + 2 controls × 50 steps on 4–8 GPU, reusing
   one box (memory: GPU-idle box reuse). It is built entirely from already-verified
   machinery (PowerSGD codec green, anchor substrate verified in #25) plus EF (#24,
   now unblocked) and a config flag for KL.

**Pre-registered predictions (so the result is honest):**
- If compression-as-exploration is real: `val(r=38, EF on, KL on) > val(dense+KL)`,
  with `r=77` intermediate and `r=154` ≈ dense. The win comes from the LOW-rank arm.
- If it is not (SFT-like): val is monotone increasing in r, saturating at dense; no
  cell beats dense+KL. EF still closes most of the plain-PowerSGD gap (recovers to
  ~dense), which is a real comm-savings result even without the surplus.
- The α=0 collapse arm + KL diagnostic (in flight) is the **pre-flight gate**: if KL
  does NOT bound the α=0 collapse, the guardrail is unreliable and we must add a hard
  length cap before running the low-rank arms.

**Decision rule.** Surplus CONFIRMED ⇒ 2-seed the winning cell, then sweep finer rank
around `r*` and write the result up as the headline. Surplus REJECTED but EF reaches
dense ⇒ pivot the headline to "comm-eff matches dense at materially lower comm" (the
GOAL.md parity+savings bar, with the exploration question answered negative). Either is
a publishable, decisive outcome.

---

## 5. Open dependencies on mechanist (to close before finalizing)

- **[Q1 — pivotal]** Is PowerSGD's `(I−P)·G` residual zero-mean/step-decorrelated
  (→ Lever 3 valid as-is) or a coherent low-rank bias (→ needs Lever 2 first)? + an SNR
  estimate. **This determines whether rank-as-temperature works directly or only after
  error-feedback.**
- **[Q2]** Three-way split of the α=0 sign-disagreement (staleness / compression / EMA)
  — confirms whether the collapse is "adversarial because stale" (freshen) or "the sign
  operator is fundamentally wrong" (abandon — current read).
- **[Q3]** Does blend's M_anchor injection carry direction-correct signal vs the true
  dense gradient, or only magnitude? Decides if Lever 5 has any live content.
- **[Q4]** Is there an EXISTING fingerprint (higher sustained entropy / wider train↔
  rollout gap) that compression already adds exploration in the healthy arms? Empirical
  seed for the whole thesis.

---

## 6. One-paragraph thesis (the elevator version)

In SFT, gradient compression is pure information loss because the objective is a fixed
target — every dropped bit is descent lost. In RL the objective is non-stationary,
exploration-limited, implicitly regularized by gradient-sign cancellation, and coupled
to a self-shaping reward, so the *sign* of compression's effect depends entirely on
whether the compression deviation is a **zero-mean fluctuation** (rides the implicit
regularizer, perturbs the policy off its greedy path, sustains exploration → can BEAT
dense via SGD-noise flat-minima + RL exploration) or a **persistent structured bias**
(breaks the regularizer, accumulates under no-KL/no-entropy, drives the length-explosion
reward-hack → collapse, the α=0 result). The signed_ema collapse is the negative image
that locates the cliff; the path to surpass dense runs along the opposite edge: a small
PowerSGD rank (maximal zero-mean exploration noise) with **error-feedback** removing the
bias and a **small KL** guardrail making the noise safe to inject — and the decisive,
falsifiable test is whether some compressed rank `r* < H` beats dense+KL on GSM8K val@50.
