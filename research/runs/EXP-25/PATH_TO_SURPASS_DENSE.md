# Path to Surpass Dense — Compression-as-Exploration in Communication-Efficient RL

**Status: CONVERGED with mechanist (round 2).** This document develops the operator's
thesis — *that communication-efficient training can SURPASS dense in RL (unlike SFT, where
compression is pure information loss)* — into a rigorous, falsifiable program. It is the
`strategist` deliverable for the `surpass-dense` team (task #2). The bias/noise
characterization the central argument hinges on was grounded by `mechanist`
(`COLLAPSE_GRADIENT_FLOW_ANALYSIS.md`, task #1) over a full converging exchange — their
findings **falsified the naive routes and reshaped the plan** (see §2.6, §5). Net honest
position: no >dense edge has been demonstrated by what has run; the surviving testable
surpass hypothesis is *variance-controlled zero-mean perturbation*, gated by a single cheap
codec-free experiment (the Gaussian-noise probe, §4). Citations:
`runs/EXP-25/COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` (mechanist), `DEEP_FINDINGS.md`,
`ENTROPY_COLLAPSE_FINDINGS.md`, `diagnostics/ENTROPY_COLLAPSE_WATCH.md`, issue #24,
`verl/workers/comm_eff/{spectral_filter,activation_mask}.py`, and W&B
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
| **plain PowerSGD r77 (frozen Q)** | `(I−P)·G` dropped — the off-low-rank-subspace residual | **MECHANIST-RESOLVED [Q1]: deterministic structured BIAS**, not zero-mean noise. Q frozen for the whole step ⇒ the SAME (H−r) directions dropped every forward. Low-magnitude (recon_rel_err ~0.025, stationary), direction-faithful (kept subspace = dominant energy) | val ~0.74 ≈ dense — benign | **dense-grade signal; NO exploration-noise channel** (frozen-Q PowerSGD is the worst case for the thesis) |
| inject (γ=1) | adds orthogonal scale-matched M_anchor component | tiny (‖M‖≪‖G‖ after rescale) — inert | inert (EXP-23 A2 0.6967) | neither helps nor hurts |
| blend (η=0.5) | convex pull toward scale-matched M_anchor; cos(G,M)≈0.001 | shrinks step to 0.71× along ~orthogonal M | inert (EXP-23 A3 0.6861) | direction-uncorrelated; not a live surpass lever |

**MECHANIST [Q1] resolved the pivotal cell — and it reshapes the whole thesis.** The
detailed gradient-flow analysis (`COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` §2) settles that
plain PowerSGD's residual is the **opposite** of zero-mean exploration noise: with Q
*frozen* for the whole global step, the same off-subspace directions are dropped on every
forward — a *deterministic, structured, low-magnitude, direction-faithful BIAS*
(`reconstruction_rel_error` ~0.025 and **stationary** across the run; the PowerSGD-only
control ties dense at 0.741). There is **no fresh per-step randomness** to act as an
SGD-noise exploration source. This **falsifies "rank-as-temperature via PowerSGD noise"
as originally drafted** — that lever assumed a zero-mean fluctuation that does not exist
for frozen-Q PowerSGD. The consequence is decisive for the program: *the surpass edge
cannot come from "PowerSGD compression noise explores," nor from "the anchor corrects the
(tiny) compression bias" (that is mechanically PARITY, §2.6).* The thesis must be rebuilt
on channels that supply signal **dense does not already use** — which is exactly what §2.6
and the revised levers (§3) do.

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

### 2.4 Prior art — the thesis is grounded in known theory but unexplored in RL

Two of the three pillars are well-established theory; the third (compression-as-
exploration *in RL*) appears genuinely novel, which is both the opportunity and the
reason there is "no recipe to copy" (GOAL.md):

- **Zero-mean gradient noise → flat minima (PILLAR 1, established).** Implicit Gradient
  Regularization (Barrett & Dherin, ICLR 2021) shows GD implicitly penalizes large
  gradient norm and biases toward flat minima that generalize better and are robust to
  parameter perturbation. The stability analysis of SGD noise (arXiv 2207.02628) adds
  the load-bearing nuance for us: it is the **alignment** of the noise with the loss
  geometry that selects flat minima — i.e. the noise must be the *right kind* (zero-mean
  along the right directions), exactly the bias-vs-fluctuation distinction in §2.1. This
  is why "more noise" is not automatically good and why rank-as-temperature (Lever 3) is
  a *sweep for an interior optimum*, not "minimize rank."

- **Regularization prevents reward-hacking (PILLAR 2, established + directly on-point).**
  "Gradient Regularization Prevents Reward Hacking in RLHF and RLVR" (arXiv 2602.18037)
  is almost exactly our α=0 failure mode: an under-regularized RL objective gets hacked
  (our length-explosion, DEEP_FINDINGS §A2) and an explicit regularizer closes the hack
  channel. This is independent literature support for Lever 1 (KL brake) as the enabler
  that makes noise injection safe. It also reframes the whole program correctly: the
  collapse risk is *reward-hacking under insufficient regularization*, not "compression
  is bad" — so the fix is to regularize and then inject, not to stop compressing.

- **Why signed_ema specifically failed (compression theory).** Sign-based gradient
  compression is a known technique (signSGD, Sparse-SignSGD with **majority vote**,
  arXiv 2302.07475) — but its convergence guarantees REQUIRE a variance-reduction /
  majority-vote step that makes the transmitted sign an *unbiased* estimator of the true
  sign. signed_ema has no majority vote; it takes the sign from a single **stale β=0.95
  EMA**, which is a *biased* sign estimator (DEEP_FINDINGS §B3: wrong on ~50% of mass).
  So signed_ema is sign-compression with the unbiasing step removed — the literature
  predicts exactly its failure. Error-feedback (Lever 2, EF/PULSELoCo per issue #24) is
  the canonical unbiasing mechanism for the *magnitude* path; it is the principled
  successor.

- **The novelty (and the risk).** The communication-efficient-training literature
  (quantization, sparsification, error-feedback, low-rank PowerSGD) is overwhelmingly
  **federated / supervised**; it treats compression as something to *minimize the harm
  of*, and measures success as "matches dense at lower comm." A survey-level scan turns
  up essentially **no** work framing compression noise as a *productive exploration
  source in RL/RLHF*. So the operator's thesis is unexplored territory: the upside is a
  genuinely new result (compression > dense in RL); the risk is that the SFT/federated
  intuition ("compression is pure loss") simply transfers and the rank curve saturates
  at dense. EXP-SURPASS-1 (§4) is designed to settle exactly that, either way.

### 2.5 The entropy "fingerprint" — investigated and DEMOTED (dense is *confident*, not under-exploring)

There is a suggestive existing signal that compression is *already* doing the
exploration the thesis predicts — but it must be read carefully and is gated on
mechanist [Q4]. At matched early steps on the identical model/data/no-KL surface
(W&B `actor/entropy`):

- **dense** (`5e2jpho9`, val 0.7536): entropy is **FLAT and LOW** — ~0.37→0.42 over
  steps 1–8, ending 0.13 @s48, smooth throughout. Crucially, this is a **confident, not
  collapsed** regime: dense keeps stable response length (~280) and clip_ratio ≈ 0 and
  reaches the best val. A confident low-entropy policy on GSM8K is *correct* (the model
  is sure of the right tokens), not pathological (DEEP_FINDINGS §A2 point 1 makes the
  same point). This is the SAME dense-low-entropy regime the team observed earlier.
- **A0 PowerSGD r77+clean5** (`oquyeic3`, val 0.7415): entropy is **HIGH and NOISY** —
  bouncing 5.7→9.0→0.4→7.8 over steps 1–8, ending 1.55 @s48 — yet it stays healthy
  (bounded length, near-dense val).

If real, this is almost a *direct* picture of the thesis: **dense exploits at low
entropy; compression keeps the policy at high, noisy entropy (exploring) while still
reaching near-dense val — i.e. compression sustains exploration without paying for it in
performance.** The decisive missing piece is whether a comm-eff arm can be kept from
collapsing *while* exploring (that is exactly what the KL+length brakes in §4 are for) —
if so, the high-entropy exploring policy has room to find a better basin than dense's
confident one.

**MECHANIST [Q4] DEMOTED THIS to "not a clean signal."** The detailed analysis
(`COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` §6.1) shows **dense trains down to entropy 0.122 —
LOWER than ANY non-collapsing comm-eff arm — with bounded length and the BEST val**. A
confident low-entropy GRPO policy on GSM8K is *correct*, not collapsed. So "comm-eff sits
at higher entropy ⇒ it explores more ⇒ it should beat dense" does **not** follow: dense's
low entropy is exploitation of a *correct* policy, and the comm-eff arms' high/noisy
`actor/entropy` (the 5→9→0.4 bouncing) is most likely a metric/scale artifact (the
parallel ~180× gap in `rollout_probs_diff_mean`, 0.0035 dense vs 0.62 comm-eff, points
to a metric-definition difference, not a real diversity gap). **I am NOT using the
entropy fingerprint as a surpass argument.** Higher entropy is not the goal; *the right
kind of update* is. The surpass case must rest on the mechanism in §2.6, not on this
observation.

### 2.6 The parity-vs-surpass ceiling, and the three channels that survive it

Mechanist's sharpest result (`COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` §8.1) is a hard ceiling
on the *obvious* comm-eff story and it must be stated bluntly:

> **"Anchor corrects the compression bias" is mechanically PARITY, not surpass.** The
> recoverable bias is the dropped off-subspace component `(I−P)·g`, whose relative size
> is bounded by the dropped activation energy ≈ `recon_rel_err² = 0.025² ≈ 0.0006` (~0.06%
> of activation energy; the boundary weight gradient is genuinely low-rank, EXP-20/#21).
> A correction that adds this back recovers single-digit-% of the update at most — it can
> let you drop the periodic clean step (pure comm savings) but it cannot exceed dense,
> because **the stale clean anchor gradient is information dense already has fresh every
> step.** To SURPASS dense, the extra signal must come from a channel *dense does not
> already use.*

This kills two framings at once: (a) PowerSGD-noise-as-exploration (no zero-mean noise to
exploit, §2.2) and (b) anchor-corrects-compression-bias (parity-only, above). The honest
question becomes: **what information channel does a communication-efficient trainer have
that a dense single-step trainer does not?** After a full converging exchange with
mechanist (their `COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` §8.2/§9), the candidates and their
verdicts are:

- **Channel A — Variance-reduced / look-ahead gradient (anchor-EMA-as-momentum). DEMOTED
  by mechanist's decisive objection.** The idea: the anchor's β=0.95 EMA `M` is a cross-
  step variance-reduced gradient dense cannot compute in one step (n=8 GRPO advantages are
  noisy). The objection (mechanist §8.2): **Adam already supplies fresh β1=0.9 momentum**,
  so a *stale* β=0.95 EMA of a K-old, clean, smoothed gradient adds little the live
  optimizer's own momentum doesn't already provide — and it pays a staleness cost the live
  momentum doesn't. The anchor is therefore best used only for **parity-recovery** (drop
  the periodic clean step) — NOT a surpass lever. Retired from the surpass path; kept only
  as the error-feedback re-grounding flush.

- **Channel B — Compression as an implicit trust-region / subspace regularizer.** Frozen-Q
  PowerSGD constrains every update to the dominant-energy subspace `P`. In a non-stationary
  RL objective this is an implicit regularizer (constrain the step → flatter, less batch-
  overfit policy, §2.4 by *constraint* not noise). It can be tested for free via the rank
  sweep (lower r = tighter trust region). **Risk (mechanist §2.2): the constraint is too
  WEAK** — direction-faithful, recon_err only 2.5%, SNR ~42:1 — so deterministic PowerSGD
  likely doesn't regularize enough to beat dense. Kept as a cheap secondary test, not the
  lead.

- **Channel C — Explicit zero-mean exploration noise (the operator's literal thesis,
  done right). THE LEAD CHANNEL.** Mechanist's key empirical addition: the codebase
  *already has* a genuinely zero-mean knob — the **prf_mask + rescale** codec is inverted
  dropout (`E[h̃]=h`, `activation_mask.py:243`, re-sampled per global step), so its residual
  IS zero-mean by construction. BUT plain high-p mask is **too high-variance** (∝p/(1−p) ≈
  19× at p=0.95) and **stalls** (the known GOAL.md result). So the literal "compression
  noise = exploration" route is real but unusable as-is. The fix mechanist specifies (§9):
  a **variance-CONTROLLED unbiased perturbation** — three properties at once that NO
  existing codec (dense / prf_mask / PowerSGD) has: (a) **zero-mean** (explores, not
  biases — PowerSGD fails this), (b) **controlled/low variance** (trains, not stalls — the
  p=0.95 mask fails this), (c) **tunable as an exploration TEMPERATURE** (the bias-variance
  tradeoff is a swept axis). Realized as: rescaled mask at **swept/annealed p + error-
  feedback** re-injecting the dropped residual to lower variance while staying unbiased;
  with an explicit **zero-mean Gaussian-noise probe** as the codec-free decoupled control.

**The reframed thesis (converged with mechanist).** The operator's intuition — "RL under-
explores, so zero-mean perturbation can help where it would only hurt in SFT" — is
**correct in spirit and the right thing to test**, but mechanist has shown the naive routes
fail: deterministic PowerSGD has no zero-mean noise (Channel A/B parity ceiling), and the
one genuinely zero-mean codec (the mask) is too high-variance to train. The surviving,
honest surpass hypothesis is **Channel C with variance control**: *zero-mean, low-variance,
temperature-tunable* perturbation. And the decisive first test is not a codec at all — it
is the **explicit Gaussian-noise probe** (§4), the operator's thesis stripped to its
irreducible core with zero codec confound: *does ideal zero-mean tunable noise beat dense
GRPO at all?* If yes, the variance-controlled mask is the comm-efficient realization; if no,
compression-as-exploration is falsified for this surface and we stop. **Crucial honest
caveat (mechanist §8.2):** no >dense edge has been demonstrated by anything that has run;
this section specifies the experiment that *would* show one, grounded in the §1 theory of
why it could exist — it is a hypothesis with a clean test, not a claimed result.

---

## 3. The ranked levers toward beating dense (REVISED post-mechanist)

Ranked by `P(moves us above dense) × leverage / cost`, **fully converged with mechanist**.
Two big changes from the first draft: the old top lever (rank-as-exploration-noise via
PowerSGD) was falsified (frozen-Q PowerSGD has no zero-mean noise, §2.2); error-feedback
and the anchor-EMA were demoted to *parity* tools (the §2.6 ceiling + the Adam-momentum
objection). What survives as a surpass lever is **variance-controlled zero-mean
perturbation** (Channel C done right) — and the decisive first test is the codec-free
**Gaussian-noise probe**. Every lever is direction-preserving (the hard constraint below);
the KL brake + length cap are the enablers that make any perturbation safe to push.

### Lever 1 — Variance-controlled UNBIASED perturbation (Channel C, the lead surpass lever)
- **Mechanism (mechanist §9.1).** The genuinely zero-mean knob already in the stack is the
  prf_mask + rescale codec (inverted dropout, `E[h̃]=h`, re-sampled per global step,
  `activation_mask.py:243`). Plain high-p mask stalls because its variance is huge
  (∝p/(1−p) ≈ 19× at p=0.95). Fix it to satisfy all THREE required properties at once —
  **zero-mean + controlled-variance + temperature-tunable** — via: (i) **sweep/anneal p**
  as the exploration temperature (lower p = less noise; anneal high→low = explore early /
  exploit late), and (ii) **error-feedback on the dropped activation residual `(I−mask)·h`**,
  re-injecting it next step so the estimator stays unbiased at *lower* variance (the exact
  fix the high-p mask lacks).
- **Why it can beat dense.** This is the operator's literal "compression noise = exploration"
  thesis, now correctly engineered: a zero-mean, low-variance, tunable perturbation is the
  one object that can supply *productive* exploration (SGD-noise flat-minima §2.4 + RL
  exploration C2) without the bias that collapsed signed_ema or the variance that stalls the
  raw mask. Dense sits at zero injected noise; if the optimal noise temperature for this
  policy/task is > 0, the swept p finds it.
- **Falsifier (mechanist §9.1).** If no p (fixed or annealed), with EF on, beats dense, then
  compression-as-exploration is dead for this surface.
- **Cost.** Reuses the existing mask codec + the error-feedback machinery #24 was scoped
  for. p-sweep `{0.5, 0.7, 0.9}` ± anneal × EF on = 3–4 arms. Medium (needs the EF wiring).

### Lever 2 — Explicit zero-mean Gaussian-noise probe (the decisive SCIENCE GATE)
- **Mechanism (mechanist §9.2).** Add `σ·N(0,1)` to the boundary activation (or the weight
  gradient), `σ` the temperature, decorrelated per step, exactly zero-mean by construction,
  variance fully controlled by σ. Run it as a **mechanism probe, not a comm-eff arm** (it
  saves no bytes) — it isolates "noise-as-exploration" from every codec / byte-budget
  confound.
- **Why this is the first thing to run.** It is the operator's thesis stripped to its
  irreducible core: *does ideal, perfectly-controlled, zero-mean tunable noise beat dense
  GRPO AT ALL?* If even this cannot beat dense, compression-as-exploration is falsified
  **independent of any codec**, and we never pay to build the variance-controlled mask
  (Lever 1). If it CAN, the thesis is proven in principle and Lever 1 becomes the
  comm-efficient realization of a demonstrated effect. Cheapest, cleanest, most decisive
  gate in the program.
- **Cost.** Tiny code (add noise at the boundary), a σ-sweep `{small, med, large}` × 50
  steps = 3 arms; lowest engineering cost of any lever.

### Lever 3 — Compression-as-trust-region rank sweep (Channel B, cheap secondary)
- **Mechanism.** Frozen-Q PowerSGD constrains updates to the dominant-energy subspace — an
  implicit regularizer (constrain the step → flatter, less batch-overfit policy, §2.4 by
  constraint not noise). Rank `r` is the trust-region radius (lower r = tighter).
- **Why it can beat dense / risk.** If dense's full-rank low-energy tail is mostly batch-
  overfitting noise, projecting it out is free regularization. **But mechanist §2.2 says the
  constraint is likely too WEAK** (direction-faithful, SNR ~42:1) to beat dense — so this is
  a cheap secondary test, not the lead. Discriminator: non-monotone val(r) with an interior
  peak above dense confirms; monotone-saturating-at-dense falsifies.
- **Cost.** Rank sweep `r ∈ {38, 77, 154}` vs dense+KL × 50 steps; cheap; reuses green codec.

### Lever 4 — KL brake + length cap: the keystone enablers
- **Mechanism.** Small KL-to-reference (`use_kl_loss=true`, coef 0.001) bounds the policy
  excursion (divergence axis); an unconditional `max_response_length` cap 1024–2048 + length
  penalty closes the reward-hack (degeneration axis). Neither supplies surplus — together
  they are the **guardrails that make any perturbation (Levers 1–3) safe to push past
  dense's operating point** without the collapse cliff.
- **Status.** A0+KL(0.001) diagnostic in flight (team-lead relaying). Mechanist predicts
  (§7) it arrests the length explosion but does NOT itself beat the PowerSGD-only 0.741 —
  consistent with KL being the enabler, not the surplus. Prior-tightener, not a blocker (§4).
- **Caveat.** KL/length-cap change the FIXED no-KL/no-entropy surface — run as a deliberate,
  labelled new lineage, not a silent drift (DEEP_FINDINGS §d.2).

### Lever 5 — Anchor-EMA + Error-feedback PowerSGD: PARITY tools (the comm-savings banked win)
- **Re-ranked OFF the surpass path (mechanist §8.1/§8.2).** Two demoted ideas folded here:
  (a) the anchor-EMA-as-momentum (Channel A) — **Adam already has fresh β1=0.9 momentum**, so
  a stale β=0.95 clean EMA adds little and pays a staleness cost; (b) error-feedback on the
  PowerSGD residual — recovers only the ~0.06%-energy compression bias. Both are mechanically
  **parity + comm-savings** (they let you drop the periodic clean step at ~dense quality),
  NOT surpass. Run EF-PowerSGD **in parallel** as the honest comm-efficiency deliverable
  (GOAL.md parity+savings) while Levers 1/2 chase the exploration edge.

### Direction-preserving correction is a HARD CONSTRAINT, not a lever
Mechanist's central result: never use a stale signal to OVERRIDE the live update direction
(the falsified signed_ema primitive). Every lever is direction-preserving by construction
(Lever 1's residual is zero-mean and re-injected, Lever 2's noise is additive, Lever 3
projects, Lever 5's EF adds back the dropped component). The dead operators (signed_ema,
inject, blend) are retired — blend/inject inert by orthogonality, signed_ema net-harmful.

### Lever ranking summary (CONVERGED)

| rank | lever | role | beats-dense path | mechanist verdict | cost |
|---|---|---|---|---|---|
| 1 | **Variance-controlled unbiased mask + EF** | Channel C realization | zero-mean low-var tunable noise (the thesis, engineered) | the §9.1 forward primitive | medium |
| 2 | **Gaussian-noise probe** | the SCIENCE GATE | does ideal noise beat dense at all? | the §9.2 decoupled control | lowest |
| 3 | Rank-as-trust-region sweep | Channel B regularization | constrain the step | likely too weak (§2.2) | cheap |
| 4 | KL brake + length cap | enablers | make 1–3 safe to push | confirmed-good guardrails | in flight |
| 5 | Anchor-EMA / EF-PowerSGD | PARITY + comm-savings | recover toward dense (drop clean step) | NOT surpass (§8.1) | low |

---

## 4. The falsifiable experiment program

**Shared surface (fixed):** Qwen2.5-1.5B-Instruct, GSM8K, GRPO, batch 128 / mini 64,
lr 1e-6, n=8, 50 steps, 4–8 GPU. **Metric:** `val@50`
(`val-core/openai/gsm8k/acc/mean@1`). **Bar:** dense = 0.7536. **Surplus claim is
two-sided:** val@50 > 0.7536 by more than run-to-run noise (estimate noise from the
dense + A0 spread ≈ ±0.005–0.01; require a margin, ideally a 2-seed confirmation of the
winning arm).

**TWO ORTHOGONAL BRAKES, both ON in every arm (defense-in-depth).** The collapse arms
died by a *response-length-explosion reward-hack* (clip_ratio 0.46/0.92 to the 16384
cap, DEEP_FINDINGS §A2); KL and a length cap address *different* failure axes and we use
both unconditionally — the length cap costs nothing in the non-collapse cells and stops a
low-rank arm from burning hours generating 16K-token garbage:
1. **KL brake (divergence axis).** `use_kl_loss=true`, coef 0.001 — bounds the policy's
   excursion from the reference (Lever 1). This is the *entropy/divergence* guardrail.
2. **Length brake (degeneration axis), UNCONDITIONAL.** Cap `max_response_length` to
   **1024–2048** (down from 16384) and/or add an explicit length penalty in the reward.
   GSM8K healthy responses are ~170–290 tokens (DEEP_FINDINGS §A2), so a 1024–2048 cap is
   ~4–8× headroom over the healthy regime — it never bites a non-degenerate arm but hard-
   stops the reward-hack channel and bounds per-arm cost. **This is baked in regardless
   of the KL diagnostic result**: KL is the divergence brake, the cap is the orthogonal
   degeneration brake. (Note: dropping the cap from 16384 also speeds every arm, so the
   whole grid is cheaper than the EXP-25 runs.)
3. **Mandatory monitor on every arm:** the ENTROPY_COLLAPSE_WATCH T1–T7 triggers
   (length-explosion is the discriminator — kill any arm that fires composite-RED early,
   don't wait for step 50).

**Required instrumentation on EVERY surpass arm (mechanist §9, HARD gate).** Log per-step:
(a) the **dense-vs-perturbed update COSINE** (the never-logged EXP-20 success criterion) —
without it we cannot tell "noise explored productively" from "noise Adam just averaged
away"; (b) **sign-agreement(perturbed, dense) and the policy entropy + IS-gap** (to confirm
the perturbation moved the policy, not just the metric). These two numbers are the
difference between a real result and an uninterpretable one.

### The ONE experiment to run first: **the zero-mean Gaussian-noise probe (the decisive gate)**

The first experiment is **not a codec** — it is the operator's thesis stripped to its
irreducible core. Mechanist's §9.2 decoupled control, elevated to the lead because it is the
cheapest, cleanest, most decisive test in the entire program: it answers *does ideal,
perfectly-controlled, zero-mean tunable noise beat dense GRPO AT ALL?* with zero codec or
byte-budget confound. Everything downstream (the variance-controlled mask, the comm-eff
payoff) is conditional on this gate.

> **EXP-SURPASS-1 — "Does ideal zero-mean tunable noise beat dense GRPO?"**
>
> All arms on the fixed surface, KL(0.001) + length-cap(1024–2048) brakes ON everywhere,
> 50 steps, the two instrumentation gates logged:
>
> | arm | perturbation | role |
> |---|---|---|
> | **D0** dense + KL + length-cap | none | **the bar** (brake-matched, so the comparison isolates the noise) |
> | **G_lo** dense + `σ_lo·N(0,1)` on boundary activations | small zero-mean noise | low exploration temperature |
> | **G_mid** dense + `σ_mid·N(0,1)` | medium | the candidate sweet spot |
> | **G_hi** dense + `σ_hi·N(0,1)` | large | high temperature (expect over-noise → ≤ dense) |
>
> Noise is added at the boundary activation, decorrelated per step, exactly zero-mean.
> `σ` is the exploration temperature; scale σ relative to the measured boundary-activation
> RMS so the sweep is interpretable. (It saves no comm — it is a *mechanism probe*, run as
> a science control, not a comm-eff arm.)

**Why this one first.**
1. **It is the irreducible test of the operator's thesis.** If even ideal zero-mean tunable
   noise cannot beat dense, "compression-as-exploration" is falsified for this surface —
   independent of any codec — and we save all the mask+EF engineering. If some σ DOES beat
   dense, the thesis is proven in principle and the codec work has a demonstrated target.
2. **It removes every confound.** No byte budget, no basis warmup, no staleness, no bias —
   just `G_t + ξ`, `ξ` exactly zero-mean. Whatever it shows is about *noise-as-exploration*,
   nothing else. That is precisely the isolation mechanist demanded (§9.2).
3. **Cheapest + lowest-engineering in the program** (add noise at the boundary; a 4-arm
   σ-sweep × 50 steps on one box), and it directly de-risks the more expensive Lever 1.

**Pre-registered predictions (honest, mechanism-grounded — and the honest prior is SKEPTICAL).**
- **Thesis confirmed:** an interior `σ*` gives `val(σ*) > val(D0)` by > noise (a non-
  monotone temperature curve with a peak above dense). The win is at *medium* σ; G_hi
  degrades (over-noise).
- **Thesis falsified (mechanist's prior, §8.2):** val is monotone *decreasing* in σ — any
  zero-mean noise only adds variance that Adam averages away and hurts, so dense (σ=0) is
  best. Mechanist's standing position is that no >dense edge has been demonstrated and the
  existing entropy "fingerprint" is anti-correlated with val — so the honest base rate
  favors this outcome. Stating it up front keeps the result credible either way.
- **The COSINE gate (a) is the tell:** if a helping σ shows update-cosine to dense
  *dropping below ~0.98* while val *rises*, that is genuine productive exploration (the
  noise is steering, not just jittering). If val rises with cosine ≈ 1.0, the "win" is noise
  Adam absorbed and is likely seed noise — demand the 2-seed confirmation.

**Decision rule.**
- **Gate PASSES (some σ beats dense):** proceed to **EXP-SURPASS-2 = Lever 1**, the
  variance-controlled UNBIASED mask (swept/annealed p + error-feedback) — the *comm-
  efficient realization* of the now-demonstrated effect. This is the headline path.
- **Gate FAILS (dense is best at σ=0):** compression-as-exploration is falsified for this
  surface. Pivot the deliverable to the honest **parity + comm-savings** result: run
  **EF-PowerSGD** (Lever 5) to bank the comm win at ~dense quality (drop the clean step),
  and optionally the cheap **Channel-B rank sweep** (Lever 3) as a last check on the
  regularization route (mechanist's prior: too weak to surpass).

### Why this ordering (Gaussian probe → mask → parity) and not the old rank-first plan
The first draft's EXP-SURPASS-1 (rank × EF grid) assumed low-rank PowerSGD injects zero-mean
*exploration* noise. Mechanist falsified that: frozen-Q PowerSGD's residual is a deterministic
*bias*, not noise (§2.2), and EF only buys parity (§2.6). The genuinely zero-mean codec (the
mask) exists but stalls from variance. So the decisive question moved upstream — *is
zero-mean noise useful in RL here at all?* — and the cleanest way to answer it is the
codec-free Gaussian probe, BEFORE paying to engineer the variance-controlled mask. This
spends the first, cheapest box on the single experiment that can kill or confirm the whole
thesis, and only then invests in the comm-efficient realization.

---

## 5. Mechanist convergence — what is settled (was: open dependencies)

All four grounding questions are CLOSED, converged with mechanist
(`COLLAPSE_GRADIENT_FLOW_ANALYSIS.md`). The resolved answers reshaped this document:

- **[Q1 — RESOLVED]** PowerSGD's `(I−P)·G` residual is a **deterministic structured BIAS**
  (frozen Q, same off-subspace dropped every step), low-magnitude (recon ~0.025, SNR ~42:1,
  0.06% energy dropped), direction-faithful — **NOT zero-mean noise**. ⇒ killed rank-as-
  exploration-via-PowerSGD; the genuinely zero-mean knob is the *mask*, not PowerSGD.
- **[Q2 — RESOLVED]** The ~50% sign-disagreement is **structural** (coin-flip of two
  estimators of a near-zero-mean GRPO gradient), already 50% at the first warm step, uniform
  across layers — NOT staleness/compression/EMA. ⇒ signed-replacement is fundamentally
  wrong; abandon, don't freshen.
- **[Q3 — RESOLVED]** blend/inject are inert by orthogonality (`cos(G,M)≈0.001`); no live
  directional content. Retired with signed_ema.
- **[Q4 — RESOLVED]** The comm-eff "high entropy" is a **codec-warmup artifact** (coincides
  with recon 0.97→0.14 as Q warms), and even the subtler sustained ~0.1-nat entropy edge in
  the trained regime is **anti-correlated with val** (slower convergence, not exploration).
  ⇒ the entropy fingerprint is NOT a surpass argument; dropped from the case.
- **Joint converged verdict:** no >dense edge has been demonstrated by anything that has
  run; the surviving testable surpass hypothesis is **variance-controlled zero-mean
  perturbation** (Lever 1), gated by the **Gaussian-noise probe** (the EXP-SURPASS-1
  decision). The anchor-EMA and error-feedback are parity/comm-savings tools, not surpass.

---

## 6. One-paragraph thesis (the elevator version, converged)

In SFT, gradient compression is pure information loss because the objective is a fixed
target — every dropped bit is descent lost. In RL the objective is non-stationary,
exploration-limited, implicitly regularized by gradient-sign cancellation, and coupled to a
self-shaping reward, so a perturbation's effect depends entirely on whether it is a
**zero-mean fluctuation** (rides the implicit regularizer, perturbs the policy off its
greedy path → could BEAT dense via SGD-noise flat-minima + RL exploration) or a
**persistent structured bias** (breaks the regularizer, drives the length-explosion reward-
hack → the α=0 collapse). Mechanist's gradient-flow analysis falsified the naive routes:
frozen-Q PowerSGD's residual is a *deterministic bias*, not noise (so "low rank = more
exploration noise" is false), the genuinely zero-mean codec (the mask) *stalls* from
variance, and the stale clean anchor is information dense already has fresh (so anchor
correction is parity, not surpass; Adam's own momentum already covers the variance-reduction
channel). What survives is the operator's thesis *correctly engineered*: a **zero-mean,
low-variance, temperature-tunable** perturbation. The decisive, falsifiable test is the
codec-free **Gaussian-noise probe** — *does ideal zero-mean tunable noise beat dense GRPO at
all?* — with the dense-vs-perturbed update cosine logged to tell productive exploration from
noise-Adam-averages-away. If it passes, the **variance-controlled unbiased mask + error-
feedback** is the comm-efficient realization; if it fails, the honest deliverable is
**error-feedback PowerSGD** for parity at materially lower comm. The honest prior (mechanist):
no >dense edge has yet been shown — this is a clean hypothesis with a one-experiment gate,
not a claimed win.

---

## 7. References

**Internal (this fork).**
- `runs/EXP-25/COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` (**mechanist, task #1**) — the gradient-
  flow account: compression is benign (PowerSGD-only ties dense 0.741); residual is
  deterministic bias not noise (SNR ~42:1, 0.06% energy); the ~50% disagreement is
  structural; length-hack not low-entropy is the killer; the mask is the zero-mean knob but
  stalls; §8.1 parity-vs-surpass ceiling; §9 forward primitive (variance-controlled unbiased
  mask + EF, Gaussian probe). The load-bearing input to this strategy.
- `runs/EXP-25/DEEP_FINDINGS.md` — the signed_ema α-sweep dose-response, the √2 sign-
  disagreement signature, the monotonicity (signed_ema is net-harmful), the ranked
  improvement menu (error-feedback #24 = top successor).
- `runs/EXP-25/ENTROPY_COLLAPSE_FINDINGS.md` — α=0 root-cause: magnitude-preserving
  sign-SGD with persistent EMA signs → length-explosion reward-hack; per-hypothesis
  verdicts; the rollout-correction/IS analysis.
- `runs/EXP-25/verdict.md` — STOP (best-α 0.7066 ≤ falsification line 0.7114).
- `diagnostics/ENTROPY_COLLAPSE_WATCH.md` — the T1–T7 collapse triggers; length-
  explosion is the discriminator (the mandatory guardrail for every surpass-dense arm).
- Issue #24 — error-feedback on the PowerSGD residual + basis-aligned/staleness-aware
  blend; the measured `cos(G_powersgd, M_anchor) ≈ 0.001` orthogonality.
- `verl/workers/comm_eff/spectral_filter.py` — `signed_ema_matrix` (:268), `blend_matrix`
  (:238), `inject_matrix` (:207), `update_anchor` (:181).
- W&B (`shamanework-pl/verl_compression_research`): dense `5e2jpho9` (0.7536), A0
  PowerSGD r77+clean5 `oquyeic3` (0.7415), α=0.5 `1wulaelw` (0.7066), α=0.3 `r8kc702g`
  (0.6164), α=0.0 `uyrpaftw` (0.3541).

**External prior art.**
- Barrett & Dherin, *Implicit Gradient Regularization*, ICLR 2021 —
  https://openreview.net/forum?id=3q5IqUrkcF (GD biases toward flat minima; the
  zero-mean-noise pillar).
- *The alignment property of SGD noise and how it helps select flat minima: a stability
  analysis*, arXiv 2207.02628 — https://arxiv.org/pdf/2207.02628 (noise must be
  *aligned*/right-kind to select flat minima → the bias-vs-fluctuation distinction; why
  rank-as-temperature is an interior-optimum sweep, not "minimize rank").
- *Gradient Regularization Prevents Reward Hacking in RLHF and RLVR*, arXiv 2602.18037 —
  https://arxiv.org/pdf/2602.18037 (under-regularized RL gets reward-hacked; regularizer
  closes the hack channel → supports Lever 1 KL-brake as the enabler; mirrors our α=0
  length-explosion hack).
- *Sparse-SignSGD with Majority Vote for Communication-Efficient Distributed Learning*,
  arXiv 2302.07475 — https://arxiv.org/pdf/2302.07475 (sign compression needs majority-
  vote/variance-reduction to be unbiased → explains why signed_ema's single stale-EMA
  sign is biased and fails).
- Vogels et al., *PowerSGD* (the low-rank codec used here) and error-feedback / PULSELoCo
  (issue #24) — the canonical unbiasing mechanism for the magnitude path.
