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

### 2.5 The candidate empirical fingerprint — dense is *confident*, comm-eff *explores* (CAVEATED, mechanist-gated)

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
that a dense single-step trainer does not?** Three candidates survive, and they are the
spine of the revised program. (Mechanist is pressure-testing each; tagged
**[MECHANIST-PENDING-R2]** where their read is still incoming.)

- **Channel A — Variance-reduced / look-ahead gradient (the EMA-as-momentum channel,
  mechanist-named in §8.1).** Dense uses a single-step gradient `G_t`. The anchor's
  β=0.95 EMA `M` is a *variance-reduced* estimate of the gradient averaged over ~20 ticks
  — a quantity the dense single-step trainer **does not have**. With n=8 GRPO advantages
  that are genuinely noisy (high-variance per-coordinate PG, §1.2 of the analysis: the
  per-coordinate gradient is near-zero-mean = high relative variance), a *direction-
  preserving* blend `G_used = G_noisy + λ·(M − proj_{G}(M))` could give a lower-variance,
  more reliable descent direction than dense's single noisy step. **This is the most
  defensible surpass mechanism** because it identifies a concrete channel dense lacks
  (the cross-step average) and it is direction-preserving (does NOT repeat the signed_ema
  error). The bar shifts: not "correct the compression bias" but "supply a variance-
  reduced gradient dense cannot compute in one step." [MECHANIST-PENDING-R2]

- **Channel B — Compression as an implicit trust-region / subspace regularizer.** Frozen-Q
  PowerSGD constrains every update to the dominant-energy subspace `P` (drops the low-
  energy tail deterministically). In a *non-stationary* RL objective, restricting updates
  to the stable high-energy subspace is an implicit regularizer — it prevents the step
  from chasing low-energy, batch-specific directions that overfit *this* rollout set,
  biasing toward flatter, better-generalizing policies (the IGR / flat-minima pillar,
  §2.4, achieved by *constraint* rather than by *noise*). This predicts comm-eff > dense
  via **regularization, not exploration** — and it is consistent with mechanist's
  stationary-recon-error result (§2.3: a stable, constant constraint). The risk
  (mechanist's §2.2 caveat): if the constraint is too *weak* (direction-faithful,
  recon_err only 2.5%), it may not regularize enough to matter. The rank sweep is then a
  test of *this* channel: lower rank = tighter trust region = stronger regularization —
  is there an `r*` whose regularization beats dense? [MECHANIST-PENDING-R2]

- **Channel C — Explicit zero-mean exploration via STOCHASTIC compression.** The §2.2
  "no zero-mean noise" result is specific to *frozen-Q* PowerSGD. A compression whose
  residual is zero-mean *by construction* — stochastic rounding/quantization, per-step
  re-sampled random-mask sparsification, or PowerSGD with a **re-randomized basis each
  step** — restores a genuine per-step fluctuation `ξ`, reviving the original SGD-noise /
  RL-exploration argument for real. This is the closest to the operator's literal "noise
  = exploration" thesis, but it is now correctly scoped to *stochastic* codecs, not the
  deterministic low-rank one we have. The risk: random projections may destroy the
  dominant-subspace signal (§2.2) faster than they add useful exploration; variance must
  be tuned. [MECHANIST-PENDING-R2]

**The reframed thesis.** Compression-as-exploration in its literal form requires Channel
C (stochastic codec). But the *stronger and more defensible* surpass argument is Channel
A (variance-reduced gradient dense cannot compute) and/or Channel B (implicit trust-region
regularization) — both of which beat dense not by adding noise but by supplying a
*better-conditioned update* than a single dense step. The operator's intuition — "RL
under-explores, so perturbing/regularizing the update can help where it would only hurt in
SFT" — is **correct in spirit**, and survives mechanist's falsification of the naive
PowerSGD-noise route, but its mechanism is variance-reduction + regularization (A+B), with
explicit exploration (C) as a separate testable variant.

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
- The α=0 + KL(0.001) diagnostic (in flight; team-lead relaying) is a **prior-tightener,
  not a blocker**: if KL bounds the α=0 collapse (entropy holds, length stays bounded),
  it directly proves the KL brake decouples noise-injection from collapse — strong
  support for running the low-rank arms aggressively. If KL does NOT bound it, the
  unconditional length cap (§4 brake 2) is the backstop and the low-rank arms still run,
  just with the cap doing more of the work. Either way EXP-SURPASS-1 is launchable; the
  diagnostic only sets how much we lean on KL vs the length cap.

**Decision rule.** Surplus CONFIRMED ⇒ 2-seed the winning cell, then sweep finer rank
around `r*` and write the result up as the headline. Surplus REJECTED but EF reaches
dense ⇒ pivot the headline to "comm-eff matches dense at materially lower comm" (the
GOAL.md parity+savings bar, with the exploration question answered negative). Either is
a publishable, decisive outcome.

### How EXP-SURPASS-1 adapts to mechanist's Q1 answer (decision-ready either way)

The pivotal Q1 (is PowerSGD's `(I−P)·G` residual zero-mean noise, or coherent bias?)
does not change *whether* we run the grid — it changes *which cell we expect to win*
and *how we read a null*. The design is robust to both branches:

- **If Q1 = zero-mean / step-decorrelated (the residual already looks like exploration
  noise).** Then rank IS a temperature knob directly. Expectation: the **EF-OFF
  low-rank** arm (`r=38, EF off, KL on`) may already beat dense, because EF would only
  *remove* the productive noise. The EF-ON row becomes the *control that should be
  WORSE or equal* (it sanitizes away the exploration). This would be the cleanest
  possible confirmation: "the dropped component IS the exploration; adding it back
  (EF) costs us the surplus." We'd then sweep rank finer on the EF-OFF row.

- **If Q1 = coherent low-rank bias (the residual is systematic, accumulates).** Then
  EF-OFF low-rank is *dangerous* (bias → drift → collapse, a milder cousin of α=0), and
  the surplus, if any, lives in the **EF-ON** row: EF removes the bias, leaving whatever
  zero-mean fluctuation remains as the exploration source. Expectation: EF-ON beats
  EF-OFF at every rank, and the surplus (if real) is `r=38/77, EF on, KL on`. If even
  EF-ON saturates at dense, the SFT intuition transferred and the answer is "match, not
  beat" (still a comm-savings win).

- **Either branch shares the same falsifier and the same controls**, so we commit to the
  full 2×3 grid + 2 controls now and let Q1 set the prior on which cell to 2-seed. The
  KL guardrail (Lever 1) is ON in every cell precisely so the EF-OFF low-rank arm is
  survivable enough to *measure* even in the bias branch — that is what de-risks running
  the grid before Q1 fully lands.

**Stage-gating to save compute.** Run in two waves on one box: **Wave A** =
`{dense+KL, r=77 EF-off (floor ref), r=38 EF-off, r=38 EF-on}` — 4 arms that already
bracket the most informative corners (current floor, the high-noise cell both with and
without de-biasing, and the bar). Read Wave A against the Q1 prior; only spend **Wave B**
(`r=154` both rows + `dense+entropy`) if Wave A shows a non-monotone signal worth
resolving. This halves the expected spend if the answer is an early, clean "no surplus."

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

---

## 7. References

**Internal (this fork).**
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
