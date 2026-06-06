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

**HONEST CURRENT VERDICT (stated up front, not softened).** *Every comm-eff arm that has
run sits AT or BELOW dense:* PowerSGD-only 0.741, A0 fresh-clean 0.7415, signed_ema-best
0.707 — all ≤ dense 0.7536. So **no >dense edge has been demonstrated.** The nuance
(mechanist round-2, §2.5): PowerSGD *does* carry a **REAL, uncompressed-rollout-corroborated**
higher policy diversity than dense in the trained regime (rollout perplexity 1.40 vs 1.24 at
s25, measured on the hook-free vLLM generator — not a measurement artifact) — but that
diversity **does NOT convert to reward** (score lags, val ties-not-beats). So the precise
verdict is: **"compression injects real, sustained, UNCONVERTED exploration; it reaches
dense-grade parity but no surpass edge has been shown."** The surpass thesis is therefore
**hypothetical** and rests on a *conversion* mechanism we have not yet tested — a CONTROLLED
(zero-mean, variance-tunable, temperature-dialed) perturbation that could harness the
diversity dense lacks. This document specifies the minimal experiment that could reveal (or
kill) it. Anything stronger than "here is the falsifiable test" would be overclaiming.

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

**MECHANIST [Q4] RESOLVED THIS — and it splits into two pieces, both pointing away from a
surpass edge.** (`COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` §6.1/§8 + the Q4 quantification.)

1. **The early 5→9→0.4 bouncing is a codec-WARMUP ARTIFACT, not exploration.** It coincides
   exactly with the cold-basis reconstruction error 0.97→0.14 as Q's power-iteration
   converges (steps 1–4); `actor/entropy` is computed on the actor-train forward, which the
   PowerSGD hook garbles while the basis is cold. Drop the warmup spike from any claim.
2. **The trained-regime fingerprint is REAL — uncompressed-rollout-corroborated — but it
   does not CONVERT (mechanist round-2 disambiguation).** Two independent measurements agree:
   (a) at matched warmed steps PowerSGD-r77 sustains ~**0.08–0.12 nat higher `actor/entropy`**
   than dense (s25: 0.335 vs 0.222; s45: 0.266 vs 0.146); (b) decisively, the **uncompressed
   vLLM generator's perplexity** (`rollout_corr/rollout_ppl`, a separate inference engine with
   NO compression hooks) is also higher for psgd (s25 1.401 vs dense 1.238; s45 1.283 vs
   1.150). Because the *uncompressed* sampling policy is genuinely more diffuse, **the higher
   entropy is a REAL policy property, not a training-forward measurement artifact.** This
   *rescues* a piece of the thesis: compression DOES inject real, sustained policy diversity.
   **BUT it does not convert to reward:** psgd's score lags dense at every step (s25 0.688 vs
   0.786) and val ties-not-beats (0.741 vs 0.754); the extra diversity is *lost, not
   harnessed.* And dense trains to 0.122 entropy — lower than any healthy comm-eff arm — with
   the best val (confident ≠ collapsed).
3. **IS-gap correction (do not misuse this metric).** `rollout_probs_diff_mean` is ~0.0035
   for BOTH dense AND psgd (no diversity difference *in this metric*), and ~0.62 for BOTH
   anchor/merger arms. The ~180× gap is **anchor-arms vs non-anchor-arms** (the anchor circuit
   changes the metric's normalization), **NOT dense-vs-compression** — my earlier draft mis-
   attributed it. So: dense-vs-psgd IS comparable on this metric (both 0.0035); merger-vs-dense
   is NOT. The clean exploration proxy is **`rollout_corr/rollout_ppl`** (comparable across all
   runs, uncompressed) — use it, not the IS-gap, going forward.

**Net (refined honest framing): the exploration fingerprint is REAL and uncompressed-
corroborated, but UNCONVERTED.** This is a *stronger* and more accurate seed than "probably an
artifact": compression already produces genuine, sustained policy diversity that dense does not
have — yet it currently fails to convert into reward (score lags, val ties). So the thesis is
NOT "compression explores better → beats dense" (the data refutes the second clause); it IS
"compression injects real unconverted diversity, and the open question is whether a *controlled*
mechanism can harness it into reward." Higher entropy is not the goal; *converting* it is.
That is exactly what EXP-SURPASS-1 tests: whether a *controlled* zero-mean perturbation (mask-p,
§2.6/§4) can convert the diversity into reward where the *uncontrolled* compression entropy does
not (the psgd entropy is an untunable byproduct of a biased codec; the mask-p knob is zero-mean,
variance-controlled, and temperature-tunable — the three properties the harness needs). The
surpass case rests on that mechanism, not on the bare entropy observation.

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

- **Channel C — Explicit zero-mean exploration noise via the prf_mask, with p as the
  TEMPERATURE dial. THE LEAD CHANNEL — and the in-stack, comm-efficient primitive already
  exists.** The **prf_mask + rescale=true** codec is inverted dropout (`h*mask/(1−p)`,
  `E[h̃]=h` exactly, `activation_mask.py:243,257`, re-sampled per global step) — so its
  residual is **genuinely ZERO-MEAN and step-decorrelated** by construction, the structural
  opposite of PowerSGD's fixed bias. And it is **tunable**: `p` is the drop probability, so
  the per-coordinate noise variance is exactly **p/(1−p)** — `p` IS the exploration
  temperature dial. So the mask, swept over p, satisfies all three required properties in
  ONE existing codec: **zero-mean + variance-tunable + comm-efficient** (it transmits only
  the kept `(1−p)·H` coords/token). My earlier claim that "no in-stack codec has all three"
  was wrong — it does; the dial is `p`.
- **The untested regime is the whole point.** EXP-16 only ever ran p=0.9/0.95 — the
  **high-variance** end (p/(1−p) = 9× / 19×) — where the mask stalls. The **low-to-mid p**
  regime (0.1 / 0.3 / 0.5 → variance 0.11× / 0.43× / 1.0×) has **never been tested for the
  beat-dense question.** That is exactly where a zero-mean perturbation could be mild enough
  to *train* yet nonzero enough to *explore*. The mask-p sweep is therefore the genuine,
  in-stack, falsifiable test of the operator's thesis (and it is the headline of §4).
- **The comm-savings ↔ exploration tension (state it honestly).** `p` couples two things:
  noise variance AND comm savings (kept = (1−p)·H). Low p = mild noise but modest savings
  (p=0.5 → 1.0× var, 50% comm cut); the big comm wins live at high p (p=0.95 → 5% kept) —
  exactly where it stalls. So the dream outcome is a **mid p** that beats dense *and* still
  cuts comm; a low-p-only win is a real thesis confirmation with modest savings; and if only
  the high-comm-savings p's stall, the comm-eff-AND-surpass goal is in tension and we fall
  back to parity (EF-PowerSGD). The sweep is designed to find where on the ladder (if
  anywhere) the perturbation helps before it stalls. **Optional variance-control arm**
  (error-feedback on the dropped mask residual, or `clean_cadence`) at the most-promising p
  to push the usable p higher — the comm-eff lever if a raw-mask sweet spot exists but is
  too low-p to save much.

#### Every candidate zero-mean noise knob in our stack — explicitly evaluated

The decisive question (is there ANY genuinely zero-mean, step-decorrelated noise source?)
deserves a complete enumeration, not just the winner. Evaluated with mechanist:

| candidate | zero-mean? | step-decorrelated? | variance | comm-eff? | verdict |
|---|---|---|---|---|---|
| **prf_mask + rescale, p as the dial** | **YES** — inverted dropout `E[h̃]=h`, re-sampled/step | YES | **TUNABLE via p** (p/(1−p): 0.11×@p0.1 → 19×@p0.95) | **YES** (transmits (1−p)·H) | **THE LEAD** — in-stack zero-mean+tunable+comm-eff knob; low-to-mid p UNTESTED → Lever 1 / EXP-SURPASS-1 |
| **PowerSGD rank** (lower r) | **NO** — fixed off-subspace bias (frozen Q) | NO (same dims every step) | low | yes | **WRONG knob** — lowering r adds *bias*, not noise; EF's job is to *remove* it (→ parity test, Lever 3) |
| **explicit Gaussian** `σ·N(0,1)` | **YES** by construction | YES | **fully tunable via σ** | NO (probe only) | **the codec-free decoupled CONTROL** — Lever 2, confirms the mask result isn't a codec artifact |
| **higher rollout temperature** | n/a (output-space, not gradient) | per-rollout | tunable | yes (rollouts already non-PP) | **distinct, cheap headroom control (T0)** — see below |
| **dropout-style stochasticity** | YES (if rescaled) | YES | tunable via p | partial | = the prf_mask case; subsumed by Lever 1 |
| **anchor-EMA** | NO (biased stale average) | NO (smoothed) | low | yes | not noise — variance-reduction, loses to Adam momentum (Channel A) |

**On rollout temperature (the team-lead's candidate worth its own line).** This is a
*different kind* of exploration than gradient-space noise: it widens the rollout
distribution directly (output-space), is essentially free (rollouts are already non-PP
vLLM, out of scope for compression), and is the *standard* RL exploration knob. It is worth
including as a **cheap orthogonal control arm**: dense + raised rollout temperature. If
raising rollout temperature alone lifts dense above 0.7536, then "RL is exploration-limited
here" is confirmed *independent of compression* — and the comm-eff question becomes "can a
zero-mean gradient perturbation match or beat that cheaper output-space exploration?" If
raising temperature does NOT help dense, that is itself strong evidence this surface is
*not* exploration-limited at the policy level, which would lower the prior on the whole
thesis. Either way it is one cheap dense arm that calibrates how much exploration headroom
exists at all — I am adding it to EXP-SURPASS-1 as control **T0**.

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

Ranked by `P(moves us above dense) × leverage / cost`, **fully converged with mechanist +
the team-lead's mask-p reframe**. Big changes from the first draft: the old top lever
(rank-as-exploration-noise via PowerSGD) was falsified (frozen-Q PowerSGD has no zero-mean
noise, §2.2); error-feedback and the anchor-EMA were demoted to *parity* tools (§2.6 +
Adam-momentum). The lead surpass lever is the **prf_mask with p as the exploration-
temperature dial** — the in-stack, comm-efficient, genuinely zero-mean knob whose low-to-mid
p regime is untested for beat-dense. Every lever is direction-preserving (the hard
constraint below); the KL brake + length cap are the enablers that make any perturbation
safe to push.

### Lever 1 — prf_mask with p-as-temperature (Channel C, THE LEAD — in-stack, comm-efficient, zero-mean)
- **Mechanism.** prf_mask + rescale=true is inverted dropout (`h*mask/(1−p)`, `E[h̃]=h`,
  re-sampled per global step, `activation_mask.py:243,257`) — genuinely zero-mean, step-
  decorrelated. `p` (the drop probability) is the **exploration temperature**: per-coordinate
  noise variance = `p/(1−p)`, and the codec transmits only the kept `(1−p)·H` coords/token,
  so the same dial sets BOTH exploration strength AND comm savings. Sweep `p` — no new codec,
  no new math, the primitive already exists and is comm-efficient.
- **Why it can beat dense.** This is the operator's literal "compression noise = exploration"
  thesis with the *correct* knob: a zero-mean perturbation supplies productive exploration
  (SGD-noise flat-minima §2.4 + RL exploration C2) without the bias that collapsed signed_ema.
  Dense sits at p=0 (zero injected noise); if the optimal noise temperature is > 0, some p
  beats dense.
- **The untested regime = the whole opportunity.** EXP-16 only tested p=0.9/0.95 (variance
  9×/19×) — the stall zone. **Low-to-mid p (0.1/0.3/0.5 → variance 0.11×/0.43×/1.0×) is
  UNTESTED for beat-dense.** That is where the noise may be mild enough to train yet nonzero
  enough to explore.
- **Falsifier.** If val is monotone-declining in p (saturating at dense as p→0), zero-mean
  exploration noise does not help GRPO on this surface — thesis falsified cleanly, in-stack,
  no new primitive needed.
- **Variance-control enhancement (only if a sweet spot exists but is too low-p to save much).**
  Add **error-feedback on the dropped mask residual `(I−mask)·h`** (re-inject next step) or a
  periodic `clean_cadence` — this lowers effective variance at a *given* p, pushing the usable
  (trainable) p HIGHER toward the big-comm-savings end. This is the comm-eff lever; deploy it
  at the most-promising p from the raw sweep.
- **Cost.** The raw p-sweep reuses the existing mask codec with ZERO new code (just config) —
  cheapest possible surpass test. The EF enhancement reuses the #24 machinery. ~3 p-arms + 2
  controls, then 1 EF arm if warranted.

### Lever 2 — Explicit zero-mean Gaussian-noise probe (the codec-free DECOUPLED CONTROL)
- **Mechanism (mechanist §9.2).** Add `σ·N(0,1)` to the boundary activation, `σ` the
  temperature, decorrelated per step, exactly zero-mean, variance fully controlled by σ.
  Run as a **mechanism probe, not a comm-eff arm** (it saves no bytes).
- **Role: the confound-killer for Lever 1.** If the mask-p sweep shows an edge, the Gaussian
  probe confirms it is *zero-mean-noise-as-exploration* and not a mask-specific codec
  artifact (PRF structure, the rescale gain, the kept-coord pattern). If the mask sweep
  nulls, the Gaussian probe is the cleaner second opinion that *no* zero-mean noise helps —
  the strongest possible falsification (codec-independent). It answers the irreducible
  question — *does ideal zero-mean tunable noise beat dense GRPO AT ALL?*  Run alongside or
  immediately after the mask sweep.
- **Why not first.** The mask-p sweep is *also* the comm-efficient deliverable and reuses an
  existing primitive at zero code cost, so it is the better first spend; the Gaussian probe is
  the science control that *interprets* the mask result (mask-edge real vs codec-artifact; or
  mask-null corroborated codec-independently). If even ideal zero-mean noise cannot beat
  dense, compression-as-exploration is falsified **independent of any codec.**
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
| 1 | **prf_mask p-as-temperature sweep** (+EF if needed) | Channel C, IN-STACK + comm-eff | zero-mean tunable noise; low-to-mid p untested | the genuine in-stack test | cheapest (config only) |
| 2 | **Gaussian-noise probe** | codec-free DECOUPLED CONTROL | does ideal noise beat dense at all? | interprets/corroborates the mask result | lowest code |
| 3 | Rank-as-trust-region sweep | Channel B regularization (parity test) | constrain the step | likely too weak (§2.2) | cheap |
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
high-noise arm from burning hours generating 16K-token garbage:
1. **KL brake (divergence axis).** `use_kl_loss=true`, coef 0.001 — bounds the policy's
   excursion from the reference (Lever 4). This is the *entropy/divergence* guardrail.
2. **Length brake (degeneration axis), UNCONDITIONAL.** Cap `max_response_length` to
   **1024–2048** (down from 16384) and/or add an explicit length penalty in the reward.
   GSM8K healthy responses are ~170–290 tokens (DEEP_FINDINGS §A2), so a 1024–2048 cap is
   ~4–8× headroom over the healthy regime — it never bites a non-degenerate arm but hard-
   stops the reward-hack channel and bounds per-arm cost. **This is baked in regardless
   of the KL diagnostic result**: KL is the divergence brake, the cap is the orthogonal
   degeneration brake. (Note: dropping the cap from 16384 also speeds every arm, so the
   whole sweep is cheaper than the EXP-25 runs.)
3. **Mandatory monitor on every arm:** the ENTROPY_COLLAPSE_WATCH T1–T7 triggers
   (length-explosion is the discriminator — kill any arm that fires composite-RED early,
   don't wait for step 50).

**Required instrumentation on EVERY surpass arm (mechanist §9, HARD gate).** Log per-step:
(a) the **dense-vs-perturbed update COSINE** (the never-logged EXP-20 success criterion) —
without it we cannot tell "noise explored productively" from "noise Adam just averaged
away"; (b) **sign-agreement(perturbed, dense) and the policy entropy + IS-gap** (to confirm
the perturbation moved the policy, not just the metric). These two numbers are the
difference between a real result and an uninterpretable one.

### The ONE experiment to run first: **the prf_mask p-as-temperature sweep**

The headline test is **in-stack and comm-efficient**: sweep the prf_mask drop probability
`p` as an exploration temperature. The mask with rescale=true is genuinely zero-mean
(inverted dropout, §2.6), `p` sets both the noise variance (p/(1−p)) and the comm savings
((1−p)·H kept), and the **low-to-mid p regime is untested for beat-dense** (EXP-16 only ran
the high-variance stall zone p=0.9/0.95). This is the operator's "compression noise =
exploration" thesis given a real, runnable, falsifiable test with an existing primitive at
near-zero code cost. The Gaussian probe (Lever 2) runs alongside as the codec-free control
that *interprets* the mask result.

> **EXP-SURPASS-1 — "Is there a mask temperature p that beats dense via zero-mean exploration?"**
>
> All arms on the fixed surface, codec=prf_mask + rescale=true, KL(0.001) + length-cap
> (1024–2048) brakes ON everywhere, 50 steps, instrumentation gates logged:
>
> | arm | config | noise var (p/(1−p)) | comm (kept) | role |
> |---|---|---|---|---|
> | **D0** dense + KL + length-cap | p=0 (no mask) | 0 | 100% | **the bar** (brake-matched) |
> | **M1** mask p=0.1 | 0.11× | 90% | mild noise, modest savings |
> | **M3** mask p=0.3 | 0.43× | 70% | the candidate low-stall sweet spot |
> | **M5** mask p=0.5 | 1.0× | 50% | mid noise + real comm savings (the dream cell) |
> | **G_mid** dense + `σ·N(0,1)`, σ matched to M3/M5 var | (matched) | 100% (probe) | **codec-free control** — mask-edge real or artifact? |
> | **T0** dense + raised rollout temperature | — | 100% (probe) | **headroom calibration** — does ANY exploration help dense? |
>
> Mask is re-sampled per global step (zero-mean), keyed on `global_step`. The M-arms are the
> comm-efficient surpass test; G_mid + T0 are science controls (save no comm). If an M-arm
> shows an edge but only at low p (modest savings), add the **EF variance-control arm**
> (error-feedback on the dropped mask residual) at the best p in a follow-up to push the
> usable p higher toward bigger savings.

**Why this one first.**
1. **It is the genuine, in-stack, comm-efficient test of the thesis.** Unlike the Gaussian
   probe (saves no comm) and the rank sweep (tests bias, not exploration), the mask-p sweep
   is *both* the exploration test *and* the comm-eff deliverable — a win here is the dream
   result (surpass AND save comm), and it reuses an existing primitive at config-only cost.
2. **The untested regime is exactly where the thesis could live.** EXP-16 stalled at high p
   (variance 9-19×); low-to-mid p (0.11-1.0× variance) has never been run for beat-dense.
   This is the cheapest way to find whether a trainable-yet-exploratory p exists.
3. **The Gaussian + T0 controls make it interpretable.** G_mid (σ matched to the mask
   variance) tells us a mask edge is *zero-mean-noise-as-exploration*, not a mask codec
   artifact. T0 tells us whether this surface is exploration-limited at all (if even rollout
   temperature can't lift dense, the prior on the whole thesis drops).

**Pre-registered predictions (honest — the prior is SKEPTICAL, per mechanist §8.2).**
- **Thesis confirmed:** an interior `p*` gives `val(p*) > val(D0)` by > noise (non-monotone
  temperature curve, peak above dense). Dream outcome: `p*` is mid (e.g. 0.5) → surpass +
  50% comm cut. Acceptable: `p*` low (e.g. 0.1) → thesis confirmed, modest savings (then EF
  to push p up).
- **Thesis falsified (mechanist's prior):** val is monotone *decreasing* in p — any mask
  noise only adds variance Adam averages away, dense (p=0) is best. Mechanist's standing
  position is no >dense edge demonstrated + entropy anti-correlated with val, so the honest
  base rate favors this. Stated up front to keep the result credible.
- **The COSINE gate is the tell:** a helping p with update-cosine-to-dense *dropping below
  ~0.98* while val rises = genuine productive exploration (noise steering, not jittering). val
  up with cosine ≈ 1.0 = noise Adam absorbed = likely seed noise → demand 2-seed confirmation.

**Decision rule.**
- **A mask arm beats dense:** 2-seed it, then sweep p finer around `p*` and add the EF
  variance-control arm to push the trainable p toward bigger comm savings. **This is the
  headline surpass + comm-savings result.** Cross-check with G_mid (real vs artifact).
- **No mask arm beats dense:** check G_mid (does *any* zero-mean noise help? if not, thesis
  falsified codec-independently) and T0 (is the surface exploration-limited at all?). Then
  pivot the deliverable to **parity + comm-savings** — run **EF-PowerSGD** (Lever 5) to bank
  the comm win at ~dense quality; optionally the cheap **rank sweep** (Lever 3) as a last
  check on the regularization route (prior: too weak).

### EXP-SURPASS-2 candidate — the CREDIT-ASSIGNMENT variant (follows from "real diversity that doesn't convert")

Mechanist's round-2 result (§2.5) — compression already produces *real, uncompressed-
corroborated* policy diversity (rollout_ppl 1.40 vs dense 1.24) that **fails to convert to
reward** — points to a second, mechanistically-distinct surpass path: the conversion failure
may be a **credit-assignment** problem, not a noise problem. If the diverse rollouts ARE being
sampled but GRPO's group-relative advantage with n=8 doesn't reward the rare *better* ones
strongly enough, the diversity is averaged away before it can steer the policy. **Cheap probe:
raise n (rollouts per prompt) ONLY on the compressed arm** (e.g. n=8→16) and ask whether the
real diversity then converts. Mechanism: more samples per group → the rare high-reward
completions the diffuse policy already produces get surfaced and get a stronger group-relative
advantage → diversity→reward. This is a surpass path that *needs compression's exploration*
(dense is less diffuse, gains less from extra n) **plus denser credit** — a genuine
compression-specific edge, not available to dense. Run it as EXP-SURPASS-2 if the mask-p sweep
shows the same leak-not-convert signature as psgd (real diversity, flat val): it tests whether
the bottleneck is *generating* exploration (mask-p's job) or *converting* it (n's job). Cost:
1–2 arms (compressed + raised n vs compressed baseline), with the dense+raised-n control to
confirm the gain is compression-specific. (Higher n raises rollout cost — keep the response
cap on.)

### Why this ordering (mask-p sweep → controls → parity) and not the old rank-first plan
The first draft's EXP-SURPASS-1 (rank × EF grid) assumed low-rank PowerSGD injects zero-mean
*exploration* noise. Mechanist falsified that: frozen-Q PowerSGD's residual is a deterministic
*bias*, not noise (§2.2), and EF only buys parity (§2.6). The genuinely zero-mean knob is the
*mask*, and `p` is its temperature dial — but only the high-variance stall zone was ever run.
So the headline test is the **mask-p sweep in its untested low-to-mid regime**: it is in-stack,
comm-efficient, reuses an existing primitive at config-only cost, and is *both* the exploration
test and the comm-eff deliverable. The Gaussian probe and rollout-temperature arm are the
science controls that interpret it; the rank sweep is demoted to a separate bias/parity check.

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
- **Joint converged verdict (+ team-lead's mask-p reframe):** no >dense edge has been
  demonstrated by anything that has run; the surviving testable surpass hypothesis is a
  **zero-mean, variance-tunable perturbation**, and the in-stack realization is the
  **prf_mask with `p` as the exploration-temperature dial** (low-to-mid p untested for
  beat-dense) — the EXP-SURPASS-1 headline, with the Gaussian-noise probe + rollout-temp arm
  as the interpreting controls. The anchor-EMA and error-feedback are parity/comm-savings
  tools, not surpass; the PowerSGD rank sweep tests *bias-tolerance/regularization*, not
  exploration. **Open item:** fold in the in-flight A0+KL(0.001) diagnostic when it lands
  (prior: KL arrests length, does not beat 0.741 = enabler not surplus).

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
variance-tunable** perturbation — and the in-stack realization already exists: the **prf_mask
with `p` as the exploration-temperature dial** (zero-mean inverted dropout, variance p/(1−p),
comm-efficient). The decisive, falsifiable test is the **mask-p sweep in its untested
low-to-mid regime** (EXP-16 only ran the high-p stall zone) — *is there a p that perturbs
productively before it stalls, beating dense?* — with the codec-free Gaussian-noise probe and
a rollout-temperature arm as the controls that interpret it, and the dense-vs-perturbed update
cosine logged to tell productive exploration from noise-Adam-averages-away. If a mask arm
wins, that is *both* surpass *and* comm-savings (the dream), pushed further with error-feedback
variance control; if none wins, the honest deliverable is **error-feedback PowerSGD** for
parity at materially lower comm. The honest prior (mechanist): no >dense edge has yet been
shown — this is a clean hypothesis with a one-experiment gate, not a claimed win.

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
