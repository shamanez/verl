# Generalization & Future Directions — Communication-Efficient PP-GRPO

> **Scope.** How the two-circuit comm-efficient GRPO trainer (Qwen2.5-1.5B-Instruct /
> GSM8K) generalizes to larger tasks, models, and a real decentralized swarm; an
> honest bytes-vs-stability account of **both** mergers (error-feedback `delayed_ef`
> = "B2", and `signed_ema`); and the open **surpass-dense** question — what routes
> could genuinely beat dense, respecting the closed-frontier nulls.
>
> **Status of every claim below** is one of: **PROVEN** (in `research/runs/SUMMARY.md`),
> **CLOSED** (a falsified frontier lever, do not re-propose), **PROJECTED** (a scaling
> extrapolation, not yet measured), or **OPEN** (a candidate route). Peer verdicts from
> `theorist` (validity) and `systems` (feasibility) are folded in per-route and dated.

---

## 0. Ground truth (the numbers this report is anchored to)

| quantity | value | status |
|---|---|---|
| Codec | PowerSGD low-rank, `r=77`, activation basis | PROVEN |
| Fast-path gradient comm | bytes ratio ≈ **0.0505** (band 0.0504–0.0506, identical across all compressed arms; ~5% of dense) | PROVEN |
| Compression scope | **only the PP-boundary gradient** is projected; the **DP axis is NOT compressed** and **rollouts/generation (vLLM) are out of scope** | PROVEN |
| No-merger floor | val@50 **0.6300** | PROVEN |
| **B2 `delayed_ef`** (λ=1, β_anc=0) | val@50 **0.735–0.754** (first proof 0.7528) = **dense parity** | PROVEN SOTA |
| `signed_ema` (α=0.5) | val@50 **0.7271** (EXP-32, **valid-M**), dominated by B2, unstable | PROVEN (valid-M) |
| **Dense reference** | val@50 band **≈ 0.75–0.78**; apples-to-apples current-code draw (`73ntu76u`) = **0.7839** | PROVEN (band, variance-dominated) |
| Eval noise (single draw, seed 0) | ≈ **±0.024** | PROVEN |

**Canonical ordering:** `0.6300` (no-merger floor) `< signed_ema 0.7271 < B2 0.7528 ≈ dense 0.75–0.78`.
*(Note: the legacy `≈0.70`/`0.7066` signed_ema figure is the **invalid-M** EXP-25 measurement and is
**not** used here — per the SUMMARY Evidence Boundary, only post-#29 **valid-M** numbers count for
anchor-circuit claims. The valid-M signed_ema number is **0.7271**, EXP-32.)*

**The surpass bar is the dense *band*, and the apples-to-apples current-code draw is
0.7839 — above B2's parity band.** Goal 4 in `GOAL.md` (a surpass) is the only open
goal; Goals 1–3 (stable / parity / savings) are met. A credible surpass must clear
the band with margin, not land inside ±0.024 of a single dense draw.

---

## 1. The async target at scale

### 1.1 What the substrate actually simulates

The fixed `delay_K=5` lock-step substrate is a **clean simulation** of one pipeline
boundary. The deployment target is **super-asynchronous** and is the load-bearing
"practical future use" point:

- **Fast SWARM** — many heterogeneous worker instances, each running the normal actor
  train forward/backward, compressing the PP-boundary gradient with PowerSGD through a
  **shared, read-only** projection basis `Q`. Bytes per boundary exchange ≈ 5% of dense.
- **Slow ANCHOR** — a *single, fixed* compute node that maintains the full-coverage,
  DP-reduced gradient EMA `M` over all 196 decoder matrices, and is the **only** updater
  of `Q`. It runs an uncompressed clone pass and broadcasts `Q` (and `M`, for the merger)
  to the swarm.

The anchor serves the swarm **over a network**. Weight/gradient shipping takes variable
time, so **the anchor is always behind the swarm, never ahead** — by construction. This
is not a limitation to engineer away; it is the defining property of the regime, and it
fixes which methods are admissible.

### 1.2 Why "always lagging" is the right abstraction (and what it rules out)

Let the swarm advance the policy at rate `Δθ` per optimizer tick and the anchor refresh
incur a round-trip latency of `L` ticks (ship weights out → backward → ship `M`/`Q` back).
The anchor's reference is then stale by `K_eff ≈ L` ticks, and `L` grows with:

- **Swarm size** — more workers feeding one anchor ⇒ longer aggregation/queueing before a
  refresh; the anchor's single-node throughput is fixed while the swarm's collective
  step-rate scales with worker count. So `K_eff` grows *roughly linearly* with the ratio
  (swarm step-rate)/(anchor throughput).
- **Network diameter** — geo-distributed links add fixed propagation + jitter to every
  ship; `K_eff` inherits a variable, occasionally-large tail.
- **Stragglers / drops** — fires arrive late, jittered, or dropped; the realized staleness
  distribution has a heavy right tail, not a fixed 5.

**Consequence (admissibility filter — operator-locked, see `async-anchor-single-node-fast-swarm`):**
- The anchor can **never lead / predict** future swarm weights ⇒ **delay-compensation,
  Nesterov-lead, staleness-extrapolation are RULED OUT.**
- Admissible levers use the anchor only as a **lagging, trusted reference**, must **degrade
  gracefully as staleness grows** (never blow up as the anchor ages), and any derived state
  must be **cross-rank/cross-instance identical** (a per-instance buffer diverges the swarm).
- **Practical-future-use is a first-class selection criterion**: a lever that only works at a
  tight fixed `K` has no future; one that survives or exploits variable lag does.

This is exactly why B2 (`delayed_ef`) is the right *primitive*: it is **additive and
magnitude-preserving** (`G_corr = G_comp + λ·δ`, `δ = M_rep − G_comp_ring`). The correction
`δ` is the codec residual reconstructed from the stale anchor; as the anchor ages, `δ`
shrinks toward the pure compressed step rather than diverging. The two-circuit structure is
therefore **mandatory and not collapsible** — it is the only structure that lets a single
slow node keep a fast swarm's compressed gradients dense-comparable without ever leading them.

### 1.3 Why two circuits beat the single-circuit alternatives

| alternative | failure mode |
|---|---|
| Fast circuit only (PowerSGD, no anchor) | biased low-rank step; **no-merger floor 0.6300** — far below parity |
| Periodic full-rank "clean step" | full-H transfer (not comm-efficient) **and** would itself be stale on a real link; amortized comm ≈ 4× not 20× (`clean-step-realism-confound`) |
| Anchor leads / extrapolates | RULED OUT — the anchor physically cannot see the swarm's future |

The two-circuit split is the unique decomposition that (a) keeps the *hot* path at ~5%
comm, (b) confines the *expensive* full-coverage work to one node, and (c) tolerates the
staleness that a real network forces. **This is the generalizable contribution**, independent
of model/dataset.

---

## 2. Scaling axes — does parity hold, what breaks

PROJECTED unless marked. Each axis is rated parity-risk **LOW / MED / HIGH**.

### 2.1 Bigger models (more & larger boundary matrices) — parity risk **LOW–MED**

- **Boundary count.** A deeper model has more PP-stage boundaries and more decoder matrices
  (here: 28 layers × 7 = 196). The anchor's `M` and `Q` work **per-matrix**, so the method
  scales structurally — but the anchor's per-refresh compute and the `Q`-broadcast volume
  grow linearly with matrix count. The anchor node becomes the throughput bottleneck first.
- **Rank scaling.** PowerSGD `r=77` was byte-matched to mask `p=0.95` at `H=1536`
  (`0.05·1536 ≈ 77`). For larger `H`, holding the **bytes ratio** fixed means `r ∝ H` (rank
  scales with hidden size). But boundary-gradient spectra get *heavier-tailed* with scale, so
  a fixed *fraction* of rank may capture **less** of the gradient energy ⇒ the codec residual
  `δ` carries more, and the anchor's job (reconstructing that residual on stale data) gets
  harder. **Open question:** does the ~5% ratio hold, or must `r/H` rise with model size to
  preserve parity? This is the single most important scaling unknown.
- **What breaks first:** the **anchor-node throughput** (one node, full-coverage `M`), then
  **codec fidelity** if `r/H` is held fixed against a heavier-tailed spectrum.

### 2.2 Longer context — parity risk **LOW**

The codec compresses **gradients at the boundary**, whose dimensionality is set by
`hidden_size × (matrix shape)`, **not** sequence length. Longer context inflates activation
*memory* and the anchor clone's footprint (already the binding constraint — actor token
budget halved to 18432 to fit the anchor clone), but does **not** change the compressed
object's rank structure. Parity should hold; the risk is **memory/OOM on the anchor clone**,
an engineering limit, not a method limit.

### 2.3 Harder datasets (beyond GSM8K) — parity risk **MED–HIGH**

GSM8K is short-horizon, verifiable-reward math. Harder regimes stress the method where it is
*least* tested:

- **Longer reasoning chains / sparser reward** ⇒ higher-variance advantages ⇒ noisier
  per-tick gradients ⇒ the codec residual `δ` is noisier and the stale anchor reconstructs a
  noisier target. Parity may degrade unless the anchor's staleness `K` is **tightened** (which
  the network may not allow) — a genuine scaling tension.
- **Multi-domain / curriculum data** ⇒ the gradient direction is *non-stationary*; a `K`-stale
  anchor reconstructs an out-of-date direction. This is where **variable-staleness tolerance**
  (§1.2) stops being a nicety and becomes the parity-determining property.
- **Note on the dense bar:** on harder data the dense ceiling itself moves; "parity" must be
  re-measured against a matched dense control per dataset, never imported from GSM8K.

### 2.4 More swarm workers — parity risk **MED**

More workers ⇒ (a) larger effective batch / more rollouts per tick (helps), but (b) longer
anchor staleness `K_eff` (§1.2, hurts) and (c) more `Q`-consumers to keep cross-rank identical
(a correctness, not accuracy, constraint). The **net** effect on parity is the central
empirical question for the decentralized regime: does the extra data diversity from more
workers offset the extra staleness? This is also where the **surpass** routes (§4) live —
because more workers is the one axis that injects *new data-distribution information*, not
just a better reconstruction of the same gradient.

### 2.5 Scaling summary

| axis | parity risk | what breaks first |
|---|---|---|
| Bigger models | LOW–MED | anchor-node throughput; then codec `r/H` vs heavy-tailed spectrum |
| Longer context | LOW | anchor-clone memory (engineering) |
| Harder datasets | MED–HIGH | stale anchor reconstructs non-stationary direction |
| More swarm workers | MED | staleness `K_eff` grows; offset by data diversity (= the surpass hope) |

---

## 3. Both mergers at scale — the bytes-vs-stability trade-off, honestly

The merger is *how* the anchor `M` corrects the fast gradient. Two families matter.

### 3.1 Error-feedback `delayed_ef` (B2) — the reference

```
δ(t)      = M_rep − G_comp_ring(t − K)
G_corr(t) = G_comp(t) + λ·δ(t),   λ = 1,  β_anc = 0
```

- **Additive, magnitude-preserving, robust.** Reaches **dense parity** (0.735–0.754) at ~5%
  comm. As the anchor ages, `δ` decays toward the pure compressed step — graceful degradation,
  exactly the §1.2 admissibility property.
- **Comm cost of the merger itself:** the anchor must ship the **full-precision `M`** (or its
  delta) to the swarm. This is a *low-frequency* transfer (every `K` ticks) but **full-width** —
  so the merger's amortized comm is **non-trivial and grows with model size**. At scale this is
  the cost that makes one consider the sign-based family.
- **Verdict at scale:** **preferred whenever the anchor↔swarm link can afford periodic
  full-width `M` transfer.** It is the safe default — stable, parity-proven, and its quality is
  insensitive to `β_anc` across [0, 0.5] (EXP-33: flat free-averaging region).

### 3.2 `signed_ema` — the cheap-comm, unstable candidate

```
G_corr ≈ |G|·sign(M)
```

- **Cheaper comm — PROJECTED 1-bit sign traffic.** *Honest framing:* the current code computes
  `G_corr = α·G_noisy + (1−α)·|G_noisy|·sign(M)` **locally** (`spectral_filter.py:430`) — magnitude
  from the fast compressed grad, **sign from the local anchor EMA `M`**. The 1-bit saving is a
  **projected deployment capability**: in a real swarm the anchor could broadcast only `sign(M)`
  (1 bit/coordinate) instead of the full-precision `M` that B2 needs (§3.1). It is **not** a measured
  saving in the lock-step sim — the sim transmits nothing. This *projected* comm advantage is the
  *only* reason to keep `signed_ema` alive.
- **But unstable.** α→0 is a sign-SGD sharpening spiral; entropy collapse / length-explosion. The
  valid-M α=0.5 measurement (EXP-32) reaches **0.7271** — above the 0.6300 floor but **dominated by
  B2 (0.7528)** — and the legacy α=0.5 50-step survival was **censored** (already spiraling at steps
  47–48 in EXP-25/27). The instability spans both the cold-`M` (β=1) and the sign-reversal regimes —
  it is a property of putting an `M`-carrier merger in the loop, not a tuning artifact
  (`entropy-collapse-alpha0-signed-ema`, `exp25-collapse-gradient-flow`).
- **Verdict at scale:** **only preferable when the comm budget tightens to the point that
  full-width `M` transfer is infeasible** (very large models + very thin anchor↔swarm links) AND
  a stability fix is found. As of now it is **not promoted**; α=0.5 is the only setting worth
  tracking in future comparisons.

### 3.3 The honest trade-off

| | `delayed_ef` (B2) | `signed_ema` (α=0.5) |
|---|---|---|
| Merger comm | full-precision `M`, low-frequency (PROVEN) | **1-bit sign(M)** — far cheaper (PROJECTED; computed locally today) |
| Quality | **dense parity (0.735–0.754)** | **0.7271** (valid-M), dominated |
| Stability | robust, graceful under staleness | **unstable** (sharpening spiral, censored survival) |
| Use when | link can afford periodic `M` | comm budget is the hard wall **and** stability is fixed |

**Bottom line:** B2 is the reference at every scale we can currently justify. `signed_ema` is a
**bytes-floor escape hatch** — its 1-bit traffic is the right answer only when full-width `M` is
the binding constraint, and only if its instability is solved. The roadmap (§5) carries a
*conditional* signed-EMA stabilization item gated on that budget pressure actually appearing.

---

## 4. The surpass question — genuinely-open routes

### 4.1 The ceiling that closes the obvious routes — and the formal escape test

`theorist`'s load-bearing result (empirically confirmed by EXP-31/33):

> **You cannot beat dense by reweighting, accumulating, perturbing, or de-noising a *stale
> estimate of dense*.** The correction `δ = M_rep − G_comp_ring` reconstructs the dense
> gradient on stale data; every admissible anchor-usage lever is some operation on that stale
> dense estimate ⇒ it asymptotes to dense, never exceeds it.

**Formal test (theorist, theory.md §5.3).** Let **σ(M) = σ(g(θ_t), g(θ_{t−K}))** be the sigma-algebra
of the current + stale dense gradient *means*. Any deterministic `Φ(G_comp, M)` is σ(M)-measurable ⇒
**capped at dense**. A route **ACCEPTS** (can surpass) **iff it injects information OUTSIDE σ(M)**.
There are exactly **three** admissible escape categories:

1. **Curvature / second-order** — uses a quantity *beyond a gradient mean* (Hessian-vector,
   preconditioner from swarm-gradient *spread*, Fisher / natural-gradient). *Test: needs more than a
   first moment.* A merely reweighted/accumulated gradient (EXP-31 L2/L3) is **inside** σ(M) ⇒ REJECT.
2. **Conversion-positive exploration** — must move the **greedy argmax** (training-time mode
   relocation), not merely widen eval-time entropy. *Test: changes argmax π, with a mandatory
   dense × {T,n} control.* Raising n + rollout temperature is the only compression-*specific* knob;
   likely a pass@k edge, **not** a greedy surpass.
3. **Cross-rank second moment** — disagreement-as-signal (descend where ranks agree, damp where they
   fight = SAM-style robust direction). Swarm variance is information the *mean* `M` discards. *Test:
   the signal lives in the cross-rank 2nd moment, stays cross-rank-identical after aggregation, and
   tolerates variable staleness.* Isotropic perturbation (EXP-31 L4) is uncorrelated noise ⇒ REJECT.

**A surpass route must land in one of these three categories** (or, separately, claim a *test-time
generalization* edge the fixed-point ceiling does not adjudicate — see R4). This is the filter every
candidate below is tagged against.

### 4.2 CLOSED frontier — do NOT re-propose (all null on the valid-M circuit)

- **EXP-31 anchor-usage tournament:** perturbation (σ=0.01), δ-momentum (μ=0.5 parity / μ=0.9
  regress), adaptive dose (ratio/cosine, parity), control-variate (covariance gate failed,
  cov≈0), sub-basis rank-2 tail (early boost, no surpass).
- **EXP-33 β_anc sweep** {0, .25, .5, .75, 1}: 0.738 / 0.740 / 0.753 / 0.722 / degenerate.
  β∈[0,0.5] flat free-averaging; β=0.5 nominal best but inside ±0.024 noise; β=1 cold-`M`
  collapse. β=0 stays default.
- Also dead (with the *why*, confirmed by `systems`): **constant λ>1** (dose escalation); **naive
  `M`-EMA** (β>0 just averages staleness); **sign-replacement** — closed *structurally*, not just
  dominated: **50.4% sign-disagreement at the first warm step**, flat, **not fixable by α / delay_K /
  β**; **Step-C / forward-Q** — *anti-converts* because vLLM **rollouts run uncompressed**, so `Q` must
  preserve train↔rollout consistency; **generation-side compression** — out of scope by design
  (rollouts uncompressed); **anchor-lead / delay-compensation** — violates async-realism (anchor always
  lags); **delta-subbasis** (frozen null); **`ef_powersgd` damped (EXP-27)** — STOP, ignites
  length-explosion at step ~61 with no val gain over its 0.7210 parent.

These are all operations on the stale dense estimate (σ(M)-measurable). The ceiling explains *why* they
are null.

### 4.3 The candidate routes (each must escape §4.1)

Each route is annotated with **[theorist: validity]** and **[systems: feasibility]** verdicts.
*(Status: both cross-exams are complete. **theorist** has ruled all five routes against the σ(M)
ceiling (R3 ACCEPT, R5 admissible-heavy, R1 partial-escape/pass@k-likely, R4 escapes-by-scope-change,
R2 REJECT); **systems** has ruled feasibility (R1 runnable; R3/R4/R2 needs-build; R5 substrate
corroborated). What remains per route is **empirical** — the experiment that would confirm the
conditional escape — not an open verdict.)*

---

#### R1 — Swarm rollout diversity (raise `n` + rollout temperature) — **most-vetted lead; rank in §4.4**

**Mechanism.** Raise rollouts-per-prompt `n` and/or rollout sampling temperature `T` **on the
compressed fast circuit**, with a **mandatory matched dense × {T, n} control**.

**Why it escapes the ORIGINAL ceiling — but lands on a NEW one (theorist's precise ruling).** `n`
and `T` change *which* gradient is computed: they move `g(θ_t)` **itself**, so they are **outside
σ(M)** for the baseline-(T,n) σ-algebra. R1 is therefore **not** a `Φ(G_comp, M)` reconstruction
lever and the hard-reject list does **not** bind it. **HOWEVER**, the mandatory dense × {T,n} control
**defines a new ceiling = dense-at-matched-(T,n)**, and R1 beats *that* **only if compression and
high-T/n INTERACT** — the compressed circuit must **convert** the extra exploration into a better
trajectory than dense does *at the same (T, n)*. That **compression × exploration interaction is the
one genuinely live, untested, compression-specific bet**; **without** it, compressed-at-high-T simply
re-lands at dense-at-high-T (ceiling holds at the new operating point). The hope rests on the diffuse
policy a compressed run produces (measured: rollout_ppl 1.40 vs dense 1.24) having **steeper
d(val)/dn** — but note this same diffuseness **already failed once** to convert on the greedy bar (the
PowerSGD null), so the interaction is a hope, not a mechanism.

**The honest caveats (from `surpass-dense-conversion-spine`).**
- Compression is **TRAIN-ONLY** (`state.py` TRAIN_TAG); rollouts are vLLM with no hooks. So R1
  does **not** make the *generation* more diverse via compression — it raises diversity with the
  *sampling knobs* (n, T) and asks whether the **trained policy** (which the compressed gradient
  shaped) converts that diversity better.
- **Val is greedy mean@1** — it reads only the **mode**. Diversity is invisible to the bar unless
  it **relocates the trained argmax** (ROUTE B). The decisive discriminator is the **pass@k
  coverage curve**: if (compressed − dense) pass@k advantage **grows with k**, the edge is
  compression-specific; flat-in-k ⇒ generic and dense catches up (ROUTE A, secondary).
- **`rollout.n=8` is LOCKED** in the fixed control surface ⇒ raising it is an **off-axis change
  requiring separate justification**, but it is explicitly sanctioned as a **new lineage with a
  matched dense control** (the conversion spine names this the lead bet). The dense × {T, n} GATE
  *was never actually run* — every executed experiment went down the gradient/merger axis — so
  R1's foundational gate is **genuinely open**.

**Promise × feasibility:** highest of the candidates, but honest prior **< 20%**; most likely
outcome is "Route-A-only" (a real pass@k edge, greedy ties). This is the prior team's vetted lead
and the only compression-*specific* bet.

> **[theorist: validity — PARTIAL ESCAPE → NEW ceiling (theorist, DIRECT). n/T move `g(θ_t)` itself ⇒
> outside σ(M) (not a `Φ(G_comp,M)` lever, hard-reject list doesn't bind). BUT the mandatory dense×{T,n}
> control defines a NEW ceiling = dense-at-matched-(T,n); R1 beats it ONLY if compression×exploration
> INTERACT (the one genuinely-live compression-specific bet). Likely pass@k/Route-A only; greedy-mode
> surpass = long shot (the codec's diffuse policy already failed to convert on greedy once).]**
> **[systems: feasibility — RUNNABLE at n≤8 + T-sweep; n>8 = verify box headroom first (systems,
> confirmed). `rollout_n` (default 8) + rollout temperature are exposed launcher knobs; the dense ×
> {T,n} control is mandated and was never run. **n>8 caveat:** static batching keeps per-micro-batch
> memory flat, but generation memory + rollout buffer + log_prob/ref passes scale with batch×n, so n=16
> risks vLLM KV-cache/generation OOM on the same 4×H200/8×H100 box (sized for n=8 + 16K response) and
> lengthens each step ⇒ headroom-check or a bigger box. Two code-verified gates: (1) codec is TRAIN-ONLY
> ⇒ a *generation* bet on n/T, not the codec; (2) val is greedy mean@1 ⇒ diffuse policy invisible unless
> it relocates the mode. Decisive discriminator = pass@k coverage curve. Off the locked "generation"
> axis ⇒ new lineage.]**

---

#### R4 — Compression as a flat-minima regularizer (test-time generalization edge) — **rank in §4.4**

**Mechanism.** Hypothesis that the low-rank projection is an **implicit regularizer** (it
systematically removes off-subspace gradient components every fast step), biasing the solution
toward flatter minima that **generalize better** than dense on a held-out / harder split — even if
**train reward only reaches parity**.

**Why it escapes the ceiling — by REDEFINING the goal (theorist-confirmed).** It does **not** try to
beat dense on the *training objective* at all — it **concedes train-objective parity** and moves the
goalposts to a **different metric** (test / OOD generalization risk). The σ(M) ceiling is a statement
about the *train* gradient / train-objective optimum; it says **nothing** about test-distribution risk,
so R4 escapes it. **But this is admissible only as a SCOPE CHANGE: it does NOT meet GOAL.md's GSM8K
greedy-mean@1 bar — it redefines it.**

**Caveats / risk (sharpened by theorist).** EXP-31's L4 perturbation (isotropic ξ, σ=0.01) was a
**regularization control** and was **null** — but it was *isotropic* and touched the anchor nowhere.
**Compression bias `(I−P)g` is a COHERENT, fixed-direction perturbation each step** — formally **more
like a fixed preconditioner than stochastic SGD noise**. So the flat-minima story is **plausible but
NOT guaranteed**: a *coherent* bias is **as likely to steer toward a WORSE basin as a flatter one —
50/50 at best** (biased ≠ flatness-seeking). **Two gates, not one:** (1) a real **OOD eval** (no split
wired today — GSM8K-only); **and** (2) a **sharpness / Hessian-trace measurement** of the B2 vs dense
optimum (the flat-minima mechanism needs its *own* evidence, not just an OOD number). Cheapest form of
(1): eval-only over **existing** B2-vs-dense checkpoints, no retrain.

> **[theorist: validity — ESCAPES by SCOPE-CHANGE (theorist-confirmed): targets a metric the σ(M)
> ceiling does not bound (test/OOD risk), so it escapes — but it REDEFINES the goal and does NOT meet
> GOAL.md's GSM8K greedy-mean bar. Coherent fixed-direction bias ⇒ flat-minima plausible-not-guaranteed,
> ~50/50 it's a worse basin. Gate on OOD eval AND a sharpness/Hessian-trace measurement.]**
> **[systems: feasibility — NOT MEASURABLE AS-IS; needs an OOD split wired (systems, confirmed). Data is
> GSM8K-only; no held-out/OOD/MATH split exists (zero hits for ood/MATH-lighteval/SVAMP/ASDiv in launchers
> + FIXED_CONTROL_SURFACE). `val_files` is a config path, so an alternate test parquet is read-only
> (measurement-class like `test_freq`), but it must be *prepared* (a second eval dataset, e.g.
> SVAMP/ASDiv/MATH, + matching reward fn) and it changes the locked measurement surface. **Cheapest
> version (systems): eval-only on EXISTING B2 vs dense checkpoints on the new split — no retrain.** Genuinely
> novel angle: a test-time edge orthogonal to the train-reward bar everything else targets.]**

---

#### R3 — Cross-rank second moment / disagreement-as-signal (SAM-style) — **theorist category 3; rank in §4.4**

> **Reframed (theorist scaffold).** My first framing — a *heterogeneous-staleness ensemble* that
> averages staleness-varied gradients — is **REJECT**: averaging over staleness is still averaging
> *stale dense means* ⇒ a noisier σ(M)-measurable estimate, capped at dense. The **category-3 escape**
> is different and genuine: use the **cross-rank second moment** (swarm gradient *disagreement*), which
> the mean `M` discards, as a signal.

**Mechanism.** The fast swarm produces many per-rank gradients. Their **mean** is what `M`/the codec
already use. Their **cross-rank variance / disagreement** is information *outside* σ(M): descend
confidently where ranks **agree**, damp the step where they **fight** (a SAM-style, variance-aware
robust direction). The DP axis is **not compressed**, so per-rank gradients are available pre-projection
to form this second moment.

**Why it escapes the ceiling (theorist's load-bearing argument).** σ(M) contains only the gradient
*means* `g(θ_t), g(θ_{t−K})`. A variance-adaptive / SAM-style direction **provably optimizes a
DIFFERENT objective than `E[g]`** (it trades the mean-descent for a robustness-penalized objective) ⇒
it is **not** σ(M)-measurable and injects genuinely-new information — exactly theorist's category 3 —
**and it lives natively on THIS substrate** (the DP swarm: many independent per-rank gradients per step,
on the *uncompressed* axis, over different data shards; no new architecture). This is **not** the
isotropic perturbation EXP-31 L4 ran (uncorrelated noise, REJECT) — the signal is the *structured*
cross-rank disagreement, which both the dense mean **and** the anchor `M` (itself a DP-mean) discard.

**Async-realism caveat.** The derived robust direction **must stay cross-rank-identical after
aggregation** (a per-instance buffer diverges the swarm — §1.2) and **tolerate variable staleness**.
Whether a cross-rank-2nd-moment estimate remains well-defined when ranks report at *different*
staleness is the open async question for theorist.

> **[theorist: validity — ACCEPT (theorist, DIRECT): explicitly named as "the mathematically-most-
> promising route I'd ADD" — the one signal genuinely outside σ(M) that lives on THIS substrate (the DP
> swarm); variance is not a function of the gradient means (his §5.4 cat-3). (The staleness-ensemble
> framing stays REJECT.) The remaining gate is the async constraint: the estimate must stay
> cross-rank-identical after aggregation + tolerate variable staleness.]**
> **[systems: feasibility — NEEDS BUILD (systems-confirmed): `delay_K` is a fixed scalar
> (`comm_eff.py:136`) and `AnchorStalenessQueue` serves a deterministic `t−K` (`anchor.py:243-258`); the
> circuit is cross-rank-identical by mandate. No cross-rank second-moment state exists; forming a
> per-coordinate cross-rank variance needs a new all-reduce of the second moment (the DP axis is
> uncompressed, so per-rank gradients are reachable) + a variance-aware writeback. Heavier than a merger
> knob.]**

---

#### R2 — Anchor as advantage-baseline (objective-side, not gradient-side) — **likely ceiling-bound; rank in §4.4**

**Mechanism.** Use the stale `M` to form a **variance-reduction baseline on the GRPO advantage**
(the objective side), explicitly *different* from EXP-31's control-variate which gated the
*gradient*.

**Why it probably doesn't escape — now with a concrete type-mismatch (systems).** In GRPO the advantage
is **already a group-relative (leave-one-out-style) baseline over the n=8 rollouts** — the within-group
mean *is* the variance-reduction baseline. So the question is what a stale `M` adds *on top*. **And `M`
is a gradient-space object** (a per-matrix EMA over the 196 decoder weight matrices), **not a return /
advantage estimate** — it does not naturally produce a per-sequence scalar baseline. There is a
**type-mismatch** that any wiring must bridge, and once bridged the most likely result is **re-deriving
the existing group baseline** (σ(M)-measurable ⇒ ceiling). Kept only to ask theorist whether *any*
objective-side use escapes; absent a positive answer it is **retired**.

> **[theorist: validity — REJECT (theorist, DIRECT): a baseline from `M` is deterministic-in-(G_comp,M)
> ⇒ σ(M)-measurable ⇒ can't move the fixed point, capped at dense; + EXP-31 L1 cov(G_comp,M)≈0 (nothing
> to cancel ⇒ adding an M-baseline RAISES variance). No objective-side escape identified.]**
> **[systems: feasibility — WIREABLE but mechanism UNDERSPECIFIED (systems, confirmed). Genuinely not yet
> tested and distinct from EXP-31's L1 gradient control-variate (cov≈0, skipped). BUT: the GRPO advantage
> is already a group-LOO baseline, and `M` is gradient-space (per-matrix EMA, not a per-sequence scalar)
> ⇒ a type-mismatch to bridge + the async cross-rank-identical/staleness constraint. Lower-confidence-
> feasible than R1; theorist must rule it is not just the existing group baseline.]**

---

#### R5 — Anchor-curvature preconditioner (second-order) — **candidate, theorist category (a)**

**Mechanism.** The anchor already maintains a full-coverage, DP-reduced gradient EMA `M`. Have it
*also* maintain a cheap **curvature estimate** — a diagonal Hessian / second-moment, or a
finite-difference `(M_t − M_{t−1})` curvature proxy along the update direction — and broadcast it
(cross-rank identical, like `Q`) as a **preconditioner** the swarm applies to its compressed step.

**Why it targets the one escape my other routes miss.** This is theorist's category **(a):
curvature / second-order structure dense-SGD-at-this-LR does not use.** It is the only category that
injects information dense *structurally* lacks, rather than reconstructing/reweighting a first-order
quantity ⇒ in principle the strongest surpass route.

**The sharp validity caveat I must flag (code-verified).** The dense reference is **AdamW** (verl
default betas; `_generated_ppo_trainer.yaml`), which **already applies a diagonal second-moment
preconditioner** (`v_t`). So R5 does **not** escape by "adding curvature dense lacks" — dense already
has *diagonal* curvature. R5's escape is **narrower and must be one of:**
1. **Off-diagonal / block curvature** Adam's diagonal `v_t` cannot represent; **or**
2. **Less-noisy diagonal curvature** from the anchor's *full-coverage, DP-reduced* gradient vs Adam's
   noisy per-microbatch running `v_t`; **or**
3. A **cross-rank-shared** curvature the swarm's *per-shard* Adam states structurally cannot form
   (each worker sees only its shard; the anchor sees the whole).
If none of these holds, R5 **collapses one level up** — it becomes "a noisier estimate of what Adam
already computes," i.e. the ceiling again. **This is the precise question for theorist.**

**Async-admissibility (favorable).** Curvature changes *slowly* relative to gradients, so a *stale*
curvature estimate hurts far less than a stale gradient — staleness-tolerance is a **strength** here,
not a liability. A bounded diagonal preconditioner also **degrades gracefully** as it ages (it is a
scaling, not a direction), and it is naturally cross-rank-identical (anchor-owned, broadcast). So R5
is the rare route whose math *likes* the async-realism constraints.

> **[theorist: validity — ADMISSIBLE-but-"heavy" (theorist, DIRECT; category 1). ESCAPES *only* if
> anchor-curvature beats Adam's own `v_t` via off-diagonal / lower-noise / cross-rank-shared structure
> (else collapses to a noisier Adam — the AdamW-`v_t` overlap is the binding constraint, not compute).
> The remaining gate is empirical: does it beat `v_t`?]**
> **[systems: feasibility — NEEDS-BUILD (MODERATE, not greenfield) — systems-confirmed. The anchor
> maintains only `M` (grad EMA, `ema_device=cpu`) + `Q` — no preconditioner/curvature/Fisher state. BUT
> a **partial primitive already exists**: `powersgd_activation.py`'s basis-family sketches already
> compute grad second moments (the `grad` family's `V = Gᵀ(G·P)`; the `ticket` family returns a per-dim
> grad-2nd-moment vector, shape (H,)) — *transient* (built to construct `Q`, not maintained/applied), but
> they prove the diagonal-2nd-moment computation is **already in-tree and cross-rank-reduced**. So R5 =
> "maintain a diagonal across steps + apply + broadcast," not "invent it." Broadcast of a per-matrix
> diagonal (same shape as `M`) is **within the OOM/comm budget** (≈doubles `M`'s ~6 GB CPU state to ~12
> GB host RAM, not HBM; same DP-reduce+broadcast pattern). *(Gotcha: the `m1/m2/m3` in `anchor.py` are
> cosine geometry-probe telemetry, NOT optimizer moments.)*]**

---

### 4.4 Route ranking (all theorist verdicts in; remaining gates are empirical — see §4.3 slots)

Ordered by **(promise of a *real* surpass) × (feasibility)**. The two genuine *fixed-point* bets are
the **reframed R3** and **R5** — both inject a **second moment** (cross-rank disagreement / curvature)
outside σ(M), theorist categories 3 and 1. **R3 is now ranked #1**: theorist has **confirmed it ACCEPTs**
(the one genuinely-outside-σ(M) signal) and — unlike R5 — it has **no AdamW-`v_t` overlap problem** to
clear, so its escape is *unconditional on optimizer overlap* (only its async constraints remain open).
**R5** stays a strong #2, conditional on beating Adam's diagonal `v_t`. **R1** is the only
compression-*specific* exploration bet (category 2) but likely a pass@k edge, not a greedy surpass.
**R4** is a separate *test-time* generalization claim the fixed-point ceiling does not adjudicate. **R2**
is the lone σ(M)-measurable REJECT among the candidates.

**Best-by-criterion (theorist's framing — the practical reading of the rank):** the single linear rank
hides that the routes win on *different* axes. **R3 = #1 for a TRAIN-objective surpass** (strongest
math: it provably optimizes a different objective than `E[g]`) — *the route most likely to actually beat
dense.* **R1 = #1 for a near-term RUNNABLE bet** (no new code at n≤8; it is the thing to *run first*,
even though its prior is pass@k-only). **R4 = the generalization fallback** (concede train parity, bet
on OOD). So the program runs R1 first (cheapest signal), builds R3 in parallel (highest payoff), and
keeps R4 as the orthogonal-axis hedge.

*(Two adjacent ideas were considered and SUBSUMED, not ranked: **exploiting the on/off-policy gap** is
not a separate route — the anchor's staleness is **already** an off-policy signal folded into B2's `δ`,
so exploiting it collapses back to σ(M) unless the new information is the swarm's data-heterogeneity,
which **is** R3. **Curriculum** is admissible but generic — it changes the data distribution like R1
(upstream), not a compression-specific lever.)*

| rank | route | theorist category | escapes σ(M)? | prior | gating verdict still needed |
|---|---|---|---|---|---|
| 1 | **R3 cross-rank 2nd moment (SAM-style)** | **3 — disagreement** | **YES — ACCEPT (theorist, DIRECT)**; "the mathematically-most-promising route", variance is outside σ(M) | cond. med–high | only the async guard: cross-rank-identical after aggregation? variable-staleness-tolerant? + systems build |
| 2 | **R5 anchor-curvature preconditioner** | **1 — curvature** | ADMISSIBLE ("heavy", theorist); escapes **iff** it beats Adam's `v_t` (off-diag / lower-noise / cross-rank-shared) | cond. high | direction settled; open: empirically does it beat Adam's `v_t`, or collapse to noisier-Adam? |
| 3 | **R1 swarm diversity (n + T)** | **2 — exploration** | ADMISSIBLE (theorist: "likely pass@k only"); greedy surpass **iff** conversion-positive | <20%, likely Route-A | direction settled; open: does it relocate the greedy argmax, or pass@k-only? |
| 4 | **R4 compression-as-regularizer** | — (test-time, not fixed-point) | **ESCAPES by SCOPE-CHANGE (theorist)**; redefines goal to OOD/test, NOT the GSM8K greedy bar | low–med (~50/50 basin) | settled (scope-change); gate on OOD eval **+** sharpness/Hessian-trace measurement |
| 5 | R2 anchor-as-advantage-baseline | — (objective-side) | **NO — REJECT (theorist**: "everything deterministic-in-(G_comp,M) is capped"**)** | very low | settled REJECT; retired unless an objective-side escape is found |

**Re-ranking triggers (R3 verdict in; others gated on theorist's direct tags):**
- **R3 took #1** because theorist confirmed it ACCEPTs AND it has **no Adam-overlap problem** (Adam has
  no cross-rank-disagreement term) — so its escape is unconditional on optimizer overlap; only its async
  constraints (cross-rank-identical after aggregation, variable-staleness-tolerant) remain to verify. If
  those async constraints *fail*, R3 drops and R5 reclaims #1.
- **R5** is #2, conditional: it stays a top bet *only if* its curvature beats Adam's diagonal `v_t`
  (else it collapses to noisier-Adam). If confirmed, R3 and R5 are co-leads (two independent 2nd-moment
  escapes — run both).
- **R1** escape is contingent on **conversion-positivity**. If theorist rules R1 pass@k-only (not
  greedy-mode relocation), it stays a secondary/Route-A edge — run the cheap R1-GATE + R4-OOD probes
  before any expensive arm.
- **R2** is expected REJECT (σ(M)-measurable); retired unless theorist finds an objective-side escape.
  None of these change the Tier structure of the roadmap.

---

## 5. Roadmap — ranked, math-justified, respecting the fixed control surface

Every item runs on top of the **B2 SOTA launcher**
(`vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh`), varying **one knob**, on 4×H200 / 8×H100,
val every 25 steps, with a **matched dense control** wherever the comparison is to dense.

### Tier 1 — the surpass bets (highest expected information)

1. **R3-2ndMOMENT: cross-rank disagreement-aware step** (category 3) — **the lead bet (theorist
   ACCEPT-confirmed)**. Form a per-coordinate **cross-rank second moment** (the swarm's gradient
   disagreement; the DP axis is uncompressed so per-rank gradients are reachable) and use it to **damp
   the step where ranks fight, trust it where they agree** (SAM-style). **Mandatory control:** matched
   dense (which has no disagreement term). Decisive metric: val beats the dense band with margin.
   **Async guard (the remaining gate):** the derived direction must be **cross-rank-identical after
   aggregation** and stable as staleness varies — verify both, or it diverges the swarm. **No
   Adam-overlap problem** (Adam has no cross-rank term) ⇒ the *cleanest* category-3 escape, which is why
   it leads.
2. **R5-CHEAP: anchor diagonal-curvature preconditioner** (category 1). *(Gated on theorist
   confirming the curvature carries structure Adam's `v_t` lacks.)* **Build is moderate, not greenfield**
   — `powersgd_activation.py`'s basis sketches already compute a cross-rank-reduced diagonal grad second
   moment (the `grad`/`ticket` families), just transiently; R5 = maintain it across steps + apply +
   broadcast (within the OOM/comm budget — same shape/path as `M`, kept on CPU). Start with the
   **cheapest** variant — a diagonal proxy from `M_t`, `M_{t−1}` (no extra backward) — broadcast it
   anchor-owned (cross-rank identical) and apply as a preconditioner. **Mandatory control:** matched
   dense-AdamW, so the comparison isolates *anchor-curvature beyond Adam's own diagonal*. Decisive
   metric: does val beat the dense band with margin (not just ±0.024)? Diagnostic: cosine between the
   anchor preconditioner and Adam's `v_t` — high cosine ⇒ collapse to noisier-Adam (kill); low cosine ⇒
   genuine new structure. Heavier off-diagonal / HVP variants only if the cheap diagonal shows signal
   **and** stay within the diagnostics-OFF OOM tier.
3. **R1-GATE: dense × {T, n} surface calibration** (category 2). *Cheap kill, never run.* Sweep dense
   over a small {T ∈ 0.7,1.0,1.2} × {n ∈ 8,16} grid; if **no (T, n) lifts dense above its band**, the
   diffuse-policy hypothesis is dead before spending on the compressed arm. **Off-axis (n locked) ⇒
   new lineage with matched dense control; justify in the plan.** **Box-headroom note (systems):**
   T-sweep at n=8 is runnable as-is; **n=16 risks vLLM KV-cache/generation OOM** on the same box
   (generation memory scales with batch×n) — headroom-check or a bigger box before the n=16 cells.
   Decisive metric set: rollout_ppl, best-of-group reward, within-group reward variance, **greedy val
   (bar)**, **pass@k coverage curve**. *(R1 verdict: theorist PARTIAL-escape → must beat the new
   dense-at-(T,n) ceiling via a compression×exploration interaction.)*
4. **R1-MAIN: compressed (B2) × {T, n} vs the dense surface.** Only if the gate shows *any* lift.
   Discriminator: does (compressed − dense) **pass@k advantage grow with k**? Grows ⇒
   compression-specific edge (the dream); flat ⇒ Route-A-only, log as secondary, **not** a surpass.

### Tier 2 — the generalization edge

5. **R4: structured-regularizer OOD probe.** *Cheapest version (systems): eval-only on the **existing**
   B2 vs dense checkpoints on a new split (SVAMP / ASDiv / MATH) — **no retrain***, just a read-only
   eval pass (measurement-class like `test_freq`). Build = prepare the OOD parquet + matching reward fn.
   If B2 ≥ dense OOD while train-parity holds, that is a genuine surpass on the axis that matters — a
   test-time edge orthogonal to the train-reward bar everything else targets. *(Gated on theorist
   confirming the gap is well-defined; systems confirmed the eval-only path is feasible once the split
   is prepared.)*

### Tier 3 — deployment-realism (parity-preservation, not surpass)

6. **Variable-staleness robustness** (the *parity-only* sibling of R3 — NOT a surpass route).
   **Requires a small build** (verified: `delay_K` is a fixed integer and `AnchorStalenessQueue` serves
   a deterministic `t − delay_K`; no distribution primitive exists). Extend the queue to draw `K` from a
   **staleness distribution** (heavy right tail) and confirm B2 **degrades gracefully** (no ignition,
   parity held within noise). This validates §1.2's "tolerate variable staleness" claim and de-risks the
   whole decentralized story — *expected parity, not surpass, but load-bearing for the deployment
   narrative.* (Build must preserve cross-rank identical staleness per refresh, or the swarm diverges.)
7. **Scaling smoke test: `r/H` vs spectrum.** On the *current* model, measure how much
   boundary-gradient energy `r=77` captures (it is the proxy for §2.1). If the captured fraction is
   already marginal at `H=1536`, that predicts `r/H` must rise with model size to hold parity — the
   single most important scaling number to know before any larger-model claim.

### Conditional — only if comm budget tightens

8. **signed_ema stabilization (1-bit traffic).** *Do not start unless a real budget pressure makes
   full-width `M` transfer infeasible.* Then: attack the sharpening spiral (the entropy-collapse
   watch P1/P2/P3 + early-gate E1 triggers) *before* claiming the 1-bit-comm win. Today B2 dominates
   it; this item exists only as the bytes-floor escape hatch.

### Explicitly NOT on the roadmap

The §4.2 closed frontier (all anchor-usage levers, β_anc, generation-side mask/Gaussian,
anchor-lead, constant λ>1). They are operations on a stale dense estimate and the ceiling explains
their nullity.

---

## 6. One-paragraph executive summary

The two-circuit structure — a single slow, always-lagging, full-coverage anchor that owns the
projection basis, serving a fast swarm that exchanges only ~5%-compressed boundary gradients — is
the **generalizable contribution**, and it is the right abstraction for decentralized / geo-distributed
RL training precisely because it tolerates the unavoidable, variable staleness of a real network
without ever letting the anchor lead. **B2 `delayed_ef`** (additive, magnitude-preserving, robust)
is the reference merger at every scale we can justify and holds **dense parity at ~5% comm**;
**`signed_ema`** is a 1-bit-traffic escape hatch worth pursuing **only** if comm budgets tighten past
the point of full-width `M` transfer **and** its sharpening-spiral instability is solved. Parity is
**projected to hold** for longer context (memory-bound only) and is **most at risk** for harder,
non-stationary datasets and for `r/H` against heavier-tailed spectra at larger models. The
**surpass-dense** question stays open against a hard theorist ceiling — *you cannot beat dense by
operating on a stale estimate of dense; a surpass must inject information that stale dense estimate
structurally lacks* (theorist's three escape categories: **(a)** curvature/second-order, **(b)**
conversion-positive new exploration that **moves the greedy mode**, **(c)** multi-rank disagreement that
is information not noise). The formal test: a route surpasses **iff it injects information outside
σ(M) = σ(g(θ_t), g(θ_{t−K}))** — the stale + current dense-gradient *means*. The **two genuine
fixed-point bets both inject a second moment** σ(M) lacks. **The lead is R3, reframed (cross-rank
second moment / disagreement-as-signal, SAM-style, category (c)** — descend where swarm ranks agree,
damp where they fight; the cross-rank *variance* is information the mean `M` discards. Theorist has
**confirmed it ACCEPTs** (the one genuinely-outside-σ(M) signal), and unlike R5 it has **no Adam-overlap
problem**, so it is the cleanest escape — *the remaining gate is purely the async constraint* (stays
cross-rank-identical after aggregation + tolerates variable staleness)**)**. **Second is R5
(anchor-curvature preconditioner, category (a)** — conditional on the curvature carrying structure
AdamW's own diagonal `v_t` lacks: off-diagonal, lower-noise, or cross-rank-shared, else it collapses to
a noisier Adam; its math notably *likes* staleness, since curvature ages slowly**)** — if confirmed, R3
and R5 are co-leads, two independent 2nd-moment escapes. **R1 (swarm rollout diversity, n + T,
category (b))** is the only compression-*specific* exploration bet but likely a pass@k edge unless it
relocates the greedy argmax; **R4 (compression as a structured flat-minima regularizer)** is a distinct
*test-time / OOD* generalization claim the fixed-point ceiling does not adjudicate; **R2
(anchor-as-advantage-baseline)** is the lone σ(M)-measurable REJECT. *(Note: my first R3 framing — a
heterogeneous-staleness ensemble — is itself REJECT, just noisier stale-dense; only the cross-rank-2nd-
moment reframe escapes. Its parity-only sibling, variable-staleness robustness, stays on the roadmap as
deployment-realism, not a surpass.)* **Both cross-exams are complete: theorist has ruled all five routes
against the σ(M) ceiling and systems has ruled feasibility** — the remaining gates are *empirical*
(R3's async guard, R5's beat-Adam test, R1's conversion-positivity), the experiments that would confirm
each conditional escape, not open verdicts.
