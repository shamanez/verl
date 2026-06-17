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
| Fast-path gradient comm | bytes ratio ≈ **0.0505** (~5% of dense) | PROVEN |
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

### 4.1 The ceiling that closes the obvious routes

`theorist`'s load-bearing result (empirically confirmed by EXP-31/33):

> **You cannot beat dense by reweighting, accumulating, perturbing, or de-noising a *stale
> estimate of dense*.** The correction `δ = M_rep − G_comp_ring` reconstructs the dense
> gradient on stale data; every admissible anchor-usage lever is some operation on that stale
> dense estimate ⇒ it asymptotes to dense, never exceeds it.

**A surpass route must therefore inject information that is structurally ABSENT from a stale
dense gradient.** This is the filter every candidate below must pass.

### 4.2 CLOSED frontier — do NOT re-propose (all null on the valid-M circuit)

- **EXP-31 anchor-usage tournament:** perturbation (σ=0.01), δ-momentum (μ=0.5 parity / μ=0.9
  regress), adaptive dose (ratio/cosine, parity), control-variate (covariance gate failed,
  cov≈0), sub-basis rank-2 tail (early boost, no surpass).
- **EXP-33 β_anc sweep** {0, .25, .5, .75, 1}: 0.738 / 0.740 / 0.753 / 0.722 / degenerate.
  β∈[0,0.5] flat free-averaging; β=0.5 nominal best but inside ±0.024 noise; β=1 cold-`M`
  collapse. β=0 stays default.
- Also dead: constant λ>1, naive `M`-EMA, sign-replacement, Step-C/forward-Q, **generation-side
  mask/Gaussian as a surpass lever** (train-only ⇒ repeats the PowerSGD null), anchor-lead,
  delta-subbasis.

These are all operations on the stale dense estimate. The ceiling explains *why* they are null.

### 4.3 The candidate routes (each must escape §4.1)

Each route is annotated with **[theorist: validity]** and **[systems: feasibility]** verdicts.
*(Status at this writing: peer verdicts were requested early and repeatedly but had not landed by
finalization; per the cross-exam protocol, theorist/systems slots that did not return are marked
**"unvetted (\<peer\>)"** with my stated prior, and any feasibility fact I could verify directly from
the code is labeled **"code check by strategist"**. These should be confirmed by the peers before
acting.)*

---

#### R1 — Swarm rollout diversity (raise `n` + rollout temperature) — **LEAD**

**Mechanism.** Raise rollouts-per-prompt `n` and/or rollout sampling temperature `T` **on the
compressed fast circuit**, with a **mandatory matched dense × {T, n} control**.

**Why it escapes the ceiling.** The information injected is **exploration diversity in the
rollout/data distribution** — it changes *which* gradient is being compressed, not *how* a fixed
gradient is reconstructed. The ceiling is about gradient-reconstruction fidelity (the codec +
anchor side); R1 acts **upstream of the codec entirely**, on the data-generating distribution. The
compression-specific bet is that the diffuse policy a compressed run produces (measured:
rollout_ppl 1.40 vs dense 1.24) has **steeper d(val)/dn** — more within-group reward variance to
convert — so at matched (T, n) the compressed run's extra diversity pays where dense's does not.

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

> **[theorist: validity — unvetted (theorist); my prior: ESCAPES — it acts upstream of the codec on
> the rollout/data distribution, changing *which* gradient is compressed, not how a fixed one is
> reconstructed; honest deliverable is likely a pass@k coverage edge (Route A), greedy-mode relocation
> (Route B) is the high bar]**
> **[systems: feasibility — unvetted (systems); code check by strategist: RUNNABLE via launcher
> overrides as a new lineage (n + rollout T), with a matched dense × {T,n} control; off the locked
> "generation" axis ⇒ requires the separate justification the conversion spine already supplies]**

---

#### R4 — Compression as a flat-minima regularizer (test-time generalization edge) — **#2**

**Mechanism.** Hypothesis that the low-rank projection is an **implicit regularizer** (it
systematically removes off-subspace gradient components every fast step), biasing the solution
toward flatter minima that **generalize better** than dense on a held-out / harder split — even if
**train reward only reaches parity**.

**Why it escapes the ceiling.** It does **not** try to beat dense on the *training objective* at
all — it claims a **test-time generalization edge**. The ceiling is about matching the dense
*gradient*; R4 concedes the gradient and bets on the *solution's* generalization. This is a
categorically different claim, so the ceiling does not bind it.

**Caveats / risk.** EXP-31's L4 perturbation (isotropic ξ, σ=0.01) was a **regularization control**
and was **null** — but it touched the anchor nowhere and was *isotropic*. **Compression noise is
structured and biased** (it removes the off-subspace component; 42:1 SNR, ~0.06% dropped per
`exp25-collapse-gradient-flow`), so R4 is **not** identical to L4 — it is a *structured* regularizer,
not isotropic noise. **The blocker:** GSM8K's greedy val on the *same* distribution may make
"generalization gap" ill-defined; R4 needs an **OOD / harder eval split** (e.g. GSM-hard, MATH
subset) to even be measurable, and the project's fixed control is GSM8K. Feasibility hinges on
whether an OOD eval split is wireable as a *measurement-only* knob (validation is read-only, so this
may be sanctioned like `test_freq`).

> **[theorist: validity — unvetted (theorist); my prior: ESCAPES — it concedes the training objective
> and claims a *test-time / OOD* generalization edge, a different claim than matching the dense gradient;
> open question for theorist: does GRPO-on-GSM8K make the generalization gap ill-defined?]**
> **[systems: feasibility — unvetted (systems); code check by strategist: WIREABLE read-only]** — `val_files` is a
> config path (`_generated_ppo_trainer.yaml`); pointing it at a different test parquet (GSM-hard / a MATH
> subset) is a **measurement-only knob** (validation is read-only, no training change — same class as
> `test_freq`). The only build is *preparing* the OOD parquet + a matching reward fn; no trainer-code
> change. Awaiting systems' confirmation.

---

#### R3 — Heterogeneous-staleness ensemble — **#3 (most async-native, likely ceiling-bound)**

**Mechanism.** In the real swarm, workers consume `Q`/`M` at **different** staleness. Treat the
**cross-worker spread** of those staleness-varied gradients as a structured signal — e.g. an
ensemble/averaging that the lock-step `K=5` sim cannot produce.

**Why it *might* escape — and why it probably doesn't.** Variable staleness is genuinely a property
of the deployment regime the lock-step sim suppresses. **But** averaging over staleness is still
averaging over *stale dense estimates* — by §4.1 the result is a (noisier) stale dense estimate, so
the ceiling likely **holds**. Its value is more as a **robustness/parity-preservation** result under
realistic variable lag than as a surpass route. **It is the most deployment-native item**, so it
belongs on the roadmap even if it is parity-only.

> **[theorist: validity — unvetted (theorist); my prior: COLLAPSES — averaging over staleness is still
> averaging stale dense estimates ⇒ a (noisier) stale dense estimate; value is robustness, not surpass]**
> **[systems: feasibility — unvetted (systems); code check by strategist: NEEDS BUILD]** — `delay_K` is a **single
> integer** in `comm_eff.py` (`delay_K: int`, B2=5) and `AnchorStalenessQueue.get_stale` returns the
> deterministic `t − delay_K` snapshot (retaining only `delay_K+1` snapshots). There is **no jitter /
> distribution / per-worker staleness** primitive today — the substrate supports only **fixed uniform
> K**. R3 (and the Tier-3 robustness item) is therefore **not config-only**: it requires extending the
> queue to serve a *staleness distribution*. Awaiting systems' confirmation.

---

#### R2 — Anchor as advantage-baseline (objective-side, not gradient-side) — **#4 (likely ceiling-bound)**

**Mechanism.** Use the stale `M` to form a **variance-reduction baseline on the GRPO advantage**
(the objective side), explicitly *different* from EXP-31's control-variate which gated the
*gradient*.

**Why it probably doesn't escape.** In GRPO the advantage is already **group-mean-normalized**, so
a baseline is a control variate on top of that — and a *stale* baseline injects only a stale
estimate of the same quantity ⇒ likely collapses to the ceiling (and EXP-31's gradient-side
control-variate already failed with cov≈0). Kept only to ask theorist whether *any* objective-side
use of the anchor escapes; absent a positive answer it is **retired**.

> **[theorist: validity — unvetted (theorist); my prior: COLLAPSES — a stale baseline on the
> group-mean-normalized GRPO advantage is bias/variance on a stale estimate of the same quantity; key
> open question for theorist: does *any* objective-side anchor use escape the ceiling?]**
> **[systems: feasibility — unvetted (systems); code check by strategist: NEEDS NEW CODE (advantage-side
> hook does not exist; the only anchor combiners today are gradient-side mergers in `spectral_filter.py`)]**

---

### 4.4 Route ranking (my priors; peer verdicts unvetted at finalization — see §4.3 slots)

| rank | route | escapes ceiling? (my prior) | prior | gating verdict still needed |
|---|---|---|---|---|
| 1 | **R1 swarm diversity (n + T)** | YES (acts upstream of codec, on data dist) | <20%, likely Route-A | **theorist**: greedy-mode relocation possible, or pass@k-only? |
| 2 | **R4 compression-as-regularizer** | YES (test-time edge, concedes train objective) | low–med | **theorist**: is the gen-gap well-defined on GRPO/GSM8K? |
| 3 | R3 heterogeneous-staleness | WEAK (likely noisier stale-dense) | low | **theorist**: confirm collapse; value = robustness, not surpass |
| 4 | R2 anchor-as-advantage-baseline | NO (likely B/V collapse) | very low | **theorist**: does *any* objective-side anchor use escape? |

**Re-ranking trigger:** if theorist rules R1 can only deliver a pass@k edge (not greedy-mode
relocation), R1 and R4 effectively tie — both become "edge on a different axis than the locked greedy
bar" — and the program should run **both** measurement-cheap probes (R1-GATE, R4 OOD) before either
expensive arm. If theorist finds an objective-side escape for R2, it jumps the ranking (it would be the
first route to inject info on the *objective* side). Neither changes the Tier-1/2/3 roadmap structure.

---

## 5. Roadmap — ranked, math-justified, respecting the fixed control surface

Every item runs on top of the **B2 SOTA launcher**
(`vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh`), varying **one knob**, on 4×H200 / 8×H100,
val every 25 steps, with a **matched dense control** wherever the comparison is to dense.

### Tier 1 — the surpass bets (highest expected information)

1. **R1-GATE: dense × {T, n} surface calibration.** *Cheap kill, never run.* Sweep dense over a
   small {T ∈ 0.7,1.0,1.2} × {n ∈ 8,16} grid; if **no (T, n) lifts dense above its band**, the
   diffuse-policy hypothesis is dead before spending on the compressed arm. **Off-axis (n locked) ⇒
   new lineage with matched dense control; justify in the plan.** Decisive metric set:
   rollout_ppl, best-of-group reward, within-group reward variance, **greedy val (bar)**, **pass@k
   coverage curve**. *(Gated on theorist confirming R1 escapes the ceiling.)*
2. **R1-MAIN: compressed (B2) × {T, n} vs the dense surface.** Only if the gate shows *any* lift.
   Discriminator: does (compressed − dense) **pass@k advantage grow with k**? Grows ⇒
   compression-specific edge (the dream); flat ⇒ Route-A-only, log as secondary, **not** a surpass.

### Tier 2 — the generalization edge

3. **R4: structured-regularizer OOD probe.** Measure B2 vs dense on a **held-out / harder eval
   split** (read-only measurement knob, like `test_freq`; no training change). If B2 ≥ dense OOD
   while train-parity holds, that is a genuine surpass on the axis that matters. *(Gated on
   systems confirming an OOD split is wireable read-only, and theorist confirming the gap is
   well-defined.)*

### Tier 3 — deployment-realism (parity-preservation, not surpass)

4. **R3: variable-staleness robustness.** **Requires a small build** (verified: `delay_K` is a fixed
   integer and `AnchorStalenessQueue` serves a deterministic `t − delay_K`; no distribution primitive
   exists). Extend the queue to draw `K` from a **staleness distribution** (heavy right tail) and
   confirm B2 **degrades gracefully** (no ignition, parity held within noise). This validates §1.2's
   "tolerate variable staleness" claim and de-risks the whole decentralized story — *expected parity,
   not surpass, but load-bearing for the deployment narrative.* (Build must preserve cross-rank
   identical staleness per refresh, or the swarm diverges — §1.2.)
5. **Scaling smoke test: `r/H` vs spectrum.** On the *current* model, measure how much
   boundary-gradient energy `r=77` captures (it is the proxy for §2.1). If the captured fraction is
   already marginal at `H=1536`, that predicts `r/H` must rise with model size to hold parity — the
   single most important scaling number to know before any larger-model claim.

### Conditional — only if comm budget tightens

6. **signed_ema stabilization (1-bit traffic).** *Do not start unless a real budget pressure makes
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
operating on a stale estimate of dense* — and the only routes that plausibly escape it inject
information that ceiling lacks: **R1 (swarm rollout diversity, n + T, acting upstream of the codec on
the data distribution — the vetted lead, prior <20%, gate never run)** and **R4 (compression as a
structured flat-minima regularizer claiming a test-time / OOD generalization edge, conceding the
training objective entirely)**. R3 (heterogeneous staleness) and R2 (anchor-as-advantage-baseline)
most likely collapse back to the ceiling and are kept for robustness / completeness, not as surpass
bets.
