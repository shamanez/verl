# Can communication-efficient GRPO SURPASS dense? — feasibility + theoretical paths

> **Author:** theorist (team `beat-dense-grpo`, task #1) · 2026-06-13
> **Question:** starting from the current best comm-eff solution (B2), can the method beat — not just
> match — the dense baseline at GSM8K val@50 greedy mean@1? Honest priors, grounded in this program's
> largely-falsified surpass history.
> **Sources (read, not invented):** `runs/EXP-30/{verdict.md, PATH_FORWARD.md,
> final_synthesis/{best_solution_verdict,forward_checks}.md, pathforward/{mechanist,critic,strategist}.md,
> PROGRESS.md}`, `LOG.md`, `runs/SUMMARY.md`, `.claude/GOAL.md`, dense-rerun log (`73ntu76u`), and the
> surpass-dense memory lineage (`surpass-dense-conversion-spine`, `exp25-collapse-gradient-flow`).
> **Read-only** except this file. Every number is quoted/recomputed from those artifacts.

---

## 0. Verdict in one line

**P(B2-lineage comm-eff beats same-config dense at GSM8K val@50 greedy, beyond nondeterminism, within
the realistic substrate) ≈ 8–12%.** The single defensible number is **~10%**. The overwhelmingly most
likely realistic outcome is **parity (seed-replicated 0.75–0.78 band overlap with dense), with a
possible secondary pass@k/diversity edge that does *not* show up on the greedy bar.** Beating dense is
not the expected result of any route on the current map; it is a low-probability upside that one route
(additive off-principal sub-basis, "candidate f") could deliver, and even that route's prior is below
even-odds. This is consistent with the program's entire surpass-dense history, which has falsified every
naive route and left exactly one surviving theoretical bet.

---

## 1. The current gap — what "beat" even means now

The dense baseline was **corrected upward** on 2026-06-13. The honest bar is no longer the old
`5e2jpho9` 0.7536 (old code); it is the **same-code, same-hyperparameter** dense rerun `73ntu76u`:
trajectory **0.0788@0 / 0.7566@25 / 0.7839@50** (recomputed from the log,
`val-core/openai/gsm8k/acc/mean@1`). So:

| quantity | val@50 | note |
|---|---|---|
| **dense — same code, same config (the real bar)** | **0.7839** | `73ntu76u`, all comm_eff counters 0 |
| dense — old code (historical) | 0.7536 | `5e2jpho9`, predates this branch |
| **B2 — best comm-eff (K-delayed exact residual)** | **0.7528** | `u9okvgzz` / ext100 `b59ncque` |
| B2 gap vs same-config dense | **−0.031 (≈96%)** | ~41 problems of 1319 |

Two facts make the bar harder than "−0.031 looks small":

1. **Step 0 is establishing PARITY, not beating.** B2 reaches the *old* dense draw (−0.0008) but sits
   ~3 pts below the *current-code* draw. The dense val@50 is best read as a **band ≈ 0.75–0.78** across
   two single draws, with **rollout nondeterminism ≈ ±0.024/draw** (measured: B2 0.7036@25 vs ext100
   0.7278@25, *identical config*). Binomial SE at N=1319 is **±0.0119** (critic T2). So a single B2 seed
   at 0.7528 is **statistically indistinguishable from dense** — but equally is **not proven equal**.
   The binding prerequisite for *any* "surpass" claim is **seed-replicating both B2 and dense** so the
   bar is a distribution, not a point. You cannot beat a number you have not pinned.

2. **To "beat" dense you must exceed the *upper* of two noisy draws by more than noise.** A defensible
   surpass claim needs B2-lineage ≥ ~0.79 *with replicates*, i.e. **+0.04 over B2's current point**, on
   the greedy mean@1 bar — about **53 more correct problems**. That is a large move relative to every
   one-knob delta this program has measured (the entire merger-vs-no-merger gain is +0.12, but that only
   buys *parity*; the residual gain over the blend is +0.0106; A0-clean over floor is +0.011). **Nothing
   on the current map produces a +0.04 over B2 with any confidence.**

**Framing that matters for honesty:** B2's headline was always "near-parity, NOT established." The
question "can we surpass" is therefore two gates deep — first *establish parity* (seed replicates,
~5 GPU-hr, the binding statistical fix), then *exceed it*. The priors below are for the second gate
*conditional on the first being won*, because there is no sense pricing surpass against an unpinned bar.

---

## 2. The program's surpass-dense history — what is already falsified (so we don't relitigate)

This program has run the surpass question hard for ~7 experiments. The naive routes are **dead**, and
the reasons are mechanistic, not incidental. Any new surpass bet must clear all of these.

| route | status | why it cannot surpass (cite) |
|---|---|---|
| **Compression noise as a regularizer** | **FALSIFIED** | PowerSGD's dropped residual `(I−P)·G` is a **persistent BIAS, not zero-mean noise** — `Q` converges to a stable dominant subspace (recon flat ~0.024), so the *same* off-subspace component is dropped every step (`exp25-collapse-gradient-flow`). SNR ~42:1, dropped energy 0.058%. A biased, low-variance, fixed-direction error is the *opposite* of a beneficial exploration regularizer; it is exactly the thing error-feedback exists to *remove*. F1 (EXP-30) sharpens this: the codec error is ~92% of the compressed gradient's energy and near-orthogonal to the truth — it is contamination to cancel, not noise to exploit. |
| **Anchor / stale-EMA as extra information** | **FALSIFIED** | A stale anchor is **not extra-dense information** — dense sees the full fresh activations; the anchor sees a *stale, compressed* view. Adam already carries fresh β1=0.9 momentum, so a stale β=0.95 EMA "adds little" (`exp25-collapse-gradient-flow`). The anchor's job is *parity recovery* (replace the unrealizable clean step), not *surpass*. |
| **signed_ema / sign-replacement merger** | **DEAD** | Sign-disagreement with the live gradient is ~50% (a coin-flip) and **structural**, not staleness — destroys the per-coordinate sign-cancellation that regularizes the GRPO step; net-harmful at every α (EXP-25 STOP). |
| **Error-feedback / entropy routes** | **PARITY, not surpass** | EF on the residual recovers the dropped bias → reaches parity (B2 is the success case). It does not *exceed* dense because it is *reconstructing* the dense gradient, not improving on it. Entropy-shaping is RED-FLAGGED: `entropy_coeff>0` pushes the exact length-hack axis the (absent) brakes can't fight. |
| **mask-p / Gaussian as a "generation" lever** | **DEMOTED** | The activation mask is **TRAIN-ONLY** (`state.py` TRAIN_TAG; rollouts are uncompressed vLLM). So mask/Gaussian/PowerSGD are the *same train-side gradient-perturbation class* — and that class already proved (PowerSGD) it makes a more-diffuse policy that **fails to convert** (`surpass-dense-conversion-spine`). Generating *more* diversity repeats the PowerSGD null. |
| **update-energy / hybrid-Q (gradient basis)** | **DEAD** | EXP-26 Step C: anti-converts. Because rollouts are uncompressed, a gradient-tuned `Q` degrades the forward activation reconstruction the codec must serve. |
| **clean_cadence (periodic dense step)** | **EXCLUDED** | Not communication-efficient (full-H transfer) and would itself be stale on a real PP link. The whole point of the anchor circuit is to *replace* it. Even when it worked it only *tied* dense (0.7415), never beat it. |

**The single surviving theoretical bet on record** (`exp25-collapse-gradient-flow` §9,
`surpass-dense-conversion-spine`): a **zero-mean, variance-controlled, TUNABLE perturbation** that acts
as *beneficial exploration noise* (not bias) — coupled, ideally, to something that lets that diversity
**convert to reward on the greedy bar** (denser credit, or a training-time mode relocation). This is the
only route the mechanist did not falsify, and it was rated **<20% prior** even before EXP-30. EXP-30
*lowered* that prior, for the reason in §4.

---

## 3. The decisive structural reason surpassing is hard: the conversion ceiling + the greedy bar

Two findings, one from the surpass team and one re-confirmed in weight space by EXP-30, jointly explain
why comm-eff keeps *tying* and never *beating*:

**(a) Comm-eff is CONVERSION-limited, not generation-limited.** Compression already makes a *genuinely
more diffuse* policy — the uncompressed vLLM generator shows `rollout_ppl` 1.40 (PowerSGD) vs 1.24
(dense) at step 25, a real ~13% diversity edge, measured on an engine with *no* compression hooks
(`exp25-collapse-gradient-flow`). **And it fails to pay**: score 0.688 vs 0.786, val ties (0.741) not
beats. The diversity is *lost, not harnessed*. The bottleneck is the *conversion* of diversity → reward,
and nothing in the compression mechanism improves conversion — if anything the biased gradient *hurts*
it.

**(b) The bar is GREEDY mean@1, which is blind to diversity unless the mode moves.** Val is
`do_sample=False, n=1, T=0` (`surpass-dense-conversion-spine`). A more-diffuse policy spreads probability
mass *off* the mode; greedy decoding reads *only* the mode. So **any diversity advantage is invisible to
the bar unless training relocates the trained argmax** — which is a much stronger requirement than
"explore more." This is the structural reason a compression-induced diversity edge cannot, by
construction, show up as a greedy-val surpass: the only diversity that counts is diversity that changes
which single answer the greedy decode picks, and a biased/noisy gradient is more likely to *blur* the
mode than to *sharpen it onto a better answer*.

**(c) EXP-30's F1 closes the door tighter still.** On identical (batch, θ), the compressed gradient is
**~92% orthogonal codec error**, with the true gradient at ~⅓ norm and statistically ⊥ (cos +0.007).
The whole engineering achievement of B2 is to *cancel* that 92% artifact and recover something close to
the *dense* gradient (telescoping: `G_corr ≈ G_anc_rep`). **B2's ceiling is therefore the dense gradient
itself** — it is an unbiased reconstruction of dense-on-(slightly-stale)-data. A faithful reconstruction
of the dense update **cannot beat dense**; at best it equals it (minus the staleness drift). The mild
late decay in ext100 (0.7536@50 → 0.7400@100) is the staleness drift showing up — evidence that B2 is
*below* dense at horizon, not above. **To beat dense, comm-eff must inject something dense does NOT
have, and do it on the greedy bar.** That is the crux of §4–§5.

---

## 4. Can B2 specifically be pushed past dense? (the δ-residual + perturbation angle)

The task asks the sharp version: B2's update is ≈ dense-on-stale-data; can a **deliberate controlled
perturbation on top** (or a partial residual λ<1 as a noise knob, or residual-compression that injects
structured noise) plausibly beat dense? **Argued honestly, the answer is: almost certainly no for the
perturbation framings, and only-maybe for one structural variant.** Walking each:

### 4a. Partial residual λ<1 as a "noise knob" — NO (it re-admits bias, not zero-mean noise)

λ=1 is the *unique* exact-telescoping value: it fully subtracts the 92%-energy codec artifact. Setting
**λ<1 deliberately leaves a fraction (1−λ) of that artifact uncancelled** (`forward_checks` candidate c).
But that uncancelled remainder is the **biased, near-orthogonal, fixed-direction codec error** — exactly
the thing §2 row 1 falsified as a regularizer. It is not zero-mean noise; it is structured contamination
pointing ~90° off the true gradient. Re-admitting it is pure downside on the conversion axis and *lowers*
val toward the plain floor (0.6300). λ<1 is a damping knob at best (it might gentle the late decay), and
it is **dominated** by shrinking delay_K, which reduces the staleness drift *without* re-admitting
artifact. **λ<1 cannot surpass; it can only lose less slowly.** (λ>1 is worse: it over-subtracts,
injecting a scaled `−G_comp` that is a persistent large-magnitude push → *raises* the carrier → ignition
risk. Forbidden by the carrier law.)

### 4b. A deliberate exogenous perturbation on top of B2 — NO (the carrier law convicts it)

This is the program's hardest-won negative and it directly kills the "add exploration noise" instinct.
**The no-KL/no-entropy GRPO surface has no brake on response length.** EXP-25/27 established the
**carrier law**: any **persistent, fixed-direction (exogenous) force** added to the update rectifies —
via the token-mean ratchet (~86× tail amplification) — into a length-explosion reward-hack that destroys
val. The discriminator is *not* dose (EXP-27 proved dose-capping only *delays* ignition ~20 steps,
"lag-not-dose"); it is **carrier presence**. A "controlled perturbation on top" of B2 is, almost by
definition, a persistent exogenous direction — the precise thing that ignites. The one reason B2 itself
survives is that its δ is **endogenous**: it cancels the circuit's *own* artifact and re-injects the
*current* objective's gradient, adding no direction dense wouldn't also follow (mechanist §5e, the
endogenous-vs-exogenous discriminator). **The moment you add a deliberate perturbation that is NOT the
honest dense direction, you create an exogenous carrier and re-enter the EXP-27 failure mode.** A
zero-mean *stochastic* perturbation (fresh i.i.d. each step) escapes the *persistence* clause in
principle — but then EXP-16's Gaussian-probe mechanism bites: on near-zero GRPO coordinates Adam's
per-coordinate √v normalization turns small zero-mean noise into `sign(noise)` = a **random walk** on
those coordinates (`exp25-collapse-gradient-flow`, EXP-16 pearson 0.006 at near-dense norm). That is
update-space damage, not exploration. **Neither persistent nor i.i.d. perturbation has a credible path to
surpass; both have credible paths to harm.**

### 4c. Residual-compression that injects *structured* noise — NO as a surpass lever (YES as a savings lever)

Compressing δ to ≪77 columns (candidate b) is a real **GOAL-3 (bytes) upside** — δ inherits the
gradient's rank-~2 concentration (m7), so it should sketch into ~8–16 columns losing almost nothing. But
the *truncation error* it introduces is, again, a **biased, structured** residual (the dropped
low-energy tail), not beneficial zero-mean noise. Its expected effect on val is **≤ 0** (it can only
degrade B2's reconstruction of the dense gradient). It is worth doing because it can hold parity at
*strictly better savings* — which is half of "done" — but it is **not a surpass mechanism.** Filing it
under "beat dense" would be a category error.

### 4d. The honest read on B2-as-base

B2's mechanism is *subtractive reconstruction of the dense gradient*. Every "perturbation on top"
framing either (i) re-admits the biased artifact (λ<1), (ii) creates an exogenous carrier that ignites
(deliberate perturbation), or (iii) adds biased truncation error (residual compression). **None inject
something dense lacks while staying on the safe side of the carrier law and the greedy bar.** B2 is a
parity engine by construction; pushing it *past* dense requires leaving the "reconstruct the dense
gradient" frame entirely — which only §5 attempts.

---

## 5. The one route with a non-trivial (but sub-even) surpass prior: additive off-principal sub-basis

The only candidate on the entire forward map that *attacks the root cause* rather than reconstructing
dense is **R5 / candidate (f): an additive off-principal replay-gradient sub-basis** (strategist §3,
mechanist §4, `forward_checks` f). The theoretical case for why it *could* beat dense — and the honest
case for why its prior is still <50% — both follow directly from the measured geometry:

**Why it could exceed parity (the steelman):**
- F1 + m7 say the true RLVR gradient lives in ~2 directions that are **statistically orthogonal to, and
  outside, the act-basis Q's subspace** (cos(G_true, G_comp) ≈ 0; off-principal share 0.68 in EXP-26).
  The act-basis codec *structurally misses* the gradient's principal directions.
- B2 already *proves those missing directions are recoverable and helpful* when injected additively (δ =
  A − C injects them via the anchor's full-rank replay backward).
- A first-class, separately-sketched gradient sub-basis (rank 2–4) routed *only* into the correction term
  — leaving the forward act-Q untouched (so it dodges the EXP-26 Step-C corner) — would inject the
  rank-~2 true direction **more cleanly and possibly more completely than B2's telescoping approximation**
  (B2's cancellation is imperfect by the K-tick drift term; a direct sub-basis has no drift term). If the
  act-basis is *also* mildly hurting conversion (by feeding the optimizer a basis misaligned with where
  reward concentrates), routing the true directions directly could in principle let the optimizer convert
  the existing diversity edge (§3a) that plain compression leaves on the table.

**Why the prior is still well below even-odds (the honest deflation):**
1. **The ceiling argument (§3c) still bites.** A sub-basis sketched from the *replay (= valid PG)*
   gradient injects the **same true gradient dense already follows**. Injecting the dense gradient's
   own principal directions, more cleanly, gets you *to* dense, not *past* it. To beat dense the
   sub-basis would have to inject a direction that is *better than the dense gradient* — and there is no
   measured source of such a direction. The route's realistic ceiling is **clean parity**, with surpass
   requiring an unproven "compression-specific conversion" effect (§3a) to actually materialize on the
   greedy bar — which the entire conversion-spine analysis rates <20%.
2. **The carrier law gates it hard.** A *static* gradient sub-basis injected every step **is** the
   persistent exogenous direction the carrier law convicts. It is admissible *only* K-delayed /
   fire-refreshed / β_anc=0 — and even then it carries the m6 ≈ 0.62 base persistence (τ/cadence ≈ 2,
   marginal). It inherits B2's *censored* stability, plus a fresh ignition surface from the injected
   directions. Non-trivial probability mass sits on "ignites" rather than "surpasses."
3. **It is the most expensive and most-confounded bet** (~20 GPU-hr, substantial new code), and it is
   *gated* behind establishing the B2 baseline (you cannot claim surpass against a single-seed 0.7528)
   and behind the Q-rotation diagnostic (if Q already rotates enough, B2's simpler form suffices and the
   sub-basis is unnecessary; if Q is frozen, the sub-basis is *necessary* just to reach the directions —
   which is a parity argument, not a surpass argument).

**My prior on (f) specifically:** P(seed-replicated val above the dense band, beyond noise) ≈ **12–18%**;
P(clean parity, no surpass) ≈ 45%; P(parity-but-censored / mild decay like B2) ≈ 25%; P(ignites) ≈
15–20%. It is the *only* route I would give double-digit surpass odds — and it is still more likely to
*tie* or *ignite* than to *beat*.

---

## 6. The pass@k vs greedy-mean@1 angle — the realistic "win" that isn't a greedy surpass

This is where comm-eff is *most* likely to show a genuine, defensible advantage — and it is **not** the
greedy bar the GOAL defines.

- The measured diversity edge is real (rollout_ppl 1.40 vs 1.24 from the *uncompressed* generator; the
  steps-5–45 ~0.08–0.12 nat entropy edge is corroborated, not a codec-warmup artifact —
  `exp25-collapse-gradient-flow`). A more-diffuse policy with the *same* mode quality has, by
  construction, a **higher pass@k for k>1**: more independent samples cover more of the answer space.
- The conversion-spine analysis split this cleanly: **Route A** = eval-time diversity → pass@k (a
  *more-lenient* bar; log as SECONDARY, never the goalpost) vs **Route B** = training-time diversity
  relocates the mode → greedy mean@1 (the *real* bar). The strategist's honest most-likely outcome was
  **"Route-A-only: a real pass@k edge, greedy ties."**
- The **discriminating signature** if anyone runs it: the **pass@k coverage curve**. If
  (compressed − dense) pass@k advantage **grows with k**, the edge is *compression-specific* (real); if
  it is **flat in k**, it is generic and dense catches up. With a mandatory dense×{T,n} control (raising
  dense's temperature/samples is the cheap kill — if dense matches the curve, there is no comm-eff edge).

**Honest verdict on this angle:** there is a **plausible (~30–40%) chance comm-eff shows a real pass@k /
diversity edge over dense** — a genuine, publishable "compression buys exploration coverage" result. But
this is a **different claim** from beating dense on the GOAL's greedy bar, and conflating them would
overclaim. On the *defined* bar (greedy mean@1), the diversity edge is invisible unless it relocates the
mode (§3b), which it has not been shown to do and which a biased gradient works *against*. **The pass@k
edge is the most likely "win," and it is a secondary win.**

---

## 7. Honest verdict — the number, defended, and the most likely outcome

**P(beat same-config dense at GSM8K val@50 greedy mean@1, seed-replicated, beyond ±0.024 nondeterminism,
on the realistic comm-eff substrate) ≈ 10%** (range 8–12%).

**Defense of the number** (decomposed, so it is falsifiable, not a vibe):
- It must be **conditional on first establishing parity** (seed replicates). Unconditionally — i.e.
  including the ~50–60% chance the program never even pins a seed-replicated parity and the surpass
  question stays moot — the number is lower, ~5–7%. I quote the *conditional-on-parity-established*
  figure (~10%) because that is the decision-relevant one for "should we chase surpass."
- The mass comes almost entirely from **one route (f, additive sub-basis): P(surpass | run) ≈ 12–18%**,
  times P(it gets run and clears its gates and the baseline is pinned) ≈ 0.6–0.7, ≈ **8–12%**. Every
  other route has P(surpass) ≈ 0 by construction (they reconstruct dense, re-admit bias, or live on the
  wrong bar). I am **not** double-counting them.
- It is bounded *above* by the **§3c ceiling argument**: B2's mechanism, and the only credible surpass
  route's mechanism, both *reconstruct or inject the dense gradient's own directions* — a faithful copy
  of dense cannot beat dense; surpass requires a compression-*specific* conversion effect that the
  conversion-spine analysis rates <20% and that has never been observed to move the greedy bar.
- It is bounded *below* away from ~0 only because (f) is a genuine root-cause attack on a *measured*
  defect (basis mismatch) with B2 as positive proof the missing directions help — that is more than the
  falsified routes ever had.

**The most likely realistic outcome, ranked:**
1. **Parity, seed-replicated (~45%)** — comm-eff (B2 or B2+shorter-K or B2+sub-basis) lands inside the
   0.75–0.78 dense band, indistinguishable from dense within noise. This **is the program's real
   deliverable**: GOAL parity (criterion 2) on the realistic substrate, *with* a measured savings number
   (GOAL-3, via the honest-bytes accounting + residual-compression). "Stable + parity + savings +
   reproducible" with no surpass is a complete, honest win against the GOAL as written — the GOAL asks
   for **≥ dense within noise**, which parity satisfies.
2. **A secondary pass@k / diversity edge (~30–40%, can co-occur with #1)** — a real
   "compression-buys-coverage" result on a more-lenient bar, logged as secondary, *not* a greedy surpass.
3. **Mild late decay / censored-stability (~25%)** — B2-lineage holds parity at the measurement point
   but drifts below dense at horizon (the ext100 0.7400@100 pattern), i.e. comm-eff is *slightly below*
   dense, not above.
4. **Genuine greedy surpass (~10%)** — only via (f), only if the compression-specific conversion effect
   materializes on the greedy bar, only after parity is pinned. The upside case, not the base case.
5. **Ignition / STOP (~10–15% on any carrier-bearing surpass attempt)** — the cost of pushing harder
   (sub-basis, λ>1, deliberate perturbation) on a brakeless surface.

**The one sentence.** Surpassing dense on the greedy bar is a **low-probability (~10%) upside available
through a single gated route (additive off-principal sub-basis), not the expected result** — the
expected result is **seed-replicated parity plus a likely secondary pass@k edge**, which is itself a
complete win against the GOAL; the honest recommendation is to **pin parity first (seed replicates +
honest bytes), treat pass@k as the realistic "win," and treat greedy-surpass as a scoped, gated, sub-even
bet — not the headline.**

---

## 8. What this implies for the program (one paragraph, for the ranked-program task #2)

Spend the cheap tier first — **seed-replicate B2 (pins the bar, ~5 GPU-hr), honest-bytes accounting
(closes GOAL-3, ~0 GPU-hr), and the C2−C3 / Q-rotation read (gates the basis route, ~0 GPU-hr)** — none
of which can be wasted and all of which the surpass question *depends on*. Then **shorter delay_K** as
the cheapest on-mechanism push (reduces the staleness drift that puts B2 *below* dense at horizon; it
defends parity, it does not promise surpass). Keep **residual-compression** as the GOAL-3 *savings*
upside (not a surpass lever — don't mislabel it). Reserve **the additive off-principal sub-basis** as the
*one* scoped, gated, run-later bet that could surpass — explicitly contingent on parity being pinned, the
basis being shown frozen (so the sub-basis is the *only* way to reach the missing directions), and a
mandatory dense×{T,n} pass@k control to separate a real compression-specific edge from a generic one. Do
**not** relitigate the BLOCKED routes (β_anc-EMA, signed_ema, hybrid-Q, clean_cadence, dense re-run,
λ-grids, deliberate exogenous perturbation) — each is falsified or carrier-convicted, and re-running them
burns compute re-confirming known negatives.
