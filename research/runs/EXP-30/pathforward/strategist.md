# EXP-30 path-forward — ranked program toward GOAL.md

> Strategist deliverable for team `exp30-pathforward`, task #3. Ranks the next-experiment
> program after EXP-30's PASS. Sources read: `.claude/GOAL.md`, `runs/EXP-30/verdict.md`
> (F1–F5), `runs/EXP-30/stepA_gate.md`, `.claude/plans/30.md`, issue #28 (+ the m1–m7
> comment 4693870612), `runs/SUMMARY.md`, `LOG.md`. Numbers are quoted, not re-derived.
>
> **This file proposes; it does not authorize.** No threshold below softens a pre-registered
> rule; no cell here launches without the operator flipping an issue to `status:approved`.

## 0. Where the program actually stands (one paragraph, so the ranking is legible)

EXP-30 produced the program's **first emission-free correction-carrying cell**: B2
(`G_corr = G_comp + δ`, δ = K-delayed exact codec residual on the EXP-29 replay substrate,
λ=1, β_anc=0) reached **val@50 = 0.7528** — past the 0.7210 realistic floor (+0.0318), past
the 0.7414 parity bar, **0.0008 under dense 0.7536** — with zero post-warmup emission and a
bounded, *declining* δ-ratio (1.37→1.03). Three facts now bound everything downstream:

1. **GATE-B1 is closed on valid M** (med m1 0.0121 ≪ 0.10 bar; paired-frac 0.57 ≪ 0.80). The
   blend/convex-combination class is dead even on a generator-consistent gradient — validity
   does *not* open the blend geometry. This retires a whole operator family, not one dose.
2. **F1 weight-space geometry**: at identical (batch, θ), cos(δ, G_comp_ring) ≈ −0.92…−0.98
   with ‖δ‖/‖G_comp‖ ≈ 1.05 ⇒ algebraically ‖G_true‖ ≈ 0.33·‖G_comp‖ and
   cos(G_true, G_comp) ≈ 0. **The codec error is the dominant component of the fast
   gradient, not a perturbation.** Residual/EF-style operators have ~all the headroom; blend
   has ~none. This is the weight-space confirmation of EXP-26's 0.318 activation-proxy.
3. **m6 ≈ 0.62 cross-fire persistence + m7 stable-rank ≈ 2**: the valid carrier is *not*
   memoryless even at β_anc=0 (carrier-law risk is partly intrinsic), and the replay gradient
   is rank-~2 in an ambient 1536 — **the codec's rank-77 capacity is not the constraint;
   act-basis MISMATCH is** (F3). Every stability claim in the verdict is 50-step **CENSORED**
   (EXP-27 ignited at ~61).

So the open frontier is no longer "does any correction help" (B2 answers yes) but **(a) is
B2's win real past the ignition window, (b) which *adjacent* operators are now cheap to settle
given the map, and (c) does the win survive an honest savings accounting.** The ranking below
is by **decision-value-per-GPU-hr**: how much of the remaining GOAL surface a cell closes per
unit compute, counting closures (a clean negative that retires a family) as full value.

---

## 1. The ext100 decision tree (the binding next measurement — already running)

The 100-step B2 extension (`exp30_B2_ext100`, identical settings, ledger EXP-30-EXT, box
i_40697545) is **the gate on everything else in this program.** Until it returns, the 0.7528
PASS is a censored point estimate and no successor should spend compute, because every
candidate inherits B2's substrate and B2's m6 ≈ 0.62 carrier. The four pre-registrable outcomes
and what each *forces* (not suggests):

| ext100 outcome | what it means | immediate follow-up | rank impact |
|---|---|---|---|
| **(i) emission-free AND val ≥ ~0.74 @75/100** | B2 is a *real* converging comm-eff trainer, not first-passage luck; the K-delayed telescoping EF is GOAL-1+GOAL-2 on the realistic substrate (modulo GOAL-3 honest bytes) | go straight to **R1 (honest-bytes cell)** + **R5 (λ-robustness, ONE point)**; B2 becomes the launcher-promotion candidate | this is the **win branch** — the program pivots from "find a converter" to "harden + measure + reproduce the converter" |
| **(ii) emission-free BUT val decays toward floor by 100** | the correction is stable but its *parity* was a 50-step transient (Adam eats the early δ benefit; the codec artifact re-accumulates as Q stops rotating) | diagnose decay source with **R3-diagnostic (Q-rotation telemetry, zero-GPU re-read of ext100 + one short probe)** before any new λ/operator; decay-by-Q-freeze ⇒ basis route (R3) jumps rank | parity is **not** banked; #28's plain@100 control becomes load-bearing as the drift reference |
| **(iii) ignites ~61 (matches EXP-27)** | the m6 ≈ 0.62 carrier IS the EXP-27 carrier; β_anc=0 + endogeneity (F4) did NOT buy stability, only a censored 50-step window | **STOP the δ-residual lineage at λ=1**; the carrier law convicts the *valid* carrier too; re-route to **R4 (small-β_anc is then carrier-blocked, see §4)** OR surface fixes (Cell C of #28) | the win evaporates; F4's "endogenous ⇒ no pump" hypothesis is falsified; this is the **highest-information negative** |
| **(iv) ignites LATER than 61 (e.g. ~75–90)** | endogeneity *delays* but does not *prevent* (the EXP-27 "lag-not-dose" pattern, but with a longer lag) | same STOP as (iii) for λ=1, but the lag-vs-dose curve now has a valid-carrier point — feeds the carrier-law model; do NOT chase with λ-capping (EXP-27 proved dose-capping delays not prevents) | δ-residual retired at this dose; the *mechanism* question (why endogenous still pumps) becomes the deliverable |

**Decision-grade reading:** outcomes (i) and (ii) keep the residual route alive and make R1
(honest bytes) the top priority; (iii)/(iv) kill the route at λ=1 and elevate the basis-mismatch
route (R3) and/or surface fixes. **Cost already sunk** (operator-authorized, running). **Do not
launch any §3 cell until ext100 returns** — it is the cheapest possible disambiguator and it is
free.

One caveat the team must hold: outcome (i)'s "val ≥ 0.74" is *not* a dense comparison past
step 50 — **no dense data exists past 50** (operator directive: dense never re-run). val@75/100
is a trajectory-shape and stability read, compared to B2's own 0.7528@50 and to #28's plain@100
drift reference, never to a dense@100 number that does not exist.

---

## 2. #28 redesign in light of B2 (is current-step EF still worth building?)

#28 builds **true current-step error feedback on the codec's own dropped residual**
(`u_t = x_t + e_t; e_{t+1} = u_t − C(u_t)`, carrier-free, telescoping, zero extra bytes) +
**plain@100** as the H_carrier/H_generic control. B2 is *additive evidence* for #28's mechanism
(K-delayed telescoping EF works in production), not a substitute. The redesign question:
**does B2's success change whether #28's current-step Cell A is worth ~20 GPU-hr?**

**Answer: the plain@100 control (Cell B) is now MORE valuable; Cell A (the current-step EF
build) is contingent on the ext100 outcome and on a sharpened inert-risk.**

- **Cell B (plain@100) — keep, raise priority.** It was already #28's "single cleanest
  discriminator" (H_carrier vs H_generic). After EXP-30 it acquires a *second* job: it is the
  **no-carrier drift reference for the B2 ext100 trajectory**. If ext100 decays (outcome ii) or
  ignites (iii/iv), plain@100 tells us whether the surface itself drifts/ignites at horizon
  *without any merger* — the only way to attribute B2's behavior to the correction vs the
  substrate. This is ~5 GPU-hr (config-only, no code) and **can never be wasted**. **It should
  run regardless of ext100** — ideally on the same box class, and it is the one #28 cell I would
  approve immediately.

- **Cell A (current-step codec EF) — worth building ONLY if a sharpened inert-risk clears, and
  the priority is ext100-conditional.** Two facts from EXP-30 cut against rushing it:
  1. **F3 + #28's own pre-registered inert-risk collide.** B2 works via a *K-delayed* residual
     that is refreshed at anchor fires; #28's Cell A works via the codec's *current-step*
     dropped residual, which on a projector codec is only transmitted **when Q rotates**
     (frozen Q ⇒ e ∈ range(I−P) forever ⇒ C(x+e)=C(x), the residual never crosses the link).
     EXP-25 measured the act-energy Q as near-converged (recon flat ~0.024). **m7 (rank-~2
     gradient, act-basis mismatch) is the strongest signal yet that Q barely rotates in the
     directions that matter** — which is exactly the regime where #28 Cell A degenerates to
     plain + a monotonically-growing e with no sawtooth (the pre-registered inert outcome).
  2. **B2 already demonstrated the telescoping mechanism converts**, so #28 Cell A's *novelty*
     is narrowed to: "does the *current-step, codec-internal* form (no anchor backward, truly
     zero-byte) also convert, or is it inert-by-Q-convergence?" That is a real and distinct
     question (B2 needs the anchor's full-rank replay backward; #28 Cell A needs nothing but
     the codec's own residual — strictly cheaper at run-time and more on-mission for GOAL-3),
     but it is **lower decision-value than ext100 and R1** because B2 already banked the
     mechanism-works result.

  **Redesign recommendation for #28 Cell A:** add a **cheap pre-gate** mirroring EXP-30's
  geometry-gate discipline — a ~1–2 GPU-hr Q-rotation probe (or a zero-GPU re-read of ext100's
  per-refresh Q-rotation telemetry, see R3) that measures `‖Q_new − Q_old‖_F` / top principal
  angle per refresh. **If Q is effectively frozen, Cell A is pre-registered inert and should NOT
  spend the 20 GPU-hr** — the result is decided by the probe ("EF inert by Q-convergence" → the
  lever is Q-refresh policy = R3, a different issue). If Q rotates materially, Cell A is a
  legitimate, cheaper-than-B2, more-on-mission converter test and should run. **This is the
  EXP-30 lesson applied to #28: gate the expensive build on a cheap geometry measurement.**

**Net:** #28 splits. Cell B (plain@100) → approve now, ~5 GPU-hr, dual-purpose control.
Cell A (current-step EF) → gate on a Q-rotation probe; build only if Q rotates; otherwise the
"inert" outcome is itself the answer and points at R3.

---

## 3. The basis-mismatch route (m7 says rank-2 gradient, act-basis misses it)

**This is the deepest finding and the highest-ceiling route — but it is the one with a live
landmine (EXP-26 Step C).** F3/m7: the valid replay gradient has stable rank ≈ 1.8–2.05 in
ambient 1536 and top-1% coordinate mass ≈ 0.60. The rank-77 codec has *abundant* capacity for a
rank-2 object. So the codec doesn't fail by under-capacity; it fails because **Q (built from
stale-weight forward activations, q_basis=act) does not contain the gradient's principal
directions** (F1's cos(G_true, G_comp) ≈ 0 is the same fact in weight space).

**The landmine, stated precisely so any proposal must clear it:** EXP-26 Step C **falsified
update-energy / hybrid Q** — building Q to capture *update* (gradient) energy instead of
activation energy **anti-converts** (because rollouts are uncompressed, the codec only sits on
the training-backward boundary; a Q tuned to gradient energy degrades the activation
reconstruction the forward pass depends on, and the net effect was negative). So the naive fix
"just make Q a gradient basis" is **already dead.** Any basis redesign MUST avoid that exact
corner.

**What is admissible vs what is the same falsified corner:**

- **A replay-gradient-sketch basis (build Q, or an *augmenting* sub-basis, from the EXP-29
  replay gradients) — is this a NEW corner or the same one?** It is a **genuinely new corner,
  but only if framed as an ADDITIVE off-principal sub-basis, not a replacement of the act-basis
  Q.** The reasoning:
  - EXP-26 Step C replaced/blended the *forward-activation* Q with an update-energy Q, which
    broke the forward reconstruction (the anti-convert). That is forbidden.
  - But B2's δ already proves the missing direction is **recoverable and helpful** when injected
    additively (δ cancels the codec artifact AND injects the true direction — F1). A
    **projection-split** operator that keeps the act-basis Q for the forward/recon path
    *unchanged* and routes a *small, separately-sketched* gradient sub-basis only into the
    correction term (the null space of the act-Q) is structurally the *projection-based split*
    the plan-30 §Pre-execution-framing explicitly listed as an unsampled point — and it does
    NOT touch the forward Q, so it does NOT re-enter the Step-C corner.
  - **However:** m7's rank-2 gradient with m1≈0 (gradient orthogonal to G_comp) plus m6≈0.62
    persistence means a sketched gradient sub-basis is itself a **persistent exogenous-ish
    direction** — the carrier law applies. A static gradient sub-basis injected every step is
    *exactly* the persistent fixed-direction force the carrier law convicts. So a replay-sketch
    basis is admissible **only in the K-delayed, fire-refreshed, β_anc=0 form** (the same
    discipline that let B2 survive 50 steps), and it is **subject to the same ext100 censoring
    caveat** — it cannot be cleared until B2's own ext100 settles whether that discipline holds
    at horizon.

- **Verdict on the basis route:** it is the **highest-ceiling, highest-risk** route. It is
  admissible **iff** (a) it augments rather than replaces the act-basis Q (avoids Step C), (b)
  it is delivered K-delayed/fire-refreshed/β_anc=0 (respects the carrier law), and (c) ext100
  first establishes that this delivery discipline survives horizon. **It is NOT the next cell to
  run** — it is gated behind ext100 (the discipline must hold) and behind R3-diagnostic (we must
  first know whether Q is frozen, because if Q already rotates enough, the simpler #28-CellA /
  B2 forms suffice and the sketch basis is unnecessary complexity). Rank it as a **research-issue
  to scope now, run later.**

---

## 4. Small-β_anc EMA per m6 ≈ 0.62 — admissible or carrier-law-blocked?

**Largely carrier-law-blocked. m6 ≈ 0.62 is bad news for this route, and the verdict already
flags it (F2).** The reasoning chain:

- The carrier law (#27 post-mortem): ignition needs a persistent fixed-direction force,
  `‖Σ_t e_t‖ ~ λ·T·‖G‖`, which requires the carrier's **autocorrelation time ≫ cadence**.
- The plan-30 §Pre-execution-framing made m6 *the explicit safety measurement* for the
  small-β_anc EMA variant: **m6 LOW ⇒ a short-effective-memory EMA plausibly stays below the
  ignition scale and is a legitimate follow-on; m6 HIGH ⇒ persistence risk is intrinsic to the
  valid anchor signal itself, even β_anc=0 carriers inherit it.**
- m6 came back ≈ **0.62** (real cross-pair fires 3–8: 0.59–0.75). That is **not low.** The valid
  anchor signal *already* carries autocorrelation 0.62 across one cadence interval at β_anc=0.
  An EMA with β_anc > 0 would *compound* that on top of an already-persistent carrier — pushing
  the effective autocorrelation time up, straight into the regime the carrier law convicts.
- **Therefore:** a small-β_anc EMA is **NOT cleared by this run** and should **not** be the next
  cell. The pre-registered safety measurement returned the unsafe value. Proposing it now would
  be re-learning the EXP-27 lesson (persistence ignites; dose-capping only delays).

- **The one narrow opening (and it is contingent, not a recommendation to run):** if ext100
  returns outcome (i) (B2 emission-free AND val holds to 100), then the β_anc=0 δ-residual is
  empirically shown to survive horizon *despite* its own 0.62 carrier — which would be evidence
  that the F4 endogeneity argument (δ cancels the circuit's own artifact, so it doesn't pump
  length the way an *exogenous* carrier does) is real. **Only in that world** does a small-β_anc
  variant become re-arguable, and even then **only on the δ-residual (endogenous) object, never
  on an exogenous M-blend**, and **only with a fresh carrier-law analysis** that accounts for
  the 0.62 base persistence. Absent that, small-β_anc EMA is **blocked.** Rank: **do not run; it
  is the route most likely to burn a cell re-confirming a known negative.**

---

## 5. Honest-bytes accounting for GOAL-3 (the anchor-circuit traffic cell)

**This is underweighted in the current program and I rank it second only to ext100, because it
is the one cell that can turn a "parity" claim into a "parity AND savings" claim — which is half
of "done" — and it is cheap.** GOAL-3 requires inter-stage communication volume "measured and
materially lower than dense, reported as a concrete number." Every cell asserts
`comm/bytes_ratio ≈ 0.0505` (~19.8×) — **but that ratio counts only the fast compressed
boundary traffic.** It does NOT count the anchor circuit's traffic, which the verdicts have
honestly flagged as a *caveat* but never *measured*:

- the **DP all-reduce of M** (full-coverage, all 196 matrices, every cadence=5 ticks),
- the **broadcast of Q** to every DP rank each refresh,
- for the **B2 / replay route specifically**: the replay machinery (CPU snapshots,
  the fire-aware ring) — mostly host-side and *not* inter-stage, but the M/Q traffic IS.

The standing caveat ("M/δ traffic is anchor-side, cadence-amortized") is an *argument* that it's
small; **GOAL-3 demands a number, not an argument.** B2's headline 0.7528-at-19.8× is
**provisional on this** — if the honest amortized ratio is, say, 8× instead of 19.8×, the result
is still a win but the headline number is wrong, and we'd be reporting a savings we don't have.

- **What the cell is:** a **measurement / accounting pass, not a training run.** It can largely
  be done **zero-GPU** from B2's existing telemetry + the codec/anchor code: count
  bytes-per-refresh for the M all-reduce and Q broadcast (sizes are known: 196 matrices, rank
  77, H=1536, cadence 5), amortize over the cadence interval, add to the fast-path bytes, and
  report a single honest inter-stage ratio. A short on-box instrumented run (~1–2 GPU-hr, byte
  counters on the anchor collectives) would *validate* the hand-count against measured wire
  bytes if the operator wants belt-and-suspenders.
- **Decision value:** it directly closes GOAL-3 for the winning configuration (B2 / ext100). It
  also **prices the basis route and #28 Cell A** — if a replay-sketch sub-basis adds another
  broadcast, this accounting tells us its byte cost up front, which is a GOAL-3 input to whether
  R3 is worth it. **It can never be wasted** and it has no length-explosion risk.
- **Why now:** the moment ext100 returns outcome (i) or (ii), the very next question an operator
  will ask is "what's the *real* savings number?" Having the accounting ready converts a parity
  result into a GOAL-1+2+3 result in one step.

---

## Ranked program (by decision-value-per-GPU-hr)

| rank | cell | GOAL crit | cost (GPU-hr) | decision value | risk | what existing evidence already answers |
|---|---|---|---|---|---|---|
| **0** | **ext100** (B2 → 100 steps) — *already running* | 1 (de-censor), 2 | 0 (sunk) | **gates the whole program**: real-converter vs censored-luck vs ignites | n/a (running) | nothing — this is THE open measurement; F2 m6≈0.62 says it's a real coin-flip |
| **1** | **R1 — honest-bytes accounting** (mostly zero-GPU; ≤2 GPU-hr to validate) | **3** | turns "parity" into "parity + measured savings"; prices every successor's byte cost | ~none (measurement, no training) | bytes_ratio 0.0505 counts ONLY fast path; M all-reduce + Q broadcast UNMEASURED — only a *caveat* today |
| **2** | **R2 — plain@100** (#28 Cell B; config-only) | 1 (attribution) | H_carrier/H_generic discriminator + the no-carrier drift reference for ext100 | low (plain emitted nothing in 50 steps; ~5 GPU-hr) | plain@50 = 0.6437 emission-free, but 50-step is CENSORED (the airtight version is @100) |
| **3** | **R3-diagnostic — Q-rotation telemetry** (zero-GPU re-read of ext100 + ≤1 GPU-hr probe) | gates 4,5 below | decides whether #28 Cell A is inert and whether the basis route is needed at all | ~none | EXP-25 recon flat ~0.024 (Q near-converged) + m7 rank-2 mismatch ⇒ STRONG prior Q is frozen; never directly measured per-refresh |
| **4** | **R4 — #28 Cell A** (current-step codec EF) — *iff R3 shows Q rotates AND ext100 ≠ ignite* | 1,2,**3** | tests the truly-zero-byte, anchor-free converter (most on-mission for GOAL-3) | medium (carrier-free by design, but censored until run); ~20 GPU-hr | B2 banked "telescoping EF converts"; novelty narrowed to current-step/codec-internal form; inert-risk live (F3+m7) |
| **5** | **R5 — basis-mismatch / replay-sketch sub-basis** (K-delayed, augmenting, β_anc=0) — *scope now, run after ext100+R3* | 1,2 (highest ceiling) | the only route that attacks the *root* (F1 cos≈0 / m7 mismatch) rather than correcting around it | **high** — must avoid EXP-26 Step C corner AND clear the carrier law; ~15–20 GPU-hr | EXP-26 Step C killed update-energy Q; admissible ONLY as additive off-principal sub-basis, K-delayed |
| **—** | ~~small-β_anc EMA~~ | — | **BLOCKED** | high (re-confirms known negative) | **m6 ≈ 0.62 returned UNSAFE**; the pre-registered safety measurement convicts it; carrier law |
| **—** | ~~blend / convex-combination, any dose~~ | — | **CLOSED** | — | GATE-B1 closed on valid M (med m1 0.0121, F1 cos(G_true,G_comp)≈0) — blend has ~zero headroom |
| **—** | ~~signed_ema on valid M~~ | — | **DEAD** | — | EXP-25/26: sign-replacement convicted structurally (coin-flip @ delay 0) independent of M validity |
| **—** | ~~update-energy / hybrid Q~~ | — | **DEAD** | — | EXP-26 Step C: anti-converts (rollouts uncompressed) — any basis fix must avoid this corner |
| **—** | ~~dense re-run~~ | — | **FORBIDDEN** | — | operator directive 2026-06-10 |

**Budget shape:** if ext100 wins (i), the cheap top of the ladder (R1 + R2 + R3-diagnostic) is
**~7–8 GPU-hr total** and closes GOAL-3 + attribution + the inert-risk diagnosis before any new
expensive build. R4/R5 are the only ~15–20 GPU-hr cells and both are **gated** behind that cheap
tier — exactly the EXP-30 discipline (cheap geometry/accounting gate before the expensive
training cell) carried forward. This keeps the next decision cycle inside one ~24 GPU-hr budget
even in the run-everything case, and as little as ~8 GPU-hr in the gate-it-out case.

---

## Next 3 issues to open (concrete, one-line hypothesis each)

1. **Issue: "B2 honest inter-stage byte accounting (GOAL-3 closure for the converging
   config)."** *Hypothesis:* the anchor circuit's amortized M-all-reduce + Q-broadcast traffic,
   added to the fast-path bytes, still yields a materially-lower-than-dense inter-stage ratio —
   reported as a single concrete number — and B2's parity therefore comes WITH measured savings,
   not just an asserted 19.8× that ignores the anchor traffic. *(Mostly zero-GPU; depends on
   ext100 only for which config to price. This is the cell that converts a parity result into a
   "done"-shaped result.)*

2. **Issue: "Q-rotation telemetry + #28 Cell A inert-gate."** *Hypothesis:* on the locked
   substrate the act-basis Q is effectively frozen in the gradient-relevant directions
   (`‖Q_new−Q_old‖_F` / top principal angle ≈ 0 per refresh, consistent with EXP-25 recon-flat
   and m7 rank-2 mismatch), which would mean #28's current-step codec EF is inert-by-Q-
   convergence and the real lever is Q-refresh policy — so #28 Cell A should be GATED on this
   measurement (run only if Q rotates), exactly as EXP-30 gated B1/B2 on the geometry probe.
   *(Re-read ext100 telemetry first, then ≤1 GPU-hr probe if needed. Also splits #28: approve
   plain@100 Cell B immediately as the dual-purpose no-carrier drift control.)*

3. **Issue: "K-delayed additive replay-gradient sub-basis (off-principal correction channel)."**
   *Hypothesis:* a small gradient sub-basis sketched from the EXP-29 replay gradients and routed
   ONLY into the K-delayed, fire-refreshed, β_anc=0 correction term — leaving the act-basis Q
   that owns the forward/recon path UNCHANGED — injects the rank-2 true direction that F1/m7 show
   the act-basis misses, converting B2's near-parity into a clear surpass, WITHOUT re-entering
   EXP-26 Step C (it augments rather than replaces Q) and WITHOUT a new exogenous carrier (it
   inherits B2's endogenous K-delayed discipline). *(Highest ceiling, highest risk; SCOPE now,
   RUN only after ext100 confirms the delivery discipline survives horizon and R3 shows the
   simpler forms don't already suffice.)*
