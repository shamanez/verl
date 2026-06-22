# Debate A — the diagonal trap & parity-vs-surpass (cross-challenge resolution)

**Referee:** cross-challenge referee (adversarial). **Date:** 2026-06-22.
**Participants (positions adjudicated):** off-policy-theorist (`01-off-policy-theory/notes.md`)
and algorithm-architect (`04-algorithm-design/notes.md`).
**Charter:** run the explicit theorist-vs-architect debate the brief requires on the
**diagonal trap** and **parity-vs-surpass**, stress-test the (largely aligned) positions
for residual cracks, and rule on each. I do **not** rubber-stamp agreement: where the two
notes *agree*, I treat the shared claim as the thing most in need of a hostile read,
because a shared blind spot survives precisely because neither author attacks it.

> **Grounding independently re-verified by the referee (not taken from the notes' self-citation):**
> - **GOAL.md** (the file, §"Async-realism constraint"): "the anchor is **always lagging,
>   never leads** … (⇒ **no delay-compensation / anchor-lead**)." The prohibition is real
>   project policy, not the architect's framing.
> - **EXP-38 verdict.md H3** (`research/runs/EXP-38/verdict.md` lines 34–37): forward `h`
>   top-1 singular direction holds **99.1%/98.6%** energy, **top-1 overlap = 1.0 at every
>   lag k=1…40**; but **top-77 overlap ≈ 0.77 (GSM8K) / 0.71 (Big-Math)**, flat; backward
>   `grad_h` rank-90 = **105 (GSM8K) / 180 (Big-Math)**, task-dependent and growing.
> - **Basis Rotation** (Jung, Shin, Lee, ICML 2026, arXiv:2602.03515; `02-async-delayed-lit/notes.md`
>   §1B/§4, **[fetched]**): rotates the optimizer into the **Hessian eigenbasis** (off-diagonal,
>   globally-shared ⇒ cross-rank-identical) to restore Adam adaptivity under staleness;
>   **stationary objective + depth-linear (structured) delay**; the note explicitly flags
>   "re-derive for variable delay + non-stationary RL" — the convergence certificate does
>   **not** transfer.
> - **The seed** (`reports/anchor-future-projection/theory-and-literature-2026-06-22.md`
>   §3–4): the "rotation `EXP-38` measured *is* `H·Δθ`"; the diagonal-trap kill-check is
>   stated there exactly as the architect relays it, and it acts on the **gradient-field
>   geometry** (un-rotating the stale *gradient*), confirming it is a gap-(a) operation.
>
> Every grounding claim the two notes lean on checks out. The cracks below are therefore
> **not** citation errors — they are residual *reasoning* gaps in the aligned positions.

---

## T1 — Does the primary recommendation (vi) raise the staleness budget, or merely relocate the stale signal?

**Claim at issue (architect, 04 §3 Primary).** Recasting the anchor into an activation-space
slow calibrator **dissolves** the budget question rather than fighting it, because the
transported object — the activation basis `Q` (and, in the curvature variant, the Hessian
eigenbasis) — is **staleness-invariant** (EXP-38 H3: forward-`h` subspace overlap *flat*
across lag), whereas the gradient *mean/sign* is not (`ρ→0` by k≈10). The budget for the
forward codec is "effectively unbounded." Both notes endorse this as the robust primary.

**Strongest challenge (the skeptic's case — sharper than the architect's own self-doubt).**
The anchor still computes everything on stale weights `θ_{t−K}`. Renaming its output a
"Q/curvature calibrator" risks being a **shell game**: you have moved the staleness from
the *gradient* to the *statistic*, and then asserted (not proven) that the statistic is
the safe place to keep it. Three concrete cracks:

1. **The "overlap 1.0 flat" claim is true only at rank 1.** EXP-38 H3 (re-read by the
   referee) shows top-**1** overlap = 1.0 flat, but top-**77** overlap is **0.77/0.71** —
   far below 1.0, *flat but not full*. The note's prose slides between "top-1 overlap =
   1.0" (the headline) and "Q overlap flat" (the claim it actually uses), and these are
   not the same statement. A rank-1 codec inherits the 1.0 guarantee; a **higher-rank Q
   does not**. So "the budget question dissolves" is airtight *only* for a rank-1 forward
   codec — exactly the regime where the codec carries almost no information beyond the
   single dominant activation direction.

2. **The backward link is where the project actually lives, and it is NOT staleness-clean.**
   The forward `h` is rank-1; but H3's own "codec-decisive" finding is the **forward/backward
   asymmetry**: backward `grad_h` is rank **105/180**, *task-dependent and growing*. EXP-38
   measured subspace-overlap-flatness for the forward `h`; it did **not** establish the same
   flatness for the rank-105/180 backward `grad_h` subspace. The architect transports a
   *separate, higher-rank* backward codec (04 §3 "Concrete shape") but **assumes** it inherits
   the forward link's staleness-tolerance. That assumption is **unverified** — and it is the
   load-bearing one, because the backward codec is the high-rank object that actually shapes
   the gradient that updates the weights.

3. **"Demote to calibrator" can hide a re-injected gradient.** If the slow statistic `Q`
   the anchor maintains is then used to *reconstruct/decode* the compressed gradient, the
   decoded gradient is still a function of stale information. The claim "it never provides a
   step direction" (EMA-PG calibrator role) is clean *only* if `Q` enters as a **projection
   basis** (a subspace the live gradient is expressed in) and **not** as a value that is
   added to the step. The note asserts the former but the calibrator-sufficiency kill-test
   (04 §3, "decompose the merged update into Q-projection vs M-folding") is precisely the
   test that has **not yet been run** — it is listed as "runnable on EXP-38 now," i.e. *open*.

**Strongest rebuttal (steelman of the aligned position).** The theorist supplies the
decisive structural distinction the challenge needs to answer: the two-gap decomposition
(01 §1). The staleness error splits into the **slow-varying** parameter-point/curvature-geometry
part (gap a, `O(L_H·K·η·‖ū‖)`, bounded by weight drift, half-life ≈16 ticks) and the
**fast-varying** policy-mean part (gap b, `Δ_dist`, unbounded, half-life ≈1 step). The
*subspace/eigenbasis is a property of the gradient field's geometry* — it lives in the
**slow** part. EXP-38 measured this directly: weight half-drift ≈7.9 global steps but
behaviour-signal half-drift ≈1 step, and *the subspace overlap tracks the slow weight
clock, not the fast behaviour clock* (overlap flat across k=1…40 ≈ the weight half-drift
window). So transporting the **subspace** stale is safe **because the subspace is in the
gap-(a)/curvature-geometry regime**, whereas transporting the **gradient mean** stale is
unsafe **because the mean is in the gap-(b)/policy regime**. This is not a rename — it is a
principled claim about *which projection of the stale signal you keep*: you keep the part
that is slow (geometry) and discard the part that is fast (the mean/direction).

**Resolution.** *The budget is genuinely raised for the forward, rank-1, projection-basis
use of the anchor — and only there.* Precisely:

- **Safe to transport stale: a low-rank subspace / energetic direction / (arguably) the
  Hessian eigenbasis.** These are slow-varying geometric objects (gap-(a) regime), and
  EXP-38 H3 confirms the forward top-1 direction is *exactly* staleness-invariant. The
  budget question genuinely *dissolves* here. The theorist's gap-(a)/gap-(b) split is the
  rigorous reason, and it is the strongest argument either note makes.
- **NOT established safe to transport stale: the gradient mean/sign** (the architect already
  concedes this — it is the whole point of demoting it), **and — the crack the notes miss —
  the *high-rank* backward `grad_h` subspace and the *full* rank-77 forward subspace.**
  EXP-38 proves flatness for the top-1/forward direction and a *flat-but-only-0.77/0.71*
  overlap for top-77; it does **not** prove the rank-105/180 backward subspace is staleness-
  flat. The eigenbasis variant is in the same boat: an eigenbasis is a *higher-rank* geometric
  object than a rank-1 activation direction, and its staleness-flatness is **assumed by
  analogy, not measured**.

**Flagged residual gap (referee's addition to the audit trail).** The architect's
calibrator-sufficiency and backward-codec-sizing kill-tests (04 §3 (b)/(c)) are listed as
"runnable on EXP-38 now" but **not yet run**. Until the backward `grad_h` subspace overlap-vs-
lag curve is measured (the same H3 analysis the forward `h` got), the claim "the budget
question dissolves" should be stated as **"dissolves for the forward rank-1/few projection
basis; budget for the higher-rank backward codec and any eigenbasis transport is an OPEN
empirical question, expected-but-not-proven staleness-tolerant."** With that qualification,
(vi) **upholds** as the robust primary — it relocates the signal *into the slow part of the
error*, which is a real budget raise, not a shell game. Without that qualification the note
**overclaims** the backward/eigenbasis half.

**Verdict: UPHELD, qualified.** The transported *forward subspace* is genuinely in the
slow (gap-a/geometry) signal and the budget is genuinely raised; the *backward codec and
eigenbasis* staleness-tolerance is assumed-not-proven and must be demoted to an open
empirical claim. The forward `h` (rank-1) is provably safe to transport stale; the backward
`grad_h` (rank 105/180) is **not yet** shown to be.

---

## T2 — Does (vii) cross-rank-2nd-moment clear the diagonal trap?

**Claim at issue (architect, 04 §1(vii)/§3 Secondary).** A cross-rank-disagreement term
clears the diagonal trap **iff** it is added in **objective form** — a `λ·R(Var_r)` penalty
(or SAM-style ascent-descent) whose gradient `∇_θ Var_r` *changes the fixed point* — and
**fails** in **step form** (`g/√Var`), which is a per-coordinate diagonal preconditioner =
"a better Adam diagonal" = the trap. The theorist concurs (01 §5.3/§6.2 OQ5): the cross-rank
2nd moment is information outside `σ(M)` and is *not* reach-capped by `τ_dist`.

**Strongest challenge.** Is the objective-vs-step distinction actually a distinction, or a
restatement? A skeptic presses two ways:

1. **The fixed-point collapse argument.** At a stationary point of the penalized objective
   `J(θ) + λ·R(Var_r(θ))`, the gradient condition is `∇J + λ·∇R(Var_r) = 0`. If `R` is
   smooth and `Var_r` is approximately separable across coordinates near the optimum (a
   diagonal Hessian of the penalty), then a *Newton/preconditioned* solver for this fixed
   point uses `diag(∇²[J+λR])` — and the penalty's contribution to that diagonal is a
   **per-coordinate variance-derived rescale**. A skeptic claims: *near the fixed point, the
   objective-form's effect on the step is again diagonal*, so the "different fixed point"
   is reached by what looks like a better diagonal. If true, the distinction evaporates
   asymptotically.

2. **The "is `∇_θ Var_r` even computable from what we have" challenge.** `Var_r` is the
   variance across **concurrent same-`θ` rank gradients** (data-variance, NOT cross-θ
   staleness). To get `∇_θ Var_r` you need the derivative of the per-rank gradient w.r.t.
   `θ` — a **second-order** object across the ensemble. The architect waves at "all-reduced
   sufficient statistics," but does not exhibit the estimator. If the only tractable
   estimator is a finite-difference that reduces to `g/√Var`-like quantities, the step-form
   is what you actually get, and the objective-form is a theoretical fiction.

**Strongest rebuttal.** The challenge's point (1) **conflates the gradient with the
preconditioner**. The diagonal trap is specifically about the **preconditioner** `v_t`:
Adam's update is `g ⊘ √v_t`, a per-coordinate *rescale of a fixed-direction gradient*. The
objective-form does something categorically different: it **adds a new term to the gradient
itself**, `g_total = ∇J + λ∇R(Var_r)`. This changes the **direction** the optimizer
descends, not the per-coordinate scaling of a fixed direction. Crucially, `∇_θ Var_r` is
**not** of the form `(diagonal matrix)·∇J` — it is a *separate vector field* that does not
vanish where `∇J` vanishes (that is the literal meaning of "different fixed point"). A
diagonal preconditioner *cannot* produce a term that survives where `∇J=0`; only a genuine
addition to the gradient can. So the objective-form is **outside** anything Adam's diagonal
`v_t` can express, *by construction*, regardless of whether some auxiliary solver's diagonal
happens to look variance-derived. Point (1)'s "Newton-on-the-penalty" picture is a red
herring: the project's control is **first-order Adam**, not Newton; the surpass test is
"does the *first-order* update direction carry information `σ(M)`/diagonal-`v_t` lacks," and
`∇R(Var_r)` plainly does.

On challenge (2): the theorist's own σ(M) memo (01 §5.2, the surpass-routes recipe) names
the admissible recipe — "same-θ CONCURRENT gradients, objective-level, async-resolved from
all-reduced sufficient statistics." `Var_r` *is* an all-reduced statistic (mean and mean-of-
squares of per-rank gradients are two all-reduces). Its gradient `∇_θ Var_r = 2·𝔼_r[(g^{(r)}
− ḡ)·∂_θ g^{(r)}]` requires a Hessian-vector product per rank — *expensive but well-defined*,
and the SAM literature shows the ascent-descent surrogate (perturb `θ` toward the
disagreement-increasing direction, then descend) approximates it **without** an explicit
Hessian. So the estimator exists; the architect's failure to *exhibit* it is a completeness
gap, not a soundness gap.

**Resolution.** *The objective-vs-step distinction is sound, and the architect's ruling is
correct.* The clean reason: **the diagonal trap is a statement about preconditioners
(rescaling a fixed direction); the objective-form is a statement about the gradient
(adding a new direction).** A per-coordinate diagonal *cannot* produce a force that
persists where `∇J=0`; `∇_θ Var_r` *does* (different fixed point). Therefore the objective-
form genuinely injects information outside `σ(M)` and outside Adam's diagonal `v_t`, while
the step-form `g/√Var` is exactly the trap (it rescales the *existing* direction per
coordinate and vanishes where `∇J=0`).

**The cleanest test that separates them** (referee's operationalization, sharper than the
notes': they propose only a temporal-dispersion Spearman proxy, which is a *weak stand-in*
and cannot tell objective from step). The discriminating test is a **fixed-point probe**,
not a cosine-lift:

> Train two short variants from a common checkpoint with a **cheap multi-rank capture**
> (per-rank `g^{(r)}` at the same `θ_t`): (A) step-form `g ⊘ √(v_t + Var_r)`; (B)
> objective-form `g + λ∇R(Var_r)` (SAM surrogate). **At a point where the dense gradient
> `∇J ≈ 0`** (a converged or near-converged minibatch), measure the update norm. The
> step-form's update **→ 0** (it rescales a vanishing gradient); the objective-form's
> update **stays bounded away from 0** (the variance term is non-zero there). A non-zero
> update at `∇J≈0` is the *signature of a different fixed point* and is the thing a
> diagonal preconditioner provably cannot produce. This is the discriminator; cosine-lift
> on EXP-38 cannot see it (EXP-38 is n=1, no cross-rank ensemble, 04 §1(vii) — confirmed).

**Verdict: UPHELD.** The objective-form clears the diagonal trap; the step-form is the trap.
The distinction is sound on the preconditioner-vs-gradient grounds above. **Caveat carried
forward:** the architect did not *exhibit* the estimator (completeness gap, 04 §4.3), and
the clean kill-test is **not** laptop-only (needs a multi-rank capture — confirmed n=1
limitation). The fixed-point probe above is the correct discriminator and should replace
the temporal-dispersion proxy as the gating test.

---

## T3 — Is (iv) learned extrapolation truly gap-(b)-capped, even in its best (Basis-Rotation off-diagonal) form?

**Claim at issue (theorist, 01 §3.3 corollary + §5.2; architect concurs, 04 §3 Secondary).**
The reach bound: a curvature/extrapolation correction `R_K ≈ H·Δθ` repairs **only gap (a)**;
gap (b) `Δ_dist` **dominates** the measured decorrelation (`τ_dist ≈ 3–4` ticks GSM8K, ≈0
Big-Math). So even a *perfect off-diagonal un-rotation* cannot restore `ρ(K)` past `τ_dist`.
Therefore (iv) is gap-(b)-capped — small reach on GSM8K, ≈0 on Big-Math — and (vii) is
preferred for surpass.

**Strongest challenge (the architect's Basis-Rotation card, played hard).** Basis Rotation
(arXiv:2602.03515, **[fetched]**) does **not** re-aim the stale gradient mean. It acts on the
**optimizer geometry** — it rotates Adam's coordinate frame into the Hessian eigenbasis so
the diagonal `v_t` becomes curvature-aligned. The theorist's reach bound is derived for an
operator that *un-rotates the stale gradient* (`g(θ_{t−K}) ↦ g(θ_t)`), i.e. an operator
acting on the **gradient**. A skeptic asks: *does the reach bound even apply to an operator
that acts on the **preconditioner geometry** rather than on the gradient?* If Basis-Rotation
improves the *conditioning* of every step (including the live compressed step), it could
help on a gap-(b)-dominated task **not** by repairing the stale gradient but by making the
optimizer better-conditioned in general — a benefit orthogonal to the gap-(a)/gap-(b) split.

**Strongest rebuttal (steelman of the reach bound).** The rebuttal must concede the
challenge's premise and then contain it. Two moves:

1. **Concede: Basis-Rotation's *conditioning* benefit is real and is NOT what the reach
   bound caps.** The reach bound caps *un-rotating a stale gradient back to the live one*
   (restoring `ρ(K)→1`). Basis-Rotation's eigenbasis-conditioning is a **different** benefit:
   it improves the *live* update's curvature-adaptivity. The theorist's bound is silent on
   that — correctly, because that benefit is not "repairing staleness," it is "better
   preconditioning," which is available **with or without** an anchor.

2. **Contain: the conditioning benefit is a PARITY benefit on this control, by the diagonal
   trap's own logic — UNLESS the eigenbasis injects off-diagonal curvature dense-Adam lacks.**
   Here the two notes' frameworks **collide**, and the collision is the real finding. The
   diagonal trap says: the control is dense-**Adam** (diagonal `v_t`). If Basis-Rotation is
   applied **to the dense control too** (the honest comparison), then dense-Adam-in-eigenbasis
   is the new control, and the comm-eff method must beat *that*. If Basis-Rotation is applied
   **only to the comm-eff arm**, then the comm-eff arm is using off-diagonal curvature
   (information outside `σ(M)` and outside the diagonal `v_t`) that the control does not have
   — *that* is a genuine surpass lever, **but it is surpass via curvature-conditioning, not
   via staleness-repair**, and it is therefore **still subject to the diagonal trap's
   resolution**, not the reach bound's. The reach bound and the diagonal trap are answering
   **different questions**: reach bound = "how much of the *stale-gradient decorrelation* can
   curvature undo?" (answer: only the gap-(a) fraction, ≈0 on Big-Math); diagonal trap = "can
   off-diagonal curvature beat dense-Adam at all?" (answer: yes, if genuinely off-diagonal
   and not given to the control).

**Resolution.** *The reach bound stands for the operation it actually bounds, and Basis-
Rotation does not escape it — but the two notes mis-state the relationship by treating
"Basis-Rotation off-diagonal form of (iv)" as a single thing.* Decompose:

- **(iv) as staleness-repair** (un-rotate `g(θ_{t−K}) ↦ g(θ_t)` to restore `ρ`): **gap-(b)-
  capped, confirmed.** A perfect off-diagonal `Ĥ·Δθ` removes only `e^{param}` (gap a). On
  Big-Math `τ_dist≈0` (`ρ(k1)=0.018`), so the reach is ≈0. The theorist's corollary is
  correct and Basis-Rotation, *used to un-rotate the stale gradient*, does not change it —
  un-rotating recovers the stale gradient's *parameter-point* shift, never its *distribution*
  shift. **No curvature route can beat the reach bound on a gap-(b)-dominated task**, because
  the gap-(b) error is a property of the *data measure*, and no operator on the *weight-space
  gradient field* (which is what `H` is) can touch the measure. This is airtight: `H = ∇²_θ`
  *with the data held fixed* (01 A2) — it is *definitionally blind to* `Δ_dist`.

- **Basis-Rotation as curvature-conditioning** (rotate the *live* optimizer frame): **NOT a
  staleness-repair lever at all, so the reach bound does not apply to it** — but it is then a
  **diagonal-trap question**, and the answer is: it is **parity** if the control also gets it,
  and only **surpass** if (a) it is withheld from the control (a defensible comm-eff-specific
  advantage *only if* the eigenbasis is a free by-product of the anchor that dense training
  would not compute) **and** (b) it is genuinely off-diagonal (which Basis-Rotation is, by
  construction). Even then, **§T3's surpass is a `K`-independent conditioning win, not a
  budget raise** — it does not let the anchor be *more stale*; it makes the geometry better
  at any staleness.

**The decisive ruling for the operator:** *No curvature route — including Basis-Rotation —
can beat the reach bound on a gap-(b)-dominated task* (Big-Math, `τ≈0`), because curvature is
definitionally a fixed-data (gap-a) object and gap (b) is a moved-measure object. Basis-
Rotation's value, where it has any, is as a **general preconditioner-conditioning** move
(diagonal-trap-governed, `K`-independent), **not** as a staleness-budget raiser. The two
notes are right that (iv)-as-extrapolation is capped; they are imprecise in implying Basis-
Rotation rescues (iv) — it changes the *subject* (conditioning, not repair), and on the
conditioning subject it is diagonal-trap-governed parity unless withheld from the control.

**Verdict: UPHELD with a sharpened distinction.** (iv)-as-staleness-repair is gap-(b)-capped;
**no** curvature route escapes that on a gap-(b)-dominated task (definitional: `H` holds data
fixed). Basis-Rotation does **not** overturn the cap — it is a different (conditioning) lever,
diagonal-trap-governed and `K`-independent, and a surpass only if off-diagonal **and** withheld
from the control. The architect's preference for (vii) over (iv) for surpass **upholds**, and
on the *strengthened* ground that the reach bound is not merely empirical-on-the-measured-tasks
but **definitional** for the gap-(b) component.

---

## T4 — Admissibility ruling: trajectory-continuation vs anchor-lead

**Claim at issue.** GOAL.md (verified) forbids "delay-compensation / anchor-lead"; the
2026-06-22 discussion + both notes argue **trajectory-continuation ≠ lead** (continuing the
anchor's own observed `(θ, g)` history is like momentum; DANA-style forecasting of the
*swarm's future state*, arXiv:1907.11612, is the canonical inadmissible form). The theorist
explicitly defers: "the admissibility ruling is a project-policy decision, not a theorem"
(01 §5.4). Both notes agree this gates **only** the capped route (iv), **not** (vi) or (vii).

**Strongest challenge (defending the prohibition).** "Trajectory-continuation" is a
euphemism. Operationally, `R_K: g(θ_{t−K}) ↦ ĝ(θ_t)` produces an **estimate of the gradient
at a weight point the anchor has not reached** — `θ_t` is *ahead* of the anchor's `θ_{t−K}`.
Whether you call the supervision "the anchor's own past trajectory" or "the swarm's future,"
the *output* is a gradient aimed at a **more-advanced** policy than the one the anchor
evaluated. In the real async target (one slow anchor, fast swarm), `θ_t` **is** the swarm's
state, which the slow anchor has *not* reached — so "continue the anchor's trajectory to
`θ_t`" is **operationally identical** to "forecast the swarm's current state." The
distinction collapses in the actual deployment. The prohibition exists precisely to stop the
anchor from *pretending to know where the fast circuit is*, and extrapolation does exactly
that.

**Strongest rebuttal (defending continuation).** The distinction is **what supervises the
operator and what variable it extrapolates in**, and it does not collapse:

1. **DANA forecasts in *time/iterate-count*** — it predicts `θ` at a future *step index*
   using a momentum/velocity model of the iterate sequence, i.e. it *assumes a trajectory
   the swarm has not produced*. Trajectory-continuation here extrapolates in the **measured
   `Δθ`** that *already happened*: at correction time the swarm **has** moved to `θ_t`, and
   `Δθ = θ_t − θ_{t−K}` is an **observed, all-reduced quantity** (the live circuit knows its
   own current weights; the anchor's `θ_{t−K}` is known). `H·Δθ` is a first-order correction
   evaluated at a **known** displacement — it is *interpolation to a known present*, not
   *forecast of an unknown future*.

2. **The "lead" prohibition targets *temporal* leading** — the anchor producing a signal for
   a state *later in wall-clock than any state it has seen*. `R_K` with observed `Δθ` produces
   a signal for the **current** `θ_t` (the present), using the anchor's stale gradient plus a
   correction for the *already-realized* weight motion. It does not race ahead of wall-clock;
   it catches up to the present. That is the opposite of leading.

**Resolution (reasoned recommendation; operator makes the final call).** The cleanest
decision rule is a **two-part admissibility test**, and it cuts the euphemism precisely:

> **A curvature/extrapolation correction is async-admissible iff:**
> **(R1) Observed-displacement-only.** It extrapolates *only* in quantities that are
> **already realized and all-reduced** at correction time — concretely, `Δθ = θ_t − θ_{t−K}`
> where `θ_t` is the live circuit's **current** (not predicted) weights. It must **not**
> contain any model of `θ` at a *future* tick `t+m`, nor any velocity/momentum forecast of
> where the swarm *will* go. (This is the line between continuation and DANA.)
> **(R2) Cross-rank-identical + variable-staleness-tolerant.** `Ĥ` (or the eigenbasis) is fit
> from **all-reduced sufficient statistics** (identical on every rank) and across a **range**
> of realized lags, not a hardwired τ. A per-rank-fit or fixed-τ operator is inadmissible
> regardless of R1.

Under this rule: **trajectory-continuation that corrects to the *observed present* `θ_t`
using the realized `Δθ` is admissible** (it satisfies R1 — `θ_t` is observed, not forecast);
**any operator that predicts a future `θ_{t+m}` or uses a velocity model to guess where the
swarm is heading is inadmissible** (it violates R1 — it is DANA-class lead). The challenge's
"the distinction collapses in deployment" is **answered by R1's insistence that `θ_t` be the
*observed* current weights**: in the real async target the slow anchor does *not* need to
forecast the swarm — at correction time the swarm broadcasts (or the all-reduce exposes) its
current `θ_t`, so `Δθ` is observed, not predicted. *If* an implementation cannot observe
`θ_t` and must *guess* it, then and only then does extrapolation become inadmissible lead —
and R1 catches exactly that case.

**Scope reminder (both notes correct):** this ruling gates **only** route (iv), which is
**reach-capped anyway** (T3). It does **not** gate the primary (vi) or secondary (vii) —
neither forecasts anything (vi transports a slow statistic; vii uses concurrent same-θ
disagreement). So even an operator ruling that (iv) is *inadmissible* leaves the primary and
secondary recommendations **untouched**. The admissibility question is therefore **not on the
critical path** — it only decides whether the (capped, low-value) curvature route is allowed,
and I recommend it be permitted *only* under R1+R2 with a documented GOAL.md carve-out, **and
only after** (iv)'s offline cosine-lift falsifier passes (it is predicted to fail on Big-Math
and barely pass on GSM8K — so the admissibility carve-out may be moot).

**Verdict: ADMISSIBLE under R1+R2 (continuation to the *observed* present is not lead);
INADMISSIBLE if `θ_t` must be forecast.** Operator confirms. Gates only the capped route (iv).

---

## Implications for the report's recommendation

The debate **upholds the architect's layered recommendation** — primary **(vi)** activation-
space recast / anchor-as-Q-calibrator, secondary **(vii)** cross-rank disagreement-AS-
OBJECTIVE, with **(iv)** gradient-extrapolation demoted and gated — and **strengthens two of
its load-bearing claims** while **qualifying one**:

1. **(vi) upheld, qualified (T1).** The budget raise is real *and principled* — the
   transported subspace is in the **slow gap-(a)/geometry** part of the staleness error
   (theorist's two-gap split is the rigorous reason), not a rename of the stale gradient. But
   the "budget dissolves" claim is airtight **only for the forward rank-1 projection basis**.
   The note must demote the **backward codec (rank 105/180)** and the **eigenbasis variant**
   to *"expected-but-unproven staleness-tolerant"* — EXP-38 measured overlap-flatness only for
   the forward `h`, and the *full* rank-77 overlap is 0.77/0.71, **not** 1.0. **Action: run the
   already-listed backward-`grad_h` overlap-vs-lag and calibrator-sufficiency kill-tests before
   stating the backward claim as fact.**

2. **(vii) over (iv) for surpass — upheld and strengthened (T2, T3).** The objective-vs-step
   distinction is **sound** (the diagonal trap governs *preconditioners*; the objective-form
   adds a *gradient* term that survives where `∇J=0` — a place no diagonal can reach). The
   reach-bound cap on (iv) is **stronger than the notes state**: it is **definitional** (H
   holds the data measure fixed, so it is *constitutively* blind to gap (b)), not merely
   empirical-on-the-measured-tasks — so **no** curvature route, **including Basis-Rotation**,
   beats it on a gap-(b)-dominated task. Basis-Rotation does not rescue (iv); it is a separate,
   `K`-independent *conditioning* lever that is diagonal-trap-governed parity unless off-diagonal
   **and** withheld from the dense control.

3. **One genuine hole flagged (not in either note).** Both notes lean on "EXP-38 H3: overlap
   1.0 flat" as the empirical keystone of the primary, but H3's 1.0 is **rank-1-only**; the
   high-rank objects the method actually transports (backward codec, eigenbasis) do **not**
   inherit it, and the relevant kill-tests are **listed-but-unrun**. This is the aligned
   positions' shared blind spot: they generalized a rank-1 invariance to higher-rank transports
   without measurement. It does not overturn the primary (the forward-rank-1 use is solid and
   sufficient for a parity-at-larger-K budget raise), but it **caps how much of (vi) is
   currently *proven*** versus *projected*.

4. **Kill-test correction (T2).** The notes' temporal-dispersion Spearman proxy for (vii) is a
   weak stand-in that **cannot distinguish objective-form from step-form**. The discriminating
   test is the **fixed-point probe** (non-zero update at `∇J≈0` ⇒ different fixed point ⇒ not a
   diagonal), which requires a cheap multi-rank capture — confirming the architect's own
   "not-fully-laptop-only" caveat for (vii).

**Net:** no recommendation is overturned. (vi) and (vii) survive a hostile read; (iv) is
correctly demoted on a *strengthened* (definitional, not just empirical) reach bound and a
clean admissibility rule (continuation-to-observed-present is not lead). The single
substantive correction is to the **proof status** of (vi)'s backward/eigenbasis half — proven
for forward rank-1, **open** for the high-rank transports — which the report should state
honestly rather than carry the notes' rank-1-to-all-rank overgeneralization.
