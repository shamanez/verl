# 04 — Algorithm design: using the stale anchor at large/variable K

**Author:** algorithm-architect (teammate 4 of 4).
**Date:** 2026-06-22.
**Charter:** synthesize the three theory/lit notes (01 off-policy theory, 02 async/optimization
lit, 03 multi-timescale lit) plus the on-file seeds into a concrete candidate set, recommend ONE,
and give each candidate a **GPU-free offline kill-test** runnable on the EXP-38 `(θ, g)` drift
tensors. Design only — no GPU, no `verl/` edits.

> **Standard.** Paper-appendix tone. Every empirical number is cited to a teammate note or a source
> artifact; nothing is invented. The three load-bearing inherited results, which I treat as fixed
> and do not relitigate:
>
> 1. **Aggregation obstruction** (01 §4.0): a per-sample IS ratio cannot be applied post-hoc to an
>    already-summed/all-reduced gradient `M`, and we have no fresh `π_{θ_t}` probabilities. ⇒ IS /
>    V-trace / Retrace are **inapplicable** to the anchor; the only post-hoc lever IS-family theory
>    permits is a **single scalar** dose `w̄·M` (01 (12)). Candidate (iii) is therefore a documented
>    dead-end, not a live option.
> 2. **Reach bound** (01 §3.3, §5.2): the staleness error splits `e_K = e^{param} + Δ_dist`; a
>    curvature/extrapolation correction `R_K ≈ H·Δθ` repairs ONLY the parameter-point gap
>    `e^{param}` (gap a), but the **distribution gap `Δ_dist` (gap b) dominates** the measured
>    decorrelation (`τ_dist ≈ 3–4` ticks GSM8K, `≈ 0` Big-Math; 01 §3.3, EXP-38 H1/H2). So even a
>    *perfect off-diagonal un-rotation* cannot restore the stale→live cosine past `τ_dist`. This
>    **caps the learned-extrapolation route** at the gap-(a) fraction and turns the offline cosine-
>    lift test into a falsifier.
> 3. **σ(M) ceiling + diagonal trap** (GOAL.md; 01 §5; seed §4): any deterministic `Φ(G_comp, M)`
>    is σ(M)-measurable ⇒ caps at parity; surpass needs information outside σ(M) (curvature /
>    conversion-positive exploration / cross-rank 2nd moment), and a curvature lift that collapses to
>    a per-coordinate **diagonal** is "a better Adam diagonal" = still parity. Off-diagonal is the
>    decisive surpass kill-check.

**The open question (restated).** Design a robust, theoretically-grounded way to use the anchor at
**large / variable K** without ignition (signed_ema sign-spiral) or drag (delayed_ef wrong-direction
pull) — i.e. **raise the usable staleness budget**. PRIMARY target = **PARITY-at-larger-K** (raise
the budget). SURPASS is a secondary, must-be-labelled aspiration, and per the reach bound it is
gap-(b)-capped on the measured tasks.

**The settled substrate (do NOT relitigate):** anchor-on-PowerSGD r=77, paired replay, anchor-owns-Q,
`signed_ema(α=0.25, β_anc=0.50)` as the research merger / `delayed_ef(λ=1, β_anc=0)` = B2 parity SOTA
at 5/5 (GOAL.md; SUMMARY.md). The EXP-31/#33 null levers (perturbation, δ-momentum, plain adaptive-λ,
control-variate gating, sub-basis, β-averaging) are σ(M)-measurable nulls and are **not** re-proposed
as surpass.

---

## §1 Candidate catalogue

For each candidate: **(1)** mechanism; **(2)** PARITY-at-larger-K vs SURPASS label (+ σ(M)-escape
category if surpass); **(3)** σ(M) verdict + diagonal-trap verdict; **(4)** async-admissibility
verdict; **(5)** a concrete GPU-free offline kill-test on the EXP-38 tensors with a pass/fail
threshold.

Notation for the kill-tests: the EXP-38 captures are `runs/EXP-38/{gsm8k,big-math}/` — 1071 fp32
tensors/arm + manifest, the `(θ, g)` (and boundary-`h`, `grad_h`) drift dataset, both datasets never
merged (EXP-38 verdict; seed §7). `g_t ≜ g(θ_t)` dense gradient at tick `t`; baselines (raw, no
correction): GSM8K `cos@k5 = 0.176`, `cos@k10 = 0.023`; Big-Math `cos@k1 = 0.018` (01 §3.3 table,
EXP-38 H1). Weight half-drift ≈ 7.9 global steps ≈ 16 ticks; behaviour-signal half-drift ≈ 1 global
step (EXP-38 H2). Forward `h` rank ≈ 1, top-1 subspace overlap = 1.0 flat across lag; backward
`grad_h` rank-90 = 105 (GSM8K) / 180 (Big-Math) (EXP-38 H3).

---

### (i) Staleness-scaled / age-decayed dose on the merger

**(1) Mechanism.** Down-weight the merged correction by the **realized** lag: apply `λ(K)·δ` (or
`w̄(K)·M`) where `λ` decreases with the observed staleness `K` (or with a measurable staleness proxy
such as the residual ratio `‖δ‖/‖g_m‖`). Two realizations already exist in the code: the held-
correction `μ^age` age-decay (`spectral_filter.py:702–779`, scales the held correction by `μ^age`,
`age = current_step − last_refresh_step`; staleness_theorist R-S1) and the ratio-gated adaptive-λ
(`_adaptive_lambda`, `spectral_filter.py:784–846`, with a low `lambda_cap`; R-S2). External
realizations: VCPO's **ESS-scaled LR** (a scalar staleness dose, 02 §1; arXiv:2602.17616), Zhang-2016
staleness-aware ASGD `η_eff = η/τ` (02 1A; arXiv:1511.05950).

**(2) Label: PARITY-at-larger-K.** It is the **only post-hoc lever the aggregation obstruction
permits** (01 §4.0: a constant scalar on the summed vector). It shrinks the wrong-direction vector;
it does not re-aim it.

**(3) σ(M): inside.** `λ(K)·M` is `M` times a measurable scalar ⇒ σ(M)-measurable (01 §5.1).
**Diagonal-trap: N/A (scalar).** A scalar is below even the diagonal — it is a single gain on the
whole vector. Cannot surpass by construction.

**(4) Async: fully admissible.** Lagging-reference-only ✓ (uses M as-is, no lead). Cross-rank-
identical ✓ if `K`/the proxy is computed from DP-identical quantities (`current_step` is DP-identical;
`‖δ‖/‖g_m‖` is post-all-reduce) — R-S1/R-S2 are explicitly cross-rank-identical (01 §5.4;
staleness_theorist). Variable-staleness-tolerant ✓ **by construction** (it keys on *realized* lag) —
this is its whole point and the reason two-timescale theory blesses it (03 §1: the rigorous license is
a vanishing **dose ratio** `b_n/a_n → 0`, i.e. a small/decaying dose, NOT a small staleness).

**(5) Offline kill-test.** This one is *primarily an on-GPU stability lever*, but it has a cheap
offline proxy. On the EXP-38 tensors, compute the **per-tick misaligned-energy fraction**
`m(k) = ‖M_⊥‖/‖M‖` where `M_⊥` is the component of the k-stale gradient orthogonal to `g_t`
(`m(k) = √(1 − ρ(k)²)` since norm-ratio ≈ 1). The dose that keeps the integrated tangential push (01
(10), carrier law) below the 5/5-stable level is `λ(K) ≲ λ_5 · m(5)/m(K)`. **Pass/fail:** confirm
`m(k)` is monotone increasing in `k` on both tasks (it must be, from the cosine decay) and that a
`λ(K) ∝ 1/m(K)` schedule reproduces the per-tick tangential dose of the *stable* 5/5 point at K=20.
If even `λ(20)` driving the dose to the 5/5 level requires `λ(20) → 0` (because `m(20)/m(5) → ∞` as
`ρ(20)→0`), that is the **honest negative result**: at K=20 the only safe dose is ~0, i.e. dosing
*degenerates to "turn the anchor off"* and cannot raise the budget while keeping the anchor useful.
GSM8K: `m(5)=√(1−0.176²)=0.984`, `m(20)=√(1−(−0.008)²)≈1.000` ⇒ ratio ≈ 1.016, so a *fixed* dose is
already near-flat in misaligned energy — meaning dose-decay barely changes the carrier at K=20
because the gradient is **already** almost all carrier. **This predicts (i) is a weak budget-raiser:**
it caps ignition by shrinking an already-orthogonal vector toward zero, recovering the no-merger floor,
not parity. (Consistent with EXP-31 δ-momentum/adaptive-λ nulls; SUMMARY.md.)

---

### (ii) Trust-region / alignment cap on the correction's contribution

**(1) Mechanism.** Instead of (or on top of) scaling, **project out / clip the wrong-direction
component** of the correction before merging. Concretely: given the live compressed gradient
`G_comp` and the anchor correction `c` (= `δ` for delayed_ef, or `|G|·sign(M)` for signed_ema), gate
the part of `c` that is anti-aligned with `G_comp` — e.g. `c' = c − max(0, −⟨c, ĝ⟩)·ĝ` (remove the
negative-projection component onto the unit live gradient `ĝ`), or hard-clip the merged step's
2nd-moment. External realizations: **GAC / Gradient Alignment Control** (02 1D, §4; arXiv:2603.01501)
— gradient-projection control that regulates progress along *stale-aligned* directions, with a
**convergence proof under bounded staleness**, and which independently reports our exact ignition
signature (stale gradients are abnormally cosine-aligned → overshoot); **M2PO** 2nd-moment IS-weight
constraint (02 1D; arXiv:2510.01161 — "prosperity before collapse" = our stable@5/ignite@20 budget,
cross-rank-computable); PPO-style clip on the *correction* (a trust region preventing drift, 01 §4.1).

**(2) Label: PARITY-at-larger-K.** A projection/clip is a defensive operation: it removes the
ignition-driving component but cannot inject information dense-Adam lacks.

**(3) σ(M): inside.** `P·M` with `P` formed from `(G_comp, M)` is a deterministic `Φ` ⇒ σ(M)-
measurable (01 §5.1: "a direction projection (GAC-style) is σ(M)-measurable"). **Diagonal-trap: N/A
for surpass** — it is a directional/2nd-moment clamp, not a curvature lift; it is correctly labelled
parity-only and never claims off-diagonal information.

**(4) Async: admissible.** Lagging-reference-only ✓. Cross-rank-identical ✓ **only if** the
projection direction and the clip threshold are formed from all-reduced quantities (the GAC
projection onto a *shared* direction, the M2PO 2nd-moment computed from DP-reduced sufficient
statistics) — a per-rank-fitted projection would break the shared-Q/shared-M invariant, so the design
must use the DP-mean `M` and DP-mean `G_comp` (both already all-reduced) to form `P`. Variable-
staleness-tolerant ✓ (GAC needs only a *bound* on staleness, not a known value; 02 §2).

**(5) Offline kill-test.** On the EXP-38 tensors, simulate the gate and measure whether it **removes
the carrier without removing the signal**. For each lag `k`, decompose the k-stale gradient
`M = M_∥ + M_⊥` relative to `g_t` (`M_∥` = projection onto `g_t`, `M_⊥` = orthogonal). The gate keeps
`M_∥` and discards (some of) `M_⊥`. **Metric:** the *retained useful fraction*
`R(k) = ‖M_∥‖/‖M‖ = |ρ(k)|`, and the *carrier suppression* `1 − ‖gated M_⊥‖/‖M_⊥‖`. **Pass/fail:**
the gate is worth running on GPU iff at the target K it retains a non-trivial aligned signal — propose
**`|ρ(K)| ≥ 0.10` at the target K** (i.e. there is *some* descent-aligned component to keep). GSM8K
`|ρ(20)| = 0.008`, `|ρ(10)| = 0.023` — **fails at K=20 and K=10** (essentially no aligned component to
retain; the gate degenerates to "discard the whole anchor"); `|ρ(5)| = 0.176` **passes at K=5**.
Big-Math `|ρ(1)| = 0.018` — **fails even at K=1**. **Prediction:** (ii) raises the budget to roughly
**K ≈ 5–8 on GSM8K** (where an aligned remnant exists to protect) and **does nothing on Big-Math**.
It is a *robustifier of the existing 5/5 regime*, not a large-K unlock. This is the cleanest honest
statement: alignment-capping can only preserve alignment that still exists.

---

### (iii) IS-reweighted anchor — DOCUMENTED DEAD-END (do not propose as live)

**(1) Mechanism (for the record).** Reweight the anchor by an importance ratio `w(τ) =
π_{θ_t}(τ)/π_{θ_{t−K}}(τ)` to make it unbiased for `∇J(θ_t)` (the textbook off-policy PG, 01 (7)).

**Why it is dead** (01 §4.0, the key result of the theory note): IS-family corrections form a
**per-sample** ratio and reweight the **summand before aggregation** (01 (11)). The anchor `M` is a
*single gradient vector* — the per-sample scores are already summed and DP-mean all-reduced (and
PowerSGD-compressed); the trajectories are gone. Reweighting is nonlinear in the summands
(01 (12): `Σ w_i x_i ≠ (mean w_i)·Σ x_i` unless `w_i` is constant), so **no per-sample ratio survives
aggregation** — the only post-hoc option is a single scalar, which is candidate (i), not an IS
correction. And we have **no fresh `π_{θ_t}` probabilities** by design (the comm-eff premise is that
the fast circuit moved on). The obstruction is *informational and upstream of* the IS variance
blow-up — we never even get to pay the variance.

**The only re-entry** (01 §4.3): keep the stale per-sample scores + logits and do **one extra
forward pass of `π_{θ_t}`** over the stale trajectories to form per-token `w_i` (the "missing old
logits" fix, Guan 2026, arXiv:2605.12070). This (a) is no longer a single transported gradient — it
is full per-sample off-policy GRPO; (b) re-incurs the IS variance `∝ exp(D_2(π_t‖π_{t−K}))` (01 (8)),
which at `K>τ` (`ρ→0`, `D_2` large) is exactly the clip-induced-bias regime; (c) **collapses the
comm-eff premise into ordinary async off-policy GRPO** (the VCPO/VESPO/GIPO research line, 02 1D) —
a different project, not a correction to *this* anchor.

**Verdict: not a live candidate.** No σ(M)/async/kill-test rows — it is removed from contention by
the aggregation obstruction. Recorded here so the report can state crisply *why* the entire IS/V-
trace/Retrace toolbox is off the table.

---

### (iv) Learned extrapolation `R_K ≈ H·Δθ` (un-rotate the stale gradient)

**(1) Mechanism.** The parameter-point gap is exactly the Hessian-vector product to first order:
`g(θ_t) = g(θ_{t−K}) + H(θ_{t−K})·Δθ + O(‖Δθ‖²)`, so the measured "rotation" of the stale gradient
**is** `H·Δθ` (01 §1.1; seed §3; EXP-38 norm-ratio ≈ 1.0 ⇒ rotation not rescale). Build an operator
`R_K` that adds the rotation back, in three rungs of increasing power: **(a) fixed-linear** — DC-ASGD
diagonal Taylor `g + λ·diag(g⊙g)·Δθ` (02 1A; arXiv:1609.08326) or the Nesterov-async fixed `(1−γ)`
look-ahead (02 1B; the project seed, arXiv:2505.01099); **(b) learned** — fit `R_K` on the anchor's
own `(θ, g)` trajectory, evaluated on held-out lags (seed §7); **(c) learned + error-feedback** — use
the sparse anchor refreshes as delayed ground truth `g_true`, regress the residual `r = g_true − R̂_K g`,
and accumulate `r` as classic EF so prediction bias does not accumulate (seed §5; Karimireddy 2019 EF,
arXiv:1901.09847; structurally identical to B2 `delayed_ef`). **Beyond-diagonal requirement** (seed
§4): a diagonal `Ĥ` can only rescale per-coordinate — it cannot rotate — so it is σ(v_t)-adjacent to
Adam and caps at parity; the rotation EXP-38 measured (norm-ratio ≈ 1) is precisely the off-diagonal
signature, so the operator must be genuinely off-diagonal. **Concrete beyond-diagonal + cross-rank-
identical realization: Basis Rotation** (02 1B, §4; arXiv:2602.03515) — rotate the optimizer into the
local **Hessian eigenbasis** so Adam's coordinate-wise `v_t` becomes curvature-aligned; the eigenbasis
is a globally-shared quantity ⇒ naturally cross-rank-identical, and it restored Adam's adaptivity under
pipeline staleness (same loss in 76.8% fewer iters, 1B LLM).

**(2) Label: PARITY-at-larger-K (rungs a/b); SURPASS-aspirational (rung c + off-diagonal), σ(M)-escape
category = curvature / 2nd-order (R5).** Fixed-linear and any σ(M)-measurable learned form are parity;
the *only* surpass-capable instance is an **off-diagonal** `Ĥ` (Basis-Rotation-class), because the
Hessian is information dense-Adam's diagonal `v_t` does not contain (01 §5.2; seed §4).

**(3) σ(M): a fixed-linear or past-dense-gradient-finite-difference `R_K` is σ(M)-measurable (01 §5.1,
the `M + β(g_{t−K} − g_{t−2K})` example) ⇒ parity. A learned `Ĥ` that estimates *curvature* (the
gradient's local derivative, not a function of the gradient means alone) is outside σ(M).**
**Diagonal-trap: this is THE candidate where the trap bites.** A learned `R_K` that collapses to a
per-coordinate gain is a better Adam diagonal ⇒ parity *regardless of how much cosine it recovers*.
Surpass requires the fitted operator's lift to come from **off-diagonal** (cross-coordinate) structure.

**(4) Async: CONDITIONAL — the contested candidate.** Cross-rank-identical ✓ **only if** `Ĥ` (or the
eigenbasis) is fit from **all-reduced sufficient statistics** so every rank computes the identical
operator (Basis-Rotation's eigenbasis is shared by construction; a per-rank-fitted `Ĥ` is inadmissible,
seed §6). Variable-staleness-tolerant ✓ **only if** fit across a *range* of lags, not one fixed τ (the
seed's fixed-τ `(1−γ)` is brittle; 02 §2 — "make its parameters functions of all-reduced slow-varying
statistics rather than a hardwired τ"). **Lagging-reference-only: DISPUTED.** Extrapolation *is* a form
of delay-compensation; GOAL.md line 62 forbids "delay-compensation / anchor-lead". The resolution
argument (seed §6; discussion-2026-06-22): weight-space extrapolation **continues the anchor's own
observed `(θ, g)` trajectory** — it does *not* forecast the swarm's future state — and trajectory-
continuation ≠ anchor-lead. **The off-policy math is agnostic** (01 §5.4: "the admissibility ruling is
a project-policy decision, not a theorem"); DANA's *future-θ forecasting* (02 1A; arXiv:1907.11612) is
the canonical **inadmissible** form, and continuation is distinct from it. ⇒ **flag as open-question
for the lead** (Debate A). This gates whether the only surpass-capable curvature lever is even on the
table.

**(5) Offline kill-test (THE central GPU-free gate, seed §7).** On the EXP-38 `(θ, g)` tensors, for
each rung fit `R_K` and measure the **stale→live cosine LIFT** on held-out lags:
- **Rung (a) fixed-linear:** apply DC-ASGD diagonal correction / fixed look-ahead; measure `ρ'(k)`.
- **Rung (b) learned (low-rank, off-diagonal-capable):** fit `R_K` on the ~50-dim active subspace
  (gradient rank-90 ≈ 50 GSM8K / 78 Big-Math, stable-rank ≈ 3 ⇒ tractable low-dim operator; EXP-38
  H3 / seed §1) on a train split of lags, evaluate `ρ'(k)` on a **held-out** lag split.
- **Rung (c) learned + EF:** add residual-EF using held-out refresh points as ground truth.

**Pass/fail (the falsifier):** propose **GSM8K `cos@k5: 0.176 → ≥ 0.40`** (seed §7). If no rung lifts
materially above baseline, the idea **dies on the laptop, no GPU** — and the reach bound (01 §3.3)
*predicts* it largely will, because gap (b) dominates: a perfect un-rotation removes only the small
slow `e^{param}` part and leaves the large fast `Δ_dist` part. **Diagonal-trap probe (mandatory):**
decompose the fitted `Ĥ` (or the per-coordinate gain achieving the lift) into **diagonal vs off-
diagonal energy** (e.g. compare the lift from a diagonal-only fit vs a full/low-rank-off-diagonal fit;
or measure the off-diagonal Frobenius mass of `Ĥ`). A purely-diagonal lift is **PARITY-only** and must
be labelled so *up front, regardless of cosine recovered*; only an off-diagonal lift is a surpass
candidate, and even then only up to the reach bound. **Big-Math:** run separately; `cos@k1 = 0.018`
(near-orthogonal in one tick) ⇒ predict the lift is ~0 (gap (b) is ~100% at k=1) and the candidate is
**infeasible on the hard task**. A method that lifts only GSM8K is still a scoped result.

---

### (v) Multi-timescale optimizer with explicit timescale separation

**(1) Mechanism.** Treat the anchor as the **slow** recursion in a two-timescale stochastic-
approximation (TTSA) coupling and enforce separation via a **step-size / dose RATIO** `b_n/a_n → 0`
(03 §1; Borkar 1997, Vidyasagar 2026 arXiv:2603.14481): the fast compressed circuit runs at rate
`a_n`, the anchor contributes at a vanishing relative rate `b_n`, so the biased stale contribution
`Σ (dose)·(bias)` stays **summable** and the fast circuit dominates. The literature-faithful reading
(03 §1, §4.1) is decisive: **the rigorous license is a vanishing DOSE, not a small staleness** — so
this is candidate (i)'s theoretical justification, sharpened to *decay the dose with K* (e.g.
`dose ∝ 1/K`, or gate off above a τ) rather than hope a fixed dose tolerates the lag.

**Does a pure lag break the SA premise?** **Yes, partially** (03 §1, §4.1, the load-bearing
distinction). Every *working* slow/fast stabilizer (Polyak/EMA, mean-teacher, Lookahead, SWA, SlowMo,
DDPG-soft-target) uses a **smoothed-current** slow signal — a low-pass of the *current* iterate at
**bounded distance** `O(ε)` from `θ_t`. Our anchor is a **pure lag** `θ_{t−K}` whose distance is
**unbounded in K**; the convex-combination contraction those proofs rely on does not apply. The *one*
rigorous license for a slow circuit (TTSA) buys convergence via the dose ratio — but it **assumes a
fixed equilibrium / martingale (zero-mean) noise**, and our anchor bias is **persistent, non-zero-
mean, and over a non-stationary GRPO objective** (03 §1 caveat; the bias is structural off-policy-ness,
not averageable noise). ⇒ TTSA gives **boundedness / stall-avoidance, not parity-beating signal** from
the stale gradient.

**(2) Label: PARITY-at-larger-K (at best; really "stall-avoidance").** Not surpass — separation only
controls how fast you converge to *some* fixed point assuming one exists; it cannot remove a non-
stationarity bias (03 §4.1 (iii)).

**(3) σ(M): inside** (a dosed `M` is σ(M)-measurable). **Diagonal-trap: N/A (scalar dose).**

**(4) Async: admissible** (it *is* the dose lever, candidate (i)'s admissibility) — and it adds the
specific prescription that the dose should **decay with realized K**, which is exactly the variable-
staleness-tolerant, cross-rank-identical form.

**(5) Offline kill-test.** Same misaligned-energy proxy as (i), plus a **summability check**: verify
that a `dose ∝ 1/K` schedule keeps the *cumulative* tangential displacement `Σ_t dose(K_t)·m(K_t)·‖M‖`
(carrier law, 01 (10)) **bounded** as the simulated lag distribution widens, whereas a fixed dose lets
it grow linearly. **Pass/fail:** the decaying-dose partial-sum stays `O(√T)`-like (cancelling) while
the fixed-dose partial-sum is `O(T)` (integrating). This is a *consistency check on the dose schedule*,
not a parity claim — it confirms the prescription "decay the dose with K" prevents ignition, which is
the boundedness TTSA promises. It cannot show a budget *raise* in the sense of recovering parity at
large K (the reach bound forbids that for any σ(M)-measurable lever).

---

### (vi) Activation-space recast / anchor-as-Q-(or-curvature)-calibrator

**(1) Mechanism.** **Stop transporting a stale gradient. Transport a slow, cross-rank-identical
*statistic* instead, so the off-policy gap `Δ_dist` never forms.** Two layers, both from EXP-38's
next-method recommendation (EXP-38 verdict §"Next-method recommendation"):
- **Forward link — low-rank activation codec with a frozen / slowly-refreshed Q.** Forward `h` is
  rank ≈ 1 (top-1 holds 99.1%/98.6% energy) and top-1 subspace overlap = 1.0 *flat across lag k=1…40*
  (EXP-38 H3) ⇒ **Q is intrinsically staleness-tolerant**; the anchor's job becomes maintaining a
  slowly-varying Q/codec basis, not providing a gradient. This is the cadence_analyst R-Q1 insight
  (decouple Q-refresh from delay_K; refreshing Q is nearly free — no stale backward — and Q is already
  cross-rank-identical via `sync_basis`+broadcast). **EMA-PG** (03 §3.1; arXiv:2602.04417) is the
  literature validating exactly this *demotion*: it replaces the GRPO anchor with an EMA used as a **KL-
  proximity reference (a calibrator, NOT a re-injected gradient)** and proves it stable — the smoothed-
  current + calibrator design choice, same algorithm (GRPO, ~1.5B), opposite to our lagged-gradient
  design, proven stable.
- **Backward link — a SEPARATE, higher-rank codec** sized per task (≥105 GSM8K / ≥180 Big-Math; EXP-38
  H3), since `grad_h` is *not* as compressible as `h` and grows with task hardness. Never symmetric
  forward/backward budgets.
- **(curvature-calibrator variant)** Transport the **Hessian eigenbasis** (Basis Rotation, 02 §4) as
  the slow shared statistic rather than the activation Q — a globally-shared, slowly-varying, cross-
  rank-identical curvature object that the fast Adam preconditioner rotates into. This is the bridge
  between (vi) and the off-diagonal-curvature surpass route (iv-c).

**(2) Label: PARITY-at-larger-K (the robust budget-raiser); the eigenbasis variant is SURPASS-
adjacent (curvature category) but transported as a *statistic*, dodging both the off-policy gap and
the aggregation obstruction.** The key structural win: a *statistic* (basis Q / eigenbasis / 2nd-
moment) is **staleness-tolerant in a way a gradient is not** — EXP-38 H3 shows the activation subspace
does not decorrelate with lag, so "large K" is not even a problem for it. The budget question
*dissolves* for the forward codec.

**(3) σ(M): the forward-Q recast is OUTSIDE the gradient-`σ(M)` framing entirely** — it does not fold a
stale gradient mean into the step at all; it changes *what is compressed*, so the σ(M) gradient-ceiling
analysis does not even apply to the forward link (the ceiling is about deterministic functions of
gradient *means*; a codec basis is a different object). **Diagonal-trap:** the activation-Q version
makes no curvature claim ⇒ parity, correctly labelled. The **eigenbasis variant** is the off-diagonal
curvature object (it *is* the beyond-diagonal information) ⇒ the only part that could surpass, and it
clears the diagonal trap by construction (an eigenbasis rotation is inherently off-diagonal; seed §4,
02 §4).

**(4) Async: the MOST admissible candidate.** Lagging-reference-only ✓ (the anchor is demoted to a
slow calibrator; it never provides a step direction — EMA-PG's calibrator role). Cross-rank-identical
✓✓ (Q is already all-reduced+broadcast and cross-rank-identical; the eigenbasis is globally shared by
construction). Variable-staleness-tolerant ✓✓ (EXP-38 H3: Q overlap is *flat* across lag — staleness
literally does not degrade it within k≤40). This is the candidate that satisfies the async north-star
**without tension** — no extrapolation/lead question arises because nothing is forecast.

**(5) Offline kill-test.** Largely **already passed by EXP-38**: forward `h` rank ≈ 1, top-1 overlap
= 1.0 flat across lag (H3) — directly confirms a frozen/slow Q is staleness-tolerant on *both* tasks.
The remaining offline checks: **(a) forward-codec headroom** — confirm rank-1/few captures ≥ the same
energy at the target K as at K=5 (it does — overlap flat); **(b) backward-codec sizing** — confirm
`grad_h` rank-90 (105 GSM8K / 180 Big-Math) on the held tensors to set the *separate* backward budget;
**(c) calibrator-only ablation proxy** — measure how much of the *step* the anchor currently
contributes as a *gradient* vs how much is recoverable from Q alone (decompose the merged update into
its Q-projection vs its M-folding component on the EXP-38 captures). **Pass/fail:** the forward recast
passes if Q-overlap at target K ≥ 0.95·(Q-overlap at K=5) — **already true** (flat). The eigenbasis-
curvature variant inherits candidate (iv)'s cosine-lift kill-test (does the eigenbasis-rotated stale
gradient lift `cos@k5 ≥ 0.40` via off-diagonal structure?).

---

### (vii) Cross-rank 2nd-moment / disagreement-as-objective (the σ(M)-escape independent of reach)

**(1) Mechanism.** Use the **disagreement between concurrent same-θ rank gradients** as a *signal*,
not the stale-vs-live drift. With per-rank gradients `{g^{(r)}}` at the *same* `θ_t` (data-variance,
NOT cross-θ staleness), the cross-rank 2nd moment `Var_r[g^{(r)}]` is information dense-Adam's diagonal
`v_t` does not contain. **Required form: variance-AS-OBJECTIVE** — add a term whose gradient is
`∇_θ Var_r` (a SAM-style ascent-descent on disagreement, or a `λ·R(Var_r)` penalty) so the **fixed
point itself changes**; NOT a `g/√Var` *step* (which reduces to a per-coordinate diagonal preconditioner
= the diagonal trap, "a better diagonal than Adam's"; surpass-routes memo via 01 §5.3, GOAL.md). The
disagreement is resolved via **all-reduced sufficient statistics** (cross-rank-identical, slow-varying,
staleness-tolerant) — so it is the surpass route the theorist flags as **not depending on gap-(a)
reach** (01 §5.2, §6.2 OQ5; EXP-38 next-method #6): it injects a *different fixed point*, not an un-
rotation, so it is **not capped by `τ_dist`** and may dominate on gap-(b)-dominated tasks (Big-Math).
External anchors: M2PO's cross-rank-computable 2nd-moment trust (02 1D), VCPO's variance-control framing
(02 §1).

**(2) Label: SURPASS, σ(M)-escape category = cross-rank 2nd moment (disagreement-as-objective).** This
is the one genuinely-surpass candidate that the reach bound does **not** cap (01 §5.2: "may be the more
robust surpass bet on gap-(b)-dominated tasks").

**(3) σ(M): OUTSIDE** — `Var_r` over concurrent rank gradients is not a function of the gradient *means*
`(g_t, g_{t−K})`; it is a second-moment across the DP ensemble, information σ(M) does not contain (01
§5.2; GOAL.md surpass categories). **Diagonal-trap: PASSES iff variance-AS-OBJECTIVE.** The decisive
kill-check (shared with the theorist, Debate B): the **step-form** `g/√Var` collapses to a diagonal and
**fails** (≡ a better Adam diagonal); the **objective-form** (`∇_θ Var_r` term, SAM-style) has a
different fixed point and **passes**. The architect's ruling: only the objective-form is admissible as
surpass; the step-form is the diagonal trap in disguise.

**(4) Async: admissible IF resolved from all-reduced statistics.** Lagging-reference-only ✓ (the
correction from all-reduced 2nd-moment sufficient statistics is slow-varying and staleness-tolerant —
01 surpass-routes recipe: "same-θ CONCURRENT gradients [data-var, NOT cross-θ drift], objective-level,
async-resolved from all-reduced sufficient statistics ⇒ cross-rank-identical"). Cross-rank-identical ✓
(by construction — it is *built from* the cross-rank reduction). Variable-staleness-tolerant ✓ (the
sufficient statistics are slow-varying). **Caveat:** this is the *least* developed in-project and needs
the objective-form spelled out; it is a research bet, not a drop-in.

**(5) Offline kill-test.** The EXP-38 captures are **n=1 per task** (single rank trajectory) — they do
**not** contain a cross-rank ensemble, so the *direct* disagreement-predicts-error test is **not fully
runnable** on EXP-38 alone (flagged limitation). The runnable proxy: use the **temporal** gradient
ensemble as a stand-in for the cross-rank ensemble and test whether **gradient-dispersion predicts
descent error** — i.e. does the variance of `{g_{t−j}}` over a short window correlate with `‖g_t −
mean‖` (does high local dispersion flag where the mean step is least trustworthy)? **Pass/fail:**
Spearman `ρ(dispersion, error) ≥ 0.3` on a held window would support that a disagreement signal carries
exploitable information; near-zero would weaken (but not kill — the cross-θ temporal proxy is a weak
stand-in for same-θ cross-rank, so a null here is inconclusive). **A clean test requires a cheap multi-
rank capture** (a short instrumented run logging per-rank `g^{(r)}` at the same `θ_t`) — this is the
one candidate whose kill-test is not fully laptop-only, which the recommendation accounts for.

---

## §2 Comparison table

| # | Candidate | (2) Parity vs Surpass | (3a) σ(M) | (3b) Diagonal trap | (4) Async-admissible | (5) Offline kill-test runnable on EXP-38? | Predicted offline outcome |
|---|---|---|---|---|---|---|---|
| (i) | Staleness-scaled / age-decayed dose | **Parity** (only post-hoc lever IS-obstruction permits) | inside | N/A (scalar) | ✓✓ full | yes (misaligned-energy proxy) | weak: caps ignition → no-merger floor, not parity (GSM8K `m(20)/m(5)≈1.0`) |
| (ii) | Trust-region / alignment cap (GAC/M2PO) | **Parity** | inside | N/A (clamp, parity-labelled) | ✓ (if `P` from all-reduced) | yes (retained-aligned-fraction `|ρ(K)|`) | budget → K≈5–8 GSM8K; nothing Big-Math (`|ρ(1)|=0.018`) |
| (iii) | IS-reweighted anchor | **DEAD-END** | — | — | — | — | removed by aggregation obstruction (01 §4.0); re-entry = ordinary async off-policy GRPO |
| (iv) | Learned extrapolation `R_K≈H·Δθ` | Parity (a/b); **Surpass** (c + off-diag) = **curvature** | inside (fixed/finite-diff); **outside** (learned curvature) | **THE trap bites** — off-diag required | **CONDITIONAL** (lead-vs-continuation = Debate A; cross-rank-identical + range-of-lags required) | **yes — THE central gate** (cosine-lift + diagonal-vs-off-diagonal decomp) | reach bound predicts mostly capped; GSM8K maybe, Big-Math ~0 (`cos@k1=0.018`) |
| (v) | Multi-timescale (dose-ratio `b_n/a_n→0`) | **Parity** / stall-avoidance | inside | N/A (scalar dose) | ✓✓ (= dose, decay with K) | yes (summability of carrier partial-sum) | confirms "decay dose with K" prevents ignition; not a parity raise |
| (vi) | Activation-space recast / anchor-as-Q-(or-curvature)-calibrator | **Parity** (robust); eigenbasis variant **Surpass**-adjacent (curvature) | **outside the gradient-σ(M) framing** (changes what's compressed) | N/A forward-Q; eigenbasis = off-diag (passes) | **✓✓ most admissible — no lead tension** | **largely already passed by EXP-38 H3** (Q overlap flat across lag) | forward recast confirmed staleness-tolerant on BOTH tasks; budget question dissolves |
| (vii) | Cross-rank 2nd-moment (disagreement-as-objective) | **Surpass** = cross-rank 2nd moment (**not reach-capped**) | **outside** | **passes IFF variance-AS-OBJECTIVE** (step-form = trap) | ✓ (if resolved from all-reduced stats) | **partial** — EXP-38 is n=1, no cross-rank ensemble; needs a cheap multi-rank capture | temporal-dispersion proxy only; clean test needs a short multi-rank run |

---

## §3 Recommendation

### The recommendation, in one sentence

**Raise the budget by recasting the anchor into an activation-space slow calibrator (candidate vi) —
the robust, async-clean, theory-and-EXP-38-supported budget-raiser — and IF surpass is wanted, the
only credible non-reach-capped bet is cross-rank disagreement-as-objective (candidate vii), with the
explicit caveat that the gradient-extrapolation route (iv) is gap-(b)-capped and should be gated on
its cheap offline cosine-lift falsifier before any GPU spend.**

This is a **layered** recommendation; the layering is justified below.

### Primary (robust budget-raiser): (vi) activation-space recast + anchor-as-calibrator

**Why this is the most defensible recommendation.** The open question is "use the anchor at large/
variable K without ignition/drag." Candidate (vi) **dissolves** the question rather than fighting it:

1. **It removes the off-policy gap at its source.** Every other gradient-space candidate inherits
   `Δ_dist` (the dominant, unbounded gap b; 01 §3.3) and can at best *manage* it. (vi) transports a
   *statistic* (the activation basis Q / a curvature eigenbasis), and EXP-38 H3 proves that statistic
   is **staleness-tolerant on both tasks** (forward `h` rank ≈ 1, top-1 subspace overlap = 1.0 *flat
   for k = 1…40*). The thing that decorrelates in ≈ 3 ticks (the gradient direction) is no longer what
   we transport; the thing we transport (the subspace) does not decorrelate within the measured window.
   **The staleness budget for the forward codec is effectively unbounded** — this is the only candidate
   for which "large K" is provably a non-problem, and it holds on Big-Math too (overlap 0.71 flat),
   where every gradient-reuse route is infeasible.
2. **It is async-clean with no open tension.** The anchor is demoted to a slow, cross-rank-identical
   Q/codec calibrator that *never provides a step direction* — exactly the EMA-PG calibrator role (03
   §3.1, proven stable in GRPO at 1.5B) — so the lead-vs-continuation admissibility question (which
   gates candidate iv) **never arises**. Q is already all-reduced + broadcast (cross-rank-identical)
   and EXP-38-flat (variable-staleness-tolerant). It satisfies the async north-star by construction.
3. **It is the convergent recommendation of the prior work.** EXP-38's own next-method recommendation
   is: compress in *activation* space, split forward/backward codec budgets, demote the anchor to a slow
   Q/codec calibrator (not a gradient provider), set K & backward rank per task. (vi) *is* that
   recommendation, now cross-checked against the theory (01: the gradient route is obstructed +
   reach-capped) and the multi-timescale lit (03: the literature's working slow circuits are either
   smoothed-current or calibrators, never re-injected lagged gradients).

**Concrete shape of (vi).** Forward link: low-rank (rank-1/few) activation codec with a slowly-
refreshed Q on its **own fast cadence**, decoupled from `delay_K` (cadence_analyst R-Q1: nearly free,
no stale backward). Backward link: a **separate, higher-rank** codec sized per task (≥105 / ≥180;
EXP-38 H3). Anchor: maintains Q (and optionally a KL-proximity calibration, EMA-PG-style), does **not**
fold a gradient into the step. K and backward rank set **per task** (EXP-38 #5).

**Falsifiable offline kill-test for the primary.** (Mostly discharged by EXP-38; the residual checks:)
- **Forward-codec staleness-tolerance:** Q-overlap at the target K ≥ `0.95 × (Q-overlap at K=5)`.
  **Status: PASS** on both tasks — overlap is flat across k = 1…40 (EXP-38 H3). This is the headline:
  the budget-raiser's core claim is *already empirically true*.
- **Backward-codec sizing:** `grad_h` rank-90 on the held tensors = 105 (GSM8K) / 180 (Big-Math); set
  the separate backward rank ≥ these. **Status: measured** (EXP-38 H3).
- **Calibrator-sufficiency:** decompose the current merged update on the EXP-38 captures into its Q-
  projection component vs its M-folding component; the recast is justified if the Q-projection carries
  the descent signal and the M-folding is the carrier-injecting part. **Threshold:** the M-folding
  component's alignment with `g_t` is ≤ the stale-gradient cosine (≈ 0.18 at k5, → 0 at k≥10), i.e.
  the gradient-folding is mostly carrier and safe to drop. **Runnable on EXP-38 now.**

**Parity-vs-surpass call for the primary: PARITY-at-larger-K (robust).** (vi) raises the budget; it
does not by itself surpass dense (the forward-Q recast injects no information dense-Adam lacks). That
is the correct, honest target — the PRIMARY objective per the charter.

### Secondary (surpass-aspirational, only if surpass is the goal): (vii) cross-rank disagreement-as-objective

**Why (vii) over (iv) for surpass.** Both are σ(M)-escapes, but they differ on the reach bound — the
single most important theoretical fact for the surpass question:
- **(iv) learned extrapolation is gap-(b)-capped.** Curvature `R_K ≈ H·Δθ` repairs only `e^{param}`
  (gap a); `Δ_dist` (gap b) dominates the decorrelation (01 §3.3 corollary), so even a perfect off-
  diagonal un-rotation cannot restore `ρ(K)` past `τ_dist` (≈ 3–4 ticks GSM8K, ≈ 0 Big-Math). Its
  surpass-reach is the gap-(a) *fraction* of `e_K`, which is small on GSM8K and ≈ 0 on Big-Math. It
  also carries the unresolved lead-vs-continuation admissibility question (Debate A).
- **(vii) disagreement-as-objective is NOT reach-capped.** It injects a *different fixed point* (a
  variance-AS-OBJECTIVE term whose gradient is `∇_θ Var_r`), not an un-rotation of the stale gradient,
  so the `τ_dist` ceiling does not bind it (01 §5.2, §6.2 OQ5). On gap-(b)-dominated tasks (Big-Math,
  where (iv) is hopeless) it is the *only* surpass route with a live mechanism.

**The surpass kill-check (Debate B, shared with the theorist):** (vii) passes the diagonal trap **only
in its variance-AS-OBJECTIVE form**. A `g/√Var` *step* collapses to a per-coordinate diagonal = a
better Adam diagonal = parity (the diagonal trap). The architect's ruling: **only the objective-form
(SAM-style ascent-descent on cross-rank disagreement, or a `λ·R(Var_r)` penalty) is admissible as
surpass**; the step-form is the trap in disguise. This is the central point to settle with the
theorist.

**Falsifiable kill-test for the secondary (honest about runnability):** the clean test —
*does cross-rank disagreement at the same `θ_t` predict descent error?* — needs a cross-rank ensemble
that **EXP-38 (n=1) does not contain**. The laptop proxy is the temporal-dispersion stand-in
(Spearman `ρ(dispersion, error) ≥ 0.3` on a held window); a null there is **inconclusive** (weak
stand-in). ⇒ the secondary requires a **cheap, short multi-rank capture** (log per-rank `g^{(r)}` at
the same `θ_t` for a few steps) before committing — this is the one place the recommendation is not
fully GPU-free, and the recommendation states it plainly.

**If a single offline gate must be named for the surpass aspiration:** run candidate (iv)'s cosine-
lift falsifier first (it *is* fully laptop-only), because it is the cheapest way to *bound* the gap-(a)
reach and thereby quantify how much (iv) could ever buy: **GSM8K `cos@k5: 0.176 → ≥ 0.40` with an
off-diagonal lift.** If (iv) fails this (the reach bound predicts it likely will, especially on Big-
Math), that result simultaneously (a) kills (iv) and (b) *strengthens* the case for (vii) as the only
non-reach-capped surpass route — a productive falsification either way.

### Async-admissibility resolution (incl. the trajectory-continuation-vs-anchor-lead tension)

- **Primary (vi):** **no tension.** The anchor is a slow calibrator that never forecasts or leads; Q
  is cross-rank-identical and EXP-38-flat across lag. Fully admissible under GOAL.md as written.
- **Secondary (vii):** admissible **iff** the disagreement correction is resolved from **all-reduced
  sufficient statistics** (cross-rank-identical, slow-varying, staleness-tolerant) — which is its
  native form. No lead.
- **(iv), if pursued:** the lead-vs-continuation question is **a project-policy ruling for the lead,
  not a theorem** (01 §5.4). My architect's position for Debate A: **trajectory-continuation of the
  anchor's own observed `(θ, g)` history is admissible** (it continues a known past, like momentum;
  it is *not* DANA-style forecasting of the swarm's future state, which is the canonical inadmissible
  form, 02 1A). But because (iv) is reach-capped *anyway*, this ruling is **not on the critical path**
  for the primary recommendation — it only matters if the lead wants to pursue the (capped) curvature
  route, and even then GOAL.md line 62 may need an explicit carve-out for "trajectory-continuation,
  cross-rank-identical, variable-staleness-tolerant." I recommend the lead resolve it, but it does
  **not** gate (vi) or (vii).

### Honest bottom line

> **Raise the budget via (vi) — robust, async-clean, and its core claim (forward-codec staleness-
> tolerance) is already empirically true in EXP-38.** This is the defensible deliverable for the
> PRIMARY target. IF surpass is wanted, the only credible bet is (vii) cross-rank disagreement-AS-
> OBJECTIVE, because (iv) gradient-extrapolation is **gap-(b)-capped** (prediction: its offline cosine-
> lift fails or barely passes on GSM8K and is ≈ 0 on Big-Math) — and the surpass route must clear the
> diagonal trap via the *objective* form, never the `g/√Var` step form. The dose/cap/multi-timescale
> levers (i)/(ii)/(v) are **parity-only robustifiers of the existing 5/5 regime** and are worth wiring
> as guardrails, but they do not raise the budget at large K because at large K the stale gradient is
> already almost pure carrier (GSM8K `cos@k20 ≈ 0`).

---

## §4 Provenance table + open questions

### §4.1 Provenance (claim / candidate → source)

| # | Claim / candidate | Source |
|---|---|---|
| 1 | Aggregation obstruction kills IS/V-trace/Retrace for the anchor; only a scalar dose survives | 01 §4.0 (12); candidate (iii) |
| 2 | Reach bound: `R_K≈H·Δθ` repairs only gap (a); gap (b) `Δ_dist` dominates decorrelation (`τ_dist≈3–4` GSM8K, ≈0 Big-Math) | 01 §3.3 §5.2; EXP-38 H1/H2 |
| 3 | σ(M) ceiling: deterministic `Φ(G_comp,M)` capped at parity; surpass needs curvature / conversion-positive exploration / cross-rank 2nd moment | GOAL.md; 01 §5; seed §4 |
| 4 | Diagonal trap: diagonal `Ĥ` / `g/√Var` step = better-Adam-diagonal = parity; off-diagonal / objective-form needed to surpass | seed §4; 01 §5.3; candidates (iv),(vii) |
| 5 | EXP-38 cosine table (GSM8K 0.507/0.176/0.023/−0.008; Big-Math 0.018/0.011/0.004); norm-ratio ≈ 1 (rotation) | 01 §3.3; EXP-38 H1; seed §1 |
| 6 | Weight half-drift ≈ 7.9 global steps; behaviour-signal half-drift ≈ 1 step ⇒ gap (b) dominates | EXP-38 H2; 01 §3.3 |
| 7 | Forward `h` rank ≈ 1, top-1 subspace overlap = 1.0 flat across lag k=1…40 (BOTH tasks) ⇒ Q staleness-tolerant | EXP-38 H3; candidate (vi) |
| 8 | Backward `grad_h` rank-90 = 105 (GSM8K) / 180 (Big-Math) ⇒ separate, higher, per-task backward codec | EXP-38 H3; candidate (vi) |
| 9 | Next-method recommendation: activation-space, split fwd/bwd codec, demote anchor to slow Q-calibrator, K & bwd-rank per task, surpass via cross-rank-2nd-moment or curvature not stale-gradient | EXP-38 verdict §Next-method; candidate (vi),(vii) |
| 10 | `μ^age` age-decay + ratio-gated adaptive-λ exist in code, cross-rank-identical (dose levers) | staleness_theorist R-S1/R-S2 (`spectral_filter.py:702–846`); 01 §5.4; candidate (i) |
| 11 | Decouple Q-refresh from delay_K (R-Q1); Q-refresh nearly free (no stale backward); Q already cross-rank-identical | cadence_analyst R-Q1 (`powersgd_activation.py:343–359`, `transformer_impl.py:1474, 2081–2082`); candidate (vi) |
| 12 | TTSA license is a vanishing dose-RATIO `b_n/a_n→0`, NOT a small staleness; pure lag breaks the smoothed-current premise; TTSA assumes fixed equilibrium + martingale noise (our bias is structural) | 03 §1, §4.1; Borkar 1997, Vidyasagar 2026 (arXiv:2603.14481); candidate (v) |
| 13 | EMA-PG: replace GRPO anchor with EMA as KL-proximity CALIBRATOR (not gradient), proven stable, same algorithm (GRPO ~1.5B) — validates the demotion | 03 §3.1; arXiv:2602.04417; candidate (vi) |
| 14 | Basis Rotation: rotate optimizer into Hessian EIGENBASIS (off-diagonal, globally-shared ⇒ cross-rank-identical) restores Adam adaptivity under staleness | 02 §4, 1B; arXiv:2602.03515; candidates (iv),(vi) |
| 15 | GAC: gradient-projection control, convergence proof under bounded staleness, reports our exact stale-aligned-overshoot ignition signature | 02 1D §4; arXiv:2603.01501; candidate (ii) |
| 16 | M2PO: 2nd-moment IS trust constraint = staleness budget ("prosperity before collapse" = stable@5/ignite@20), cross-rank-computable | 02 1D; arXiv:2510.01161; candidates (ii),(vii) |
| 17 | VCPO: ESS-scaled LR = scalar staleness dose; TIS-spike-then-collapse demo = our ignition | 02 §1, 1D; arXiv:2602.17616; candidates (i),(vii) |
| 18 | DC-ASGD `H·Δθ` = the gradient-space un-rotation form, but its Hessian approx is DIAGONAL (diagonal trap); stationary proof | 02 1A; seed §2; arXiv:1609.08326; candidate (iv) |
| 19 | Nesterov-async seed: fixed-linear `(1−γ)` look-ahead, FIXED-τ (brittle to variable delay), σ(M)-measurable (parity) | 02 1B; seed §2; arXiv:2505.01099; candidate (iv) |
| 20 | DANA future-θ forecasting = the canonical INADMISSIBLE anchor-lead form (distinct from trajectory-continuation) | 02 1A; arXiv:1907.11612; candidate (iv) admissibility / Debate A |
| 21 | Error feedback (Karimireddy 2019) makes a biased compressor converge; B2 `delayed_ef` is EF on codec residual; EF-on-prediction-residual is the same trick | seed §5; 02 1C; arXiv:1901.09847; candidate (iv-c) |
| 22 | Offline cosine-lift kill-test + diagonal-vs-off-diagonal decomposition; threshold GSM8K `cos@k5 0.176→≥0.40`; run tasks separately | seed §7; 01 §5.3 hand-off; candidate (iv) |
| 23 | Carrier law: persistent biased pull integrates `O(T)`, zero-mean noise cancels `O(√T)` | 01 §2.3 (10); candidates (i),(v) kill-tests |
| 24 | EXP-37: latency is the failure knob; 5/5 stable, 20/20 breaks BOTH mergers; no-merger floor 0.6300, dense band 0.75–0.78 | SUMMARY.md EXP-37; staleness_theorist §0 |
| 25 | EXP-38 captures n=1 per task ⇒ no cross-rank ensemble ⇒ (vii)'s clean kill-test needs a cheap multi-rank capture | EXP-38 verdict (n=1); candidate (vii) |

### §4.2 Open questions for the cross-challenge

1. **Debate A — trajectory-continuation vs anchor-lead (for the lead, gates candidate iv only).** Does
   GOAL.md line 62 ("no delay-compensation / anchor-lead") forbid trajectory-continuation extrapolation
   that continues the anchor's *own* observed `(θ, g)` history (as opposed to DANA-style forecasting of
   the swarm)? The off-policy math is agnostic (01 §5.4); my position is continuation is admissible and
   distinct from lead, but it may need an explicit GOAL.md carve-out. **Not on the critical path for the
   primary (vi) or secondary (vii) recommendation** — flagged because it gates the reach-capped
   curvature route (iv).
2. **Debate B — the diagonal trap, with the theorist (the decisive surpass kill-check).** I claim
   candidate (vii) surpasses **only in its variance-AS-OBJECTIVE form** and the `g/√Var` *step* form
   collapses to the diagonal trap; and candidate (iv) surpasses **only with an off-diagonal `Ĥ`**, a
   diagonal lift being parity *regardless of cosine recovered*. Does the theorist concur that (a) the
   objective-form genuinely yields a different fixed point (not reducible to a diagonal preconditioner),
   and (b) the cosine-lift kill-test's diagonal-vs-off-diagonal decomposition is the correct
   operationalization of "outside σ(M)"?
3. **Is the reach bound's prediction for (iv) confirmed offline?** The recommendation *predicts* the
   cosine-lift falsifier fails or barely passes on GSM8K (`cos@k5 0.176 → <0.40`) and is ≈ 0 on Big-Math
   (`cos@k1 = 0.018`). If a learned off-diagonal `R_K` *does* lift GSM8K `cos@k5 ≥ 0.40`, that would
   partially reopen (iv) as a scoped GSM8K-only budget-raiser — a falsifiable disagreement with the
   reach-bound reading. **Hand-off to whoever runs the offline gate.**
4. **Does (vii) need a multi-rank capture before any GPU commitment?** EXP-38 (n=1) cannot run the clean
   disagreement-predicts-error test. Is a short instrumented multi-rank capture (per-rank `g^{(r)}` at
   the same `θ_t`) in scope as the cheapest next diagnostic, or does the temporal-dispersion laptop
   proxy suffice to gate the bet? **Hand-off to lead / analyst.**
5. **Forward/backward codec split — engineering feasibility.** cadence_analyst R-Q2 flags that
   decoupling Q-refresh cadence from `delay_K` requires the replay ring to key retention on the
   M-refresh cadence, not the Q one (`AnchorReplayRing._keep_residue`, `anchor.py:426`). Is the
   separate-backward-codec + decoupled-Q-cadence shape of (vi) expressible without breaking the ring's
   fire-aware retention? **Hand-off to systems.**

### §4.3 Limitations of this note

- The offline-kill-test thresholds (`cos@k5 ≥ 0.40`; `|ρ(K)| ≥ 0.10`; Spearman `≥ 0.3`) are *proposed*
  operating points inherited from the seed (§7) or chosen here for falsifiability; they are not derived
  constants and should be sanity-checked by the analyst against the actual EXP-38 spectra.
- The misaligned-energy arithmetic in (i)/(v) uses `m(k) = √(1−ρ(k)²)` with norm-ratio ≈ 1 (EXP-38 H1);
  this is exact only to the extent the norm-ratio is exactly 1 (it is 0.89–1.01 across lags).
- Candidate (vii)'s kill-test is **not fully laptop-only** (EXP-38 is n=1, no cross-rank ensemble);
  this is stated in §1(vii), §3, and OQ4 — the one honest gap in the GPU-free guarantee.
- All literature claims are relayed from teammates 01/02/03 with their verification status (02 marks
  Basis-Rotation/GAC/VCPO/EMA-PG as **[fetched]**; some bibliography rows are search-verified or
  codex-reused — see 02 §5, 03 ledger). I did not independently re-fetch; flagged per their notes.
- I did not propose a fully-specified objective-form for (vii) (the `R(Var_r)` penalty / SAM ascent-
  descent is named, not derived) — it is a research bet requiring the theorist's sign-off (Debate B),
  not a drop-in.
