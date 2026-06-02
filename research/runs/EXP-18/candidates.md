# EXP-18 / M4 — Theoretical candidate enumeration (sequence step 0)

> **MANDATE deliverable (plan §"THE WORKING MODE", §Success criteria).** Written
> BEFORE any candidate run. Enumerates the complete set of continuous,
> STALE-anchor gradient corrections under test — each with (1) mechanism,
> (2) rationale tied to `g_mask = g_true + b + ξ`, (3) how it respects the three
> HARD CONSTRAINTS, (4) predicted ≤50-step curve-match behavior. At least one
> candidate (C2) is **derived from the current spectral correction** (turning its
> *reweighting* of `G_mask` into *additive injection* of the missing component).
> This file is appended with a **results-driven theory** after every iteration
> (observe → theorize → propose-new → test).

---

## 0. The theory we are reasoning from

**The estimator decomposition.** The masked actor gradient is a biased, noisy
estimator of the true (unmasked) GRPO gradient:

```
g_mask = g_true + b + ξ
```

- `g_true` — the true GRPO actor gradient at the current weights θ_t (what the
  dense run follows).
- `b` — a **systematic, curvature-aligned bias** (a delta-method term from masking
  the boundary activations and the RMSNorm `1/RMS` backward compounding over the 7
  pipeline boundaries `[3,7,11,15,18,21,24]`). `b` **accumulates** across a masking
  window and, left uncorrected, eventually flips the ascent projection — the masked
  run stalls at the ~0.13 reward floor (`runs/SUMMARY.md`, `findings/NEXT_RESEARCH.md`).
- `ξ` — zero-mean masking noise. Rescale (`h·mask/(1-p)`, ON permanent) unbiases the
  *activation* (`E[h̃]=h`) and bounds `Var(ξ)` (≈`p/(1-p)`≈9× at p=0.9). **Rescale does
  NOT remove `b`** — the gradient bias survives because it is a nonlinear/curvature term.

**The orthogonality finding (the load-bearing failure of the as-implemented method).**
EXP-16 `anchor@2 + spectral@2` (no clean step) → GSM8K val **0.080 ≈ random**, pearson
**~0.004**, inert. Root cause: at p=0.9, masking rotates `G_mask` **nearly orthogonal**
to `g_true` (cos≈0). The `SpectralFilter` (`spectral_filter.py`) computes

```
G_proj = α·G_mask + (1-α)·U diag(d) (Uᵀ G_mask V) diag(d) Vᵀ ,   d_i = s_i/(s_i+τ)
```

where `(U,S,V)` is the SVD basis of the stale-anchor EMA `M_anchor`. **`G_proj` is a
purely LINEAR function of `G_mask`.** `M_anchor` enters ONLY as projection *geometry*
(its singular vectors + scalar Tikhonov weights `d_i∈[0,1)`); the **vector** `M_anchor`
itself is never added. So when `G_mask ⊥ g_true`, the content `Uᵀ G_mask V` carries
~no energy in the directions that matter, and **no linear reweighting of a
near-orthogonal vector can synthesize the missing component.** This is exactly why the
clean step works (it *applies* the true gradient) while spectral did not (it only *used*
the true gradient as a basis).

**The anchor signal we have (`anchor.py`, `transformer_impl._maybe_comm_eff_anchor_refresh`).**
Every `cadence=5` steps the anchor runs ONE **unmasked** GRPO-actor-loss fwd/bwd on a
**deep-cloned module** loaded with `delay_K=5`-**stale** weights `θ_{t-5}`, over the
**current** batch's rollouts/advantages, taking **no optimizer step**. It yields a RAW
per-target 2D gradient `G_anchor ≈ g_true(current batch; θ_{t-5})` — the **stale true
gradient**. Today `G_anchor` flows ONLY into the EMA `M_anchor ← β·M_anchor +
(1-β)·G_anchor` (`feed_anchor_grads_into_ema`) and is **never applied to the
optimizer**. **Wiring `M_anchor` into the update as a force is the load-bearing change.**

**The staleness model.** `M_anchor ≈ g_true(θ_{t-K})`. For small `K·lr`
(`K=5`, `lr=1e-6` → weights barely move over 5 steps) `g_true(θ_{t-5}) ≈ g_true(θ_t)`,
so `M_anchor` is a usable estimate of the **current** true-gradient *direction*. The
correction must work from this stale estimate alone (Constraint 2) — it can NEVER assume
a fresh anchor.

**Scale caveat (important for every injection candidate).** Rescale inflates `‖G_mask‖`
≈9× relative to a dense/true gradient, so `‖G_mask‖ ≫ ‖M_anchor‖`. A naïve `G_mask +
M_anchor` is swamped — the true-gradient force barely rotates the update. **Every
injection candidate must scale-match** the injected term to `‖G_mask‖` (Adam's `m̂/√v̂`
is scale-invariant and verl grad-clips, so what matters is the *direction* we steer
`G_mask` toward, not the absolute magnitude — settled decision, `runs/SUMMARY.md`).

**Coverage caveat.** The as-implemented hook caps correction at `max_targets=4` matrices
(an EXP-7 *discovery* smoke value). Four matrices out of ~196 targeted 2D decoder
matrices cannot move the training curve. **Every candidate raises `max_targets=-1`
(all targeted matrices)** so the correction actually covers the network. Injection modes
carry **no SVD/basis cost**, so `-1` is cheap; only the per-matrix `M_anchor` EMA is
stored (offload to `ema_device=cpu` if HBM-bound).

---

## 1. HARD CONSTRAINTS every candidate obeys (plan §HARD CONSTRAINTS)

1. **No periodic full-gradient optimizer step.** `clean_cadence=0` always. The only
   full-fidelity passes allowed are the anchor's **STALE** reference passes that FEED the
   correction — never a fresh dense gradient applied as the update.
2. **Staleness mandatory.** `delay_K=5`, never 0, never 20. A correction that needs a
   fresh anchor is INVALID (it would not survive a decentralized/pipeline-parallel
   deployment where the full gradient always lands ~K steps old).
3. **Supply the missing component, do not reweight.** The correction must ADD the part
   `G_mask` lacks — not linearly reweight `G_mask` in a subspace (that is precisely why
   the as-implemented spectral filter was inert under orthogonality).

---

## 2. Candidate enumeration

Fixed across all candidates: `MASK_P=0.9`, `rescale=ON`, `mask_recompute=ON`,
`ANCHOR_CADENCE=5`, `ANCHOR_DELAY_K=5` (STALE), `CLEAN_CADENCE=0`. The correction is the
ONE varying dimension, explored serially.

### C0 — As-implemented spectral reweighting (the FLOOR, not a candidate)
- **Mechanism.** `spectral_filter.correct_matrix` as committed: two-sided Tikhonov
  reweighting of `G_mask` in `M_anchor`'s SVD basis (the formula in §0).
- **Why it is the floor.** Inert by orthogonality (GSM8K 0.080, pearson ~0.004). Run
  UNCHANGED as sequence step 1b (`curvematch_spectral_baseline_c5_d5`) and cached as the
  reference every candidate must beat. **This is the explicit departure point for C2.**

### C1 — Stale-anchor additive injection (direction-matched true-gradient force) — RUN FIRST
- **Mechanism.** At the grad-correction hook, for each targeted 2D matrix, read the stale
  true-gradient EMA `M_anchor` and **ADD** it as a scale-matched force:
  ```
  G_corr = G_mask + γ · (‖G_mask‖_F / (‖M_anchor‖_F + ε)) · M_anchor
  ```
  New knob `spectral.correction_mode="inject"` selects this path (default `"reweight"` =
  C0, byte-identical); `spectral.inject_gamma=γ` (start 1.0). `max_targets=-1`.
- **Rationale (`g_mask=g_true+b+ξ`).** `M_anchor ≈ g_true(stale)`. Adding a scale-matched
  copy injects energy **along the true ascent direction** — exactly the directions
  masking rotated away (where `G_mask≈0` by orthogonality, so the add is **constructive,
  not a reweight**). This partially cancels `b` by re-supplying `+g_true`. `γ` tunes how
  hard we steer toward the (stale) truth vs the (biased) masked gradient.
- **Constraints.** (1) `M_anchor` is the stale anchor EMA — no clean step. (2) fed by
  `delay_K=5` `G_anchor`. (3) **additive** supply of `g_true`'s direction — not a
  reweight of `G_mask`. ✓
- **Predicted curve.** Should lift reward **off the 0.13 floor toward dense**. `γ` too
  small → undershoot (stays near floor); too large → over-weights the stale direction →
  overshoot/oscillation. Diagnostic to log: live **cos(G_mask, M_anchor)** (confirms
  orthogonality on the REAL anchor) and the injected-norm ratio.

### C2 — Complement-projection injection (DERIVED FROM the spectral correction) — the spectral-derived candidate
- **Mechanism.** Reuse the same anchor signal the `SpectralFilter` already holds, but
  replace its *reweighting* with *additive injection of the orthogonal complement*: add
  the part of `M_anchor` that `G_mask` does NOT already span —
  ```
  P_{G_mask}(M_anchor) = (⟨G_mask, M_anchor⟩ / ‖G_mask‖²) · G_mask        # projection onto span(G_mask)
  G_corr = G_mask + η · scale · (M_anchor − P_{G_mask}(M_anchor))          # inject the COMPLEMENT
  ```
  (`scale = ‖G_mask‖/‖M_anchor‖` for the same scale-match reason as C1.) This is the
  **minimal edit that turns the spectral filter from a reweighting of `G_mask` in a
  subspace into an additive injection of the missing component** — the literal §(i)
  mandate ("turning reweighting into additive injection / complement-projection").
- **Rationale.** Orthogonality (cos≈0) ⇒ `P_{G_mask}(M_anchor) ≈ 0` ⇒ the injected term
  **≈ the full `M_anchor`** = the entire missing true direction. So **under the measured
  orthogonality, C2 *collapses to C1*** — which is why C1 is the clean first realization
  and C2 is the principled framing (the complement form also stays correct when cos is
  NOT ≈0, avoiding double-counting the aligned part). The complement of `g_true` w.r.t.
  `g_mask` is precisely what `b` rotated away.
- **Constraints.** ✓ (stale `M_anchor`; additive complement; `delay_K=5`).
- **Predicted curve.** ≈ C1 at the orthogonal limit; strictly safer when alignment grows
  mid-run (it never re-injects energy `G_mask` already has). Run if C1 shows signal but
  drifts late (suggesting the aligned-part double-count matters as training proceeds).

### C3 — Explicit `b`-estimator (stale masked-minus-unmasked bias removal)
- **Mechanism.** Extend the anchor pass to compute, at the SAME stale weights `θ_{t-5}`,
  **both** an unmasked grad `G_anchor` AND a masked grad `G_anchor^mask`. Their
  difference is a direct estimate of the bias:
  ```
  b̂_stale = G_anchor^mask − G_anchor        # = (g_mask − g_true) at θ_{t-5}
  G_corr   = G_mask − λ · b̂_stale            # subtract the estimated bias  (EMA-smoothed b̂)
  ```
  New knob `correction_mode="bias_est"`, `bias_lambda=λ`. Requires a second (masked)
  stale pass in `_maybe_comm_eff_anchor_refresh` (still cadence=5).
- **Rationale.** The most literal attack on `g_mask=g_true+b+ξ`: **measure `b` and
  subtract it.** Valid because `b` is slowly varying (curvature-aligned, accumulates over
  windows — `findings/NEXT_RESEARCH.md`), so `b̂_stale ≈ b_current`.
- **Constraints.** (1) both stale passes are reference passes, no optimizer step. (2)
  `delay_K=5`. (3) supplies `−b` (the missing correction). ✓
- **Predicted curve.** Most principled de-biasing; should track dense best **if** `b` is
  slowly varying. Risks: `b̂` is itself stale+noisy; doubles the anchor compute. Run if
  direction-injection (C1/C2) lifts reward but does not *track* dense step-for-step.

### C4 — Staleness-aware anchor aggregation (refinement axis on the winner)
- **Mechanism.** Vary how `M_anchor` aggregates stale `G_anchor` to reduce lag: (a) lower
  `beta_anc` (→0.5, →0.0 = raw last `G_anchor`); (b) linear extrapolation across two
  refreshes `M̂ = G_anchor(t) + [G_anchor(t) − G_anchor(t−cadence)]`; (c) age-decay the
  injection weight `γ_eff = γ·ρ^{age}` between refreshes.
- **Rationale.** C1/C2/C3 assume `M_anchor ≈ g_true(current)`; `delay_K=5` + a heavy EMA
  (`β=0.95`) makes it lag. Better aggregation sharpens the estimate → better step-for-step
  tracking. (The "different aggregations/decays of the anchor signal" direction.)
- **Constraints.** ✓ (all variants use only the stale anchor).
- **Predicted curve.** A multiplier on whichever base candidate wins; tune `β`/extrapolation
  AFTER a base candidate shows it lifts off the floor.

### C5 — Boundary-activation injection (alternative locus; heavier change; fallback)
- **Mechanism.** Inject the missing component at the **7 boundary activations** (where the
  mask zeroed dims) instead of in parameter-gradient space: use the stale anchor's UNMASKED
  boundary activations to estimate the per-(token,channel) activation the mask removed and
  add it back at the boundary forward, so the downstream backward carries the missing signal.
- **Rationale.** `b` originates at the masked boundaries (RMSNorm `1/RMS` compounding).
  Correcting at the **source** (activations) is cheaper/more local than reconstructing `b`
  in parameter space.
- **Constraints.** ✓ but touches `activation_mask.py` + the boundary forward in
  `transformer_impl.py` — the most invasive change.
- **Predicted curve.** Potentially the most faithful (corrects at source); deferred unless
  parameter-space injection (C1–C3) plateaus.

---

## 3. Run order + rationale

1. **C1 (`exp/18-anchorinject-c5d5`)** — simplest realization of the load-bearing idea
   (inject the stale true gradient as a scale-matched force). Establishes *whether
   injection moves reward off the floor at all*. Logs live cos(G_mask, M_anchor) — the
   first direct measurement of orthogonality on the live anchor.
2. **C2** — if C1 lifts reward but drifts late → switch to the complement form (the
   spectral-derived candidate) to stop double-counting the aligned part.
3. **C3** — if direction-injection lifts but does not *track* dense step-for-step → measure
   and subtract `b` explicitly.
4. **C4** — refinement axis (anchor freshness) applied to whichever base candidate shows life.
5. **C5** — fallback locus if parameter-space injection plateaus.

STOP at the **first** candidate clearing the bar (mean `|Δreward|≤0.05` over steps 1..50
AND final `|Δreward|≤0.05`, no collapse, slope-sign match, `pg_loss` tracks, `grad_norm`
finite, constraints verified). Per `iterations:3`, at most 3 REVISE cycles on this lineage.

---

## 4. Results-driven theory log (appended after each iteration — the observe→theorize→propose loop)

### References cached (2026-06-03)
- **Dense TARGET** (`metrics/curvematch_dense_ref_50step.jsonl`, 50/50): reward **0.135→0.868**, steep rise to ~0.75 by step 10 then plateau ~0.78–0.85; grad_norm 0.32–0.49, no NaN.
- **Spectral FLOOR** (`metrics/curvematch_spectral_baseline_c5_d5.jsonl`, 50/50, rc=0): flat **mean 0.135** (0.111–0.164) — inert-by-orthogonality CONFIRMED on the live anchor at c5/d5. The "beat-this" baseline.
- **Anchor-OOM engineering finding:** the anchor's UNSHARDED full backward OOMs at 36864 tok/gpu on 4×H200 (vLLM + FSDP + the ~3 GB clone). Fix = launcher-documented halve to `PPO_MAX_TOKEN_LEN_PER_GPU=18432` (no method change). Validated (floor ran 50 steps, anchor fired 40×, OOM=0). Every anchor-ON cell inherits it.

### Iteration 1 — C1 (stale-anchor additive injection) — config refinement BEFORE the result
**Observation (pre-result, from the live launch diagnostics):** at the first C1 launch (`seed_anchor_cache=true`, launcher default), `inject_matrix` fired ~3137× by step 2 while `anchor_backwards=0` — i.e. the injection was adding the **seeded random** `M_anchor` (a deterministic PSD basis), NOT the real stale anchor gradient (which only populates `M_anchor` once the live anchor fires ~step 3). The runner's "cos(G_mask,M_anchor)≈0" was therefore partly a **seed artifact** (random vs G_mask is trivially orthogonal), not the real G_mask-vs-true-gradient measurement.
**Theory (tie to the method):** for an ADDITIVE-injection correction, seeding `M_anchor` injects a random-direction force at γ=1, and the `beta_anc=0.9` EMA makes the seed persist (≈0.9^k decay over k anchor fires) — contaminating the injected direction for much of the 50-step window. This is benign for the REWEIGHT floor (seed = geometry only) but corrupts INJECT (seed = an added force).
**Refinement proposed + applied:** relaunch C1 with `seed_anchor_cache=false` ⇒ `M_anchor` starts at ZERO ⇒ injection is a no-op until the live anchor fires, then injects the REAL stale true-gradient direction (scale-matched). Killed the seeded launch at step 2 (minimal waste), relaunched clean via `c1_relaunch.sh`.

### Iteration 1b — C1 — a LOAD-BEARING ANCHOR-CIRCUIT BUG found + fixed (the clean run revealed it)
**Observation:** the clean (seed=false) C1 launch OOM'd at step 7 AND logged `spectral_corrections=2744` with **ZERO `[EXP-18][inject]` lines** — injection no-op'd entirely — while the anchor logged `targets=196 ||dM_anchor||≈0.027`. The inject-path param name was `model.layers.0._fsdp_wrapped_module.self_attn.q_proj.weight` (per-layer FSDP infix).
**Root cause (a genuine bug in the committed anchor circuit, not the method):** `build_anchor_module`'s `copy.deepcopy` fails on the HF-monkey-patched model → the fallback `ModelClass(cfg)` clone has PLAIN (non-infixed) names. Exact-name matching then failed at two boundaries: (1) the fallback param-copy + the `delay_K`-stale snapshot-load matched **0/338** params → the clone ran on **RANDOM init weights** → `G_anchor` was garbage (nonzero, hence the misleading `||dM_anchor||≈0.027`); (2) the EMA was keyed under clone (non-infixed) names while `inject_matrix` read under live (infixed) names → `M_anchor` read as ZERO → injection no-op. **Implication: the as-implemented anchor circuit never delivered the real stale gradient under nested FSDP — prior anchor-experiment inertness (EXP-16, 0.080) was confounded by this, not purely orthogonality.**
**Fix (`exp/18-anchorinject-c5d5` @ e65e2c98a):** `_canon(name)` strips the `._fsdp_wrapped_module` infix at every name-key boundary — EMA access in `spectral_filter.py` (ensure/anchor_on/update_anchor/refresh_basis/correct_matrix/inject_matrix), the snapshot-load in `transformer_impl.py`, and the fallback copy in `anchor.py`. +4 CPU key-consistency tests (feed under clone-name, read under live-name → resolves to one canonical key). VALIDATED on the box: `[anchor-load] loaded 338/338 stale params (canon-matched)` (was 0/338), thousands of `[inject]` lines with REAL `cos(G_mask,M_anchor)`∈{−0.085,−0.161,0.046,−0.001,…} (near-orthogonal — orthogonality now confirmed on the REAL stale gradient), `||inj||/||G_mask||≈1.0`, `||dM_anchor||` 0.027→1.5e-4 (real small unmasked grad). OOM fixed with `ema_device=cpu`.
**RESULT (C1, valid+fixed run, killed at step 34 — terminal collapse):** reward 0.134→0.163(s9)→0.095(s16)→0.020(s24)→**0.000(s33)**. `curve_match` vs dense (`curve_match_C1.md`): **mean |Δreward| = 0.611** (tol 0.05 → FAIL), final |Δ| = 0.805, **slope sign OPPOSITE dense** (cand −0.13 vs dense +0.67), collapsed to 0.0 — **WORSE than the floor** (floor mean |Δ|=0.596). VERDICT: **REVISE (catastrophic collapse).**
**Theory — WHY (tie to b/orthogonality/staleness):** the additive form `G_corr = G_mask + γ·scale·M_anchor` (cos≈0, so complement≈M_anchor) keeps the **biased `G_mask` at full weight** and tacks on an **orthogonal, ‖G_mask‖-scale-matched** force → the update is ~45° from both directions at √2·‖G_mask‖ magnitude. Because `G_mask` is rescale-inflated (~9×), scale-matching the injection to it makes the added force ~9× a natural gradient — a large, stale, orthogonal perturbation that follows neither the masked nor the true ascent direction → it **destroys** the policy (reward→0), it doesn't correct it. ADD cannot escape `G_mask`'s wrong direction; it only adds a perpendicular component. Confirms: to follow the true gradient you must **REPLACE/downweight `G_mask`**, with a **stable (non-inflated) magnitude**.
**Next candidate proposed FROM this evidence → C2 = convex BLEND** (`runs/EXP-18/candidate-02-C2-blend-spec.md`): `G_corr = (1-η)·G_mask + η·scale·M_anchor`, η=0.7. Replaces the biased direction with the scale-matched stale true gradient at a STABLE magnitude (‖G_corr‖≤‖G_mask‖ — fixes C1's √2 blow-up). Decisive test: does steering ~70% toward the (stale) true gradient lift reward off the floor? If it ALSO collapses → the staleness itself (delay_K=5 + beta=0.9 EMA) is the limiter → C3 = blend + lower beta_anc / extrapolation; if even η→1 (pure stale-grad descent) can't match dense → STOP finding (realistic-staleness target unreachable with this family). Counts as REVISE iteration 1 of 3.

### Iteration 2 — C2 (convex blend η=0.7, beta_anc=0.9) — REVISE (slow decline)
**RESULT (killed step 23):** reward 0.133→0.116(s5)→0.111(s14)→**0.031(s23)**, declining. `curve_match` vs dense (`curve_match_C2.md`): **mean |Δreward| = 0.531** (FAIL), final |Δ| = 0.744, slope OPPOSITE dense (cand −0.10 vs +0.64). Stable magnitude held (‖G_corr‖/‖G_mask‖≈0.76, NO C1-style √2 blow-up, grad_norm 2–3 not 5–6) so it did NOT violently collapse — but it **slowly degrades** rather than lifts. VERDICT: **REVISE.**
**Theory — WHY:** the blend fixed C1's magnitude pathology (stable, no collapse) yet steering ~70% of the update along the stale true-gradient direction still drives reward DOWN, not up. So the problem is not magnitude — it is the **DIRECTION/quality of the stale anchor gradient itself**. Two suspects: (i) the `beta_anc=0.9` EMA averages ~10 refreshes × cadence-5 = ~50 steps of history; over the steep early trajectory (dense 0.13→0.75 in 10 steps) the true-gradient direction rotates fast, so the EMA is a stale "smear" of obsolete directions; (ii) **a deeper anchor-loss suspicion**: the anchor reuses the **masked** `old_log_probs` with an **unmasked** new forward, so its GRPO importance ratio ≠ 1 — the "true gradient" may be importance-ratio-corrupted, not the clean unmasked gradient.
**Next candidate FROM this evidence → C3 (FINAL iter 3/3) = blend η=0.7 + `beta_anc=0.0`** (`c3_relaunch.sh`, config-only). Single-variable change from C2: use the RAW last 5-stale gradient (no EMA smear). Decides suspect (i): if C3 lifts → the EMA averaging was washing out the signal (β is tunable → progress/PASS-path); if C3 ALSO degrades → the `delay_K=5` staleness itself (and/or the ratio-corruption) is the limiter, and **direct forcing of the masked gradient toward the stale anchor gradient does NOT reproduce dense → STOP** with a real negative finding, and the importance-ratio-clean anchor becomes the next-cycle direction (beyond iterations:3). Counts as REVISE iteration 2 of 3.
