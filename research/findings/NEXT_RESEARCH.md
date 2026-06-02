# NEXT_RESEARCH — the next cycle

**Directive (head researcher):** find a *working* method combining a **spectral correction** with an **anchor circuit** as a **cheap, continuous surrogate for the periodic dense "clean" step.**

**The moment of truth (why this direction).** Masking alone makes a biased, noisy gradient that does not learn at high mask rates. But **periodically passing a full dense gradient every K steps re-anchors training and reaches results comparable to dense** (GSM8K parity; clean-resettable sawtooth, EXP-17) — proof the signal is *recoverable*, not lost. If a *sparse* full gradient does it, a **cheaper continuous spectral correction** (driven by an anchor = a low-frequency true-gradient reference) should recover the same at a fraction of the comm. The clean step is expensive (full inter-stage comm every K steps, savings capped `(K-1)/K·p`); the prize is a correction applied *every* masked step that matches it at lower comm. The bar (Constraints 1–2 below): target the bias directly — not reweight `G_mask` — and clear the harder-task case where even the clean step stalls.

Grounding: `runs/SUMMARY.md` (the proven result + why, knobs, settled base, dataset prep).

## What we know

| dataset | base (no RL) | masked p=0.9 + clean@20 | dense | reading |
|---|---|---|---|---|
| GSM8K | 0.715 | 0.735 (EXP-17) | 0.741 | parity = **elicitation** |
| Big-Math | 0.480 | 0.55 flat (EXP-19) | 0.61 (EXP-20) | **stalls** = gradient-fidelity/SNR limit |

The mask is not a policy — it's a biased, high-variance **estimator of the true GRPO gradient**: `g_mask = g_true + b + ξ`. Rescale unbiases the *activation*, not the gradient. `b` is a systematic curvature-aligned bias (delta-method term) that **accumulates** across a masking window; clean@K is error-feedback re-sync that injects `g_true` and resets `b` before it flips the ascent projection.

## HARD CONSTRAINT 1 — the prior anchor+spectral failed, by orthogonality

EXP-16 `anchor@2+spectral@2`, no clean steps → GSM8K **0.080 (≈random)**, inert, pearson still ~0.004. The `SpectralFilter` linearly reweights `G_mask` inside the unmasked-anchor SVD subspace — but `G_mask` is ~orthogonal to the true direction (cos≈0), so no linear projection of it can synthesize the missing `b`-correction. And the clean anchor gradient is **never applied** (only feeds the EMA basis). **The redesign must attack `b` directly and let the anchor's true-gradient reach the optimizer as a force — not reweight `G_mask` in a subspace.**

## HARD CONSTRAINT 2 — the bar is higher than "match clean@K"

On Big-Math even clean@20 stalls (~0.55) while dense learns (~0.61); the clean step's grad_norm there is ~2× smaller (smaller `‖g_true‖`). If hard tasks are in scope, the correction must recover signal the clean step itself does not.

## The empirical target — match the dense training curve

This is RL, not supervised learning: there is no recipe to port, so the correction must be **found empirically**. The concrete, measurable target:

**Make the masked+correction training curve match the dense training curve within ≤50 steps** — GSM8K, anchor `cadence`=5, staleness `delay_K`=5 (the anchor refreshes every 5 steps from 5-step-stale weights — a realistic latency, see `runs/SUMMARY.md`).

- **Reference:** dense run (`COMM_EFF_ENABLED=false`), ≤50 steps, GSM8K, EXP-17 shape (lr=1e-6, n=8, train_batch=128, mini_batch=64). Log the per-step curve (reward + loss + val); cache it.
- **Candidate:** masked (p=0.9, rescale ON, `mask_recompute=true`) + the correction under test, anchor `cadence`=5, `delay_K`=5, **`clean_cadence` OFF** (the correction must stand on its own — no leaning on frequent clean steps). ≤50 steps, same logging.
- **Match metric:** per-step distance between the curves (mean |reward_masked − reward_dense| over the 50 steps + the final gap + the within-window slope). "Match" = the candidate **tracks dense across the whole trajectory** within a stated tolerance, not just at the end — the strong evidence that the correction recovers the true gradient step-by-step, under realistic staleness.

**Two hard constraints on the search:** (a) **staleness is mandatory** — `delay_K`=5, never 0; a realistic decentralized / PP system can never deliver a *fresh* full gradient (it always lands ~K steps stale), so a correction that only works with a fresh anchor is invalid; (b) **the periodic clean step was only an existence proof** that the signal is recoverable — `clean_cadence` stays OFF and is **not** reintroduced. The only full-fidelity passes allowed are the anchor's *stale reference* passes that feed the correction — never a fresh full gradient applied directly as the optimizer update.

## The recursive search loop

A pure-research issue runs an agent that **recursively explores corrections**, every iteration backed by a real run:

0. **Enumerate all candidate corrections theoretically first** — a complete list, each with its mechanism, rationale (why it should make the masked curve track dense), how it respects the constraints above, and predicted behavior — *before* running anything.
1. Establish the dense reference curve (one run, cached).
2. Propose a correction (a refinement of a prior attempt, or a new approach).
3. Patch it on an `exp/*` branch; run masked+correction (≤50 steps, cadence 5, delay_K 5, clean_cadence OFF); log the curve.
4. Compare to dense; compute the match metric; record what improved / regressed.
5. Not matched → refine and loop. Matched → lock it, confirm on a longer run.

No correction is "good" until its ≤50-step curve is shown next to dense. The agent is expected to go **beyond** the starting list below.

## Candidate corrections to explore (open — starting points, not a prescription)

All must respect Constraint 1 (supply the component `G_mask` lacks — don't merely reweight `G_mask` in a subspace) and work under `delay_K`=5 staleness:
- Estimate and **remove the masking bias `b`** using the anchor's (stale) true-gradient signal, applied as a force on the masked gradient (with a staleness-aware decay).
- Act on the **boundary activations** (cheap, local) rather than the full parameter gradient.
- Subspace methods that **add a missing component** rather than reweight an existing one.
- Different aggregations / decays of the anchor signal across its 5-step staleness.

*Where the code is:* the anchor circuit is already plumbed (`verl/workers/comm_eff/anchor.py`: a `delay_K`-stale isolated clone producing the raw unmasked gradient; today it only feeds the EMA, never the optimizer). Wiring its signal into the actual update is the load-bearing code change.

## Success bar
- **Primary (this cycle):** the masked+correction ≤50-step GSM8K training curve matches the dense curve within tolerance, at cadence 5 / delay_K 5 / `clean_cadence` OFF.
- **Then:** confirm on one longer run (curve still tracks; final GSM8K val ≥ dense within noise) and report **net inter-stage comm vs dense**. cadence-5 anchor passes count against savings; raising cadence for savings is the follow-on **once matching holds** — match first, optimize comm second.

## Supporting diagnostics (optional, as the search needs them)
- **p-sweep** — `COMM_EFF_MASK_P ∈ {0.9,0.7,0.5,0.3,0.1}`: how p-sensitive is GSM8K learning, and how much a lower p alone closes the curve gap (p is also a savings knob).
- **clean-only ablation** — dense-every-K vs masked+clean@K: is masking contributing learning, or are the clean steps doing it? (Honesty check on the framing.)

*Savings metric:* boundary-activation volume not communicated; report **net of the correction's (and the cadence-5 anchor's) overhead**, vs dense.
