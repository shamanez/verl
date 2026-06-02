# NEXT_RESEARCH — the next cycle

**Directive (head researcher):** find a *working* method combining a **spectral correction** with an **anchor circuit** as a **cheap, continuous surrogate for the periodic dense "clean" step.**

**Why this is the direction:** the clean step is the existence proof that periodic true-gradient re-anchoring works (clean-resettable sawtooth, EXP-17). But it is expensive — full inter-stage comm every K steps, savings capped at `(K-1)/K·p`. The prize: a correction applied *every* masked step that recovers what the clean step recovers, at lower comm.

Grounding: `findings/theory/REPORT.md` (the why), `runs/SUMMARY.md` (knobs, settled base, dataset prep).

## What we know

| dataset | base (no RL) | masked p=0.9 + clean@20 | dense | reading |
|---|---|---|---|---|
| GSM8K | 0.715 | 0.735 (EXP-17) | 0.741 | parity = **elicitation** |
| Big-Math | 0.480 | 0.55 flat (EXP-19) | 0.61 (EXP-20) | **stalls** = gradient-fidelity/SNR limit |

The mask is not a policy — it's a biased, high-variance **estimator of the true GRPO gradient**: `g_mask = g_true + b + ξ`. Rescale unbiases the *activation*, not the gradient. `b` is a systematic curvature-aligned bias (delta-method term) that **accumulates** across a masking window; clean@K is error-feedback re-sync that injects `g_true` and resets `b` before it flips the ascent projection. (Derivation: REPORT §2.)

## HARD CONSTRAINT 1 — the prior anchor+spectral failed, by orthogonality

EXP-16 `anchor@2+spectral@2`, no clean steps → GSM8K **0.080 (≈random)**, inert, pearson still ~0.004. The `SpectralFilter` linearly reweights `G_mask` inside the unmasked-anchor SVD subspace — but `G_mask` is ~orthogonal to the true direction (cos≈0), so no linear projection of it can synthesize the missing `b`-correction. And the clean anchor gradient is **never applied** (only feeds the EMA basis). **The redesign must attack `b` directly and let the anchor's true-gradient reach the optimizer as a force — not reweight `G_mask` in a subspace.**

## HARD CONSTRAINT 2 — the bar is higher than "match clean@K"

On Big-Math even clean@20 stalls (~0.55) while dense learns (~0.61); the clean step's grad_norm there is ~2× smaller (smaller `‖g_true‖`). If hard tasks are in scope, the correction must recover signal the clean step itself does not.

## Run these two cheap experiments FIRST (pure-config, before any redesign)

**EXP-A — p-sweep (decisive).** Sweep `COMM_EFF_MASK_P ∈ {0.9,0.7,0.5,0.3,0.1}` on **both** GSM8K and Big-Math, all else at the EXP-17 shape, `clean_cadence=20` fixed.
- *Question:* is the wall mask-rate/SNR (Big-Math climbs as p drops → find threshold `p*`) or does it genuinely need a correction (flat at all p)?
- *Prediction:* GSM8K insensitive at high p; Big-Math climbs toward dense below some `p*`. Climbs → a cheap correction buys effective-low-p at high savings. Flat → the bias isn't a mask-rate artifact; the correction must attack `b`.

**EXP-B — clean-only ablation (honesty check).** Fixed K=20 on GSM8K: Arm 1 = dense step every K, **no update between**; Arm 2 = masked+clean@K (= EXP-17).
- *Question:* is masking contributing learning, or are the clean steps doing it all? Compare final val + within-window reward slope.
- *Decides framing:* Arm2 ≫ Arm1 → "masking supports learning" is honest; Arm1 ≈ Arm2 → downgrade to "masking doesn't destroy what clean steps learn."

## The redesign (only after EXP-A/B)

Attack `b`; use the anchor as a stale true-gradient **reference that contributes a component `G_mask` lacks**; apply the correction **continuously**. Starting hypotheses:
1. Estimate `b`'s dominant component (closed-form delta-method term, or an EMA of `(g_anchor_stale − g_mask)` on refresh steps) and subtract it every masked step.
2. The anchor is already plumbed (`anchor.py`: K-stale isolated clone, raw unmasked grad) but its gradient is never applied — making it reach the optimizer as a correction is the load-bearing code change.
3. Prefer correcting at the boundary activations (cheap, local) over a full parameter-gradient correction.

**Success bar (ties to GOAL "done"):** reach **≥ masked-clean@K val with the clean step OFF (or sparser)**, at **net inter-stage comm strictly below the clean@K baseline** — savings reported *net of the correction's own overhead*. GSM8K parity (0.7415) minimum; Big-Math improvement if EXP-A shows the wall is correctable, else clear Constraint 2.

*Savings metric:* boundary-activation volume not communicated. masked+clean@K = `((K-1)/K)·p` (EXP-17 K=20,p=0.9 → ~85.5%). A continuous correction targets ~p every step but must subtract its own comm cost. (Distinct from "clean-step sparsity," the looser ~95% figure.)
