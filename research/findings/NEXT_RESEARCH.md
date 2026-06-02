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

## Experiment protocol (new runs)

**GSM8K, ≤25 steps** — the standard fast/cheap testbed for this cycle (the spectral-correction iteration and the two gating runs below). 25 steps is more than enough to see the early trajectory, the clean/correction re-anchoring, and the within-window reward slope. Use a **short clean cadence (≈5)** so several cycles fit in 25 steps (clean@20 would give only one). Hold the rest of the EXP-17 shape: rescale ON, `mask_recompute=true`, no-KL no-entropy, lr=1e-6, n=8, train_batch=128, mini_batch=64. **Big-Math is an optional later confirmation** once something works on the cheap GSM8K testbed — not part of the fast loop. (25-step runs are for development/comparison; a final parity claim needs one longer confirmation run — see Success bar.)

## Run these two cheap experiments FIRST (pure-config, before any redesign)

**EXP-A — p-sweep.** Sweep `COMM_EFF_MASK_P ∈ {0.9,0.7,0.5,0.3,0.1}` on GSM8K, ≤25 steps, clean@5.
- *Question:* how p-sensitive is GSM8K learning? Theory predicts it stays ~insensitive at high p (elicitation needs only a coarse gradient); p is also itself a savings knob (boundary volume ∝ p).
- *Optional confirmation (later, 1 run):* repeat at the most informative p on Big-Math for the decisive `p*` question — does lowering p unlock the hard task, or is the bias not a mask-rate artifact (→ the correction must attack `b`)?

**EXP-B — clean-only ablation (honesty check).** GSM8K, ≤25 steps, K=5: Arm 1 = dense step every K, **no update between**; Arm 2 = masked+clean@K.
- *Question:* is masking contributing learning, or are the clean steps doing it all? Compare val + within-window reward slope.
- *Decides framing:* Arm2 ≫ Arm1 → "masking supports learning" is honest; Arm1 ≈ Arm2 → downgrade to "masking doesn't destroy what clean steps learn."

## The redesign (only after EXP-A/B)

Attack `b`; use the anchor as a stale true-gradient **reference that contributes a component `G_mask` lacks**; apply the correction **continuously**. Starting hypotheses:
1. Estimate `b`'s dominant component (closed-form delta-method term, or an EMA of `(g_anchor_stale − g_mask)` on refresh steps) and subtract it every masked step.
2. The anchor is already plumbed (`anchor.py`: K-stale isolated clone, raw unmasked grad) but its gradient is never applied — making it reach the optimizer as a correction is the load-bearing code change.
3. Prefer correcting at the boundary activations (cheap, local) over a full parameter-gradient correction.

**Success bar (ties to GOAL "done"):** on the 25-step GSM8K testbed, the correction (clean step OFF or sparser) must **hold the masked+clean@K trajectory** at **net inter-stage comm strictly below the clean@K baseline** (savings reported *net of the correction's overhead*). The final **parity claim (GSM8K val ≥ 0.7415)** is confirmed on **one longer run** once the 25-step testbed looks good; Big-Math improvement only if its optional p-run shows the wall is correctable, else clear Constraint 2.

*Savings metric:* boundary-activation volume not communicated. masked+clean@K = `((K-1)/K)·p` (EXP-17 K=20,p=0.9 → ~85.5%). A continuous correction targets ~p every step but must subtract its own comm cost. (Distinct from "clean-step sparsity," the looser ~95% figure.)
