# EXP-31 — the next surpass lever (worked out, execution-ready)

The sub-basis exploration established the ceiling: **amplifying a direction the model already
descends speeds the *path* to dense's optimum, not the *destination*** → parity, not surpass.
To beat dense, comm-eff must reach a **better (flatter / better-generalizing) optimum**. That is
a *regularization* lever, not an amplification one. This doc specifies it so it can be run the
moment a GPU is available.

## Thesis (why this can surpass when amplification can't)
Greedy val `mean@1` rewards the **mode** of the policy. A flatter minimum generalizes better →
the greedy mode lands in a better basin → higher val. Beneficial gradient noise (SGLD / SAM-style)
is the canonical way to bias SGD toward flat minima. Crucially, **the comm-eff substrate already
injects structured gradient perturbation** (the codec drops ~92% of the gradient energy) — but
that perturbation is **biased** (harmful, per EXP-25: 42:1 SNR, not zero-mean). The lever is to
**replace the biased codec noise with a zero-mean, tunable perturbation** → turn the compression's
side-effect from a *liability* into a *regularizer*. This is the "zero-mean + tunable perturbation"
route flagged (but never run on this substrate) in the prior strategy work.

## Mechanism — `perturb_sigma` (a new, default-OFF merger knob)
In `delayed_ef_matrix`, AFTER the existing `g_corr = gm + λ·(δ_B2 + δ_subbasis)`:
```
σ = self.perturb_sigma                       # default 0.0 ⇒ skip ⇒ bitwise B2/Cell-D
if σ > 0:
    gnorm = ‖g_corr‖
    ξ = randn(g_corr.shape, gen=seed(step, target))   # cross-rank-IDENTICAL seed
    ξ = ξ / ‖ξ‖                                        # unit
    g_corr = g_corr + σ · gnorm · ξ                    # ‖perturbation‖ = σ·‖g_corr‖
```
- **Zero-mean over steps** (fresh seed each step) ⇒ E[update] unchanged ⇒ unbiased descent + exploration.
- **Cross-rank-identical** (seed = f(step, target), same on every DP rank) ⇒ no rank divergence (same discipline as the sub-basis SVD seed).
- **σ relative to ‖g_corr‖** ⇒ scale-free, tunable. **Default σ=0 ⇒ bitwise-B2** (off-path parity), so it composes cleanly with everything already on vast-ai-workload.
- **Local, zero added communication** ⇒ stays "strictly comm-efficient" (the comm-eff substrate is untouched; this is a free local gradient modification).

Optional refinement (v2): project ξ into the **act-null-space** `(I − Q_actQ_actᵀ)` — perturb only
the directions the codec drops (structurally-matched noise) rather than isotropic. Needs the
per-target↔boundary Q_act map (the sub-basis granularity issue); start isotropic.

## Sweep + success criterion (the GPU run, when a box is available)
- Substrate = B2 (delayed_ef λ=1) on the operator box; `COMM_EFF_SPECTRAL_PERTURB_SIGMA ∈ {0.05, 0.10, 0.20}`, 50 steps, seed 0, all other knobs at B2 defaults + `disable_custom_all_reduce`.
- **SURPASS** iff a σ's val@50 band clears dense (0.7506) by > pooled-SE (0.020), certified by seed replicates (the deferred Cell F). **PARITY/NULL** otherwise (and σ that hurts → confirms the codec noise was already near the useful ceiling).
- Also pin the deferred dense×3 / B2×3 bands in the same session (the box-stop cut these short).

## Build state
NOT yet implemented (operator wound down to the claim layer; no GPU). When greenlit: ~40-line
addition mirroring the `delta_subbasis` knob (config field + merger term + actor.yaml decl +
launcher passthrough + CPU tests: σ=0 bitwise-B2, σ>0 zero-mean + cross-rank-identical), default OFF,
on `vast-ai-workload`. Est. ~1 GPU-hr per σ.

## Honest prior
Low (~10–15%). Most perturbation routes were falsified in prior work; this is the cleanest survivor
and the only *destination-changing* (vs path-speeding) lever left inside the locked frame. If it
nulls, the rigorous conclusion is **comm-eff = dense-parity at ~5% comm cost** (already established),
and a genuine surpass would require relaxing a locked constraint (codec rank, delay_K, or the
greedy-only eval) — i.e. a new issue, not an EXP-31 cell.
