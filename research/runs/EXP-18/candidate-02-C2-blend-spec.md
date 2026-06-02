# Candidate 2 (C2) — convex blend toward the scale-matched stale true gradient — spec

> Results-driven from C1's failure. C1 (`correction_mode=inject`, γ=1) ADDED a scale-matched
> force orthogonal to `G_mask` → reward COLLAPSED 0.13→0.02 (policy destroyed). Diagnosis:
> ADD keeps the biased `G_mask` at full weight + tacks on an equal-magnitude orthogonal vector
> → update is 45° from both directions at √2·‖G_mask‖ → follows neither, destabilizes. To
> actually follow the (stale) true gradient we must **REPLACE/downweight `G_mask`**, with a
> **stable magnitude** (no √2 blow-up). Branch `exp/18-anchorblend-c5d5` from `exp/18-anchorinject-c5d5`
> (keep the canon fix + inject mode; ADD a blend mode).

## Mechanism
```
scale  = ‖G_mask‖ / (‖M_anchor‖ + eps)          # scale-match M_anchor to G_mask's magnitude
G_corr = (1-η)·G_mask + η·scale·M_anchor          # convex blend (REPLACE, not add)
```
- Magnitude stable: ‖G_corr‖ ≤ ‖G_mask‖ (for orthogonal terms, = ‖G_mask‖·√((1-η)²+η²)) — fixes C1's √2 blow-up.
- At η→1: update ≈ scale-matched stale true gradient (decisive test: does descending the STALE true gradient match dense?). At η→0: ≈ `G_mask` (floor). 
- **Run η=0.7 first** (70% toward the true direction, 30% G_mask — decisively steer toward truth while retaining some masked signal + stability). This is the clean test of "does steering the update toward the stale true gradient — at a sane magnitude — lift reward off the floor?"
- Constraints: uses the STALE M_anchor (delay_K=5) as the force (Constraint 1 ✓ — not a fresh full grad), works under staleness (Constraint 2 ✓), and SUPPLIES the true direction by replacing the biased one (Constraint 3 ✓ — not a subspace reweight).

## Code change (4 files — mirror the C1 inject pattern)
1. `verl/workers/config/comm_eff.py` (CommEffSpectralConfig): allow `correction_mode ∈ {reweight, inject, blend}`; add `blend_eta: float = 0.5` + validation `0 ≤ blend_eta ≤ 1`.
2. `verl/trainer/config/actor/actor.yaml`: add `blend_eta` (and ensure `correction_mode` already lists blend) — verl struct-mode rejects unknown keys (the C1 lesson).
3. `verl/workers/comm_eff/state.py` (build): pass `blend_eta=float(getattr(spec_cfg,"blend_eta",0.5))` into SpectralFilter.
4. `verl/workers/comm_eff/spectral_filter.py`:
   - `__init__`: accept+store `blend_eta`; allow "blend" in the correction_mode assert.
   - add `blend_matrix(self, name, g_mask)` (canon the name first, same guards as inject_matrix):
     ```python
     name = _canon(name); self.ensure_anchor(name, g_mask)
     anc = self.anchor_on(name, g_mask.device).to(torch.float32); gm = g_mask.to(torch.float32)
     eps=1e-12; gm_norm=torch.linalg.norm(gm); anc_norm=torch.linalg.norm(anc)
     if anc_norm<=eps or gm_norm<=eps: return g_mask
     eta=self.blend_eta; scale=gm_norm/(anc_norm+eps)
     g_corr=(1.0-eta)*gm + eta*scale*anc
     cos=((gm*anc).sum()/(gm_norm*anc_norm+eps)).item()
     print(f"[comm_eff][EXP-18][blend] {name} eta={eta} cos(G_mask,M_anchor)={cos:.4f} "
           f"||G_corr||/||G_mask||={(torch.linalg.norm(g_corr)/(gm_norm+eps)).item():.4f}", flush=True)
     return g_corr.to(g_mask.dtype)
     ```
   - `update_anchor` basis-cache skip: extend to `self.correction_mode in ("inject","blend")` (blend needs no SVD basis either).
   - `apply_spectral_correction_to_params` dispatch: add `elif getattr(spectral,"correction_mode","reweight")=="blend": g_proj=spectral.blend_matrix(name, full)`.
5. CPU test: assert blend at η=1 → ≈ scale·M_anchor (within tol), η=0 → G_mask exactly, key-consistency (feed clone-name, blend live-name).

## Launch (reuse box 39132674; inherits ALL C1 fixes)
Same env as the fixed C1 (ema_device=cpu, max_targets=-1, seed_anchor_cache=false, 18432, anchor c5/d5, clean=0) but `correction_mode=blend`, `EXPERIMENT_NAME=curvematch_anchorblend_c5_d5`, + Hydra `"$@"`: `actor_rollout_ref.actor.comm_eff.spectral.correction_mode=blend actor_rollout_ref.actor.comm_eff.spectral.blend_eta=0.7`.
Pins (INVALID if violated): ANCHOR_DELAY_K=5, CLEAN_CADENCE=0, ANCHOR_CADENCE=5, MAX_RESPONSE 16384.

## If C2 (η=0.7) also collapses
→ the STALE true gradient direction itself is the limiter (too stale at delay_K=5 + beta=0.9 EMA). Next: C3 = blend with LOWER beta_anc (fresher anchor, e.g. 0.5/0.0) or staleness extrapolation (candidates.md C4), or the explicit b-estimator (candidates.md C3). If even η→1 (pure stale-grad descent) can't match dense, that is the STOP finding (realistic-staleness target unreachable with this family).
