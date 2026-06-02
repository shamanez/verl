# Candidate 1 (C1) — stale-anchor additive injection — exact code-change spec

> Hand this to the experiment-runner for the `exp/18-anchorinject-c5d5` dispatch.
> Branch `exp/18-anchorinject-c5d5` from `origin/vast-ai-workload`. The change is
> SURGICAL — 3 files, all in the plan's `target_modules`. The new knobs reach
> Hydra via the launcher's `"$@"` passthrough (NO launcher edit needed; the fields
> must exist in the structured config or OmegaConf rejects them).

## Mechanism (from candidates.md C1/C2)
For each targeted 2D matrix, ADD the scale-matched complement of the stale
true-gradient EMA `M_anchor` to the masked gradient — supplying the component
masking removed, instead of reweighting `G_mask`:
```
P            = (<G_mask, M_anchor> / ||G_mask||^2) * G_mask        # part of M_anchor already in span(G_mask)
M_complement = M_anchor - P                                        # the MISSING component
scale        = ||G_mask|| / (||M_anchor|| + eps)                   # rescale inflates ||G_mask|| ~9x → scale-match
G_corr       = G_mask + inject_gamma * scale * M_complement
```
Under the measured orthogonality (cos≈0) `P≈0` so this ≈ scale-matched direct
injection of the stale true gradient (C1). `inject_gamma` (start **1.0**) tunes it.

## File 1 — `verl/workers/config/comm_eff.py` (CommEffSpectralConfig)
Add two fields (place after `basis_cache: str = "cache"`):
```python
    # EXP-18/M4: correction mode. "reweight" (default) = the as-implemented
    # two-sided Tikhonov reweighting of G_mask (byte-identical to every prior
    # config). "inject" = ADD the scale-matched complement of the stale anchor
    # EMA M_anchor (supply the missing true-gradient component; the M4 redesign).
    correction_mode: str = "reweight"
    # EXP-18/M4: injection strength for correction_mode="inject" (force along the
    # stale true-gradient direction, scale-matched to ||G_mask||). Unused under
    # "reweight". >= 0.
    inject_gamma: float = 1.0
```
Add validation in `__post_init__` (near the other spectral checks):
```python
        if self.spectral.correction_mode not in ("reweight", "inject"):
            raise ValueError(
                f"comm_eff.spectral.correction_mode must be one of (reweight, inject); "
                f"got {self.spectral.correction_mode!r}"
            )
        if self.spectral.inject_gamma < 0.0:
            raise ValueError(f"comm_eff.spectral.inject_gamma must be >= 0; got {self.spectral.inject_gamma}")
```

## File 2 — `verl/workers/comm_eff/state.py` (CommEffState.build)
In the `if spec_enabled:` block, pass the two new fields into `SpectralFilter(...)`:
```python
                correction_mode=str(getattr(spec_cfg, "correction_mode", "reweight")),
                inject_gamma=float(getattr(spec_cfg, "inject_gamma", 1.0)),
```
(Add them to the existing keyword-arg list; nothing else changes.)

## File 3 — `verl/workers/comm_eff/spectral_filter.py`
**(a) `SpectralFilter.__init__`** — accept + store the new knobs:
```python
        correction_mode: str = "reweight",
        inject_gamma: float = 1.0,
        ...
        assert correction_mode in ("reweight", "inject"), correction_mode
        self.correction_mode = str(correction_mode)
        self.inject_gamma = float(inject_gamma)
```
**(b) `update_anchor`** — skip the SVD basis cache in inject mode (inject needs
no basis; computing 196 full SVDs per refresh would stall the run):
```python
        if self.basis_cache == "cache" and self.correction_mode != "inject":
            self._basis[name] = compute_basis(new, svd_mode=self.svd_mode, rank=self.rank)
```
**(c) New method `inject_matrix`** (next to `correct_matrix`):
```python
    def inject_matrix(self, name: str, g_mask: torch.Tensor) -> torch.Tensor:
        """EXP-18/M4 additive injection: G_corr = G_mask + gamma*scale*(M_anchor - P_Gmask(M_anchor)).

        Supplies the component of the stale true-gradient EMA M_anchor that G_mask
        does NOT already span (the part masking rotated away), scale-matched to
        ||G_mask|| (rescale inflates ||G_mask|| ~9x; Adam+grad-clip make the
        *direction* the load-bearing quantity). Under orthogonality (cos≈0) the
        projection ~0 and this is scale-matched direct injection of M_anchor.
        Returns G_corr with g_mask's shape/dtype/device.
        """
        self.ensure_anchor(name, g_mask)
        anc = self.anchor_on(name, g_mask.device).to(torch.float32)
        gm = g_mask.to(torch.float32)
        eps = 1e-12
        gm_norm = torch.linalg.norm(gm)
        anc_norm = torch.linalg.norm(anc)
        if anc_norm <= eps or gm_norm <= eps:
            return g_mask  # anchor not warmed / zero grad → no-op
        coeff = (gm * anc).sum() / (gm_norm * gm_norm + eps)   # <G_mask,M_anchor>/||G_mask||^2
        complement = anc - coeff * gm
        scale = gm_norm / (anc_norm + eps)
        g_corr = gm + self.inject_gamma * scale * complement
        # Diagnostic: cosine(G_mask, M_anchor) — measures orthogonality on the LIVE anchor.
        cos = (coeff * gm_norm / (anc_norm + eps)).item()
        print(f"[comm_eff][EXP-18][inject] {name} cos(G_mask,M_anchor)={cos:.4f} "
              f"gamma={self.inject_gamma} scale={scale.item():.4f} "
              f"||inj||/||G_mask||={(torch.linalg.norm(self.inject_gamma*scale*complement)/(gm_norm+eps)).item():.4f}",
              flush=True)
        return g_corr.to(g_mask.dtype)
```
**(d) `apply_spectral_correction_to_params`** — dispatch on mode (replace the
single `g_proj = spectral.correct_matrix(name, full)` line):
```python
        if getattr(spectral, "correction_mode", "reweight") == "inject":
            g_proj = spectral.inject_matrix(name, full)
        else:
            g_proj = spectral.correct_matrix(name, full)
```
(Everything else — discovery log, `relative_change`, writeback, `spectral_corrections++` — unchanged. For inject mode `relative_change` = ||injected||/||G_mask|| = the injection ratio, a useful metric.)

## Launch (reuse the search box; NO launcher edit)
Same env as the spectral floor PLUS the new knobs via Hydra `"$@"` passthrough,
and `MAX_TARGETS=-1` so the correction covers ALL targeted matrices (inject has
no SVD cost; M_anchor EMA ≈5 GB/rank — fits H200/H100):
```
PROJECT_NAME=comm_eff_curve_match_m4 EXPERIMENT_NAME=curvematch_anchorinject_c5_d5 \
COMM_EFF_ENABLED=true \
COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=0 \
COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_MAX_TARGETS=-1 \
TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=False TEST_FREQ=100000 USE_DYNAMIC_BSZ=True \
NGPUS_PER_NODE=<provisioned count> \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=inject \
  actor_rollout_ref.actor.comm_eff.spectral.inject_gamma=1.0
```
Constraint pins (INVALID run if violated): `ANCHOR_DELAY_K=5` (launcher default 20!), `CLEAN_CADENCE=0`, `ANCHOR_CADENCE=5`.

## CPU unit-test before launch (cheap, catches the math/wiring)
On the box (or laptop) run the existing comm_eff tests + a quick inject sanity:
`python -m pytest tests/workers/comm_eff/test_spectral_filter.py -q` (the
`alpha=1` no-op and shape tests must still pass — inject is an ADDITIVE path that
does not touch `correct_matrix`). Then a 1-step check that `correction_mode=inject`
is accepted by the config merge.
