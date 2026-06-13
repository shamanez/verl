# Cell D design — stale-anchor rank-2 sub-basis, additive into δ (weight-gradient realization)

**Status:** design locked by orchestrator 2026-06-13. Headline greedy-surpass bet.
**Mandate:** surpass dense 0.7839 (greedy mean@1, beyond pooled-SE 0.020), comm-eff LOCKED (PowerSGD r=77 act forward codec untouched).

## The mechanism (one line)
Amplify the dominant low-rank "true direction the act-basis structurally misses,"
harvested from the **stale anchor gradient**, added **additively** into the B2
correction term — forward codec Q untouched.

```
δ_B2(t)       = M_rep − G_comp_ring(t−K)            # existing B2 residual (refreshed at fires, held between)
δ_subbasis(t) = rank_{r_sb}( S )                    # NEW: top-r_sb SVD of the source S
G_corr(t)     = G_comp + λ·( δ_B2 + δ_subbasis )    # λ=1, β_anc=0 (both LOCKED)
```
- **family=tail (default):** `S = δ_B2`. δ_B2 = (uncompressed stale grad) − (act-compressed stale grad on the SAME batch,θ) = the **act-deflated weight gradient** = exactly the off-act-principal direction (the weight-space realization of the plan's `tail = G_b − P_Qact(G_b)`). Its top-`r_sb` SVD is the dominant direction the act codec drops.
- **family=grad (REVISE fallback):** `S = M_rep` (raw stale anchor gradient top-`r_sb`, not deflated).
- `r_sb ∈ {2,4}`, default **2** (geometry_sizing.md: median stable-rank 1.93, 53.6% of targets ≤2, off-principal energy 0.682, coherent at K=5).

## Why this realization (deviation from the plan's literal activation-family vehicle — justified)
The plan said "reuse `_compute_family_V('tail')/_build_family_Q`" (activation-space, per-boundary Q). The wiring read shows that route is **infeasible-as-written**:
- Family Q is built per **boundary layer** (`decoder_boundary_indices`, 8 boundaries) in **activation space** (H×r). The merger corrects **196 weight matrices** (`q,k,v,o,gate,up,down_proj` × 28). 8↔196 is not 1:1, and the H-axis match differs per matrix (q_proj [1536,1536] vs down_proj [1536,8960]) → axis ambiguity. `build_and_dump_family_sketches` currently only **dumps** the family Q; it is never consumed by the per-target merger.
- The merger ALREADY has, per target, the two tensors whose difference IS the act-deflated weight gradient (`M_rep`, `G_comp_ring`). Harvesting the tail in **weight-gradient space, per target** is the natural fit for a weight-gradient merger, needs **no** boundary↔target mapping, and is what actually amplifies the optimizer update direction.

This realization is **more faithful in effect** (it amplifies the weight-update direction the codec misses) and **strictly safer**: it touches NO forward path, so Step-C avoidance is automatic (forward Q checksum == act-only run, trivially — we never read or write Q). If it nulls, the REVISE ladder is family tail→grad, then r_sb 2→4; the activation-family route is the last resort.

## Correctness invariants — satisfied BY CONSTRUCTION
- **off-path parity (hard):** `delta_subbasis_rank=0` ⇒ the sub-basis branch is **skipped entirely** (not computed-then-zeroed) ⇒ `G_corr = G_comp + λ·δ_B2` = B2 bitwise. New knobs default OFF. ✓
- **limiting-case identity (hard):** `λ=0` keeps the existing FIRST early-return (`return g_comp`, same object) — sub-basis logic is after it, never reached. ✓
- **Step-C avoidance (hard):** implementation is confined to `delayed_ef_matrix` (weight-grad space). Forward `q_basis='act'` and `self._basis` are never read/written ⇒ recon_rel_error stays in the act band, forward Q checksum identical. ✓ (STRONGER than the plan's passive-family approach.)
- **detached / no-STE (hard):** `M_rep` (anchor, detached) and `ring_grad` (detached CPU fp32) carry no autograd history; δ_subbasis = SVD of a detached tensor → detached. The forward projector is untouched. ✓
- **determinism / multi-rank agreement (hard):** δ_B2 is identical across DP ranks (M_rep DP-MEAN-reduced, ring FSDP-mean under same agg_loss). The randomized SVD MUST use a **fixed-seed generator** (e.g. `torch.Generator().manual_seed(powersgd_seed ⊕ target_salt)`) so `svd_lowrank` returns bit-identical columns on every rank. ✓ (assert via the existing cross-rank check on a probe.)
- **scale contract (#25 mean-vs-sum, hard):** δ_subbasis is built from δ_B2 / M_rep, which already honor DP-mean; the SVD applies no rescaling. ✓ Add a unit test mirroring `test_delayed_ef_exp30`.
- **backend/memory (hard):** rank-2 `torch.svd_lowrank(S, q=r_sb, niter=2)` per target = a few small matmuls; tensors are transient + freed. No new persistent GPU state (the ring stays CPU fp32). Re-confirm max_mem < 30.77 at 50 steps on the on-box probe.

## Code locations (blast radius = 2 files + launcher; NO transformer_impl.py / powersgd_activation.py needed)
1. `verl/workers/comm_eff/spectral_filter.py`
   - `__init__`: add `delta_subbasis_rank: int = 0`, `delta_subbasis_family: str = "tail"`; store; seed the per-target SVD generator off `base_seed`/powersgd seed.
   - `delayed_ef_matrix`: after `delta` is resolved (line ~513), **if `self.delta_subbasis_rank > 0`**: `S = delta` (tail) or `anc` (grad); `delta_sb = self._subbasis_delta(name, S, r_sb)`; `correction = delta + delta_sb` else `correction = delta`; `g_corr = gm + lam * correction`. Add a `_subbasis_delta(name, S, r)` helper: deterministic low-rank reconstruction `U[:,:r] diag(s[:r]) V[:,:r]ᵀ` via seeded `svd_lowrank`; fp32; detached; shape-guarded (skip+count if degenerate).
   - Add counters `delayed_ef_subbasis_applied` + a per-fire scalar `subbasis_energy_ratio = ‖δ_subbasis‖/‖δ_B2‖` (median) on the existing `[comm_eff][EXP-30][delayed_ef]` log line.
2. `verl/workers/config/comm_eff.py`
   - `SpectralConfig`: add `delta_subbasis_rank: int = 0`, `delta_subbasis_family: str = "tail"` (+ `r_delta: int = 0` for Cell C, default 0 = OFF). `__post_init__`: validate rank ≥ 0, family ∈ {tail, grad}, r_delta ≥ 0.
   - Wherever `SpectralFilter(...)` is constructed from config, pass the two (three) new knobs through.
3. Launcher `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
   - Add `${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK:-0}`, `${..._FAMILY:-tail}`, `${COMM_EFF_SPECTRAL_R_DELTA:-0}` passthroughs to the Hydra arg list (OFF defaults preserve every existing run).

## CPU tests to add/extend (run in the worktree, no GPU — gate the push)
- off-path parity: rank=0 path returns the exact B2 `g_corr` tensor (reuse `test_delayed_ef_exp30` fixtures); byte-compare.
- limiting-case: λ=0 still returns the `g_comp` object identity.
- sub-basis math: `_subbasis_delta(S, r=full)` reconstructs S to <1e-5; rank-2 on a known rank-2 S is exact; output detached.
- determinism: two generators with the same seed give identical columns; assert cross-"rank" identity on a synthetic δ.
- scale contract: feeding a SUM-reduced S inflates ‖δ_subbasis‖ by world_size (mirrors `test_delayed_ef_exp30`).

## On-box probe (AFTER Cell A frees the box) — the hard gates that need a GPU
1–2 steps with `delta_subbasis_rank=2 family=tail`: assert recon_rel_error in act band (NOT 0.68), forward-Q checksum == Cell-A act-only run, multi-rank δ_subbasis agreement, no NaN/OOM, max_mem<30.77. Any hard-gate fail ⇒ STOP+fix on branch.

## Run
Cell D production = Cell A's B2 config + `COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK=2 COMM_EFF_SPECTRAL_DELTA_SUBBASIS_FAMILY=tail`, 50 steps, test_freq=25, seed 0. Headline: best val@50 ≥ 0.79; certified by Cell F (seed-mean − dense-mean > 0.020).
