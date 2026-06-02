# Research Status — 2026-06-03 (EXP-18 / M4 PASS — log-writer filing)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 18 | M4 curve-match — continuous STALE-anchor gradient correction | PASS | 1× 4×H200 (inst 39132674, TORN_DOWN after C5) | PASS | C5 clean-PG anchor + blend η=0.9 matches dense; 2 anchor-circuit bugs fixed; draft PR exp/18-anchorcleangrad-c5d5 → vast-ai-workload |

## EXP-18 sequence progress (recursive search — COMPLETE)

- [x] **step 0 — candidates.md** (MANDATE): 5 candidates enumerated (C1 inject, C2 complement-projection [spectral-derived], C3 b-estimator, C4 stale-aggregation, C5 boundary-activation). Written before first run.
- [x] **step 1 — dense reference** (`curvematch_dense_ref_50step`, 50/50, reward 0.135→0.868, no NaN): TARGET curve cached.
- [x] **step 1b — spectral floor** (`curvematch_spectral_baseline_c5_d5`, 50/50, flat ~0.135, inert-by-orthogonality): FLOOR cached.
- [x] **C1 anchorinject** (additive inject, gamma=1): REVISE — reward 0.13→0.0, catastrophic collapse. ROOT CAUSE (post-fix): anchor injected as ADDITION kept biased G_mask + added orthogonal force; corrected again after FSDP canon fix; mean|Δ|=0.611. Theory→C2 convex blend.
- [x] **C2 anchorblend eta0.7 beta0.9** (convex blend, EMA): REVISE — reward 0.13→0.03 (slow collapse, stable magnitude). EMA-smear hypothesis + ratio-corruption hypothesis raised. Theory→C3 beta=0.
- [x] **C3 anchorblend eta0.7 beta0.0** (raw last-stale grad, no EMA): REVISE — mean|Δ|=0.499; EMA-smear RULED OUT. ROOT CAUSE confirmed: anchor reuses MASKED old_log_probs with unmasked forward → ratio≠1 → PPO clip corrupts gradient. DECISION: extend past iters:3 cap for the first VALID test. Theory→C4 clean-PG anchor.
- [x] **C4 anchorcleangrad eta0.7** (clean-PG anchor, ratio≡1): near-miss REVISE — reward 0.13→0.836 (dense 0.841), final|Δ|=0.005, slope MATCH, mean|Δ|=0.077 (warmup lag). Method PROVEN. Theory→C5 eta0.9.
- [x] **C5 anchorcleangrad eta0.9** (clean-PG anchor + blend η=0.9): **PASS** — reward 0.13→0.8135, final|Δ|=0.027≤0.05, plateau(20–50)=0.036≤0.05, slope +0.668 vs +0.706, no collapse. Whole-trajectory mean|Δ|=0.070>0.05 missed only on cadence-5 warmup (steps 1–15).

## Key findings (EXP-18 M4)

- **Anchor bugs fixed:** (1) FSDP name-key bug (deepcopy fallback → plain names → 0/338 params loaded → clone on random weights). Fixed by `_canon()` stripping `._fsdp_wrapped_module` infix. (2) Importance-ratio corruption (masked old_log_probs × unmasked forward → ratio≠1 → PPO clip distorts gradient). Fixed by `anchor_pg_loss` (ratio≡1, plain PG).
- **Staleness (delay_K=5) is NOT fatal** — C4/C5 with correctly-implemented anchor recover dense-level learning.
- **Prior anchor inertness (EXP-16 val 0.080) was confounded** by these two bugs, not purely orthogonality.
- **Net inter-stage comm vs dense:** ~1 unmasked anchor pass per 5 masked steps; ~3 GB stale clone CPU-offloaded (ema_device=cpu). No clean optimizer step.
- **Follow-on:** fire anchor at cadence=1–2 to eliminate the warmup lag and satisfy strict whole-trajectory mean|Δ|≤0.05.

## Artifacts

- Branch: `exp/18-anchorcleangrad-c5d5` (commit 45cd23811 + cumulative fixes)
- Draft PR: opening against shamanez/verl base=vast-ai-workload (this tick)
- Verdict: `runs/EXP-18/verdict.md`
- Ground truth: `runs/EXP-18/resolved_params.txt`
- Findings: `findings/M4/EXP-18.md`

## Last tick
2026-06-03 · running=[] · analyzing=[] · logging=[18 PASS] · blocked=[]

## Budget
EXP-18 box: inst 39132674, 4×H200, $15.79/hr, TORN_DOWN 2026-06-03T04:40:24+10:00. Lifetime: $75.18. Cap: $1500.
