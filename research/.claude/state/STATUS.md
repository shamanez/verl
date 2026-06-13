# Research Status — 2026-06-13 (EXP-31 surpass-dense drive)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 31 | Surpass dense via stale-anchor sub-basis merger | RUNNING (Cell A) + Cell D code in flight | 1×4H200 (i_40806688, OPERATOR box) | — | Cell A = B2-reproduce substrate control, monitored; Cell D merger being implemented on exp/31 (CPU); geometry done (r_sb=2) |
| 28 | EXP-28 TRUE error-feedback successor | PLAN_READY? (kind:experiment, no status label) | — | — | not approved; out of scope this drive |

## EXP-31 cell progress
- **Cell A (B2 reproduce)** — RUNNING on operator box 40806688 (4×H200). tmux `exp-31-104_202_252_41`, ledger row EXP-31 owns heartbeat. 15/15 controlled-variable assertions PASS (delayed_ef λ=1, β_anc=0, r=77 act, cadence=delay_K=5, clean=0, replay, snapshot cpu, seed 0, 50 steps, test_freq=25). In vLLM-init window. **monitor-cellA** active (bg).
- **ANALYSIS (geometry sizing)** — DONE. `runs/EXP-31/geometry_sizing.md`: stable-rank 1.93 (53.6% ≤2) → r_sb=2; off-principal energy 0.682 (F1 reproduced); honest-byte denom 3.70×; Cell C r_δ=16 then 8 (only beats 3.70× if compressed δ REPLACES full-rank anchor-M traffic).
- **Cell D (headline merger)** — code being implemented on `exp/31-subbasis-merger` (CPU/worktree, parallel). Design locked in `runs/EXP-31/cellD_design.md`: δ_subbasis = rank-r_sb SVD of δ_B2 (tail) added additively into delayed_ef; weight-gradient realization (per-target, forward Q untouched ⇒ Step-C avoidance automatic; rank-0 = B2 bitwise). **runner-cellD-impl** active (bg).
- **Cell C (savings)** — pending (blocked on Cell A; optional).
- **Cell F (certification)** — pending (blocked on Cell A + Cell D production).

## Last tick
2026-06-13 · running=[31 Cell A] · implementing=[31 Cell D] · analyzing=[] · blocked=[31 Cell C/F]

## Budget
$/hr now: $14.27 (1× operator 4×H200) · max_gpu_hr cap 96 · OPERATOR BOX — no teardown without operator confirmation.
