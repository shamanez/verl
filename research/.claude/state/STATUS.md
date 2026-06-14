# Research Status — 2026-06-14 (EXP-31 closed — PARITY accepted)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 31 | Surpass dense via stale-anchor sub-basis merger | DONE / PARITY (operator-accepted) | 1×4H200 (i_40806688, OPERATOR box — stopped Vast-side, NOT destroyed) | PARITY | dense-here 0.7506; B2 0.7400 = statistical parity (gap 0.011 < ±0.024 noise); sub-basis = early accelerant only; seed bands deferred (box stopped); branch exp/31-subbasis-merger pushed/unmerged |
| 28 | EXP-28 TRUE error-feedback successor | PLAN_READY? (kind:experiment, no status label) | — | — | not approved; out of scope this drive |

## EXP-31 cell summary (CLOSED)
- **Cell A (B2 reproduce)** — DONE. val@50=0.7400; bytes_ratio=0.0505; max_mem 30.755; vLLM fix (disable_custom_all_reduce) confirmed as controlled variable. W&B p87f8mfr.
- **ANALYSIS (geometry sizing)** — DONE. r_sb=2; off-principal energy 0.682; honest-byte denom 3.70x.
- **Cell D r2 (rank-2 tail sub-basis, constant weight)** — DONE. val@50=0.6983 (below B2 0.7400); val@25=0.7293 (+0.036 vs B2 early). Over-amplifies near convergence — regresses. W&B dxy1ba36.
- **Cell D γ-decay50 (rank-2 tail, γ decays 1→0 over 50 steps)** — DONE. val@50=0.7210; fixes regression but tempers early gain. W&B p3vrcxdq.
- **Dense rerun (THIS config, seed 0)** — DONE. val@50=0.7506. DENSE REFRAME: the 0.7839 ref was a different box. B2 (0.7400) = PARITY with dense-here (gap 0.011 < ±0.024).
- **Cell D hold25-decay25 (γ=1 through step 25 then ramp to 0)** — PARTIAL. val@25=0.7066; val@50 lost to Vast-side box stop (~23:13 UTC). Box 40806688 stopped/held, disk intact.
- **Cell C (savings)** — NOT run (no time after parity accepted).
- **Cell F (seed certification)** — NOT run (box stopped; bands deferred per operator).

## Box status
- **i_40806688** (EXP-31 operator box, 4×H200): Vast-side STOPPED (not destroyed); restart-queued; disk intact. NOT torn down per operator decision. Potential band-pin if box restarts.

## Last tick
2026-06-14 · running=[] · analyzing=[] · logging=[31 closed PARITY] · blocked=[]

## Budget
Experiment closed. No active billing box (i_40806688 stopped, not running).
