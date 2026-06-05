# Research Status — 2026-06-05T04:38:43+00:00 (log-writer · EXP-23 DONE)

## Active
No active experiments. 0 live Vast.ai instances. No idle GPU bleed.

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 23 | Stale full-grad re-anchor for PowerSGD (delay_K=5 + inject/blend) | DONE | 1×4H200 (i_39447338, torn down 2026-06-05T04:31+10) | STOP | Hypothesis FALSIFIED. max(A2,A3)=0.6967 ≤ floor+0.02=0.7114. cos(G,M)≈0.001 (structural orthogonality). Integration worked. Next lever: EXP-24 error-feedback. Launcher wiring MERGED via shamanez/verl#14 (squash 9edea6105); exp/23 branch deleted. |
| 20 | PowerSGD activation codec (parent) | DONE | — | PASS | A0=0.7415 (r=77 clean@5), dense ceiling=0.7536 — EXP-23 reference points (not re-run) |

## EXP-23 outcome summary
- A1 no-refresh floor: val@50=0.6914
- A2 stale inject (γ=1.0, delay_K=5): val@50=0.6967 (+0.005 vs floor — INERT)
- A3 stale blend (η=0.5, delay_K=5): val@50=0.6861 (−0.005 vs floor — INERT)
- Reference: A0 fresh-clean@5 (EXP-20)=0.7415, dense=0.7536
- Falsification predicate: max(0.6967,0.6861)=0.6967 ≤ 0.6914+0.02=0.7114 → TRUE → STOP
- Mechanism: cos(G_powersgd, M_anchor)≈0.001 (near-orthogonal throughout; PowerSGD r=77 discards the directions M lives in)
- All 6 smoke hard-gate invariants passed; circuits fired (anchor_backwards=20, spectral_corrections=80/arm); codec health green; 0 NaN/OOM

## Milestone M6 progress
- 1 PASS (EXP-20). Milestone summary requires ≥2 PASS entries. Not yet written.
- EXP-23 = STOP; does NOT count toward M6 PASS tally.
- Follow-up: EXP-24 (error-feedback PowerSGD residual + staleness-aware blend η∝1/K) — next M6 candidate.

## Last tick
2026-06-05T04:38:43+00:00 · running=[] · analyzing=[] · logging=[23] · blocked=[]

## Budget
$/hr now: $0.00 (0 instances) · EXP-23 final: ~$100.66 lifetime spend (check_budget.py). Monthly cap $1500 — headroom intact.
