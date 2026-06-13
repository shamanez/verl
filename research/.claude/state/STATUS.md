# Research Status — 2026-06-13T17:05:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 31 | Surpass the dense baseline (beat-dense program from B2) | PLAN_READY (research:claim) | — | — | hand-off for next session; `runs/EXP-30/beat_dense/` + issue body |
| 30 | Generator-consistent M geometry gate + B1/B2 + controls | **DONE / CLOSED** | torn down (i_40765004) | **PASS** | B2 0.7528 (≈96% dense); 6-run decomposition; PR #17 merged; #31 successor |
| 29 | Anchor on-policy replay | DONE | — | PASS | PR #16 merged; substrate donor |
| 27 | Damped ef_powersgd merger | DONE | — | STOP | lineage closed |
| 26 | EF PowerSGD + Q families | DONE | — | REVISE | ef 0.7210, M6 record |

## EXP-30 final result (current hyperparameters, val@50)
dense same-config **0.7839** (band 0.75–0.78) · **B2 δ-residual 0.7528** (≈96%, near-parity not established) · B1 blend 0.7422 · C2 plain+Q no-merge 0.6300 · C3 frozen-Q 0.0925@25 (killed, no learning). **Q-update = dominant lever (+~0.5); merger +0.123.** Honest savings ~4×.

## Canonical EXP-30 docs
`verdict.md` (record + 6-run decomposition) · `beat_dense/{program,feasibility}.md` (#31 hand-off) · `stepA_gate.md`. Intermediate team syntheses deleted (git history).

## Last tick
2026-06-13T17:05:00+10:00 · running=[] · closed=[30] · planned=[31] · blocked=[]

## Budget
**0 live instances** (all torn down, API-verified). EXP-30 complete.
