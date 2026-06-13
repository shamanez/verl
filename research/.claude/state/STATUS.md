# Research Status — 2026-06-13T16:53:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 31 | Surpass the dense baseline (beat-dense program from B2) | PLAN_READY (research:claim) | — | — | hand-off for next session; current-situation + run-order in issue + `runs/EXP-30/beat_dense/` |
| 30 | Generator-consistent M geometry gate + gated B1/B2 + controls | PASS · closing | 1×4H200 (i_40765004, tear down on C3 done) | **PASS** | B2 0.7528 (≈96% dense, near-parity); PR #17 merged. C3 frozen-Q last cell running; close + teardown imminent |
| 29 | Anchor on-policy replay | DONE | — | PASS | PR #16 merged; substrate donor for EXP-30 |
| 27 | Damped ef_powersgd merger | DONE | — | STOP | lineage closed |
| 26 | EF PowerSGD + Q families | DONE | — | REVISE | ef 0.7210, M6 record |

## EXP-30 result (current hyperparameters, val@50)
dense (same-config) ≈ **0.78** (band 0.75–0.78) · best comm-eff **B2 = 0.7528** (≈96%, near-parity not established) · plain no-merge C2 0.6300 (merge-value +0.123) · blend B1 0.7422 · frozen-Q C3 pending. Honest comm savings ~4×.

## Canonical EXP-30 docs (intermediate team syntheses deleted 2026-06-13 → git history)
`verdict.md` (record + findings) · `beat_dense/{program,feasibility}.md` (forward, issue #31) · `stepA_gate.md` (gate).

## Last tick
2026-06-13T16:53:00+10:00 · running=[30:C3] · closing=[30] · planned=[31] · blocked=[]

## Budget
$11.58/hr live (i_40765004, 4×H200) · STRICT: tear down as soon as C3 done.
