# Research Status — 2026-06-25

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 41 | M4 look-ahead anchor (delay_K=20, fixed-linear) | RUNNING (monitor active) | external 4×H200 i_42465843 (team) | — | probe PASSED 10/10 hard invariants; cells A(5/5 disabled)→B(fixed_linear 20/20) launched, 100 steps each, monitor dispatched |

## Last tick
2026-06-25 · running=[41] · analyzing=[] · logging=[] · blocked=[]

## EXP-41 progress
- ✅ Attach (external, team, box 42465843, 4×H200) — RUNNING ledger row written by runner
- ✅ Fire-forcing invariant probe (cadence=delay_K=1, fixed_linear) — ALL 10 hard gates PASS (runs/EXP-41/probe-invariants.md). No commit-hotfix loop needed.
- 🔄 Cell A (onsurface-5over5-reference, lookahead disabled, 100 steps) — running
- ⏳ Cell B (lookahead fixed_linear 20/20, 100 steps) — chained after A
- ⏳ Cell C (learned_linear) — CONDITIONAL: only if analyst finds B clean-but-underlifted
- ⏳ analyst → verdict.md → teardown (team account) → log-writer

## Budget
External operator-provided box (dph recorded 0 on ledger). max_gpu_hr cap 96. Teardown the moment science is captured.
