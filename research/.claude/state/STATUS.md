# Research Status — 2026-05-30

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 16 | Short-run stability matrix (mask/rescale/clean/spectral) | READY_TO_RUN (status:approved) | — | — | Standard orchestrator routing: `experiment-runner` → `training-log-monitor` → `analyst` → `log-writer`. `code_change:true` (branch `exp/16-short-run-stability-matrix`). No run started yet; awaiting the next orchestrator tick. |
| 11 | M3 — 100-step M95+AP GRPO vs dense baseline | NOT_CLAIMED | — | — | `kind:experiment`, `milestone:M3`, but no `research:claim`, no `status:*`, no plan. Triage/planning domain — out of orchestrator scope. |
| 10 | M3 — DP gradient compression (PowerSGD/DiLoCo) scope | NOT_CLAIMED | — | — | `kind:experiment`, `milestone:M3`, no `research:claim`/`status:*`/plan. Triage/planning domain — out of orchestrator scope. |

`baseline` (dense control, `.claude/plans/baseline.md`) is a design template, not a gating EXP-run — `#16 depends_on: []`.

## Last tick
2026-05-30 · running=[] · analyzing=[] · logging=[] · blocked=[] · ready=[16]

## Budget
$/hr now: $0 (no active Vast.ai instance) · spent today: $0 · #16 budget cap: ≤ $60 total, ≤ $24/hr/instance, ≤ 96 GPU-hr, ≤ 12 h wall-clock

## Notes
- Kill switch clear (`~/.claude-kill-switch` absent).
- gh default repo: `shamanez/verl-compression-research` (issue queue). Code PRs target `shamanez/verl` base `vast-ai-workload`.
