# Research Status — 2026-05-30T03:12:18+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 16 | Short-run stability matrix (mask/rescale/clean/spectral) | RUNNER_PLANNING (lead approval gate) | — | — | Operator authorized lead-led launch. Team `exp-16-stability` created; 12 tasks seeded (T0–T7,A5,V,PR, deps wired). Runner spawned (background) under propose-then-wait gate — awaiting its execution+code-change plan; lead approves before ANY spend/code change. Monitor/auditor/synthesizer dispatched just-in-time per the lead-log execution model. |
| 11 | M3 — 100-step M95+AP GRPO vs dense baseline | NOT_CLAIMED | — | — | `kind:experiment`, `milestone:M3`, but no `research:claim`, no `status:*`, no plan. Triage/planning domain — out of orchestrator scope. |
| 10 | M3 — DP gradient compression (PowerSGD/DiLoCo) scope | NOT_CLAIMED | — | — | `kind:experiment`, `milestone:M3`, no `research:claim`/`status:*`/plan. Triage/planning domain — out of orchestrator scope. |

`baseline` (dense control, `.claude/plans/baseline.md`) is a design template, not a gating EXP-run — `#16 depends_on: []`.

## Last tick
2026-05-30T03:12 · running=[] · analyzing=[] · logging=[] · blocked=[16 → operator decision] · skipped=[11,10 unclaimed]

## Budget
$/hr now: $0 (no active Vast.ai instance) · spent today: $0 · #16 budget cap: ≤ $60 total, ≤ $24/hr/instance, ≤ 96 GPU-hr, ≤ 12 h wall-clock

## Notes
- Agent teams enabled: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`, claude 2.1.156 (≥ 2.1.32 minimum met).
- Kill switch clear (`~/.claude-kill-switch` absent).
- gh default repo: `shamanez/verl-compression-research` (issue queue). Code PRs target `shamanez/verl` base `vast-ai-workload`.
