# PROGRESS — research orchestration audit log

Append-only. Subagents (triage, planner, runner, analyst, log-writer, codex-bridge) and the Stop hook (`teardown-finished-runs.sh`) write one line per significant event in the format:

```
[<ISO timestamp>] [<agent-name> #<exp-id-if-applicable>] <one-line description>
```

The orchestrator greps this file each tick for these structured markers to route work:

| Marker | Meaning | Routed to |
|---|---|---|
| `STUCK: <ctx>` | runner hit verl-internal it can't resolve | `codex-bridge --mode=code-rescue` |
| `RESCUE_REQUEST: math <ctx>` | planner/analyst needs derivation review | `codex-bridge --mode=math-rescue` |
| `MILESTONE_PASS: M<X>` | log-writer wrote a SUMMARY for a milestone | `codex-bridge --mode=adversarial` |
| `VERIFY_TIMEOUT:` / `BROKER_DIED:` | codex-bridge hit the watchdog | demote plan, ping human |
| `MANUAL_REVIEW_NEEDED: <ctx>` | anything that needs a human (no offers, codex down, ...) | flagged in STATUS, no auto-action |
| `LAUNCH_FAILED: EXP-<N>` / `LAUNCH_FAILED_TIER:` | experiment-runner couldn't launch | depends; usually walks to next tier or stops |
| `BUDGET_EXCEEDED: EXP-<N>` | run blew through `max_gpu_hr` | teardown hook destroys, orchestrator notes |
| `TEARDOWN_FAILED` | Stop hook couldn't destroy an instance | loud warning — operator must check `vastai show instances` |
| `PR_SKIPPED: EXP-<N>` | log-writer refused to draft PR (wrong default repo) | flagged, no auto-action |
| `NEEDS_PLAN: #<N>` | orchestrator saw a `research:claim` issue with no plan | triage owes a plan |

Anything else here is informational. The hook that used to spam `[session-id] Edit` per file write has been removed — only meaningful events are recorded.

---
[2026-05-28T00:45:10+10:00] [reset] Event log cleared for a fresh next-issue session. Prior history is in git (commit 95e8cc38 and earlier): EXP-3 (M1 dense GRPO baseline — PASS, val 0.087→0.789) and EXP-4 (M2 comm_eff no-op scaffolding — merged shamanez/verl#1, issue #4 closed status:done) are complete. Backlog #5–#11 await planning. All Vast instances torn down; runs.jsonl ledger preserved. Carryover follow-up: launcher `vast_baseline_qwen25_1p5b_grpo_gsm8k.sh:196` done.flag bug must be fixed before any multi-cell smoke.
[2026-05-28T00:51:39+10:00] [research-planner #5] plan written
[2026-05-28T00:52:01+10:00] [triage] dispatched 1 planners, 0 issues already planned
[2026-05-28T01:04:41+10:00] [codex-bridge --mode=verify] result=CONCERNS
[2026-05-28T01:30:02+10:00] [experiment-runner #5] launched on 1 instance (38098877, 4xH200, tier_idx=1) dph=14.74; 3 smoke cells (p95/p90/disabled) running on exp/5-actor-mask
[2026-05-28T01:33:13+10:00] [orchestrator] tick: EXP-5 RUNNING (i_38098877 4xH200 $14.74/hr); verify=CONCERNS->VERIFIED; cell p95 in model-load; pre-created done.flag dir on box to keep 3-cell chain from aborting under set -e (launcher bug confirmed unpatched on exp/5)
