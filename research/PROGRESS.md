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
