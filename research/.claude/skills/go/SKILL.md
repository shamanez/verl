---
name: go
description: "Resumable driver: detect where issue N is in its lifecycle from labels + ledger and run every remaining stage to terminal. Pauses only at the approval gate or a needs:human flag. Prefer the phase entry points (/build, /plan, /execute) for day-to-day work; /go is the resume-from-anywhere fallback."
argument-hint: "<issue-number>"
allowed-tools: Bash, Read, Glob, Grep, Agent, Skill, AskUserQuestion
---

# /go <N> — drive one issue to terminal, from wherever it is

Stage detection is labels + ledger FIRST, files second — so it survives
deleted plans/runs and session restarts.

```bash
source .claude/skills/_lib.sh
st=$(issue_status <N>)          # "" | planned | approved | running | pass | revise | stop | done
row=$(ledger_row_by_issue <N>)  # "" | PROVISIONED | RUNNING | EXTERNAL | TORN_DOWN
has_human_flag <N> && echo "PAUSED: needs:human/awaiting:approval label set"
```

**Pause label first:** if `has_human_flag <N>` — interactive: show the latest
`needs:human` comment on the issue, ask the operator whether it is resolved;
resolved → `clear_human_flags <N>` and continue; not resolved → stop.
Unattended: stop immediately (the label is the durable pause signal).

| observed | next stage |
|---|---|
| no `status:*` label | `/plan <N>` |
| `planned` | `/approve <N>` (interactive) — unattended: print digest, `flag_awaiting_approval <N>`, STOP |
| `approved`, no live row | `/launch <N>` (analysis kind → `/analyze <N>`; implementation/brainstorm/literature → `/close <N>`) |
| `approved`/`running`, row RUNNING / PROVISIONED / EXTERNAL | `/monitor <N>` (it handles PROVISIONED-wait and EXTERNAL itself) |
| `running`, row TORN_DOWN, results in `runs/<id>/` | `/analyze <N>` |
| `running`, row TORN_DOWN, NO results, no verdict | `/launch <N>` — its relaunch exception accepts exactly this state, bounded by `launch_attempts ≤ 3` |
| `pass` / `stop` / `revise` | `/close <N>` (a REVISE child was already filed by /analyze — check PROGRESS for `REVISE_CHILD:`) |
| `done` | print the issue close comment; nothing to do |

`depends_on` gates read `pass|stop|done` as terminal (a closed parent shows `done`).

Run stages in order until one of: `status:done`, `awaiting:approval` set,
`needs:human` set, or a stage refuses. Surface the refusal reason verbatim —
a refusal is information for the human, not a retry target.

## Hard rules

- Never skip the approval gate; never self-approve.
- Each stage invocation is single-shot; bounded retries live INSIDE stages
  (ledger counters), not here. /go never loops on a refusing stage.
- One issue per /go. Parallel issues = parallel sessions, each in its own
  worktree (`claude --worktree <id>`); state converges on the primary
  checkout's ledger automatically (`_lib.sh`).

## Appendix — unattended (days-long) form, optional

The default workflow is one-issue-per-window with a human nearby
(`/plan` window → `/execute` window). If you genuinely need a no-human loop:

```
/bg /goal Issue <N> is terminal: status:done and box TORN_DOWN — or the issue
gained a needs:human/awaiting:approval label. Each turn: run /go <N> once,
print status + evidence. Stop after 120 turns.
```

Every /goal condition MUST include the label escape + a turn bound so an
impossible criterion cannot burn money forever.
