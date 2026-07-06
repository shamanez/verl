---
name: execute
description: "Phase-3 entry point — drive an APPROVED issue through launch → monitor → analyze → close in one window. Refuses pre-approval states (those belong to /plan). Resumable: detects the stage from labels + ledger, like /go."
argument-hint: "<issue-number>"
allowed-tools: Bash, Read, Glob, Grep, Agent, Skill, AskUserQuestion
---

# /execute <N> — approved plan → status:done (phase 3 of 3)

Run this phase in a FRESH window (the /plan window is closed; context stays
small). Everything /execute needs travels via labels + ledger + the issue
body; nothing depends on the planning window's context.

## Preconditions

```bash
source .claude/skills/_lib.sh
st=$(issue_status <N>)
```
- `st` empty or `planned` → refuse:
  `#<N> is not approved — run /plan <N> (it ends at the human gate)`.
  /execute NEVER plans and NEVER approves.
- `has_human_flag <N>` → interactive: show the latest `needs:human` comment,
  ask the operator; resolved → `clear_human_flags <N>` and continue; else
  stop. Unattended: stop.
- `st == done` → print the issue close comment; nothing to do.

## Steps

Run the `/go <N>` dispatch loop (Skill: go) from wherever labels + ledger say
the issue is — from `approved` that is: `/launch` → `/monitor` → `/analyze` →
`/close` (analysis kinds skip straight to `/analyze`;
implementation/brainstorm/literature to `/close`). Terminal states:
`status:done` + box `TORN_DOWN`, or a `needs:human` pause.

## Hard rules

- Same as /go: single-shot stages, ledger-bounded retries, never loop on a
  refusal, GPU never sits unwatched.
- One issue per window. Parallel issues = parallel windows, each in its own
  worktree (`claude --worktree <N>-<slug>`).
