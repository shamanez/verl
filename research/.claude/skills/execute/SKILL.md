---
name: execute
description: "Phase-3 entry point — drive an APPROVED issue through launch → monitor → analyze → close in one window. Refuses pre-approval states (those belong to /plan). Resumable: detects the stage from labels + ledger, like /go. `--gpu auto|ask` picks hands-off provisioning vs pause-for-operator-box."
argument-hint: "<issue-number> [--gpu auto|ask] [--attach <instance-id | \"ssh …\">] [--account team|private]"
allowed-tools: Bash, Read, Glob, Grep, Agent, Skill, AskUserQuestion
---

# /execute <N> — approved plan → status:done (phase 3 of 3)

Run this phase in a FRESH window (the /plan window is closed; context stays
small). Everything /execute needs travels via labels + ledger + the issue
body; nothing depends on the planning window's context.

## Flags (forwarded verbatim to /launch)

- `--gpu auto` — hands-off: when the runner's PREPARE phase (implementation +
  CPU gates, laptop-only) is green, it provisions by itself and runs to done.
- `--gpu ask` — prepare everything, then PAUSE: the issue gets a
  `needs:human` "READY FOR GPU" comment and this window stops. You come back
  with `/execute <N> --attach <id>` (your own box — login registered via
  vast-attach) or `/execute <N> --gpu auto` to let it provision.
- `--attach <id>` — use an operator-provided box (implies no provisioning).
  Accepts a bare Vast instance-id **or** the full SSH login string (quoted:
  `--attach "ssh -i <key> -p <port> root@<host> …"`) — the endpoint is parsed
  from it, no reverse-lookup loop.
- No flag → the plan's `gpu_mode:` key, else `project.yaml
  default_compute.gpu_mode`. Resolution: CLI > plan > project default.

## Preconditions

```bash
source .claude/skills/_lib.sh
st=$(issue_status <N>)
```
- `st` empty or `planned` → refuse:
  `#<N> is not approved — run /plan <N> (it ends at the human gate)`.
  /execute NEVER plans and NEVER approves.
- `has_human_flag <N>` → interactive: show the latest `needs:human` comment,
  ask the operator. A **READY FOR GPU** flag + a `--attach`/`--gpu auto` flag
  on THIS invocation counts as resolved: `clear_human_flags <N>` and continue
  straight into /launch. Otherwise resolved → clear and continue; else stop.
  Unattended: stop (unless the READY-FOR-GPU + flag case above applies).
- `st == done` → print the issue close comment; nothing to do.

## Steps

Run the `/go <N>` dispatch loop (Skill: go) from wherever labels + ledger say
the issue is — from `approved` that is: `/launch` (with the flags above) →
`/monitor` → `/analyze` → `/close` (analysis kinds skip straight to
`/analyze`; implementation/brainstorm/literature to `/close`). Terminal
states: `status:done` + box `TORN_DOWN`, a READY-FOR-GPU pause (`--gpu ask`),
or a `needs:human` pause.

## Hard rules

- Same as /go: single-shot stages, ledger-bounded retries, never loop on a
  refusal, GPU never sits unwatched.
- GPU-idle rule (project.yaml `verification.gpu_idle_rule`): implementation
  finishes on the laptop BEFORE any box exists — in both gpu modes.
- GPU-occupancy rule (project.yaml `verification.gpu_occupancy_rule`): the
  moment a box exists, getting/keeping training running IS the priority —
  fix→relaunch cycles preempt all deferrable laptop work (target < 30 min
  box-idle per incident); it is fine for issues to surface on the box, fix
  them fast and iterate. Never pad occupancy with a known-broken config.
- One issue per window. Parallel issues = parallel windows, each in its own
  worktree (`claude --worktree <N>-<slug>`).
