# researcher_steps.md — command index

One issue = one lifecycle = seven commands, each `/<command> <issue-number>`,
each self-contained (labels applied automatically — you never touch `gh`).
Design + rationale: [`.claude/HARNESS_DESIGN.md`](.claude/HARNESS_DESIGN.md);
all config: [`.claude/project.yaml`](.claude/project.yaml).

| stage | command | what it does | your involvement |
|---|---|---|---|
| file | `/new-issue "<one-liner>"` | issue + `research:claim` + `kind:*` labels | describe the experiment |
| plan | `/plan <N>` (add `deep` for multi-stage) | plan file + `status:planned` + stub comment | answer questions if asked — planning is where deliberation belongs |
| gate | `/approve <N>` | digest → your yes/no → `status:approved` | **the one decision that is yours** |
| launch | `/launch <N>` (`--attach <id>` for your own box) | branch `exp/<N>-<slug>`, provision-or-attach, launch, `status:running` | none |
| watch | `/monitor <N>` | bounded watch cycles → teardown the moment results sync | none (pauses + flags if a human call is needed) |
| judge | `/analyze <N>` | verdict.md, WandB backfill, `status:pass\|revise\|stop` | none |
| finish | `/close <N>` | teardown check, LOG+SUMMARY, PR with results → merge, `status:done`, issue closed | none |

**Drive it all with one command:** `/go <N>` — detects the current stage from
labels + ledger and runs everything remaining, pausing only at `/approve` or a
`MANUAL_REVIEW_NEEDED`. Simple experiment end-to-end:

```
/new-issue "does signed_ema α=0.4 hold parity at cadence 10/10?"   → #64
/go 64        # plans, waits for your approve, then runs to done
```

**Days-long unattended run** (after you've approved):

```
/bg /goal Issue 64 is terminal: status:done, box TORN_DOWN, LOG entry — or
PROGRESS.md has AWAITING_APPROVAL/MANUAL_REVIEW_NEEDED/STUCK for #64. Each
turn run /go 64 once and print stage + evidence. Stop after 120 turns.
```

**Parallel issues:** one session per issue, each in its own worktree —
`claude --worktree 64-ema-sweep` — zero cross-contamination; all sessions
share the primary checkout's ledger automatically.

## Overview / control

```bash
/status                      # fleet table: issue | stage | box | burn | next command
tail -30 PROGRESS.md         # append-only audit; grep MANUAL_REVIEW_NEEDED|STUCK|AWAITING_APPROVAL
jq -c . .claude/state/runs.jsonl                     # the ledger
python scripts/check_budget.py --month               # spend vs project.yaml budget caps
bash .claude/skills/vast-teardown/run.sh <id>        # manual teardown
touch ~/.claude-kill-switch                          # instant pause (rm to resume)
```

## Human-only skills (never run by the loop)

- `/de-bloat <run-id>` — fold a DONE experiment into runs/SUMMARY.md and delete
  its artifacts. Requires `DEBLOAT_OPERATOR_ACK=1`; the model cannot self-invoke.
- `codex-verify` — external plan/derivation review, planning-time only.

## Failure modes

| symptom | fix |
|---|---|
| a stage refuses | it printed the named reason + the right next command — that's the design, not an error |
| plan or `runs/<id>/` was deleted | commands degrade to labels+ledger; `/go <N>` tells you what's recoverable |
| box not torn down | `bash .claude/skills/vast-teardown/run.sh <instance_id>` |
| `MANUAL_REVIEW_NEEDED` in PROGRESS.md | read the line, decide, re-run `/go <N>` |
| attached box needed for analysis only | `vast-attach --manual` → status EXTERNAL (never auto-reaped; you own teardown) |

One-time prerequisites: `~/.config/verl-research/secrets.env` (`-rw-------`),
`gh repo set-default --view` → `shamanez/verl-compression-research`,
`which claude gh vastai uv`.
