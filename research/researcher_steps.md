# researcher_steps.md — command index

Full operator manual (workflow, compute, data flow, money, design rationale):
**https://claude.ai/code/artifact/33a69614-404b-42fe-9fea-029f1c73d874**
(hosted HTML — human-only; agent-facing truth is `.claude/project.yaml` +
`.claude/skills/*/SKILL.md`, which this index and the manual link to, never restate).

## Three phases, one FRESH window each

```
/build "does signed_ema α=0.4 hold parity at cadence 10/10?"   → files #N
/plan <N> [deep]    # plan → written INTO the issue body → ends at YOUR approve
/execute <N>        # launch → monitor → analyze → close → status:done
```

Parallel issues: one window per issue, each in its own worktree
(`claude --worktree <N>-<slug>`); all sessions share the primary checkout's
ledger automatically. `/go <N>` is the resume-from-anywhere fallback
(unattended `/bg /goal` form: manual appendix).

## The 7 stages underneath (each `/<command> <issue>`, labels automatic)

| stage | command | your involvement |
|---|---|---|
| file | `/new-issue "<one-liner>"` | describe the experiment |
| plan | `/plan <N>` (`deep` for multi-stage) | answer questions — deliberation belongs here |
| gate | `/approve <N>` | **the one decision that is yours** |
| launch | `/launch <N>` (`--attach <id>`, `--account team\|private`) | none |
| watch | `/monitor <N>` | none (flags `needs:human` if a call is needed) |
| judge | `/analyze <N>` | none |
| finish | `/close <N>` | none (close comment = the verdict record) |

## Overview / control

```bash
/status                                  # fleet table, printed (no file)
jq -c . .claude/state/runs.jsonl         # the ledger
python scripts/check_budget.py --month   # spend vs project.yaml caps
bash .claude/skills/vast-teardown/run.sh <id>            # manual teardown
bash .claude/hooks/install-reaper-cron.sh --status       # hourly money backstop
touch ~/.claude-kill-switch              # instant pause (rm to resume)
```

Pause signals are LABELS on the issue: `needs:human` (reason in a comment),
`awaiting:approval`. A refusing stage prints the named reason + the next
command — that is the design, not an error.

## Human-only skills (never run by the loop)

- `/de-bloat <id>|--all-terminal` — fold terminal runs into `runs/SUMMARY.md`,
  delete artifacts. Requires `DEBLOAT_OPERATOR_ACK=1`; invariant test:
  `scripts/test_debloat_invariant.sh`.
- `codex-verify` — external plan review, planning-time only.

One-time prerequisites: `~/.config/verl-research/secrets.env` (`-rw-------`),
`gh repo set-default --view` → `shamanez/verl-compression-research`,
`which claude gh vastai uv`.
