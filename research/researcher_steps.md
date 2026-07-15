# researcher_steps.md — command index

Full operator manual (workflow, prompting examples, compute, money, design doctrine):
**https://com-eff-rlvr.pages.dev/harness/** (the site's Autonomous Harness tab;
mirror: https://claude.ai/code/artifact/33a69614-404b-42fe-9fea-029f1c73d874).
Human-only — agent-facing truth is `.claude/project.yaml` +
`.claude/skills/*/SKILL.md`, which this index and the manual link to, never restate.

## Three phases, one FRESH window each

```
/build "does signed_ema α=0.4 hold parity at cadence 10/10?" [kind:…]   → files #N
/plan <N> [deep]              # plan → written INTO the issue body → ends at YOUR approve
/execute <N> [--gpu auto|ask] # launch → monitor → analyze → close → status:done
```

`kind:` is optional — explicit wins, else inferred (experiment | ablation |
implementation | analysis | brainstorm | literature); it decides the lifecycle
shape, so state it whenever inference could guess wrong.

`--gpu auto` (default, project.yaml `default_compute.gpu_mode`) = hands-off:
implementation + CPU gates finish on the laptop, then the harness provisions
by itself. `--gpu ask` = same laptop-side preparation, then the issue pauses
with a **READY FOR GPU** comment until you hand over a box
(`/execute <N> --attach <id>`) or authorize provisioning
(`/execute <N> --gpu auto`). Either way a GPU is never up while code is being
written (`verification.gpu_idle_rule`).

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
| launch | `/launch <N>` (`--gpu auto\|ask`, `--attach <id>`, `--account team\|private`) | none (ask-mode: hand over a box when flagged) |
| watch | `/monitor <N>` | none (cheap machine-monitor watches; Opus classifier on anomaly; flags `needs:human` if a call is needed) |
| judge | `/analyze <N>` | none |
| finish | `/close <N>` | none (close comment = the verdict record; report page published to the site; checkboxes ticked; local footprint auto-cleaned) |

## Where results live (nothing about a finished run stays local)

- **Verdict SSOT**: the issue close comment.
- **Report page**: https://com-eff-rlvr.pages.dev/runs/ — one HTML page per
  issue, auto-published + pushed by /close (the push is the Cloudflare Pages
  deploy). Small artifacts land in the report repo's gitignored
  `artifacts/<id>/`; big ones in R2 under
  `autonomous-harness-rlvr-compression/<id>/` (project.yaml `reports:`).
- **Local**: `runs/SUMMARY.md` (one row per issue — the offline fallback) +
  `PROGRESS.md` (THE one session file: capped tick echo + end-of-session
  checklist). LOG.md is retired.

## Overview / control

```bash
/status                                  # fleet table, printed (no file)
jq -c . .claude/state/runs.jsonl         # the ledger
python scripts/check_budget.py --month   # spend vs project.yaml caps
bash .claude/skills/vast-teardown/run.sh <id>            # manual teardown
bash .claude/hooks/install-reaper-cron.sh --status       # hourly money backstop
touch ~/.claude-kill-switch              # instant pause (rm to resume)
```

Pause signals are LABELS on the issue: `needs:human` (reason in a comment —
incl. the `--gpu ask` READY-FOR-GPU pause), `awaiting:approval`. A refusing
stage prints the named reason + the next command — that is the design, not an
error.

## Human-only skills (never run by the loop)

- `/de-bloat <id>|--all-terminal` — manual batch fallback for leftover
  dirs (the normal path is /close's automatic cleanup sweep,
  `scripts/close_cleanup.sh`). Requires `DEBLOAT_OPERATOR_ACK=1`; invariant
  test: `scripts/test_debloat_invariant.sh`.
- `codex-verify` — external plan review, planning-time only.

One-time prerequisites: `~/.config/verl-research/secrets.env` (`-rw-------`),
`gh repo set-default --view` → `shamanez/verl-compression-research`,
`which claude gh vastai uv aws`,
`bash .claude/hooks/check-workspace.sh` (sessions MUST open in research/ of the
CURRENT checkout — agents/hooks/backstops register from there), and after any
checkout move: `bash .claude/hooks/install-reaper-cron.sh` (re-stamps the cron
on the new path; `--status` warns on drift).
