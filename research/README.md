# research/ — Claude Code research harness for verl

An additive scaffold beside the upstream verl codebase (it never modifies verl). It runs
research experiments — hypothesis → plan → [human gate] → Vast.ai launch → analysis → finding —
across two long-running Claude Code `/loop` sessions (planning + execution). What it researches is
decided **per issue**, not by any pinned doc.

## Project in one line

A **communication-efficient, pipeline-parallel GRPO trainer** — Qwen2.5-1.5B-Instruct + GSM8K.
What "done" means and where we are: [`.claude/GOAL.md`](.claude/GOAL.md).

## Canonical docs

| For… | Read |
|---|---|
| What "done" means (north-star) | [`.claude/GOAL.md`](.claude/GOAL.md) |
| Operating config — repos, secrets, vast template, compute defaults, branch policy | [`.claude/project.yaml`](.claude/project.yaml) |
| Operator manual — workflow, human gate, kill switch, troubleshooting | [`researcher_steps.md`](researcher_steps.md) |
| Engineering map of the method | [`../CODE_WALKTHROUGH.md`](../CODE_WALKTHROUGH.md) |

## Layout

- `.claude/agents/` — leaf subagents: research-planner, experiment-runner, training-log-monitor, analyst, log-writer.
- `.claude/playbooks/` — the two coordinator loops (triage, orchestrator), run at the top level of a `/loop` session.
- `.claude/plans/` — `TEMPLATE.md` + one `<issue>.md` plan per claimed issue.
- `runs/` — per-experiment runtime artifacts + the `runs.jsonl` ledger (created by the harness).
- `scripts/` — `analyze.py`, `check_budget.py`, `diff_against_baseline.py`.

## Start the loop

```bash
cd /Users/shamane/Documents/verl/research
# Session A — planning watcher
claude   # then:  /bg /loop 60m Read .claude/playbooks/triage.md and execute it.
# Session B — executor
claude   # then:  /bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it.
```

The planner writes a plan and labels the issue `status:planned`; **you** review it and flip to
`status:approved`; the orchestrator then drives provisioning → training → verdict → log entry.

## Kill switch

```bash
touch ~/.claude-kill-switch   # pause all agent tool calls
rm ~/.claude-kill-switch      # resume
```

## Don't touch

`../verl/`, `../verl/.claude/`, `../verl/.codex/`, `../verl/.agent/`, `../AGENTS.md`,
`../pyproject.toml`, `../setup.py` — upstream-owned; the `protect-upstream.sh` PreToolUse hook
enforces it. (`../CLAUDE.md` is fork-specific — edit it from the repo root, not from here.)
