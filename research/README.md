# research/ — Claude Code research harness for verl

An additive scaffold beside the upstream verl codebase (it never modifies verl).
It drives research issues through a seven-stage lifecycle — file → plan →
[human approve] → launch → monitor → analyze → close — one self-contained
command per stage, each `/<command> <issue>`.

## Project in one line

A **communication-efficient, pipeline-parallel GRPO trainer** — Qwen2.5-1.5B-Instruct + GSM8K.
What "done" means and where we are: [`.claude/GOAL.md`](.claude/GOAL.md).

## Canonical docs (single source each; link, don't duplicate)

| For… | Read |
|---|---|
| What "done" means (north-star) | [`.claude/GOAL.md`](.claude/GOAL.md) |
| **The commands** — one per stage, failure modes, human-only skills | [`researcher_steps.md`](researcher_steps.md) |
| Operating config — repos, labels, naming, budget, verification policy, compute | [`.claude/project.yaml`](.claude/project.yaml) |
| Architecture + audit rationale (why it's built this way) | [`.claude/HARNESS_DESIGN.md`](.claude/HARNESS_DESIGN.md) |
| Engineering map of the method | [`../CODE_WALKTHROUGH.md`](../CODE_WALKTHROUGH.md) |

## Layout

- `.claude/skills/` — the stage commands (`new-issue`, `plan`, `approve`, `launch`,
  `monitor`, `analyze`, `close`, `go`, `status`) + compute skills (`vast-*`) +
  human-only `de-bloat`/`codex-verify` + the shared `_lib.sh`.
- `.claude/agents/` — leaf subagents the commands dispatch: research-planner,
  experiment-runner, training-log-monitor, analyst, log-writer.
- `.claude/plans/` — `TEMPLATE-fast.md` / `TEMPLATE-deep.md` + one `<issue>.md` per issue.
- `runs/<N>-<slug>/` — ALL run artifacts (metrics, verdict.md, report.html, run.json);
  `runs/SUMMARY.md` is the durable fold-in record. Nothing run-related lives anywhere else.
- `.claude/state/` — `runs.jsonl` ledger + STATUS.md (machine state; not versioned).
- `scripts/` — `analyze.py`, `check_budget.py`, `capture_resolved_config.py`, ….

## Quick start

```bash
cd /Users/shamane/Documents/verl/research && claude
/new-issue "does signed_ema α=0.4 hold parity at cadence 10/10?"
/go <N>          # plans → waits for your /approve → runs to status:done
```

## Kill switch

```bash
touch ~/.claude-kill-switch   # pause all agent tool calls
rm ~/.claude-kill-switch      # resume
```

## Don't touch

`../verl/`, `../verl/.claude/`, `../AGENTS.md`, `../pyproject.toml`, `../setup.py` —
upstream-owned; the `protect-upstream.sh` PreToolUse hook enforces it
(writable only on `exp/*` / `vast-ai-workload` / `autonomous-harness-*` branches).
