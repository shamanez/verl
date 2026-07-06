# research/ — Claude Code research harness for verl

An additive scaffold beside the upstream verl codebase (it never modifies verl).
It drives research issues through a seven-stage lifecycle — file → plan →
[human approve] → launch → monitor → analyze → close — one self-contained
command per stage, surfaced as three phase windows: `/build` → `/plan <N>` →
`/execute <N>`.

## Project in one line

A **communication-efficient, pipeline-parallel GRPO trainer** — Qwen2.5-1.5B-Instruct + GSM8K.
What "done" means and where we are: [`.claude/GOAL.md`](.claude/GOAL.md).

## Canonical docs (single source each; link, don't duplicate)

| For… | Read |
|---|---|
| What "done" means (north-star) | [`.claude/GOAL.md`](.claude/GOAL.md) |
| **The commands** — phases, stages, human-only skills | [`researcher_steps.md`](researcher_steps.md) |
| Operating config — repos, labels, naming, budget, verification policy, compute | [`.claude/project.yaml`](.claude/project.yaml) |
| **The operator manual** (workflow, compute, money, design rationale — human-only, hosted) | [manual](https://claude.ai/code/artifact/33a69614-404b-42fe-9fea-029f1c73d874) |
| Engineering map of the method | [`../CODE_WALKTHROUGH.md`](../CODE_WALKTHROUGH.md) |

## Layout

- `.claude/skills/` — phase entry points (`build`, `plan`, `execute`) + the stage
  commands (`new-issue`, `approve`, `launch`, `monitor`, `analyze`, `close`, `go`,
  `status`) + compute skills (`vast-*`) + human-only `de-bloat`/`codex-verify` +
  the shared `_lib.sh`.
- `.claude/agents/` — leaf subagents the commands dispatch: research-planner,
  experiment-runner, training-log-monitor, analyst, log-writer.
- `.claude/plans/` — `TEMPLATE-fast.md` / `TEMPLATE-deep.md` (+ read-only legacy
  files). The plan itself lives in the GITHUB ISSUE BODY; the local copy is a
  gitignored cache under `.claude/state/plan-cache/`.
- `runs/<N>-<slug>/` — ALL run artifacts (metrics, verdict.md, report.html, run.json);
  `runs/SUMMARY.md` is the durable fold-in record. Nothing run-related lives anywhere else.
- `.claude/state/` — `runs.jsonl` ledger + plan cache (machine state; not versioned).
- `scripts/` — `analyze.py`, `check_budget.py`, `capture_resolved_config.py`, ….

## Quick start

```bash
cd /Users/shamane/Documents/verl/research && claude
/build "does signed_ema α=0.4 hold parity at cadence 10/10?"   # → #N
# fresh window: /plan <N>      (ends at your approve)
# fresh window: /execute <N>   (runs to status:done)
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
