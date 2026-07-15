# research/ — Claude Code research harness for verl

An additive scaffold beside the upstream verl codebase (it never modifies verl).
It drives research issues through a seven-stage lifecycle — file → plan →
[human approve] → launch → monitor → analyze → close — one self-contained
command per stage, surfaced as three phase windows: `/build` → `/plan <N>` →
`/execute <N>`.

## Project in one line

A **communication-efficient, pipeline-parallel GRPO trainer** —
`Qwen/Qwen2.5-Math-1.5B` on MATH train/test.
What "done" means and where we are: [`.claude/GOAL.md`](.claude/GOAL.md).

## Canonical docs (single source each; link, don't duplicate)

| For… | Read |
|---|---|
| What "done" means (north-star) | [`.claude/GOAL.md`](.claude/GOAL.md) |
| **The commands** — phases, stages, human-only skills | [`researcher_steps.md`](researcher_steps.md) |
| Operating config — repos, labels, naming, budget, verification policy, compute | [`.claude/project.yaml`](.claude/project.yaml) |
| Current MATH/RELEX experiment and selection status | [`../docs/experiments/relex_rank1_report.html`](../docs/experiments/relex_rank1_report.html) |
| **The operator manual** (workflow, compute, money, design rationale — human-only, hosted) | [manual](https://claude.ai/code/artifact/33a69614-404b-42fe-9fea-029f1c73d874) |
| Engineering map of the method | [`../CODE_WALKTHROUGH.md`](../CODE_WALKTHROUGH.md) |

## Layout

- `.claude/skills/` — phase entry points (`build`, `plan`, `execute`) + the stage
  commands (`new-issue`, `approve`, `launch`, `monitor`, `analyze`, `close`, `go`,
  `status`) + compute skills (`vast-*`) + human-only `de-bloat`/`codex-verify` +
  the shared `_lib.sh`.
- `.claude/agents/` — leaf subagents the commands dispatch: research-planner,
  experiment-runner (PREPARE laptop-phase → COMPUTE box-phase), machine-monitor
  (cheap Sonnet health poller — the default watcher), training-log-monitor
  (Opus classifier, dispatched on anomaly), analyst, log-writer.
- `.claude/plans/` — ONLY `TEMPLATE-fast.md` / `TEMPLATE-deep.md`. The plan
  itself lives in the GITHUB ISSUE BODY; the local copy is a gitignored cache
  under `.claude/state/plan-cache/`.
- `runs/<N>-<slug>/` — run artifacts (metrics, verdict.md, run.json) — VOLATILE:
  /close publishes the record to the reports site + R2, then its cleanup sweep
  (`scripts/close_cleanup.sh`) deletes the dir. `runs/SUMMARY.md` (one row per
  issue) is the only durable local index; `PROGRESS.md` is THE one session file.
- `.claude/state/` — `runs.jsonl` ledger + plan cache (machine state; not versioned;
  swept per-issue by /close's cleanup).
- `scripts/` — `analyze.py`, `check_budget.py`, `capture_resolved_config.py`,
  `publish_run_report.py` (report page + artifacts + R2), `close_cleanup.sh`, ….

## Where finished-run results live

Issue close comment (verdict SSOT) → report page on
**https://com-eff-rlvr.pages.dev/runs/** (auto-published by /close; push =
Cloudflare Pages deploy) → big artifacts in R2
(`autonomous-harness-rlvr-compression/<run_id>/`) → small artifacts in the
report repo's gitignored `artifacts/<run_id>/`. Config: `project.yaml reports:`.

## Quick start

```bash
cd <your-checkout>/research && claude   # research/ of the CURRENT checkout (verify: bash .claude/hooks/check-workspace.sh)
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
(writable only on `exp/*` / `autonomous-harness-*` branches).
