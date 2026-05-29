# research/ — long-running Claude Code + Codex research harness

This subtree is an **additive scaffold** sitting beside the upstream verl codebase. It does NOT modify verl. The harness orchestrates research experiments (hypothesis → plan → codex-verify → Vast.ai launch → analysis → finding) on a fleet of three long-running Claude Code sessions plus a Codex bridge.

The setup is **task-agnostic** — what research it runs is decided per-issue by you, not by any pinned doc.

## Start here

1. Read **`researcher_steps.md`** — the single operator manual. It walks you through the two-phase workflow (planning vs implementation), the hybrid gate, the M0 smoke test, the compute profile, codex-bridge, and troubleshooting.
2. Skim the **design plan** at `/Users/shamane/.claude/plans/first-go-throuhg-the-deep-pearl.md` — that is the contract this directory implements.
3. Inspect **`.claude/agents/`** — five leaf-subagent definitions (research-planner, codex-bridge, experiment-runner, analyst, log-writer). The two coordinator workflows (`triage`, `orchestrator`) live under **`.claude/playbooks/`** and are executed at the top level of the `/loop` session, because Claude Code subagents cannot spawn other subagents.

## Don't touch

- `../verl/`, `../verl/.claude/`, `../verl/.codex/`, `../verl/.agent/`, `../AGENTS.md`, `../pyproject.toml`, `../setup.py` — those belong to upstream. The `protect-upstream.sh` PreToolUse hook enforces this at the harness level. (`../CLAUDE.md` is fork-specific now; edit it from the repo root, not from here.)

The project **north-star** — what "done" means — lives at [`.claude/GOAL.md`](.claude/GOAL.md). The harness is issue-first: every operating fact an agent needs lives in this `research/` tree.

## How to start the loop

```bash
cd /Users/shamane/Documents/verl/research

# Session A — planning watcher
claude   # then in-session:
/bg /loop 60m Read .claude/playbooks/triage.md and execute it.

# Session B — autonomous executor
claude   # then in-session:
/bg /loop 30m Read .claude/playbooks/orchestrator.md and execute it.

# Optional Session C — milestone goal
/goal milestone M<N> has >=2 PASS experiments AND research/findings/M<N>/SUMMARY.md exists
```

See `researcher_steps.md` for the full procedure, the human-gate semantics, the kill switch, and troubleshooting.

## Kill switch

```bash
touch ~/.claude-kill-switch
# (resume)
rm ~/.claude-kill-switch
```
