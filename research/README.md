# research/ — long-running Claude Code + Codex research harness

This subtree is an **additive scaffold** sitting beside the upstream verl codebase. It does NOT modify verl. The harness orchestrates research experiments (hypothesis → plan → [human gate] → Vast.ai launch → analysis → finding) across two long-running Claude Code `/loop` sessions (planning + implementation), with Codex available as an **operator-invoked** review skill (`codex-verify`) — never on the autonomous path.

The setup is **task-agnostic** — what research it runs is decided per-issue by you, not by any pinned doc.

## The project, in three bullets (read the canonical docs for detail)

A **communication-efficient, pipeline-parallel GRPO trainer** — per-(token,channel) activation masking at the pipeline-stage boundaries; Qwen2.5-1.5B-Instruct + GSM8K.

- **Settled base.** Mask (p=0.9 at the 7 boundaries) **+ rescale (inverted-dropout `1/(1-p)`, ON, permanent) + a true dense "clean" gradient every K steps (`clean_cadence`)**. Mask cross-pass consistency is solved; rescale is settled (its job is unbias, not grad-norm); judge on **val/score, not grad_norm**.
- **Proven result.** masked+clean@K is **stable** (clean-resettable sawtooth, no ratchet) and reaches **dense parity on GSM8K** (0.735 vs 0.741) — but that is **elicitation** (base already 0.715). On harder **Big-Math** it **stalls flat (~0.55)** while dense learns (~0.61): a gradient-fidelity limit, not a missing ceiling. Anchor + spectral *as implemented* did **not** work (GSM8K 0.080, fails by orthogonality) — the clean step is the only lever that worked.
- **Frontier.** Redesign anchor + spectral as a **cheap, continuous surrogate** for the periodic clean step, grounded in the delta-method curvature bias — not the failed anchor-gradient-SVD. Gated by two cheap experiments: the **p-sweep** and the **clean-only ablation**.

**Canonical docs** (one coherent story; start here):

| For… | Read |
|---|---|
| What "done" means + where we are | [`.claude/GOAL.md`](.claude/GOAL.md) |
| Durable run record + result + why + knob surface | [`runs/SUMMARY.md`](runs/SUMMARY.md) |
| The next-cycle plan | [`findings/NEXT_RESEARCH.md`](findings/NEXT_RESEARCH.md) |
| Engineering map of the method | [`../CODE_WALKTHROUGH.md`](../CODE_WALKTHROUGH.md) |

## Start here

1. Read **`researcher_steps.md`** — the single operator manual. It walks you through the two-phase workflow (planning vs implementation), the human gate, the compute profile, the operator-invoked codex review, and troubleshooting.
2. Read **`.claude/GOAL.md`** (the project north-star — what "done" means) and **`.claude/project.yaml`** (the authoritative operating config: repos, secrets, vast template, compute defaults, branch policy). Together they are the contract this directory implements.
3. Inspect **`.claude/agents/`** — five leaf-subagent definitions (research-planner, experiment-runner, training-log-monitor, analyst, log-writer). The two coordinator workflows (`triage`, `orchestrator`) live under **`.claude/playbooks/`** and are executed at the top level of the `/loop` session, because Claude Code subagents cannot spawn other subagents.

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
