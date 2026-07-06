---
name: plan
description: Write the plan for a research issue (fast tier by default, deep tier for multi-stage research). Stage 2 — auto-labels status:planned. Planning is where all questions, brainstorming and adversarial review belong.
argument-hint: "<issue-number> [deep]"
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# /plan <N> [deep] — issue → plan file

## Preconditions (check, then act — never loop)

```bash
source .claude/skills/_lib.sh
gh issue view <N> --json title,body,labels,url,state   # must be OPEN
```
- Issue closed → refuse: `issue #<N> is closed`.
- `status:pass|stop|done` label → refuse: `#<N> is terminal — file a new issue`.
- Plan file already exists → say so and show its digest; only rewrite if the
  operator asked to re-plan (a plan may carry human edits — never clobber silently).

## Tier choice

- **fast** (default): single hypothesis, ≤ ~6 cells, one launch round.
  Template: `.claude/plans/TEMPLATE-fast.md`. Target ≤ 4 KB.
- **deep**: operator passed `deep`, or the issue is genuinely multi-stage
  (sequential gated stages / days-long / wide design space).
  Template: `.claude/plans/TEMPLATE-deep.md`. Target ≤ 15 KB.

## Steps

1. Dispatch ONE `research-planner` subagent:
   ```
   You are research-planner for issue #<N>, tier=<fast|deep>.
   Follow .claude/agents/research-planner.md. Write .claude/plans/<N>.md.
   ```
2. **Deep tier only — this is the sanctioned place for heavy deliberation:**
   before dispatching, you MAY run a judge-panel workflow (draft 2–3 approaches
   → score → synthesize) and hand the winner to the planner; you MAY ask the
   operator clarifying questions. Never do any of this after approval.
3. After the planner returns, verify + label:
   ```bash
   plan_exists <N> || { progress "STUCK: planner produced no plan for #<N>"; exit 1; }
   set_status_label <N> planned
   ```
4. Post a STUB comment on the issue (≤ 15 lines: plan path, cell list, budget,
   success bar — NEVER the full plan body).
5. Print the plan digest + `Next: /approve <N>`.

## Rules

- ONE planner dispatch per invocation. Planner fails → log
  `STUCK: planner #<N> <reason>` to PROGRESS.md and stop. `/plan <N>` again is
  the retry — a human decision, not a loop.
- Cell names in the plan must pass `lint_cell_name` (no c1/armA opacity).
- Ablations must name a PASSed parent in `depends_on`.
