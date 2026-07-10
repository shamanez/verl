---
name: plan
description: Phase-2 entry point — write the plan INTO the GitHub issue body (fast tier default, deep for multi-stage research), then continue straight into the /approve human gate. Auto-labels status:planned. Planning is where all questions, brainstorming and adversarial review belong.
argument-hint: "<issue-number> [deep]"
allowed-tools: Bash, Read, Glob, Grep, Agent, Skill, AskUserQuestion
---

# /plan <N> [deep] — issue → plan (in the issue body) → human gate

Run this phase in a FRESH window. It ends either **approved** (interactive)
or parked at **`awaiting:approval`** (unattended). The plan's single source
of truth is the **GitHub issue body**, between `<!-- plan:start -->` /
`<!-- plan:end -->` markers; local files under `.claude/state/plan-cache/`
are a gitignored derived cache (offline reads + worktrees).

## Preconditions (check, then act — never loop)

```bash
source .claude/skills/_lib.sh
gh issue view <N> --json title,body,labels,url,state   # must be OPEN
plan_fetch <N> && echo "plan already present — digest only unless asked to re-plan"
```
- Issue closed → refuse: `issue #<N> is closed`.
- `status:pass|stop|done` label → refuse: `#<N> is terminal — file a new issue`.
- Plan block already present → show its digest; only rewrite if the operator
  asked to re-plan. `plan_publish` replaces ONLY the marked block — the claim
  text and any human notes outside the markers are never touched. Human edits
  made to the plan block ON GitHub are authoritative: every stage re-fetches,
  so never overwrite them silently.

## Tier choice

- **fast** (default): single hypothesis, ≤ ~6 cells, one launch round.
  Template: `.claude/plans/TEMPLATE-fast.md`.
- **deep**: operator passed `deep`, or the issue is genuinely multi-stage
  (sequential gated stages / days-long / wide design space).
  Template: `.claude/plans/TEMPLATE-deep.md`.

Plan length is **not capped** — the issue body holds ~64 KB and plans use a
fraction of it (`plan_publish` refuses only if the assembled body nears that
real GitHub limit, so it never silently truncates). Write what the plan needs:
keep it scannable and cut filler/duplication, but **never** trim safety-gate
content (money gates, silent-failure contracts) to make it shorter.

## Steps

1. Dispatch ONE `research-planner` subagent:
   ```
   You are research-planner for issue #<N>, tier=<fast|deep>.
   Follow .claude/agents/research-planner.md. Draft to $(plan_path <N>),
   then plan_publish <N> "$(plan_path <N>)".
   ```
2. **Deep tier only — this is the sanctioned place for heavy deliberation:**
   before dispatching, you MAY run a judge-panel workflow (draft 2–3 approaches
   → score → synthesize) and hand the winner to the planner; you MAY ask the
   operator clarifying questions. Never do any of this after approval.
3. After the planner returns, verify + label:
   ```bash
   plan_fetch <N> || { progress "STUCK: planner published no plan for #<N>"; exit 1; }
   set_status_label <N> planned
   ```
4. Print a ≤ 20-line plan digest, then **continue into the gate**: invoke the
   `approve` skill (`/approve <N>`) in this same window — planning and the
   human decision are one phase. Unattended: print the digest,
   `flag_awaiting_approval <N>`, and STOP.

## Rules

- ONE planner dispatch per invocation. Planner fails → log
  `STUCK: planner #<N> <reason>` to PROGRESS.md and stop. `/plan <N>` again is
  the retry — a human decision, not a loop.
- Cell names in the plan must pass `lint_cell_name` (no c1/armA opacity).
- Ablations must name a PASSed parent in `depends_on`.
- NO stub comment — the plan IS the issue body (the 2026-06-12 "stub comment"
  directive is retired). Issue comments stay terse: verdicts, `needs:human`
  reasons, close-out one-liners.
