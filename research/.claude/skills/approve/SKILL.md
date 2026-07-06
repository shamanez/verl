---
name: approve
description: The human gate. Show the plan digest, get an explicit yes/no from the operator, then flip labels automatically (status:planned -> status:approved). The ONLY stage that requires a human decision; the human never touches gh/labels by hand.
argument-hint: "<issue-number>"
allowed-tools: Bash, Read, AskUserQuestion
---

# /approve <N> — human go/no-go, mechanics automated

Approval is a MONEY decision. The decision is the human's; every keystroke
around it is the harness's.

## Steps

1. ```bash
   source .claude/skills/_lib.sh
   plan_exists <N> || die "no plan for #<N> — run /plan <N> first"
   [[ "$(issue_status <N>)" == "planned" ]] || echo "note: #<N> is status:$(issue_status <N>)"
   ```
2. Print a ≤ 20-line digest of `.claude/plans/<N>.md`: hypothesis, tier, kind,
   cells (names + one-line each), success bar, `max_gpu_hr` × `max_dph` worst
   case in dollars, code_change + target_modules.
3. Sanity-check before asking (flag, don't silently pass): success criteria
   machine-checkable? budget sane? cell names readable? `target_modules`
   confined to allowed paths? Numeric hypothesis?
4. Ask the operator (AskUserQuestion): **approve / edit first / reject**.
   - **approve** → `set_status_label <N> approved` → print `Next: /launch <N>  (or /go <N> to drive to completion)`.
   - **edit first** → apply the operator's requested plan edits, re-show digest, ask again.
   - **reject** → ask close-or-keep: close → `gh issue close <N>` +
     `rm .claude/plans/<N>.md`; keep → leave at `status:planned`.
5. Append one PROGRESS line: `[approve] #<N> <decision>`.

## Rules

- NEVER approve autonomously. In an unattended session, print the digest,
  append `AWAITING_APPROVAL: #<N>` to PROGRESS.md, and stop.
- Deep-tier plans: this is the last sanctioned point for adversarial review —
  offer (don't force) an operator-invoked `codex-verify --mode verify --plan
  .claude/plans/<N>.md` pass. After approval, no more review loops anywhere.
