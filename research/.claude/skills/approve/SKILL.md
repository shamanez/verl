---
name: approve
description: The human gate. Show the plan digest (fetched from the GitHub issue body), get an explicit yes/no from the operator, then flip labels automatically (status:planned -> status:approved). The ONLY stage that requires a human decision; the human never touches gh/labels by hand.
argument-hint: "<issue-number>"
allowed-tools: Bash, Read, AskUserQuestion
---

# /approve <N> — human go/no-go, mechanics automated

Approval is a MONEY decision. The decision is the human's; every keystroke
around it is the harness's. Normally reached from inside `/plan <N>` (same
window); invokable standalone for a re-gate.

## Steps

1. ```bash
   source .claude/skills/_lib.sh
   plan_fetch <N> || die "no plan block in issue #<N> — run /plan <N> first"
   [[ "$(issue_status <N>)" == "planned" ]] || echo "note: #<N> is status:$(issue_status <N>)"
   ```
   `plan_fetch` first, always — a human may have edited the plan block on
   GitHub since it was written; GitHub is the plan's SSOT.
2. Print a ≤ 20-line digest of `$(plan_path <N>)`: hypothesis, tier, kind,
   cells (names + one-line each), success bar, `max_gpu_hr` × `max_dph` worst
   case in dollars, code_change + target_modules.
3. Sanity-check before asking (flag, don't silently pass): success criteria
   machine-checkable? budget sane? cell names readable? `target_modules`
   confined to allowed paths? Numeric hypothesis?
4. Ask the operator (AskUserQuestion): **approve / edit first / reject**.
   - **approve** → `set_status_label <N> approved && clear_human_flags <N>` →
     print `Next: open a FRESH window → /execute <N>`.
   - **edit first** → apply the operator's requested edits to
     `$(plan_path <N>)`, `plan_publish <N> "$(plan_path <N>)"`, re-show the
     digest, ask again. (Edits the operator already made on GitHub are picked
     up by the `plan_fetch` — never overwrite them with a stale cache.)
   - **reject** → ask close-or-keep: close → `gh issue close <N>` (the plan
     block stays in the closed issue as the record); keep → leave at
     `status:planned`.
5. Append one PROGRESS line: `[approve] #<N> <decision>`.

## Rules

- NEVER approve autonomously. In an unattended session, print the digest,
  run `flag_awaiting_approval <N>` (durable label + PROGRESS echo), and stop.
- Deep-tier plans: this is the last sanctioned point for adversarial review —
  offer (don't force) an operator-invoked `codex-verify --mode verify --plan
  $(plan_path <N>)` pass. After approval, no more review loops anywhere.
