---
name: status
description: "One-screen read-only overview printed to stdout: every open research issue with its stage, pause labels, live boxes with burn rate, untracked-box leak check, and recent PROGRESS flags. Writes NOTHING — there is no status file to go stale."
argument-hint: "[issue-number]"
allowed-tools: Bash, Read, Glob, Grep
---

# /status [N] — where is everything?

Read-only. Never dispatches, never labels, never tears down, never writes a
file — the fleet view is derived fresh from labels + ledger every time
(a cached STATUS.md would only go stale; it was deleted deliberately).

```bash
source .claude/skills/_lib.sh
gh issue list --state open --json number,title,labels \
  -q '.[] | select(.labels[].name=="research:claim") | [.number, .title, ([.labels[].name | select(startswith("status:") or startswith("kind:") or .=="needs:human" or .=="awaiting:approval")] | join(","))] | @tsv'
[[ -f "$LEDGER" ]] && jq -c 'select(.status=="RUNNING" or .status=="PROVISIONED" or .status=="EXTERNAL")' "$LEDGER"
bash .claude/skills/vast-cost/run.sh          # burn + leak check (read-only)
tail -15 "$PROGRESS" | grep -E 'MANUAL_REVIEW_NEEDED|STUCK|AWAITING_APPROVAL|TEARDOWN_FAILED' || true
```

Print one table: `#N | title | status | kind | pause | box (id, $/hr, age) | next command`.
The "pause" column shows `needs:human` / `awaiting:approval` labels; the
"next command" column is /go's dispatch table applied per issue.

With an argument, show one issue in depth: labels, ledger row, run dir
contents (or "deleted"), verdict digest, WandB group link.
