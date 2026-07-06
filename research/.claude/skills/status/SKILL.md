---
name: status
description: "One-screen read-only overview: every open research issue with its stage, live boxes with burn rate, untracked-box leak check, and recent PROGRESS flags. Rewrites .claude/state/STATUS.md as a side effect."
argument-hint: "[issue-number]"
allowed-tools: Bash, Read, Glob, Grep
---

# /status [N] — where is everything?

Read-only. Never dispatches, never labels, never tears down.

```bash
source .claude/skills/_lib.sh
gh issue list --state open --json number,title,labels \
  -q '.[] | select(.labels[].name=="research:claim") | [.number, .title, ([.labels[].name | select(startswith("status:") or startswith("kind:"))] | join(","))] | @tsv'
[[ -f "$LEDGER" ]] && jq -c 'select(.status=="RUNNING" or .status=="PROVISIONED" or .status=="EXTERNAL")' "$LEDGER"
bash .claude/skills/vast-cost/run.sh          # burn + leak check (read-only)
tail -15 "$PROGRESS" | grep -E 'MANUAL_REVIEW_NEEDED|STUCK|AWAITING_APPROVAL|TEARDOWN_FAILED' || true
```

Print one table: `#N | title | status | kind | box (id, $/hr, age) | next command`.
The "next command" column is /go's dispatch table applied per issue.

With an argument, show one issue in depth: labels, ledger row, run dir
contents (or "deleted"), verdict digest, WandB group link.

Then overwrite `.claude/state/STATUS.md` with the same table + timestamp
(the durable copy for other sessions; no other stage writes it).
