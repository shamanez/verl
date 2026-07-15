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

### Burn rate + leak check (`vast-cost/run.sh`, read-only)

Line 19 runs the money-visibility check (was its own `vast-cost` skill until the
2026-07-15 fold — same script, now surfaced through `/status`). It sums
`dph_total` over running instances on BOTH accounts (private + team) and flags any
live instance with no owning live ledger row as `UNTRACKED (possible LEAK)` — the
orphan class that silently bleeds money (a provision orphan, or a teardown that
no-opped under the wrong account). Emits a machine-readable
`VAST_COST: burn_rate_dph=<X> projected_24h_usd=<Y> untracked=<0|1>`. Read-only:
never calls `vastai destroy`, never echoes API keys, and exits 0 even if one
account errors so a single-account hiccup never hides the other's spend. On
`untracked=1`, investigate then `vast-teardown <id>` (it resolves the account).
Run it standalone any time with `bash .claude/skills/vast-cost/run.sh`.

With an argument, show one issue in depth: labels, ledger row, run dir
contents (or "deleted"), verdict digest, WandB group link.
