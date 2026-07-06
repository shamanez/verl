---
name: close
description: "Finish an issue: verify teardown, write LOG + SUMMARY, commit deliverables to the issue branch, open a PR with results and merge it, label status:done, close the issue. Stage 7 — the single exit point for every issue."
argument-hint: "<issue-number>"
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# /close <N> — verdict → landed + torn down + logged

## Preconditions (graceful with deleted artifacts)

```bash
source .claude/skills/_lib.sh
st=$(issue_status <N>); kind=$(plan_field <N> kind experiment)
case "$st" in
  pass|stop|revise) ;;
  done) echo "#<N> already done"; exit 0 ;;
  approved) case "$kind" in brainstorm|literature|implementation) ;;   # plan/PR IS the deliverable — no verdict stage
            *) die "#<N> is status:approved — /launch or /analyze first";; esac ;;
  *) die "#<N> is status:$st — /analyze <N> first (close needs a verdict)";;
esac
row=$(ledger_row_by_issue <N>); id=$(jq -r '.id // empty' <<<"$row")
[[ -z "$id" ]] && { slug=$(plan_field <N> slug); [[ -n "$slug" ]] && id="<N>-$slug"; }
[[ -z "$id" ]] && id="issue-<N>"   # last resort: plan deleted too — close from labels/LOG only
```
- Verdict file missing but label is terminal → proceed using the label + LOG
  as the source (deleted-run degradation); note it in the PR body.
- `kind: brainstorm|literature|implementation` → plan (+ any PR) IS the
  deliverable: commit + PR it, label done, close. No box ever existed.

## Steps

1. **Teardown check first (money before paperwork).** If the ledger row is
   `RUNNING|PROVISIONED` → run `vast-teardown` now; verify `TORN_DOWN`. No
   harness box may outlive its issue. **`EXTERNAL` rows are operator-managed:**
   do NOT auto-destroy — ask (interactive) or append
   `MANUAL_REVIEW_NEEDED: external box <id> still up after /close #<N> — tear
   down when done` and continue (the box may serve other work).
2. Dispatch ONE `log-writer` subagent with `run_id=<id> issue=<N>`. It owns:
   LOG.md prepend, `runs/SUMMARY.md` row, REPRODUCIBILITY + launcher promotion
   from `resolved_params.txt` (PASS only), and the branch/PR/merge below.
3. **Branch + PR + merge — every issue with something to land:**
   - Ensure `exp/<id>` exists (create from `origin/vast-ai-workload` if
     /launch never ran — e.g. analysis kinds).
   - Commit deliverables to it: plan file, `runs/<id>/verdict.md` +
     `report.html` + `run.json` + `resolved_params.txt`, LOG/SUMMARY deltas,
     any code/launcher changes. Bulk artifacts stay gitignored.
   - `git diff origin/vast-ai-workload..exp/<id>` empty → skip PR, log
     `PR_SKIPPED: #<N> nothing to land`, continue.
   - Else: push; `gh pr create --repo <project.yaml github.code_repo>
     --base vast-ai-workload --head exp/<id>` with a body carrying: verdict,
     the results table (metric | value | target | source), box/cost line from
     the ledger, WandB group link. Then `gh pr merge --squash --delete-branch`.
     Merge conflict/failure → leave the PR open, log
     `MANUAL_REVIEW_NEEDED: PR merge #<N>`, continue to labels (never hang on
     git).
4. `set_status_label <N> done` and `gh issue close <N> --comment "<verdict
   one-liner + PR link + runs/<id>/>"`.
5. Print: what landed, PR URL, teardown confirmation, cost total.
   Optionally suggest (never run) `/de-bloat <id>` — that skill is human-only.

## Hard rules

- code PRs go to `shamanez/verl` base `vast-ai-workload` — NEVER upstream,
  NEVER base main.
- Idempotent: re-running must not duplicate LOG entries, PRs, or comments.
- This stage never re-opens analysis or re-litigates the verdict.
