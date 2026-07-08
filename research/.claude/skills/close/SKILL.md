---
name: close
description: "Finish an issue: verify teardown, publish the run report to the reports site, write the SUMMARY row, commit deliverables to the issue branch, open a PR with results and merge it, tick the plan's checkboxes, label status:done, close the issue, then CLEAN UP the local footprint. Stage 7 — the single exit point for every issue."
argument-hint: "<issue-number>"
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# /close <N> — verdict → landed + torn down + published + cleaned

## Preconditions (graceful with deleted artifacts)

```bash
source .claude/skills/_lib.sh
st=$(issue_status <N>); kind=$(plan_field <N> kind experiment)
case "$st" in
  pass|stop|revise) ;;
  done) ;;   # NOT a bare exit — a crash between steps can leave a done label
             # with an unposted comment / unpublished report / uncleaned dir.
             # Resume the post-verdict tail idempotently: skip to step 4 and
             # run only what is missing (comment absent → post it; page absent
             # → publish; leftovers present → cleanup). All present → print
             # "#<N> already done" and stop.
  approved) case "$kind" in brainstorm|literature|implementation) ;;   # plan/PR IS the deliverable — no verdict stage
            *) die "#<N> is status:approved — /launch or /analyze first";; esac ;;
  *) die "#<N> is status:$st — /analyze <N> first (close needs a verdict)";;
esac
row=$(ledger_row_by_issue <N>); id=$(jq -r '.id // empty' <<<"$row")
[[ -z "$id" ]] && { slug=$(plan_field <N> slug); [[ -n "$slug" ]] && id="<N>-$slug"; }
[[ -z "$id" ]] && id="issue-<N>"   # last resort: plan deleted too — close from labels/SUMMARY only
```
- Verdict file missing but label is terminal → proceed using the label + the
  SUMMARY row / issue thread as the source (deleted-run degradation); note it
  in the PR body.
- `kind: brainstorm|literature|implementation` → plan (+ any PR) IS the
  deliverable: commit + PR it, label done, close. No box ever existed.

## Steps

1. **Teardown check first (money before paperwork).** If the ledger row is
   `RUNNING|PROVISIONED` → run `vast-teardown` now; verify `TORN_DOWN`. No
   harness box may outlive its issue. **`EXTERNAL` rows are operator-managed:**
   do NOT auto-destroy — ask (interactive) or run `flag_human <N> "external
   box <id> still up after /close — tear down when done"` and continue (the
   box may serve other work).
2. Dispatch ONE `log-writer` subagent with `run_id=<id> issue=<N>`. It owns:
   the `runs/SUMMARY.md` row, REPRODUCIBILITY + launcher promotion from
   `resolved_params.txt` (PASS only), and the branch/PR/merge below. (The
   report publish is NOT its job — it happens in step 5, after the close
   comment exists, because the report page renders that comment.)
3. **Branch + PR + merge — every issue with something to land** (BASE = the
   project.yaml `code_pr_base_branch`, resolved the way log-writer does —
   never a hardcoded branch name):
   - Ensure `exp/<id>` exists (create from `origin/$BASE` if /launch never
     ran — e.g. analysis kinds).
   - Commit deliverables to it: `runs/<id>/verdict.md` + `run.json` +
     `resolved_params.txt`, SUMMARY delta, any code/launcher changes. The
     plan is NOT a file deliverable — it lives in the issue body. Bulk
     artifacts stay gitignored (they go to R2 via the publish script).
   - `git diff origin/$BASE..exp/<id>` empty → skip PR, log
     `PR_SKIPPED: #<N> nothing to land`, continue.
   - Else: push; `gh pr create --repo <project.yaml github.code_repo>
     --base $BASE --head exp/<id>` with a body carrying: verdict,
     the results table (metric | value | target | source), box/cost line from
     the ledger, WandB group link. Then `gh pr merge --squash --delete-branch`.
     Merge conflict/failure → leave the PR open, `flag_human <N> "PR merge
     failed"`, continue to labels (never hang on git).
4. **Tick the plan's checkboxes** — a done issue must never show unticked
   boxes (operator directive 2026-07-08). For every success-criteria /
   progress box the verdict proves ✓: `plan_tick <N> "<literal substring of
   the line>"` (index() match, NOT a regex — raw checkbox text is safe), or
   `plan_tick <N>` with no pattern when the verdict is PASS-all-green.
   Criteria the verdict marks ✗ stay UNTICKED and are named in the close
   comment.
5. **Close, label, publish — in THIS order (each sub-step idempotent):**
   a. If the issue is not already CLOSED with a `VERDICT` comment:
      `gh issue close <N> --comment` — body = `VERDICT:` line, headline
      results (criterion | observed | target), cost line from the ledger
      (gpu-hr × $/hr), PR link, **report-page link**
      (`<reports.site_url>/runs/<id>.html` — the URL is deterministic, so it
      can be stated before the page is written), WandB group. The close
      comment is the per-issue verdict SSOT.
   b. `set_status_label <N> done` (label AFTER the comment — a crash between
      the two leaves a resumable state, not a stranded one).
   c. **Publish the run report** (now the close comment exists — the page
      renders it): `python3 scripts/publish_run_report.py --issue <N>
      --run-id <id>` → one HTML page in the reports repo (`project.yaml
      reports:`), a card on the Experiment Runs tab, small artifacts to the
      repo's gitignored `artifacts/<id>/`, big ones to R2 `<prefix>/<id>/`,
      then commit+push the report repo (the push IS the Cloudflare Pages
      deploy). **Nonzero exit / `REPORT_PUBLISH_PARTIAL`** →
      `flag_human <N> "report publish failed/partial — <id>"`, SKIP step 6,
      and stop — re-running /close resumes here.
6. **Cleanup sweep (the last step — nothing about a done run stays local).**
   `bash scripts/close_cleanup.sh <N> <id>` — guarded (status:done + no live
   row ACROSS ALL ledger rows + SUMMARY row + report page present), it
   deletes `runs/<id>/`, the plan cache, torn-down handles, compacts the
   ledger, and sweeps the issue's PROGRESS ticks. A guard refusal is a named
   reason to surface, not an error to retry.
7. Print: what landed, PR URL, report URL, teardown confirmation, cost total.

## Hard rules

- code PRs go to `shamanez/verl` base = project.yaml `code_pr_base_branch`
  (this harness line's own branch) — NEVER upstream, NEVER base main.
- Publish BEFORE cleanup: step 6 only runs after the SUMMARY row, close
  comment, and report page all exist — close_cleanup's guards enforce the
  SUMMARY row and report page mechanically; a failed/partial publish stops
  at step 5c.
- Idempotent: re-running must not duplicate SUMMARY rows, PRs, report cards,
  or comments — and a `status:done` re-entry resumes the post-verdict tail
  (see Preconditions) instead of exiting blind.
- This stage never re-opens analysis or re-litigates the verdict.
