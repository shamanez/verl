---
name: analyze
description: "Judge a finished run: verify the box is down, run verification commands, write verdict.md (PASS/REVISE/STOP), backfill WandB tail, auto-label status:pass|revise|stop. Stage 6. One bounded pass — no verification loops."
argument-hint: "<issue-number>"
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# /analyze <N> — results → verdict

## Preconditions (degrade, never stall)

```bash
source .claude/skills/_lib.sh
row=$(ledger_row_by_issue <N>); id=$(jq -r '.id // empty' <<<"$row")
# GPU-free kinds (and deleted ledgers) have no row — derive the id from the plan:
[[ -z "$id" ]] && { slug=$(plan_field <N> slug); [[ -n "$slug" ]] && id="<N>-$slug"; }
[[ -z "$id" ]] && die "cannot resolve a run id for #<N> (no ledger row, no plan slug) — nothing to analyze"
```
- `kind: analysis` plan (GPU-free): no ledger row is expected — run the plan's
  `## Verification commands` locally, verdict GO=PASS / NO-GO=STOP. Skip all
  box checks.
- Box still `RUNNING`/`PROVISIONED` → refuse: `/monitor <N>` owns it until
  results sync + teardown.
- `runs/<id>/verdict.md` already exists → show it, `Next: /close <N>`, stop
  (idempotent).
- `runs/<id>/` DELETED but issue is `status:pass|stop` → terminal already;
  point at `/close <N>` or the runs/SUMMARY.md row. If not terminal and the run dir is
  gone → `die "results for #<N> were deleted — relaunch or close"`.

## Steps

1. Dispatch ONE `analyst` subagent with `run_id=<id> issue=<N>`. It reads
   `runs/<id>/run.json` + metrics; the plan file only if it still exists
   (success criteria); otherwise the criteria snapshot inside run.json.
2. The analyst applies the **default predicate** unless the plan overrides:
   - **PASS** — every success-criteria box ✓ (a clean symmetric negative that
     the plan declared falsifiable is also PASS).
   - **REVISE** — fixable miss AND revise depth < `iterations` (ledger
     `revise_depth`); must emit ≤ 3 `next_actions {knob, from, to, rationale}`.
   - **STOP** — hypothesis falsified / budget exhausted / depth exhausted /
     unmeasured (a result that wasn't measured is never PASS).
3. WandB tail backfill: last 1–2 steps from `runs/<id>/metrics/train.log` if
   the uploader dropped them (`scripts/backfill_wandb.py`).
4. Label from the verdict: `set_status_label <N> <pass|revise|stop>`.
5. REVISE → file the child issue NOW (`/new-issue` semantics: title
   `REVISE of #<N>: <knob change>`, body = next_actions, `depends_on: [<N>]`),
   bump `revise_depth` on the ledger row, append the durable flag
   `progress "REVISE_CHILD: #<child> from #<N> — needs /plan"` (so /status and
   unattended goal-judges can see it), and print `Next: /plan <child>`.
6. Print verdict + evidence lines (metric=value vs target, source file) +
   `Next: /close <N>`.

## Hard rules

- ONE analyst pass. No adversarial re-verification of the verdict, no
  multi-analyst fan-out, no "let me double-check with another agent". If the
  verdict looks untrustworthy, run `flag_human <N> "verdict doubt: <doubt>"`
  and stop — the human decides whether to re-run analysis or invoke
  `codex-verify`.
- Never invent numbers; every verdict value must be greppable from
  `runs/<id>/metrics/` or `analysis.log`.
- Divergence (NaN/exploding grad-norm) in metrics ⇒ STOP with the step number.
