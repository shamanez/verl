---
name: de-bloat
description: "HUMAN-ONLY: fold a COMPLETED experiment into runs/SUMMARY.md and delete its bulky artifacts (run dir, plan cache, stale handles). Never invoked by the autonomous loop — deleting science is a deliberate operator act."
disable-model-invocation: true
allowed-tools: Bash
---

# /de-bloat <id> [id …] | --all-terminal [--dry-run] — operator-only artifact pruning

Collapses a finished experiment's footprint into one `runs/SUMMARY.md` row and
deletes the rest. **This is the manual batch FALLBACK for leftover
dirs** — the normal path is /close's own cleanup sweep
(`scripts/close_cleanup.sh`), which does this per-issue automatically once the
record is durably published. **Two independent human-only gates:**
1. `disable-model-invocation: true` — the model can never auto-fire this skill;
   only the operator typing `/de-bloat` triggers it.
2. `run.sh` refuses without `DEBLOAT_OPERATOR_ACK=1`. When (and only when) the
   operator invoked this skill themselves, run:
   ```bash
   DEBLOAT_OPERATOR_ACK=1 bash .claude/skills/de-bloat/run.sh <id> [--dry-run]
   # or the batch form — folds EVERY terminal run in one pass:
   DEBLOAT_OPERATOR_ACK=1 bash .claude/skills/de-bloat/run.sh --all-terminal [--dry-run]
   ```
   An autonomous session must NEVER set that variable — it may only *suggest*
   the command in its close-out summary.

`--all-terminal` enumerates every `runs/*/` dir and lets the per-id guards
decide: live-ledger, pending-work, and baseline ids are refused with a printed
reason, so exactly the terminal runs fold. Run `--dry-run` first, read the
list, then run it for real.

## Ids

Accepts the canonical `<N>-<slug>` (e.g. `61-math-ablation`), older `EXP-44`
/ `44` ids, and named dirs (`MOAT-45-ANALYSIS`) — matched verbatim against
`runs/`, word-bounded against SUMMARY (EXP-4 never matches EXP-44).

## What it does (per id)

1. Appends one `| id | date | verdict | headline | issue | PR |` row to
   `runs/SUMMARY.md` (from plan → run.json fallbacks; idempotent — same
   format the log-writer uses at /close).
2. Deletes `runs/<id>/` and the derived plan cache
   `.claude/state/plan-cache/<N>.md` (the plan's SSOT is the GitHub issue
   body — nothing is lost; rich detail lives on the report page,
   project.yaml `reports:`).
3. Tidies `.claude/state/vast-handles/*.json` for TORN_DOWN instances.

Does NOT commit — review `git status`, then commit.

## The invariant this relies on (tested)

**A terminal issue's preconditions never require its run dir.** Every stage
derives state labels + ledger FIRST; `runs/<id>/` is evidence, not state.
`scripts/test_debloat_invariant.sh` proves it hermetically (run it after
touching _lib.sh or this skill): post-deletion, the ledger row still resolves,
`snapshot_get`/`plan_field` degrade to defaults, SUMMARY carries the record,
and re-running de-bloat is a no-op.

## Refusals (guards, not errors)

- The baseline (`baseline`/`3`/`EXP-3`) — permanent control.
- Any id with a live ledger row (RUNNING / PROVISIONED / EXTERNAL).
- Any id with no verdict.md AND no SUMMARY row (pending work).

## When

After `/close <N>` is fully done (status:done, PR merged, box TORN_DOWN) and
you no longer need the raw artifacts. The harness stays fully functional after
deletion: every stage command degrades gracefully via labels + ledger +
SUMMARY (that is a tested design property, not luck).
