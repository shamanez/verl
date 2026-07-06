---
name: de-bloat
description: "HUMAN-ONLY: fold a COMPLETED experiment into runs/SUMMARY.md and delete its bulky artifacts (run dir, plan file, stale handles). Never invoked by the autonomous loop — deleting science is a deliberate operator act."
disable-model-invocation: true
allowed-tools: Bash
---

# /de-bloat <id> [id …] [--dry-run] — operator-only artifact pruning

Collapses a finished experiment's footprint into one `runs/SUMMARY.md` row and
deletes the rest. **Two independent human-only gates:**
1. `disable-model-invocation: true` — the model can never auto-fire this skill;
   only the operator typing `/de-bloat` triggers it.
2. `run.sh` refuses without `DEBLOAT_OPERATOR_ACK=1`. When (and only when) the
   operator invoked this skill themselves, run:
   ```bash
   DEBLOAT_OPERATOR_ACK=1 bash .claude/skills/de-bloat/run.sh <id> [--dry-run]
   ```
   An autonomous session must NEVER set that variable — it may only *suggest*
   the command in its close-out summary.

## Ids

Accepts the canonical `<N>-<slug>` (e.g. `61-math-ablation`), legacy `EXP-44`
/ `44`, and legacy slug dirs (`MOAT-45-ANALYSIS`) — matched verbatim against
`runs/`, word-bounded against LOG/SUMMARY (EXP-4 never matches EXP-44).

## What it does (per id)

1. Appends one `| id | milestone | what | result | merged |` row to
   `runs/SUMMARY.md` (from plan → run.json → LOG fallbacks; idempotent).
2. Deletes `runs/<id>/` and the plan file `.claude/plans/<N>.md`.
3. Tidies `.claude/state/vast-handles/*.json` for TORN_DOWN instances.

Does NOT commit — review `git status`, then commit.

## Refusals (guards, not errors)

- The baseline (`baseline`/`3`/`EXP-3`) — permanent control.
- Any id with a live ledger row (RUNNING / PROVISIONED / EXTERNAL).
- Any id with no verdict.md AND no LOG entry (pending work).

## When

After `/close <N>` is fully done (status:done, PR merged, box TORN_DOWN) and
you no longer need the raw artifacts. The harness stays fully functional after
deletion: every stage command degrades gracefully via labels + ledger +
SUMMARY (that is a tested design property, not luck).
