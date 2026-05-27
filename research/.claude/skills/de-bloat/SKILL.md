---
name: de-bloat
description: Fold a COMPLETED experiment into runs/SUMMARY.md and delete its bulky artifacts (run dir incl. *.bundle, plan file, stale handle). Run only after an entire issue is done. Never touches the baseline.
allowed-tools: Bash
---

# de-bloat

Keeps the harness repo lean. After an experiment's issue is **fully done** (verdict
written, PR merged if any, instance torn down), this skill collapses that experiment's
heavyweight footprint into a single concise row in `runs/SUMMARY.md` and removes the rest
— exactly the manual cleanup applied to EXP-4/EXP-5, made repeatable and safe.

## Usage

```
$CLAUDE_PROJECT_DIR/.claude/skills/de-bloat/run.sh EXP-<N> [EXP-<M> ...]
$CLAUDE_PROJECT_DIR/.claude/skills/de-bloat/run.sh --dry-run EXP-<N>     # preview, no writes/deletes
```

Accepts `EXP-5`, `5`, or `EXP-5`-style ids.

## What it does (per id)

1. **Appends a one-line row** to the `runs/SUMMARY.md` table (newest first), derived from
   the plan (`title`, `milestone`), the verdict (`VERDICT: PASS|REVISE|STOP`), and any PR
   number found in `LOG.md`/`PROGRESS.md`. Skipped if a row for that id already exists.
2. **Deletes** `runs/EXP-<N>/` (including the multi-MB `exp.bundle`) and `.claude/plans/<N>.md`
   — `git rm` when tracked, plain `rm` otherwise.
3. **Light tidy:** removes `.claude/state/vast-handles/*.json` for instances already
   `TORN_DOWN` in the ledger (never live ones).

It does **not** commit or push — review `git status` and `runs/SUMMARY.md`, then commit.

## Hard guards (it refuses rather than risk the baseline or live work)

- **Never the baseline.** `baseline` / id `3` / `EXP-3` / `runs/baseline/` / `plans/baseline.md`
  are the permanent dense control and are always left intact.
- **Never a live experiment.** Refuses if the id has a `RUNNING` or `PROVISIONED` ledger row
  (tear it down first — that's `vast-teardown`'s job).
- **Never an undone experiment.** Refuses if the run dir exists but there is no
  `verdict.md` and no `LOG.md` entry for the id.
- **Idempotent.** Re-running on an id whose artifacts are already gone and whose row is
  already in `SUMMARY.md` is a no-op.

## When to run

After the orchestrator has driven an issue to its terminal state (LOG entry written, PR
merged, instance torn down). Typically the last manual step before moving to the next issue.
The ledger rows in `runs.jsonl` are left as historical record; only handle files and bulky
artifacts are removed.
