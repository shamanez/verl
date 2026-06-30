# Research Status — 2026-06-30T23:00+10:00

## Phase: M4 collection DONE → analysis phase (start in a new session at #44)

EXP-43 (the shared dense full-weight per-tick trajectory) is collected, verified, logged,
and CLOSED. The async-upload speedup is merged to vast-ai-workload (default-on for collection
runs, box-validation pending). No live boxes. The next work is GPU-free analysis: #44 → #56.

## Issue pipeline

| EXP | Title | State | Verdict | Notes |
|---|---|---|---|---|
| 43 | Dense regime-A FULL-weight per-tick → R2 | DONE · **CLOSED** | pass | 80/80, 160/160 R2 snapshots verified, codec OFF, GSM8K 0.7809. Box TORN_DOWN (~$18.60). Trace + index published. |
| 42 | M4 weight-projection accuracy vs horizon | DONE · closed | pass | count-sketch instrument SUPERSEDED by EXP-43 raw full-weights. |
| 41 | M4 look-ahead anchor | DONE · closed | STOP | falsified. |
| 44–56 | M4 weight-proj analysis (sweep engine, predictors, per-layer/block, plots, verdict) | OPEN, unplanned | — | all `kind:analysis` (GPU-free); depend only on the EXP-43 trace; each carries the R2-access-pattern comment. **#44 is the entry point.** |

## Ready for #44 (analysis entry point)
- **Trace (self-describing):** `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/` (`tick_<N>/tick_<N>.pt`, 160 bf16 full-model snapshots, n_matrices=338, ~492 GB R2-only).
- **Index manifests:** in R2 at `.../weights/{full_manifest,r2_manifest}.jsonl` AND tracked in-repo at `runs/EXP-43/regimeA/weights/`.
- **Access discipline (MANDATORY):** stream layer/block-wise → per-layer/block intermediates → combine → HTML; never bulk-download (~492 GB = out-of-disk). Doc: `research/reports/r2-access-pattern-for-analysis.md` (also on every issue #44–#56).
- **Dependency:** #43 PASS + closed → #44–#56 unblocked.
- Certified verdict + provenance kept at `runs/EXP-43/` (verdict.md, manifests, resolved_params, train log); plan file `.claude/plans/43.md` removed (folded into `runs/SUMMARY.md` §Milestone M4); ephemeral monitor/heartbeat/handle noise trimmed.

## Last tick
2026-06-30T23:00+10:00 · running=[] · analyzing=[] · logging=[] · blocked=[]
Close-out: #43 CLOSED, plan removed, manifests co-located in R2, SUMMARY + plans-index updated, async-upload merged to vast-ai-workload. Cleanup complete; ready for the #44 analysis session.

## Budget / box
No live boxes. EXP-43 box 43197578 (1×H200, team) TORN_DOWN; run cost ~$18.60. No leak.
