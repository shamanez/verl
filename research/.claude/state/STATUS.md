# Research Status — 2026-06-30T22:36+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 43 | Dense regime-A FULL-weight per-tick → R2 | DONE | 1×H200 i_43197578 (team, external) TORN_DOWN | pass | 80/80 steps, 160/160 R2 snapshots verified (verify --r2 PASS, max_rel_norm_err=0.0001), codec OFF (counters all 0), no NaN, WandB a51waqza backfilled. Canonical M4 weight-proj spine root PUBLISHED: s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/. Issue #43 status:pass; close it. |
| 42 | M4 weight-projection accuracy vs horizon | DONE | TORN_DOWN | pass | CLOSED (count-sketch instrument SUPERSEDED by EXP-43 raw full-weights). |
| 41 | M4 look-ahead anchor | DONE | TORN_DOWN | STOP | falsified. |

Downstream M4 weight-proj issues #44–#56 (OPEN) all consume the EXP-43 R2 trace; none launched yet.

## Last tick
2026-06-30T22:36+10:00 · running=[] · analyzing=[] · logging=[] · blocked=[]
EXP-43 PASS logged. Box 43197578 TORN_DOWN (operator-directed, after the teardown gate was satisfied). No live boxes; no leak.

## Budget / box
No live boxes. EXP-43 box 43197578 (1×H200, team) TORN_DOWN at 2026-06-30T22:36+10:00; run cost ~$18.60 (5.08 h × $3.6635/hr). Old box 43190371 (ABORTED relaunch) GONE.

## Deliverable (EXP-43) — DELIVERED
160/160 per-tick FULL-weight bf16 snapshots (ALL params, n_matrices=338, real shapes) in R2 at s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/ (key form tick_<N>/tick_<N>.pt, ~492 GB, R2 ONLY). Manifests (full_manifest.jsonl + r2_manifest.jsonl, 160/160 verified) synced to runs/EXP-43/regimeA/weights/. WandB a51waqza backfilled to step 80. Box torn down. This R2 prefix is the canonical spine root every M4 weight-proj issue (#44–#56) cites.
