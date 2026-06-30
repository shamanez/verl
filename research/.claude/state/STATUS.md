# Research Status — 2026-06-30T18:27+10:00

## 🔒 EXP-43 TEARDOWN GATE (operator directive 2026-06-30) — DO NOT TEAR DOWN box 43197578 UNTIL ALL hold:
1. `done.flag` present AND tmux `exp43a` reached step 80/80.
2. R2 holds ~160 `tick_*.pt` objects under `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/`.
3. `r2_manifest.jsonl` is ALL `verified:true` AND `verify_full_weight_dump.py runs/EXP-43/regimeA/weights --r2 --r2-sample 5 --tol 0.01` = PASS.
4. `full_manifest.jsonl` + `r2_manifest.jsonl` synced to `research/runs/EXP-43/regimeA/weights/`.
5. WandB backfilled to final step.
Only THEN: analyst writes verdict.md → log-writer → teardown via vast-teardown skill (team acct).
WHILE RUNNING: do NOT write/let-be-written any `runs/EXP-43/verdict.md` (it is the #1 auto-teardown trigger); refresh `runs/EXP-43/metrics/incoming.log` EVERY tick BEFORE the sweep (stale>30min is the only live auto-trigger). Weights are upload-then-delete, so each tick is already persisted to R2 as it completes; the gate just ensures the FULL set is collected + verified before the box dies.

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 43 | Dense regime-A FULL-weight per-tick → R2 | RUNNING (~5/80) | 1×H200 i_43197578 (team, external) | — | smoke PASSED; 80-step live, ~11 R2 objects climbing, 0 NaN/crash, upload-then-delete OK. Monitor re-dispatched (prior died API-timeout). |
| 42 | M4 weight-projection accuracy vs horizon | DONE | TORN_DOWN | pass | CLOSED. |
| 41 | M4 look-ahead anchor | DONE | TORN_DOWN | STOP | falsified. |

## Last tick
2026-06-30T18:27+10:00 · running=[43 @~5/80] · analyzing=[] · logging=[] · blocked=[]
Monitor re-dispatched (robust: 45s cadence, 25-min cap). Heartbeat fresh (0.8 min). Teardown sweep no-op (healthy box preserved). All auto-teardown triggers verified inert/gated (verdict absent, budget 16h-out + dph=0-disabled, heartbeat fresh).

## Budget / box
ONE live box: 43197578 (1×H200, team), ~$3.66/hr, under $8 cap. Budget-cap teardown not before 2026-07-01 09:45 (run finishes ~5h from 17:45 start). Old box 43190371 GONE (no leak).

## Deliverable (EXP-43)
~160 per-tick FULL-weight bf16 snapshots (ALL params, n_matrices≈338, real shapes) in R2; manifests synced; WandB backfilled; box torn down ONLY after the gate above.
