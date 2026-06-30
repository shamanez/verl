# Research Status — 2026-06-30T16:40+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 43 | Weight-proj: collect dense GRPO **FULL weight matrices** (all params, bf16, per-tick, regime A) | READY (code done) | external 1×H200 i_43190371 (team) **ABORTED+held** | — | `status:approved`. First attempt (count-sketch) ABORTED by operator mid-load. Study REDEFINED to RAW FULL weights of EVERY param; sketch/calib + the `select_all` subset toggle REMOVED; per-tick cadence (~160 snapshots) + Cloudflare R2 upload-then-delete-local added; code DONE + pushed `vast-ai-workload`. New session: attach 43190371, fetch, smoke, launch. |
| 42 | M4 weight-projection accuracy vs horizon | DONE | TORN_DOWN | pass | CLOSED. Narrow 196-matrix count-sketch study; its sketch-analysis tooling removed this session. |
| 41 | M4 look-ahead anchor (delay_K=20, fixed-linear) | DONE | TORN_DOWN | STOP | fixed-linear θ̂ falsified; cell B length-explosion. |

## Last tick
2026-06-30T16:40+10:00 · running=[] · aborted+held=[43] · analyzing=[] · logging=[] · blocked=[]
EXP-43 first attempt operator-aborted; instrument redefined + reimplemented (full-weights);
code committed + pushed to vast-ai-workload; box held for new-session relaunch.

## Pipeline state — EXP-43 = RAW FULL weights on R2 (operator, 2026-06-30)
The count-sketch (k=4096) instrument AND the `select_all`/196-substring subset toggle were REMOVED.
`WeightTrajObserver` now dumps the FULL weights of EVERY floating param (the whole model, no subset,
no compression). Cadence is set by `per_tick`: `true` => one snapshot per optimizer TICK
(`full/tick_<tick>.pt`, ~160 over 80 steps), `false` => per step (`full/step_<gs>.pt`); each manifest
row carries both `global_step` and `tick` so per-tick subsamples to per-step. A shared Cloudflare R2
sink (`verl/workers/comm_eff/r2_sink.py`, bucket `shamane-pluralis`, creds from the env) uploads each
snapshot then DELETES the local `.pt` after a verified upload, so local disk is staging only — the
~492 GB per-tick trajectory never lands on the box/laptop. The same sink routes the gradient capture
dumps. Plan `.claude/plans/43.md` reconciled. Local validation: py_compile OK; 39 CPU tests pass
(`test_r2_sink`, `test_weight_traj_observer`, `test_capture_writer_r2`, `test_ef_powersgd`); Hydra
instantiate with `per_tick=true r2_enabled=true` runs the validators clean.

## What the NEW session does (the run was handed off)
1. Attach external box `43190371` (1×H200, team) — still up; `attach_box` in the plan. If reaped, provision per chain.
2. On box: `git fetch origin vast-ai-workload && git checkout -fB vast-ai-workload origin/vast-ai-workload`; assert
   `grep _dump_full verl/workers/comm_eff/capture.py` hits and `grep -c "class CountSketch" ...` is 0.
3. Ensure the box has `aws` CLI + R2 creds in `$HOME/.verl_auth.env` (R2_*). Smoke 1-2 steps with
   `WEIGHT_TRAJ_R2_ENABLED=true` → confirm `full/tick_*.pt` upload to R2, the local `.pt` is deleted, and
   `r2_manifest.jsonl` rows are `verified:true`; then launch the 80-step regime-A cell:
   `exp42_run_cell.sh regimeA` with `RUN_DIR=/workspace/runs/EXP-43 WEIGHT_TRAJ_PER_TICK=true WEIGHT_TRAJ_R2_ENABLED=true WEIGHT_TRAJ_FULL_DTYPE=bf16`.
4. Gates: ~160 R2 objects under `verl-research/EXP-43/regimeA/weights/full/` (n_matrices≈338, real shapes),
   `r2_manifest.jsonl` verified, no NaN, comm_eff counters 0.
5. Local disk stays near-empty (upload-then-delete). Sync the small manifests/logs to the MacBook,
   backfill WandB, THEN tear down. Analysis pulls weights from R2 on demand (subsample first tick/step).

## Budget / box
External operator box `43190371` (team) is UP and idle (training killed). Intentionally NOT torn down
(operator chose keep-up for the relaunch). CAVEAT: the harness auto-reaper may destroy an idle external
box after ~30-60 min of stale heartbeat; if so, the new session re-attaches/re-provisions per the chain.
Teardown is a MUST after the new run's trace syncs + gates pass.
