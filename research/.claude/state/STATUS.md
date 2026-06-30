# Research Status — 2026-06-30T16:40+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 43 | Weight-proj: collect dense GRPO **FULL weight matrices** (bf16, per step, regime A) | READY (code done) | external 1×H200 i_43190371 (team) **ABORTED+held** | — | `status:approved`. First attempt (count-sketch) ABORTED by operator mid-load. Study REDEFINED to FULL per-step weights; sketch/calib code REMOVED; code DONE + pushed `vast-ai-workload@12202b0e`. New session: attach 43190371, fetch, smoke, launch. |
| 42 | M4 weight-projection accuracy vs horizon | DONE | TORN_DOWN | pass | CLOSED. Narrow 196-matrix count-sketch study; its sketch-analysis tooling removed this session. |
| 41 | M4 look-ahead anchor (delay_K=20, fixed-linear) | DONE | TORN_DOWN | STOP | fixed-linear θ̂ falsified; cell B length-explosion. |

## Last tick
2026-06-30T16:40+10:00 · running=[] · aborted+held=[43] · analyzing=[] · logging=[] · blocked=[]
EXP-43 first attempt operator-aborted; instrument redefined + reimplemented (full-weights);
code committed + pushed to vast-ai-workload; box held for new-session relaunch.

## Pipeline state — EXP-43 REDEFINED to FULL weights (operator, 2026-06-30)
The count-sketch (k=4096) weight-trajectory instrument was REMOVED. `WeightTrajObserver` now
dumps the FULL weight matrices once per training step (bf16 default, `full/step_<gs>.pt` +
`full_manifest.jsonl`), no compression. Code change is DONE + committed + pushed:
`shamanez/verl` `vast-ai-workload@12202b0e`. Plan `.claude/plans/43.md` reconciled (code_change:true,
status:DONE; new session fetches vast-ai-workload + launches, no re-patch). Local validation:
py_compile OK; observer + `verify_full_weight_dump.py` CPU-smoked PASS.

## What the NEW session does (the run was handed off)
1. Attach external box `43190371` (1×H200, team) — still up; `attach_box` in the plan. If reaped, provision per chain.
2. On box: `git fetch origin vast-ai-workload && git checkout -fB vast-ai-workload origin/vast-ai-workload`; assert
   `grep _dump_full verl/workers/comm_eff/capture.py` hits and `grep -c "class CountSketch" ...` is 0.
3. Smoke 1-2 steps → confirm `full/step_*.pt` load as ≈338 real tensors; then launch the 80-step regime-A cell:
   `exp42_run_cell.sh regimeA` with `RUN_DIR=/workspace/runs/EXP-43 WEIGHT_TRAJ_SELECT_ALL=true WEIGHT_TRAJ_FULL_DTYPE=bf16 WEIGHT_TRAJ_FULL_EVERY=1`.
4. Gates: 80 `full/step_*.pt` (n_matrices≈338, real shapes), no NaN, `verify_full_weight_dump.py` PASS, comm_eff counters 0.
5. Sync ~246 GB to MacBook (needs ≈250 GB free), backfill WandB, THEN tear down.

## Budget / box
External operator box `43190371` (team) is UP and idle (training killed). Intentionally NOT torn down
(operator chose keep-up for the relaunch). CAVEAT: the harness auto-reaper may destroy an idle external
box after ~30-60 min of stale heartbeat; if so, the new session re-attaches/re-provisions per the chain.
Teardown is a MUST after the new run's trace syncs + gates pass.
