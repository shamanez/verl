# Research Status — 2026-06-30T15:55+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 43 | Weight-proj: collect shared dense GRPO weight traj (select_all observer, regime A) | LAUNCHING | external 1×H200 i_43190371 (team), attaching | — | `status:approved`, kind:experiment, M4. First tick: runner dispatched to attach + launch regime-A cell. Box SSH-probed healthy (H200 143771 MiB, idle, 400G free). code_change=false (instrument already on vast-ai-workload). |
| 42 | M4 weight-projection accuracy vs horizon | DONE | external boxes TORN_DOWN | pass | CLOSED (status:pass). The narrow 196-matrix study; de-bloated. EXP-43 widens to select_all (~338 matrices), regime A only. |
| 41 | M4 look-ahead anchor (delay_K=20, fixed-linear) | DONE | external 4×H200 TORN_DOWN | STOP | fixed-linear θ̂ falsified; cell B collapsed via length-explosion. |

## Last tick
2026-06-30T15:55+10:00 · running=[] · launching=[43] · analyzing=[] · logging=[] · blocked=[]

## Pipeline state
EXP-43 is the ROOT of the M4 weight-proj dependency spine (depends_on: []). First tick: the
experiment-runner is attaching the operator-provided external 1×H200 (instance 43190371, team
account) and launching the single dense regime-A collection cell (exp42_run_cell.sh regimeA,
WEIGHT_TRAJ_SELECT_ALL=true, RUN_DIR=/workspace/runs/EXP-43). No provisioning — external box,
do NOT auto-replace. Awaiting runner's RUNNING report → then dispatch training-log-monitor.

## Acceptance gates (this run is ACCEPTED iff ALL FOUR hold)
1. Widened instrument fired: select_all=True banner + first manifest.jsonl n_matrices ≈ 338 (NOT 196).
2. Trajectory >= 80 steps (≈160 ticks, ~160 sketch_tick_*.npz), no NaN/Inf.
3. Sketch fidelity <= 5% rel vs on-box exact fp32 calib.jsonl at grid deltas=[10]×horizons=[5,10,20].
4. Dense regime codec-OFF: powersgd_applications=0, anchor_backwards=0, spectral_corrections=0.
Then: trace synced to research/runs/EXP-43/regimeA/weights/, WandB backfilled to step 80, box TORN_DOWN.

## Budget
External operator box (team account), max_dph $8.0, max_gpu_hr 14. Single live instance once attached.
Teardown is a MUST after the trace syncs + gates pass — external boxes are not exempt.
On terminal gate failure: log STUCK + MANUAL_REVIEW_NEEDED, do NOT auto-provision a replacement.
