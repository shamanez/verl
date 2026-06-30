# Research Status — 2026-06-30T17:46+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 43 | Dense regime-A FULL-weight per-tick → R2 | RUNNING (launching) | 1×H200 i_43197578 (team, external) | — | runner attaching + R2 smoke + 80-step launch on the NEW operator box. Old box 43190371 confirmed DESTROYED (no leak). |
| 42 | M4 weight-projection accuracy vs horizon | DONE | TORN_DOWN | pass | CLOSED. 196-matrix count-sketch study; sketch tooling removed. |
| 41 | M4 look-ahead anchor (delay_K=20, fixed-linear) | DONE | TORN_DOWN | STOP | fixed-linear θ̂ falsified. |

## Last tick
2026-06-30T17:46+10:00 · running=[43 launching] · analyzing=[] · logging=[] · blocked=[]
Tick1 of new session: EXP-43 ABORTED row (operator stop, study redefined) → new external box 43197578
SSH-probed healthy (H200 idle, aws-cli present, R2_* + HF + WandB in $HOME/.verl_auth.env, /workspace/verl
on vast-ai-workload, 400G 1% used). experiment-runner dispatched (bg) to attach + R2 smoke + launch 80-step
regime-A FULL-weight per-tick collection. Stale dead-box handle 43190371.json removed.

## Budget / box
ONLY ONE live box: 43197578 (1×H200, team), $3.66/hr (under $8 cap), max_gpu_hr 16. Direct
`vastai show instances` (team acct) confirms old box 43190371 is GONE — no billing leak.
Teardown sweep DEFERRED to next turn to avoid a runs.jsonl write-race with the runner's
PROVISIONED→RUNNING promotion; budget proven safe by the direct vast query above.

## Deliverable (EXP-43)
~160 per-tick FULL-weight bf16 snapshots (ALL params, n_matrices≈338, real shapes — NO sketch, NO
196-subset) at s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/, upload-then-delete-local;
manifests synced to research/runs/EXP-43/regimeA/weights/; WandB backfilled; box torn down.
