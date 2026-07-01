# EXP-57 box tracking (fp32 weight-trajectory collection)

## STATUS: ✅ COMPLETE — BOX TORN DOWN (2026-07-01T11:20Z)

VERDICT **PASS**. 160/160 fp32 snapshots R2-verified under
`s3://shamane-pluralis/verl-research/EXP-57/regimeA/weights/full/`,
`verify_full_weight_dump.py --r2` PASS (max_rel_norm_err=0), codec-OFF,
no NaN/Inf, WandB `ztre15q8` backfilled to step 80. Instance 43311909
destroyed (private account); ledger row EXP-57 = TORN_DOWN. Nothing live.

---

_Historical (during the run):_ It was intentionally NOT registered in `runs.jsonl`
(to keep it out of the no-heartbeat-30min auto-teardown reaper, which killed the
first widened EXP-43 attempt). Teardown was MANUAL, per the plan's
COLLECTION-RUN TEARDOWN GATE.

- instance_id: **43311909**
- account: **private** (`VAST_API_KEY` — NOT team)
- ssh: `ssh -i ~/.ssh/vast_ai -p 15490 root@210.157.233.86`
- gpu: 1×H200, ~$3.16/hr
- tmux: `exp57a`
- launched: 2026-07-01 ~09:29 UTC (box time)
- run cmd: `RUN_DIR=/workspace/runs/EXP-57 WEIGHT_TRAJ_PER_TICK=true WEIGHT_TRAJ_FULL_DTYPE=fp32 WEIGHT_TRAJ_R2_ENABLED=true bash research/scripts/weight_traj_run_cell.sh regimeA`
- R2 prefix: `s3://shamane-pluralis/verl-research/EXP-57/regimeA/weights/full/`
- box logs: `/workspace/runs/EXP-57/regimeA/train_regimeA_internal.log` (training),
  `/workspace/runs/EXP-57/regimeA/driver.log` (launcher banner),
  `/workspace/runs/EXP-57/cell_stdout.log` (driver echoes)
- manifests: `/workspace/runs/EXP-57/regimeA/weights/{full_manifest.jsonl,r2_manifest.jsonl}`

## Teardown gate (ALL must hold before destroy)
1. `r2_manifest.jsonl` 160/160 rows `verified:true`, `local_bytes==remote_bytes`
2. manifests synced to `research/runs/EXP-57/regimeA/weights/`
3. `verify_full_weight_dump.py research/runs/EXP-57/regimeA/weights --r2 --r2-sample 8 --tol 0.01 --expect 160` → PASS
4. clean `comm_eff_close` (no permanent upload failure)
5. WandB backfilled to final step

Then: `VAST_API_KEY=$VAST_API_KEY vastai destroy instance 43311909 -y` (private key).
