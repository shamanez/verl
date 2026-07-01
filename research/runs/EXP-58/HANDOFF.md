# EXP-58 — HANDOFF (probe GREEN, box held, collection NOT launched)

**Status:** probe gate PASSED all 5 hard invariants. Box held RUNNING+ALIVE for the next agent.
**Do NOT tear down the box** (operator directive — another agent owns the collection).

## Box access
- instance_id **`43387501`** · 1×H200 · TEAM account (`VAST_ACCOUNT=team`) · $3.03/hr
- `ssh -i ~/.ssh/vast_ai -p 40381 root@145.241.107.153`  (key `~/.ssh/vast_ai_name` also attached)
- world_size **W=1** · repo on box: `/workspace/verl` · runs on box: `/workspace/runs/EXP-58`
- ledger row: `runs.jsonl` id=EXP-58 status=RUNNING inst=43387501 acct=team started 01:33 max_gpu_hr=96
- **heartbeat keepalive** running locally (PID in `runs/EXP-58/heartbeat_keepalive.pid`) — touches `metrics/incoming.log` every 10 min so the no-heartbeat-30min reaper won't kill the idle box. Kill it once the collection is launched (training keeps the heartbeat fresh on its own).

## Code
- branch **`exp/58-ckpt-r2`** on origin `shamanez/verl` (base `vast-ai-workload`). Deployed on the box.
  - `325acd70` — checkpoint→R2 on-the-go hook (4 target_modules edits + `research/scripts/ckpt_r2_collection_cell.sh` + 12 CPU tests)
  - `740bc3cb` — hotfix: `shift` the phase arg in the collection cell
- tests: 12/12 new (`tests/trainer/ppo/test_ckpt_r2_mirror.py`) + 28/28 `test_r2_sink.py` PASS

## Probe verdict — 5/5 hard invariants PASS
Evidence: `runs/EXP-58/probe-artifacts/`
1. method-OFF byte-parity — PASS (0 `[ckpt_r2]` markers on OFF save path; R2 relpath set == OFF on-disk set)
2. on-the-go / disk-bounded — PASS (R2 rows verified DURING training; local shards deleted after verify; peak ≈19 GB, 0 leftover .pt)
3. resume completeness — PASS (R2 `latest_checkpointed_iteration.txt`=2 resolves `global_step_2` from R2 alone; model+optim+extra_state+data.pt+fsdp_config+huggingface present)
4. drain-barrier fails loud — PASS (`close() OK: n_uploaded=22 n_errors=0`)
5. FSDP1 summon-safe + no NaN/OOM — PASS (NO_SHARD, pg_loss finite 0.0157/0.0062, OOM=0, peak 45/143 GB)

## Exact resolved config for the 1000-step collection
```
strategy=fsdp (FSDP1, NO_SHARD @ W=1)   use_orig_params=true   comm_eff.enabled=false (anchor OFF)
disable_custom_all_reduce=true
MAX_RESPONSE_LENGTH=4096 (probe value)  ROLLOUT_TP=1  ROLLOUT_GPU_MEM_UTIL=0.45  ALLOW_SINGLE_GPU=1
ppo_micro_batch_size_per_gpu=1  ppo_max_token_len_per_gpu=18432  train_batch_size=128  ppo_mini_batch_size=64
trainer.checkpoint_r2_enabled=true  trainer.max_actor_ckpt_to_keep=null  SAVE_FREQ=20  TOTAL_TRAINING_STEPS=1000  TOTAL_EPOCHS=2
CKPT_R2_ASYNC=true CKPT_R2_DELETE_LOCAL=true CKPT_R2_WORKERS=4 CKPT_R2_MAX_STAGED_GB=50
WEIGHT_TRAJ: dump_dtype=fp32 per_tick=false FULL_EVERY=20 r2_enabled=true
R2 prefixes: verl-research/EXP-58/regimeA/weights  and  verl-research/EXP-58/regimeA/checkpoints  (distinct)
```
**Launch command (on the box):**
```
CKPT_R2_ENABLED=true ALLOW_SINGLE_GPU=1 ROLLOUT_TP=1 ROLLOUT_GPU_MEM_UTIL=0.45 \
  bash research/scripts/ckpt_r2_collection_cell.sh collection
```

## ⚠️ Two decisions before launch
1. **Runtime:** ~267 s/step at resp=4096/n=8 on 1×H200 → **~74 h wall / ~$225** for 1000 steps (exceeds the plan's 18 h soft deadline; ~74 GPU-hr is under the 96 cap but with little margin). Operator OOM-fallback `MAX_RESPONSE_LENGTH=2048` (or a larger rung) would materially cut this. **No OOM occurred** (45/143 GB), so resp=4096 is a *speed/cost* choice, not a fit constraint. — OPERATOR DECISION PENDING.
2. **R2 pre-clean:** the probe left `global_step_1/` + `global_step_2/` under `…/regimeA/checkpoints/` in R2. They don't collide with collection steps {20…1000} but delete them so `verify_ckpt_r2_mirror.py` sees exactly the 50 expected step-dirs. Not deleted (destructive; under hold).

## Deliverable (definition of done)
- `weights/` R2 manifest: 50 `verified:true` fp32 snapshots (steps 20..1000)
- `checkpoints/` R2 manifest: complete `verified:true` object set for each of the same 50 `global_step_<N>/`
- 1000 clean steps, no NaN; dry restore resolves step 1000 from R2 tracker; on-the-go + disk-bounded confirmed
- verify: `research/scripts/verify_full_weight_dump.py`, `verify_ckpt_r2_mirror.py` (analyst adds), `analyze.py`
- on finish: backfill last 1–2 WandB steps; tear down ONLY after BOTH manifests show 50 verified:true + synced.
