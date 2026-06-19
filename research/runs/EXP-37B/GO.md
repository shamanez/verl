# EXP-37B — GO runbook (single run, NOT yet launched)

Everything is staged. This box (team account, instance `41680420`, `84.8.106.109:40206`,
H200×4, dph $12.88) is **operator-managed and intentionally NOT in `runs.jsonl`** so the
`teardown-finished-runs` Stop hook cannot reap it while idle. Launching is the two-step
"GO" block below: it (1) registers the run as RUNNING and (2) starts training in tmux.

## Pre-flight already done (2026-06-20)
- [x] SSH reachable on `~/.ssh/vast_ai` AND `~/.ssh/vast_ai_name` (sync-metrics/monitor use the latter)
- [x] `~/.config/verl-research/secrets.env` on box = HF_TOKEN + WANDB_API_KEY only, chmod 600, no VAST leak
- [x] GSM8K parquet pre-prepped on box (`~/data/gsm8k`, train 7473 / test 1319)
- [x] cgroup `pids.max`=27392 (>=4096, safe), all 4 GPUs idle
- [x] verl on `vast-ai-workload`, accel-base launcher present, torch 2.11.0+cu130 / vllm 0.20.2
- [x] `/workspace/runs/EXP-37B/{launch.sh,config.yaml}` rsynced to box
- [x] local `runs/EXP-37B/{handles,metrics,config.yaml,launch.sh}` materialised

## GO — run this from `/Users/shamane/Documents/verl/research` (laptop)

```bash
cd /Users/shamane/Documents/verl/research
HANDLE=$(cat runs/EXP-37B/handles/41680420.json)
ROW=$(jq -nc --argjson h "$HANDLE" --arg t "$(date -Iseconds)" --argjson ts "$(date +%s)" \
  '{id:"EXP-37B", handles:[$h], started_at:$t, started_at_epoch:$ts,
    max_gpu_hr:48, per_node_gpus:4, total_gpus:4, dph:12.878654970760232,
    chosen_tier_idx:0,
    chosen_tier_query:"num_gpus=4 gpu_name=H200 gpu_ram>=140 reliability>=0.95 rentable=true verified=true",
    vast_account:"team", instance_id:41680420, status:"RUNNING"}')
echo "$ROW" >> .claude/state/runs.jsonl
# fresh heartbeat so the row owns its incoming.log path
mkdir -p runs/EXP-37B/metrics && touch runs/EXP-37B/metrics/incoming.log

# launch in a detached tmux; banner -> launch-banner.log, training -> /workspace/train.log
ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=accept-new -p 40206 root@84.8.106.109 \
  "tmux new -d -s exp-37b-84_8_106_109 'bash /workspace/runs/EXP-37B/launch.sh > /workspace/runs/EXP-37B/launch-banner.log 2>&1'"

# flip the issue label
gh issue edit 37 --repo shamanez/verl-compression-research --add-label status:running --remove-label status:approved
```

## After GO — verify the launch took (within ~10 min)
```bash
ssh -i ~/.ssh/vast_ai -p 40206 root@84.8.106.109 \
  "tmux ls; echo ---; grep -E 'cadence=5|delay_K=5|total_training_steps=100|train_batch_size=128|total_epochs=2' /workspace/train.log | tail -8"
```
Expect the banner to show `anchor.cadence=5 delay_K=5`, `total_training_steps=100`, `total_epochs=2`.
Per the plan, for THIS run the banner's 5/5 is trustworthy (no override clobber — it IS the default).

## Then dispatch the monitor (orchestrator step)
The orchestrator's next tick sees status:RUNNING with no monitor and dispatches
`training-log-monitor` (background) for EXP-37B. To do it by hand, dispatch the
training-log-monitor subagent with the handle above and plan `.claude/plans/37.md`.

## DO NOT
- Do NOT pass any trailing Hydra args or `COMM_EFF_ANCHOR_*` overrides (5/5 is the default;
  an override leak => latency-not-realized => REVISE).
- Do NOT register a PROVISIONED row and walk away — the 15-min PROVISIONED-stale teardown
  trigger would reap the box. Register RUNNING only as part of GO (immediately before launch).
