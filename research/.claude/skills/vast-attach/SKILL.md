---
name: vast-attach
description: Register an ALREADY-RUNNING, operator-provided Vast.ai box as an EXTERNAL handle the harness can use without provisioning — and that the teardown hook + vast-teardown skill will NEVER auto-destroy. Companion to vast-provision for the "bring-your-own-box" fast path.
allowed-tools: Bash
---

# vast-attach

Skip the ~1–3 min provision + ~5–8 min warm-up: hand the harness a box you already
have running and start immediately. The box is marked **`external: true`** on both
the handle JSON and the ledger row, so **neither** teardown path will destroy it —
its lifecycle is yours.

## Usage

```bash
# Real Vast box — the API resolves ssh/gpu details from just the instance id:
bash .claude/skills/vast-attach/run.sh --instance-id 41680420 --account team

# Or give them explicitly (any box, Vast or not):
bash .claude/skills/vast-attach/run.sh \
  --instance-id 41680420 --ssh-host 84.8.106.109 --ssh-port 40206 --num-gpus 4 --account team

# Just write the handle, do NOT add a ledger row (purest manual path — harness
# stays completely unaware of the box, so nothing can ever touch it):
bash .claude/skills/vast-attach/run.sh --instance-id <id> --ssh-host H --ssh-port P --num-gpus N --no-register
```

| flag | meaning |
|---|---|
| `--instance-id` | **required.** The Vast instance id, or any label for a non-Vast box. |
| `--ssh-host` / `--ssh-port` / `--num-gpus` | the box's SSH endpoint + GPU count. Auto-resolved from the Vast API if omitted and the id is a real Vast box. |
| `--gpu-name` / `--gpu-ram` / `--dph` | optional metadata (display only; external boxes are never budget-torn-down). |
| `--account` | `team` \| `private` (default `private`) — recorded on the handle/row. |
| `--exp-id` | the run id to file under (default `ATTACH-<instance-id>`). |
| `--no-register` | write the handle JSON only; do **not** add a `runs.jsonl` row. |

## What it does

1. Writes `runs/<EXP_ID>/handles/<instance_id>.json` in the **same schema as
   vast-provision** (`schema_version "1"`, `ssh_login`, `num_gpus`, …) **plus
   `external: true`**, and prints a `VAST_HANDLE: {...}` line (so the
   experiment-runner's rsync+launch path consumes it unchanged).
2. Unless `--no-register`, appends a ledger row `status:"RUNNING", external:true`
   so the orchestrator/monitor/analyst can see it — but the teardown machinery skips it.

## The external contract (why this is safe)

- The **teardown Stop hook** (`teardown-finished-runs.sh`) passes `external:true`
  rows straight through — it never destroys them, even on verdict / stale-heartbeat /
  budget.
- The **vast-teardown skill** refuses an instance id whose ledger row is
  `external:true` unless you pass `--force` (`--include-external`).
- ⇒ An attached box stays up across as many experiments as you want; **you** tear
  it down (`vast-teardown --force <id>`, or just destroy it on Vast directly).

## When to use

- You already have a warm box (manual provision, or held from a prior run) and want
  to start training/analysis NOW without paying the provision+warmup tax.
- You want one long-lived box to serve several back-to-back experiments.

For a brand-new box the harness should own and auto-tear-down, use `vast-provision`
instead — not this.
