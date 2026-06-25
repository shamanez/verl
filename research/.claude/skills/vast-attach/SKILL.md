---
name: vast-attach
description: Register an ALREADY-RUNNING, operator-provided Vast.ai box as an EXTERNAL handle the harness can use without provisioning (skip the provision+warmup tax). external is PROVENANCE only — the box is still torn down after its run or on request, like any box. Companion to vast-provision for the "bring-your-own-box" fast path.
allowed-tools: Bash
---

# vast-attach

Skip the ~1–3 min provision + ~5–8 min warm-up: hand the harness a box you already
have running and start immediately. The box is marked **`external: true`** (provenance:
the harness didn't provision it) — but it is **still torn down** after its run completes
or on request. Teardown is a must; external is not an exemption.

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

## The external flag (provenance, NOT teardown protection)

`external: true` only records that the operator ATTACHED this box (the harness did
not provision it). **It does not exempt the box from teardown** — teardown is a must:

- The **teardown Stop hook** tears an external box down on the same triggers as a
  provisioned one (verdict written / heartbeat stale / budget exceeded).
- The **vast-teardown skill** destroys an external box like any other (no `--force`).
- The one external-specific behaviour: on an **env-failure**, the harness does NOT
  auto-provision a replacement (you hand-picked a box; there's no SKU chain to walk) —
  it tears the box down and surfaces `MANUAL_REVIEW_NEEDED` instead.

Want a box the harness will NOT track or tear down? Use `--no-register` (or skip the
skill and just SSH in): with no ledger row the harness never sees it, and **you** own
its teardown entirely.

## When to use

- You already have a warm box (manual provision, or held from a prior run) and want
  to start training/analysis NOW without paying the provision+warmup tax. The box is
  torn down after its run completes (or on request); to keep it across runs, use
  `--no-register` and manage (and tear down) it yourself.

For a brand-new box the harness should own and auto-tear-down, use `vast-provision`
instead — not this.
