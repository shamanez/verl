---
name: vast-attach
description: Register an ALREADY-RUNNING, operator-provided box so the harness can use it without provisioning ("bring-your-own-box"). Probes SSH before registering. Default lifecycle = reaped like any box; --manual = operator-managed EXTERNAL (never auto-reaped). Companion to vast-provision.
allowed-tools: Bash
---

# vast-attach

Skip the ~1–3 min provision + ~5–8 min warm-up: hand the harness a box you
already have. The skill **ssh-probes the box first** (refuses to register an
unreachable one), writes a provision-schema handle, and registers a ledger row.

## Usage

```bash
# Training run on your own box (reaped like any provisioned box; 24 gpu-hr backstop):
bash .claude/skills/vast-attach/run.sh --exp-id 63-anchor-ema-sweep \
  --instance-id 41680420 --account team

# Operator-managed analysis/download box (NEVER auto-reaped — you own teardown):
bash .claude/skills/vast-attach/run.sh --instance-id 41680420 --manual

# Handle only, no ledger row (harness completely unaware):
bash .claude/skills/vast-attach/run.sh --instance-id <id> --ssh-host H --ssh-port P --num-gpus N --no-register
```

| flag | meaning |
|---|---|
| `--instance-id` | **required.** Vast id (ssh/gpu auto-resolved from the API) or any label for a non-Vast box |
| `--exp-id` | run id to file under — use the canonical `<N>-<slug>` (default `ATTACH-<id>`) |
| `--ssh-host/--ssh-port/--num-gpus` | explicit endpoint (auto-resolved for real Vast ids) |
| `--account` | `team` \| `private` (default private) — stamped on handle+row; teardown reads it back |
| `--manual` | ledger `status:EXTERNAL`: tracked by vast-cost, **never auto-reaped**; teardown is YOUR explicit act |
| `--max-gpu-hr` | budget backstop for the default (RUNNING) lifecycle; default 24 |
| `--no-probe` | skip the ssh reachability probe (non-standard boxes) |
| `--no-register` | handle JSON only, no ledger row |

## Lifecycle semantics (the box-43495538 lesson, mechanized)

- **default** → `status:"RUNNING", external:true, max_gpu_hr:<cap>`: the reaper
  applies ALL normal triggers (verdict / heartbeat / budget). Use for training
  runs the harness drives end-to-end (`/launch <N> --attach <id>`).
- **`--manual`** → `status:"EXTERNAL"`: the reaper never touches it and
  vast-cost counts it as tracked (not a leak). Use for operator-managed
  analysis/download boxes where a heartbeat-reap mid-work would destroy data.
  `vast-teardown <id>` still destroys it and flips the row when you're done.
- On env-failure of any attached box the harness tears it down but never
  auto-provisions a replacement (you hand-picked it) — `MANUAL_REVIEW_NEEDED`.

## What it writes

1. `runs/<exp-id>/handles/<instance_id>.json` **and**
   `.claude/state/vast-handles/<instance_id>.json` (provision schema +
   `external:true`) + a `VAST_HANDLE: {...}` stdout line.
2. Unless `--no-register`: one locked ledger append (schema above).

For a brand-new box the harness should own end-to-end, use `vast-provision`.
