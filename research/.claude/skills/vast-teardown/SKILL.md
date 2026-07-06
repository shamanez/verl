---
name: vast-teardown
description: Destroy one or more Vast.ai instances by id and mark the corresponding ledger rows TORN_DOWN. Use after an experiment finishes or fails so we never bleed money.
allowed-tools: Bash
---

# vast-teardown

Companion to the `vast-provision` skill. Tears down provisioned Vast.ai instances by id and patches the local `research/.claude/state/runs.jsonl` ledger so the harness knows they're gone.

## Usage

From an agent or directly:

```
$CLAUDE_PROJECT_DIR/.claude/skills/vast-teardown/run.sh <instance_id> [<instance_id> ...]
```

Or pass a path to a handle JSON file (the runner emits these under `research/runs/<N>-<slug>/handles/`):

```
$CLAUDE_PROJECT_DIR/.claude/skills/vast-teardown/run.sh --handles research/runs/<run>/handles/
```

This tears down **any** instance id you give it — including operator-attached
(`external:true`) boxes. Teardown is a must; nothing is exempt.

## Behavior

1. Resolves auth **per instance** (`team`|`private`, default `private`) via the shared `../_vast_account.sh` resolver, reading `vast_account` in priority order: (a) the handle JSON passed via `--handles`, (b) the provision handle dir `.claude/state/vast-handles/<id>.json`, (c) the ledger row referencing the id. A `team` box is destroyed with `VAST_API_KEY_TEAM`, a `private` box with `VAST_API_KEY` (both from `~/.config/verl-research/secrets.env`). This guarantees a team-account box is never orphaned under the personal key — even for an id with no ledger row yet. If the resolved key is **empty**, the id is SKIPPED (logged + counted failed) rather than destroyed with an empty key (which the CLI would silently downgrade to the stored private config). An explicit `VAST_ACCOUNT=team|private` env var forces that account for **all** ids.
2. For each instance id, runs `vastai destroy instance <id> -y` under the resolved key. Failures are logged to `/tmp/teardown.err` but never block — the script always exits 0 even if some destroys failed, so a partially-failed teardown still lets the orchestrator move on.
3. Patches matching ledger rows in `.claude/state/runs.jsonl` to `status: "TORN_DOWN"` with `torn_down_at` and `teardown_reason: "manual"` (or `--reason <r>` to override).
4. Emits a one-line summary on stdout: `VAST_TORN_DOWN: destroyed=<N> failed=<N> reason=<R>` (a non-zero `failed` means a box may still be live — check `/tmp/teardown.err`).

## Why this exists separately from the Stop-hook teardown

The `teardown-finished-runs.sh` Stop hook handles **automatic** teardown — verdict written, heartbeat stale, budget exceeded, orphaned handles. This skill is for **explicit** teardown the operator or an agent wants to force, independent of those triggers (e.g., aborting an experiment that's heading nowhere, or cleaning up after a launch retry that left orphans).

## Session-independent backstop (optional, recommended)

The Stop-hook reaper only fires while a Claude session is open. To keep the
money backstop alive with zero sessions, install an hourly cron on the laptop:

```bash
crontab -e   # add:
17 * * * * CLAUDE_PROJECT_DIR=/Users/shamane/Documents/verl/research bash /Users/shamane/Documents/verl/research/.claude/hooks/teardown-finished-runs.sh >> /tmp/teardown.cron.log 2>&1
```

The hook is idempotent, lock-aware, and exits 0 — safe to run alongside live
sessions.
