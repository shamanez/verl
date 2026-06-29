---
name: vast-cost
description: Report live Vast.ai burn rate ($/hr) + projected 24h spend across BOTH accounts (private + team), and flag any live instance with no owning ledger row (an untracked box = likely billing leak). Read-only — never destroys. The money-visibility companion to vast-provision / vast-teardown.
---

# vast-cost

A read-only spend check, inspired by Vast's official `/vastai:cost`, extended with a
**ledger cross-check the official plugin lacks**: it flags any live instance that no
RUNNING/PROVISIONED ledger row owns — exactly the orphan class that leaks money (a
provision orphan, or a teardown that silently no-opped under the wrong account).

## Usage

```bash
bash .claude/skills/vast-cost/run.sh
```

No args. It sources `../_vast_account.sh`, loads both keys from
`~/.config/verl-research/secrets.env`, and for each account with a key runs
`vastai show instances --raw`.

## What it does

1. Sums `dph_total` over **running** instances on the **private** and **team** accounts.
2. Prints per-account: instance count, `$/hr`, and one line per instance
   (`id / status / NxGPU / $/hr`).
3. **Leak flag:** any live instance id not present in a `RUNNING`/`PROVISIONED`
   `runs.jsonl` row is marked `UNTRACKED (possible LEAK)`.
4. Emits a machine-readable summary:
   `VAST_COST: burn_rate_dph=<X> projected_24h_usd=<Y> untracked=<0|1>`.

## When to run

- Before/after provisioning, to confirm the spend you expect (and nothing else).
- Each orchestrator tick alongside `teardown-finished-runs.sh`, for standing
  spend + leak visibility (a long `/loop` may not emit a Stop between ticks).
- Any time you suspect a leak — `untracked=1` means a billing box the harness
  isn't tracking; investigate, then `vast-teardown <id>` (it resolves the account).

## Safety

Read-only. Never calls `vastai destroy`. Never echoes API keys (passes them only as a
per-command `VAST_API_KEY=…` env prefix). Exits 0 even if one account errors, so a
single-account hiccup never hides the other account's spend.
