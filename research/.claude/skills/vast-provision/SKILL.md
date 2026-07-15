---
name: vast-provision
description: Provision the cheapest vast.ai instance(s) matching a compute requirement, wait for SSH-routable state, write a handle JSON the experiment-runner consumes, and emit a one-line summary. Also registers an already-running bring-your-own box (attach mode, sibling `vast-attach/run.sh`). Companion to vast-teardown.
allowed-tools: Bash
---

# vast-provision

Picks the cheapest qualifying vast.ai offer(s), creates the instance(s), verifies real SSH,
writes a handle JSON per box, prints machine-parseable `VAST_HANDLE:` lines. Needs `vastai` +
`jq`. The ONLY sanctioned way to create instances — agents never call `vastai create` directly.

## How callers run it

Provisioning takes ~7–25 min per box (image pull + onstart), longer than one Bash call's cap:
run the skill DETACHED (`nohup bash run.sh ... > provision.log 2>&1 &`) and poll the log +
handle dir. Every `vastai` call inside is timeout-bounded (`VAST_CLI_TIMEOUT`, default 120 s),
so the skill itself never hangs; a stderr progress line is emitted on every poll.

```bash
bash $CLAUDE_PROJECT_DIR/.claude/skills/vast-provision/run.sh \
  --query "num_gpus=1 gpu_name=H200 gpu_ram>=140 cuda_max_good>=13.0 reliability>0.99 rentable=true verified=true" \
  --disk-gb 200 --max-price 6.0 --count 1 --label EXP-61
```

Defaults: `COUNT=1 DISK_GB=200 MAX_PRICE=1.0 MIN_RELIABILITY=0.99 TIMEOUT=1500s/instance
POLL_INTERVAL=15s POOL=COUNT+8` candidates; SSH identity `~/.ssh/vast_ai_name` (override
`VAST_SSH_IDENTITY`). Auth auto-sources `~/.config/verl-research/secrets.env`
(`VERL_SECRETS_FILE` overrides); handles land in `.claude/state/vast-handles/`
(`VERL_VAST_HANDLE_DIR` / `--handle-dir`).

**Secret seeding (automatic).** The instant SSH is verified, the skill pushes a
STRIPPED copy of the laptop secrets to `/root/.config/verl-research/secrets.env`
on the box (`chmod 600`) via `_seed_secrets.sh` — an allowlist of HF + WandB + R2
keys only; the Vast API keys are structurally withheld (the on-box launcher
FATALs if a VAST key leaks). This closes the launcher's
`FATAL: secrets.env not found` with no manual `scp` and no agent step. Disable
with `VERL_SEED_SECRETS=0`.

## Template auto-selection (agents never pass --image / --template-hash)

With neither flag given, the skill reads `templates.json` (this dir) — the single locked
research template `verl-research-vllm020` — and provisions via its Vast Template hash:
`hash_id` for `VAST_ACCOUNT=private`, `team_hash_id` for `VAST_ACCOUNT=team` (a Vast Template
is visible only to its owning account). Choice is logged to stderr (`auto-selected template
...`). The Template record on Vast carries the image, onstart (clones `shamanez/verl` at the
template's pinned branch, `pip install --no-deps -e .`), docker options, recommended disk, and the
`cuda_max_good>=13.0` driver filter. Maintenance: update `templates.json` and the Vast
console Template record together — that is the whole runbook.

### Team-account templates

`VAST_ACCOUNT=team` with no `team_hash_id` in `templates.json` fails fast with exit 4 BEFORE
any create (the private hash is guaranteed to 400 on the team account). Remedy: re-run with
`VAST_ACCOUNT=private`, or record a team-owned copy (identical image + docker options) in
`templates.json`.

## Accounts & SSH keys

`VAST_ACCOUNT=team|private` (default private) is resolved by `../_vast_account.sh`, which
picks `VAST_API_KEY_TEAM` vs `VAST_API_KEY` from the secrets file; the choice is stamped on
the handle as `vast_account` so teardown auths against the same account. Team accounts cannot
hold account-level SSH keys, so team boxes get the harness key(s) attached per-instance
(`vastai attach ssh`) right after create — zero keys attached ⇒ destroy + next candidate.
Private boxes use the account's uploaded keys (skill refuses to run if the account has none).

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--query`, `-q` | (required) | vast.ai `search offers` query DSL string |
| `--count`, `-n` | `1` | number of instances |
| `--disk-gb`, `--disk` | `200` | disk per instance (GB), fixed at create; 200 covers image + model + data + ckpts, bump to 400+ for 32B-class models |
| `--max-price` | `1.0` | per-instance $/hr ceiling; production plans MUST override (typically `6.0` for 1×H200) |
| `--min-reliability` | `0.99` | minimum `reliability2` |
| `--gpu-count` | unset | fail-fast filter: offers with `num_gpus == N` only |
| `--label` | unset | **exact** instance label (= run id); beats `--label-prefix` |
| `--label-prefix` | `verl-research` | label = `<prefix>:<session-id>` when `--label` unset |
| `--session-id` | new UUID | baked into label + handle |
| `--timeout` | `1500` | seconds per instance to reach running + SSH-verified |
| `--poll-interval` | `15` | seconds between polls |
| `--handle-dir` | `$VERL_VAST_HANDLE_DIR` | where `<instance_id>.json` lands |
| `--template-hash` | auto | pin a specific Vast Template (override path only) |
| `--image`, `-i` | auto | bypass the Template entirely — onstart won't run, `/workspace/verl` won't exist; not for harness use |
| `--env` / `--login` / `--onstart-cmd` | unset | raw vastai passthrough; never pass laptop credentials |
| `--no-default-filters` | off | `-n` to search (drops implicit `rentable/verified`) AND drops the skill's `direct_port_count>=1` filter |
| `--dry-run` | off | search + filter + pick, print would-be create argv, exit 0, no spend |

## Stdout contract (machine-parseable)

```
VAST_HANDLE: {"created_at":"...","dph_total":6.40,...}   # one per box; one-line JSON (jq -c, sorted keys)
VAST_PROVISIONED: count=1 total_dph=6.4000               # exactly one, terminal
```

Everything else goes to stderr — including a `BUDGET:` line printed BEFORE any
`vastai create` (count × max-price aggregate ceiling; the spend audit trail).
Downstream parses `grep '^VAST_HANDLE: '` + `jq` — keep prefixes verbatim.

## Handle JSON (`schema_version` "1" — cross-skill, do NOT rename fields)

```json
{"schema_version":"1", "instance_id":"12345", "offer_id":"98765",
 "ssh_host":"ssh4.vast.ai", "ssh_port":12345, "public_ipaddr":"1.2.3.4",
 "gpu_name":"H200", "num_gpus":1, "dph_total":6.40,
 "created_at":"2026-05-25T12:30:00Z", "label":"EXP-61", "session_id":"<uuid>",
 "ssh_login":"ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 12345 root@ssh4.vast.ai -L 8080:localhost:8080",
 "vast_account":"private"}
```

- `num_gpus` (not gpu_count) and `dph_total` (not dph) — experiment-runner sums these into
  the ledger row. `instance_id` is a string; `vast-teardown` parses this schema, so any
  change here requires a paired change there.
- `ssh_login` is the paste-ready connect command — use it verbatim; never hand-build a bare
  `ssh -p <port> root@<host>` (wrong key offered, host-key trips on reused Vast IPs).
- Written atomically (mktemp + mv) to `<handle-dir>/<instance_id>.json`; no secrets inside.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | all instances provisioned (or `--dry-run` succeeded) |
| 1 | missing CLI / credentials |
| 2 | bad arguments |
| 3 | no qualifying offers (stderr names the cheapest rejected; the runner's tier-walk treats 3 as "advance to next tier") |
| 4 | unrecoverable create error — bad key / template not accessible to this account; incl. `VAST_ACCOUNT=team` with no `team_hash_id`, which fails BEFORE any create |
| 5 | candidate pool exhausted before `--count` boxes SSH-verified (everything created on the way was already destroyed) |

## Never-leak invariants

- Every created instance is tracked until SSH-verified; an EXIT trap destroys any unverified
  box on ANY exit path (timeout, error, signal) — provision owns this because it alone knows
  the instance id before a ledger row exists.
- A mapped 22/tcp port is NOT success: a REAL `ssh ... true` probe with the offered key must
  pass (team: re-attach key once, retry) or the box is destroyed and the next candidate tried.
- Endpoints never mix: direct = `public_ipaddr` + `ports["22/tcp"]` HostPort; proxy =
  `ssh_host` + `ssh_port`. `public_ipaddr`+`ssh_port` is unreachable by construction.
- pids.max gate: after SSH verify, the skill reads the container's cgroup `pids.max`; a host
  capping it at <= 2048 deterministically SIGABRTs under FSDP+vLLM (~1700+ threads), so the
  box is auto-destroyed and the next candidate tried.
- Host-side failures (`intended_status`→stopped, OCI/CDI daemon errors) fail fast during the
  poll → destroy + advance, instead of blocking the full timeout.
- `--cancel-unavail` on every create — the scheduler can never leave a stopped-but-billing
  orphan; and `VAST_API_KEY` (or any laptop env) is never forwarded onto the instance.

## Attach mode — bring-your-own-box (sibling `vast-attach/run.sh`)

Instead of creating a box, register one you already have (skip the ~1–3 min
provision + ~5–8 min warm-up). Same end state as provisioning: an SSH-probed box
with a provision-schema handle + a ledger row, secrets seeded via
`_seed_secrets.sh` (HF + WandB + R2 only; Vast keys withheld). Reached through
`/launch --attach` / `/execute --attach`, and called directly by the
experiment-runner; the engine is `vast-attach/run.sh` (was its own skill until the
2026-07-15 fold — same script, no longer a separate skill entry).

```bash
# Harness-driven training run on your own box (reaped like any provisioned box):
bash .claude/skills/vast-attach/run.sh --exp-id 63-anchor-ema-sweep --instance-id 41680420 --account team

# From a raw SSH login string (host/port/key parsed; Vast id reverse-resolved for teardown):
bash .claude/skills/vast-attach/run.sh --exp-id 64-middle-block-freeze --issue 64 \
  --ssh-login "ssh -i ~/.ssh/vast_ai -p 15338 root@ssh8.vast.ai"

# Operator-managed analysis/download box (NEVER auto-reaped — you own teardown):
bash .claude/skills/vast-attach/run.sh --instance-id 41680420 --manual
```

Key flags: `--instance-id` (or `--ssh-login`), `--exp-id <N>-<slug>`,
`--account team|private`, `--ssh-identity <key>`, `--manual` (EXTERNAL, never
reaped), `--need-r2` (R2 write preflight), `--no-probe`, `--no-register`. Full
flag table, secret-seeding, and synthetic-id details live in the header of
`vast-attach/run.sh`.

**Lifecycle:** default → `status:RUNNING external:true` (all reaper triggers
apply; `--max-gpu-hr` cap, default 24). `--manual` → `status:EXTERNAL` (the reaper
never touches it; the burn/leak check still counts it as tracked; you run
`vast-teardown` when done). On env-failure the harness tears an attached box down
but never auto-provisions a replacement (you hand-picked it). `--ssh-login` with an
unresolvable Vast id gets a synthetic `ATTACH-<host>-<port>` id and **auto-teardown
disabled** (reaper + `vast-teardown` skip non-numeric ids) — re-attach with the
numeric `--instance-id` to restore it.

## Not this skill's job

Teardown (`vast-teardown` / the Stop hook), `runs.jsonl` writes (experiment-runner), remote
commands, tier-exhaustion retries (the runner walks the `gpu_filter_chain`).
