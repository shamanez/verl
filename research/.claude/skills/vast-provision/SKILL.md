---
name: vast-provision
description: Provision the cheapest vast.ai instance(s) matching a compute requirement, wait for SSH-routable state, write a handle JSON the experiment-runner consumes, and emit a one-line summary. Companion to vast-teardown.
allowed-tools: Bash
---

# vast-provision

Self-contained provisioning skill for the verl research harness. Given a
vast.ai search query and a docker image, it picks the cheapest qualifying
offer(s), creates the instance(s), waits until each is SSH-routable, and
prints one `VAST_HANDLE: <json>` line per instance plus a final
`VAST_PROVISIONED:` summary. The downstream `experiment-runner` agent
captures those handle lines verbatim (see
`.claude/agents/experiment-runner.md` step 4.2); the `vast-teardown` skill
later consumes the same JSON shape.

## When to invoke

- `experiment-runner` walks its plan's `gpu_filter_chain` and calls this
  skill once per tier (cheapest-first); the first tier returning ≥1 handle
  wins.
- Operators may also call it directly to bring up a one-off box matching
  an ad-hoc query (M0 smoke tests, debug sessions).

## Default Template (the research harness contract)

This skill is the **single source of truth** for how the research harness
provisions Vast.ai instances. Agents must call it with only the
per-experiment knobs (`--query`, `--max-price`, `--count`, `--disk-gb`)
and let the skill pick the locked Template from `templates.json`. Everything
about the runtime — docker image, container `--shm-size` / `--cap-add`,
onstart script (clones `shamanez/verl @ vast-ai-workload`, pip-installs
verl `--no-deps`), recommended disk, CUDA driver filter — lives in the
Template record on Vast.ai and is referenced by `templates.json`.

| Field | Value | Source |
|---|---|---|
| Active template name | `verl-research-vllm020` | `templates.json` |
| Active template hash | `3b0f8b726ac3036d6c007bfa13b6d75f` | `templates.json` |
| Image | `verlai/verl:vllm020.dev1` (torch 2.11.0+cu130, vllm 0.20.2) | Template (Vast) |
| Onstart | clones `shamanez/verl @ vast-ai-workload`, `pip install --no-deps -e .` | `onstart.verl-vllm020.sh` |
| Docker options | `--shm-size=10g --cap-add=SYS_ADMIN -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false -e RAY_DISABLE_USAGE_STATS=1` | Template (Vast) |
| Recommended disk | 200 GB | Template (Vast) |
| Driver filter | `cuda_max_good >= 13.0` | Template (Vast) `extra_filters` |

**Rules of engagement:**
- Agents (orchestrator, experiment-runner) MUST NOT pass `--template-hash`
  or `--image` — leave both unset. The skill auto-fills from
  `templates.json` and emits a stderr line `vast-provision: auto-selected
  template '<name>' hash=<HEX> image=<IMAGE>` so the choice is auditable.
- Agents MUST NOT call `vastai create instance` directly. The skill exists
  precisely so the cgroup / onstart / image-pin / SSH-key / handle-write
  invariants are enforced in one place.
- If the template needs to change (new vllm, new image, new onstart),
  update `templates.json` AND the Vast.ai Template record together — see
  "Templates" below for the maintenance procedure. The hash in
  `templates.json` is the authoritative pointer; the previous hashes list
  records every prior version so we can verify history.

## Required env

| Var | Purpose | Notes |
|---|---|---|
| `VAST_API_KEY` | vast.ai REST credential | Auto-sourced from `~/.config/verl-research/secrets.env` (override path with `VERL_SECRETS_FILE`). Agents calling `/vast-provision` cold do NOT need to `source` first — the skill handles it. |
| `VERL_SECRETS_FILE` | optional path override | Default: `~/.config/verl-research/secrets.env`. |
| `VERL_VAST_HANDLE_DIR` | optional override | Default: `$CLAUDE_PROJECT_DIR/.claude/state/vast-handles`. |
| `CLAUDE_PROJECT_DIR` | research dir | Defaults to the dir three levels above this skill. |

## Expected wait time (the skill blocks the calling shell)

A `vastai create instance` returns a contract id in seconds, but the box is
only **reachable** after vast.ai's host pulls the docker image, extracts
layers, and starts the container's SSH daemon. For the
`verlai/verl:vllm020.dev1` image (~30 GB compressed, ~60 GB extracted) on a
freshly-allocated host that hasn't seen the image before, expect:

| Phase | Typical wait |
|---|---|
| `vastai create instance` returns contract id | ~3-10 s |
| Host downloads image layers | 5-20 min (network dependent) |
| Docker extracts layers + starts container | 30-90 s |
| Template's `onstart` (clone verl + `pip install --no-deps`) | 30-90 s |
| `ports["22/tcp"]` mapped, SSH-routable | total ~7-25 min |

The skill emits a stderr progress line on every poll (default every 15 s)
showing elapsed time, `actual_status`, whether port 22 is mapped, and the
latest line of `status_msg` (the docker pull progress). Example:

```
vast-provision: [+90s poll #6] actual_status=loading port22=none msg='3b1c1909ac70: Pull complete|c78c94e14151: Pulling fs layer'
```

If you want to peek from another shell at any time:

```bash
source ~/.config/verl-research/secrets.env
vastai show instance <id> --raw \
  | python3 -c 'import sys, re; m = re.search(r"\"status_msg\":\s*\"((?:[^\"\\\\]|\\\\.)*)\"", sys.stdin.read(), re.S); print(m.group(1) if m else "")'
```

vast.ai's API returns `status_msg` with raw newlines inside a JSON string,
which is technically invalid JSON — that's why the snippet above uses a
regex instead of `jq`. The skill handles this internally.

If the timeout expires (`--timeout`, default `1500`), the skill exits 5
**but the instance is still running and billing**. Either re-attach via
`vastai show instance <id>` to debug, or tear it down with
`vast-teardown`. The Stop hook will also catch it on session end.

## Cost & safety contract (read first)

This skill spends real money. The defaults are the safety floor:

- `--max-price 1.0` per instance × `--count 1` = **\$1.00/hr aggregate ceiling** by default. Multi-GPU production plans override explicitly.
- Before any `vastai create instance` call, the skill emits a `BUDGET:` line on stderr naming the per-instance ceiling × count. That line is the audit trail even when the Bash tool is allowlisted.
- The skill **refuses to provision** if your vast.ai account has zero uploaded SSH keys — without that you'd get a billable but unreachable box.
- `--dry-run` runs the search/filter/pick stages and prints the exact `vastai create` argv that would run, without spending. Use it freely.
- `--cancel-unavail` is passed to every `vastai create`, so if the scheduler can't place us, vast.ai returns an error instead of leaving a stopped-but-billing instance.

> The skill **never** forwards `VAST_API_KEY` (or any laptop env) onto
> the provisioned instance. Issue #1's stripped-secrets discipline is
> preserved here at the boundary.

External CLIs required on the laptop: `vastai`, `jq`. `uuidgen` if
available; otherwise a timestamp falls back.

## Usage

**Recommended (research harness path):** call WITHOUT `--image` or
`--template-hash`. The skill auto-reads the active Template from
`templates.json` (the single canonical record) and provisions with the
locked image + onstart + docker options. This is the path
`experiment-runner` must use — agents never name the template hash
explicitly.

```bash
bash $CLAUDE_PROJECT_DIR/.claude/skills/vast-provision/run.sh \
  --query "num_gpus=4 gpu_name=H100 gpu_ram>=80 cuda_max_good>=13.0 reliability>=0.97 rentable=true verified=true" \
  --disk-gb 200 \
  --max-price 24.0 \
  --count 1
# stderr emits:  vast-provision: auto-selected template 'verl-research-vllm020' hash=... image=verlai/verl:vllm020.dev1
```

**Override path** (rare — exists only so the skill is reusable outside
the research harness): pass `--template-hash <HASH>` to pin a different
Template, or `--image <IMAGE>` to bypass the Template entirely and
provision a plain instance. Bypassing the Template means the onstart
won't run and `/workspace/verl` won't be cloned — the runner would have
to handle that itself, which is what we're explicitly NOT doing.

The experiment-runner's slash form (`/vast-provision count=1 query="..."
disk_gb=200 max_price=...`) maps one-to-one onto the long flags below.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--query`, `-q` | (required) | vast.ai `search offers` query DSL string |
| `--image`, `-i` | (auto from `templates.json` if exactly one template; **do not pass** in the research harness) | docker image tag. Specifying this bypasses the Template entirely — the onstart won't run and `/workspace/verl` won't be cloned. |
| `--count`, `-n` | `1` | number of **instances** to provision |
| `--gpu-count` | unset | optional **per-host** GPU sanity-check; when set, filters offers to `num_gpus == N`. Use to harden the query if you want a fail-fast guard. |
| `--disk-gb`, `--disk` | `200` | disk allocation per instance, in GB. Default sized for typical research run (image + HF model + dataset + checkpoints). See "Disk sizing" below for rules of thumb. |
| `--max-price` | `1.0` | per-instance `$/hr` ceiling (USD). Sized for a single-GPU ad-hoc smoke; multi-GPU production plans (8×H100, etc.) MUST pass an explicit override (typically `24.0`). |
| `--min-reliability` | `0.95` | minimum `reliability2` score |
| `--env` | unset | raw `vastai --env` string (`"-e VAR=val -p 8000:8000"`). Use only when you need port forwards; do NOT pass laptop credentials. |
| `--login` | unset | `vastai --login` string for private docker registries (e.g. `"ghcr-user -p TOKEN"`) |
| `--onstart-cmd` | unset | inline shell to run at container start (passed to `vastai --onstart-cmd`) |
| `--timeout` | `1500` | seconds to wait for each instance to become running + SSH-routable. Default sized for a ~30 GB image pull + the template's onstart (verl clone + `pip install --no-deps`). |
| `--poll-interval` | `15` | seconds between `vastai show instance` polls |
| `--handle-dir` | `$VERL_VAST_HANDLE_DIR` | where to drop `<instance_id>.json` |
| `--label-prefix` | `verl-research` | instance label prefix; final label = `<prefix>:<session-id>` |
| `--session-id` | new UUID | session id baked into the label and handle |
| `--template-hash` | auto from `templates.json` if exactly one template defined | provision via a saved vast.ai Template (`vastai create instance --template_hash <hash>`). When set, `--image` is optional; if both are passed, `--image` overrides the template's image (vast.ai documented merge semantics: scalar request fields win over template). See `templates.json` for the canonical hashes. **In the research harness, leave unset — the skill auto-selects `verl-research-vllm020`.** |
| `--no-default-filters` | off | pass `-n` to `vastai search` so its implicit `rentable=true verified=true` filters are disabled (your query is then fully authoritative) |
| `--dry-run` | off | print chosen offer(s) + would-be `vastai create` command(s) and exit 0 |
| `-h`, `--help` | — | print this guide |

## Stdout contract (machine-parseable)

```
VAST_HANDLE: {"created_at":"2026-05-25T12:30:00Z","dph_total":18.40,"gpu_name":"H100",...}
VAST_PROVISIONED: count=1 total_dph=18.4000
```

- One `VAST_HANDLE:` line per successfully provisioned instance. The
  payload is one-line JSON (jq -c), sorted by key.
- Exactly one terminal `VAST_PROVISIONED:` line.
- Anything else (auth banner, progress, errors) goes to **stderr**.

Downstream agents (`experiment-runner`) parse these lines via `grep
'^VAST_HANDLE: '` then `jq` — keep the prefix verbatim.

## Handle JSON schema (locked, `schema_version = "1"`)

```json
{
  "schema_version": "1",
  "instance_id":    "12345",
  "offer_id":       "98765",
  "ssh_host":       "ssh4.vast.ai",
  "ssh_port":       12345,
  "public_ipaddr":  "1.2.3.4",
  "gpu_name":       "H100",
  "num_gpus":       8,
  "dph_total":      18.40,
  "created_at":     "2026-05-25T12:30:00Z",
  "label":          "verl-research:<session-id>",
  "session_id":     "<session-id>",
  "ssh_login":      "ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 12345 root@ssh4.vast.ai -L 8080:localhost:8080"
}
```

`ssh_login` is the **paste-ready** connect command — the fixed SSH form every consumer must
use (`-i ~/.ssh/vast_ai_name` so the right key is offered; `StrictHostKeyChecking=accept-new` so a
reused Vast IP doesn't trip host-key verification). The skill also prints it to stderr on
provision (`log in FIRST, then work`). Never hand-build a bare `ssh -p <port> root@<host>`.

Field-name notes (cross-skill contract — do NOT silently rename):

- `num_gpus` (not `gpu_count`) — experiment-runner sums this to populate
  the ledger row's `total_gpus`.
- `dph_total` (not `dph`) — experiment-runner sums this for the ledger
  row's `dph`.
- `instance_id` — the int the `vastai` CLI calls `new_contract`. Stored
  as a string for forward compatibility.

If a field is missing because the vast.ai API didn't return it, the skill
falls back to the chosen offer's value (e.g. `gpu_name` / `num_gpus`)
rather than emitting an empty handle.

The skill writes the handle atomically (`mktemp` + `mv`) to
`<handle-dir>/<instance_id>.json` so a parallel `vast-teardown` never
sees a half-written file.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | all instances provisioned (or `--dry-run` succeeded) |
| 1 | missing CLI / env / credentials |
| 2 | bad arguments |
| 3 | no qualifying offers under the given filters (stderr names the cheapest rejected offer) |
| 4 | `vastai create instance` failed |
| 5 | instance never became running + SSH-routable inside `--timeout` |
| 6 | host-side failure detected (CDI device error, OCI runtime error, intended_status flipped to stopped) — destroy + retry on a different host |

The runner's tier-walk treats exit 3 as "advance to next tier" and other
non-zero codes as "log LAUNCH_FAILED_TIER and retry up to 3 times" — see
`experiment-runner.md` step 4.

## Behavior

1. **Auth**: `VAST_API_KEY` must be set in the calling shell (sourced
   from `~/.config/verl-research/secrets.env`). The skill logs that it
   read the var (without echoing the value); exits 1 if it's missing.
2. **Search**: `vastai search offers "$QUERY" -o dph_total --raw` →
   ascending-by-price JSON array.
3. **Filter**: drop offers missing price/reliability, then enforce
   `reliability2 >= --min-reliability` and `dph_total <= --max-price`.
   Optionally filter on `num_gpus == --gpu-count`.
4. **Pick cheapest N distinct hosts** (de-dup by `host_id`, fall back to
   `id`). If fewer than `--count` qualify, exit 3 with the cheapest
   rejected offer named in stderr.
5. **Create** each instance with `vastai create instance <offer> --image
   ... --disk ... --ssh --direct --cancel-unavail --label
   verl-research:<session-id> --raw`. `--cancel-unavail` is mandatory:
   we never want a stopped, billable orphan if the scheduler can't place
   us.
6. **Wait**: poll `vastai show instance <id> --raw` every
   `--poll-interval`s until `actual_status == "running"` and an SSH
   route (`ssh_host` + a host port mapped to container 22) is present.
   Prefer the direct route via `public_ipaddr` + `ports["22/tcp"][0].HostPort`;
   fall back to `ssh_host` + `ssh_port`.
7. **Handle write**: atomic `mktemp` → `mv`. Permissions left as the
   shell's default umask — no secrets in the file.
8. **Emit** `VAST_HANDLE:` line, then move to the next instance.

## Dry-run

`--dry-run` runs the search + filter + pick stages and prints one
`dry-run offer:` line per chosen offer (id, gpu_name, num_gpus, price,
reliability) and one `dry-run vastai command:` line with the would-be
argv. `--login`'s argument is redacted. No `vastai create` is called.

## Cross-skill contract with `vast-teardown`

`vast-teardown` reads `instance_id` from handle JSON files (`--handles
<path>` or positional ids), runs `vastai destroy instance <id>`, and
patches `runs.jsonl` rows. This skill must keep `schema_version = "1"`
and the `instance_id` field name in lockstep with `vast-teardown`'s
parser. Any schema change here requires a paired change there.

## Disk sizing (`--disk-gb`)

Vast.ai's disk is the host filesystem inside the container — image layers,
HF model cache, dataset, checkpoints, and vLLM KV cache all share it. Once
chosen at provision time it cannot be expanded without re-creating the
instance. Plan generously; over-provisioning by 20-50 GB is much cheaper
than paying for a re-pull.

Rule of thumb:

| Footprint | Size |
|---|---|
| `verlai/verl:vllm020.dev1` image (extracted) | ~30 GB |
| `/workspace/verl` git checkout + `pip install` | ~1 GB |
| HF model cache (Qwen2.5-0.5B → ~2 GB; Qwen3-1.7B → ~4 GB; Qwen3-4B → ~9 GB; Qwen3-32B → ~65 GB) | model-dependent |
| GSM8K parquet (train + test) | ~0.1 GB |
| Math/MATH dataset | ~0.5 GB |
| vLLM KV cache + intermediates at runtime | a few GB |
| Checkpoint per save (FSDP-sharded, bf16) | ~1-2× model size; multiply by `save_freq` if you keep more than one |

So `--disk-gb 200` (the default) covers ~30 (image) + ~10 (model up to 4B) + ~5 (data) + ~20-50 (vLLM + intermediates) + headroom — fine for the M0 baselines. Bump to **400-500 GB** when training Qwen3-32B+ or when you keep multiple checkpoints. Set tighter (e.g. `--disk-gb 80`) only for image-only smoke tests where nothing downloads a model.

When in doubt, ask `vastai show offers` for a feel for what's available — disk is priced into `dph_total` so doubling it can push you over `--max-price`.

## Choosing `cuda_max_good` correctly (driver gotcha)

The vast.ai offer field `cuda_max_good` is the **native** CUDA ceiling of the
host's NVIDIA driver. It is NOT the value `nvidia-smi` prints inside the
container — `nvidia-smi` reports the CUDA Forward Compatibility ceiling,
which is optimistic on workstation/consumer GPUs (forward-compat is only
honored on datacenter GPUs: A100/H100/L40/T4 with R450/R470/R535/R570 LTS).

Practical rule: **set the filter to match the image's torch CUDA suffix**.

| Image | torch | required `cuda_max_good` | typical driver |
|---|---|---|---|
| `verlai/verl:vllm020.dev1` | `2.11.0+cu130` | `>=13.0` | 580.x+ |
| `verlai/verl:vllm018.dev1` | `2.6.0+cu124` (approx) | `>=12.4` | 550.x+ |

If you provision with too-loose a filter, CUDA kernel launches return
`Error 803: system has unsupported display driver / cuda driver combination`.
Python imports still work because nothing has touched a kernel yet — the
failure surface is at `torch.cuda.get_device_name(0)` / vllm engine init.

The current Template (`3b0f8b726ac3036d6c007bfa13b6d75f`) encodes
`cuda_max_good >= 13.0` in its `extra_filters`, so when you provision via
`--template-hash 3b0f8b726ac3036d6c007bfa13b6d75f` you get this filter for
free. (Always re-check `templates.json` for the current hash — the hash
mutates on every template update.)

## Templates (vast.ai's saved provisioning configs)

A vast.ai **Template** is a server-side JSON record bundling `image + tag +
onstart + env + ports + ssh_direct + recommended_disk_space +
extra_filters` and is referenced by `hash_id` (a 32-hex string). The
research harness owns one Template:

| Name | hash_id | id | Purpose |
|---|---|---|---|
| `verl-research-vllm020` | `3b0f8b726ac3036d6c007bfa13b6d75f` | `447527` | base for all M0+ verl GRPO experiments on shamanez/verl@vast-ai-workload |

The Template's onstart script (`onstart.verl-vllm020.sh` in this dir) clones
`https://github.com/shamanez/verl @ vast-ai-workload` into `/workspace/verl`
and runs `pip install --no-deps -e .`. The `--no-deps` is load-bearing: the
verlai image's torch / vllm / megatron / transformer-engine / deepep versions
were CI-validated together by verlai. Touching them at install time would
silently break vllm rollouts at training time, so the onstart hard-fails if
they drift.

The Template's `env` field (Vast's "Docker options") additionally carries:

```
--shm-size=10g --cap-add=SYS_ADMIN
-e PYTHONUNBUFFERED=1
-e TOKENIZERS_PARALLELISM=false
-e RAY_DISABLE_USAGE_STATS=1
```

The first two come from verl's official docker docs. Vast.ai silently
strips `--cap-add=SYS_ADMIN` (CapEff bit 21 stays unset) and the
`--shm-size` is ignored in favor of the host's default (typically 50%
of RAM — much larger than 10g anyway, so harmless). The three `-e` flags
DO take effect and trim Ray's idle thread footprint, which matters on
hosts with low cgroup pids caps (see "Container limits" below).

Maintenance:

```bash
# inspect (active templates owned by this account)
curl -s "https://console.vast.ai/api/v0/users/me/templates/?api_key=$VAST_API_KEY" \
  | jq '.templates[] | select(.name == "verl-research-vllm020")'

# update — DANGER: omitted fields are nulled out. ALWAYS pass the full
# field set on every update (name, image, onstart-cmd, ssh, direct,
# disk_space, search_params, env, desc). Test the resulting hash via
# the API listing above before promoting it into templates.json.
vastai update template <old-hash> \
  --name verl-research-vllm020 \
  --image verlai/verl:vllm020.dev1 \
  --env '--shm-size=10g --cap-add=SYS_ADMIN -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false -e RAY_DISABLE_USAGE_STATS=1' \
  --onstart-cmd "$(cat onstart.verl-vllm020.sh)" \
  --ssh --direct --disk_space 200 \
  --search_params 'verified=true rentable=true external=false num_gpus>=1 cuda_max_good>=13.0 reliability>=0.97' \
  --desc 'shamanez/verl@vast-ai-workload on verlai/verl:vllm020.dev1 ...'

# safer: create a new template and delete the old one
vastai create template ...     # (same flags as update)
vastai delete template --template-id <orphan-template-id>
```

**Warning learnt the hard way (EXP-vast-1p5b-smoke):**

1. `vastai update template <hash>` REPLACES the record with the fields
   you pass. Any field you don't include (`--name`, `--image`,
   `--onstart-cmd`, `--ssh`, `--direct`, `--disk_space`, ...) gets nulled
   out, leaving a broken template. Always pass the full set.
2. `vastai delete template --template-id <N>` takes the numeric
   `template_id`, not the `hash_id`. Different versions of the same
   template can share an `id`, so always read both `id` and `hash_id`
   from the API before deleting. We bricked an active template once by
   passing the wrong id.
3. The hash is content-derived: rebuilding a template with identical
   fields gives back the same `hash_id`. This means `templates.json`
   stays stable across template recreates if the content is unchanged.

## Container limits on Vast.ai (cgroup PIDs cap — read this)

Vast.ai sets each container's cgroup PIDs limit via the host's docker
daemon `--pids-limit` flag, which is **not template-configurable** and
**read-only from inside the container**:

```bash
$ cat /sys/fs/cgroup/pids/pids.max
1792                                        # observed on host machine_id=16297
7680                                        # observed on host machine_id=43919
$ echo 65535 > /sys/fs/cgroup/pids/pids.max
bash: ... Read-only file system             # even with SYS_ADMIN requested
```

Why this matters: verl's full FSDP + vLLM + Ray stack on a single GPU
**peaks at ~1700 threads** at the FSDP→vLLM weight-transfer boundary
(Ray raylet + dashboard sub-modules + vLLM `multiproc_executor` +
EngineCore + ZMQ bucketed weight transfer's per-call sockets, each
spawning ZMQ I/O threads via `pthread_create`).

| Host pids.max | Outcome |
|---|---|
| `1792` | Dies inside `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py:169` at `zmq_socket()` with SIGABRT. Surface error: `Resource temporarily unavailable (src/thread.cpp:241)` → `EngineCore: Executor failed.` → `ray.exceptions.ActorDiedError`. Misleadingly looks like a thread / FD limit. |
| `7680` | Works — peak observed 4226/7680 (55%) during a Qwen2.5-1.5B GRPO step. |

What does NOT help on a too-tight host (we tried them all):

- `ulimit -n 65535` / `ulimit -u 65535` — bumps the wrong limits
- `RAY_DISABLE_DASHBOARD=1` + `OMP_NUM_THREADS=1` + `MKL_NUM_THREADS=1`
  + `TOKENIZERS_PARALLELISM=false` — shaved ~300 threads but still hit
  the 1792 ceiling
- Tighter rollout config (`ROLLOUT_N=1`, smaller batch, smaller token
  budgets) — peak went from 1700+ to 1650, still over
- `--cap-add=SYS_ADMIN` in template's docker-options — Vast silently
  strips it (CapEff bit 21 remains unset)
- Switching to `actor_rollout_ref.rollout.name=hf` — fails because
  verl's `_ROLLOUT_REGISTRY` only registers `(vllm, async)`,
  `(sglang, async)`, `(trtllm, async)`; `hf` / `naive` are not
  selectable through the trainer config

What DOES help: **filter to hosts with generous pids.max**. There is
no Vast.ai search filter for `pids.max` — you have to provision and
check. Recommended pattern:

```bash
# 1. provision
bash $CLAUDE_PROJECT_DIR/.claude/skills/vast-provision/run.sh \
     --template-hash <hash> --query "..." --max-price ...

# 2. immediately after VAST_HANDLE: probe the host
HANDLE=$(ls $CLAUDE_PROJECT_DIR/.claude/state/vast-handles/*.json | tail -1)
HOST=$(jq -r .ssh_host  "$HANDLE")
PORT=$(jq -r .ssh_port  "$HANDLE")
PIDS_MAX=$(ssh -i ~/.ssh/vast_ai_name -p $PORT root@$HOST \
              'cat /sys/fs/cgroup/pids/pids.max')

# 3. if <= 1792, tear down and retry on a different host
if (( PIDS_MAX <= 2048 )); then
    iid=$(jq -r .instance_id "$HANDLE")
    bash $CLAUDE_PROJECT_DIR/.claude/skills/vast-teardown/run.sh \
         --reason "pids-max=$PIDS_MAX too tight for verl vLLM stack" "$iid"
    # retry with a different host (use --query 'machine_id=...' to exclude)
fi
```

Worth adding to `experiment-runner`'s post-provision smoke phase
eventually; for now it's manual.

## What this skill deliberately does NOT do

- Push `VAST_API_KEY` to the instance. The bootstrap (HF / WandB
  login, repo sync, container start) happens **on the instance**, via
  the locked Vast.ai template's onstart script (see
  `onstart.verl-vllm020.sh` in this directory and the `image` /
  `docker_options` fields in `templates.json`). `HF_TOKEN` and
  `WANDB_API_KEY` are forwarded via `-e VAR=...` from the laptop env;
  `VAST_API_KEY` is never forwarded.
- Run remote commands. Use `experiment-runner` step 6+ for that.
- Tear down. That's `vast-teardown`'s job (or the `teardown-finished-runs.sh`
  Stop hook).
- Patch `runs.jsonl`. That's `experiment-runner` step 5.
- Retry on tier exhaustion. The runner walks the chain; the skill stays
  single-purpose.
