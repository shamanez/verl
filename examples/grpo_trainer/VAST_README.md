# Vast.ai communication-efficient GRPO launchers

These launchers run the current qboot-v2 pipeline on the
`Qwen/Qwen2.5-Math-1.5B` + MATH train/test surface. The reference command is:

```bash
bash examples/grpo_trainer/run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh composite
```

The explicit launcher enables communication efficiency. Generic Hydra
configuration keeps `actor_rollout_ref.actor.comm_eff.enabled=false`, so code
outside this launch path remains dense by default.

## Current launcher map

| Launcher | Purpose |
| --- | --- |
| `run_qwen25_math_1p5b_relex_qboot_v2_comparison_fsdp.sh composite` | Latest completed qboot-v2 reference arm |
| `run_qwen25_math_1p5b_rank1_relex_fsdp.sh` | Direct Qwen-Math/MATH method launcher |
| `vast_comm_eff_engine_grpo.sh` | Shared FSDP + vLLM engine used by the MATH wrappers |
| `run_qwen25_math_1p5b_relex_comparison_fsdp.sh dense` | Dense control on the same model, data, and GRPO surface |

The current method settings are:

- PowerSGD activation projection at rank 77 with synchronized, warm-started,
  activation-derived `Q` and fast-Q bootstrap;
- a paired dense anchor at cadence/delay 20/20 optimizer ticks, scoped to one
  PPO mini-batch, with replay/snapshots on CPU and anchor-owned `Q`;
- RELEX rank-1 weight projection with progressive W2 → W3 → W4 history,
  window 4/minimum 2, strength 1, `auto` trajectory routing, and
  `stale_correct`;
- signed-EMA `M` over all floating targets, beta 0.50, alpha 0.25, stored on
  CPU.

See `examples/grpo_trainer/COMM_EFF_CONFIG.md` for the current run defaults.

## Preparing a Vast.ai box

### Locked template

Provision with the `verl-research-vllm020` entry in
`research/.claude/skills/vast-provision/templates.json`. Its image supplies the
CUDA, PyTorch, and vLLM stack, and its startup installs this checkout in editable
mode with `--no-deps`.

The provisioning tools place only the runtime Hugging Face, W&B, and R2
credentials required by the harness. The launcher reads
`~/.config/verl-research/secrets.env`; the template startup does not embed
secrets.

### Operator-provided box

For a bare instance:

1. Copy the private checkout to `/workspace/verl` without `.git`, local run
   outputs, caches, or laptop credentials.
2. Install the FSDP + vLLM dependency subset supported by this checkout, then
   run `pip install --no-deps -e /workspace/verl`.
3. Create `~/.config/verl-research/secrets.env` with only the runtime
   credentials needed on the box and set mode `600`.
4. Prepare MATH train/test parquet files under `/workspace/data/math`, or set
   `DATA_DIR` to another prepared directory.
5. Run the explicit reference launcher above from `/workspace/verl`.

The shared engine can prepare MATH when its `DATA_DIR` lacks `train.parquet`
and `test.parquet`. Reference experiments should pre-prepare those files using
the canonical project data path so their inputs are deliberate and auditable.

## What the launcher pins

The Qwen-Math wrapper pins the scientific surface before delegating to the
shared engine:

| Area | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-Math-1.5B` |
| Data | Prepared MATH train/test parquet files |
| Train / PPO mini-batch | 512 / 256 prompts |
| Rollouts per prompt | 8 |
| Prompt / response length | 1024 / 3072 tokens |
| Actor learning rate | `1e-6` |
| Reference KL | `low_var_kl`, coefficient `0.001` |
| Total steps / validation cadence | 100 / 25 |
| Activation projection | PowerSGD rank 77 |
| Anchor cadence / delay | 20 / 20 optimizer ticks |
| RELEX | `rank1_relex`, W4 retention, minimum 2, strength 1 |
| Anchor gradient state | Signed EMA, all floating, beta 0.50, alpha 0.25, CPU |

Every launcher prints its resolved model, data files, GRPO shape, and
communication-efficient settings before starting Python. Training also runs
under shell tracing, so the fully expanded Hydra command is present in
`train.log`.

## Canonical launcher and experiment overrides

An experiment should call the existing launcher and override only the variable
being tested. For example, to test the larger anchor data scope while retaining
the current reference everywhere else:

```bash
COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
  EXPERIMENT_NAME=qboot_v2_rollout_scope \
  bash examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh
```

Hydra overrides may be appended after the launcher arguments when an
environment variable is not exposed. The resolved command, rather than a
handwritten run note, is authoritative for what executed.

## Branch bootstrap on the box

The template can use a restricted fetch refspec, so fetching a branch may only
update `FETCH_HEAD`. Bootstrap a requested branch from that object and verify
the resulting commit:

```bash
git fetch origin "$BASE_BRANCH" || exit 1
git checkout -B "$BASE_BRANCH" FETCH_HEAD
want=$(git ls-remote origin "refs/heads/$BASE_BRANCH" | cut -f1)
[[ -n "$want" && "$(git rev-parse HEAD)" == "$want" ]] || exit 1
```

Checking the commit is essential because `checkout -B` always gives the local
branch the requested name, even when the fetched object is not the intended
remote tip.

## Iteration loop

The checkout is installed editable, so code or launcher updates do not require
reprovisioning:

```text
laptop: update and push the experiment branch
box:    cd /workspace/verl && fetch the exact branch commit
box:    run the Qwen-Math/MATH launcher
```

Keep run-specific configuration in its experiment branch and run directory.
Only a human-reviewed, validated configuration should change the canonical
launcher defaults.

## Host checks

### Process limit

The FSDP + vLLM + Ray stack can approach the container process limit. Check the
host before a long run:

```bash
cat /sys/fs/cgroup/pids/pids.max
```

Use a host with at least 4096 available process IDs. A low container limit is
not repairable from inside the container; reprovision on a different host.

### File descriptors

The shared engine raises the file-descriptor limit to 65535 before launching
Python. This is required for Ray and vLLM socket creation but does not replace
the process-limit check.

## Secret hygiene

- Keep the on-box secrets file owner-readable only (`chmod 600`).
- Never place laptop provisioning credentials on the training box.
- Never put secret values in launch arguments, logs, run manifests, commits, or
  issue comments.
- Verify the running environment contains only the credentials required by the
  training process.
