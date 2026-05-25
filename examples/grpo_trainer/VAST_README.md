# Vast.ai launchers (`vast_*.sh`)

Branch: **`vast-ai-workload`** on `shamanez/verl`. This is the stable home
for vast.ai-specific GRPO smoke + research launchers. Upstream
`verl-project/verl` knows nothing about these files; we keep the fork's
mainline mergeable with upstream by isolating fork-specific scripts to
this branch and to the `vast_*` naming convention.

## Convention

Filename: `vast_<scenario>_<model>_<algo>_<dataset>.sh`. Examples:

- `vast_smoke_qwen25_0p5b_grpo_gsm8k.sh` — single-GPU smoke
- `vast_baseline_qwen3_4b_grpo_gsm8k.sh` — M0 baseline (to be added)
- `vast_baseline_qwen3_1p7b_grpo_gsm8k.sh` — cheap-tier baseline (to be added)

Each launcher:

1. Sources `~/.config/verl-research/secrets.env` on the box (HF + WandB
   only; never reads `VAST_API_KEY`).
2. Hard-fails if `VAST_API_KEY` is present on the box (defense-in-depth
   against accidentally scp'ing the laptop's full secrets file).
3. Reuses upstream's per-model launcher (e.g.
   `run_qwen3_4b_fsdp.sh`) and overrides Hydra knobs via CLI. This keeps
   the upstream-mergeable surface narrow.

## How this composes with the harness

The `verl-research-vllm020` vast.ai Template (in
`research/.claude/skills/vast-provision/templates.json`) clones THIS branch
of THIS fork into `/workspace/verl` at instance start and runs
`pip install --no-deps -e .` (preserves the verlai image's bundled torch /
vllm / megatron / TE / deepep — drift would silently break vllm rollouts).

The `vast-provision` skill provisions an instance from the Template. From
that point on, **everything** the box needs to run an experiment is in this
git checkout — no scp, no `/tmp` scripts, no laptop-side file transfers.

## Iteration loop (e.g. fitting GPUs, fighting OOM)

```text
laptop:  edit examples/grpo_trainer/vast_smoke_qwen25_0p5b_grpo_gsm8k.sh
laptop:  git commit -am "smoke: drop batch to 4 for OOM"
laptop:  git push origin vast-ai-workload
box:     cd /workspace/verl && git pull
box:     bash examples/grpo_trainer/vast_smoke_qwen25_0p5b_grpo_gsm8k.sh
```

Repeat. No need to re-provision the box for code changes — the verl
install is editable (`pip install -e .`), so a `git pull` is enough.

## Why the protect-upstream hook allows writes here

The research harness's `research/.claude/hooks/protect-upstream.sh` refuses
edits under `verl/` unless the current branch matches `exp/*` OR
`vast-ai-workload` (the named exception added when this branch became
the home for fork-specific launchers).

## Secret hygiene checklist on the box

- [ ] `~/.config/verl-research/secrets.env` contains ONLY `HF_TOKEN` and
      `WANDB_API_KEY`.
- [ ] `grep -E '^export (VAST|STREAMS_VAST)' ~/.config/verl-research/secrets.env`
      returns nothing (exit 1). Note the `^export ` prefix — the laptop's
      secrets file uses `export NAME=...` form, so the bare-name grep
      misses everything and produces a zero-byte stripped file.
- [ ] Inside the running container (or current SSH session) `env | grep
      ^VAST` returns nothing.

## Vast.ai host gotcha — check `cgroup/pids.max` BEFORE running the smoke

The verl FSDP + vLLM + Ray stack peaks at ~1700 pthreads at the
FSDP→vLLM weight-transfer boundary. Vast.ai hosts vary in their docker
daemon `--pids-limit` configuration (observed 1792 vs 7680 on two
different A100-80GB hosts). On a 1792 host the smoke dies inside
`verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` with the
misleading `Resource temporarily unavailable (src/thread.cpp:241)` ZMQ
error. **Always probe before running:**

```bash
ssh -i ~/.ssh/vast_ai -p <port> root@<host> 'cat /sys/fs/cgroup/pids/pids.max'
# >= 4096 → safe to run
# 1792    → tear down, provision a different host
```

The cgroup file is read-only from inside the container even with
`--cap-add=SYS_ADMIN` (Vast silently strips it). The only fix is
host-selection. Full diagnostic notes in
`research/.claude/skills/vast-provision/SKILL.md` ("Container limits on
Vast.ai (cgroup PIDs cap)" section) and the post-mortem at
`research/runs/EXP-vast-1p5b-smoke/PROGRESS.md`.

## ulimit -n bump (already in the launcher)

`vast_smoke_qwen25_*.sh` bump `ulimit -n 65535` before any python3 call.
The default 1024 makes verl's Ray + vLLM ZMQ socket allocation hit
EMFILE during EngineCore init. The bump is necessary but not sufficient
— the cgroup pids cap is the harder wall (see previous section).
