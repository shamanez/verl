# Dense baseline reproducibility manifest

This run **MUST** be reproducible by any teammate. The contract: re-running the
launcher below from the recorded commit, on the same template, should produce
the same WandB curve and final val score (±noise from rollout sampling).

## Source-of-truth: the launcher

| Location | Commit (HEAD) | File blob SHA | sha256(file) |
|---|---|---|---|
| Local laptop | `b4566654` (vast-ai-workload) | `b600dfaeb2eddca7c5846cb239ef7a44b417e5ee` | `b6d4d59440c6a8bf8596a06d8d26111e6db349d04a8e727e41384592bd6c355c` |
| Remote `origin/vast-ai-workload` | `81e4ab32` | `b600dfaeb2eddca7c5846cb239ef7a44b417e5ee` | `b6d4d59440c6a8bf8596a06d8d26111e6db349d04a8e727e41384592bd6c355c` |
| Vast box `/workspace/verl/` | `81e4ab32` | `b600dfaeb2eddca7c5846cb239ef7a44b417e5ee` | `b6d4d59440c6a8bf8596a06d8d26111e6db349d04a8e727e41384592bd6c355c` |

**All three blobs are bit-identical.** The local laptop is one commit ahead of remote (`b4566654` is an autosave commit that does NOT touch the launcher) — `git show b4566654 -- examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` returns nothing. The Vast box ran exactly the launcher that's on `origin/vast-ai-workload@81e4ab32`.

Verbatim snapshot of the launcher used by this run is checked in alongside this manifest at `launcher.snapshot.sh` (same `sha256: b6d4d594…`).

## Container provenance

| | |
|---|---|
| Vast template name | `verl-research-vllm020` |
| Vast template hash | `6485b9625ddd6d25a5f2f09b9f7fde17` |
| Docker image | `verlai/verl:vllm020.dev1` |
| Bundled torch | `2.11.0+cu130` |
| Bundled vllm | `0.20.2` |
| Onstart action | clones `shamanez/verl @ vast-ai-workload` into `/workspace/verl`, runs `uv pip install --no-deps -e .` |

## Compute

| | |
|---|---|
| GPU | 4 × H200 (141 GB HBM each) |
| Vast instance id | `37881404` |
| Vast machine id | `100313` |
| Tier chosen | `chosen_tier_idx=1` (`num_gpus=4 gpu_name=H200 gpu_ram>=140 cuda_max_good>=13.0 reliability>=0.97`) |
| Hourly cost | $16.054/hr (`dph_total`) |

## Re-running

From any laptop with `~/.config/verl-research/secrets.env` (HF + WandB + VAST keys) and the SSH key registered on Vast:

```bash
cd /path/to/verl
git fetch origin
git checkout vast-ai-workload
git reset --hard 81e4ab32                       # the exact commit this run used

# Provision an identical box
source ~/.config/verl-research/secrets.env
bash research/.claude/skills/vast-provision/run.sh \
  --query 'num_gpus=4 gpu_name=H200 gpu_ram>=140 cuda_max_good>=13.0 reliability>=0.97 rentable=true verified=true' \
  --max-price 24.0 --count 1 --disk-gb 200

# On the box (template's onstart already cloned the repo + installed verl):
ssh -i ~/.ssh/vast_ai -p <port> root@<host> '
  tmux new -ds grpo-baseline "
    cd /workspace/verl &&
    git pull &&
    bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh
  "
'
```

The launcher reads its hyperparameters from env-vars with defaults pinned in
the file (see `launcher.snapshot.sh` here). Override one only with intent;
unknown overrides will diverge from this baseline.

## Resulting HF checkpoints (private)

- `gshasiri/qwen25-1p5b-grpo-gsm8k-baseline-step50` — global_step 50
- `gshasiri/qwen25-1p5b-grpo-gsm8k-baseline-step100` — global_step 100 (final, since this run was stopped at step 100 to conserve budget)
