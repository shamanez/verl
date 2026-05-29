# Plan: baseline

> **The dense control.** verl run with **no modifications** — vanilla GRPO,
> Qwen2.5-1.5B-Instruct on GSM8K. Every compression experiment overlays its curve
> against this reference. () Result:
> `val/test_score` 0.0872 → 0.7892 over 100 steps on 4×H200.

## Experiment
- id:            baseline
- title:         Qwen2.5-1.5B-Instruct dense GRPO baseline on GSM8K (verl, unmodified)
- issue:         https://github.com/shamanez/verl-compression-research/issues/3
- kind:          experiment
- milestone:     M1
- created_at:    2026-05-26T00:00:00Z
- baseline_run:  none   # this IS the baseline

## Kind (drives orchestrator routing)

`experiment` — runs on Vast.ai, analyst executes the predicate, log-writer files a verdict comment + LOG entry on completion. No code change, so no draft PR.

## Hypothesis

A clean dense GRPO baseline of **Qwen2.5-1.5B-Instruct** on **GSM8K** runs end-to-end on a Vast.ai multi-GPU instance (**4..8 H100/H200**, never single-GPU), provisioned from the locked `verl-research-vllm020` template (hash `6485b9625ddd6d25a5f2f09b9f7fde17`) which auto-clones `shamanez/verl @ vast-ai-workload` and pip-installs verl editable. The launcher `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` (committed on that branch) runs **2 epochs over the GSM8K train split** (~7.5K prompts, ~117 GRPO optimizer steps total) with rollout `n=8`, prompt context `1024`, response context up to `16384`, global train batch `128`, PPO mini-batch `64`, dynamic-token-batched microbatching (`ppo_max_token_len_per_gpu=36864`), FSDP without offload, gradient checkpointing on, vLLM TP=2, validating on the GSM8K test split every 25 steps. Falsifiable thresholds (all must hold): within **12 hours wall-clock** and under **$60 total Vast.ai spend**, the run completes both epochs with **no NaN/Inf in any loss/grad/reward field**, **`train/reward_mean` strictly increases by ≥ 0.05** between the first 25-step window and the final logged step, **`train/grad_norm` median over the run < 5.0**, and **`val/test_score` improves by ≥ +0.05** (5 percentage points pass@1) between step 0 and the final eval. WandB logs the full curve under project `verl_compression_research` / experiment `qwen25_1p5b_grpo_gsm8k_baseline`. This produces the dense reference curve later compression experiments overlay against. **Not a smoke test.**

## Background pointers
- prior findings: (none) — this is the first real experiment on the harness
- referenced docs:
  - `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` (the committed launcher on `vast-ai-workload`)
  - `examples/grpo_trainer/VAST_README.md` (vast-ai launcher conventions)
  - `research/.claude/skills/vast-provision/templates.json` (locked Vast.ai template registry)
  - `research/.claude/skills/vast-provision/SKILL.md` (`cuda_max_good>=13.0` rationale)
  - Vast template `verl-research-vllm020` hash `6485b9625ddd6d25a5f2f09b9f7fde17` (image `verlai/verl:vllm020.dev1`, torch 2.11.0+cu130, vllm 0.20.2)

## Experiment design
```yaml
sweep_grid:
  # No sweep — this is a single dense baseline configuration. The model/algorithm/batch knobs are
  # all fixed (see "Model + training config" table below). The only runner-driven dimension is
  # the GPU tier chosen from gpu_filter_chain, which is recorded in the PROVISIONED ledger row.
  (n/a): []
baselines:
  - dense_grpo_qwen25_1p5b_gsm8k    # this run IS the dense reference; no further baseline to compare against
ablations: []                        # ablations belong to later issues with depends_on: [baseline]
seed_replicates:  1                  # single seed for the reference curve; replicates come later if needed
fanout_max:       1                  # one cell, one instance
```

## Compute budget (HARD CAPS)
```yaml
gpu_count:        1                       # one Vast.ai instance (multi-GPU within it)
gpu_filter_chain:                         # multi-GPU only (4..8); cheapest-first per issue body
  - "num_gpus=4 gpu_name=H100 gpu_ram>=80 cuda_max_good>=13.0 reliability>=0.97 rentable=true verified=true"   # preferred: cheapest 4×80GB tier
  - "num_gpus=4 gpu_name=H200 gpu_ram>=140 cuda_max_good>=13.0 reliability>=0.97 rentable=true verified=true"  # H200 4× — ~30% pricier but ~1.5× faster
  - "num_gpus=8 gpu_name=H100 gpu_ram>=80 cuda_max_good>=13.0 reliability>=0.97 rentable=true verified=true"   # 8×H100 fallback when 4× is dry
  - "num_gpus=8 gpu_name=H200 gpu_ram>=140 cuda_max_good>=13.0 reliability>=0.97 rentable=true verified=true"  # 8×H200 last (most expensive)
max_dph:          24.0                    # per-instance $/hr ceiling (covers all four tiers)
max_gpu_hr:       96                      # 12 h wall-clock × 8 GPUs ceiling
max_parallel:     1
wall_clock_hr:    12
iterations:       2                       # REVISE child-experiment depth cap on this lineage
```

The chain above is the **issue-overridden** chain (not the planner default). All tiers carry `cuda_max_good>=13.0` because the `verlai/verl:vllm020.dev1` image ships torch 2.11.0+cu130 — sub-13 drivers silently break vLLM at engine init. All tiers carry `reliability>=0.97` (tighter than the default 0.95) because a 12-hour run cannot afford a mid-run host eviction. All tiers carry `num_gpus` in {4, 8} — single-GPU offers are HARD-FAILED by the launcher.

If all four tiers exhaust with zero offers under `max_dph`, the runner appends `MANUAL_REVIEW_NEEDED: no offers in any tier` to PROGRESS.md and stops.

The template's image, container `--shm-size=10g --cap-add=SYS_ADMIN`, onstart script (clones `shamanez/verl@vast-ai-workload` and `pip install --no-deps -e .`), disk default (200 GB), and driver filter are all owned by the locked Template referenced in `research/.claude/skills/vast-provision/templates.json`. The plan MUST NOT name `template_hash` or `image`; the `vast-provision` skill auto-reads `templates.json` and pins the Template.

## Success criteria

### Provision + access
- [ ] Instance provisioned via the `verl-research-vllm020` Template (hash `6485b9625ddd6d25a5f2f09b9f7fde17`) — verified by `vastai show instance <id> --raw | jq .template_hash_id` matching the recorded hash
- [ ] `nvidia-smi -L | wc -l` inside the container returns a value in **{4, 5, 6, 7, 8}**; single-GPU offers are HARD-FAILED by the launcher
- [ ] `cat /sys/fs/cgroup/pids/pids.max` inside the container is `>= 4096` (Vast cgroup PIDs gotcha — the launcher hard-fails on `<= 2048`)
- [ ] `/workspace/verl` exists and `git -C /workspace/verl rev-parse --abbrev-ref HEAD` returns `vast-ai-workload`
- [ ] `python3 -c 'import verl, vllm, torch; print(verl.__file__, vllm.__version__, torch.__version__)'` returns the bundled image versions (no pip drift)

### Secret hygiene
- [ ] On the box (outside the container, then inside): `grep -E '^export VAST' ~/.config/verl-research/secrets.env` returns nothing (exit 1)
- [ ] `env | grep -E '^VAST'` inside the running launcher returns nothing
- [ ] `HF_TOKEN` + `WANDB_API_KEY` both present and non-empty inside the launcher

### Training
- [ ] Training log contains a `global_step` value `>= 100` (end-of-epoch-2 is ~step 116; analyst greps for the highest `global_step` and confirms `>= 100`)
- [ ] No line containing `loss`, `grad_norm`, `policy_loss`, or `reward` contains `nan`, `NaN`, `inf`, or `Inf`
- [ ] `train/reward_mean` at the final logged step exceeds `train/reward_mean` at step 25 by **≥ 0.05** (positive learning signal)
- [ ] `train/grad_norm` stays finite throughout and **median over all logged steps < 5.0**
- [ ] At least one validation eval ran (one or more rows tagged `val/...` in `train.log` or WandB)
- [ ] Final `val/test_score` exceeds the step-0 `val/test_score` by **≥ +0.05** (5 percentage points pass@1 on GSM8K test)
- [ ] WandB run exists under project `verl_compression_research` with experiment name `qwen25_1p5b_grpo_gsm8k_baseline` and shows **≥ 100** step rows

### Budget
- [ ] Total Vast.ai spend on this instance **≤ $60** from `vastai create` to last criterion verification
- [ ] Total wall-clock from launcher start to final eval **≤ 12 hours**

All criteria are machine-checkable — the analyst greps `runs/baseline/metrics/*.jsonl`, the rsynced `train.log`, the Vast.ai instance ledger, and queries WandB for the literals above.

## Verification commands

The analyst runs exactly these commands and captures stdout to `runs/baseline/analysis.log`.

```bash
# 1. Train log + numeric criteria
grep -E 'global_step|nan|NaN|inf|Inf|reward_mean|grad_norm|val/' \
    runs/baseline/metrics/train.log | head -200

python research/scripts/analyze.py runs/baseline --emit verdict.md

# 2. Budget + wall-clock
python research/scripts/check_budget.py runs/baseline

# 3. No baseline diff yet (this run IS the dense baseline). Skip diff_against_baseline.py.

# 4. WandB sanity (run row count + val/test_score start vs final)
wandb run --project verl_compression_research --name qwen25_1p5b_grpo_gsm8k_baseline

# 5. Secret-hygiene grep on the rsynced train.log + on-box env dump
grep -E 'VAST_API_KEY|VAST_API|^VAST=' runs/baseline/metrics/train.log || echo "OK: no VAST leak in log"

# 6. Template hash attribution (from the PROVISIONED ledger row)
jq -r '.template_hash_id' runs/baseline/handles/*.json
```

## Analyst predicate

- **PASS** iff every box in `## Success criteria` is checked.
- **REVISE** if at most `iterations` (=2) boxes are unchecked AND the analyst can name a concrete next-action knob change for each. Output `next_actions:` as a yaml list of `{knob, from, to, rationale}` objects. Preferred REVISE knobs (cheapest first):
  - `{knob: KL_LOSS_COEF, from: 0.001, to: 5e-4, rationale: "allow more policy drift if reward_mean plateaus"}`
  - `{knob: ACTOR_LR, from: 1e-6, to: 3e-6, rationale: "bump LR if grad signal is too small"}`
  - `{knob: ROLLOUT_N, from: 8, to: 16, rationale: "lower-variance advantage estimate if reward is noisy"}`
  - `{knob: TOTAL_EPOCHS, from: 2, to: 3, rationale: "cheap extra epoch if curve is still improving at end"}`
  - On OOM only: `{knob: actor.fsdp_config.param_offload, from: false, to: true, rationale: "CPU-offload the actor optimizer state before reducing MAX_RESPONSE_LENGTH — user mandate forbids reducing 16K"}`
- **STOP** if any of:
  - `train/reward_mean` is flat or decreasing across the whole 2 epochs — recipe is broken on this hardware tier; route to codex `code-rescue` with `train.log`
  - hypothesis is falsified on the headline numeric thresholds (`val/test_score` delta `< 0`, or NaN/Inf in loss/grad/reward)
  - budget exhausted (`spend > $60` OR `wall_clock > 12 h`)
  - 2 REVISE cycles already consumed on this lineage

## Code change
```yaml
code_change: false
target_modules: []
```

The launcher is already committed on `vast-ai-workload` and the template's onstart `git pull`s it. No `verl/` patch in this experiment — this is the dense reference. Compression code changes come in later issues with `code_change: true` and a `codex-bridge --mode=verify` gate.

## Dependencies
```yaml
depends_on: []
```

This is the first real experiment. No prior PASS required.

## Rescue triggers
```yaml
escalate_to_codex_if:
  - "VAST_API_KEY found in container env"
  - "VAST key leaked into container"
  - "VERIFY_TIMEOUT:"
  - "no offers in any tier"
  - "cgroup pids.max .* too tight"
  - "RuntimeError: CUDA out of memory"
  - "torch.distributed"
  - "vllm.engine"
  - "NaN detected"
  - "Error 803: system has unsupported display driver"
  - "MANUAL_REVIEW_NEEDED: vast-provision template auto-default missing"
```

The orchestrator greps PROGRESS.md each tick for these patterns and routes to `codex-bridge` in the appropriate mode (security-rescue for the VAST leak patterns, code-rescue for the runtime / CUDA / NaN patterns, provision-rescue for the offers/template patterns).

## Notes for runner

- **Provision via the skill only.** Call `/vast-provision count=1 query="<tier>" disk_gb=200 max_price=24.0` per tier; let the skill auto-select the locked Template from `templates.json`. Do NOT pass `--template-hash` or `--image`. Before the offer search starts, confirm the stderr line `vast-provision: auto-selected template 'verl-research-vllm020' hash=6485b9625ddd6d25a5f2f09b9f7fde17 image=verlai/verl:vllm020.dev1` appears; if not, abort and append `MANUAL_REVIEW_NEEDED: vast-provision template auto-default missing` to PROGRESS.md.
- **Walk the chain cheapest-first.** Stop at the first tier with ≥1 offer ≤ `max_dph`. Record the chosen tier on the PROVISIONED ledger row so the analyst can attribute results.
- **Teardown via the skill only.** `.claude/skills/vast-teardown/run.sh <instance_id>` or the `teardown-finished-runs.sh` Stop hook. Never call `vastai destroy instance` directly.
- **Do not change the template.** The template's `verl_repo`/`verl_branch` are the only valid source of `/workspace/verl`. Code iteration flows: laptop edit → `git push origin vast-ai-workload` → `git pull` on the box → re-run the launcher. **No scp'd scripts. No `/tmp/` workarounds. No per-experiment template recreation.**
- **Stripped secrets are mandatory.** Push only `HF_TOKEN` + `WANDB_API_KEY` to `~/.config/verl-research/secrets.env` on the box. The laptop's `VAST_API_KEY` MUST NOT cross the boundary. The launcher hard-fails if it sees it.
- **Multi-GPU is mandatory.** If a Vast offer somehow provisions <4 GPUs, the launcher exits 1 immediately. Tear down and retry on the next offer or next tier.
- **Probe `cgroup/pids.max` before launch.** Vast's docker daemon caps process counts at `--pids-limit` (host-specific, NOT template-configurable). On hosts with `pids.max <= 2048` the verl FSDP + vLLM + Ray stack dies inside `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` with a misleading ZMQ error. The launcher refuses to start on such hosts. Re-provision on a different `machine_id`.
- **Launch under tmux** on the remote host so the launcher survives SSH disconnect:
  ```bash
  ssh -p <port> root@<host> 'tmux new -ds grpo-baseline "cd /workspace/verl && git pull && bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh"'
  ```
- **`done.flag`** — the launcher touches `runs/qwen25_1p5b_grpo_gsm8k_baseline/done.flag` on clean exit. Rsync that path back to `runs/baseline/done.flag` for the orchestrator's `RESULTS_READY` transition.
- **`runs.jsonl` row** — the runner registers PROVISIONED immediately after handle capture (before rsync), per `experiment-runner.md` step 5, so the Stop hook can tear down even on launch failure.
- **Model + training knobs are FIXED** by the launcher and the issue body's Model + training config table — `MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct`, `ROLLOUT_N=8`, `TRAIN_BATCH_SIZE=128`, `PPO_MINI_BATCH_SIZE=64`, `MAX_PROMPT_LENGTH=1024`, `MAX_RESPONSE_LENGTH=16384`, `ppo_max_token_len_per_gpu=36864`, `use_dynamic_bsz=True`, `enable_gradient_checkpointing=True`, `ref.param_offload=True`, `actor.param_offload=False`, `ROLLOUT_TP=2`, `ROLLOUT_GPU_MEM_UTIL=0.4`, `ACTOR_LR=1e-6`, `KL_LOSS_COEF=0.001`, `ENTROPY_COEFF=0`, `TOTAL_EPOCHS=2`, `SAVE_FREQ=50`, `TEST_FREQ=25`, `trainer.logger=[console,wandb]`. Do not deviate without a REVISE plan.

## Notes for analyst

- **Headline criteria:** all checkboxes above, with the *learning signal* being `val/test_score` improving by ≥ +0.05 from step-0 to final. `train/reward_mean` rising monotonically is the secondary indicator (it is noisier than `val/test_score`).
- **`p95` is n/a here** — this is a dense baseline, not a compression/staleness experiment. The headline is pure learning quality + cost.
- **OOM mid-run** falsifies the OOM-avoidance posture; produce REVISE with `actor.fsdp_config.param_offload=True` as the first `next_action`. Do NOT propose reducing `MAX_RESPONSE_LENGTH` — the user mandate is 16K.
- **NaN / Inf in any of `loss`, `grad_norm`, `policy_loss`, `reward`** is automatic STOP + codex `code-rescue` route, not REVISE.
- **Spend > $60 or wall-clock > 12 h** is automatic STOP, not REVISE — the budget criterion is part of the hypothesis.
- **Missing WandB run** is automatic STOP — `trainer.logger=[console,wandb]` is mandatory for the compression-curve overlay in later experiments.
