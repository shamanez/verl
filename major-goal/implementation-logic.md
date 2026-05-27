# Implementation Logic For GRPO Issue Creation

This file is the source of truth for creating GitHub issues that test the
paper method in verl GRPO. It replaces the deleted planning drafts. Keep issue
bodies self-contained: the autonomous research agents under `research/` do not
read `major-goal/` by contract.

## Mission

Create a small, non-duplicative issue set in
`shamanez/verl-compression-research` to test communication-efficient
activation masking with GRPO in this fork:

- Code repo: `/Users/shamane/Documents/verl`
- Research issue repo: `shamanez/verl-compression-research`
- Dense GRPO baseline issue:
  `https://github.com/shamanez/verl-compression-research/issues/3`
- Baseline launcher:
  `/Users/shamane/Documents/verl/examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`

Do not create another dense baseline issue. Issue #3 already passed a real
100-step Qwen2.5-1.5B-Instruct GSM8K GRPO run.

## Non-Negotiable GRPO Compatibility

All communication-efficient changes must coexist with the normal verl GRPO
training loop. The same codebase must still run standard dense GRPO by changing
only configuration/launcher arguments. In particular,
`examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` must continue
to train normal GRPO when `comm_eff.enabled=false` or when no comm-eff overrides
are supplied.

This is a must-have acceptance condition for every code-changing issue:

- Communication-efficient behavior must be opt-in and disabled by default.
- Existing GRPO launcher semantics must remain valid.
- The default actor/ref/rollout/checkpoint/weight-sync path must not require
  new anchor resources, new masks, or new spectral state.
- A dense no-op smoke must prove that adding the scaffolding does not change
  the normal GRPO path before any masking, anchor, or spectral feature is
  enabled.
- The implementation should be modular and config-driven so later theoretical
  variants can be swapped without rewriting the GRPO trainer loop.

## Paper Logic To Preserve

Primary paper artifact:

- `/Users/shamane/Documents/verl/major-goal/LLM_adaptation_neurips.pdf`

The method is:

1. Fast circuit: train continuously with deterministic in-graph random
   activation masks at logical pipeline boundaries.
2. Anchor circuit: run occasional unmasked forward/backward passes from a
   stale weight snapshot and return clean anchor gradients asynchronously.
3. Spectral correction: maintain an EMA of anchor gradients per targeted
   matrix, compute an SVD basis, and use it to denoise masked gradients before
   AdamW updates.
4. DP compression: combine the PP-side method with PowerSGD and Streaming
   DiLoCo when testing low-bandwidth throughput.

Do not implement top-k activation masking for the paper method. The PDF states
that random PRF masking is required because top-k introduces structured bias
that the spectral filter cannot remove.

Default paper knobs:

- Mask variants: `M90` uses `p=0.90`; `M95` uses `p=0.95`.
- Anchor replicas: `Z=1`.
- Anchor staleness/cadence: use `K=20` as the default operating point.
- Anchor EMA: `beta_anc=0.95`.
- Spectral blend: `alpha=0.3`.
- Spectral damping: `tau=1e-3`.
- Spectral SVD: full thin SVD for faithful reproduction; low-rank SVD is a
  separate scalability ablation.
- DP compression target: PowerSGD-64 plus Streaming DiLoCo/local horizon 10 or
  higher when testing 200 Mbps throughput.

Activation masking details:

- Apply masks in graph as `h_tilde = h * mask`.
- Do not add forward `1/(1-p)` rescaling in the first GRPO implementation.
  Algorithm A in the PDF writes the direct product, and the faithful simulator
  notes that rescaling at `p=0.95` destabilizes bf16 training.
- Generate masks from a shared deterministic PRF. The key must include at
  least layer/boundary id, global optimizer step, microbatch identity, sequence
  shard identity when present, hidden size, and a base run seed. The key must
  not depend on activation values.
- For a model with `L` decoder blocks and logical `pp_size`, choose boundary
  layer indices by evenly partitioning block indices into `pp_size` shards and
  taking the last block index of each shard except the final shard. This gives
  `[1,3,5,7,9,11,13]` for a 16-layer model with `pp_size=8`.
- For Qwen2.5-1.5B GRPO, derive `L` and hidden size from
  `model.config`; do not hardcode Llama layer counts.

Spectral correction formula for each targeted 2D matrix:

```text
M_anchor = beta_anc * M_anchor + (1 - beta_anc) * G_anchor
M_anchor = U S V^T
d_i = s_i / (s_i + tau)
X = U^T G_mask V
G_filt = U diag(d) X diag(d) V^T
G_proj = alpha * G_mask + (1 - alpha) * G_filt
```

The masked gradient is never discarded. Anchor gradients provide geometry, not
a direct replacement update.

Ordering requirement:

- Paper order is masked forward/backward, DP synchronization, anchor EMA/SVD
  refresh if an anchor gradient arrived, spectral correction, AdamW update.
- The simulator applies spectral correction before its manual DP all-reduce;
  because the filter is linear when every rank has the same anchor cache, this
  commutes with dense mean all-reduce.
- In verl/FSDP, each issue must explicitly test where full or sharded gradients
  are available. The correction must happen after masked actor backward and
  before `optimizer.step()`, without corrupting FSDP reduction or clipping.

## Supervised Reference Files

Use these files from `/Users/shamane/Documents/comm-eff-ft` only as
implementation references. Do not copy unrelated supervised-training
machinery into verl GRPO.

Core simulator/source-of-truth files:

- `/Users/shamane/Documents/comm-eff-ft/sim_v2/ddp_train.py`
  - Fast/anchor loop, mask enable schedule, anchor queue, spectral correction,
    manual DP all-reduce, optimizer ordering.
- `/Users/shamane/Documents/comm-eff-ft/sim_v2/activation_mask.py`
  - Forward hooks, PRF random masks, no forward rescale, shared mask seed.
- `/Users/shamane/Documents/comm-eff-ft/sim_v2/spectral_filter.py`
  - Anchor EMA, SVD, Tikhonov weights, two-sided projection, alpha blend,
    checkpoint state.
- `/Users/shamane/Documents/comm-eff-ft/sim_v2/anchor_circuit.py`
  - Staleness queue, full K-stale model snapshot, unmasked anchor
    forward/backward, target-gradient return protocol.
- `/Users/shamane/Documents/comm-eff-ft/sim_v2/model.py`
  - Target matrix selection. For the first verl implementation, target 2D
    decoder matrices that receive gradients; skip norms, biases, embeddings,
    and lm head unless an issue explicitly asks for an ablation.

Paper detail files:

- `/Users/shamane/Documents/comm-eff-ft/paper/sections/experiments_section.tex`
- `/Users/shamane/Documents/comm-eff-ft/paper/sections/dp_compression_section.tex`
- `/Users/shamane/Documents/comm-eff-ft/paper/sections/staleness_appendix.tex`
- `/Users/shamane/Documents/comm-eff-ft/paper/sections/datasets_appendix.tex`

Reference-only or older supervised-hook files:

- `/Users/shamane/Documents/comm-eff-ft/scripts/activation_mask.py`
- `/Users/shamane/Documents/comm-eff-ft/scripts/spectral_filter.py`
- `/Users/shamane/Documents/comm-eff-ft/scripts/anchor_callback.py`
- `/Users/shamane/Documents/comm-eff-ft/scripts/anchor_correction.py`
- `/Users/shamane/Documents/comm-eff-ft/scripts/activation_compress.py`

Ignore for the first GRPO issue set:

- `/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/`
- `/Users/shamane/Documents/comm-eff-ft/sim_v2/codebook/`
- Signed-EMA, signSGD, rank-1 projection, top-k, quantization, and other
  ablation variants unless a later issue specifically tests them.

## Paper Experiment Facts

Main paper SFT experiments use Llama-3.2-1B in the main text/simulator and a
SmolLM3-Mid recipe in later appendix text. For this verl task, do not migrate
the whole SFT benchmark suite into GRPO. Use the paper logic above and test it
on the existing GRPO GSM8K baseline.

Paper variants:

- `Baseline`
- `Baseline+DP`
- `M90`
- `M90+AP`
- `M95`
- `M95+AP`

Main paper environment:

- 10 epochs
- learning rate `2e-5`
- context length 2048
- batch size 32
- fast mesh `7 x 8`
- anchor mesh `1 x 8`
- throttled links at 200 Mbps

GRPO adaptation target for this repo:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: GSM8K
- Algorithm: GRPO
- Baseline launcher:
  `/Users/shamane/Documents/verl/examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`

## Existing GRPO Baseline

Issue #3 passed and should be referenced as the dense baseline:

- Issue:
  `https://github.com/shamanez/verl-compression-research/issues/3`
- Harness baseline id: `EXP-3`
- WandB run:
  `https://wandb.ai/shamanework-pl/verl_compression_research/runs/wybop525`
- Hardware: 4xH200
- Training stopped at global step 100 to conserve budget.
- GSM8K validation accuracy improved from `0.0872` at step 0 to `0.7892`
  at step 100.
- Rollout reward mean improved from about `0.126` to `0.874`.
- Latest actor grad norm was finite and below 5.
- No NaN/Inf was observed.
- Total instance spend was about `$12.76`.

Baseline launcher defaults:

- `algorithm.adv_estimator=grpo`
- `algorithm.use_kl_in_reward=False`
- `actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct`
- `actor_rollout_ref.rollout.n=8`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`
- `data.train_batch_size=128`
- `actor_rollout_ref.actor.ppo_mini_batch_size=64`
- `data.max_prompt_length=1024`
- `data.max_response_length=16384`
- `actor_rollout_ref.actor.optim.lr=1e-6`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`
- `trainer.total_epochs=2`
- `trainer.save_freq=50`
- `trainer.test_freq=25`

## Required GRPO Two-Step Smoke

Every code-changing compression issue must require a real two-step GRPO smoke
before any longer run. The smoke must execute the forward and backward path,
not only import tests or unit tests.

In verl GRPO, distinguish trainer rollout steps from actor optimizer substeps:
one trainer step generates a rollout batch, computes rewards/advantages, then
`_update_actor` may run several optimizer substeps over that same fixed batch.
With the baseline launcher defaults, `train_batch_size=128`, `rollout.n=8`,
`ppo_mini_batch_size=64`, and `ppo_epochs=1`, each trainer step produces
1024 sequences and runs two actor optimizer substeps. The smoke template below
keeps the same structure at smaller scale: 8 prompts, 2 rollouts, and 4-prompt
PPO mini-batches produce two actor optimizer substeps per trainer step on both
4-GPU and 8-GPU runs. Increasing `ppo_epochs` multiplies the number of
substeps on the same rollout data.

Minimum path that must run for two trainer rollout steps:

1. Rollout generation through vLLM.
2. Old log-prob recomputation.
3. Reference log-prob when KL loss requires the reference policy.
4. Reward extraction and GRPO advantage computation.
5. Actor `update_actor`.
6. Actor train forward/backward.
7. Gradient processing, clipping, optimizer step, scheduler step.
8. Weight update/sync back to rollout replicas.

Use the baseline launcher identity and only shrink batch, rollout count,
sequence length, validation, and step count for smoke. A valid smoke shape is:

```bash
PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=<unique-smoke-name> \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=4 \
ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=256 \
MAX_RESPONSE_LENGTH=256 \
PPO_MAX_TOKEN_LEN_PER_GPU=4096 \
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 \
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
TOTAL_EPOCHS=1 \
bash /Users/shamane/Documents/verl/examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  trainer.total_training_steps=2 \
  trainer.val_before_train=False \
  actor_rollout_ref.actor.ppo_epochs=1 \
  <comm_eff overrides for the issue>
```

For anchor-specific smokes, do not use the paper default `K=20` if the test is
only two trainer steps. A K=20 anchor will not arrive in that window. Use a
test-only override such as `anchor_delay_K=1` and `anchor_cadence=1`, or seed a
known anchor cache, so the smoke proves that at least one unmasked anchor
gradient is produced and applied. Keep `K=20` for the later paper-cadence or
100-step comparison issue.

The issue-creating agent must tell implementers to read the launcher before
finalizing the command and preserve the multi-GPU mandate: 4 to 8 H100/H200
GPUs through the locked Vast template, not single GPU.

Two-step smoke acceptance criteria:

- Training reaches exactly or at least global step 2.
- The run executes at least two actor optimizer substeps; with baseline smoke
  defaults it should execute four substeps across two trainer steps.
- `actor/grad_norm` is finite on every actor optimizer substep reported.
- Actor loss, KL, entropy, rollout score/reward, and log-prob metrics contain
  no NaN/Inf.
- At least one actor parameter changes between step 0 and step 2.
- Compression metrics expected by that issue are present.
- Masking is disabled outside actor training unless the issue explicitly tests
  a later compressed-rollout extension.

## verl Files To Inspect

Entrypoints and trainer flow:

- `/Users/shamane/Documents/verl/verl/trainer/main_ppo.py`
- `/Users/shamane/Documents/verl/verl/trainer/main_ppo_sync.py`
- `/Users/shamane/Documents/verl/verl/trainer/ppo/ray_trainer.py`
- `/Users/shamane/Documents/verl/verl/trainer/ppo/core_algos.py`
- `/Users/shamane/Documents/verl/verl/workers/utils/losses.py`

Actor/ref/rollout workers and engines:

- `/Users/shamane/Documents/verl/verl/workers/engine_workers.py`
- `/Users/shamane/Documents/verl/verl/workers/engine/base.py`
- `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py`
- `/Users/shamane/Documents/verl/verl/workers/engine/megatron/transformer_impl.py`
- `/Users/shamane/Documents/verl/verl/workers/engine/automodel/transformer_impl.py`
- `/Users/shamane/Documents/verl/verl/workers/engine/automodel/utils.py`
- `/Users/shamane/Documents/verl/verl/workers/rollout/`

Checkpoint and weight-sync boundaries:

- `/Users/shamane/Documents/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- `/Users/shamane/Documents/verl/verl/utils/checkpoint/megatron_checkpoint_manager.py`
- `/Users/shamane/Documents/verl/verl/utils/megatron_utils.py`

Config files:

- `/Users/shamane/Documents/verl/verl/workers/config/engine.py`
- `/Users/shamane/Documents/verl/verl/workers/config/actor.py`
- `/Users/shamane/Documents/verl/verl/trainer/config/ppo_trainer.yaml`
- `/Users/shamane/Documents/verl/verl/trainer/config/engine/fsdp.yaml`
- `/Users/shamane/Documents/verl/verl/trainer/config/engine/megatron.yaml`
- `/Users/shamane/Documents/verl/verl/trainer/config/engine/automodel.yaml`

New module locations to propose in issues:

- `/Users/shamane/Documents/verl/verl/workers/comm_eff/__init__.py`
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/activation_mask.py`
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/spectral_filter.py`
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/anchor.py`
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/dp_compress.py`
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py`
- `/Users/shamane/Documents/verl/verl/workers/config/comm_eff.py`

Tests to propose:

- `/Users/shamane/Documents/verl/tests/workers/comm_eff/test_activation_mask.py`
- `/Users/shamane/Documents/verl/tests/workers/comm_eff/test_spectral_filter.py`
- `/Users/shamane/Documents/verl/tests/workers/comm_eff/test_anchor_queue.py`
- `/Users/shamane/Documents/verl/tests/workers/config/test_comm_eff_config.py`

## GRPO Integration Boundaries

Masking must be actor-training-only for the first compression tests:

- Enable masking only on the actor train path:
  `RayPPOTrainer._update_actor` ->
  `ActorRolloutRefWorker.update_actor` ->
  `TrainingWorker.train_mini_batch` ->
  `TrainingWorker.train_batch` ->
  `BaseEngine.train_batch` ->
  engine `forward_backward_batch`.
- Do not enable masking during rollout generation.
- Do not enable masking during old log-prob recomputation.
- Do not enable masking during reference log-prob computation.
- Do not enable masking during validation.
- Do not enable masking during checkpoint save/load/export.
- Do not enable masking in `infer_batch`.

The first issues should force a no-op/off-by-default path before any masking:

- `comm_eff.enabled=false` must preserve dense GRPO metrics and behavior.
- Config defaults must be disabled.
- Existing GRPO launchers must run unchanged unless overrides are supplied.
- The baseline command must remain a valid way to train normal dense GRPO; the
  compression path is selected only by explicit `comm_eff.*` arguments.
- The compressed path must not fork the trainer into a separate GRPO algorithm.
  It should wrap or augment actor training while preserving the same rollout,
  log-prob, reward, advantage, actor update, optimizer, and weight-sync
  sequence used by normal GRPO.

Main caveats before issue creation:

- Do not frame the task as porting supervised HuggingFace Trainer logic. The
  GRPO target is the actor-training gradient path inside a rollout-generation
  RL loop.
- Do not start with a separate anchor GPU or Ray rank. Verl already colocates
  actor, reference, rollout, vLLM sleep/wake, and checkpoint-engine weight sync
  under Ray/FSDP. Start with same-process or same-worker anchor emulation.
- The anchor must consume the rollout-expanded GRPO actor batch, including
  responses, `response_mask`, `old_log_probs`, `advantages`, and optional
  `ref_log_prob`. It must not generate new responses, recompute rewards, or use
  supervised next-token loss.
- Early issues must test GRPO mechanics and invariants before making quality or
  throughput claims. Useful early hypotheses are finite losses/grad norms,
  nonzero actor parameter deltas, expected mask ratios, no contamination of
  non-train paths, and successful weight sync.
- Distinguish trainer rollout steps from actor optimizer substeps. A single
  trainer step may reuse one rollout batch over multiple PPO mini-batches, so
  mask keys and anchor cadence must account for optimizer substeps and
  microbatch identity.
- FSDP gradient access is a discovery item. Issues must determine whether
  correction can safely operate on full matrices, DTensors, flat parameters, or
  local shards before promising paper-faithful spectral correction.
- Spectral correction must happen after actor backward and before clipping and
  `optimizer.step()`. Whether it should be before or after an FSDP reduction is
  an implementation question that must be tested.
- Weight sync is a separate post-update invariant. Anchor state, masks, and
  spectral caches must not be synced to rollout replicas or alter rollout
  weights except through the normal actor optimizer update followed by
  checkpoint-engine weight sync.
- Do not mix DP compression into the first actor-mask/anchor correctness
  issues. Add PowerSGD/Streaming-DiLoCo only after the GRPO-specific masked and
  anchor paths pass.
- Full-context anchor backward can OOM on the long-response baseline. Anchor
  issues may use two-step smoke lengths, bounded subsets, or microbatch
  accumulation, but any deviation from whole rollout-expanded batch anchoring
  must be logged explicitly.
- The paper/simulator use direct `h * mask` without forward rescaling for the
  first faithful implementation. Do not overclaim theoretical unbiasedness for
  the no-rescale GRPO port.

FSDP caveat:

- FSDP may reduce or shard gradients during backward. Any spectral correction
  issue must state exactly whether it operates on full gradients, flat/sharded
  gradients, or per-parameter local shards. If the full matrix is unavailable,
  the issue should start with a minimal unsharded/single-engine proof or add a
  specific FSDP-aware design task.

Anchor design note for GRPO:

- Test first: a same-loop, periodic anchor refresh that runs one unmasked GRPO
  actor-loss backward over the whole rollout-expanded batch, does no optimizer
  step, updates the anchor EMA/SVD once, and reuses that cached basis for every
  fast PPO mini-batch until the next refresh.
- Later/heavier: compute a separate unmasked anchor gradient per PPO mini-batch
  and apply a matched spectral basis per fast mini-batch. This is more exact,
  but multiplies anchor compute by `train_batch_size / ppo_mini_batch_size`
  and by `ppo_epochs`.

## Minimal Issue Families

Create focused issues, not one omnibus issue. Use `kind: experiment` with
`code_change: true` for any issue that must run the two-step GRPO smoke. Use
`kind: implementation` only for pure local/unit work that does not need a Vast
training run.

Recommended non-duplicate issue families:

1. No-op config and two-step dense parity smoke.
2. Actor-only activation masking with PRF masks and two-step smoke.
3. Mask-disabled invariants for rollout, old log-prob, ref log-prob,
   validation, checkpoint, and `infer_batch`.
4. Spectral filter unit coverage plus gradient-application point discovery for
   FSDP.
5. Anchor queue/state with K-stale snapshots and a same-process anchor
   emulation for GRPO smoke.
6. Full M95+AP two-step GRPO smoke using `p=0.95`, `alpha=0.3`, `tau=1e-3`,
   `beta_anc=0.95`, and a smoke-only anchor cadence such as `K=1` so an
   anchor gradient arrives during the smoke. A later comparison issue should
   restore paper cadence `K=20`.
7. Optional DP-compression issue after M95+AP smoke passes.
8. Optional 100-step compressed GRPO comparison against issue #3 after all
   two-step smokes pass.

Each issue must include:

- Falsifiable `hypothesis:` with numeric thresholds.
- `kind: experiment` and `code_change: true` when a GRPO smoke run is required.
- `target_modules:` with exact verl paths.
- `baseline_run: EXP-3` when comparing against the dense baseline.
- `depends_on:` entries using harness ids such as `EXP-3` or earlier issue
  ids in the new sequence.
- A two-step smoke command template.
- Machine-checkable success criteria.
- Analyst predicate: PASS, REVISE, STOP.
- Rescue triggers for NaN/Inf, missing metrics, FSDP reduction problems,
  rollout/ref path contamination, OOM, and Vast host `pids.max` problems.

## Do Not Duplicate

- Do not recreate `verl-grpo-implementation-issues.md`,
  `allignment-handbook-modeifications.md`, or `core-task.md`; their useful
  content is consolidated here.
- Do not create a new dense GRPO baseline issue; issue #3 is the baseline.
- Do not create issues for the whole SFT paper benchmark suite unless the user
  separately asks for supervised reproduction.
- Do not create top-k, quantization, random projection, codebook, signed-EMA,
  signSGD, or rank-1-projection issues in the first pass.
- Do not ask future harness agents to read `major-goal/`; issue bodies must
  carry the exact context needed by `research-planner`.
