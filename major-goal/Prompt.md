# Prompt For Issue-Populating Agent

You are an issue-populating agent. Your only job is to create GitHub issues in
`shamanez/verl-compression-research` for testing the communication-efficient
activation-masking paper method with verl GRPO. Do not implement code, do not
edit verl files, do not provision Vast instances, and do not open PRs.

Primary goal: create a dependency-ordered issue sequence that determines
whether the paper's masked-anchor spectral correction can be safely adapted to
verl GRPO actor training without contaminating rollout generation, old
log-prob computation, reference log-prob computation, reward extraction, GRPO
advantage computation, checkpointing, validation, or actor-to-rollout weight
sync.

Do not frame the work as a direct port of supervised HuggingFace Trainer logic.
The first implementation target is a GRPO-safe, opt-in actor-training
augmentation. Standard dense GRPO must continue to work in the same codebase by
changing only configuration/launcher arguments. The baseline launcher
`/Users/shamane/Documents/verl/examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
must still train normal GRPO when `comm_eff.enabled=false` or no comm-eff
overrides are supplied.

Read these first:

- `/Users/shamane/Documents/verl/major-goal/implementation-logic.md`
- `/Users/shamane/Documents/verl/major-goal/LLM_adaptation_neurips.pdf`
- `/Users/shamane/Documents/verl/CLAUDE.md`
- `/Users/shamane/Documents/verl/research/researcher_steps.md`
- `/Users/shamane/Documents/verl/research/.claude/project.yaml`
- `/Users/shamane/Documents/verl/research/.claude/plans/TEMPLATE.md`
- `/Users/shamane/Documents/verl/research/.claude/agents/research-planner.md`
- Existing dense baseline issue:
  `https://github.com/shamanez/verl-compression-research/issues/3`

Before creating anything, check for duplicates:

```bash
gh issue list --repo shamanez/verl-compression-research --state all --search "Qwen2.5 GRPO GSM8K activation mask"
gh issue list --repo shamanez/verl-compression-research --state all --search "M95 AP GRPO"
gh issue list --repo shamanez/verl-compression-research --state all --search "spectral anchor GRPO"
gh issue list --repo shamanez/verl-compression-research --state all --search "two-step GRPO smoke"
gh pr list --repo shamanez/verl --state open --search "activation mask GRPO"
gh pr list --repo shamanez/verl --state open --search "comm_eff"
```

If an existing open issue already covers the same work, do not create a
duplicate. Comment only if a material scope correction is needed.

Create only focused issues. Do not create a large omnibus issue. Do not create
another dense baseline issue; issue #3 is the dense baseline and already
passed. The issues must be self-contained because the research harness agents
do not read `major-goal/`.

For every issue that changes verl code and must exercise GRPO, use:

```yaml
kind: experiment
code_change: true
```

Do not use `kind: implementation` for those, because the harness treats
implementation issues as verify-only and will not run the required two-step
GRPO training path.

Each issue must include:

- Label expectation: `research:claim`.
- `milestone:`. Use a coherent milestone sequence such as `M2` for
  implementation smokes and `M3` for compressed GRPO comparisons.
- `hypothesis:`. It must be falsifiable and numeric.
- `baseline_run: EXP-3` when the issue compares to the passed dense baseline.
- `depends_on:`. Use harness ids such as `EXP-3` for the passed issue #3
  baseline where relevant, and chain later issues on earlier smoke issues.
- `target_modules:` with exact file paths from `implementation-logic.md`.
- Exact baseline launcher path:
  `/Users/shamane/Documents/verl/examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- A normal-GRPO compatibility requirement: all new behavior must be opt-in,
  disabled by default, and the same launcher/code path must remain capable of
  dense GRPO by setting `comm_eff.enabled=false` or omitting comm-eff overrides.
- RL-specific invariants: rollout generation, old log-prob recomputation,
  reference log-prob computation, reward extraction, GRPO advantage
  computation, validation, checkpoint save/load, inference, and weight sync
  must remain unmasked/unmodified unless a later issue explicitly scopes a
  separate extension.
- The two-step smoke requirement:
  rollout generation, old log-prob, reference log-prob when enabled,
  reward/advantage computation, actor forward/backward, gradient handling,
  optimizer step, scheduler step, and weight sync must run for two trainer
  rollout steps. Make clear that each trainer step may contain multiple actor
  optimizer substeps over the same rollout batch.
- A concrete smoke command template based on:

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
  <issue-specific comm_eff overrides>
```

- For anchor-specific two-step smokes, do not require paper cadence `K=20`.
  Use a smoke-only override such as `anchor_delay_K=1` and
  `anchor_cadence=1`, or a seeded anchor cache, so at least one unmasked
  anchor gradient is produced and applied inside the smoke. Reserve `K=20` for
  the later paper-cadence or 100-step comparison issue.
- For the first anchor issues, do not allocate a separate anchor GPU, separate
  Ray rank, or separate resource pool. Start with same-process or same-worker
  anchor emulation over the existing GRPO actor-loss path. A dedicated anchor
  resource is a later design issue only after the same-worker path passes.
- Success criteria that are machine-checkable.
- Analyst predicate with PASS, REVISE, and STOP.
- Compute budget with multi-GPU Vast H100/H200 only, 4 to 8 GPUs, locked
  template from `project.yaml`, and a small smoke budget.
- Rescue triggers for NaN/Inf, non-finite `actor/grad_norm`, OOM, FSDP
  sharding/reduction errors, missing metrics, rollout/ref contamination,
  cgroup `pids.max <= 2048`, and secrets leakage.

Create issues for this minimal sequence:

1. No-op `comm_eff.enabled=false` config plus two-step dense parity smoke.
   Purpose: prove the integration scaffolding does not change issue #3 GRPO
   behavior when disabled and that the baseline launcher still trains normal
   dense GRPO by arguments alone.

2. Actor-only PRF activation masking plus two-step smoke.
   Purpose: apply random in-graph masks only during actor training
   forward/backward, with rollout, old log-prob, ref log-prob, validation,
   checkpoint, and `infer_batch` confirmed unmasked.

3. Mask-disabled invariant tests.
   Purpose: make contamination impossible by testing all non-actor-training
   paths explicitly.

4. Spectral filter and FSDP gradient-application discovery.
   Purpose: port the paper formula and find the correct point between actor
   backward and `optimizer.step()` where spectral correction can be applied
   without corrupting FSDP sharding/reduction.

5. Anchor queue and same-process anchor emulation.
   Purpose: test the first anchor design without a separate anchor GPU/rank:
   same-loop periodic unmasked GRPO actor-loss backward over the
   rollout-expanded batch, no optimizer step, one EMA/SVD refresh, cached
   basis reused across fast PPO mini-batches. If full-batch anchor backward is
   too large for the smoke, allow microbatch accumulation or a bounded
   deterministic subset, but require the deviation to be logged.

6. Full M95+AP two-step GRPO smoke.
   Purpose: run `p=0.95`, `alpha=0.3`, `tau=1e-3`, `beta_anc=0.95`, and
   a smoke-only anchor cadence such as `K=1` through the complete two-step
   GRPO path, proving at least one anchor gradient is produced and applied.

7. DP compression follow-up only if the prior issues pass.
   Purpose: add PowerSGD/Streaming-DiLoCo issue scope without mixing it into
   activation masking or anchor correctness.

8. 100-step compressed GRPO comparison against issue #3 only after two-step
   smokes pass.
   Purpose: compare M95+AP GRPO at paper cadence `K=20` to the passed dense
   baseline on GSM8K.

Do not make the first anchor issue compute a separate anchor gradient per PPO
mini-batch. That is a later/heavier fidelity ablation after the whole-batch
same-loop anchor refresh passes.

Do not create issues for:

- Re-running the dense baseline from issue #3.
- Whole supervised SFT benchmark reproduction.
- Top-k activation masking.
- Quantized activation transport.
- Random projection/codebook activation compression.
- Signed-EMA, signSGD, rank-1 projection, or other simulator ablations.
- Single-GPU smokes.
- A separate anchor GPU/Ray-rank implementation in the first issue sequence.

Use concise issue bodies, but include enough exact context that
`research-planner` can write `.claude/plans/<N>.md` without reading
`major-goal/`.
