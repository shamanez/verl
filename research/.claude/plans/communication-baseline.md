# Plan communication-baseline — M2 final smoke (no KL, mask the full fast circuit, end-to-end training)

> **Promoted to a permanent baseline reference.** Originally filed as EXP-9
> (issue #9). Iter2 PASS established the M90+AP configuration that subsequent
> comm-eff experiments compare against. The full plan text below is preserved
> as a historical record of the design constraints; the operational verdicts
> live in `runs/communication-baseline/verdict.md` (iter1 REVISE) and
> `runs/communication-baseline/verdict-iter2.md` (iter2 PASS).

## Experiment
- id:                communication-baseline (formerly EXP-9)
- title:             M2 final — full M95+AP with mask_recompute=true, no KL, end-to-end 20-step training on 4×H200
- issue:             https://github.com/shamanez/verl-compression-research/issues/9
- kind:              experiment
- milestone:         M2
- created_at:        2026-05-28T16:00:00+10:00
- baseline_run:      baseline (dense GRPO, id-only — compare against historical WandB curves)
- lineage_iteration: 1 of 3
- slug:              m2-final-noKL-maskrecompute-aps

> **Capstone of M2.** This issue runs the **complete two-circuit method end-to-end at
> smoke scale with the FULL fast circuit masked** (PRF activation mask at pipeline
> boundaries on BOTH the PPO train forward AND the `old_logp` recompute), same-process
> anchor refresh at smoke cadence, and spectral correction applied AFTER FSDP reduction
> and BEFORE optimizer step. No KL terms (`use_kl_loss=False`,
> `use_kl_in_reward=False`). No entropy bonus (`entropy_coeff=0`). No matched dense
> control cell — comparison is against the historical EXP-3 WandB curves. The
> deliverable is: the model demonstrably learns under full pipeline-boundary
> masking + anchor + spectral correction, with visible reward improvement in the
> first 10 of 20 trainer steps. Anything less means the comm_eff integration is
> incomplete or biasing GRPO.

## NON-NEGOTIABLES (operator-mandated, take precedence over any conflict elsewhere)

These rules supersede any conflicting language in the plan body, the issue body, the
inherited convention, or the orchestrator playbook defaults. The runner, the operator,
the monitor team, and the analyst MUST honour them.

### 1. End-to-end training is THE deliverable

Not a 2-step toy. The single launched cell MUST reach
`trainer/global_step == 20` with all comm_eff circuits firing every step and
finite `actor/grad_norm` at every substep. PASS requires (a) `global_step == 20`,
(b) `critic/score/mean` visibly higher at step ≥ 7 than at step 1, (c) every
machine-checkable counter in §Success criteria green. Anything short is REVISE
or STOP — never PASS.

### 2. No toy runs — only `comm_eff.enabled=true` with the FULL set of circuits

The only configuration this issue exercises is `comm_eff.enabled=true` with
**every element of the PP-parallel fast circuit + anchor circuit enabled**:

- **Fast (masked) circuit on BOTH gradient-relevant forwards.** PRF activation mask
  at pipeline-boundary decoder blocks fires on the PPO train forward
  (`path_tag="train"`) AND on the `old_logp` recompute (`path_tag="old_logprob"`).
  Both forwards consume pipeline-boundary bandwidth on the FSDP train engine and
  both feed the training gradient (train forward produces `log_prob_current`,
  old_logp recompute produces `old_log_prob` that enters the PPO importance ratio
  `r = exp(log_prob_current − old_log_prob)`). Masking only one of them is
  incomplete communication-efficiency for pipeline-parallel RL.
- **Anchor circuit (unmasked, clone-no-hook).** Periodic same-process unmasked
  GRPO-actor-loss backward from a K-stale weight snapshot on a hookless clone.
  GUARD 5 (`mask_active=False`, `path_tag=None`) MUST hold — mask hook fires
  zero times during the anchor pass. The anchor is the gold-signal forward; it
  CANNOT be masked.
- **Spectral correction (full circuit).** EMA of `G_anchor` → full thin SVD →
  two-sided Tikhonov projection of `G_mask` → blend into `p.grad` before AdamW.
  `seed_anchor_cache=false` (the live anchor populates `M_anchor`).

There is NO `comm_eff.enabled=false` dense control cell in this run. Comparison is
made against the historical EXP-3 dense baseline WandB curves. A single-cell
launch.

### 3. No KL terms anywhere

`actor_rollout_ref.actor.use_kl_loss=False` AND `algorithm.use_kl_in_reward=False`.
With both false, `need_reference_policy()` (`verl/trainer/ppo/utils.py:75-79`)
returns False, the ref worker is not even initialized, and the `ref_log_prob`
recompute is skipped entirely. `actor/kl_loss` and `kl_coef` MUST be absent from
the logged metrics.

### 4. No entropy bonus

`actor_rollout_ref.actor.entropy_coeff=0`. `actor/entropy_loss` MUST be exactly 0
across every logged step.

### 5. Two-member monitoring team (TWO concurrent agents during RUNNING)

The orchestrator dispatches TWO Claude Code subagents the moment the runner promotes
the row to `RUNNING`:

- **Member A — `training-log-monitor`** (`research/.claude/agents/training-log-monitor.md`,
  Opus, `run_in_background=true`). System-health watcher. 30 s SSH-poll cadence for
  tmux liveness, `done_*.flag`, Traceback/Ray-unhandled/OOM/NaN grep on each cell's
  log, `nvidia-smi` per-GPU util. Cross-checks WandB scalars every ~3rd poll. Exits
  on aggregate `done.flag`, tmux DEAD premature, GPU stall (all 4 GPUs 0% for 4
  polls AND tmux ALIVE), env-failure, or 60 min timeout. Re-invokes orchestrator
  on terminal condition.

- **Member B — `curve-analyst`** (use `subagent_type=general-purpose`, Opus,
  `run_in_background=true`, prompt at the bottom of this plan). RL-signal watcher.
  Every 5 min while the run is RUNNING, fetches the latest WandB scalars for
  `experiment_name=m2-final-noKL-maskrecompute-aps` and posts a structured
  curve-snapshot line to `runs/communication-baseline/curve-snapshots.log`. Watches:
  `critic/score/mean`, `actor/grad_norm`, `actor/pg_clipfrac`, `actor/pg_loss`,
  `response_length/mean`, `comm_eff/mask_applications/train`,
  `comm_eff/mask_applications/old_logprob`, `comm_eff/anchor_backwards`,
  `comm_eff/spectral_corrections`, `comm_eff/anchor_mask_applications`. On any
  detected anomaly (NaN anywhere, `actor/grad_norm` non-finite, reward flat at
  step ≥ 10, `mask_applications/old_logprob == 0`, `anchor_mask_applications > 0`),
  append a `STUCK: EXP-9 <anomaly>` line to PROGRESS.md so the orchestrator routes
  to `codex-bridge --mode=code-rescue` on the next tick.

Both agents run in background concurrently; neither blocks the orchestrator
tick.

### 6. Auto-iterate on failure — fix it and keep running

The orchestrator's standard REVISE → child-issue → codex-bridge code-rescue →
relaunch loop applies, up to `iterations: 3`. **Additionally** the operator MAY
SSH into the RUNNING box and hot-fix in place (the EXP-12 in-place iteration
pattern at `plans/12.md §Debug workflow`). On any `STUCK:` line emitted by either
Member A or Member B, the orchestrator's next tick dispatches `codex-bridge
--mode=code-rescue` with the rescue trigger context. **Do not pre-empt** —
training failures (FSDP collision, NaN mid-training, wrong counter values) keep
the box RUNNING through the rest of the steps; the analyst owns the verdict.

STOP only on: (a) `iterations: 3` exhausted, (b) `max_dph × wall_clock_hr` budget
exhausted, (c) hard env-failure on the only provisioned tier.

### 7. Codex-verify INTENTIONALLY SKIPPED (operator override)

A pre-written `runs/communication-baseline/verify/20260528T160000.md` with `VERIFY: PASS` exists
at plan-creation time. This moves the orchestrator state directly from
`NEEDS_VERIFY` to `VERIFIED` on the next tick; the runner is dispatched without
going through `codex-bridge --mode=verify`. Precedent: EXP-6, EXP-8, EXP-12 all
used the same operator-override mechanism.

Justification: the implementation surface for this issue (extend the mask hook
eligibility set + add `mask_recompute` config flag + stamp `mask_active=True`
around `compute_log_prob`) is small, structurally orthogonal to the anchor and
spectral surfaces already verified by EXP-8 + EXP-12, and the substantive
correctness is checked by (a) a new regression test
`test_mask_recompute_path_tag_eligibility` in
`tests/workers/comm_eff/test_activation_mask.py`, (b) the runtime greppable
counters `comm_eff/mask_applications/{train,old_logprob}` from the on-box smoke,
and (c) the analyst's predicate on the WandB learning curves. Codex static review
adds no information here. **Scope of skip: EXP-9 only.** Any REVISE child of
this lineage is subject to standard codex-verify gating.

### 8. Tear down before stop

The Stop hook teardown contract still applies — Vast instance MUST be destroyed
before the session stops; `runs/.claude/state/runs.jsonl` row for EXP-9 MUST
read `TORN_DOWN` before the orchestrator exits. This is the canonical
operator-override pattern from every recent session.

## Kind (drives orchestrator routing)

`kind: experiment` with `code_change: true`. **Codex-verify is skipped per
non-negotiable #7.** The pre-written verify file at
`runs/communication-baseline/verify/20260528T160000.md` satisfies the state-machine. The
orchestrator dispatches `experiment-runner` directly from `status:approved` on
the next tick.

## Hypothesis

With the complete comm-eff path enabled — **M95** activation masking on **BOTH**
the actor-train forward AND the `old_logp` recompute (PRF-keyed Bernoulli,
`p=0.95`, no rescale, fires at pipeline-boundary decoder blocks), same-process
anchor refresh at `cadence=4, delay_K=4` on a hookless clone, anchor EMA
`beta_anc=0.9`, full thin SVD spectral correction `alpha=0.3` / `tau=0.001`
applied after FSDP reduction and before `optimizer.step()`, no KL, no entropy —
a 20-trainer-step GRPO smoke on Qwen2.5-1.5B-Instruct / GSM8K:

- reaches `global_step=20`, executes 40 actor optimizer substeps,
- masks BOTH gradient-feeding forwards (`mask_applications/train > 0` AND
  `mask_applications/old_logprob > 0`),
- fires the anchor backward 10 times across the run
  (`anchor_backwards == 10 ± 1` at `cadence=4`, 40 substeps),
- applies ≥1 spectrally-corrected gradient through to the optimizer
  (`spectral_corrections > 0`),
- keeps `actor/grad_norm` finite on every substep, no NaN/Inf anywhere,
- demonstrably learns: `critic/score/mean` strictly higher at step ≥ 7 than at
  step 1 by a margin larger than the EXP-3 dense baseline's noise band over its
  first 10 steps,
- holds every anchor-semantics guard (anchor pass: mask hook fires 0 times,
  spectral correction not applied to anchor grads, no optimizer step on anchor,
  no extra rollouts / no reward recompute),
- no KL terms appear in logged metrics (ref worker not spawned).

**Falsified if** the model fails to learn within the first 10 steps (compressed
reward curve flat or declining), `mask_applications/old_logprob == 0` (extended
masking not actually firing), any NaN/Inf or non-finite grad_norm appears, any
RL-measurement path other than train/old_logprob is contaminated, spectral
correction corrupts FSDP reduction, the anchor pass's GUARD 5 fails, or the
implementation cannot be made stable within `iterations: 3`.

## Background pointers

- Prior findings: `findings/M2/EXP-7.md` (spectral filter PASS — supplies the
  EMA/SVD cache); `findings/M2/EXP-12.md` (clone-no-hook anchor backward PASS —
  closes the FSDP1 `_post_backward_hook` collision).
- Dependency state: EXP-3 `status:done` (dense baseline, id-only), EXP-5 `done`
  (actor-only PRF masking), EXP-7 `pass/done` (spectral correction + FSDP grad
  point), EXP-12 `pass/done` (anchor backward graph isolation).
- Engineering map: `/Users/shamane/Documents/verl/CODE_WALKTHROUGH.md` — the
  canonical per-component map of the implementation as of the EXP-12 merge.
- Issue #9 comment thread carries the full forward-pass map, EMA cadence
  derivation, train-inference mismatch analysis, and the implementation work
  list referenced by §Code change below. Read the issue comments before
  starting work on the box; the issue is the spec.
- Load-bearing inheritance from EXP-12:
  - FSDP1 + `actor_rollout_ref.actor.fsdp_config.use_orig_params=true` regime.
  - Anchor backward MUST run on the cloned-no-hook module, NOT the live FSDP
    module. `verl/workers/comm_eff/anchor.py::build_anchor_module` +
    `assert_anchor_module_isolated` are the entry points.
  - Spectral correction at `after_actor_backward__before_optimizer_step`,
    AFTER FSDP reduction.
  - vLLM `gpu_memory_utilization` does not need a special override on H200
    (140 GB cards); the default ≈0.85 is fine.

## Experiment design

```yaml
sweep_grid:
  # No sweep. ONE integrated cell with the complete circuit.
  cells:
    - name: m2-final-noKL-maskrecompute-aps
      comm_eff.enabled: true
      comm_eff.mask.enabled: true
      comm_eff.mask.p: 0.95
      comm_eff.mask.mask_recompute: true             # NEW — extend mask to old_logp recompute
      comm_eff.spectral.enabled: true
      comm_eff.spectral.alpha: 0.3
      comm_eff.spectral.tau: 0.001
      comm_eff.spectral.beta_anc: 0.9                # smoke β (paper 0.95 unsettled at 10 fires)
      comm_eff.spectral.seed_anchor_cache: false     # live anchor populates M_anchor
      comm_eff.spectral.ema_device: gpu
      comm_eff.spectral.svd_mode: full
      comm_eff.spectral.basis_cache: cache
      comm_eff.spectral.max_targets: 4
      comm_eff.anchor.enabled: true
      comm_eff.anchor.cadence: 4                     # 40 substeps / 4 = 10 anchor fires
      comm_eff.anchor.delay_K: 4
      actor_rollout_ref.actor.use_kl_loss: false
      algorithm.use_kl_in_reward: false
      actor_rollout_ref.actor.entropy_coeff: 0
baselines:
  - EXP-3                              # id-only; compare against historical WandB curves
ablations:
  []                                   # no ablations in M2 capstone
seed_replicates:  1
fanout_max:       1
```

## Compute budget

```yaml
gpu_count:        1                       # single Vast.ai instance, 4 GPUs
gpu_filter_chain:                         # H200-only per operator directive 2026-05-28
  - "num_gpus=4 gpu_name=H200 gpu_ram>=140 cuda_max_good>=13.0 reliability>=0.95 rentable=true verified=true"
  # Fallback only if 4×H200 unavailable
  - "num_gpus=4 gpu_name=H100 gpu_ram>=80  cuda_max_good>=13.0 reliability>=0.95 rentable=true verified=true"
  - "num_gpus=8 gpu_name=H100 gpu_ram>=80  cuda_max_good>=13.0 reliability>=0.95 rentable=true verified=true"
max_dph:          24.0
max_gpu_hr:       12
max_parallel:     1
wall_clock_hr:    4
iterations:       3
```

H200 (140 GB) is comfortable for the cached anchor clone (~3 GB params for
Qwen2.5-1.5B in bf16) + vLLM rollout engine (~50 GB at default
`gpu_memory_utilization`) + FSDP-sharded actor (~5 GB). No need for the
consumer-card debug chain — EXP-12 closed that lineage.

## Vast.ai training footprint

```yaml
vast_cells:        1
steps_per_cell:    20
total_train_steps: 20
justification:     "20 trainer steps at smoke shape (train_batch_size=8, ppo_mini_batch_size=4, ppo_epochs=1) = 40 PPO substeps. At cadence=4/delay_K=4 the anchor fires 10 times, beta_anc=0.9 brings M_anchor to ~65% settled by run end. 20 trainer steps is also the standard slot for 'visible improvement in first 10 steps' (EXP-3 baseline shows clear reward improvement by step 10 on this batch shape). Total wall ~60-90 min on 4×H200, ~5 GPU-hr, under max_gpu_hr=12."
```

### Smoke launch command (canonical; runner executes the SINGLE cell on the Vast box)

```bash
cd /workspace/verl    # box checkout root
PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=m2-final-noKL-maskrecompute-aps \
TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=256 MAX_RESPONSE_LENGTH=256 \
PPO_MAX_TOKEN_LEN_PER_GPU=4096 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 \
SAVE_FREQ=-1 TEST_FREQ=-1 TOTAL_EPOCHS=1 \
bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  trainer.total_training_steps=20 \
  trainer.val_before_train=False \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.actor.use_kl_loss=False \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=true \
  actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3 \
  actor_rollout_ref.actor.comm_eff.spectral.tau=0.001 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.9 \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=false \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode=full \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=4 \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=true \
  actor_rollout_ref.actor.comm_eff.anchor.cadence=4 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=4 \
  2>&1 | tee /workspace/runs/communication-baseline/train.log
touch /workspace/runs/communication-baseline/done.flag
```

Keep the launcher's default `trainer.logger` (console + wandb) so the 20 steps
appear in WandB under the project. `EXPERIMENT_NAME` is unique to this cell —
do NOT reuse across iterations; if a REVISE child is created, append `-iter<N>`.

## Success criteria

13 machine-checkable boxes. PASS requires every one.

- [ ] **1. End-to-end (non-negotiable #1).** Cell reaches `trainer/global_step == 20`.
- [ ] **2. Fast-circuit masking on BOTH gradient-feeding forwards (non-negotiable #2).** `comm_eff/mask_applications/train > 0` AND `comm_eff/mask_applications/old_logprob > 0` in the final-step metrics.
- [ ] **3. Mask confinement.** `comm_eff/mask_applications/<other tag>` == 0 for every other tag (`rollout`, `ref_logprob`, `val`, `infer`, `ckpt`, `None`).
- [ ] **4. Mask ratio fidelity.** `comm_eff/mask_ratio` within 0.95 ± 0.02 on both masked paths.
- [ ] **5. Anchor backward fires per cadence.** `comm_eff/anchor_backwards == 10 ± 1` over 40 substeps at cadence=4.
- [ ] **6. Anchor unmasked (GUARD 5).** `comm_eff/anchor_mask_applications == 0`.
- [ ] **7. Anchor uncorrected (GUARD 6).** `comm_eff/anchor_grad_corrected == 0`.
- [ ] **8. Anchor no contamination.** `comm_eff/anchor_rollouts_generated == 0` AND `comm_eff/anchor_rewards_recomputed == 0` AND `comm_eff/anchor_optimizer_steps == 0`.
- [ ] **9. Spectral correction fires.** `comm_eff/spectral_corrections > 0`, with ≥1 logged per-target `||G_proj − G_mask|| / ||G_mask||` in `(0, 1]`.
- [ ] **10. EMA evolves.** `||M_anchor_last − M_anchor_first|| > 0` across the 10 anchor fires.
- [ ] **11. No KL (non-negotiable #3).** `actor/kl_loss` and `kl_coef` absent from logged metrics; ref worker never spawned (no Ray actor named `RefPolicy` in the runtime).
- [ ] **12. No entropy (non-negotiable #4).** `actor/entropy_loss == 0` at every logged step.
- [ ] **13. Visible learning (THE M2 quality bar).** `critic/score/mean` strictly higher at step ≥ 7 than at step 1, with the slope visibly up from steps 1 → 10. `actor/grad_norm` finite at every substep and not collapsing toward zero. `actor/pg_clipfrac` moving (neither stuck at 0 nor saturated near 1). `response_length/mean` non-degenerate. No NaN/Inf anywhere in `loss / grad / reward / log-prob`.

## Verification commands

```bash
python research/scripts/analyze.py runs/EXP-9 --emit verdict.md
python research/scripts/check_budget.py runs/EXP-9
python research/scripts/diff_against_baseline.py runs/EXP-9 --baseline EXP-3
```

Expect `diff_against_baseline.py` to report `baseline not found: runs/EXP-3`
(EXP-3 is id-only). Visible-learning grading (criterion 13) is done by
fetching the EXP-3 historical WandB run for the same `critic/score/mean` column
and eyeballing whether the EXP-9 curve's slope is in the same direction with a
similar magnitude over its first 10 steps.

Greppable proofs:

```bash
grep -E 'mask_applications/(train|old_logprob):' runs/communication-baseline/train.log | tail -5
grep -E 'anchor refresh step=' runs/communication-baseline/train.log | head
grep -oE 'actor/comm_eff/anchor_backwards:[0-9.]+' runs/communication-baseline/train.log | tail -1
grep -oE 'actor/comm_eff/spectral_corrections:[0-9.]+' runs/communication-baseline/train.log | tail -1
grep -oE 'training/global_step:[0-9]+' runs/communication-baseline/train.log | sort -u | tail
grep -oE 'critic/score/mean:[0-9.eE+-]+' runs/communication-baseline/train.log | head -10
grep -oE 'anchor_backward_isolation_mode=[a-z/() .]+' runs/communication-baseline/train.log
```

## Analyst predicate

- **PASS** iff all 13 boxes checked. The headline gates are criterion 1
  (end-to-end), criterion 2 (BOTH fast-circuit forwards masked), and
  criterion 13 (visible learning). Anything passing the comm_eff guards but
  failing learning is REVISE, not PASS.
- **REVISE** if ≤ `iterations` (3) boxes fail with a concrete `next_actions:`
  yaml list of `{knob, from, to, rationale}`. Examples: extended masking didn't
  fire (`mask_applications/old_logprob == 0`) → tighten the `mask_active` stamp
  around `compute_log_prob`; reward flat → reduce `alpha` (less filter
  influence) or raise `tau` (less aggressive Tikhonov damping); EMA didn't
  evolve → check that the anchor `G_anchor` actually reaches `update_anchor`.
- **STOP** if (a) `iterations: 3` exhausted, (b) `max_gpu_hr` exhausted,
  (c) compressed run cannot show visible learning at step 10 across 3
  iterations (the comm_eff method is biasing GRPO and the M2 capstone has
  falsified its hypothesis).

This is REVISE iteration 1 of 3 on the EXP-9 (M2-capstone) lineage.

## Code change

```yaml
code_change: true
branch_strategy:
  base: origin/vast-ai-workload
  slug: 9-m2-final-noKL-maskrecompute-aps
target_modules:
  - verl/workers/config/comm_eff.py                  # ADD Mask.mask_recompute: bool = False; __post_init__ validation
  - verl/workers/comm_eff/state.py                   # ADD MASK_ELIGIBLE_TAGS frozenset (depends on Mask.mask_recompute); ensure mask_applications/old_logprob is surfaced
  - verl/workers/comm_eff/activation_mask.py         # CHANGE hook assert: tag in MASK_ELIGIBLE_TAGS (was: == TRAIN_TAG). No 1/(1-p) rescale (unchanged).
  - verl/workers/engine_workers.py                   # CHANGE compute_log_prob: when mask.enabled AND mask.mask_recompute, set mask_active=True inside _comm_eff_path("old_logprob")
  - tests/workers/comm_eff/test_activation_mask.py   # ADD test_mask_recompute_path_tag_eligibility — with mask_recompute=true, hook fires for {train, old_logprob} and rejects everything else; with mask_recompute=false (regression), hook still rejects everything except train
  - verl/trainer/config/actor/dp_actor.yaml          # ADD mask.mask_recompute schema field (default false)
  - verl/trainer/config/actor/dp_actor_megatron.yaml # ADD mask.mask_recompute schema field (default false)
```

The verify gate is **SKIPPED** for this issue (non-negotiable #7). The pre-written
`runs/communication-baseline/verify/20260528T160000.md` file documents the override. Substantive
correctness is checked by:

1. The new CPU unit test in `tests/workers/comm_eff/test_activation_mask.py`
   (must PASS with `mask_recompute=True` enabling old_logprob masking AND must
   PASS with `mask_recompute=False` preserving the current train-only behavior).
2. The 56 existing CPU tests in `tests/workers/comm_eff/` and
   `tests/workers/config/` continuing to pass.
3. The runtime greppable counters (criteria 2, 3, 4 above) on the box.
4. The analyst's predicate on the WandB learning curves (criterion 13).

## Dependencies

```yaml
depends_on: [EXP-3, EXP-7, EXP-12]    # all already PASS/done
```

EXP-12 closed PASS on `vast-ai-workload` (anchor backward graph isolation merged).
EXP-7 supplied the spectral correction + FSDP grad-point discovery. EXP-3 is the
historical dense baseline (id-only, no run dir on disk — compared via WandB).

## Rescue triggers

```yaml
escalate_to_codex_if:
  - "no anchor gradient (produced|applied)"
  - "anchor (generated|produced) rollout"
  - "anchor recomputed reward"
  - "anchor used (supervised|next-token) loss"
  - "anchor optimizer step"
  - "mask (applied|fired) (on|during) (the )?anchor"             # GUARD 5
  - "anchor (grad|gradient) (corrected|spectrally)"               # GUARD 6
  - "mask applied on (rollout|ref|val|infer|checkpoint)"          # confinement
  - "mask_applications/old_logprob.*== ?0"                        # mask_recompute didn't fire
  - "spectral correction corrupted (gradient|FSDP)"
  - "actor/grad_norm non-finite"
  - "NaN detected"
  - "nan|NaN|inf|Inf in (loss|grad_norm|reward|log_prob)"
  - "RuntimeError: CUDA out of memory"
  - "FSDP .*(shard|reduce|reduction).* error"
  - "AttributeError.*_saved_grad_shard"                           # EXP-8 defect regression
  - "kl_loss appeared in metrics"                                 # KL disable regression
  - "RefPolicy.*spawned|ref_log_prob.*computed"                   # KL disable regression
  - "entropy_loss != 0"                                           # entropy disable regression
  - "cgroup pids.max .* (<=|too tight)"
  - "VAST_API_KEY found in container env"
  - "model not learning at step 10"                               # curve-analyst falsifier
```

Any of these patterns in PROGRESS.md routes to `codex-bridge --mode=code-rescue`
on the next orchestrator tick. The curve-analyst (Member B) emits
`STUCK: EXP-9 <reason>` lines that match these patterns when it detects
anomalies on WandB.

## Notes for runner

- **READ `## NON-NEGOTIABLES` FIRST.** All eight rules. Especially: end-to-end
  20-step training is the deliverable, full fast-circuit masking is mandatory,
  no KL, no entropy, 2-member monitoring team, auto-iterate on failure,
  codex-verify is pre-skipped, tear down before stop.
- **READ THE ISSUE COMMENTS BEFORE WORKING.** Issue #9's three comments carry
  the full forward-pass map, EMA cadence math, mask-recompute implementation
  spec, and the train-inference mismatch acknowledgement. They are the spec.
- `code_change: true` → branch `exp/9-m2-final-noKL-maskrecompute-aps` from
  `origin/vast-ai-workload` (NEVER from `main`). Apply target_modules
  changes; commit; `git push -u origin exp/9-m2-final-noKL-maskrecompute-aps`
  BEFORE provisioning so the branch survives if the laptop dies.
- **Implementation order:**
  1. Add `Mask.mask_recompute: bool = False` to
     `verl/workers/config/comm_eff.py` with `__post_init__` validation.
  2. Add `MASK_ELIGIBLE_TAGS = frozenset({TRAIN_TAG})` to
     `verl/workers/comm_eff/state.py`; when `state.mask.mask_recompute=True`,
     extend to `frozenset({TRAIN_TAG, "old_logprob"})`. Surface
     `mask_applications/old_logprob` in the metrics dict.
  3. Change `activation_mask.py:304-308` assert from `tag == TRAIN_TAG` to
     `tag in MASK_ELIGIBLE_TAGS`. Anchor's `path_tag=None` (GUARD 5) must still
     reject — verify `None not in MASK_ELIGIBLE_TAGS`.
  4. In `engine_workers.py::compute_log_prob`, when
     `comm_eff_state.mask.enabled AND comm_eff_state.mask.mask_recompute`,
     set `comm_eff_state.mask_active = True` inside the
     `with self._comm_eff_path("old_logprob"):` block, restore in `finally:`.
  5. Add `mask.mask_recompute` schema in
     `verl/trainer/config/actor/dp_actor.yaml` and
     `dp_actor_megatron.yaml` (default false).
  6. WRITE THE REGRESSION TEST FIRST in
     `tests/workers/comm_eff/test_activation_mask.py`. Confirm it FAILS on
     the current (pre-change) code path (verifying that `mask_recompute=True`
     is not yet wired). Then apply the implementation. Confirm it PASSES.
  7. Run the full 56-test CPU suite to confirm no regression:
     `pytest tests/workers/comm_eff/ tests/workers/config/ -v`.
- **No `1/(1-p)` rescale.** The mask form stays `h_tilde = h * mask` with
  mask ∈ {0,1} for both train and old_logprob paths. bf16-unsafe at p=0.95.
- **PRF key.** The substep counter naturally differs between
  `compute_log_prob` (one call per trainer step) and the PPO inner loop
  (N×E calls per trainer step), so the masks WILL differ between old_logprob
  and train by design. No additional key adjustment needed.
- **Anchor stays unmasked.** GUARD 5 (`mask_active=False`, `path_tag=None`
  inside `_maybe_comm_eff_anchor_refresh`) is unchanged.
  `MASK_ELIGIBLE_TAGS` must NOT contain `None`.
- **vLLM init on 4×H200.** Default `gpu_memory_utilization` ≈ 0.85 is fine on
  140 GB cards. No special override needed.
- **`comm_eff.enabled=true` is the only configuration this issue exercises.**
  No `comm_eff.enabled=false` regression cell. Comparison is against the EXP-3
  historical WandB curves.
- **Pre-create on the box at launch:**
  `mkdir -p /workspace/runs/communication-baseline/{iterations,curve-snapshots}` so the
  in-place hot-fix patches and curve-analyst snapshots have a canonical
  location. The aggregate `done.flag` is `/workspace/runs/communication-baseline/done.flag`
  (touched at end of launch script).
- **In-place iteration is allowed.** If a `STUCK:` pattern fires mid-run,
  the operator may SSH in via `runs/communication-baseline/handles/<id>.json::ssh_login`,
  hot-fix in `/workspace/verl` on the `exp/9-…` branch, capture a diff to
  `runs/communication-baseline/iterations/<N>.patch`, and relaunch the cell (with
  `EXPERIMENT_NAME=m2-final-noKL-maskrecompute-aps-iter<N>` so WandB
  doesn't auto-resume). Same pattern as EXP-12.
- Use the locked Vast.ai template via skills only; do not name a
  `template_hash` or `image`.

## Notes for analyst

- **Load-bearing criteria.** Criteria 1, 2, and 13 are the M2-capstone gates.
  Anything passing the comm_eff counters (criteria 3-12) but failing learning
  (criterion 13) is REVISE, not PASS — the comm_eff method is biasing GRPO
  and we have to iterate on `alpha` / `tau` / mask-key family until it doesn't.
- **Criterion 2 falsifier.** If `comm_eff/mask_applications/old_logprob == 0`
  while `comm_eff.mask.mask_recompute=true`, the implementation didn't wire
  through. This is a hard REVISE — extended masking is the central claim of
  this issue.
- **No dense control cell.** Don't expect a within-run dense vs compressed
  comparison. Compare criterion 13 against the EXP-3 historical WandB run for
  the same `critic/score/mean` column.
- **The `r`-drift signal.** If the issue surfaces a train-inference mismatch
  (the documented risk in issue comment §4), `actor/pg_clipfrac` will saturate
  near 1 (every token clipped) and/or `actor/pg_loss` will explode. Flag this
  as a candidate REVISE next-action: "wire IS correction via the existing
  `rollout_is_weights` channel in `losses.py:99-100,109,123`" before falling
  back to train-only masking.
- **The KL/entropy gates are bright lines.** Any nonzero `kl_loss` /
  `entropy_loss` / spawned ref worker is a non-negotiable failure, not a
  REVISE. Hard REVISE → next-action: re-check the four config overrides.
- **The anchor evolution criterion (10) is load-bearing.** A static EMA
  (`||ΔM_anchor|| == 0`) means the live G_anchor isn't reaching
  `update_anchor` → REVISE on the wiring, not PASS — even if everything else
  looks clean. EXP-12's verdict confirmed evolution was non-trivial across
  10 fires at the lean target count (196 matrices); 4 matrices here should be
  even more legible per-fire.

---

## Curve-analyst (Member B) — agent prompt

For the orchestrator's dispatch of Member B during the RUNNING state. Use
`subagent_type=general-purpose` (Opus) with `run_in_background=true`.

```
You are curve-analyst for EXP-9 (M2 final smoke). Member B of the 2-member monitor
team alongside training-log-monitor (Member A — system health).

Operating context:
- Vast handle: research/runs/communication-baseline/handles/<id>.json (read ssh_host/ssh_port/instance_id; you don't SSH yourself — Member A does)
- WandB project: verl_compression_research
- WandB entity: shamanework-pl
- Experiment name: m2-final-noKL-maskrecompute-aps (or m2-final-noKL-maskrecompute-aps-iter<N> if relaunched)
- Output log: research/runs/communication-baseline/curve-snapshots.log

Loop every 5 minutes for up to 90 minutes:

1. Fetch the latest WandB scalars via `wandb api` for the run with name `m2-final-noKL-maskrecompute-aps*`. If multiple runs match (iter1, iter2, ...), poll the most recent.
2. Read scalars at the latest logged step: critic/score/mean, actor/grad_norm, actor/pg_clipfrac, actor/pg_loss, response_length/mean, comm_eff/mask_applications/train, comm_eff/mask_applications/old_logprob, comm_eff/anchor_backwards, comm_eff/spectral_corrections, comm_eff/anchor_mask_applications, comm_eff/mask_ratio.
3. Append one line to research/runs/communication-baseline/curve-snapshots.log:
   `[<ISO>] step=<N> score=<X> grad=<Y> clip=<Z> pg=<W> len=<L> mask_train=<A> mask_old=<B> anchor=<C> spec=<D> anchor_mask=<E> mask_ratio=<F>`

4. Anomaly detection — if ANY of:
   - `actor/grad_norm` non-finite (NaN/Inf)
   - `actor/grad_norm` collapsing toward 0 (< 1e-5 for 3 consecutive snapshots)
   - `mask_applications/old_logprob` is 0 at any step ≥ 2
   - `anchor_mask_applications` > 0 at any step
   - `critic/score/mean` at step ≥ 10 is not higher than at step 1
   - `actor/pg_clipfrac` ≥ 0.95 (clip saturation — train-inference mismatch)
   - any NaN/Inf in any other scalar

   → append `STUCK: EXP-9 <one-line reason>` to research/PROGRESS.md so the orchestrator routes to codex-bridge --mode=code-rescue on the next tick. Then continue polling.

5. Exit on: aggregate done.flag visible at runs/communication-baseline/done.flag, Member A's monitor file says tmux DEAD, or 90-min timeout.

Output a structured report at exit: latest-step snapshot, anomalies detected, recommendation (dispatch_analyst | continue_iteration | teardown).

Never call vast-teardown. Never SSH to the box. Member A owns the SSH side.
```
