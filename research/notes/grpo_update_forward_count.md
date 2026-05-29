# GRPO update and forward count for `vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`

Scope: this is a read-only trace of the launcher and code paths. No source code was changed. The counts below exclude validation, checkpoint save/load, and rollout weight sync. Validation runs extra rollout generation, but no PPO/actor update.

## Settings used

From `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`:

- Train prompts per global step: `TRAIN_BATCH_SIZE=128`
- Rollouts per prompt: `ROLLOUT_N=8`
- Generated sequences per global step: `128 * 8 = 1024`
- Actor PPO mini-batch: `PPO_MINI_BATCH_SIZE=64` prompts
- Effective actor mini-batch after rollout expansion: `64 * 8 = 512` sequences
- Actor PPO epochs: `ppo_epochs=1`
- Total training steps: `TOTAL_TRAINING_STEPS=100`
- Objective assumptions here: `use_kl_loss=False`, `use_kl_in_reward=False`, `entropy_coeff=0`

## PPO updates per global step

The trainer repeats the rollout batch by `rollout.n`, then `_update_actor()` multiplies `actor.ppo_mini_batch_size` by `rollout.n` before passing `mini_batch_size` to `train_mini_batch()`.

Formula:

```text
generated sequences per global step = train_batch_size * rollout_n
effective PPO mini-batch sequences = ppo_mini_batch_size * rollout_n
PPO optimizer steps per global step =
  (generated sequences / effective PPO mini-batch sequences) * ppo_epochs
= (1024 / 512) * 1
= 2
```

So one trainer global step performs **2 actor PPO optimizer updates**. Across `100` configured global steps, this is **200 actor optimizer updates**.

## Logical forward passes per global step

Counting logical actor/rollout stages, not every internal dynamic micro-batch:

| Stage | Count | Why it exists | Output names |
|---|---:|---|---|
| vLLM rollout generation | 1 generation stage over 1024 requests | Sample 8 completions for each of 128 prompts | `responses`, `response_mask`, `input_ids`, `attention_mask`, `position_ids`; `rollout_log_probs` only if rollout `calculate_log_probs=True` |
| Old policy log-prob forward | 1 actor inference forward over 1024 sequences | Build the PPO denominator/stable anchor for `ratio = exp(log_prob - old_log_prob)` | worker returns `log_probs`, `entropy`; trainer stores `old_log_probs`, `entropys` then drops `entropys` after metrics |
| Reference log-prob | 0 in this setting | Skipped because both KL switches are false | If enabled, would store `ref_log_prob` |
| Actor train forward/backward | 2 forward/backward passes over 512 sequences each | One per PPO mini-batch; each computes current policy log-probs and takes one optimizer step | model output `log_probs`; loss variable `log_prob`; metrics include `actor/pg_loss`, `ppo_kl`, clip fractions, `grad_norm` |

Therefore, per global step under these settings:

- **3 logical actor FSDP forwards**: 1 old-log-prob forward + 2 train forwards.
- **2 backward passes**: one for each PPO mini-batch.
- **2 optimizer steps**: one after each train forward/backward.
- **0 reference forwards** and **0 critic forwards**: no KL ref policy and GRPO disables critic by default.
- Including rollout generation as a model-using stage: **4 logical stages** per global step, but the rollout stage is autoregressive vLLM generation, not a single FSDP `self.module(...)` call.

## Strict count with dynamic batching off and micro-batch size 1

If `use_dynamic_bsz=False` and both actor/log-prob per-GPU micro-batch sizes are `1`, then each FSDP micro-forward contains exactly one trajectory per data-parallel rank.

Let `dp_size = NGPUS_PER_NODE` for this launcher's FSDP actor path, because sequence parallel is not enabled.

## Verdict on masking

For the communication-efficient research question, the hard rule should be:

> Every actor FSDP forward inside the trainer that consumes the rollout trajectory for the PPO update must be masked.

In this single global step, that means:

1. The forward that computes `old_log_probs` must be masked.
2. The forward that computes current-policy `log_probs` for the actor loss must be masked.

If `old_log_probs` is dense/unmasked but current `log_probs` is masked, the PPO ratio compares two different computation graphs/subnetworks. That no longer cleanly simulates a communication-efficient actor; it corrupts the meaning of `ratio = exp(log_prob - old_log_prob)`.

Also, "masked" should not mean independent arbitrary randomness per forward call. The stronger rule is:

> For a fixed global step, the same trajectory/token/activation coordinate must get the same mask in every trainer-side actor forward where that coordinate appears.

That includes the old-log-prob forward, the train forward for the PPO mini-batch containing that trajectory, and any gradient-checkpoint replay inside the masked train backward region. Independent masks can be studied as a separate noisy ablation, but they are not the clean communication-efficient baseline.

This rule applies to trainer actor forwards in the PPO update. It does not require masking validation, checkpoint save/load, rollout weight sync, or a skipped reference model. The rollout generation stage is vLLM generation, not the FSDP actor trainer forward counted here.

### Per trajectory

Each generated trajectory stream goes through the actor FSDP model:

1. **Once for old log-prob forward**: forward-only, output worker `log_probs`, stored by trainer as `old_log_probs`.
2. **Once for actor training**: forward + backward, output worker/model `log_probs`, used in PPO loss as current `log_prob`.

So with `ppo_epochs=1`, each trajectory has **2 actor model forwards inside the trainer**. The two PPO mini-batches do not replay the same trajectory twice; they split the 1024 trajectories into two disjoint 512-sequence optimizer updates.

### Per global step

Global micro-forward executions across all data-parallel ranks:

```text
old_log_prob micro-forwards = 1024 trajectories * 1 = 1024
train micro-forwards        = 1024 trajectories * 1 = 1024
ref micro-forwards          = 0
critic micro-forwards       = 0

total actor FSDP micro-forwards inside trainer = 2048
```

Per rank, these happen as:

```text
old_log_prob forwards per rank = 1024 / dp_size
train forwards per rank        = 1024 / dp_size
total forwards per rank        = 2048 / dp_size
```

Examples:

| Actor DP size | Old-log-prob forwards/rank | Train forwards/rank | Total actor forwards/rank |
|---:|---:|---:|---:|
| 4 | 256 | 256 | 512 |
| 8 | 128 | 128 | 256 |

Masking implication for this research setup: every trainer-side actor model entry in the global step is required to be masked. Under the micro-batch-1 assumption, each trajectory appears in **2 required masked full-model passes** per global step: one old-log-prob forward and one train forward. Across the whole global step, this is **2048 required masked actor FSDP micro-forwards**. Because model gradient checkpointing is enabled, any internal checkpoint replay inside the masked train backward region should also reuse the same mask key; it is part of the same trainer-side masked execution, not a dense escape path.

## Symbolic mask rule

Let a generated trajectory be indexed by:

```text
z = (prompt_index i, rollout_index r)
```

Let a response token position be `s`, a masked boundary/layer be `b`, and a hidden channel be `c`. At trainer global step `t`, define the mask as a deterministic keyed function:

```text
M_t(z, s, b, c) = BernoulliKeep(seed, t, z, s, b, c)
```

The key must **not** include:

```text
forward_call_id
ppo_minibatch_id
microbatch_id
fsdp_rank
packed_token_offset
checkpoint_replay_id
```

For any trainer-side actor pass `a` in the same global step:

```text
a in P_t = {old_logprob pass, train pass, checkpoint replay passes}
```

the masked activation should be:

```text
h'_{a,t,z,s,b,c} = h_{a,t,z,s,b,c} * M_t(z,s,b,c) / keep_prob
```

So the old/current PPO ratio is comparing policies under the same masked subnetwork for that trajectory/token:

```text
ratio_t(z,s)
  = exp(log pi_theta^M_t(y_{z,s} | x_z)
        - log pi_old^M_t(y_{z,s} | x_z))
```

The bad/noisy version is:

```text
ratio_bad(z,s)
  = exp(log pi_theta^M_train(y_{z,s} | x_z)
        - log pi_old^M_old(y_{z,s} | x_z)),
    where M_train != M_old
```

That adds mask-sampling noise directly into the PPO denominator/numerator mismatch. It is a different experiment, not the clean communication-efficient simulation.

## How FSDP data parallelism divides these forwards

FSDP is still data-parallel in the batch dimension. It is not "pure DDP" internally, because FSDP shards parameters, gradients, and optimizer state instead of keeping a full replica on every rank. But from the data-flow/counting perspective, each DP rank receives a different slice of the trajectories and runs the same forward/backward program on its local slice.

For this launcher:

- The actor uses `strategy=fsdp`.
- `ulysses_sequence_parallel_size=1`.
- The FSDP engine computes `dp_size = torch.distributed.get_world_size() // ulysses_sequence_parallel_size`.
- Therefore `dp_size = NGPUS_PER_NODE`.
- On a 4x H200 run, actor `dp_size=4`.
- On an 8x H200 run, actor `dp_size=8`.
- `ROLLOUT_TP=2` is for vLLM rollout tensor parallelism; it is not the actor FSDP DP size.

The division is deterministic/even, not random:

1. The driver builds the global `DataProto` with 1024 trajectories.
2. The dispatch function chunks that batch into `dp_size` equal chunks along dim 0.
3. Each FSDP rank receives one chunk.
4. For `micro_batch_size_per_gpu=1`, each rank runs one trajectory per local micro-forward.
5. All ranks progress through aligned local micro-forward slots, but each rank is processing different trajectories.

So dividing by `dp_size` is correct **for per-rank local forward counts**:

```text
total actor FSDP micro-forwards across all ranks = 2048
local actor FSDP micro-forwards per rank = 2048 / dp_size
```

The global count is not reduced by FSDP. FSDP just runs those forwards in parallel across DP ranks while sharding model state and synchronizing gradients/parameters through FSDP collectives.

## Micro-batch caveat

`actor_rollout_ref.actor.use_dynamic_bsz=True` and `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True`, with max token budgets of `36864` per GPU. In the FSDP engine, dynamic batching calls `prepare_micro_batches()` and splits by token length. So the exact number of low-level `self.module(...)` forwards is data-dependent on sampled response lengths. The configured `PPO_MICRO_BATCH_SIZE_PER_GPU=1` and `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1` are not the controlling split while dynamic batching is enabled.

Useful formulas:

```text
old_log_prob low-level forwards per rank
  = dynamic_micro_batches(1024 / dp_size sequences, max 36864 tokens/GPU)

train low-level forwards per rank
  = 2 * dynamic_micro_batches(512 / dp_size sequences, max 36864 tokens/GPU)
```

At the logical PPO level, the count remains exactly **2 updates per global step**.

Gradient-checkpointing note: the launcher enables model gradient checkpointing. Checkpoint replay during train backward is not an additional trainer micro-batch, but it is still inside the trainer-side actor execution and should remain masked under the hard rule.
