# implementation-auditor brief

## Verdict

- **Anchor no-rerollout is structurally correct.** The anchor path replays the already generated batch, does not regenerate rollouts/rewards, does not step the optimizer, and uses an anchor-only PG loss with `ratio=1`.
- **Stale rollouts are matched to the intended rollout-generation snapshot in paired replay mode.** The replay ring stores one generator snapshot per trainer global step and clones mini-batches with `_comm_eff_global_step`; post-warm exactness/canary checks enforce replaying the stale batch with that global step's generation snapshot.
- **But `delayed_ef` is not always the documented exact same-batch/same-weights residual under multiple PPO mini-batch optimizer ticks.** Later mini-batch ticks within one trainer global step use an anchor snapshot from rollout-generation time, while `G_comp_ring(t-K)` was computed at live training weights after earlier mini-batch optimizer steps. That makes the batch match exact, but the weights differ by within-global-step optimizer drift.
- **Additional caveat:** `delayed_ef` is documented as `beta_anc=0`, but validation does not enforce this, and at least one run config uses `beta_anc=0.5`, turning it into an EMA-smoothed replay gradient rather than an exact per-fire residual.

## Evidence

### Anchor no-rerollout path

- `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1269-1303` documents anchor mode as no rollout/reward recompute, no optimizer step, and no correction.
- `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1736-1751` runs `_forward_backward_batch_inner(anchor_data, anchor_loss_function)` for the anchor clone.
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/anchor.py:137-225` implements `anchor_pg_loss`: it ignores `old_log_probs`, sets `ratio = 1`, and computes `per_token_pg = -advantages * log_prob`.
- `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1812-1825` asserts the anchor fired no mask hooks and took no optimizer step.
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py:311-333` initializes explicit canary counters for anchor mask applications, grad correction, rollouts, reward recomputes, and optimizer steps; `/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py:963-967` surfaces them in metrics.

### Stale snapshot and paired replay correctness

- Legacy snapshot queue logic is in `/Users/shamane/Documents/verl/verl/workers/comm_eff/anchor.py:228-272`; paired replay ring logic is in `/Users/shamane/Documents/verl/verl/workers/comm_eff/anchor.py:391-512`.
- The ring explicitly stores one generator snapshot per global step and per-tick batch clones: `/Users/shamane/Documents/verl/verl/workers/comm_eff/anchor.py:391-417`, with retention in `/Users/shamane/Documents/verl/verl/workers/comm_eff/anchor.py:420-437`.
- The FSDP engine stores one snapshot per `_comm_eff_global_step` before mini-batch training if missing: `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1422-1439`, stores retained mini-batch clones at `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1445-1452`, and retrieves exact replay batches at `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1480-1499`.
- Post-warm exact replay and minimum realized weight delay are asserted at `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1500-1518`; legacy exactness is similarly asserted at `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1535-1556`.
- Canary hashing is checked at `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1672-1693`, with helpers in `/Users/shamane/Documents/verl/verl/workers/comm_eff/anchor.py:336-388`.

Conclusion: for paired replay, the stale batch is replayed with the stale **rollout-generation** weights for its trainer global step. This is the right no-rerollout pairing.

### Mini-batch optimizer tick caveat in `delayed_ef`

- The trainer generates rollouts once per global step at `/Users/shamane/Documents/verl/verl/trainer/ppo/ray_trainer.py:1499-1505`, computes old log-probs/rewards/advantages through `/Users/shamane/Documents/verl/verl/trainer/ppo/ray_trainer.py:1530-1667`, updates the actor at `/Users/shamane/Documents/verl/verl/trainer/ppo/ray_trainer.py:1681-1684`, then syncs actor weights to rollout workers at `/Users/shamane/Documents/verl/verl/trainer/ppo/ray_trainer.py:1707-1709`.
- The trainer stamps `_comm_eff_global_step` once for the whole actor update at `/Users/shamane/Documents/verl/verl/trainer/ppo/ray_trainer.py:1331`; it is threaded into the engine at `/Users/shamane/Documents/verl/verl/workers/engine_workers.py:772-808`.
- `train_mini_batch` splits that global-step batch and calls `train_batch` for each mini-batch at `/Users/shamane/Documents/verl/verl/workers/engine_workers.py:236-303`; `BaseEngine.train_batch` zeroes grads, runs anchor, fast backward, correction, then an optimizer step at `/Users/shamane/Documents/verl/verl/workers/engine/base.py:194-226`.
- The actual optimizer step is `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:3095-3096`.
- Anchor/spectral clocks advance per `train_batch`, not per rollout global step: `anchor_step += 1` at `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1341-1343`; `spectral_step += 1` at `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:2870-2876`; docs for the optimizer-tick clock are `/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py:700-732`.
- `delayed_ef` claims exact same `(batch, theta)` as anchor at `/Users/shamane/Documents/verl/verl/workers/comm_eff/spectral_filter.py:879-887` and in config docs at `/Users/shamane/Documents/verl/verl/workers/config/comm_eff.py:279-285`.
- Implementation looks up `G_comp_ring(t-K)` by optimizer tick at `/Users/shamane/Documents/verl/verl/workers/comm_eff/spectral_filter.py:1115-1122`, and stores current raw `G_comp` before correction at `/Users/shamane/Documents/verl/verl/workers/comm_eff/spectral_filter.py:1183-1195`. The residual is `delta = anc - ring_grad` at `/Users/shamane/Documents/verl/verl/workers/comm_eff/spectral_filter.py:930-940`.

Therefore, with more than one PPO mini-batch optimizer step per rollout global step:

- The replayed **batch** for tick `t-K` is exact.
- The anchor snapshot is the rollout-generation snapshot for that batch's global step.
- But `G_comp_ring(t-K)` was computed at the live weights of optimizer tick `t-K`, which for later mini-batches are already after earlier optimizer steps in the same global step.

On the fixed control surface, `/Users/shamane/Documents/verl/research/runs/FIXED_CONTROL_SURFACE.md:28-31` sets `train_batch_size=128`, `ppo_mini_batch_size=64`, and `rollout.n=8`; `/Users/shamane/Documents/verl/verl/trainer/ppo/ray_trainer.py:1344-1355` multiplies PPO mini-batch size by `rollout.n`, so the post-rollout batch is `1024` samples and mini-batches are `512`: **two optimizer ticks per trainer global step**. With `delay_K=20` and `cadence=20`, retained ticks are multiples of 20, i.e. the second mini-batch of a global step; the ring snapshot for that batch is the previous first tick's rollout-generation snapshot, so `realized_weight_delay` becomes `K+1`, which is allowed by the `>= K` assert but is not the same weights as `G_comp_ring(t-K)`.

### Other implementation checks

- `delayed_ef`'s `beta_anc=0` assumption is not enforced: validation only checks anchor enabled plus paired replay at `/Users/shamane/Documents/verl/verl/workers/config/comm_eff.py:831-840`. `/Users/shamane/Documents/verl/research/runs/EXP-37E/config.yaml:1-24` uses `delayed_ef` with `beta_anc: 0.5`, contradicting the exact residual documentation.
- `signed_ema` is explicitly an EMA/sign operator, not an exact residual: `/Users/shamane/Documents/verl/verl/workers/comm_eff/spectral_filter.py:403-441`.
- Q ownership is coherent: compressor setup reads `anchor_owns_q` at `/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py:510-535`; fast-path basis updates are skipped at `/Users/shamane/Documents/verl/verl/workers/engine_workers.py:900-918`; direct update asserts not anchor-owned at `/Users/shamane/Documents/verl/verl/workers/comm_eff/powersgd_activation.py:708-716`; anchor hook/update/broadcast flow is `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1707-1719`, `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1765-1784`, `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:2061-2123`, backed by `/Users/shamane/Documents/verl/verl/workers/comm_eff/powersgd_activation.py:1110-1255`.
- DP anchor-gradient reduction is present and averaged over DP ranks at `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1004-1100`, with SUM/divide at `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:1080-1085`.

## Bottom line

No-rerollout anchor replay is correct, and paired replay matches stale rollouts to their rollout-generation snapshots. The main implementation caveat is the `delayed_ef` proof/claim: it is exact only when the retained tick is the first optimizer tick for that rollout global step, or when there is one PPO mini-batch optimizer step per global step. Under the current fixed surface's two mini-batch ticks per global step, later retained ticks subtract a `G_comp_ring(t-K)` computed at post-mini-batch-update weights from an anchor gradient computed at rollout-generation weights.
