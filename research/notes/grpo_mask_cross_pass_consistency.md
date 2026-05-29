# GRPO forward-mask consistency across one global step's forwards

Companion to [`grpo_update_forward_count.md`](grpo_update_forward_count.md). Scope:
the **forward activation mask + importance-sampling path only** (no anchor, no
spectral). Read-only audit of `vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`;
no source changed.

## Question

In one global step the same trajectories pass through the actor model twice — the
`old_log_prob` recompute (before the PPO loop) and the train forward (inside it). If
a given `(trajectory z, token s, boundary b, channel c)` got a **different** mask in
those two forwards, the per-token importance ratio `exp(log_prob − old_log_prob)`
would compare two different masked subnetworks and inject spurious off-policy noise.
Target rule:

```
M_t(z, s, b, c) = BernoulliKeep(seed, t, z, s, b, c)
   t=global step (same within a step, independent across steps)
   z=sample_id (trajectory)   s=position_id (token)
   b=pipeline boundary        c=hidden channel
key must NOT include: forward_call_id, ppo_minibatch_id, microbatch_id, fsdp_rank,
                      packed_offset, ckpt_replay_id
```

## Outcome

**No change needed — the IS-correctness property is already implemented, proven
against the code, and locked by tests.** The optional "global per-trajectory
`sample_id`" refactor was considered and **declined**: it does not affect IS
correctness and would change the mask realization (new baseline).

## The proof (IS correctness)

- **Key** = `(base_seed, layer_idx, global_step, sample_id, position_id, channel)`
  (`activation_mask.py:205-209`); the hook only *reads* context — no per-call counter
  (`:269-327`). Exactly `M_t(z,s,b,c)`. `layer_idx` ⇒ each pipe boundary an
  independent mask (`[3,7,11,15,18,21,24]` for L=28/pp=8); `global_step` in key ⇒
  the mask changes across steps.
- **`global_step` constant across the two forwards**: incremented only at
  `ray_trainer.py:1444` (once, pre-loop) and `:1787` (end-of-iter); old_log_prob
  (`:1577`) and update_actor (`:1683`) are the same iteration, both stamp
  `self.global_steps` (`:1274`,`:1331`); worker threads it onto the masker
  (`engine_workers.py:776-786`, `transformer_impl.py:728`).
- **Stable, packing-invariant identity**: per micro-batch
  `sample_ids=repeat_interleave(comm_eff_sample_id, seqlens)` and
  `position_ids=flat−starts` from the rmpad `cu_seqlens` (`transformer_impl.py:708-731`).
- **No row reorder/filter between the forwards**: `_balance_batch` reorders once at
  `:1541` (before both); only `union` (adds cols) + GRPO advantage (in-place,
  `core_algos.py:324-329`) run between; no `filter_groups`/dynamic sampling
  (`over_sample_rate=0`); dispatch is a deterministic contiguous `chunk(dp_size)` on a
  stable per-mesh mapping (`decorator.py:251,276`), identical for both (mesh `"actor"`).
- `mask_recompute=true` masks old_log_prob with the **same** key as train
  (`engine_workers.py:679-687`); a clean step makes both unmasked together; gradient-
  checkpoint replay and any extra PPO epoch regenerate the identical mask.
- **Test-locked** (`tests/workers/comm_eff/test_activation_mask.py`):
  `test_cross_packing_consistency` / `..._through_hook` (same token → same mask under
  any packing at a fixed step), `test_different_step_different_mask`,
  `test_boundary_indices_*`, `test_register_installs_hooks_on_boundaries_only`.

## Forward count (confirms the companion note)

Per global step (1024 seqs, mini 512, ppo_epochs=1, dynamic_bsz off, micro=1):
`old_logprob` = 1024 forward-only micro-forwards; `train` = 2 PPO updates × 512 =
1024 fwd+bwd → **2048 actor micro-forwards**, i.e. each trajectory gets exactly
**1 old + 1 train** forward, with the mask on every `(z,s,b,c)` byte-identical
across the two. So `exp(log_prob − old_log_prob)` compares the same masked
subnetwork (the clean comm-eff simulation, not the noisy `M_train ≠ M_old` version).

## Theory clarifications

- "Same mask per trajectory across the step's forwards" is the load-bearing
  IS-correctness property → **holds**.
- "Same" means **same per `(token, boundary, channel)` across forward passes**, not
  one shared vector for all tokens of a trajectory — the mask is per-(token, dim)
  (the method's spec); each token/channel draws its own bit, and that bit is reused
  across passes.
- Whether *different* trajectories share a mask is **irrelevant to IS** (each ratio
  only needs `z` to match itself across passes). Lowest variance = independent
  per-trajectory masks; batch-wide sharing would be IS-correct but highest-variance.

## Known, accepted residual (not a bug)

`comm_eff_sample_id` is a **per-rank `arange`** stamped post-dispatch
(`engine_workers.py:748`). Cross-pass correctness therefore relies on the global row
order being identical at both dispatch points — true for this launcher (no
filter/reorder), but *implicit*. Two trajectories sharing a per-rank local index
across DP ranks share a mask (a minor fidelity/variance nit). If anyone later enables
dynamic-sampling / `filter_groups`, revisit: stamping a global trajectory-unique id
once as a tensor column would make the guarantee structural and give each trajectory
an independent mask. Recorded so it isn't re-litigated as a cross-pass bug.

## Verify

```bash
cd /Users/shamane/Documents/verl
pytest tests/workers/comm_eff/test_activation_mask.py -v
```
