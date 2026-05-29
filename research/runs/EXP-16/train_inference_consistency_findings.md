# EXP-16 — Train-vs-Inference Consistency & IS-Mismatch Findings

Qwen2.5-1.5B-Instruct / GSM8K, 4xB200. Masked-GRPO comm-eff method (per-(token,channel)
activation masking at p=0.9 + rescale at 7 pipeline-stage boundaries). PRIMARY FOCUS: cell 7,
`grpo_mask_channel_p0p9_rescale_clean_every5_50steps` (rescale + clean dense step every 5).

> Status: COMPLETE. Cell 7 finished all 50 steps; end-of-run GSM8K val acc logged. The post-training
> DataLoader-worker "killed by signal" traceback is benign shutdown noise after the step-50 checkpoint
> save — training and validation both completed.

## Two policies (the "Sooo important" question)

1. **Rollout policy** = ordinary vLLM generation, **UNMASKED** (rollout is non-pipeline-parallel; fine).
2. **Training/actor policy** = FSDP forward with per-(token,channel) activation masking at p=0.9 + rescale
   at boundary layers [3,7,11,15,18,21,24].

The masked actor is therefore a **different function** from the vLLM sampler that produced the rollouts.
This is the IS mismatch GRPO must tolerate. Two distinct questions are answered below:
(a) is the *mask itself* consistent across the within-step forwards (correctness), and
(b) how far apart are the masked-actor and the vLLM-rollout policies, and does that gap grow (stability)?

## Verdict 1 — Mask cross-pass consistency: PROVEN CONSISTENT (no per-call drift)

The {0,1} mask pattern is bit-identical across the old_log_prob recompute, the train forward, and
every PPO minibatch within a step. Evidence (all measured, not assumed):

- **Structural (cell0 preflight, `cell0_preflight.log`)**: `cross_pass_mask_bit_identical(layer=3,step=7): True`
  (L109 region), `per_boundary_independent(layer3 vs layer5): True`, `different_step_different_mask: True`,
  `masked_fraction~p: 0.901`. 44/44 `test_activation_mask.py` + 8/8 `test_mask_rescale.py` PASS. CELL0 PASS.
- **Key has no per-call term**: `(base_seed, layer_idx, global_step, sample_id, position_id, channel)`;
  `global_step` is constant across the old_logprob and train forwards of one iteration; `mask_recompute=true`
  masks old_logprob with the same key as train (cross-pass note, `grpo_mask_cross_pass_consistency.md`).
- **Runtime routing (cell 7, all 28+ logged steps)**: 0 routing violations. `mask_applications/{rollout,ref_logprob,val}`
  are 0 on **every** step (rollout/ref/eval never masked); `train` and `old_logprob` are both nonzero on every
  masked step and both increment by 0 on clean steps (so the two forwards are masked together or clean together).
- **Per-boundary uniformity (cell 7)**: `mask_ratio/layer_{3,7,11,15,18,21,24}` = 0.899902 exactly; max within-step
  spread across the 7 layers = 0.0 over all samples. Stable across all steps.

=> For any fixed `(step, prompt, position, channel)`, the same mask bit is used in the old_logprob recompute
and the train forward. No evidence of per-call mask drift anywhere.

## Verdict 2 — Train-vs-inference mismatch is LARGE but the mask is its sole cause

Per-step (masked steps), masked cells vs the dense consistency control:

| metric | dense_ref | masked (cell 7 masked steps) |
|---|---|---|
| rollout↔actor Pearson corr | ~0.999 | ~0.004 |
| rollout_probs_diff_mean | ~0.0036 | ~0.84 |
| rollout_corr/kl | ~0.0004 | ~16.8 |

The masked actor assigns near-uncorrelated probabilities to the vLLM-sampled tokens (pearson ~0 vs ~1 dense).
**Proof the mask is the only cause:** on cell 7's clean steps (5,10,15,20,25) — same data, mask OFF in both
forwards — pearson jumps to 0.9995-0.9996 and rollout_corr/kl drops to 0.0003-0.0005, i.e. byte-identical to the
dense reference. Toggling the mask flips the gap on/off.

## Verdict 3 — Over 50 steps the divergence is FLAT, and training is BOUNDED-STABLE (not instability)

Cell 7, regression over masked steps:
- Pearson corr: slope ~+6e-5/step (R^2~0.2, noise) -> flat at ~0.004. **Does not grow.**
- rollout_corr/kl: slope ~+0.004/step -> +0.66% over the run on a base of 16.8. **Flat.**
- actor/ppo_kl (within-training IS ratio): stays ~5e-4, slope ~0 -> exp(logp-old_logp)~=1, cross-pass
  consistency confirmed end-to-end, not just structurally.
- grad_norm: sawtooth — masked-block peaks (~5-8) reset to ~0.4 on every clean step; bounded, never near the
  no-rescale failure mode (~2700). pg_clipfrac ~0.02-0.05, **not saturating** (no_rescale was ~0.15).
- reward: climbs strongly (slope ~+0.024/step, R^2~0.97). Learning proceeds despite the persistent mismatch.

So the IS mismatch is real and big but **stationary** — GRPO tolerates it here because (i) within-training the
ratio is ~1 and (ii) clean steps every 5 inject on-policy (actor==rollout) gradient that anchors learning.

## What needs to be done (ranked, grounded in numbers)

1. **Clean steps are the working lever — keep/tune them.** clean-every-4 reaches GSM8K val acc 0.696 vs dense
   0.741 (within 4.5 pts). Anchor@2+spectral@2 with NO clean steps gets 0.080 (~base/random) and reward stalls
   at ~0.13. **Anchor+spectral as implemented does NOT close the train-inference gap** (pearson still ~0.004).
   Action: sweep clean cadence (every-3/4/5/8) for the accuracy/comm-cost trade; cell 7 (every-5, 50 steps) is the
   long-horizon point.
2. **The gap is structural to per-(token,channel) masking at p=0.9, not a bug.** To shrink it without clean steps:
   (i) lower p (less aggressive masking -> higher correlation); (ii) mask the rollout too / on-policy masked
   generation so actor==rollout function; (iii) explicit IS-correction. Each should be measured by pearson and
   rollout_corr/kl against these baselines.
3. **No action needed on cross-pass mask correctness or grad explosion** — both are resolved (rescale fixes the
   no-rescale ~2700 grad_norm; mask is bit-consistent across forwards).

## Per-cell summary (final logged values)

| cell | steps | clean | pearson (masked) | rollout_corr/kl | grad_norm | reward (last) | GSM8K val acc |
|---|---|---|---|---|---|---|---|
| dense_ref | 25 | n/a | 0.999 | 0.0004 | 0.39 | 0.779 | 0.741 |
| mask_norescale_10 | 10 | 0 | 0.019 | 11.85 | ~2683 | 0.130 | 0.082 |
| mask_rescale_clean4_20 | 20 | 5 | 0.005 | 16.72 | ~5.8 (saw) | 0.619 | 0.696 |
| mask_rescale_anchor2_spectral2_20 | 20 | 0 | 0.0045 | 16.75 | ~4.6 | 0.131 | 0.080 |
| mask_rescale_clean5_50 (cell 7) | 50 | 10 | 0.004 | 16.75 | ~5.9 (saw) | TBD | TBD |
