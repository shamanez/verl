VERDICT: PASS

# Verdict EXP-17 — 2026-06-01T06:25:39Z

## Result
VERDICT: PASS

This is a CHARACTERIZATION run. PASS = learning + stability + stationarity + repair
+ no-drift-to-random. Every gate box below is checked. Dense parity is a bonus, not
a gate — and EXP-17 lands at near-parity anyway (final val 0.7354 vs dense 0.7415,
−0.82%) with only 5 true-gradient steps over 116. The four load-bearing long-horizon
reads all point PASS-ward: clean-step grad_norm trends DOWN (repair holding), clean-step
true-policy entropy is sharp and trending DOWN like dense (NOT climbing to random),
the train-inference gap is a clean-resettable sawtooth (NOT a monotone ratchet), and
per-clean-step repair is full and constant at every one of the 5 clean cycles.

## Success criteria
- [x] Reaches global_step >= 116 (2 epochs); no NaN/Inf in any loss/grad_norm/reward/log_prob field (observed: max step = 116, full NaN/Inf scan over all 117 reconstructed rows = clean)
- [x] actor/grad_norm finite throughout; masked-step median < 10; clean-step ~0.4 (observed: masked median = 6.32, range 3.71–9.61, all finite & <10; clean steps = 0.426/0.403/0.439/0.378/0.360)
- [x] val/test_score_final >= val/test_score_step0 + 0.05 (observed: step0 = 0.0849, final = 0.7354, delta = +0.6505; threshold 0.1349)
- [x] Clean-step grad_norm does NOT trend upward across 20/40/60/80/100 (observed: linear slope = −0.00078/step, R²=0.57 — trending DOWN; each clean step still fully re-anchors)
- [x] True policy not drifting to random: clean-step entropy sharp ~0.4, trending flat-or-down like dense (0.38→0.24), NOT climbing toward masked ~5.9; clean-step val >> 0 (observed: clean-step entropy 0.393→0.413→0.369→0.339→0.296, slope −0.0013/step R²=0.84; step20 val=0.1319, step100 val=0.7202 — all >> 0)
- [x] Train-inference gap stationary OR clean-resettable sawtooth (observed: pearson slope +4.9e-7/step R²=0.000 over masked steps; steps-since-clean binning FLAT vs position, R²=0.03; gap fully RESETS at every clean step pearson 0.004→0.9996, kl 17→0.0004 — clean-resettable sawtooth, not a ratchet. See Notes for the kl-level caveat.)
- [x] Masked-step pg_clipfrac not saturating, stays well below 0.15 (ideally <=0.08), not climbing toward saturation between clean steps (observed: masked median = 0.0365, max = 0.0470, all <0.08; within-window slope +1.1e-4 with R²=0.52 but band is 0.019–0.047 — nowhere near 0.15 saturation, NOT a tighten-cadence trigger)
- [x] Clean steps fire at exactly 20/40/60/80/100: mask counters FREEZE, grad→~0.4, ppo_kl→~1e-5, clipfrac→~4e-4, clipfrac_lower→0 (observed: clean_steps counter increments 1→2→3→4→5 at those steps; mask_applications/train & /old_logprob delta = +0 AT each clean step then +14 next step; grad 0.36–0.44; ppo_kl −3e-5…+5e-5; clipfrac 2.5e-4–4.6e-4; clipfrac_lower = 0 at all 5)
- [x] Masking confined to actor-train; anchor+spectral OFF (observed: mask_applications/{rollout,ref_logprob,val,infer,ckpt} == 0 every step; anchor_backwards == spectral_corrections == 0 every step — zero violations across 116 steps)
- [x] Learning-speed-vs-dense diagnostic REPORTED (not a gate) — see Comparisons section
- [x] Boundary-activation comm-savings number REPORTED (~85.5%; 19/20 masked at p=0.9) — see Metrics summary

## Metrics summary
- val/test_score: step0 = 0.0849 → final(116) = 0.7354 (Δ +0.6505; gate step0+0.05 = 0.1349)
- val trajectory (TEST_FREQ=10): 0:0.0849, 10:0.0826, 20:0.1319, 30:0.4882, 40:0.5534, 50:0.6899, 60:0.7036, 70:0.7248, 80:0.7339, 90:0.7187, 100:0.7202, 110:0.7225, 116:0.7354
- critic/score/mean (reward): steps-to-reward>=0.5 = step 44; final(116) = 0.7490; reward slope = +0.0064/step (R²=0.87)
- actor/grad_norm: masked median 6.32 (range 3.71–9.61, all finite); clean-step 0.426/0.403/0.439/0.378/0.360 (slope −0.00078/step)
- actor/entropy: masked ~5.90 (mask-forward artifact — IGNORE for policy health); clean-step (true policy) 0.393→0.296 (slope −0.0013/step)
- actor/pg_clipfrac: masked median 0.0365 (max 0.0470, all <0.08); clean-step ~4e-4
- train-inference gap (masked steps): pearson(actor,rollout) mean 0.004 (slope +4.9e-7/step, R²~0); rollout_corr/kl mean 17.01 (range 16.69–17.39); rollout_probs_diff_mean mean 0.846 (range 0.819–0.877)
- per-clean-step repair (just-before → at-clean): pearson 0.004→0.9996, kl ~17→0.0004, entropy ~5.9→~0.4, grad ~6.5→~0.4 — FULL and CONSTANT at all 5 clean cycles
- comm-eff counters: clean_steps = 5 (final); mask_ratio = 0.89990 (p=0.9 target); 7 boundaries [3,7,11,15,18,21,24] all at 0.8999
- boundary-activation comm savings = (19/20 masked steps) × 0.9 mask_p = 85.5% (matches plan target)
- budget: month_spent = $19.84, cap = $1500 — well within (running_count = 0; box retained for EXP-18, not charged to this verdict)

## Comparisons to baseline_run: EXP-16
Read from runs/EXP-16/metrics/{dense cell6, clean@5/50 cell7, clean@4/20 cell4} train.logs (np.float64-wrapped; reconstructed). EXP-16 has no top-level metrics/train.jsonl, so the verbatim `diff_against_baseline.py --baseline EXP-16` is a no-op cross-check; the substantive diff (baseline_diff.md) is against a reconstructed dense cell6 baseline.

| run | clean cadence | final val | steps-to-reward>=0.5 | final reward |
|---|---|---|---|---|
| EXP-17 (this) | K=20, 116 steps | 0.7354 | 44 | 0.749 |
| dense cell6 | none (every step) | 0.7415 | 6 | 0.779 |
| clean@5/50 cell7 | K=5, 50 steps | 0.7293 | 18 | 0.778 |
| clean@4/20 cell4 | K=4, 20 steps | 0.6960 | 17 | 0.619 |

Learning-speed read: EXP-17's final-val gap to dense is only −0.0061 (−0.82%) and it actually EXCEEDS both K=5 (+0.006) and K=4 (+0.039) reference trajectories on final val — sparser clean cadence (K=20) did NOT cost final quality over the long horizon. The cost is learning SPEED: steps-to-reward>=0.5 = 44 (vs dense 6, K=5 18, K=4 17) — the 4× sparser true-gradient injection means the masked windows carry the policy more slowly between re-anchors, but each clean step still fully repairs and the curve keeps climbing (reward slope +0.0064/step, R²=0.87, monotone). Reward rise is attributable to the clean+post-clean dynamics: masked windows hold roughly flat-to-slowly-rising, clean steps re-anchor to the true gradient. This is exactly the long-horizon characterization the run was designed to produce.

## Resolved parameters (ground truth)
Source: `resolved_params.txt` (extracted from train.log's main_ppo `set -x` trace, NOT the plan). 75 params captured from 1 main_ppo invocation; `resolved_cmd.txt` holds the verbatim expanded command.

Comm-eff + headline knobs (verbatim):
```
actor_rollout_ref.actor.comm_eff.enabled=true
actor_rollout_ref.actor.comm_eff.clean_cadence=20
actor_rollout_ref.actor.comm_eff.mask.enabled=true
actor_rollout_ref.actor.comm_eff.mask.p=0.9
actor_rollout_ref.actor.comm_eff.mask.rescale=true
actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true
actor_rollout_ref.actor.comm_eff.mask.pp_size=8
actor_rollout_ref.actor.comm_eff.mask.seed=0
actor_rollout_ref.actor.comm_eff.anchor.enabled=false
actor_rollout_ref.actor.comm_eff.spectral.enabled=false
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=64
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.entropy_coeff=0
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.name=vllm
algorithm.use_kl_in_reward=False
algorithm.adv_estimator=grpo
data.train_batch_size=128
data.max_prompt_length=1024
data.max_response_length=16384
trainer.total_training_steps=116
trainer.experiment_name=grpo_mask_channel_p0p9_rescale_clean_every20_2epoch
```

DIVERGENCE CHECK: NONE. Every launched value matches the plan's `## Notes for runner` env block exactly — clean_cadence=20, mask p=0.9 rescale=true recompute=true, anchor+spectral OFF, no-KL no-entropy GRPO, lr=1e-6, n=8, train_batch=128, mini_batch=64, max_response=16384, total_training_steps=116. `mask.pp_size=8` (⇒ 7 internal pipeline boundaries) is consistent with the plan's stated boundaries [3,7,11,15,18,21,24], confirmed by the 7 per-layer mask_ratio counters all reading 0.8999. The run is fully reproducible from resolved_cmd.txt.

## Notes
- METRICS RECONSTRUCTED: metrics/ contained only sync-errors.log — the per-step jsonl was never synced (sync-metrics rsync errored repeatedly on a non-existent `/workspace/runs/EXP-17/hotfix-patches` dir; see metrics/sync-errors.log). I reconstructed metrics/train.jsonl (117 rows, steps 0..116) from train.log via runs/EXP-17/reconstruct_metrics.py — the trainer prints the full verl metrics dict every step. The reconstruction reproduces the prompt's known result (final val 0.7354, clean steps at 20/40/60/80/100, clean grad_norm 20→0.426 / 40→0.403). Completion confirmed independently: done.flag present (2026-06-01T05:29:05Z), train.log shows "Final validation metrics: ...0.7354..." and reaches step:116. Two minor step-label offsets vs the prompt's recap (prompt said "80:0.7248, 100:0.7187" — the log has 70:0.7248, 80:0.7339, 90:0.7187, 100:0.7202; final 0.7354 matches exactly); all values in this verdict are grepped from the reconstructed jsonl / train.log.
- GAP-STATIONARITY CAVEAT (the one subtle read): masked-step rollout_corr/kl (16.69→17.39) and rollout_probs_diff_mean (0.819→0.877) show high-R² (0.98 / 0.80) tiny-positive slopes, which in isolation could look like a slow ratchet. The plan's decisive test resolves it: the steps-since-last-clean binning is FLAT vs position (bucket-mean pearson slope +1.6e-5, R²=0.03) and the gap RESETS COMPLETELY at every clean step (pearson 0.004→0.9996). The masked forward pass structurally produces a ~kl-17 gap regardless of true-weight quality (it is a mask artifact, not weight drift); the slow rise in the *absolute kl level* tracks the improving true policy's training-perplexity, and it is fully erased at each clean step. This is the clean-resettable sawtooth the criterion explicitly permits, NOT a monotone across-run ratchet on the true weights (whose health is read from clean-step entropy 0.39→0.30 trending DOWN and clean-step grad 0.43→0.36 trending DOWN). PASS stands.
- BOX RETAINED for EXP-18 (instance 38877541). Per the prompt, the EXP-17 ledger row is already flipped to COMPLETE — this verdict triggers NO teardown. No teardown attempted.
- FORWARD-LOOK (open as NEW follow-up issues, NOT REVISE children of EXP-17, per plan §Notes-for-analyst): (1) the K∈{10,20,40,never} comm-savings/quality sweep — EXP-17 shows K=20 holds final quality but slows learning-speed; quantifying the K↔speed↔savings frontier is the natural next axis; (2) the clean-only-vs-masked+clean ablation (issue Open Question 1 — attribute reward rise to masked windows vs clean steps); (3) the cheaper *continuous* on-policy-masked correction every step (instead of periodic re-anchoring) hinted at in the predicate's STOP branch — worth exploring even on this PASS, since the masked-window learning-speed drag is the one cost K=20 incurs.
