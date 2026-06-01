# Verdict EXP-17 — 2026-06-01T06:24:59+00:00

## Result
VERDICT: PASS
note: M0 smoke: done.flag present, no NaN — harness validated

## Success criteria
- [ ] (paste from plan; analyst marks observed values here)

## Metrics summary
### train.jsonl (117 rows)
last row keys: actor/comm_eff/anchor_backwards, actor/comm_eff/anchor_batch_fraction, actor/comm_eff/anchor_grad_corrected, actor/comm_eff/anchor_mask_applications, actor/comm_eff/anchor_optimizer_steps, actor/comm_eff/anchor_rewards_recomputed, actor/comm_eff/anchor_rollouts_generated, actor/comm_eff/clean_steps, actor/comm_eff/mask_applications, actor/comm_eff/mask_applications/ckpt, actor/comm_eff/mask_applications/infer, actor/comm_eff/mask_applications/old_logprob, actor/comm_eff/mask_applications/ref_logprob, actor/comm_eff/mask_applications/rollout, actor/comm_eff/mask_applications/train, actor/comm_eff/mask_applications/val, actor/comm_eff/mask_ratio, actor/comm_eff/mask_ratio/layer_11, actor/comm_eff/mask_ratio/layer_15, actor/comm_eff/mask_ratio/layer_18, actor/comm_eff/mask_ratio/layer_21, actor/comm_eff/mask_ratio/layer_24, actor/comm_eff/mask_ratio/layer_3, actor/comm_eff/mask_ratio/layer_7, actor/comm_eff/spectral_corrections, actor/comm_eff/spectral_step, actor/entropy, actor/grad_norm, actor/loss, actor/lr, actor/perf/cpu_memory_used_gb, actor/perf/max_memory_allocated_gb, actor/perf/max_memory_reserved_gb, actor/pg_clipfrac, actor/pg_clipfrac_lower, actor/pg_loss, actor/ppo_kl, critic/advantages/max, critic/advantages/mean, critic/advantages/min, critic/returns/max, critic/returns/mean, critic/returns/min, critic/rewards/max, critic/rewards/mean, critic/rewards/min, critic/score/max, critic/score/mean, critic/score/min, global_seqlen/balanced_max, global_seqlen/balanced_min, global_seqlen/max, global_seqlen/mean, global_seqlen/min, global_seqlen/minmax_diff, num_turns/max, num_turns/mean, num_turns/min, perf/mfu/actor, perf/mfu/actor_infer, perf/throughput, perf/time_per_step, perf/total_num_tokens, prompt_length/clip_ratio, prompt_length/max, prompt_length/mean, prompt_length/min, response/aborted_ratio, response_length/clip_ratio, response_length/max, response_length/mean, response_length/min, response_length_non_aborted/clip_ratio, response_length_non_aborted/max, response_length_non_aborted/mean, response_length_non_aborted/min, rollout_corr/chi2_seq, rollout_corr/chi2_token, rollout_corr/k3_kl, rollout_corr/kl, rollout_corr/log_ppl_abs_diff, rollout_corr/log_ppl_diff, rollout_corr/log_ppl_diff_max, rollout_corr/log_ppl_diff_min, rollout_corr/ppl_ratio, rollout_corr/rollout_log_ppl, rollout_corr/rollout_ppl, rollout_corr/training_log_ppl, rollout_corr/training_ppl, step, timing_per_token_ms/adv, timing_per_token_ms/gen, timing_per_token_ms/update_actor, timing_s/adv, timing_s/agent_loop/compute_score/max, timing_s/agent_loop/compute_score/mean, timing_s/agent_loop/compute_score/min, timing_s/agent_loop/generate_sequences/max, timing_s/agent_loop/generate_sequences/mean, timing_s/agent_loop/generate_sequences/min, timing_s/agent_loop/num_preempted/max, timing_s/agent_loop/num_preempted/mean, timing_s/agent_loop/num_preempted/min, timing_s/agent_loop/slowest/compute_score, timing_s/agent_loop/slowest/generate_sequences, timing_s/agent_loop/slowest/num_preempted, timing_s/agent_loop/slowest/prompt_length, timing_s/agent_loop/slowest/response_length, timing_s/agent_loop/slowest/tool_calls, timing_s/agent_loop/tool_calls/max, timing_s/agent_loop/tool_calls/mean, timing_s/agent_loop/tool_calls/min, timing_s/gen, timing_s/old_log_prob, timing_s/reward, timing_s/save_checkpoint, timing_s/start_profile, timing_s/step, timing_s/stop_profile, timing_s/testing, timing_s/update_actor, timing_s/update_weights, training/epoch, training/global_step, training/rollout_actor_probs_pearson_corr, training/rollout_probs_diff_max, training/rollout_probs_diff_mean, training/rollout_probs_diff_std, training/rollout_probs_diff_valid, val-aux/num_turns/max, val-aux/num_turns/mean, val-aux/num_turns/min, val-aux/openai/gsm8k/reward/mean@1, val-core/openai/gsm8k/acc/mean@1
- val-aux/openai/gsm8k/reward/mean@1: mean=0.545402 min=0.0826384 max=0.735406
- val-core/openai/gsm8k/acc/mean@1: mean=0.545402 min=0.0826384 max=0.735406
- val-aux/num_turns/min: mean=2 min=2 max=2
- val-aux/num_turns/max: mean=2 min=2 max=2
- val-aux/num_turns/mean: mean=2 min=2 max=2
- step: mean=58 min=0 max=116
- global_seqlen/min: mean=83222 min=70709 max=98814
- global_seqlen/max: mean=92220.9 min=79311 max=111209
- global_seqlen/minmax_diff: mean=8998.95 min=2017 max=23134
- global_seqlen/balanced_min: mean=87452 min=76117 max=100839
- global_seqlen/balanced_max: mean=87799.9 min=76117 max=104896
- global_seqlen/mean: mean=87552.3 min=76117 max=100841
- actor/entropy: mean=5.67397 min=0.295963 max=5.92921
- perf/mfu/actor_infer: mean=0.20613 min=0.0876416 max=0.256065
- training/rollout_probs_diff_valid: mean=1 min=1 max=1
- training/rollout_probs_diff_max: mean=0.965828 min=0.15834 max=1
- training/rollout_probs_diff_mean: mean=0.81 min=0.00342151 max=0.877218
- training/rollout_probs_diff_std: mean=0.264339 min=0.00777718 max=0.299156
- training/rollout_actor_probs_pearson_corr: mean=0.0473874 min=-0.000131607 max=0.999586
- rollout_corr/training_ppl: mean=3.60056e+07 min=1.3609 max=4.91205e+07
- rollout_corr/training_log_ppl: mean=16.6246 min=0.281787 max=17.6504
- rollout_corr/kl: mean=16.2797 min=0.000296678 max=17.3846
- rollout_corr/k3_kl: mean=15.3386 min=0.000382576 max=16.3858
- rollout_corr/rollout_ppl: mean=1.46529 min=1.28876 max=3.79762
- rollout_corr/rollout_log_ppl: mean=0.329332 min=0.244096 max=0.39793
- rollout_corr/log_ppl_diff: mean=16.2953 min=3.69764e-05 max=17.3865
- rollout_corr/log_ppl_abs_diff: mean=16.2953 min=0.00167121 max=17.3865
- rollout_corr/log_ppl_diff_max: mean=17.9818 min=0.0104252 max=21.7314
- rollout_corr/log_ppl_diff_min: mean=13.5063 min=-0.067682 max=16.3241
- rollout_corr/ppl_ratio: mean=2.63427e+07 min=1.00005 max=3.80433e+07
- rollout_corr/chi2_token: mean=837.454 min=-0.999124 max=35838
- rollout_corr/chi2_seq: mean=-0.946773 min=-1 max=0.336874
- actor/pg_clipfrac: mean=0.0344968 min=0.000252882 max=0.0469771
- actor/ppo_kl: mean=0.000983188 min=-0.000772813 max=0.00248537
- actor/pg_clipfrac_lower: mean=0.000589555 min=0 max=0.00107034
- actor/pg_loss: mean=0.027981 min=0.00490163 max=0.0993723
- actor/loss: mean=0.027981 min=0.00490163 max=0.0993723
- actor/grad_norm: mean=5.98731 min=0.359795 max=9.6063
- actor/perf/max_memory_allocated_gb: mean=75.3169 min=65.8585 max=80.0737
- actor/perf/max_memory_reserved_gb: mean=86.9985 min=69.9492 max=91.9316
- actor/perf/cpu_memory_used_gb: mean=451.861 min=429.588 max=480.089
- actor/lr: mean=1e-06 min=1e-06 max=1e-06
- actor/comm_eff/mask_applications: mean=1219.63 min=21 max=2380
- actor/comm_eff/anchor_backwards: mean=0 min=0 max=0
- actor/comm_eff/spectral_corrections: mean=0 min=0 max=0
- actor/comm_eff/anchor_mask_applications: mean=0 min=0 max=0
- actor/comm_eff/anchor_grad_corrected: mean=0 min=0 max=0
- actor/comm_eff/anchor_rollouts_generated: mean=0 min=0 max=0
- actor/comm_eff/anchor_rewards_recomputed: mean=0 min=0 max=0
- actor/comm_eff/anchor_optimizer_steps: mean=0 min=0 max=0
- actor/comm_eff/anchor_batch_fraction: mean=1 min=1 max=1
- actor/comm_eff/clean_steps: mean=2.4569 min=0 max=5
- actor/comm_eff/spectral_step: mean=0 min=0 max=0
- actor/comm_eff/mask_applications/train: mean=784.603 min=14 max=1554
- actor/comm_eff/mask_applications/rollout: mean=0 min=0 max=0
- actor/comm_eff/mask_applications/old_logprob: mean=435.026 min=7 max=826
- actor/comm_eff/mask_applications/ref_logprob: mean=0 min=0 max=0
- actor/comm_eff/mask_applications/val: mean=0 min=0 max=0
- actor/comm_eff/mask_applications/infer: mean=0 min=0 max=0
- actor/comm_eff/mask_applications/ckpt: mean=0 min=0 max=0
- actor/comm_eff/mask_ratio: mean=0.899902 min=0.899902 max=0.899902
- actor/comm_eff/mask_ratio/layer_3: mean=0.899902 min=0.899902 max=0.899902
- actor/comm_eff/mask_ratio/layer_7: mean=0.899902 min=0.899902 max=0.899902
- actor/comm_eff/mask_ratio/layer_11: mean=0.899902 min=0.899902 max=0.899902
- actor/comm_eff/mask_ratio/layer_15: mean=0.899902 min=0.899902 max=0.899902
- actor/comm_eff/mask_ratio/layer_18: mean=0.899902 min=0.899902 max=0.899902
- actor/comm_eff/mask_ratio/layer_21: mean=0.899902 min=0.899902 max=0.899902
- actor/comm_eff/mask_ratio/layer_24: mean=0.899902 min=0.899902 max=0.899902
- perf/mfu/actor: mean=0.259325 min=0.199479 max=0.275297
- training/global_step: mean=58.5 min=1 max=116
- training/epoch: mean=0.5 min=0 max=1
- critic/score/mean: mean=0.535072 min=0.108398 max=0.795898
- critic/score/max: mean=1 min=1 max=1
- critic/score/min: mean=0 min=0 max=0
- critic/rewards/mean: mean=0.535072 min=0.108398 max=0.795898
- critic/rewards/max: mean=1 min=1 max=1
- critic/rewards/min: mean=0 min=0 max=0
- critic/advantages/mean: mean=-0.0219409 min=-0.089903 max=0.000853104
- critic/advantages/max: mean=2.47487 min=2.47487 max=2.47487
- critic/advantages/min: mean=-2.22261 min=-2.47487 max=-0.935413
- critic/returns/mean: mean=-0.0219409 min=-0.089903 max=0.000853104
- critic/returns/max: mean=2.47487 min=2.47487 max=2.47487
- critic/returns/min: mean=-2.22261 min=-2.47487 max=-0.935413
- response_length/mean: mean=238.334 min=196.387 max=288.29
- response_length/max: mean=1287.55 min=639 max=16384
- response_length/min: mean=11.0776 min=2 max=56
- response_length/clip_ratio: mean=1.68373e-05 min=0 max=0.000976562
- response_length_non_aborted/mean: mean=238.334 min=196.387 max=288.29
- response_length_non_aborted/max: mean=1287.55 min=639 max=16384
- response_length_non_aborted/min: mean=11.0776 min=2 max=56
- response_length_non_aborted/clip_ratio: mean=1.68373e-05 min=0 max=0.000976562
- response/aborted_ratio: mean=0 min=0 max=0
- prompt_length/mean: mean=103.667 min=98.1094 max=110.016
- prompt_length/max: mean=186.25 min=155 max=256
- prompt_length/min: mean=68.3362 min=56 max=74
- prompt_length/clip_ratio: mean=0 min=0 max=0
- num_turns/min: mean=2 min=2 max=2
- num_turns/max: mean=2 min=2 max=2
- num_turns/mean: mean=2 min=2 max=2
- timing_s/start_profile: mean=8.34388e-05 min=6.0557e-05 max=0.000383454
- timing_s/agent_loop/num_preempted/min: mean=-1 min=-1 max=-1
- timing_s/agent_loop/num_preempted/max: mean=-1 min=-1 max=-1
- timing_s/agent_loop/num_preempted/mean: mean=-1 min=-1 max=-1
- timing_s/agent_loop/generate_sequences/min: mean=0.352634 min=0.0750711 max=1.23804
- timing_s/agent_loop/generate_sequences/max: mean=7.50887 min=5.67325 max=40.6983
- timing_s/agent_loop/generate_sequences/mean: mean=4.74526 min=3.66961 max=6.21286
- timing_s/agent_loop/tool_calls/min: mean=0 min=0 max=0
- timing_s/agent_loop/tool_calls/max: mean=0 min=0 max=0
- timing_s/agent_loop/tool_calls/mean: mean=0 min=0 max=0
- timing_s/agent_loop/compute_score/min: mean=0.00440198 min=0.00320535 max=0.00593967
- timing_s/agent_loop/compute_score/max: mean=0.915363 min=0.199045 max=1.86622
- timing_s/agent_loop/compute_score/mean: mean=0.267 min=0.0444125 max=0.659383
- timing_s/agent_loop/slowest/generate_sequences: mean=7.41494 min=5.40283 max=40.6983
- timing_s/agent_loop/slowest/tool_calls: mean=0 min=0 max=0
- timing_s/agent_loop/slowest/compute_score: mean=0.106482 min=0.00592989 max=0.908772
- timing_s/agent_loop/slowest/num_preempted: mean=-1 min=-1 max=-1
- timing_s/agent_loop/slowest/prompt_length: mean=119 min=72 max=238
- timing_s/agent_loop/slowest/response_length: mean=1081.99 min=172 max=16384
- timing_s/gen: mean=9.9674 min=8.07437 max=42.9725
- timing_s/reward: mean=0.00279398 min=3.7531e-05 max=0.00580774
- timing_s/old_log_prob: mean=5.40184 min=4.75129 max=8.43133
- timing_s/adv: mean=1.44144 min=1.302 max=1.87818
- timing_s/update_actor: mean=7.73432 min=6.93066 max=8.9028
- timing_s/update_weights: mean=3.03818 min=2.7808 max=4.09606
- timing_s/step: mean=28.3576 min=25.3134 max=61.7621
- timing_s/stop_profile: mean=0.000126702 min=4.2771e-05 max=0.000303386
- timing_per_token_ms/update_actor: mean=0.0221471 min=0.0192715 max=0.0259288
- timing_per_token_ms/gen: mean=0.040989 min=0.0330795 max=0.185821
- timing_per_token_ms/adv: mean=0.00413751 min=0.00336838 max=0.00546189
- perf/total_num_tokens: mean=350209 min=304468 max=403364
- perf/time_per_step: mean=28.3576 min=25.3134 max=61.7621
- perf/throughput: mean=3125.78 min=1368.98 max=3478.67
- timing_s/testing: mean=47.4063 min=45.7015 max=49.1441
- timing_s/save_checkpoint: mean=5.80628 min=5.66099 max=5.90226

## Comparisons to baseline_run
(diff_against_baseline.py output — paste if a baseline was specified)

## next_actions (REVISE only)
(omit unless VERDICT is REVISE)

## Notes
M0 smoke: done.flag present, no NaN — harness validated
