# REPRODUCIBILITY — EXP-18 (C5 matching run) — GENERATED

> **DO NOT HAND-EDIT.** Derived from `runs/EXP-18/resolved_params.txt` (ground truth).
> Regenerate: re-run log-writer on EXP-18.

## Run identity
- Experiment: EXP-18 · M4 curve-match
- Matching candidate: C5 (`curvematch_cleangrad_blend_e09_c5_d5`)
- Branch: `exp/18-anchorcleangrad-c5d5`
- Head commit: `45cd23811`
- Fork commit (at log-write time): `fb615809c0b8ce7a58446eaf790b5559e7e5c425`
- VERDICT: PASS
- Headline metric: reward 0.1455→0.8135 (dense 0.1348→0.8408); final|Δ|=0.027; plateau(20–50) mean|Δ|=0.036; whole-trajectory mean|Δ|=0.070

## Comm-eff + headline knobs (from resolved_params.txt, last-write-wins)

| Knob | Value |
|---|---|
| `actor_rollout_ref.actor.comm_eff.enabled` | `true` |
| `actor_rollout_ref.actor.comm_eff.clean_cadence` | `0` |
| `actor_rollout_ref.actor.comm_eff.mask.enabled` | `true` |
| `actor_rollout_ref.actor.comm_eff.mask.p` | `0.9` |
| `actor_rollout_ref.actor.comm_eff.mask.rescale` | `true` |
| `actor_rollout_ref.actor.comm_eff.mask.mask_recompute` | `true` |
| `actor_rollout_ref.actor.comm_eff.mask.pp_size` | `8` |
| `actor_rollout_ref.actor.comm_eff.mask.seed` | `0` |
| `actor_rollout_ref.actor.comm_eff.anchor.enabled` | `true` |
| `actor_rollout_ref.actor.comm_eff.anchor.cadence` | `5` |
| `actor_rollout_ref.actor.comm_eff.anchor.delay_K` | `5` |
| `actor_rollout_ref.actor.comm_eff.spectral.enabled` | `true` |
| `actor_rollout_ref.actor.comm_eff.spectral.correction_mode` | `blend` |
| `actor_rollout_ref.actor.comm_eff.spectral.blend_eta` | `0.9` |
| `actor_rollout_ref.actor.comm_eff.spectral.beta_anc` | `0.0` |
| `actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache` | `false` |
| `actor_rollout_ref.actor.comm_eff.spectral.ema_device` | `cpu` |
| `actor_rollout_ref.actor.comm_eff.spectral.max_targets` | `-1` |
| `actor_rollout_ref.actor.comm_eff.spectral.alpha` | `0.5` |
| `actor_rollout_ref.actor.comm_eff.spectral.tau` | `0.01` |
| `actor_rollout_ref.actor.comm_eff.spectral.cadence` | `1` |
| `actor_rollout_ref.actor.comm_eff.spectral.svd_mode` | `full` |
| `actor_rollout_ref.actor.comm_eff.spectral.basis_cache` | `cache` |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `64` |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `18432` |
| `actor_rollout_ref.actor.use_kl_loss` | `False` |
| `actor_rollout_ref.actor.entropy_coeff` | `0` |
| `algorithm.use_kl_in_reward` | `False` |
| `actor_rollout_ref.rollout.n` | `8` |
| `data.train_batch_size` | `128` |
| `data.max_prompt_length` | `1024` |
| `data.max_response_length` | `16384` |
| `trainer.total_training_steps` | `50` |
| `trainer.total_epochs` | `2` |
| `trainer.n_gpus_per_node` | `4` |
| `trainer.val_before_train` | `False` |
| `trainer.test_freq` | `100000` |
| `trainer.project_name` | `comm_eff_curve_match_m4` |
| `trainer.experiment_name` | `curvematch_cleangrad_blend_e09_c5_d5` |

## Verbatim resolved command (resolved_cmd.txt)

```
python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=/root/data/gsm8k/train.parquet data.val_files=/root/data/gsm8k/test.parquet data.train_batch_size=128 data.max_prompt_length=1024 data.max_response_length=16384 data.filter_overlong_prompts=True data.truncation=error algorithm.use_kl_in_reward=False actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000 actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.rollout.enable_chunked_prefill=False actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 actor_rollout_ref.rollout.n=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 trainer.critic_warmup=0 'trainer.logger=["console","wandb"]' trainer.project_name=comm_eff_curve_match_m4 trainer.experiment_name=curvematch_cleangrad_blend_e09_c5_d5 trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.save_freq=100000 trainer.test_freq=100000 trainer.total_epochs=2 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432 actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18432 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18432 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.calculate_log_probs=True algorithm.rollout_correction.rollout_is=null algorithm.rollout_correction.rollout_rs=null algorithm.rollout_correction.bypass_mode=false actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.actor.fsdp_config.use_orig_params=true actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.use_kl_loss=False algorithm.use_kl_in_reward=False actor_rollout_ref.actor.entropy_coeff=0 trainer.total_training_steps=50 trainer.val_before_train=False actor_rollout_ref.actor.comm_eff.enabled=true actor_rollout_ref.actor.comm_eff.clean_cadence=0 actor_rollout_ref.actor.comm_eff.mask.enabled=true actor_rollout_ref.actor.comm_eff.mask.p=0.9 actor_rollout_ref.actor.comm_eff.mask.rescale=true actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true actor_rollout_ref.actor.comm_eff.mask.seed=0 actor_rollout_ref.actor.comm_eff.mask.pp_size=8 actor_rollout_ref.actor.comm_eff.anchor.enabled=true actor_rollout_ref.actor.comm_eff.anchor.cadence=5 actor_rollout_ref.actor.comm_eff.anchor.delay_K=5 actor_rollout_ref.actor.comm_eff.spectral.enabled=true actor_rollout_ref.actor.comm_eff.spectral.alpha=0.5 actor_rollout_ref.actor.comm_eff.spectral.tau=0.01 actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.0 actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=false actor_rollout_ref.actor.comm_eff.spectral.ema_device=cpu actor_rollout_ref.actor.comm_eff.spectral.svd_mode=full actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache actor_rollout_ref.actor.comm_eff.spectral.max_targets=-1 actor_rollout_ref.actor.comm_eff.spectral.cadence=1 actor_rollout_ref.actor.comm_eff.spectral.correction_mode=blend actor_rollout_ref.actor.comm_eff.spectral.blend_eta=0.9
```
