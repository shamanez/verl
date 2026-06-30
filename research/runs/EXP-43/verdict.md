# Verdict EXP-43 — 2026-06-30T00:00:00Z

## Result
VERDICT: PASS

This is an acceptance check on an artifact (the per-tick FULL dense-GRPO weight
snapshots in R2), not a hypothesis-vs-baseline comparison. All five acceptance
gates hold. The certified canonical R2 trace prefix that every downstream M4
weight-proj issue (#44 through #56) cites is:

  s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/

Key form: `tick_<N>/tick_<N>.pt` (one full bf16 model snapshot per optimizer
tick). Per-tick is a superset of per-step: subsample the FIRST tick of each
`global_step` (tick 0, 2, 4, ...) to recover the 80-point per-step movement
trajectory.

## Success criteria
- [x] (FULL WEIGHTS, headline) 160 objects under the R2 full/ prefix, all r2_manifest rows verified; first sampled object loads to a state dict of n_matrices=338 real weight tensors with true shapes, NOT a sketch and NOT a 196-subset (observed: full_manifest 160 rows; first row n_matrices=338, dump_dtype=bf16; box-side aws s3 ls --recursive of the full/ prefix ~160)
- [x] (instrument) first full_manifest.jsonl row reports n_matrices in [330,346] and dump_dtype=bf16 with both global_step and tick present (observed: n_matrices=338, dump_dtype=bf16, tick=0, global_step=1)
- [x] (trajectory) cell reached >= 80 training steps AND done.flag present (observed: max logged step=80; done.flag = "2026-06-30T12:27:02Z done regimeA rc=1"; last full_manifest row tick=159 gs=80)
- [x] (numerics) no NaN/Inf in any loss field across all logged steps (observed: grep -ciE for nan|inf in internal log = 0)
- [x] (dump integrity) every r2_manifest.jsonl row is verified:true AND verify_full_weight_dump.py --r2 --r2-sample 5 reports PASS (observed: 160/160 rows verified:true; box-side verify PASS, 5/5 sampled, max_rel_norm_err=0.0001 <= 0.01 tol)
- [x] (codec-OFF) comm_eff counters all zero (observed: powersgd_applications=0.0, mask_applications=0.0, anchor_backwards=0.0, spectral_corrections=0.0 on ALL 80 logged steps; master switch comm_eff.enabled=false)
- [x] (R2 hygiene) local regimeA/weights/full/ near-empty; df never approached full (observed: no local full/ dir present, count 0 — upload-then-delete worked; box-side staging stayed near-empty)
- [x] (manifests synced) MacBook has full_manifest.jsonl (~160 rows) + r2_manifest.jsonl (~160 verified) (observed: full_manifest 160 rows, r2_manifest 160 rows / 160 verified:true, both at runs/EXP-43/regimeA/weights/)
- [x] (WandB) backfilled to final step (observed: run a51waqza, RELOG DONE to step 80, resume="must")
- [x] (teardown) box TORN_DOWN in runs.jsonl and confirmed gone (observed: ledger row id=EXP-43 instance=43197578 status=TORN_DOWN; operator confirmed gone via vastai; team account, external box)

## Metrics summary
- full_manifest.jsonl rows: 160 (target ~160 = 2 ticks/step x 80 steps)
- first full_manifest row: n_matrices=338, dump_dtype=bf16, tick=0, global_step=1 (target n_matrices in [330,346], bf16)
- last full_manifest row: n_matrices=338, dump_dtype=bf16, tick=159, global_step=80
- r2_manifest verified rows: 160 / 160 (target = total rows, all verified:true)
- box-side R2 sample verify: PASS, 5/5, max_rel_norm_err=0.0001 (target <= 0.01)
- max training step logged: 80 (target >= 80)
- NaN/Inf in loss fields: 0 (target 0)
- comm_eff counters (every step): powersgd_applications=0.0, mask_applications=0.0, anchor_backwards=0.0, spectral_corrections=0.0 (target all 0)
- local full/ staging count: 0 (target ~0, upload-then-delete)
- final val-core/openai/gsm8k/acc/mean@1: 0.7809 (provenance only; no acceptance threshold — this is a dense control trajectory)

## Comparisons to baseline_run: none
baseline_run is `none` by design. EXP-43 PRODUCES the shared dense full-weight
trajectory that downstream issues read; there is no comparison arm. The dense
codec-OFF GRPO control reached val acc 0.7809 at step 80, consistent with the
known dense regime-A band (0.75 to 0.78), so the trajectory is a healthy dense
control and not a collapsed or diverged run.

## Resolved parameters (ground truth)
Source: `resolved_params.txt` + `resolved_cmd.txt`, extracted from the verbatim
`set -x` `main_ppo` trace at runs/EXP-43/regimeA/train_regimeA_internal.log line
35 (Hydra last-wins over 125 distinct keys), NOT from the plan. `capture_resolved_config.py`
expects `train.log` at the run-dir root and the EXP-43 trace lives under
`regimeA/train_regimeA_internal.log`, so the extraction was done directly off
that trace; the resolved files were written into runs/EXP-43/.

Comm-eff + weight-traj headline knobs (last-wins, verbatim):
- actor_rollout_ref.actor.comm_eff.enabled=false        (MASTER switch — codec OFF)
- actor_rollout_ref.actor.comm_eff.probe.weight_traj.enabled=true
- actor_rollout_ref.actor.comm_eff.probe.weight_traj.per_tick=true
- actor_rollout_ref.actor.comm_eff.probe.weight_traj.dump_dtype=bf16
- actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_enabled=true
- actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_delete_local=true
- actor_rollout_ref.actor.comm_eff.probe.weight_traj.out_dir=/workspace/runs/EXP-43/regimeA/weights

Fixed control surface (matches the plan and the project control variables):
- actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct
- data.train_files=/root/data/gsm8k/train.parquet ; data.train_batch_size=128
- actor_rollout_ref.actor.ppo_mini_batch_size=64  (=> 2 optimizer ticks/step => 160 ticks / 80 steps)
- data.max_response_length=1024 ; actor_rollout_ref.rollout.n=8
- actor_rollout_ref.actor.optim.lr=1e-6
- actor_rollout_ref.actor.use_kl_loss=False ; actor_rollout_ref.actor.entropy_coeff=0  (vanilla GRPO, no-KL no-entropy)
- trainer.total_training_steps=80 ; trainer.total_epochs=2 ; trainer.test_freq=40 ; trainer.val_before_train=False
- actor_rollout_ref.actor.use_dynamic_bsz=True
- trainer.n_gpus_per_node=1 ; actor_rollout_ref.rollout.tensor_model_parallel_size=1
- trainer.experiment_name=exp42-regimeA-exp43-full

DIVERGENCE NOTE (benign, expected): the resolved trace also carries
`actor_rollout_ref.actor.comm_eff.anchor.enabled=true` and
`actor_rollout_ref.actor.comm_eff.spectral.enabled=true` as sub-module flags.
These are INERT because the master `comm_eff.enabled=false` gates the entire
codec path. The proof is in the counters: powersgd_applications / mask_applications
/ anchor_backwards / spectral_corrections are all 0.0 on every one of the 80
logged steps, so no codec operation ever fired. This is the known "bare-exported
sub-cadence echoed in the resolved command does not reflect the master gate"
pattern (anchor/spectral sub-flags default to true in the launcher but never
activate under codec-OFF). Gate 4 is satisfied on the observed counters, which
are the authoritative codec-OFF evidence, not the sub-module enable flags. No
real divergence between intended (codec-OFF dense regime A) and what ran.

## Notes
- Canonical trace for downstream M4 issues (#44-#56): `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/`, key form `tick_<N>/tick_<N>.pt`, 160 bf16 full-model snapshots (n_matrices=338), one per optimizer tick. Per-step trajectory = first tick of each global_step (ticks 0,2,4,...,158 -> steps 1..80).
- The heavy .pt live in R2 ONLY (~492 GB); do NOT pull them to the laptop. Local artifacts are the two small manifests + the internal log. The box-side integrity verify (--r2-sample 5, PASS, max_rel_norm_err=0.0001) was run on the box at close-out against the same script and the same R2 data; an independent laptop re-verify would download ~15 GB and was not necessary to certify PASS. Downstream analyses that difference consecutive snapshots must account for bf16 ~4e-3 relative error (or commission a future fp32 run, dump_dtype=fp32, ~984 GB).
- done.flag reads rc=1: this is the documented benign atexit DataLoader-SIGKILL after step 80 + final validation, NOT a failure. No Ray/FSDP traceback, no CUDA OOM, no NaN/Inf in any loss field. Judged done by step-80 reached + 160 R2 objects + done.flag, per the plan's "gate on trajectory not launcher rc" instruction.
- resolved_params.txt / resolved_cmd.txt written into runs/EXP-43/ as reproducibility provenance (the real launched command, not the plan table). RESOLVED_CONFIG_MISSING was logged to PROGRESS.md because the standard script could not auto-locate train.log at the run-dir root; the params were recovered manually from the set -x trace and the run IS reproducible.
- final val acc 0.7809 is recorded as provenance only; this collection unit has no accuracy acceptance threshold.
