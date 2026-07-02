# EXP-58 verdict

VERDICT: PASS

Big-Math 1000-step GRPO collection (Qwen2.5-1.5B-Instruct, dense = comm-eff OFF) on a
single 1×H200 (team, instance 43387501). The new config-gated checkpoint→R2 on-the-go
mirror hook is proven correct and the full artifact set is complete + verified in R2.

## Success criteria — ALL MET
- [x] probe gate GREEN — all 5 hard invariants: method-OFF byte-parity (probe-off log 0 `[ckpt_r2]` lines); on-the-go upload-then-delete (on-disk ≤ ~19 GB, disk ≤14%); resume completeness (verify PASS 2/2, tracker, dry_restore=True); drain barrier (`close() OK n_uploaded=22 n_errors=0`); FSDP1 NO_SHARD + 0 non-finite.
- [x] smallest rung, TEAM account (1×H200, `vast_account=team`, instance 43387501).
- [x] 1000/1000 steps @ MAX_RESPONSE_LENGTH=4096 on Big-Math (`data_source=DigitalLearningGmbH/MATH-lighteval`), strategy=fsdp (FSDP1), use_orig_params=true, **0 NaN/non-finite** in any loss field.
- [x] fp32 weights stream: **50/50** `verified:true` snapshots in `weights/full/step_{20..1000}.pt`, each ~6.17 GB (fp32; 0 undersized). Weights manifest: 50 rows, 50 verified:true, 50 distinct steps.
- [x] checkpoints stream: **50/50** `verified:true` `global_step_<N>/` trees — full `model_/optim_/extra_state_world_size_1_rank_0.pt` + `data.pt` + `actor/huggingface/{config,tokenizer}` + `actor/fsdp_config.json` per step; 0 zero-byte objects; fresh root `latest_checkpointed_iteration.txt`. Checkpoints manifest: 550 rows, 550 verified:true, 50 distinct steps.
- [x] on-the-go confirmed: objects accrued during training (per-save milestones while steps advanced); peak local disk bounded (~12–19 GB in-flight, delete-local), NOT keep-all ~900 GB.
- [x] dry restore: `find_latest_ckpt_path`-equivalent resolves step **1000** from the R2-mirrored tracker; all W=1 shard-triples present (`dry_restore=True`).
- [x] method-OFF sanity: byte-identical (probe-off).

## Evidence
- `research/scripts/verify_ckpt_r2_mirror.py … --expect-steps 20:1000:20 --require-tracker --emit dry-restore` → `PASS 50/50, tracker=1000, dry_restore=True`; independent recount unanimous (50 each of dirs/model_/optim_/extra_state/data.pt/fsdp_config/hf-config; 0 zero-byte).
- drain barrier: `[ckpt_r2] close() OK: n_uploaded=550 n_errors=0`.
- data: `s3://shamane-pluralis/verl-research/EXP-58/regimeA/{checkpoints,weights}`; manifests+logs → `runs/EXP-58/{manifests,collection}/`.
- science readout (NON-gating): critic/score (=Big-Math batch accuracy) rose 0.47→~0.61 (windowed) then plateaued; grad_norm ~0.24 flat, entropy 0.27→0.04 — stable GRPO. Success is artifact completeness, not reward.
- launcher `rc=1` is benign atexit teardown noise (wandb UnixTransport + DataLoader-worker-killed); NOT an R2/upload failure.

## Teardown
Box 43387501 torn down on completeness (team account, ledger TORN_DOWN, confirmed absent from account — no billing leak). WandB run 73zd1o6x backfilled to step 1000.
