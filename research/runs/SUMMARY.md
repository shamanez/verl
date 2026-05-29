# Research runs — summary

Concise record of what has run on this harness. Heavy per-experiment artifacts
were pruned to keep the repo lean; the durable record is here + git history +
the merged code.

## Permanent references (full artifacts kept)

| id | milestone | what | result | dir |
|---|---|---|---|---|
| **baseline** | M1 | Dense GRPO, Qwen2.5-1.5B-Instruct on GSM8K — verl unmodified (the control, ran before any comm-eff code change) | `val/test_score` 0.0872 → 0.7892 over 100 steps, 4×H200 | `runs/baseline/` |
| **communication-baseline** | M2 | Comm-eff M90+AP smoke proof: PRF mask p=0.9, spectral α=0.5/τ=0.01/β_anc=0.9, anchor cadence=5/delay=5, mask_recompute=true, no KL, no entropy (formerly EXP-9 iter2) | PASS 20-step smoke; mean(11-20)=0.125 vs mean(1-10)=0.069 (+82%); all 12 comm-eff guards held; mask_applications/{train,old_logprob}=280/140; anchor_backwards=10; spectral_corrections=160; ||dM_anchor|| max=1.119 | `runs/communication-baseline/` |

## Active

| id | milestone | what | result | dir |
|---|---|---|---|---|
| EXP-13 | M3 | Paper-scale extension of communication-baseline (TRAIN_BATCH=128, ROLLOUT_N=8, MAX_RESPONSE=16384) | PASS at-risk: +26% val gain steps 0→50, but grad_norm starts at 1134 step 1 and entropy collapses 6.4→0.023 by step 58. Investigation queued at `notes/investigation-prompt-grad-norm.md`. | `runs/EXP-13/` |

## De-bloated (artifacts folded into this table)

| id | milestone | what | result | merged |
|---|---|---|---|---|
| EXP-12 | M2 | REVISE child of EXP-8 — anchor backward graph isolation (cloned-no-hook module / no_sync+summon_full_params) | PASS — four on-box hot-fix iterations closed FSDP autograd-hook collision; anchor_backwards=20 with all 6 guards held | PR #5 → `vast-ai-workload` |
| EXP-8 | M2 | M2 anchor circuit: same-process K-stale unmasked GRPO-actor-loss refresh | REVISE → closed by child EXP-12 | — |
| EXP-7 | M2 | Spectral correction filter (anchor-EMA → thin SVD → Tikhonov → two-sided projection → α-blend) + FSDP gradient-application-point discovery | PASS: FSDP1 full 2D Tensor via `use_orig_params`, correction AFTER FSDP reduction / BEFORE grad clipping; `spectral_corrections` fired, rel_change in (0,1], grad_norm finite | PR #4 → `vast-ai-workload` |
| EXP-6 | M2 | Mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS: per-path counters train=28/all-RL-paths=0; 35 unit tests including 1e-6 logprob equality + checkpoint guard; ckpt leak-scan clean | PR #3 → `vast-ai-workload` |
| EXP-5 | M2 | Actor-only PRF activation masking (in-graph `h*mask`, no rescale) | PASS: mask_ratio tracks p (p95→0.950, p90→0.900, ±0.02); confined to actor-train path; grads finite, no NaN/Inf | PR #2 → `vast-ai-workload` |
| EXP-4 | M2 | `comm_eff` no-op scaffolding: config group + disabled-by-default integration hooks | Run A no-op parity validated (`comm_eff.enabled=false` == dense) | PR #1 → `vast-ai-workload` |

## Implementation locus

The verl implementation lives on `vast-ai-workload`:
- `verl/workers/config/comm_eff.py` — Hydra config schema
- `verl/workers/comm_eff/{state.py, activation_mask.py, anchor.py, spectral_filter.py}` — runtime
- `verl/workers/engine_workers.py` — `compute_log_prob` mask_active stamp (mask_recompute wiring)
- `verl/workers/engine/fsdp/transformer_impl.py` — `_comm_eff_mask_active` gating
- `tests/workers/comm_eff/` — unit tests (140 PASS / 10 skip / 2 pre-existing skip)

## Conceptual notes

- `notes/anchor-memory-cost.md` — why the anchor clone takes ~3 GB
- `notes/fast-circuit-vs-anchor-pass.md` — which of the 5 GRPO forwards get masked
- `notes/investigation-prompt-grad-norm.md` — the next investigation issue draft

## Carryover follow-ups

- Launcher `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
  inherits a `done.flag` path that fails under `SAVE_FREQ=-1`, aborting
  multi-cell smoke chains under `set -e`. EXP-5/6 worked around it on-box; a
  real fix (`$EXPERIMENT_NAME` + `mkdir -p`) still belongs in the launcher.
- Plan templates grep for `val/test_score` but verl emits
  `val-core/openai/gsm8k/acc/mean@1` — update plan templates accordingly.
- `TOTAL_EPOCHS=1 × 7473 / 128 = 58 batches per epoch` is the M3 paper-scale
  step ceiling on the current dataset; use `TOTAL_EPOCHS=2` to reach the
  intended 100-step horizon.
