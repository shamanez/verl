# Research Log (newest first)

## EXP-13 · 2026-05-28T18:55:00+00:00 · M3 · PASS (at-risk, under investigation)

Paper-scale comm-eff PP-RL demonstration — 58-step M90+AP GRPO on
TRAIN_BATCH=128, ROLLOUT_N=8, MAX_RESPONSE=16384 (iter1 OOM → iter2 PASS via
memory recipe).

- **hypothesis**: communication-baseline's PASS knobs (α=0.5, τ=0.01, p=0.9,
  β_anc=0.9, anchor cadence=5, delay_K=5, mask_recompute=true) scale to
  paper-scale rollouts without infrastructure regression; model learns on
  held-out val (TRAIN_BATCH=128 vs smoke 16); all 6 comm-eff guards hold;
  iter2 reaches step 58 (TOTAL_EPOCHS=1 dataset-epoch limit).
- **infrastructure result (✓)**: iter1 OOM (step 2, GPU 0 135 GiB / 140 GiB on
  actor MLP forward — anchor clone + dynamic-batch wedge exceeded envelope);
  iter2 PASS with memory fix (PPO_MAX_TOKEN_LEN_PER_GPU 36864→18432, vLLM
  gpu_mem_util 0.4→0.3, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True). All
  comm-eff counters scale linearly: mask_applications/train=2548,
  /old_logprob=2366; anchor_backwards=44; spectral_corrections=896; mask_ratio
  0.8999 ± 0.0001; memory plateau 125.44 GB (no leak).
- **learning result (✓ but at-risk)**: validation 0.0864 → 0.0925 → 0.1092
  across steps 0 / 25 / 50 (+26% relative gain). However grad_norm starts at
  1134 at step 1 and climbs to 1884 by step 56; entropy collapses 6.42 → 0.023;
  ppo_kl reaches 1.4 — all consistent with policy collapse driven by KL anchor
  removal AND/OR comm-eff variance amplification (IS variance under
  independent PRF masks for train/old_logprob, smaller mini-batch + halved
  wedge, possible spectral-conditioning issues).
- **investigation queued**: `notes/investigation-prompt-grad-norm.md` enumerates
  9 candidate root causes and a 4-test discriminating plan (T1 sanity-zero, T2
  baseline-batch on more GPUs, T3 mask_recompute ablation, T4 α and
  seed_anchor_cache ablation). KL stays off across all tests.
- **run dir**: `runs/EXP-13/`
- **verdict**: `runs/EXP-13/verdict.md`
- **PR merged**: `shamanez/verl#7` → `vast-ai-workload`

## communication-baseline · 2026-05-28T17:15:00+10:00 · M2 · PASS (reference)

The communication-efficient method's smoke-scale verification (formerly
EXP-9). Promoted to a permanent baseline reference alongside the dense
baseline so future comm-eff experiments can compare against it directly.

- **hypothesis**: mask_recompute extension on old-logprob recompute forward,
  combined with knob relaxation (α 0.3→0.5, τ 0.001→0.01, p 0.95→0.9), yields
  visible learning trend and sustained peak-reward sequences; all 12 comm_eff
  guards held; iter2 reaches global_step=20.
- **result**: iter1 REVISE (one-step spike, declining second half); iter2 PASS
  (sustained rising trend, mean(11-20)=0.125 vs mean(1-10)=0.0688; three 0.25
  peaks at steps 12/17/18 = 4× step 1; mask_applications=420 total,
  anchor_backwards=10, spectral_corrections=160, ||dM_anchor|| max=1.119); all
  infrastructure counters exact, GUARD 5/6 held, no KL/entropy, actor/grad_norm
  finite, 140 tests PASS / 10 skip / 2 pre-existing skip.
- **run dir**: `runs/communication-baseline/`
- **verdict**: `runs/communication-baseline/verdict-iter2.md` (PASS),
  `runs/communication-baseline/verdict.md` (iter1 REVISE)
- **PR merged**: `shamanez/verl#6` → `vast-ai-workload` (mask_recompute
  extension lives on `vast-ai-workload` as part of the comm-eff implementation)

## baseline · 2026-05-26 · M1 · PASS (dense reference)

Dense GRPO, Qwen2.5-1.5B-Instruct on GSM8K — verl unmodified (the control,
ran before any comm-eff code change was merged).

- **result**: `val/test_score` 0.0872 → 0.7892 over 100 steps on 4×H200
- **run dir**: `runs/baseline/`
- **plan**: `.claude/plans/baseline.md`
- **reproducibility**: `runs/baseline/REPRODUCIBILITY.md` (launcher SHA pinned)

---

**Older entries (EXP-4 / EXP-5 / EXP-6 / EXP-7 / EXP-8 / EXP-12)** were folded
into `runs/SUMMARY.md` during de-bloat. The full history is preserved in git
log + the merged PRs (#1, #2, #3, #4, #5).
