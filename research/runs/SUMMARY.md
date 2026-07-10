# Runs summary — one row per issue

Concise durable index (the OFFLINE fallback, and a `close_cleanup.sh` guard). The
per-issue verdict SSOT is the **issue close comment**; the rich record is the
**published report page** (project.yaml `reports:` → https://com-eff-rlvr.pages.dev/runs/),
WandB, and git history. LOG.md is retired — its rows were folded in here 2026-07-08.

| id | date | verdict | headline | issue | PR |
|---|---|---|---|---|---|
| 64-middle-block-freeze-grpo | 2026-07-11 | PASS | C(block): GSM8K 0.985/0.948 (block carries ~all RL gain); Big-Math 0.618/0.553 (clean neg, ~half). L11-15 freeze, comm-eff OFF. Boxes 44365338->44376214 (1xH200 team)+disk-full recovery. WandB 64-middle-block-freeze-grpo | #64 | shamanez/verl#24 |
| 63-deepscaler-r1d-signed-ema-k20 | 2026-07-10 | PASS (directional/truncated) | signed_ema β_anc=0.50 vs dense: b50 AIME@100 0.2125 vs dense 0.254 (within band+noise); train-reward parity; operator-cut. WandB 63-deepscaler-r1d-signed-ema-k20; box 44208646 TORN_DOWN | #63 | shamanez/verl#22 |
| 62-rlvr-models-datasets | 2026-07-07 | PASS | RLVR models+datasets enablement: 8/8 tokenizers, 5/5 schemas, 25/25 reward-preflight (CPU money gate), 25/25 GPU smoke cells non-null + no NaN, dense parity byte-identical. 5 test models × 5 math datasets; 3 integrate-only 7B/8B (CPU load). | #62 | shamanez/verl#21 |
| EXP-61 | 2026-07-04 | PASS | [M4] Math-only projector ablation: best = adaptive_linear rolling_ls_k K=3 @ Δ*=1,h=1 (ratio 0.9888); deploy fixed damped_linear at freshest anchor; do-nothing optimal at op — confirms dataset-specific | #61 | — |
| EXP-56 | 2026-07-04 | PASS | [M4] MOAT rollup: conditional winner = fixed #47 damped-linear (GSM8K op 0.9396, pred_evr +0.1171) but DATASET-SPECIFIC (Big-Math collapses to do-nothing) — NOT a universal ANCHOR default; gate on trajectory coherence | #56 | — |
| EXP-60 | 2026-07-04 | PASS | [M4] Big-Math cross-dataset validation: completed negative — NO projector beats do-nothing (GSM8K winner λ*=0.0, ratio 1.0000; consec_delta_cos≈0.15 vs 0.86 GSM8K) | #60 | — |
| EXP-49 | 2026-07-04 | PASS | [M4] self-correcting ANCHOR projector: completed negative — armA rolling_ls_k K=5 (op 0.9351) beats fixed bar 0.9396 but sub-threshold (Δ −0.0045 < 0.01) → keep fixed #47 rule | #49 | — |
| EXP-48 | 2026-07-03 | PASS | [M4] fixed 2nd-order kill-gate: completed negative — +0.2086 worse than damped-linear (1.1482 vs 0.9396), h_safe −28 → curvature dropped | #48 | — |
| EXP-47 | 2026-07-03 | PASS | [M4] ANCHOR damped-linear lane: projection HELPS on GSM8K — OOS-damped 0.940 beats naive 1.158 + hold-stale; λ*=0.3, best_δ=5, h_safe 30 steps; R²=0.535, ρ=−0.75 | #47 | — |
| EXP-58 | 2026-07-03 | PASS | [M4] Big-Math 1000-step GRPO collection: 50/50 ckpts + 50/50 fp32 weights verified:true in R2, dry_restore@1000; method-OFF byte-identical | #58 | shamanez/verl#20 |
| EXP-45 | 2026-07-02 | PASS | [M4] MOAT scorecard CONTRACT GO: SELFTEST/EMIT/SCHEMA GO (26796 rows box+laptop); hold-stale identity exact; structure partition 338 exact | #45 | — |
| EXP-44 | 2026-07-01 | PASS | [M4] offline weight-projection sweep engine ACCEPTED 8/8: 15/15 families reconstruct (rel-err 0.0); prior bf16-floor STOP overturned (category error) → differenced floor + directedness (p≈1.05, R²≈0.99) | #44 | — |
| EXP-43 | 2026-06-30 | PASS | [M4] dense GRPO FULL bf16 weight collection: 160/160 ticks × 338 matrices in R2, verify 5/5 max_rel_norm_err 1e-4; canonical M4 spine (~492 GB, R2-only) | #43 | — |
| baseline | — | reference | Dense GRPO, Qwen2.5-1.5B-Instruct on GSM8K — verl unmodified (the permanent control) | — | — |
