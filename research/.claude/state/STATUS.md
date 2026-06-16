# Research Status — 2026-06-16T23:05+10:00 (EXP-33 PAUSED — operator out of Vast credits)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 33 | β_anc EMA sweep {0,.25,.5,.75,1} on B2 delayed_ef | **PAUSED (2/5 done)** | operator 4×H200 (84.8.116.228) — DEAD (out of credits) | — | C0 val@50=**0.73844** PASS, C1 val@50=**0.73995** tie. **Resume FROM C2** (b0p50 died gs=4, no read), then C3/C4. Issue OPEN. Full resume → `.claude/plans/33.md` RESUME STATE |
| 32 | signed_ema α=0.5 on valid-M | DONE | operator (op-managed) | (closed status:done) | result 0.7271 < B2 0.7528 |
| 31 | anchor-usage 4-lever tournament | DONE | — | STOP | all-null for surpass; B2=SOTA |

## EXP-33 progress (PAUSED — resume in a new session)
DONE (banked in runs/EXP-33/{b0p00,b0p25}/ + WandB verl_compression_research_beta_sweep):
- ✅ C0 b0p00 β=0.00 → val@25=0.71418, **val@50=0.73844** (CONTROL PASS, B2 band [0.716,0.774])
- ✅ C1 b0p25 β=0.25 → val@25=0.71418, **val@50=0.73995** (TIE, +0.0015 within ±0.024)

TO RUN (resume FROM C2 — NOT C3):
- ⏳ C2 b0p50 β=0.50 → 55 steps (interrupted gs=4 by infra crash, NO valid read — re-run from scratch)
- ⏳ C3 b0p75 β=0.75 → 55 steps (not started)
- ⏳ C4 b1p00 β=1.00 → 30 steps (degenerate bracket; val@25 read)

Bar: hypothesis falsified iff any C2/C3 val@50 ≥ 0.7624 (=0.73844+0.024). C1 ties ⇒ freshness/free-averaging holding. Remaining ~17 GPU-hr.
Relaunch (per cell, fresh box): `PROJECT_NAME=verl_compression_research_beta_sweep EXPERIMENT_NAME=<b0p50|b0p75|b1p00> TOTAL_TRAINING_STEPS=<55|55|30> TEST_FREQ=25 bash examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh actor_rollout_ref.actor.comm_eff.spectral.beta_anc=<0.50|0.75|1.00> trainer.val_before_train=false`

## Last tick
2026-06-16T23:05+10:00 · running=[] · paused=[33 2/5 done, resume from C2] · analyzing=[] · logging=[] · blocked=[operator-out-of-credits]

## Budget
EXP-33: ~14 GPU-hr spent (C0+C1+partial C2); ~17 GPU-hr remaining for C2/C3/C4 (cap 96). Operator box dead (out of credits) — new box needed to resume.
