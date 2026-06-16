# Research Status — 2026-06-17 (EXP-33 RESUMED on new box i_41194490)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 33 | β_anc EMA sweep {0,.25,.5,.75,1} on B2 delayed_ef | **RUNNING (resume, 2/5 banked)** | 1×4H200 (i_41194490, 46.243.55.155:40266) | — | C0/C1 done+banked (0.73844 / 0.73995). C2 (b0p50 β=0.50) training; C3 (b0p75) + C4 (b1p00) queued. Monitor a3f8c514 active (background). |
| 32 | signed_ema α=0.5 on valid-M | DONE | operator (op-managed) | (closed status:done) | result 0.7271 < B2 0.7528 |
| 31 | anchor-usage 4-lever tournament | DONE | — | STOP | all-null for surpass; B2=SOTA |

## EXP-33 cell ledger
DONE (banked in runs/EXP-33/{b0p00,b0p25}/ + WandB verl_compression_research_beta_sweep):
- ✅ C0 b0p00 β=0.00 → val@25=0.71418, **val@50=0.73844** (CONTROL PASS, B2 band [0.716,0.774])
- ✅ C1 b0p25 β=0.25 → val@25=0.71418, **val@50=0.73995** (TIE, +0.0015 within ±0.024)

RUNNING / QUEUED on i_41194490 (resume driver C2→C3→C4):
- 🔵 C2 b0p50 β=0.50 → 55 steps — TRAINING NOW (passthroughs verified: beta_anc=0.50, val_before_train=false)
- ⏳ C3 b0p75 β=0.75 → 55 steps (queued)
- ⏳ C4 b1p00 β=1.00 → 30 steps (degenerate bracket; val@25 read; prime early-kill candidate)

Bar: hypothesis falsified iff any C2/C3 val@50 ≥ 0.7624 (=0.73844+0.024). C1 ties ⇒ freshness/free-averaging holding so far.

## Box
i_41194490 · 4×H200 · 46.243.55.155:40266 (proxy ssh2.vast.ai:34490) · $11.3523/hr · **ON project Vast account → API-destroyable → tear down via vast-teardown skill on done** · verl@vast-ai-workload 03ca9c8 (comm_eff+launchers+config byte-identical to C0/C1 commit d61607c — verified empty diff) · tmux exp-33-46_243_55_155 · keys ~/.ssh/vast_ai + ~/.ssh/vast_ai_name both accepted.

## Early-kill policy (operator directive 2026-06-17 — do NOT wait out 55 steps on a divergence)
Monitor a3f8c514 instructed to ABORT-and-REPORT immediately on: length-hack ignition (P1 ≥2 consec cap-pins /
P2 len-mean slope>0 sustained / P3 len-mean>2× early / E1 len/max>4k @steps10-30), NaN/non-finite grad,
reward/val collapse (GRPO = no critic; watch reward mean + gsm8k val acc), or OOM. Also reports val@25 the instant
it lands with a "below C0's 0.71418 & falling?" verdict. On a monitor `early_kill_cell:<name>` rec → orchestrator
kills JUST that cell's training process tree on the box so the driver's run_cell advances to the next cell
(NOT a box teardown). The divergence/val@25 IS that β-point's recorded result. STOP whole sweep only if C0 ignites
(C0 done+banked → N/A here).

## Last tick
2026-06-17 · running=[33:C2] · analyzing=[] · logging=[] · blocked=[] · monitor=background (a3f8c514)

## Budget
$/hr now: $11.35 (one box) · resume-row max_gpu_hr cap: 36 · remaining ≈ 17 GPU-hr (2 full cells + C4@30)

## On completion (HARD REQUIREMENT — operator directive, MUST terminate the instance)
When C2/C3/C4 reach val@50 (or are cleanly early-killed at val@25) and metrics rsynced:
1. dispatch `analyst` over ALL 5 cells (C0/C1 logs already local) → verdict.md + β→accuracy curve;
2. **TEAR DOWN i_41194490 via the `vast-teardown` skill** (`vastai destroy 41194490` + flip ledger row TORN_DOWN).
   This box IS on the project VAST_API_KEY account (verified: `vastai show instances` → 41194490 running) so it
   IS API-destroyable. Teardown is MANDATORY. Backstop: teardown-finished-runs Stop-hook reaps on verdict /
   heartbeat-stale / budget(>36 GPU-hr).
