# Research Status — 2026-06-17 (EXP-33 RESUMED on new box)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 33 | β_anc EMA sweep {0,.25,.5,.75,1} on B2 delayed_ef | **RUNNING (resume 3/5)** | 1×4H200 (i_41194490, 46.243.55.155:40266) | — | C0/C1 done+banked (0.73844 / 0.73995). C2 (b0p50 β=0.50) training; C3 (b0p75) + C4 (b1p00) queued. Monitor active (background). |
| 32 | signed_ema α=0.5 on valid-M | DONE | operator (op-managed) | (closed status:done) | result 0.7271 < B2 0.7528 |
| 31 | anchor-usage 4-lever tournament | DONE | — | STOP | all-null for surpass; B2=SOTA |

## EXP-33 cell ledger

| cell | β_anc | steps | status | val@25 | val@50 |
|---|---|---|---|---|---|
| C0 b0p00 | 0.00 | 55 | DONE (banked) | 0.71418 | **0.73844** (control PASS, B2 band) |
| C1 b0p25 | 0.25 | 55 | DONE (banked) | 0.71418 | **0.73995** (tie w/ C0) |
| C2 b0p50 | 0.50 | 55 | **RUNNING (this box)** | — | — |
| C3 b0p75 | 0.75 | 55 | queued | — | — |
| C4 b1p00 | 1.00 | 30 | queued (degenerate bracket; val@25 read) | — | — |

THE BAR: C0 val@50 = 0.73844. Freshness-best hypothesis FALSIFIED iff any C2/C3 val@50 ≥ 0.7624. C1 ties ⇒ freshness/free-averaging holding so far.

## Box
i_41194490 · 4×H200 · 46.243.55.155:40266 (proxy ssh2.vast.ai:34490) · $11.3523/hr · **ON project Vast account → vastai-manageable → tear down via vast-teardown skill on done** · verl@vast-ai-workload 03ca9c8 (comm_eff+launchers+config byte-identical to C0/C1 commit d61607c, verified empty diff) · tmux exp-33-46_243_55_155
Passthroughs verified in set -x trace: `spectral.beta_anc=0.50` (last-wins over env 0.0) + `trainer.val_before_train=false` (last-wins over True).

## Last tick
2026-06-17 · running=[33:C2] · analyzing=[] · logging=[] · blocked=[] · monitor=background (a3f8c514)

## Budget
$/hr now: $11.35 (one box) · resume-row max_gpu_hr cap: 36 · remaining work ≈ 17 GPU-hr (2 full cells + C4@30)

## On completion (HARD REQUIREMENT — operator directive)
When C2/C3/C4 finish (C4 may be early-killed at val@25), dispatch `analyst` over ALL 5 cells (C0/C1 logs already local), write verdict, then **TEAR DOWN i_41194490 via the vast-teardown skill** — instance is on the project Vast account, destroyable via API.
