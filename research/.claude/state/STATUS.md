# Research Status — 2026-06-17 (EXP-34 RUNNING)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 34 | signed_ema α=0.5 β_anc sweep {0.25,0.50,0.75} | **RUNNING (2/3 cells)** | 1×4H200 (i_41292294, **team**) | — | **Cell 1 (β=0.25) DONE**: val s25=0.7271, **s50=0.7612**, s55=0.7384. val@50 nominally clears bar but val@50/val@55 spread (0.023≈noise) ⇒ NOISE-FRAGILE, likely tie. **Cell 2 (β=0.50) running** (step 1, clean). Cell 3 (β=0.75) pending. Monitoring: fg-probe + bg-sleep pacer (20min). Analyst weighs both late draws. Teardown (team key, 41292294 only) on aggregate done.flag |
| 33 | β_anc sweep on B2 delayed_ef | DONE | (torn down) | PASS | flat free-averaging; β=0 stays default; max gap C2 +0.0144 < 0.024 |
| 32 | signed_ema α=0.5 on valid-M | DONE | (op-managed) | done | val@50 0.7271 < B2 0.7528 |

## EXP-34 detail

- **Box**: operator-provided team-account instance **41292294** (4×H200, $12.69/hr). NOT harness-provisioned; do NOT reprovision. Both `~/.ssh/vast_ai` + `~/.ssh/vast_ai_name` reach it.
- **Account**: `vast_account=team` on ledger + handle → teardown auths with team key. ⚠️ Team account also holds unrelated box **41267389** (1×H100, NOT ours) — teardown must target 41292294 ONLY.
- **Cells** (back-to-back, one tmux): `signed_ema_b0p25` (β=0.25) → `signed_ema_b0p50` (β=0.50) → `signed_ema_b0p75` (β=0.75). signed_ema α=0.5 fixed; B2 substrate fixed; 55 steps; val@25/50 (`val_before_train=False`).
- **W&B**: project `verl_compression_research_beta_sweep_signed_ema`, entity shamanework-pl.
- **Heartbeat**: `runs/EXP-34/metrics/incoming.log` (sync-metrics tails box `/workspace/train.log`; both SSH keys verified to reach box).
- **Headline bar**: `best_cell_val@50 − 0.7271 > 0.024` (i.e. > 0.7511) ⇒ PASS. Reference points: EXP-32 signed_ema β=0 = **0.7271**; B2 delayed_ef SOTA = **0.7528** (context only). Prior (EXP-33 flat β curve + converged thesis) favors NULL/closure STOP.
- **Teardown**: the instant aggregate `done.flag` + all 3 cells' metrics sync to laptop → dispatch `vast-teardown` (team key, instance 41292294 ONLY). NO keep-warm (operator directive).

## Last tick
2026-06-17 · running=[34] · analyzing=[] · logging=[] · blocked=[] · monitor=bg(a2e8d17)

## Budget
$/hr now: 12.69 (1 box) · EXP-34 cap: 96 GPU-hr / $24-per-instance · expected ~12–18 GPU-hr
