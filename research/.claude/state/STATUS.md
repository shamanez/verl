# Research Status — 2026-06-04T12:35+10:00 (orchestrator · EXP-23)

## Active
EXP-23 RUNNING — 1×4H200 (instance 39447338, $15.21/hr). Arms A1→A2→A3 back-to-back in tmux `exp-23-84_8_106_109`. monitor-23 active (background). Smoke gate PASSED (all 6 hard gates; no verl hotfix needed).

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 23 | Stale full-grad re-anchor for PowerSGD (delay_K=5 + inject/blend) | RUNNING | 1×4H200 (i_39447338) | — | smoke PASS. A1 launching ~12:31. launch.sh ARM-parse bug fixed + relaunched. |
| 20 | PowerSGD activation codec (parent) | DONE | — | PASS | A0=0.7415 (r=77), dense ceiling 0.7536 — EXP-23 refs (not re-run) |

## EXP-23 arm plan (one warm box, back-to-back — 4 GPUs can't split across 3 multi-GPU arms)
- **A1** no-refresh floor: anchor OFF, spectral OFF, clean_cadence=0, 36864 tok/gpu — `exp-23-A1-no-refresh`
- **A2** stale inject: anchor delay_K=5 cadence=5, spectral inject γ=1.0 cadence=5, 18432 + ema_device=cpu — `exp-23-A2-stale-inject`
- **A3** stale blend: as A2 but spectral blend η=0.5 — `exp-23-A3-stale-blend`
- HEADLINE (PASS): `max(val@50(A2),val@50(A3)) ≥ 0.7315 AND ≥ val@50(A1)+0.05`. Falsified if `≤ val@50(A1)+0.02`.
- code change f42b7f36 on `exp/23-stale-reanchor`: wired spectral.correction_mode/inject_gamma/blend_eta env vars (was silent reweight default).

## Parallel background research team — `exp23-stale-grad` (operator directive 2026-06-04)
Runs alongside training; the MANDATORY next-lever readout if A2 AND A3 both fail (plan §Falsification contingency).
- async-rl-scout — async/off-policy RL staleness-tolerance literature
- cited-reader — operator-cited papers (2602.03839, 2511.08567, 2601.04537, nrehiew OPD blog)
- grad-empirics — measured per-layer G↔M_anchor geometry + layer coverage from EXP-23 logs
- synthesizer (blocked by the 3) → `runs/EXP-23/stale_gradient_research/STALE_GRADIENT_ALTERNATIVES.md`

## Last tick
2026-06-04T12:35+10:00 · running=[23] · analyzing=[] · logging=[] · blocked=[]

## Budget
$/hr now: $15.21 (1×4H200) · EXP-23 caps: max_gpu_hr=48, max_dph=24 · est arms ≈ 24–30 GPU-hr (within cap)
