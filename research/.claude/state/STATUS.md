# Research Status — 2026-06-12T23:10:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 30 | Generator-consistent M geometry gate + gated B1/B2 (valid-M merge re-test) | **PASS** + EXT RUNNING | 1×4H200 (i_40697545, operator box; EXP-30-EXT RUNNING row) | **PASS** | B2 0.7528@50 emission-free (parity reached, dense−0.0008); PR #17 merged ca5f4b002; m1–m7 → #28. ext100 de-censoring run live (watcher armed); team exp30-pathforward writing PATH_FORWARD |
| 29 | Anchor on-policy replay (paired batch + CPU snapshots + fire-aware retention + relevance probe) | DONE | — (box destroyed) | PASS | PR #16 merged d26176b44; substrate donor for EXP-30 |
| 27 | Damped ef_powersgd merger | DONE | — | STOP | lineage closed |
| 26 | EF PowerSGD + Q families | DONE | — | REVISE | ef 0.7210 best realistic, M6 record stands |

## Last tick
2026-06-12T22:25:00+10:00 · running=[30] · analyzing=[] · logging=[] · blocked=[]

## Budget
$13.97/hr live (i_40697545, 4×H200, operator-provided) · EXP-30 cap 24 GPU-hr (≈6 box-hr) · Step A ETA ~25 min from 22:18
