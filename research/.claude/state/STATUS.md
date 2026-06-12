# Research Status — 2026-06-12T23:10:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 30 | Generator-consistent M geometry gate + gated B1/B2 (valid-M merge re-test) | RUNNING (B2 launching) | 1×4H200 (i_40697545, operator box, do-NOT-teardown) | — | Step A DONE all-green (7 post-warmup fires, gates verbatim): GATE-B1 CLOSED (m1 0.0121 — validity doesn't move geometry), GATE-B2 OPEN (m5 1.0528, loss-mismatch 0.0103). B1 never launches; B2 delayed_ef λ=1 50-step cell launching on same box |
| 29 | Anchor on-policy replay (paired batch + CPU snapshots + fire-aware retention + relevance probe) | DONE | — (box destroyed) | PASS | PR #16 merged d26176b44; substrate donor for EXP-30 |
| 27 | Damped ef_powersgd merger | DONE | — | STOP | lineage closed |
| 26 | EF PowerSGD + Q families | DONE | — | REVISE | ef 0.7210 best realistic, M6 record stands |

## Last tick
2026-06-12T22:25:00+10:00 · running=[30] · analyzing=[] · logging=[] · blocked=[]

## Budget
$13.97/hr live (i_40697545, 4×H200, operator-provided) · EXP-30 cap 24 GPU-hr (≈6 box-hr) · Step A ETA ~25 min from 22:18
