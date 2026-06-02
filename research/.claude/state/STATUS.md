# Research Status — 2026-06-02 (EXP-18 / M4 curve-match cycle active)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 18 | M4 curve-match — continuous STALE-anchor gradient correction | RUNNING (refs launching) | provisioning (4×H200→8×H100 chain) | — | seq step 0 (candidates.md) DONE; dense ref + spectral floor launching on the search box |

## EXP-18 sequence progress (recursive search)

- [x] **step 0 — candidates.md** (MANDATE): 5 candidates enumerated (C1 inject, C2 complement-projection [spectral-derived], C3 b-estimator, C4 staleness-aggregation, C5 boundary-activation). Run order C1→C2→C3.
- [⏳] **step 1 — dense reference** (`curvematch_dense_ref_50step`, COMM_EFF_ENABLED=false, 50 steps) — launching on the box (the TARGET curve).
- [⏳] **step 1b — spectral floor** (`curvematch_spectral_baseline_c5_d5`, anchor c5/d5 + spectral as-implemented, 50 steps) — chained after dense (the inert-by-orthogonality FLOOR).
- [ ] **step 2 — candidate 1** (C1 `exp/18-anchorinject-c5d5`): add `spectral.correction_mode=inject` + `inject_gamma`; ADD scale-matched complement of stale M_anchor; max_targets=-1. Pending refs completion + box reuse.

## Harness prep done this tick

- `runs/EXP-18/candidates.md` — theoretical enumeration (PASS gate).
- `research/scripts/fetch_wandb_history.py` — WandB→per-step JSONL fetcher (verl logs only console+wandb; this materializes `runs/EXP-18/metrics/<name>.jsonl` for the curve-match, survives box teardown).

## Constraint pins (every candidate — a run violating these is INVALID, not REVISE)
- `COMM_EFF_ANCHOR_DELAY_K=5` (launcher default is 20 — MUST override), never 0/20.
- `COMM_EFF_CLEAN_CADENCE=0` (clean step OFF — correction stands alone).
- `COMM_EFF_ANCHOR_CADENCE=5`; mask confined to actor-train.

## Last tick
2026-06-02 · running=[18 refs] · analyzing=[] · logging=[] · blocked=[]

## Budget
gpu_filter_chain 4×H200 ($/hr ≤24 target) → 8×H100; max_gpu_hr=96 for the WHOLE search (box reused across all cells). ledger: 1 row (EXP-18) being registered by runner.
