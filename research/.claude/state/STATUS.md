# Research Status — 2026-06-03 (EXP-18 / M4 curve-match — ACTIVE)

## Compute: 1 live box — instance 39132674, 4×H200 @ $15.79/hr (tier0). ~1.5 GPU-hr of 96 used.

## Issue pipeline
| EXP | Title | State | Vast | Verdict | Notes |
|---|---|---|---|---|---|
| 18 | M4 — continuous STALE-anchor gradient correction | RUNNING (candidate C1) | 39132674 (4×H200) | — | step0+dense+floor done; C1 launching on reused box |

## EXP-18 recursive search progress
- [x] **step 0 — candidates.md** (MANDATE): 5 candidates (C1 inject, C2 complement-proj [spectral-derived], C3 b-estimator, C4 staleness-agg, C5 boundary-act).
- [x] **step 1 — dense TARGET** cached (`metrics/curvematch_dense_ref_50step.jsonl`, 50/50): reward **0.135→0.868**, grad_norm 0.32–0.49, no NaN.
- [x] **step 1b — spectral FLOOR** cached (`metrics/curvematch_spectral_baseline_c5_d5.jsonl`, 50/50, rc=0): flat **mean 0.135** (0.111–0.164) = inert-by-orthogonality CONFIRMED. The "beat-this" baseline.
- [⏳] **step 2 — candidate C1** (`exp/18-anchorinject-c5d5`): stale-anchor additive injection (`correction_mode=inject`, `inject_gamma=1.0`, `max_targets=-1`). runner-c1 (bg) branching+patching+launching on reused box.

## Anchor-OOM fix (load-bearing, applies to every anchor-ON cell)
First floor run OOM'd in the anchor's unsharded full backward at the first anchor fire (36864 tok/gpu). FIX = launcher-documented `PPO_MAX_TOKEN_LEN_PER_GPU=18432` (no code, no response-len change). Validated end-to-end (floor ran 50 steps, anchor fired 40×, OOM=0). C1 inherits it.

## Constraint pins (every candidate — violation ⇒ INVALID, re-launch not analyze)
`COMM_EFF_ANCHOR_DELAY_K=5` (launcher default 20!) · `COMM_EFF_CLEAN_CADENCE=0` · `COMM_EFF_ANCHOR_CADENCE=5` · mask actor-train-only.

## Analysis infra (this cycle)
`scripts/fetch_wandb_history.py` (WandB→jsonl) · `scripts/parse_train_log.py` (log→jsonl fallback; dense WandB was state=crashed at 48/50 from chained-launch finalize gap) · `scripts/curve_match.py` (mean+final |Δreward| vs dense) · `scripts/cell_watch.sh` (reusable bg heartbeat/terminal watcher).

## Last tick
2026-06-03 · running=[18 C1] · analyzing=[] · logging=[] · blocked=[]

## Budget
$15.79/hr (1×4×H200) · max_gpu_hr 96 for whole search (box reused across all cells) · wall_clock_hr 12 soft.
