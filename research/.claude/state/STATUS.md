# Research Status — 2026-06-18T14:15:14+10:00 (EXP-35 RUNNING)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 35 | signed_ema α-sweep {0,.25,.5,.75,1} @ β_anc=0.50 (accel 4×H200: dyn-bsz + TP=1 + resp 2048) | **RUNNING** (monitor bg) | 1×4H200 (i_41420622, **team**) | — | reused operator box; 5-cell sequential sweep in one tmux (c3→c1→c2→c4→c5). c3 (α=0.5) active ~global_step 6/50, GPUs bursting 96–99%, no errors. |
| 34 | signed_ema α=0.5 β_anc sweep {0.25,0.50,0.75} | DONE | (torn down) | REVISE | β=0.50 peak val@50 0.7635 (clears +0.024 bar) but n=1 → replicate. EXP-35 follows up on the α axis at that β=0.50 peak. |
| 33 | β_anc sweep on B2 delayed_ef | DONE | (torn down) | PASS | flat free-averaging; β=0 default |

## EXP-35 detail

- **Box**: operator-provided warm **4×H200**, instance **41420622**, **team account**, direct SSH `-i ~/.ssh/vast_ai -p 40264 root@84.8.106.109 -L 8080:localhost:8080`. Reused / do-not-provision; do NOT teardown without operator order. dph ≈ $12.88/hr.
- **Driver**: `tmux exp-35-84_8_106_109` → `/workspace/runs/EXP-35/launch.sh` (5-cell sweep wrapper, pid 2144) → c3 launcher (pid 2150). Remaining cells c1/c2/c4/c5 auto-run back-to-back after c3 — box stays GPU-busy for the whole sweep.
- **Cells** (run order): `exp-35-c3-a050` (α=0.5, control/surface-validation gate, target val@50 ∈ [0.7395,0.7875]) → `exp-35-c1-a000` (α=0.0, sign-SGD endpoint, ignition risk) → `exp-35-c2-a025` (α=0.25) → `exp-35-c4-a075` (α=0.75) → `exp-35-c5-a100` (α=1.0, no-merger PowerSGD floor ~0.63).
- **Surface**: signed_ema correction, β_anc=0.50, B2 substrate (PowerSGD r77, anchor owns_q, replay_paired_batch, disable_custom_all_reduce). 50 steps/cell, test_freq=25 (val@25 + val@50), val_before_train=False.
- **W&B**: project `verl_compression_research_alpha_sweep_signed_ema`, entity shamanework-pl.
- **Monitor**: `training-log-monitor` dispatched in background (agent a41af464…), 30 s cadence; returns terminal report on cell-transition / done / stall / error. Heartbeat: `runs/EXP-35/metrics/incoming.log`.
- **Why no parallel job**: the 4-GPU FSDP run legitimately owns all 4 GPUs (bursts to 96–99%, free_cache_engine swings mem 13↔90GB); #35 is the only open issue. GPUs are driven by the existing sweep — nothing idle, nothing else approved to launch.

## Last tick
2026-06-18T14:15:14+10:00 · running=[35] · analyzing=[] · logging=[] · blocked=[] · monitor=bg(a41af464)

## Budget
$/hr now: $12.88 (1 box, team) · run started 14:05 (~10 min ago) · max_gpu_hr cap: 60
