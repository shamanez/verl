# Research Status — 2026-06-18T11:43+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 37 | signed_ema 100-step realism, anchor cadence/delay 20/20 | RUNNING | 1×4H200 (i_41475643, team, $12.17/hr) | — | Direct launch on operator-provisioned box. cadence/delay 20/20 verified in resolved cmd (trailing Hydra args win over bare-export 5). 100 steps, signed_ema(0.25,0.50). Background monitor active. |

## EXP-37 launch facts
- Box: team account, 4×H200, i_41475643, 104.202.252.41:20177, $12.17/hr (under $24 cap; ~4 gpu-hr expected, under max_gpu_hr 48).
- Launcher: vast_comm_eff_accel_base (LOG override → /workspace/runs/EXP-37/train.log). tmux exp-37-104_202_252_41.
- Verified resolved cmd: anchor.cadence=20, anchor.delay_K=20 (last-wins over env 5), correction_mode=signed_ema, beta_anc=0.50, signed_ema_alpha=0.25, owns_q=true, total_training_steps=100, val_before_train=False, test_freq=25, project=verl_compression_research_accel_rebaseline.
- Data prep OK (7473 train / 1319 test). No FATAL at init.
- **Gates** (plan §Success criteria): headline val@50 ≥ 0.6862 (within 0.05 of EXP-36B 0.7362); stability — no ignition/collapse/cold-M across steps 50-100; latency — anchor_backwards == 10 (NOT 40).

## Base reference (the default loop)
`vast_comm_eff_accel_base_*.sh` — signed_ema(α=0.25, β_anc=0.50), accel surface @ gpu_mem 0.55, diagnostics=false, PowerSGD r=77 anchor. ~25 min train / 50 steps.

| arm | run | val@50 |
|---|---|---|
| dense control (comm-eff OFF) | EXP-36C | 0.7657 |
| comm-eff signed_ema(0.25, 0.50) | EXP-36B | 0.7362 |

Only the merger axis may vary; all else locked (`runs/FIXED_CONTROL_SURFACE.md`).

## Last tick
2026-06-18T11:43+10:00 · running=[37] · analyzing=[] · logging=[] · blocked=[]

## Budget
$/hr now: $12.17 (1 box, team account) · max_dph cap $24 · max_gpu_hr 48
