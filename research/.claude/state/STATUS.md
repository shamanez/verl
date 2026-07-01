# Research Status — 2026-07-02T00:55:03+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 58 | Big-Math 1000-step GRPO — fp32 weights + full ckpts→R2 (on-the-go) | RUNNER_DISPATCHED | — (provisioning) | — | code_change: exp/58-ckpt-r2 (ckpt→R2 hook); team acct; ladder 1×H200 first; probe gates 1000-step collection |
| 57 | fp32 dense weight-traj→R2 | DONE | 1×… (43311909) | PASS | TORN_DOWN; feeds #45–#56 analysis |
| 43 | — | DONE | (43197578) | — | TORN_DOWN |

## Last tick
2026-07-02T00:55:03+10:00 · running=[] · provisioning=[58] · analyzing=[] · logging=[] · blocked=[]

## Budget
EXP-58 caps: max_gpu_hr=96 (probe+collection+re-probe), max_dph=$24/hr, per-rung ceilings 6/10/12/24. Team account.
