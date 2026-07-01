# Research Status — 2026-07-02T01:38:28+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 58 | Big-Math 1000-step GRPO — fp32 weights + full ckpts→R2 (on-the-go) | RUNNING (probe gate) | 1×H200 **43387501** (port 40381); orig 43383926 TORN_DOWN | — | code_change exp/58-ckpt-r2 pushed; team acct $3.03/hr; OPERATOR: stay on 1×H200, on OOM shrink resp 4096→2048 (no ladder); probe-off arm live (WandB exp-58-probe-off, 0/2 steps, no NaN/OOM) |

## Reconciliation note (2026-07-02 01:37)
Original box 43383926 auto-reaped at 01:29:58 by teardown Stop hook (PROVISIONED>15min, never promoted during long probe). Runner re-provisioned identical replacement 43387501 (same host 145.241.107.153, port 40381), re-launched probe, promoted ledger→RUNNING at 01:35 to defeat the reaper. No money leak. Operator-confirmed SSH: `ssh -i ~/.ssh/vast_ai -p 40381 root@145.241.107.153`. Untracked team box 43308497 (2×RTX4090, label lfm2-tp-erfan) = teammate Erfan's — do NOT touch.

## Last tick
2026-07-02T01:38:28+10:00 · running=[58 probe-gate on 43387501] · analyzing=[] · logging=[] · blocked=[]

## Budget
EXP-58: max_gpu_hr=96, $3.03/hr on 1×H200. Team account. No ladder escalation (operator override).
