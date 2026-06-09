# Research Status — 2026-06-10T02:45:37.623717+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 26 | Geometry audit + ef_powersgd (#25 follow-up) | OPERATOR GATE (STUCK) | 4×H200 (i_40242796, warm+idle) | — (H3 confirmed; H1/H2 blocked) | G_dense backward won't compose w/ codec-ON (grad-ckpt/loss-partial, design call). Runner recommends Option A (use G_fresh_anchor as dense ref). Awaiting operator design call + box decision. |

## Last tick
2026-06-09T22:38:52+10:00 · running=[26] · analyzing=[] · logging=[] · blocked=[]

## Budget
$/hr now: $12.84 (1×4H200) · max_gpu_hr cap: 60 · max_dph cap: $24
