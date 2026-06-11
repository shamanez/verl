# Research Status — 2026-06-11T12:01:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 27 | EXP-26.1 REVISE child — damped ef (clip 0.5, decay 0.5) → 100 steps | **RUNNING** (under monitor) | 1×4H200 (i_40493729, WARM operator box, 46.243.55.155:40569, dph $13.49) | — | Launched 2026-06-11T01:58Z on operator-provided warm box; **Vast provisioning BYPASSED per operator directive** (no vast-provision; existing instance reused). Single cell `exp27_B_ef_damped`. Pre-run banner GREEN: ef_decay=0.5 ef_clip=0.5 mode=ef_powersgd q_basis=act, locked substrate (powersgd r77 sync_basis, anchor owns_q cadence5/delayK5, clean0), capture ON (g_dense=false fresh_anchor=false → OOM-safe), expandable_segments + ema_device=cpu, val@25/50/75/100. tmux exp-27-46_243_55_155. Ledger RUNNING. Monitor (bg) a642afe2fde831a00. PASS bar: best val@{50,75,100} ≥ 0.7414 + no ignition + cos(G_comp,G_corr)≈0.95±0.05. |
| 26 | Geometry audit + ef_powersgd (#25 follow-up) | DONE — terminal verdict; issue closed | 1×4H200 (i_40242796, TORN_DOWN, destroy verified) | **REVISE** | Best realistic comm-eff = ef_powersgd 0.9/1.0 + act: val@50 **0.7210** (+7.7 over plain, cos(G_comp,G_corr)=0.9558, comm ratio 0.0506 ≈19.8×) but parity 0.7414 missed by 2.0 pts. B_plain 0.6437 < floor; Step C falsified (hybrid-Q anti-converts 0.3730). Follow-up = approved child #27 (now RUNNING). Code MERGED to vast-ai-workload (f93e3cc83). Dense ref = W&B 5e2jpho9 0.7536 (never re-run, operator directive). |

## Last tick
2026-06-11T12:01:00+10:00 · launched=[27 on warm box, provision bypassed] · running=[27] · monitor=[a642afe2fde831a00 bg] · analyzing=[] · blocked=[]

## Budget
$/hr now: ~$13.49 (1× warm operator box i_40493729, 4×H200; LEFT RUNNING — the active run needs it) · EXP-27 cap max_gpu_hr=30 (run ≈20–24 GPU-hr, under cap) · run ETA ~5–6 h wall.

## Next action
Background monitor returns a terminal report at ≤40 min (or on event: length-explosion / OOM / stall / done). On `continue_in_place_iteration` (healthy) re-watch until the cell's done.flag, then dispatch `analyst` (verdict per plan predicate). On length-ignition / `teardown_only`, the orchestrator dispatches vast-teardown. Run is ~5–6 h, so a verdict lands in a later tick/session — box stays RUNNING until then.
