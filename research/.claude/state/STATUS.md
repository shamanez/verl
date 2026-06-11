# Research Status — 2026-06-11T12:01:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 27 | EXP-26.1 REVISE child — damped ef (clip 0.5, decay 0.5) → 100 steps | **RUNNING** (under monitor) | 1×4H200 (i_40493729, WARM operator box, 46.243.55.155:40569, dph $13.49) | — | Launched 2026-06-11T01:58Z on operator-provided warm box; **Vast provisioning BYPASSED per operator directive** (no vast-provision; existing instance reused). Single cell `exp27_B_ef_damped`. Pre-run banner GREEN: ef_decay=0.5 ef_clip=0.5 mode=ef_powersgd q_basis=act, locked substrate (powersgd r77 sync_basis, anchor owns_q cadence5/delayK5, clean0), capture ON (g_dense=false fresh_anchor=false → OOM-safe), expandable_segments + ema_device=cpu, val@25/50/75/100. tmux exp-27-46_243_55_155. Ledger RUNNING. Monitor (bg) a642afe2fde831a00. PASS bar: best val@{50,75,100} ≥ 0.7414 + no ignition + cos(G_comp,G_corr)≈0.95±0.05. |
| 26 | Geometry audit + ef_powersgd (#25 follow-up) | DONE — terminal verdict; issue closed | 1×4H200 (i_40242796, TORN_DOWN, destroy verified) | **REVISE** | Best realistic comm-eff = ef_powersgd 0.9/1.0 + act: val@50 **0.7210** (+7.7 over plain, cos(G_comp,G_corr)=0.9558, comm ratio 0.0506 ≈19.8×) but parity 0.7414 missed by 2.0 pts. B_plain 0.6437 < floor; Step C falsified (hybrid-Q anti-converts 0.3730). Follow-up = approved child #27 (now RUNNING). Code MERGED to vast-ai-workload (f93e3cc83). Dense ref = W&B 5e2jpho9 0.7536 (never re-run, operator directive). |

## Last tick
2026-06-11T13:10:00+10:00 · running=[27 @ step 35/100, HEALTHY, past ignition window] · val@25=0.7134 · watcher=[b76a4q80q bg, 5-min] · analyzing=[] · blocked=[]

Progress @ step 35: val@25=0.7134 (>floor 0.6914, <PASS 0.7414); score 0.71–0.75; response_length CONTRACTING 271→170 (NO ignition; cleared the parent's step 29–42 zone); entropy plateaued ~0.76; EF dose 0.04–0.09 (parent 0.47 — damping holds); OOM/NaN/TB=0; owns-Q held; comm 0.0506. Two Sonnet monitors (a642afe2fde831a00, af9f3f20434db58ad) retired after confirming health; replaced by the cheap watcher.

## Budget
$/hr now: ~$13.49 (1× warm operator box i_40493729, 4×H200; LEFT RUNNING — the active run needs it) · EXP-27 cap max_gpu_hr=30 (run ≈20–24 GPU-hr, under cap) · run ETA ~3–4 h remaining.

## Next action
Cheap laptop-side watcher `b76a4q80q` polls every 5 min and re-invokes the orchestrator on done.flag / tmux-dead / OOM / NaN / length-ignition. On done.flag → rsync remaining artifacts + dispatch `analyst` (PASS iff best val@{50,75,100} ≥ 0.7414 + no ignition + cos(G_comp,G_corr)≈0.95; else REVISE final-cycle / STOP per plan predicate). val@50 (first PASS-relevant) ETA ~05:50–07:05Z. Box stays RUNNING until verdict; teardown is operator-gated.
