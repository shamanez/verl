# Research Status — 2026-06-11T04:25:00+00:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 27 | EXP-26.1 REVISE child — damped ef (clip 0.5, decay 0.5) → 100 steps | **DONE** — terminal STOP; issue #27 open at status:stop, awaiting operator close | 1×4H200 (i_40493729, COMPLETE+held_warm; 46.243.55.155:40569, dph $13.49/hr; OUTSIDE auto-teardown — operator decision pending) | **STOP** | Cell killed at step ~66-68 on confirmed LENGTH_EXPLOSION rescue trigger (resp_len/mean 557.6 > 509 alarm, max pinned 16384 steps 61-68, entropy 0.34→0.079, max_mem 123/143 GB OOM-imminent). score 0.73-0.84 = length-hack, not reward collapse. val@25=0.7134, best val@50=0.7202 <= 0.7210 falsify floor. Damping capped EF dose (rel_change 0.02-0.19 vs parent 0.30-0.47) but only DELAYED ignition ~20 steps, zero val gain. STOP on BOTH predicate clauses. ef_powersgd lineage terminates (revise cycle 2 of 3); EXP-26's REVISE findings stand as M6 record. WandB qa6sll3h. |
| 26 | Geometry audit + ef_powersgd (#25 follow-up) | DONE — terminal REVISE verdict; issue #26 closed status:done | 1×4H200 (i_40242796, TORN_DOWN, destroy verified) | **REVISE** (M6 record) | Best realistic comm-eff = ef_powersgd 0.9/1.0 + act: val@50 **0.7210** (+7.7 over plain, cos(G_comp,G_corr)=0.9558, comm ratio 0.0506 ≈19.8×) but parity 0.7414 missed by 2.0 pts. B_plain 0.6437 < floor; Step C falsified (hybrid-Q anti-converts 0.3730). Code MERGED to vast-ai-workload (f93e3cc83). Dense ref = W&B 5e2jpho9 0.7536 (never re-run, operator directive). ef_powersgd STANDS as best M6 result. |

## Last tick
2026-06-11T04:25:00+00:00 · running=[] · EXP-27 DONE terminal STOP · box i_40493729 HELD WARM (COMPLETE+held_warm, ~$13.49/hr until operator decision) · pipeline EMPTY

## Budget
$/hr now: ~$13.49 (1× warm idle box i_40493729, 4×H200; HELD per operator — GPUs already freed, cell killed) · EXP-27 consumed: cell ran ~66 steps of 100 target (~4h wall), killed on LENGTH_EXPLOSION · lifetime_spent_usd: $32.945 (check_budget.py) · monthly cap: $1500.

## Next action
Pipeline EMPTY. No issues in flight. The ef_powersgd lineage is TERMINATED (EXP-26 REVISE stands as M6 record; parity 0.7414 not reached). Optional un-tried knob: ef_clip→0.25 or decay-only, flagged LOW-PRIOR in the verdict (unlikely to clear the 2.0-pt gap; the length-explosion is not dose-driven). Next work requires a NEW operator-approved issue. Box i_40493729 accrues ~$13.49/hr until operator tears it down or registers a new RUNNING owner.
