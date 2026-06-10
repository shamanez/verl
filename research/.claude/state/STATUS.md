# Research Status — 2026-06-11T00:15:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 26 | Geometry audit + ef_powersgd (#25 follow-up) | DONE — terminal verdict | 1×4H200 (i_40242796, TORN_DOWN, destroy verified) | **REVISE** | Whole issue complete: A (gate) → C (falsified) → B (3 arms + r2) → E (measured). Best = ef_powersgd 0.9/1.0 + act: **val@50 0.7210** (best realistic comm-eff to date; +7.7 over plain, +1.4 over signed_ema; cos(G_comp,G_corr)=0.9558 direction-preserving; comm ratio 0.0506 ≈19.8×) but parity 0.7414 MISSED by 2.0 pts. New negatives: B_plain 0.6437 < floor 0.6914 (anchor-refresh-alone harmful, inverts EXP-25 "merger net-harmful"); Step C falsified (hybrid Q anti-converts 0.3730 — rollouts are uncompressed). r1 ignition alarm (length-explosion at clip=1.0, stochastic) carried into next_actions. Verdict + analysis posted on #26 (comments 4675017777/4675019072). Dense = W&B 5e2jpho9 per operator directive (never re-run). |
| 27 | EXP-26.1 REVISE child — damped ef (clip 0.5, decay 0.5) → 100 steps | PLAN_READY | — | — | **Awaiting operator approval**: `gh issue edit 27 --remove-label status:planned --add-label status:approved`. Plan at .claude/plans/27.md (+ posted on #27). Config-only, 1 cell, max_gpu_hr 30, branch exp/26-…@13a21c3e9 reused. |

## Last tick
2026-06-11T00:15:00+10:00 · terminal-verdict=[26 REVISE] · child-created=[27 planned] · running=[] · blocked=[27 awaiting human gate]

## Budget
$/hr now: $0 (no live instances — verified via `vastai show instances`) · EXP-26 box lifetime ≈ 24.6 h × $12.88 ≈ $316 · monthly lifetime $697.86 / cap $1500

## Next action
Operator reviews plan 27 and flips the label; the orchestrator loop then dispatches the runner (fresh provision, warm-box is gone). LOG.md entry for EXP-26 happens when the lineage terminates (PASS/STOP per playbook; REVISE keeps the lineage open).
