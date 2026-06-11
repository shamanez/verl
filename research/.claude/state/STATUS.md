# Research Status — 2026-06-11T15:25:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 28 | TRUE error-feedback on the codec's own dropped residual (carrier-free) @100 + plain@100 no-carrier control | **AWAITING TRIAGE** (`research:claim`) | — | — | Successor to #27, created 2026-06-11 by the exp27-postmortem team (Fable issue-author) from MECHANISM_ANALYSIS.md + RUN_COMPARISON.md, GOAL-criteria-mapped. Cells: A `codec_ef_100s` (the #24 frontier primitive, never actually built — sender-local e←(g+e)−C(g+e), telescoping, P(no-ignition)≈0.85), B `plain_100s` (H_carrier/H_generic discriminator, P≈0.85 clean), C conditional LABELED seq-mean guardrail (fires only on P1). All production cells capture OFF (diagnostics policy); E1+P1 watches standing. Next: triage → plan 28 → operator approval. Warm box reuse = operator's call. |
| 27 | EXP-26.1 damped ef (clip 0.5, decay 0.5) → 100 steps | **CLOSED** (completed, `status:done`) | 1×4H200 (i_40493729, **HELD WARM**, COMPLETE+held_warm, ~$13.49/hr, outside auto-teardown) | **STOP** + post-mortem | Killed @step ~66-68 on confirmed LENGTH_EXPLOSION; best val 0.7202 ≤ 0.7210 floor. 4-agent post-mortem complete: mechanism = persistent TANGENTIAL stale-EMA carrier (the implemented ef is NOT true error feedback; dose sets only the lag, near-linear 2.17×↔2.03×); carrier-specific (plain clean on identical substrate, single-knob isolate); entropy = follower ×3; α0.5 was censored-unstable (P(ignite@100)≈0.6). Docs: runs/EXP-27/{verdict.md+post-mortem, MECHANISM_ANALYSIS.md 760 lines, RUN_COMPARISON.md §§1-9}; HTML tab6; #27 comment 4677420446. Watch re-centered: E1 early gate (len/max>4k @10-30) + P1 kill (2 consecutive 16384 pins); diagnostics policy added to FIXED_CONTROL_SURFACE.md. |
| 26 | Geometry audit + ef_powersgd | CLOSED (status:done) | TORN_DOWN | **REVISE** (M6 record) | ef_powersgd 0.7210 stands as the M6 record (+7.7 over plain, ~19.8× comm); parity 0.7414 missed by 2.0. Dense ref = W&B 5e2jpho9 (never re-run). |

## Last tick
2026-06-11T15:25:00+10:00 · #27 CLOSED w/ post-mortem · #28 created `research:claim` (awaiting triage) · exp27-postmortem team (4 agents + 2 addenda) COMPLETE · running=[] · blocked=[]

## Budget
$/hr now: ~$13.49 (1× idle warm box i_40493729, HELD per operator — GPUs free; teardown or reuse-for-#28 = operator decision) · monthly cap $1500.

## Next action
1. Triage loop picks up #28 (`research:claim`) → research-planner writes `.claude/plans/28.md` → operator reviews + flips `status:planned → status:approved` → orchestrator executes (runner may REUSE warm box i_40493729 by registering a new RUNNING ledger row — operator decides at approval).
2. Operator: decide the held box (it holds nothing unique — all artifacts incl. 11 GB captures are local).
