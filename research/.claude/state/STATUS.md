# Research Status — 2026-05-28T14:30:00+10:00 (EXP-12 DONE · PASS)

## EXP-8 lineage cycle 2 of 3 — EXP-12 completed

- **EXP-12 run**: instance `38184281` 4×H200 @ $15.00/hr (chosen_tier_idx=5; fell back from one consumer attempt — `i_38182871` 8×RTX_5090 hit `env-failure: capacity-unavailable` per plan §Debug workflow table → TORN_DOWN, no second consumer re-roll per plan rule)
- **Branch**: `exp/12-anchor-detach` @ HEAD `afd43319` (four hot-fix iterations: clone-no-hook anchor + criterion-13 regression test). All 56 CPU unit tests PASS.
- **Codex-verify**: SKIPPED per plan non-negotiable #3 (pre-written `runs/EXP-12/verify/20260528T130000.md` `VERIFY: PASS`).
- **Monitor**: `training-log-monitor` (Opus, 30s cadence) completed successfully. Per-cell logs confirmed: faithful cell exit=0 @ 2026-05-28T04:52:55+00:00; lean cell exit=0 @ 2026-05-28T05:02:10+00:00.
- **SSH**: `ssh -i ~/.ssh/vast_ai -p 37927 root@35.130.230.5` · tmux `exp-12-35_130_230_5` · per-cell logs `/workspace/runs/EXP-12/train_<cell>.log` · per-cell done flags `/workspace/runs/EXP-12/done_<cell>.flag` + aggregate `done.flag`.

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 12 | REVISE child of EXP-8 — anchor backward graph isolation | **DONE** | 1×4H200 (i_38184281 $15.00/hr) | **PASS** | four on-box iterations closed FSDP collision; both anchor cells reached step:10; all 6 guards held; PR #5 drafted |
| 8 | M2 anchor circuit (K-stale unmasked GRPO refresh) | REVISE → child #12 | TORN_DOWN | REVISE | cell-2 PASS reproduces EXP-7; cells 1+3 FSDP autograd-hook collision; lineage 1/3 |
| 11 | M3 100-step M95+AP vs dense | NO_STATUS | — | — | not approved; skip |
| 10 | M3 DP gradient compression scope | NO_STATUS | — | — | not approved; skip |
| 9 | M2 full M95+AP two-step smoke | NO_STATUS | — | — | depends on EXP-8/12 PASS; skip |
| 7 | M2 spectral correction + FSDP grad-point | DONE | — | PASS | PR #4 merged |
| 6 | M2 mask contamination guard | DONE | — | PASS | PR #3 merged |
| 5 | M2 actor-only mask smoke | DONE | — | PASS | PR #2 merged |
| 3 | M1 dense GRPO baseline | DONE | — | PASS | milestone:M1 |

## Last tick
2026-05-28T14:30:00+10:00 · EXP-12 DONE PASS · PR #5 drafted to shamanez/verl base=vast-ai-workload

## Budget
Live instance: TORN_DOWN. EXP-12 total spend: ~$17.76 (4×H200 @ $15.00/hr ≈ 1h 10m wall). Cumulative lifetime spend: ~$35.12.
