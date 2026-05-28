# Research Status — 2026-05-28T14:10:00+10:00 (EXP-12 RUNNING + monitor active)

## EXP-8 lineage cycle 2 of 3 — EXP-12 launched

- **EXP-12 run**: instance `38184281` 4×H200 @ $15.00/hr (chosen_tier_idx=5; fell back from one consumer attempt — `i_38182871` 8×RTX_5090 hit `env-failure: capacity-unavailable` per plan §Debug workflow table → TORN_DOWN, no second consumer re-roll per plan rule)
- **Branch**: `exp/12-anchor-detach` pushed to origin @ `1708b3e0` (clone-no-hook anchor + criterion-13 regression test) + `1de1d2c4` (YAML schema alignment for `ema_device`/`svd_mode`/`basis_cache`/`cadence`/`delay_K`). All 56 CPU unit tests PASS including the new `test_fsdp_anchor_backward_no_collision` (fails on EXP-8 code path, passes on EXP-12 code path).
- **Codex-verify**: SKIPPED per plan non-negotiable #3 (pre-written `runs/EXP-12/verify/20260528T130000.md` `VERIFY: PASS`). State machine moves directly `NEEDS_VERIFY → VERIFIED → READY_TO_RUN`.
- **Monitor**: `training-log-monitor` (Opus, 30s cadence, 40min cap) dispatched in background. Will SSH-probe per-cell logs + nvidia-smi + WandB `comm_eff/anchor_*` and `actor/grad_norm` scalars; returns a structured report with `recommendation: <dispatch_analyst | teardown_only | continue_in_place_iteration>`.
- **SSH**: `ssh -i ~/.ssh/vast_ai -p 37927 root@35.130.230.5` · tmux `exp-12-35_130_230_5` · per-cell logs `/workspace/runs/EXP-12/train_<cell>.log` · per-cell done flags `/workspace/runs/EXP-12/done_<cell>.flag` + aggregate `done.flag`.

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 12 | REVISE child of EXP-8 — anchor backward graph isolation | **RUNNING** | 1×4H200 (i_38184281 $15.00/hr) | — | clone-no-hook fix + criterion-13 regression test; 56 CPU tests PASS; codex-verify SKIPPED per override |
| 8 | M2 anchor circuit (K-stale unmasked GRPO refresh) | REVISE → child #12 | TORN_DOWN | REVISE | cell-2 PASS reproduces EXP-7; cells 1+3 FSDP autograd-hook collision; lineage 1/3 |
| 11 | M3 100-step M95+AP vs dense | NO_STATUS | — | — | not approved; skip |
| 10 | M3 DP gradient compression scope | NO_STATUS | — | — | not approved; skip |
| 9 | M2 full M95+AP two-step smoke | NO_STATUS | — | — | depends on EXP-8/12 PASS; skip |
| 7 | M2 spectral correction + FSDP grad-point | DONE | — | PASS | PR #4 merged |
| 6 | M2 mask contamination guard | DONE | — | PASS | PR #3 merged |
| 5 | M2 actor-only mask smoke | DONE | — | PASS | PR #2 merged |
| 3 | M1 dense GRPO baseline | DONE | — | PASS | milestone:M1 |

## Last tick
2026-05-28T14:10:00+10:00 · verify=[skipped per override] · running=[12] · analyzing=[] · logging=[] · blocked=[9,10,11 not-approved]

## Budget
Live instance: `38184281` 4×H200 @ $15.00/hr. EXP-12 budget cap: $25 (per goal-string non-negotiable #4). Consumer-tier 0 churn was free (instance went TORN_DOWN before SSH became routable). Cumulative lifetime spend: ~$17.36 + EXP-12 active.
