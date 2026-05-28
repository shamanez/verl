# Research Status — 2026-05-28T12:33:00+10:00 (EXP-8 closed REVISE → child #12 created → instance torn down)

## EXP-8 lineage cycle 1 of 3 complete

- **EXP-8 run**: instance `38170973` 4×H200 @ $15.00/hr · started 12:01:43 → torn down 12:32:50 AEST · ~31 min wall · ~$7.75 spend
- **VERDICT: REVISE** — `runs/EXP-8/verdict.md`. 1/13 success criteria PASS (cell 2 anchor-off reproduces EXP-7); cells 1+3 crashed at identical FSDP `_check_grad_to_accumulate` `'NoneType'.shape` call site — anchor's `loss.backward()` collides with the FSDP fast-path `_post_backward_hook`. Lean storage knobs (cpu/lowrank/seed=false) did NOT mitigate; defect is upstream of storage, in the autograd-hook chain itself.
- **Cell-2 evidence** (load-bearing science): `training/global_step:5`, `mask_applications=70`, `spectral_corrections=40`, all `anchor_*=0`, `actor/grad_norm` 50.10 → 99.84 → 85.07 → 57.73 → 125.52 (finite). EXP-7 reproduction confirmed; M2 mask + spectral stack is sound.
- **Child issue #12** created at `status:planned` (`kind:experiment`, `milestone:M2`). Plan at `.claude/plans/12.md` mirrors plan/8 with two `next_actions`: (a) `anchor_backward_graph_isolation: live_fsdp_module → cloned_no_hook_module`, (b) `anchor_optimizer_param_group: shared → disjoint`. Plus new criterion-13 regression test (FSDP1 toy-module anchor-backward no-collision).
- **Next tick**: orchestrator detects `NEEDS_VERIFY_REVISE` on issue #12 → dispatches `codex-bridge --mode=verify` on plans/12.md. On PASS, status flips to `status:approved` → next tick experiment-runner builds `exp/12-anchor-detach` from `origin/vast-ai-workload`, cherry-picks EXP-8's clean schema+storage+spectral commits, applies the clone-no-hook fix, re-runs the 3-cell 5-step smoke.

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 12 | REVISE child of EXP-8 — anchor backward graph isolation | PLAN_READY | — | — | child of #8; awaits codex-verify (NEEDS_VERIFY_REVISE) on next tick |
| 8 | M2 anchor circuit (K-stale unmasked GRPO refresh) | **REVISE** | 1×4H200 (i_38170973 TORN_DOWN) | REVISE | cell-2 PASS reproduces EXP-7; cells 1+3 FSDP anchor-backward collision; lineage 1/3 → child #12 |
| 11 | M3 100-step M95+AP vs dense | NO_STATUS | — | — | not approved; skip |
| 10 | M3 DP gradient compression scope | NO_STATUS | — | — | not approved; skip |
| 9 | M2 full M95+AP two-step smoke | NO_STATUS | — | — | depends on EXP-8 PASS; not approved; skip |
| 7 | M2 spectral correction + FSDP grad-point | DONE | — | PASS | PR #4 merged |
| 6 | M2 mask contamination guard | DONE | — | PASS | PR #3 merged |
| 5 | M2 actor-only mask smoke | DONE | — | PASS | PR #2 merged |
| 3 | M1 dense GRPO baseline | DONE | — | PASS | milestone:M1 |

## Last tick
2026-05-28T12:33:00+10:00 · verify=[] · running=[] · analyzing=[] · logging=[] · blocked=[9,10,11 not-approved] · revise_children=[12 awaiting NEEDS_VERIFY_REVISE on next tick]

## Budget
No live Vast instances. EXP-8 spend: ~$7.75 (31 min × $15.00/hr). Cumulative lifetime on this project: ~$17.36 / monthly cap $1500. EXP-12 projected: ~$5 (clone path may shorten cells 1+3 to clean step-5 completion).

## Operator note (lesson captured)
The session's load-bearing learning, saved to memory `active-vast-monitor.md`: training is FAST; never trust `done_<cell>.flag` alone (the chain-doesn't-abort wrapper writes it through silent Ray errors); always run a 30s-cadence monitor cross-checking per-cell logs + per-GPU nvidia-smi + WandB scalars; delegate continuous monitoring to a teammate subagent. Caught the FSDP `_check_grad_to_accumulate` AttributeError this cycle ONLY because the operator pointed it out from WandB; the active monitor we now run by default will catch it natively next time.
