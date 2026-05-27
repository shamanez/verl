# Research Status — 2026-05-28T03:12:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 6 | M2 mask contamination guard (path-tag invariants) | RUNNING | 1×4H200 (i_38107546) | — | $15.05/hr; mask_on cell loading weights → val+2steps+ckpt, then mask_off. codex FAIL bypassed (false positive on gpu_count, human-authorized) |
| 7 | M2 spectral correction filter | NO_PLAN | — | — | no plan/status — triage owes (not research:claim) |
| 8 | M2 anchor circuit | NO_PLAN | — | — | triage owes |
| 9 | M2 full M95+AP two-step smoke | NO_PLAN | — | — | triage owes |
| 10 | M3 DP gradient compression scope | NO_PLAN | — | — | triage owes |
| 11 | M3 100-step M95+AP vs dense | NO_PLAN | — | — | triage owes |
| 5 | M2 actor-mask hook | DONE | — | PASS | milestone:M2, PR #2 merged (runs/SUMMARY.md) |
| 4 | M2 comm_eff no-op scaffolding | DONE | — | PASS | PR #1 merged (runs/SUMMARY.md) |
| baseline | M1 dense GRPO | DONE | — | PASS | val 0.087→0.789 |

## Last tick
2026-05-28T03:12:00+10:00 · verify=[6→bypassed] · running=[6] · analyzing=[] · logging=[] · blocked=[7,8,9,10,11 no-plan]

## Budget
$/hr now: $15.05 (i_38107546, 4×H200, pre-existing box) · run cap: max_gpu_hr=8 (≈2h wall on 4 GPUs) · MUST teardown when smoke done

## Notes
- Carryover: launcher `done.flag` bug — runner applied the on-box workaround + comm_eff Hydra override fix (actor_rollout_ref.actor.comm_eff.*) on exp/6.
