# Research Status — 2026-05-28T03:25:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 6 | M2 mask contamination guard (path-tag invariants) | DONE | — | PASS | milestone:M2, PR to draft (exp/6-mask-invariants -> vast-ai-workload) |
| 7 | M2 spectral correction filter | NO_PLAN | — | — | no plan/status — triage owes (not research:claim) |
| 8 | M2 anchor circuit | NO_PLAN | — | — | triage owes |
| 9 | M2 full M95+AP two-step smoke | NO_PLAN | — | — | triage owes |
| 10 | M3 DP gradient compression scope | NO_PLAN | — | — | triage owes |
| 11 | M3 100-step M95+AP vs dense | NO_PLAN | — | — | triage owes |
| 5 | M2 actor-mask hook | DONE | — | PASS | milestone:M2, PR #2 merged (runs/SUMMARY.md) |
| 4 | M2 comm_eff no-op scaffolding | DONE | — | PASS | PR #1 merged (runs/SUMMARY.md) |
| baseline | M1 dense GRPO | DONE | — | PASS | val 0.087→0.789 |

## Last tick
2026-05-28T03:25:00+10:00 · logged=[6] · verdict=PASS · draft_pr=[6] · milestone=M2

## Budget
run 6 complete · $9.0842 spent, 0 running instances (within caps: 8 GPU-hr / max_dph 24.0 / 3 h wall)

## Notes
- EXP-6 verdict: PASS. Path-tag contamination guard (train|rollout|old_logprob|ref_logprob|val|infer|ckpt) + assert-on-wrong-path mask hook + per-path counters + checkpoint mask-free guard all established. 35 unit tests pass; live 2-step GRPO smoke with per-path counters train=28/all-RL-paths=0; val ran with parity within noise; checkpoint leak-scan clean.
- Draft PR opened: exp/6-mask-invariants -> vast-ai-workload (code_change=true)
- M2 milestone summary: EXP-4 (PASS), EXP-5 (PASS), EXP-6 (PASS) — ready for adversarial review via findings/M2/SUMMARY.md
