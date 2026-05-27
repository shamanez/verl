# Research Status — 2026-05-28 (post-cleanup)

## Pipeline
| id | milestone | state | verdict | notes |
|---|---|---|---|---|
| baseline | M1 | DONE | PASS | dense GRPO control, val 0.087→0.789; `runs/baseline/`, `.claude/plans/baseline.md` |
| EXP-4 | M2 | DONE | PASS | comm_eff no-op scaffolding, PR #1 merged (see `runs/SUMMARY.md`) |
| EXP-5 | M2 | DONE | PASS | actor-only PRF activation masking, PR #2 merged (see `runs/SUMMARY.md`) |

No instances running. Full story: `runs/SUMMARY.md`. Backlog #6–#11 await planning.

## Budget
$/hr now: $0.00 (all torn down).

## Notes
- Repo de-bloated: EXP-4/EXP-5 run dirs removed (incl. 67 MB git bundles), folded into `runs/SUMMARY.md`; skill-test ledger rows + stale vast-handles pruned.
- `runs/**/*.bundle` + `*.log` now gitignored (history not rewritten — bundles remain in past commits).
- vast ssh-login fixed: provision emits paste-ready `ssh_login` (`-i ~/.ssh/vast_ai -o StrictHostKeyChecking=accept-new …`); `experiment-runner.md` examples corrected; "log in FIRST" surfaced on provision.
- Carryover: launcher `done.flag` bug still unpatched in `vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` (workaround applied on-box for EXP-5).
