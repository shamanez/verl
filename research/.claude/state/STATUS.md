# Research Status — 2026-05-28T00:45:10+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| — | (no active experiments) | IDLE | none running | — | EXP-3 (M1 baseline) + EXP-4 (M2 comm_eff scaffolding) complete & merged. See `findings/`, `runs/EXP-3`+`runs/EXP-4`, and git history. |

## Backlog (await triage/planning — no `status:`/`research:claim` yet)
- #5 M2 PRF activation masking · #6 M2 mask contamination guard · #7 M2 spectral correction filter · #8 M2 anchor circuit · #9 M2 full M95+AP smoke
- #10 M3 DP gradient compression scope · #11 M3 100-step M95+AP vs dense baseline

## Last tick
2026-05-28T00:45:10+10:00 · verify=[] · running=[] · analyzing=[] · logging=[] · blocked=[] · IDLE — session reset for next issue

## Budget
$/hr now: **$0.00** (no instances running) · all Vast instances torn down · monthly cap: tracked separately

## Carryover follow-ups (not yet issues)
- **Launcher done.flag bug** — `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh:196` hardcodes a `done.flag` path (default exp-name) that doesn't exist under `SAVE_FREQ=-1` → exits nonzero → aborts back-to-back smoke chains under `set -e`. Fix: `$EXPERIMENT_NAME` + `mkdir -p`. Blocks any multi-cell smoke (incl. the EXP-4 B/C relaunch for A-vs-B parity + rel-tol-1e-4).
- `.claude/worktrees/` not gitignored (minor). · `research/researcher_steps.md` still references the deleted `major-goal/core-task.md` (CLAUDE.md refs were updated; researcher_steps.md left per scope).
