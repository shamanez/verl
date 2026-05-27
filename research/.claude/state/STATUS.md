# Research Status — 2026-05-28T00:33:26+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 4 | M2 — comm_eff no-op scaffolding: dense GRPO parity smoke | DONE | 1×4H200 (i_38088784) torn down | PASS (no-op proven; B/C deferred) | Scaffolding MERGED to vast-ai-workload via shamanez/verl#1 (merge 1f376733, tip b036c656). Run A: global_step=2, comm_eff counters all 0.0, grad_norm 3.07e-4, no NaN/Inf; unit tests 10/10. Issue #4 closed status:done. exp/4-commeff-noop branch deleted (local+remote). Instance torn down 00:32. **Follow-up:** Runs B(config-default)+C(reference) NOT run — launcher `vast_baseline_...sh:196` hardcoded done.flag path fails under SAVE_FREQ=-1 → chain aborted under set -e; A-vs-B parity + rel-tol 1e-4 pending a patched relaunch |
| 3 | M1 — Qwen2.5-1.5B GRPO baseline (GSM8K) | DONE | 1×4H200 torn down | PASS | val 0.087→0.789 (+0.702) by step 100; HF step50+step100 (private); `runs/EXP-3/REPRODUCIBILITY.md` |
| 5–11 | M2/M3 backlog (PRF mask, contamination guard, spectral filter, anchor circuit, full M95+AP smoke, DP compression, 100-step) | BACKLOG | — | — | open, no `status:` label, no `research:claim`, no plan — owned by triage/planning loop; orchestrator skips |

## Last tick
2026-05-28T00:33:26+10:00 · verify=[] · running=[] · analyzing=[] · logging=[] · blocked=[] · shipped+closed=[4] (PR #1 merged, branch deleted, instance torn down)

## Budget
$/hr now: **$0.00** (no instances running) · EXP-4 total: ~$6–7 (i_38088784, 4×H200 @ $14.74/hr, ~27 min provision→teardown) · EXP-3 historical: ~$12.76 instance / ~$10.29 training · monthly cap: tracked separately

## Open follow-up (not an issue yet)
- **Launcher done.flag bug** in `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh:196`: `touch /workspace/verl/runs/qwen25_1p5b_grpo_gsm8k_baseline/done.flag` hardcodes the default experiment name and a dir that doesn't exist under `SAVE_FREQ=-1` → exits nonzero → aborts back-to-back smoke chains under `set -e`. Fix: use `$EXPERIMENT_NAME` + `mkdir -p` (or guard the touch). Needed before any multi-cell smoke (incl. the EXP-4 B/C relaunch). Also: `.claude/worktrees/` is not gitignored (minor hygiene).
