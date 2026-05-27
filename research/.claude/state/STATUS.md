# Research Status — 2026-05-28T00:09:15+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 4 | M2 — comm_eff no-op scaffolding: dense GRPO parity smoke | RUNNING | 1×4H200 (i_38088784, tier_idx=1, $14.74/hr) | — | launched 00:05, ~4 min elapsed; branch `exp/4-commeff-noop` pushed before provisioning; pre-launch local unit tests 10/10 PASS (`runs/EXP-4/verify.log`); 3 back-to-back smokes (disabled / default / reference) in tmux `exp-4-156-19-254-2`; plan gate operator-cleared, code-level diff verify still pending (later gate) |
| 3 | M1 — Qwen2.5-1.5B GRPO baseline (GSM8K) | DONE | 1×4H200 torn down | PASS | val 0.087→0.789 (+0.702, 14× threshold) by step 100; HF step50+step100 (private); `runs/EXP-3/REPRODUCIBILITY.md`. EXP-4 `depends_on:[EXP-3]` satisfied |
| 5–11 | M2/M3 backlog (PRF mask, contamination guard, spectral filter, anchor circuit, full M95+AP smoke, DP compression, 100-step) | BACKLOG | — | — | open, no `status:` label, no `research:claim`, no plan — not yet claimed for planning (triage owns); orchestrator skips |

## Last tick
2026-05-28T00:09:15+10:00 · verify=[] · running=[4] · analyzing=[] · logging=[] · blocked=[] · dispatched=[experiment-runner→EXP-4]

## Budget
$/hr now: $14.74 (EXP-4, 1×4H200) · spent on EXP-4 so far: ~$0.98 (~4 min) · EXP-4 hard caps: max_gpu_hr=8 (≈2.0 wall-clock h on 4 GPUs) / wall_clock_hr=3 / max_dph=24.0 → teardown hook caps EXP-4 at ≈$29 · EXP-3 historical: ~$12.76 instance / ~$10.29 training · monthly cap: tracked separately

## Notes
- EXP-4 plan-level codex verify was OPERATOR-CLEARED (raw codex: FAIL rev-1 / CONCERNS rev-2, judged out-of-scope for a disabled-scaffolding parity smoke; plan reverted to rev-1). Code-level verify of the `exp/4-commeff-noop` diff was NOT waived; honored cheaply pre-launch via a torch-free local unit-test gate (10/10 PASS, `runs/EXP-4/verify.log`) so no GPU was spent on unvetted scaffolding. Full codex code-level review of the diff remains a later gate (PR-draft time).
- Next tick: EXP-4 is `RUNNING` → no dispatch; sync-metrics hook pulls logs. On `runs/EXP-4/done.flag` (or dead tmux + `metrics/*.jsonl` present) → dispatch `analyst`. Benign analyst note: launcher's internal cgroup probe printed `pids.max=unknown` (cgroup-v2 host; real value 23552, recorded in PROGRESS) — the `<=2048` guard was correctly not tripped.
