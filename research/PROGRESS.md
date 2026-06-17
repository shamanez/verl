# Progress

Historical tick-by-tick orchestration output has been de-bloated. The next phase
starts from the compact handoff:

- `research/runs/SUMMARY.md`
- `research/.claude/plans/SUMMARY.md`
- `research/runs/FIXED_CONTROL_SURFACE.md`

Current working state:

- B2 `delayed_ef` is the comm-efficient reference.
- Beta `0.5` is only a nominal high draw; beta `0` remains the default.
- `signed_ema alpha=0.5` is a comparison candidate, not a promoted method.
- Prior bulky run directories and execution plans are removed.
[2026-06-17T17:08:24+10:00] [research-planner #34] plan written
[2026-06-17T17:08:49+10:00] [triage] dispatched 1 planners, 0 issues already planned
[2026-06-17T17:28:13+10:00] [orchestrator #34] LAUNCHED EXP-34 on operator team box 41292294 (4xH200, $12.69/hr) — 3 signed_ema cells {b0p25,b0p50,b0p75} back-to-back, tmux exp-34-104_202_252_41, val@25/50, vast_account=team. Cell 1 (beta_anc=0.25) started.
[2026-06-17T17:43:50+10:00] [orchestrator #34] attempt-1 hit known custom_all_reduce vLLM-IPC crash on all 3 cells (0 steps, env-failure). Applied break-glass DISABLE_CUSTOM_ALL_REDUCE=true (in-place, same box 41292294, no reprovision per operator). Attempt-2 cell 1 (b0p25) now PAST vLLM init — CUDA graphs captured, WandB connected, training started. Config verified: signed_ema/alpha=0.5/beta_anc=0.25/val_before_train=False/project correct. Re-dispatched monitor.
[2026-06-17T18:21:29+10:00] [orchestrator #34] attempt-2 HEALTHY: cell 1 (b0p25) global_step 14, reward 0.18→0.56 (steps6→14), resp_len ~237 stable (no ignition), bytes_ratio 0.0504, break-glass effective. Run is LONG: ~139-157s/step → ~2.5h/cell → ~7.5h total (~30 GPU-hr, ~$95, under 96 GPU-hr cap). Switched from heavy monitor subagent to lightweight background poll loop (per watch-logs-for-early-failures operator pref; ~zero tokens during wait). Each cell runs to val@50 (val@25 not safe to early-kill: EXP-33 showed +0.045 val@25→50 jumps). Teardown (team key, instance 41292294 only) when aggregate done.flag + metrics sync.
[2026-06-17T19:01:15+10:00] [orchestrator #34] CELL 1 val@25 = 0.7271 (val-core acc/mean@1) @ global_step 28 — ties the EXP-32 beta=0 reference (0.7278 val@25), as prior predicted. Reward 0.79, resp_len 152 (no ignition). LESSON: background bash has NO network egress (poll loop bdnwdf8yk returned 14 empty probes + did not refresh heartbeat → box went ~25min stale but survived). Only SUBAGENTS + foreground bash can SSH. Reverted to relaxed-cadence (180s) monitor subagent reusing poll.sh + refreshing incoming.log itself. Cell 1 → val@50 in ~65min.
