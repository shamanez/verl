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
