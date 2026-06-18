# Progress

Historical tick-by-tick orchestration output has been de-bloated. The next phase
starts from the compact handoff:

- `research/runs/SUMMARY.md`
- `research/.claude/plans/SUMMARY.md`
- `research/runs/FIXED_CONTROL_SURFACE.md`

Current working state:

- The **accelerated comm-eff loop** is the locked base: `signed_ema(α=0.25, β_anc=0.50)`,
  accel surface @ gpu_mem 0.55, `diagnostics=false`, PowerSGD r=77 anchor substrate.
  Launcher: `examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`.
- ~25 min train / ~28 min wall per 50-step run. Reference val@50 (n=1): comm-eff 0.7362
  (EXP-36B) vs dense 0.7657 (EXP-36C).
- The vLLM speed knobs (gpu_mem 0.75 / chunked_prefill / forward_prefetch) were tried and
  dropped — no speedup, added noise.
- Prior bulky run directories and execution plans are removed.
[2026-06-18T21:33:50+10:00] [research-planner #37] plan written
[2026-06-18T21:34:12+10:00] [triage] dispatched 1 planner (#37), skipped #35 (status:done)
[2026-06-18T21:43:33+10:00] [orchestrator #37] EXP-37 launched DIRECTLY on operator-provisioned team box i_41475643 (4×H200, $12.17/hr); cadence/delay 20/20 verified (trailing Hydra args win over bare-export 5); 100 steps, signed_ema(0.25,0.50). tmux exp-37-104_202_252_41. Dispatching background monitor.
EXP-37 exp-37-cad20-delay20 DONE 2026-06-18T12:40Z 100steps val@25=0.592 val@50=0.648 val@75=0.490 val@100=0.444 anchor_fires=10 stability=STOP(val@50<0.6862,late_len_excursion_steps93-100) wandb=fxo8chsv
[2026-06-18T22:48:22+10:00] [orchestrator] EXP-37 COMPLETE (100 steps, val@50 0.6482 < gate 0.6862, back-half length-escalation s93-100 → STOP-class; anchor_backwards=10 latency realized). Box i_41475643 REUSED for EXP-38 (EF analog: ef_powersgd 0.9/1.0, cadence/delay 20/20, 100 steps). Dispatched: EXP-38 monitor + EXP-37 analyst + 2-member delay-failure analysis team (staleness-theorist, cadence-analyst); report-lead pending.
[2026-06-18T12:49Z] EXP-38 MONITOR EXIT: ENV_FAILURE — box 41475643 (104.202.252.41:20177) unreachable from first contact (0% ping, ports 20177+22 closed, Vast API 0 instances). Training never started (0 steps, no WandB run). Recommendation: teardown_and_fallback (re-provision new box for EF run).
[2026-06-18T22:50:32+10:00] [analyst #37] verdict=STOP
[2026-06-18T22:54:44+10:00] [orchestrator] ROOT CAUSE: box i_41475643 destroyed 12:46Z by the vast-teardown SKILL invoked from CONCURRENT /loop session 2d9239b8 (PID 68399) — it reaped the finished EXP-37 box ~2min after THIS session reused it for EXP-38. EXP-38 never reached step 1 (no WandB run). Multi-session box-reuse race. Ledger: both 41475643 rows -> TORN_DOWN. EF re-run BLOCKED on operator decision (avoid re-losing a box to the same race).
[2026-06-18T23:01:39+10:00] [orchestrator] OPERATOR: switch off all H200s. VERIFIED 0 live instances on BOTH team + private accounts (i_41475643 already destroyed by concurrent session). No teardown needed; no money leaking. Ledger clean (no RUNNING/PROVISIONED rows). Delivered delay-failure HTML report.
