# Research Status — 2026-06-04T02:28:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 20 | PowerSGD-style PP activation compression (M6) | RUNNING (monitor active) | 1×4×H200 (i_39319060, $15.21/hr) | — | runner built `powersgd` codec on `exp/20-powersgd-activation` (HEAD def451e5, pushed); launched seq probe→mask p95+clean5→powersgd r102+clean5→optional dense; CPU invariants 15/15 + cfg 18/18; on-box PROBE (FSDP/bf16/grad-ckpt + ρ≈1 hard gate) is the near-term signal; monitor-exp20 dispatched (bg) |
| 19 | M5 — surpass dense baseline (epic) | UNCLAIMED | — | — | labels `kind:experiment`+`milestone:M5` only; no `research:claim`/`status:*`/plan → awaiting triage, not orchestrator's concern this tick |
| 18 | M4 curve-match | TORN_DOWN | 1×4×H200 (gone) | — | torn down 2026-06-03, no-heartbeat-30min; no verdict on disk |
| 21 | reweight on fixed anchor | TORN_DOWN | 1×4×H200 (gone) | — | torn down 2026-06-03, no-heartbeat-30min; no verdict on disk |

## Last tick
2026-06-04T02:28:00+10:00 · running=[20] · analyzing=[] · logging=[] · blocked=[] · unclaimed=[19]

## Rescue-trigger watch (surfaced for operator)
- No §Rescue-triggers patterns fired yet (run just launched). The next monitor report will surface any `powersgd NaN/Inf`, `q_cond nan/inf`, `reconstruction_rel_error→1.0`, `single-GPU fallback`, FSDP `flat_param/summon_full_params`, or probe `PROBE_FAILED`.

## Notes (corrected 2026-06-04)
- **Earlier "degraded local branch" flag was WRONG — retracted.** The runner claimed local `vast-ai-workload` had autosave commits that deleted `comm_eff`; verification disproves it: `comm_eff/` (mask/anchor/spectral/state) is fully present on the local tip AND on disk. Commit `1c75d9166` ("full codex purge + retire findings/") only retired runs/findings/logs (operator's bloat cleanup) and left `comm_eff` untouched; the autosave commits touch no `verl/` code. Local is just 2 harmless autosaves ahead of `origin/vast-ai-workload @ 1c75d9166` (identical `verl/`). The runner forked origin (harmless — same code) but mis-stated the reason. `exp/20-powersgd-activation` correctly ADDS `powersgd_activation.py` alongside the intact `comm_eff`. No operator action needed.

## Budget
$/hr now: $15.21 (1 instance, EXP-20) · max_dph cap $24/instance (OK) · max_gpu_hr 96 (worst-case ~48 GPU-hr for full seq, OK) · prior EXP-18/EXP-21 both TORN_DOWN (not billing)
