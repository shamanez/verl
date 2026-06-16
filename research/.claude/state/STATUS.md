# Research Status — 2026-06-16 (EXP-32 RUNNING — signed_ema α=0.5 valid-M closure)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 32 | signed_ema α=0.5 on CORRECTED valid-M anchor circuit (#29) | **RUNNING** (monitor active, bg) | 1×4H200 (operator box 46.243.55.134:40154, separate Vast acct) | — | config-only, B2 + ONE knob (correction_mode=signed_ema, α=0.5) via Hydra passthrough; knob confirmed live 3 ways; step-1 healthy (bytes_ratio 0.0504, no ignition); val 0/25/50; val@0=0.0826; ~60 min for 50 steps |
| 31 | Anchor-signal-usage tournament (L4/L2/L3/L1) | DONE / VERDICT STOP | 1×4H200 (i_41048644, operator) | STOP | all 4 levers NULL for surpass; B2_live@25=0.7202 / val@50=0.7354; logged M6. Surpass-dense mandate CLOSED. |
| 28 | EXP-28 TRUE error-feedback successor | PLAN_READY? (kind:experiment, no status label) | — | — | not approved; out of scope this drive |

## EXP-32 cell summary (RUNNING)

- **Cell 1 (exp32_signed_ema_a0p5_validM)** — RUNNING. B2 substrate + exactly one knob flipped: `spectral.correction_mode=signed_ema` + `signed_ema_alpha=0.5` (both via Hydra `"$@"` passthrough — the env-var route is clobbered by the launcher's hard `export ...=delayed_ef` at L52). Knob confirmed live 3 independent ways (resolved set -x trace, Hydra config tree, per-fire merger banner `correction_mode=signed_ema alpha=0.5 corrected=196` on all 4 ranks). Controlled vars asserted == B2: bytes_ratio=0.0504 (gate [0.0500,0.0510] ✓), PowerSGD r=77, anchor on/owns_q/cad5/dK5, replay_paired_batch, snapshot_device=cpu, clean=0. max_mem 17.4 GB. val@0=0.0826. No NaN/OOM/custom_all_reduce crash. response_length/mean 276.6 (no length-hack ignition).
- **Cell 2 (same-box B2 reference)** — CONDITIONAL, not yet launched. Only if idle GPU-hr remain under max_gpu_hr=48 after cell 1 banks.

## Decision point — val@25

- **Expected:** signed_ema α=0.5 valid-M lands ~0.70 — dominated by B2 (band ≈0.735–0.753), clears the corrected no-merger floor C2=0.6300. That is the EXPECTED closure PASS → early-kill at val@25, dispatch analyst.
- **Breakthrough branch:** val@25 ≥ 0.7066 AND rising → let it run to 50 (potential surpass; STOP-and-new-issue per plan).
- **Collapse branch:** val@25 < 0.690 with flat/negative slope, OR a length-hack ignition trip-wire (P1/P2/P3/E1) fires → STOP (record the dominated/ignited box as the result; read (c) for ignition).

## Box status

- **46.243.55.134:40154** (EXP-32, 4×H200, operator separate Vast acct): RUNNING. NOT in project VAST_API_KEY ⇒ operator-managed teardown (do NOT vastai-teardown). Heartbeat path runs/EXP-32/metrics/incoming.log materialized; sync-metrics + monitor reach it (both keys appended).

## Last tick

2026-06-16 · running=[32] · analyzing=[] · logging=[] · blocked=[] · monitor(bg) active on EXP-32

## Budget

EXP-32 on operator box (dph=0 in our acct; operator-billed). No project-billed instance live. max_gpu_hr=48 cap; one 50-step cell ~6-8 GPU-hr.
