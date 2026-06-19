# Research Status — 2026-06-20 (EXP-37B launched)

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 37 | EXP-37B: 5/5-latency 100-step epoch-boundary control | RUNNING | 1×4H200 (i_41680420, team, $12.88/hr) | — | launched 2026-06-20; banner verified cadence=5/delay_K=5, 100 steps, epochs=2; background monitor active |

## EXP-37B launch facts
- **Box**: operator-provisioned team-account 4×H200 — instance `41680420`, `84.8.106.109:40206`, dph $12.88 (< $24 cap; ~1 gpu-hr expected, < max_gpu_hr 48). SSH `~/.ssh/vast_ai` (vast_ai_name also accepted).
- **Cell**: `exp-37b-cad5-delay5-100step` — signed_ema(α=0.25, β_anc=0.50) accel base, **anchor cadence=5 / delay_K=5** (NOT EXP-37's 20/20), 100 steps, test_freq=25. NO trailing Hydra anchor overrides (5/5 is the accel-base default; banner is trustworthy for this run).
- **WandB**: project `verl_compression_research_accel_rebaseline` (overlays EXP-37 20/20 collapse, EXP-36B 5/5@50, EXP-36C dense@50).
- **tmux**: `exp-37b-84_8_106_109`; train log `/workspace/train.log`; banner `/workspace/runs/EXP-37B/launch-banner.log`.
- Data prep OK (7473 train / 1319 test). No FATAL at init. main_ppo started.

## Hypothesis / deliverable
Back-half (steps 50–100, centered on GSM8K epoch-2 boundary ~step 58) STABLE vs COLLAPSE classification → disambiguates whether EXP-37's post-step-50 collapse was **latency-driven** (5/5 stays stable ⇒ PASS, epoch-hypothesis rejected) or **epoch-driven** (5/5 also collapses ⇒ STOP, still a finding). NOT a surpass-dense claim.

## Gates (plan §Success criteria)
- 100 steps reached, no NaN; **latency realized: anchor_backwards == 40** (200 ticks / cadence 5), realized delay ≥ 5.
- reproduction sanity: val@50 ≥ 0.6862 (within 0.05 of EXP-36B 0.7362) — NOT the headline.
- bytes_ratio ≈ 0.0505 (fast-path Y + amortized Q only; full-dense M broadcast uncounted).

## Base reference (the default loop)
`vast_comm_eff_accel_base_*.sh` — signed_ema(α=0.25, β_anc=0.50), accel surface @ gpu_mem 0.55, diagnostics=false, PowerSGD r=77 anchor. ~25 min / 50 steps.

| arm | run | val@50 |
|---|---|---|
| dense control (comm-eff OFF) | EXP-36C | 0.7657 |
| comm-eff signed_ema(0.25, 0.50), 5/5@50 | EXP-36B | 0.7362 |
| comm-eff signed_ema(0.25, 0.50), 20/20@100 | EXP-37 | collapsed after ~step 50 |

## Pending close-out (owning session duties — per [[agent-owns-closeout-backfill-teardown]])
1. Re-dispatch monitor when the first (40-min) monitor returns TIMEOUT — full run is ~50–60 min.
2. On `done.flag`: dispatch analyst (plan §Analyst predicate), then **backfill final 1–2 WandB steps from local train.log** (async uploader drops the tail), then **teardown box 41680420 with the TEAM account** + verify 0 live.

## Last tick
2026-06-20 · running=[37B] · analyzing=[] · logging=[] · blocked=[]

## Budget
$/hr now: $12.88 (1 box, team account) · max_dph cap $24 · max_gpu_hr 48
