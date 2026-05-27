# Research Status — 2026-05-28T01:50+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 5 | M2 actor-only PRF activation masking (2-step smoke) | PASS | 1×4H200 (torn down) | PASS · mask_ratio ±0.02p, confinement OK, grad finite, no NaN/Inf, EXP-4 no-op regression clean | findings/M2/EXP-5.md filed; PR drafted exp/5-actor-mask→vast-ai-workload |

## Last tick
2026-05-28T01:50+10:00 · verify=[] · running=[] · analyzing=[] · logging=[] · blocked=[]

## Budget
EXP-5 spend: $3.70 (4×H200, tier-1; 3 h wall) — under all caps.

## Metrics archived
- EXP-5 p95: mask_ratio 0.9498/0.9502, actor/grad_norm finite 42.15/19.86, zero mask on non-actor paths
- EXP-5 p90: mask_ratio 0.8999/0.9002, actor/grad_norm finite 18.11/95.18, zero mask on non-actor paths
- EXP-5 disabled: all comm_eff counters 0 (dense no-op contract matched), grad_norm 1.13/0.37

## Notes
- Verdict filed 2026-05-28T01:45+10:00; instance torn down 01:42:27; metrics rsync complete.
- Milestone M2 now has ≥2 PASS (EXP-5 + one more expected from related cell). Checking for SUMMARY.md creation eligibility.
