# Research Status — 2026-05-28T01:33+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 5 | M2 actor-only PRF activation masking (2-step smoke) | RUNNING | 1×4H200 (i_38098877) | — | verify=CONCERNS→VERIFIED; exp/5-actor-mask @69524742 pushed; cell p95 in model-load; done.flag-dir pre-created on box to keep 3-cell chain alive |

## Last tick
2026-05-28T01:33+10:00 · verify=[] · running=[5] · analyzing=[] · logging=[] · blocked=[]

## Budget
$/hr now: $14.74 (4×H200, tier-1; tier-0 4×H100 had 0 offers) · cap/instance: $24 · run cap: 8 GPU-hr / 3 h wall

## EXP-5 cell metrics (live)
- p95: model-load/vLLM-init, no train steps yet · grad_norm: n/a · mask_ratio: n/a · NaN: none
- p90: not started
- disabled: not started

## Notes
- Carryover launcher done.flag bug (hardcoded path under SAVE_FREQ=-1) confirmed live on exp/5; mitigated by pre-creating /workspace/verl/runs/qwen25_1p5b_grpo_gsm8k_baseline on the box so `touch done.flag` succeeds → chain does not abort under set -e.
- Analyst note (from verify CONCERNS): grep mask-confinement on BOTH p95 AND p90 logs (not just p95); disabled-cell --baseline EXP-3 is the harness standard baseline-diff entry point.
