# Research Status — 2026-07-01T18:40+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 44 | Weight-proj: extend offline sweep engine (orders≥2, damped-α, learnable-at-every-order, regression, EMA; new GPU-free metrics) | DONE | — (kind:analysis, GPU-free) | PASS | Engine ACCEPTED as shared substrate for #45-#56; 8/8 success criteria. Prior STOP (BF16_FLOOR_BLOCKS) OVERTURNED — bf16 noise-floor category error (597-2167x over-estimate); corrected differenced floor on changed support + directedness discriminator (p=1.05/1.04, R²≈0.99 DIRECTED, matches probe p≈1.07) + PuLSE metrics. 15/15 families reconstruct, grouping 338/11/28 exact, bounded streaming. weight_proj_ratio>1 / h*=0 for extrapolators = VALID finding for #52-#56, not an engine failure. bf16 stays fixed (fp32 deferred → #57). Fix committed 71500bd5 on vast-ai-workload (research/scripts/ only). No PR (code already landed, promote:none). |
| 43 | Collect dense regime-A full-weight per-tick trace → R2 | DONE | 1×H200 (i_43197578, TEAM) TORN_DOWN | PASS | M4 spine trace, 160/160 snapshots R2-verified; sole input to #44. |

Other M4 weight-proj issues (#45-#56) are open at kind:analysis but NOT status:approved (awaiting human plan review). #44 is now DONE/PASS. M4 is NOT yet achieved — it requires the #45-#56 spine; #43 (data) and #44 (engine) are infra units, not the milestone science result. fp32 re-collection is deferred and tracked as issue #57.

## Last tick
2026-07-01T18:40+10:00 · running=[] · analyzing=[] · logging=[] · blocked=[]

## Budget
No live Vast.ai box. Teardown sweep rc=0 (no-op). EXP-44 was GPU-free (R2 egress only). Ledger: EXP-43 rows ABORTED + TORN_DOWN.
