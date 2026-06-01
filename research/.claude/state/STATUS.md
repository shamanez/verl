# Research Status — 2026-06-01T16:30:00+10:00

## Active flow (operator goal — strict order, GPU-idle minimized)
1. ✅ Finish EXP-17 training — DONE (116 steps, final val 0.7354 ~dense parity, no collapse).
2. ✅ Push all changes — vast-ai-workload @ 90be6ed92 + exp/17-masked-clean-every20 on origin.
3. ⏳ EXP-17 report writing — analyst running (background, separate team) → verdict.md; log-writer to follow.
4. ⏳ Big-Math harder-dataset run — EXP-18 LAUNCHED on REUSED box 38877541 (no re-provision; ~0 idle).

## Issue / run pipeline

| EXP | Title | State | Vast | Verdict | Notes |
|---|---|---|---|---|---|
| 18 | Harder-dataset collapse test — masked GRPO p=0.9 + clean@20 on Big-Math | **RUNNING** (monitor + poller) | reuse i_38877541 4×H200 $9.29/hr | — | Operator-directed (plan .claude/plans/18.md, no GH issue). data_source=math_dapo (\boxed), 20k train/500 val, 120 steps/1 epoch, clean@20/40/60/80/100/120. tmux exp-18-210_157_233_86, WandB grpo_mask_p0p9_clean20_bigmath_collapse. Collapse-test: low reward expected, only hard errors = teardown. |
| 17 | Long-horizon masked GRPO clean_cadence=20, 2 epochs | COMPLETE → reporting | i_38877541 (retained, reused by EXP-18) | (analyst running) | val 0.7354 (dense 0.741, clean@5 0.729, clean@4 0.696); 5 clean steps fired; clean-step grad_norm 0.4258→0.4026 (down=repair holding); pg_clipfrac 0.034; mask confinement + anchor/spectral-off held. WandB t03dn4nh. Ledger row COMPLETE (box protected from teardown for reuse). |
| 16 | Short-run stability matrix | DONE | 4×B200 (manual) TORN_DOWN | PASS (manual) | EXP-17 depends_on:[16] satisfied. Evidence runs/EXP-16/. |
| 11 | M3 100-step M95+AP (K=20) | NOT_CLAIMED (superseded by #17) | — | — | Out of orchestrator scope. |
| 10 | M3 DP gradient compression scope | NOT_CLAIMED | — | — | Out of orchestrator scope. |

## Box-reuse mechanism (EXP-17 → EXP-18, idle minimization)
- EXP-17 ledger row flipped RUNNING → **COMPLETE** (a non-RUNNING terminal status the teardown-finished-runs hook skips). So the report team's verdict.md will NOT trigger teardown of the shared box.
- EXP-18 registered as the new **RUNNING** owner of instance 38877541 (started_at reset for the budget clock; heartbeat protects it). When EXP-18 writes verdict.md / dies / hits max_gpu_hr, the Stop hook tears the box down normally.
- Net: GPU went EXP-17→EXP-18 with only unavoidable bring-up idle (dataset prep ran on-box CPU while GPUs were briefly free; no re-provisioning gap, no scarcity risk).

## Last tick
2026-06-01T16:30:00+10:00 · running=[18] · reporting=[17 analyst] · logging=[] · blocked=[] · skipped=[11,10 not-claimed]

## Budget
$/hr now: $9.29 (i_38877541, EXP-18 training) · EXP-17 used ~6 gpu-hr · EXP-18 ~8 gpu-hr expected (120 steps ~2h × 4 GPU) · max_gpu_hr 96 per-run cap · account credit ample.

## Watchers
- analyst-17 (background): EXP-17 verdict.md + issue #17 label.
- monitor-18 (background): EXP-18 startup health + collapse-trajectory, 40-min window.
- bash poller bevqay2wv (background): reliable EXP-18 done.flag/crash backbone (~2.8h) — the EXP-17 monitor agents stalled on long polls, so this is the dependable completion signal.

## Notes
- Kill switch clear. gh default: shamanez/verl-compression-research (issues). Code PRs → shamanez/verl base vast-ai-workload.
- Pending: analyst-17 returns → dispatch log-writer (step 3 finish). EXP-18 done.flag → analyst for EXP-18 (predicate in plans/18.md).
