# Research Status — 2026-06-01T17:20:00+10:00

## Active flow (operator goal — strict order, GPU-idle minimized)
1. ✅ EXP-17 training finished (val 0.7354 ~dense parity, no collapse).
2. ✅ Pushed (vast-ai-workload @ 9709bc1c8 + exp/17 branch).
3. ✅ EXP-17 report written (analyst PASS → verdict.md, log-writer → LOG + findings/M3/EXP-17.md).
4. ⏳ Big-Math harder-dataset run — EXP-19 (masked) + EXP-20 (dense baseline) CHAIN running on reused box, CORRECTED reward.

## Run pipeline

| EXP | Title | State | Vast | Verdict | Notes |
|---|---|---|---|---|---|
| 19 | Big-Math masked p=0.9 clean@20 (collapse test, fixed reward) | **RUNNING** (chain head; monitor+poller) | reuse i_38877541 4×H200 | — | data_source=DigitalLearningGmbH/MATH-lighteval → math_reward (float 0/1; mirrors min_rl_add). 120 steps, clean@20/40/60/80/100/120. tmux bigmath-chain-210_157_233_86, WandB grpo_mask_p0p9_clean20_bigmath_fixed. Watching step-0 val (prior crash point). |
| 20 | Big-Math DENSE baseline (comm-eff OFF) | QUEUED (chain tail) | reuse i_38877541 | — | Auto-starts after EXP-19 done.flag (run_bigmath_chain.sh). WandB grpo_dense_bigmath_baseline. Reference for EXP-19. |
| 18 | Big-Math masked (data_source=math_dapo) | SUPERSEDED | (killed) | — | Reward confounded: math_dapo default verifier scrapes "Answer:" not \boxed{} → biased/noisy. Superseded by EXP-19. |
| 17 | Long-horizon masked GRPO clean@20, 2ep (GSM8K) | DONE | i_38877541 (retained→reused) | PASS | val 0.7354; clean-step grad_norm trends down; gap clean-resettable sawtooth. |
| 16 | Stability matrix | DONE | 4×B200 (manual) TORN_DOWN | PASS | — |
| 11,10 | M3 misc | NOT_CLAIMED | — | — | Out of scope. |

## Reward-fix saga (resolved)
- EXP-18 (mine) used data_source=math_dapo → `is_correct_minerva` scrapes for "Answer:" token, ignores \boxed{} (the prompt's format) → biased reward. Operator (via monitor agent) independently diagnosed it, added a custom `math_bigmath` route returning `{"pred": None}` + designed EXP-19/EXP-20 chain.
- BUT `math_bigmath`'s `"pred": None` crashed `process_validation_metrics` (np.mean over [None,…] → TypeError) at val_before_train. Operator pointed to the `min_rl_add` branch as reference.
- **`min_rl_add` recipe** (examples/data_preprocess/math_dataset.py): `data_source="DigitalLearningGmbH/MATH-lighteval"` → `math_reward.compute_score` (last \boxed{} over full solution + is_equiv; returns **float 0/1**). Float has no pred key → val-safe. Adopted this (no verl/ edit needed; the route already exists). Verified end-to-end on-box: \boxed{6}→1.0, \boxed{7}→0.0, \boxed{\frac12}→1.0.
- The `math_bigmath` entry in verl/utils/reward_score/__init__.py (@265fca825) is now UNUSED + buggy-as-written (None pred) — do not route through it.

## Watchers (background)
- monitor-19: EXP-19 startup (confirm step-0 val passes, no TypeError) + collapse trajectory.
- poller bp9b95qzp: EXP-19 done.flag / chain done / error / tmux-death (~2.5h).
- On EXP-19 done.flag → flip EXP-19 ledger→COMPLETE, register EXP-20 RUNNING (keeps heartbeat anchored to active run), continue chain.

## Budget
$/hr now: $9.29 (i_38877541). EXP-19+EXP-20 chain ~4h ≈ 16 gpu-hr; well under 96-cap. Account credit ample.

## Notes
- Kill switch clear. gh default: shamanez/verl-compression-research. Code PRs → shamanez/verl base vast-ai-workload.
- Lesson recorded: [[gpu-idle-box-reuse]], reward-routing-for-boxed-prompts (see memory).
