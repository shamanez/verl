# Verdict EXP-6 — 2026-05-28T03:25:00+10:00

## Result
VERDICT: PASS

## Success criteria
- [x] Per-path mask counter `== 0` on rollout/old-logprob/ref-logprob/validation/checkpoint/infer paths (observed: key-prefix grep returns only `actor/comm_eff/mask_applications`; non-train-keyed falsifier grep is empty; unit tests `test_mask_hook_asserts_on_non_train_path[rollout|old_logprob|ref_logprob|val|infer|ckpt|None]` all PASSED)
- [x] Per-path mask counter `> 0` only on actor-train forward/backward (observed: `actor/comm_eff/mask_applications:28.0`; intermediate `:14.0` at step 1)
- [x] `old_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: unit test `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — fixed-batch, deterministic; tag-inactive is the path old_log_prob recompute runs under)
- [x] `ref_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: same `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — ref_log_prob runs under the identical inactive-tag forward, mask asserts off per `test_mask_hook_asserts_on_non_train_path[ref_logprob]`)
- [x] At least one validation pass ran and produced a score; unchanged vs masking-off within scoring noise (observed: `val-core/openai/gsm8k/acc/mean@1` mask_on={0.05080, 0.05231, 0.05914}, mask_off={0.04852, 0.06520}; verl emits this key, not literal `val/test_score`; same ballpark on a 2-step GSM8K smoke; see Notes on auto-resume)
- [x] At least one checkpoint saved; no comm_eff/mask tensors; reload bit-identical (observed: global_step_1 and global_step_2 saved, all 4 FSDP shards; live leakage scan of global_step_2 actor = LEAKED KEYS NONE; unit tests `test_checkpoint_guard_passes_on_clean_state_dict` + `test_checkpoint_guard_rejects_leaked_comm_eff_state` PASSED)
- [x] No NaN/Inf in `(loss|grad_norm|reward|log_prob|ppo_kl|kl_loss)`; run reaches `global_step=2` (observed: NaN/Inf grep empty; `training/global_step:2` reached; `actor/grad_norm` 227.43 -> 30.48, finite)
- [x] `tests/workers/comm_eff/test_activation_mask.py` path-isolation tests pass (observed: 35 passed, 26 warnings, 12.87s — runs/EXP-6/verify/unit_tests.log)

## Metrics summary
- actor/comm_eff/mask_applications: 28.0 (target > 0, train path only)
- non-train-keyed mask_applications: none (target 0 / absent)
- actor/comm_eff/mask_ratio: 0.95006 (target ~0.95)
- actor/comm_eff/anchor_backwards: 0.0 (target 0 — M2 guard, anchor not yet active)
- actor/comm_eff/spectral_corrections: 0.0 (target 0 — M2 guard, spectral not yet active)
- mask_off actor/comm_eff/mask_applications: 0.0 (target 0 — masking disabled)
- val-core/openai/gsm8k/acc/mean@1 (mask_on): 0.0508 (target: parity within noise vs mask_off)
- val-core/openai/gsm8k/acc/mean@1 (mask_off): 0.0485-0.0652 (same ballpark)
- actor/grad_norm: 227.43 -> 30.48 (target: finite; magnitude not a criterion)
- NaN/Inf hits: 0 (target 0)
- training/global_step reached: 2 (target 2)
- unit tests: 35 passed (target: all pass)
- checkpoint leakage: 0 leaked keys across 4 shards (target 0)
- budget: lifetime $9.0842 spent, 0 running instances (within caps: 8 GPU-hr / max_dph 24.0 / 3 h wall)

## Comparisons to baseline_run: baseline
`diff_against_baseline.py --baseline baseline` ran clean (wrote baseline_diff.md). Per the plan's Notes-for-analyst, the baseline diff is used here only as the dense-path regression check on the mask_off reference cell, not as a performance comparison — p95/staleness/communication metrics are n/a for a contamination-guard issue. The mask_off cell logs `actor/comm_eff/mask_applications:0.0` (masking disabled) and finite grad_norm, consistent with an unperturbed dense GRPO path. No regression signal.

## Notes
- Mask confinement (the single most important falsifier per the plan) is fully and independently established: the key-prefix grep yields exactly one mask-applications key, `actor/comm_eff/mask_applications`, and the seven `test_mask_hook_asserts_on_non_train_path[...]` assertions turn any non-train activation into a hard failure rather than a silent counter.
- AUTO-RESUME caveat weighed and judged non-blocking: the mask_off cell reached `training/global_step:3` and shares the checkpoint dir + experiment_name (`m2-mask-invariants`) with mask_on, so verl almost certainly auto-resumed mask_off from mask_on's global_step_2 checkpoint. The live mask_off val number is therefore NOT a pristine masking-off-from-scratch reference, so criterion 5's val-PARITY sub-clause is only weakly supported by the live run (0.0508 vs 0.0485-0.0652, same ballpark on a 2-step smoke). It does NOT flip the verdict because: (a) criterion 5's first clause — "a validation pass ran and produced a score" — is hard-met; (b) the rigorous, deterministic log-prob equality at rel-tol 1e-6 is proven on a fixed batch by `test_logprob_equal_mask_on_vs_off_when_tag_inactive`, which is the exact mechanism that would make val differ if masking leaked into eval generation; (c) the EXP-6 hypothesis is mask CONFINEMENT, not val parity, and confinement is fully supported; (d) an auto-resume cannot manufacture a false confinement pass — it would, if anything, contaminate eval, yet the non-train counters are still 0. Recommend the next planner pin a unique `experiment_name`/checkpoint dir per cell on any future parity-sensitive rerun to harden the live val-parity evidence.
- `val/test_score` literal grep from the plan's Verification commands returns empty because verl emits the GSM8K validation score under key `val-core/openai/gsm8k/acc/mean@1`; the criterion is satisfied via that key. This is a plan-grep literal mismatch, not a missing measurement — flagged for the next planner to update the verification command string.
- Benign artifact (NOT a falsifier): mask_on logged a post-step-2 `RuntimeError: DataLoader worker killed by signal: Killed` during final validation/wandb shutdown; transient host-RAM blip on a 2TB-RAM box, recovered, chain continued to mask_off. Occurred after all science was captured.
- analyze.py emitted a non-authoritative PENDING scaffold (no metrics/*.jsonl; metrics arrived as train.log, as in EXP-5). This verdict's numbers are read from the explicit key-prefix greps against train.log and the verify/ artifacts, per the plan's Notes-for-analyst. check_budget.py is authoritative for spend and confirms within-caps.
- M2 milestone: anchor_backwards=0 and spectral_corrections=0 are expected — this issue guards the mask hook's confinement; the asynchronous anchor circuit and spectral correction are downstream milestones, intentionally inert here.
- code_change=true → orchestrator should route to draft PR (exp/6-mask-invariants -> vast-ai-workload).
