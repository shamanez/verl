# Codex Adversarial — 2026-05-28T03:38:49+10:00
ADVERSARIAL: CONTESTED

<!-- Verdict header prepended by codex-bridge for orchestrator grep; full Codex exec trace + final structured verdict (also tailed with ADVERSARIAL: CONTESTED) preserved verbatim below. -->

I’ll sanity-check the local M2 files if they’re present, then evaluate the claims against the evidence as stated. I’m treating this as a methodological review, not a code fix.
exec
/bin/zsh -lc "sed -n '1,260p' findings/M2/EXP-6.md" in /Users/shamane/Documents/verl
exec
/bin/zsh -lc "sed -n '1,240p' findings/M2/SUMMARY.md" in /Users/shamane/Documents/verl
 exited 1 in 0ms:
sed: findings/M2/EXP-6.md: No such file or directory

 exited 1 in 0ms:
sed: findings/M2/SUMMARY.md: No such file or directory

exec
/bin/zsh -lc "pwd && rg --files findings/M2 . | rg 'findings/M2/(SUMMARY|EXP-5|EXP-6)\\.md"'$|AGENTS'"\\.md"'$'"'" in /Users/shamane/Documents/verl
 succeeded in 0ms:
/Users/shamane/Documents/verl
rg: findings/M2: No such file or directory (os error 2)
./AGENTS.md
./research/findings/M2/EXP-6.md
./research/findings/M2/SUMMARY.md
./research/findings/M2/EXP-5.md

codex
The files are under `research/findings/M2`, not `findings/M2`. I’m reading the local summary and EXP-6 evidence now to avoid relying only on the pasted excerpt.
exec
/bin/zsh -lc "sed -n '1,260p' research/findings/M2/SUMMARY.md" in /Users/shamane/Documents/verl
 succeeded in 0ms:
# M2 Milestone Summary

**M2 goal:** comm-eff integration smokes (mask/spectral/anchor, two-step GRPO).

Establish that the activation-masking circuit integrates into verl's GRPO actor
update, is provably confined to the actor-train forward/backward, leaves every
RL-measurement path bit-unchanged, and keeps checkpoints mask-free. M2 gates the
two downstream circuits (spectral correction, async anchor). Three PASS findings.

## What M2 establishes

1. **Integration** — the masking hook plugs into the actor-train path without
   altering the GRPO sequence (rollout → old_logprob → ref_logprob → reward →
   advantage → update_actor → weight-sync) in any of the disabled / p90 / p95 cells.
2. **Confinement** — masks fire *only* on actor-train. The explicit path-tag
   (train|rollout|old_logprob|ref_logprob|val|infer|ckpt) + assert-on-wrong-path
   guard turns silent contamination into a loud failure. Per-path counters: 28
   applications on actor-train across two substeps; **0 on every other path**.
3. **Measurement correctness** — old_log_prob and ref_log_prob recomputed with
   masking enabled equal the mask-off values within 1e-6 on a fixed batch
   (deterministic unit test). RL measurements are unperturbed by mask state.
4. **Checkpoint hygiene** — no comm_eff/mask tensors leak into saved weights
   (live leakage scan of global_step_2: 0 keys across 4 FSDP shards) and the
   guard unit tests pass.
5. **Reproducibility** — boundary set [3,7,11,15,18,21,24] is derived from
   model.config (L=28 Qwen2.5-1.5B / pp_size=8); mask_ratio tracks p within ±0.02.

## Findings

| EXP | What | Result | PR |
|---|---|---|---|
| EXP-4 | comm_eff no-op scaffolding: config group + disabled-by-default hooks | PASS — disabled == dense parity | #1 merged → vast-ai-workload |
| EXP-5 | actor-only PRF activation masking (in-graph h*mask, no rescale) | PASS — mask_ratio tracks p (p95→0.950, p90→0.900, ±0.02); confined to actor-train; grads finite | #2 merged → vast-ai-workload |
| EXP-6 | mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS — per-path counters train=28 / all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean, no NaN | #3 (draft) → vast-ai-workload |

**EXP-5 headline:** boundaries [3,7,11,15,18,21,24]; p95 mask_ratio 0.9498/0.9502,
p90 0.8999/0.9002; disabled cell all comm_eff counters 0; grads finite. (findings/M2/EXP-5.md)

**EXP-6 headline:** per-path counters train=28, rollout/old_logprob/ref_logprob/val/infer/ckpt=0;
old/ref log-prob bit-equal within 1e-6 mask-on vs mask-off; validation ran
(val-core/openai/gsm8k/acc/mean@1 parity within noise); checkpoint save/load clean
(0 leaked keys / 4 shards); anchor_backwards=0, spectral_corrections=0 (both inert
in M2, as designed); grad_norm 227.43→30.48 finite. 35 unit tests passed. (findings/M2/EXP-6.md)

## Open follow-ups

1. **Launcher done.flag bug** — `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
   hardcodes a done.flag path that fails under `SAVE_FREQ=-1`, aborting multi-cell
   chains under `set -e`. EXP-5 and EXP-6 worked around it on-box (pre-created the
   dir). A real fix (`$EXPERIMENT_NAME` + `mkdir -p`) still belongs in the launcher.
2. **EXP-6 mask_off auto-resume** — the mask_off reference cell shared
   `experiment_name`/checkpoint dir with mask_on, so verl auto-resumed it from
   mask_on's global_step_2 (mask_off reached step 3, not a fresh step 2). This only
   weakens the *live* val-parity sub-clause; confinement is unaffected (an auto-resume
   cannot manufacture a false confinement pass) and the 1e-6 log-prob equality is
   proven independently by unit test. Next planner: pin a unique experiment_name per
   cell (m2-mask-on vs m2-mask-off) for pristine val-parity evidence.
3. **Plan-grep literal mismatch** — plans grep `val/test_score`, but verl emits
   `val-core/openai/gsm8k/acc/mean@1`. Update plan templates.

## Not yet covered

- **Spectral correction** (issue #7) — gradient correction for masked params; unplanned.
- **Async anchor circuit** (issue #8) — unmasked auxiliary forward; unplanned.

Both remain intentionally inert in M2 (`spectral_corrections=0`, `anchor_backwards=0`)
and will gate on this milestone's confinement + correctness proofs.

exec
/bin/zsh -lc "sed -n '1,320p' research/findings/M2/EXP-6.md" in /Users/shamane/Documents/verl
 succeeded in 0ms:
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

exec
/bin/zsh -lc 'rg -n "mask_applications|path_tag|old_log_prob|ref_log_prob|auto-resume|resume|val-core|checkpoint|leak|1e-6|tag" research/findings/M2 -S' in /Users/shamane/Documents/verl
 succeeded in 0ms:
research/findings/M2/codex-review.md.partial:24:- It leaves every RL-measurement path (rollout, old_logprob, ref_logprob, val, infer, ckpt) bit-unchanged (1e-6 log-prob equality).
research/findings/M2/codex-review.md.partial:25:- Checkpoints are mask-free.
research/findings/M2/codex-review.md.partial:28:[Confinement] An explicit path-tag (train|rollout|old_logprob|ref_logprob|val|infer|ckpt) + an assert-on-wrong-path guard in the forward hook. Per-path counters: actor/comm_eff/mask_applications=28 on actor-train across two substeps; 0 (key absent) on every other path. Seven unit tests test_mask_hook_asserts_on_non_train_path[rollout|old_logprob|ref_logprob|val|infer|ckpt|None] PASSED — they assert that the hook RAISES if it ever fires under a non-train tag. Confinement at runtime is evidenced by a key-prefix grep on train.log: the only mask-applications metric emitted is actor/comm_eff/mask_applications; no non-train-keyed counter exists.
research/findings/M2/codex-review.md.partial:29:[Measurement correctness] old_log_prob and ref_log_prob mask-on vs mask-off equal within rel-tol 1e-6, proven by ONE unit test test_logprob_equal_mask_on_vs_off_when_tag_inactive — described as "fixed-batch, deterministic". The argument is that old_logprob/ref_logprob recompute runs under the tag-INACTIVE forward, so the mask hook is inert there.
research/findings/M2/codex-review.md.partial:30:[Live val] val-core/openai/gsm8k/acc/mean@1 mask_on={0.05080,0.05231,0.05914}, mask_off={0.04852,0.06520}, "same ballpark", 2-step GSM8K smoke.
research/findings/M2/codex-review.md.partial:31:[Checkpoint] live leakage scan loaded all 4 FSDP shards of global_step_2/actor and grepped keys for comm_eff|mask_applications|path_tag|anchor|spectral -> NONE. Plus 2 unit tests for a checkpoint guard.
research/findings/M2/codex-review.md.partial:34:KNOWN CAVEAT (self-reported): EXP-6 mask_off reference cell SHARED experiment_name (m2-mask-invariants) + checkpoint dir with mask_on, so verl auto-resumed mask_off from mask_on's global_step_2 checkpoint (mask_off reached step 3, not a fresh step 2). The verdict argues this only weakens the live val-parity sub-clause, not confinement.
research/findings/M2/codex-review.md.partial:37:1. Where could the "confinement proof" be CIRCULAR or under-powered? (e.g. does the per-path counter being 0 on non-train paths actually prove no contamination, or only that the COUNTER didn't increment? Could masking perturb a measurement path WITHOUT incrementing the counter — e.g. a hook firing before the tag is set, a stale tag from a prior phase, a path that never sets a tag at all and thus defaults to None, multiple model replicas/microbatches where only one carries the tag, FSDP/vLLM weight-sync copying a masked activation cache?)
research/findings/M2/codex-review.md.partial:38:2. Is the assert-on-wrong-path guard a sufficient SILENT-FAILURE net? What failure modes would it NOT catch? (hook not installed on the eval model at all; eval running on a separate vLLM engine the hook never touches; tag set to train during an eval that is mislabeled; the assert being compiled out / swallowed under torch.compile or no_grad or inference_mode; the guard only checking tag value not whether mask was actually applied.)
research/findings/M2/codex-review.md.partial:39:3. Is the 1e-6 log-prob equality REPRESENTATIVE or a toy fixed batch? One test, one batch — does "tag-inactive forward == mask-off forward" tautologically hold by construction (if the tag gates the hook, of course they're equal) rather than testing the real measurement path? Does it exercise the real GSM8K sequence lengths / the real boundary layers [3,7,11,15,18,21,24] / FSDP-sharded forward / bf16, or a tiny fixture?
research/findings/M2/codex-review.md.partial:40:4. Does the mask_off auto-resume undermine anything BEYOND val-parity? (Could the resumed optimizer/model state mean the "mask_off" cell is actually running a masked-trained model, so its 0 mask counter and finite grads are not an independent control at all?)
research/findings/M2/codex-review.md.partial:51:RL-measurement path bit-unchanged, and keeps checkpoints mask-free. M2 gates the
research/findings/M2/codex-review.md.partial:58:   advantage → update_actor → weight-sync) in any of the disabled / p90 / p95 cells.
research/findings/M2/codex-review.md.partial:59:2. **Confinement** — masks fire *only* on actor-train. The explicit path-tag
research/findings/M2/codex-review.md.partial:63:3. **Measurement correctness** — old_log_prob and ref_log_prob recomputed with
research/findings/M2/codex-review.md.partial:64:   masking enabled equal the mask-off values within 1e-6 on a fixed batch
research/findings/M2/codex-review.md.partial:66:4. **Checkpoint hygiene** — no comm_eff/mask tensors leak into saved weights
research/findings/M2/codex-review.md.partial:67:   (live leakage scan of global_step_2: 0 keys across 4 FSDP shards) and the
research/findings/M2/codex-review.md.partial:78:| EXP-6 | mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS — per-path counters train=28 / all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean, no NaN | #3 (draft) → vast-ai-workload |
research/findings/M2/codex-review.md.partial:84:old/ref log-prob bit-equal within 1e-6 mask-on vs mask-off; validation ran
research/findings/M2/codex-review.md.partial:85:(val-core/openai/gsm8k/acc/mean@1 parity within noise); checkpoint save/load clean
research/findings/M2/codex-review.md.partial:86:(0 leaked keys / 4 shards); anchor_backwards=0, spectral_corrections=0 (both inert
research/findings/M2/codex-review.md.partial:95:2. **EXP-6 mask_off auto-resume** — the mask_off reference cell shared
research/findings/M2/codex-review.md.partial:96:   `experiment_name`/checkpoint dir with mask_on, so verl auto-resumed it from
research/findings/M2/codex-review.md.partial:98:   weakens the *live* val-parity sub-clause; confinement is unaffected (an auto-resume
research/findings/M2/codex-review.md.partial:99:   cannot manufacture a false confinement pass) and the 1e-6 log-prob equality is
research/findings/M2/codex-review.md.partial:103:   `val-core/openai/gsm8k/acc/mean@1`. Update plan templates.
research/findings/M2/EXP-6.md:7:- [x] Per-path mask counter `== 0` on rollout/old-logprob/ref-logprob/validation/checkpoint/infer paths (observed: key-prefix grep returns only `actor/comm_eff/mask_applications`; non-train-keyed falsifier grep is empty; unit tests `test_mask_hook_asserts_on_non_train_path[rollout|old_logprob|ref_logprob|val|infer|ckpt|None]` all PASSED)
research/findings/M2/EXP-6.md:8:- [x] Per-path mask counter `> 0` only on actor-train forward/backward (observed: `actor/comm_eff/mask_applications:28.0`; intermediate `:14.0` at step 1)
research/findings/M2/EXP-6.md:9:- [x] `old_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: unit test `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — fixed-batch, deterministic; tag-inactive is the path old_log_prob recompute runs under)
research/findings/M2/EXP-6.md:10:- [x] `ref_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: same `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — ref_log_prob runs under the identical inactive-tag forward, mask asserts off per `test_mask_hook_asserts_on_non_train_path[ref_logprob]`)
research/findings/M2/EXP-6.md:11:- [x] At least one validation pass ran and produced a score; unchanged vs masking-off within scoring noise (observed: `val-core/openai/gsm8k/acc/mean@1` mask_on={0.05080, 0.05231, 0.05914}, mask_off={0.04852, 0.06520}; verl emits this key, not literal `val/test_score`; same ballpark on a 2-step GSM8K smoke; see Notes on auto-resume)
research/findings/M2/EXP-6.md:12:- [x] At least one checkpoint saved; no comm_eff/mask tensors; reload bit-identical (observed: global_step_1 and global_step_2 saved, all 4 FSDP shards; live leakage scan of global_step_2 actor = LEAKED KEYS NONE; unit tests `test_checkpoint_guard_passes_on_clean_state_dict` + `test_checkpoint_guard_rejects_leaked_comm_eff_state` PASSED)
research/findings/M2/EXP-6.md:17:- actor/comm_eff/mask_applications: 28.0 (target > 0, train path only)
research/findings/M2/EXP-6.md:18:- non-train-keyed mask_applications: none (target 0 / absent)
research/findings/M2/EXP-6.md:22:- mask_off actor/comm_eff/mask_applications: 0.0 (target 0 — masking disabled)
research/findings/M2/EXP-6.md:23:- val-core/openai/gsm8k/acc/mean@1 (mask_on): 0.0508 (target: parity within noise vs mask_off)
research/findings/M2/EXP-6.md:24:- val-core/openai/gsm8k/acc/mean@1 (mask_off): 0.0485-0.0652 (same ballpark)
research/findings/M2/EXP-6.md:29:- checkpoint leakage: 0 leaked keys across 4 shards (target 0)
research/findings/M2/EXP-6.md:33:`diff_against_baseline.py --baseline baseline` ran clean (wrote baseline_diff.md). Per the plan's Notes-for-analyst, the baseline diff is used here only as the dense-path regression check on the mask_off reference cell, not as a performance comparison — p95/staleness/communication metrics are n/a for a contamination-guard issue. The mask_off cell logs `actor/comm_eff/mask_applications:0.0` (masking disabled) and finite grad_norm, consistent with an unperturbed dense GRPO path. No regression signal.
research/findings/M2/EXP-6.md:36:- Mask confinement (the single most important falsifier per the plan) is fully and independently established: the key-prefix grep yields exactly one mask-applications key, `actor/comm_eff/mask_applications`, and the seven `test_mask_hook_asserts_on_non_train_path[...]` assertions turn any non-train activation into a hard failure rather than a silent counter.
research/findings/M2/EXP-6.md:37:- AUTO-RESUME caveat weighed and judged non-blocking: the mask_off cell reached `training/global_step:3` and shares the checkpoint dir + experiment_name (`m2-mask-invariants`) with mask_on, so verl almost certainly auto-resumed mask_off from mask_on's global_step_2 checkpoint. The live mask_off val number is therefore NOT a pristine masking-off-from-scratch reference, so criterion 5's val-PARITY sub-clause is only weakly supported by the live run (0.0508 vs 0.0485-0.0652, same ballpark on a 2-step smoke). It does NOT flip the verdict because: (a) criterion 5's first clause — "a validation pass ran and produced a score" — is hard-met; (b) the rigorous, deterministic log-prob equality at rel-tol 1e-6 is proven on a fixed batch by `test_logprob_equal_mask_on_vs_off_when_tag_inactive`, which is the exact mechanism that would make val differ if masking leaked into eval generation; (c) the EXP-6 hypothesis is mask CONFINEMENT, not val parity, and confinement is fully supported; (d) an auto-resume cannot manufacture a false confinement pass — it would, if anything, contaminate eval, yet the non-train counters are still 0. Recommend the next planner pin a unique `experiment_name`/checkpoint dir per cell on any future parity-sensitive rerun to harden the live val-parity evidence.
research/findings/M2/EXP-6.md:38:- `val/test_score` literal grep from the plan's Verification commands returns empty because verl emits the GSM8K validation score under key `val-core/openai/gsm8k/acc/mean@1`; the criterion is satisfied via that key. This is a plan-grep literal mismatch, not a missing measurement — flagged for the next planner to update the verification command string.
research/findings/M2/SUMMARY.md:7:RL-measurement path bit-unchanged, and keeps checkpoints mask-free. M2 gates the
research/findings/M2/SUMMARY.md:14:   advantage → update_actor → weight-sync) in any of the disabled / p90 / p95 cells.
research/findings/M2/SUMMARY.md:15:2. **Confinement** — masks fire *only* on actor-train. The explicit path-tag
research/findings/M2/SUMMARY.md:19:3. **Measurement correctness** — old_log_prob and ref_log_prob recomputed with
research/findings/M2/SUMMARY.md:20:   masking enabled equal the mask-off values within 1e-6 on a fixed batch
research/findings/M2/SUMMARY.md:22:4. **Checkpoint hygiene** — no comm_eff/mask tensors leak into saved weights
research/findings/M2/SUMMARY.md:23:   (live leakage scan of global_step_2: 0 keys across 4 FSDP shards) and the
research/findings/M2/SUMMARY.md:34:| EXP-6 | mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS — per-path counters train=28 / all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean, no NaN | #3 (draft) → vast-ai-workload |
research/findings/M2/SUMMARY.md:40:old/ref log-prob bit-equal within 1e-6 mask-on vs mask-off; validation ran
research/findings/M2/SUMMARY.md:41:(val-core/openai/gsm8k/acc/mean@1 parity within noise); checkpoint save/load clean
research/findings/M2/SUMMARY.md:42:(0 leaked keys / 4 shards); anchor_backwards=0, spectral_corrections=0 (both inert
research/findings/M2/SUMMARY.md:51:2. **EXP-6 mask_off auto-resume** — the mask_off reference cell shared
research/findings/M2/SUMMARY.md:52:   `experiment_name`/checkpoint dir with mask_on, so verl auto-resumed it from
research/findings/M2/SUMMARY.md:54:   weakens the *live* val-parity sub-clause; confinement is unaffected (an auto-resume
research/findings/M2/SUMMARY.md:55:   cannot manufacture a false confinement pass) and the 1e-6 log-prob equality is
research/findings/M2/SUMMARY.md:59:   `val-core/openai/gsm8k/acc/mean@1`. Update plan templates.
research/findings/M2/EXP-5.md:7:- [x] p=0.95 and p=0.90 each reach `global_step=2` and execute ≥2 actor optimizer substeps (observed: both cells log `training/global_step:2`; `update_actor` logged at 2 trainer steps; `mask_applications` 14→28 per cell — the doubling across steps reflects boundary-mask ops over multiple substeps)
research/findings/M2/EXP-5.md:9:- [x] `comm_eff/mask_applications` >0 on actor-train, ==0 on rollout/old-logprob/ref-logprob/validation/checkpoint/infer_batch (observed: the ONLY mask-applications metric emitted is `actor/comm_eff/mask_applications` = 14/28 on masked cells, 0 on disabled; zero non-actor-path-keyed mask counters exist in either masked log)
research/findings/M2/EXP-5.md:13:- [x] `comm_eff.enabled=false` matches dense GRPO no-op / EXP-4 contract (observed: disabled cell all comm_eff counters 0 — mask_applications=0, anchor_backwards=0, spectral_corrections=0; grad_norm finite 1.13/0.37; reaches global_step=2; loss/grad evolve → ≥1 param changed)
research/findings/M2/EXP-5.md:19:- mask_applications (actor-train path): p95/p90 14→28; disabled 0→0
research/findings/M2/EXP-5.md:25:`diff_against_baseline.py runs/EXP-5 --baseline EXP-3` wrote runs/EXP-5/baseline_diff.md. Per the plan, this entry point is used here only to confirm the disabled cell still tracks the dense path (EXP-4 no-op contract regression), not as a performance comparison — p95/staleness/communication metrics are n/a for this actor-only masking integration smoke. The disabled cell reproduces dense GRPO: all comm_eff counters 0, finite low grad_norm (1.13, 0.37, consistent with an un-masked KL-regularized actor vs the masked cells' inflated grad_norm), and reaches step 2. No fork of the GRPO algorithm: same rollout → old-logprob → ref-logprob → reward → advantage → update_actor → weight-sync sequence in all three cells.
research/findings/M2/EXP-5.md:29:  (1) Mask-confinement grep was run on BOTH p95 AND p90 logs (not just p95). The plan's step-4 grep `comm_eff/mask_applications.*(rollout|log_prob|ref|val|infer|checkpoint)` produces FALSE POSITIVES because metrics are emitted as one physical line per step containing `timing_s/old_log_prob`, `timing_s/ref`, etc. downstream of the mask counter. The correct check extracts the metric KEY prefix: the only mask-applications metric in either masked log is `actor/comm_eff/mask_applications`; zero rollout/log_prob/ref/val/infer/checkpoint-keyed mask counters exist. Mask confinement HOLDS for both cells — the single most important falsifier did not fire.

codex
The local evidence matches the pasted summary, including the auto-resume caveat and the “fully independently established” wording. I’m checking the actual test/hook names next because the strength of the claim depends on what those tests really exercise.
exec
/bin/zsh -lc "rg --files tests verl | rg 'comm_eff|activation|mask|grpo|fsdp|rollout'" in /Users/shamane/Documents/verl
 succeeded in 0ms:
tests/trainer/ppo/test_rollout_corr_integration.py
tests/trainer/ppo/test_rollout_corr.py
verl/trainer/ppo/rollout_corr_helper.py
verl/trainer/distillation/fsdp/losses.py
verl/trainer/distillation/fsdp/__init__.py
verl/model_merger/fsdp_model_merger.py
verl/experimental/one_step_off_policy/shell/dapo_7b_math_fsdp2_64_64_ris.sh
verl/experimental/one_step_off_policy/shell/grpo_qwen3_8b_gsm8k_fsdp2_8_8_npu.sh
verl/experimental/one_step_off_policy/shell/dapo_7b_math_fsdp2_64_64.sh
verl/experimental/one_step_off_policy/shell/dapo_7b_math_fsdp2_sglang_colocate.sh
verl/experimental/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_2_6.sh
verl/experimental/one_step_off_policy/shell/dapo_7b_math_fsdp2_sglang_4_12.sh
verl/experimental/one_step_off_policy/shell/grpo_3b_gsm8k_fsdp2_2_6.sh
verl/experimental/one_step_off_policy/shell/dapo_7b_math_fsdp2_colocate.sh
verl/experimental/one_step_off_policy/shell/dapo_7b_math_fsdp2_4_12.sh
verl/experimental/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_sglang_2_6.sh
verl/trainer/config/rollout/rollout.yaml
verl/utils/checkpoint/fsdp_checkpoint_manager.py
verl/workers/config/rollout.py
verl/workers/config/comm_eff.py
verl/trainer/config/optim/fsdp.yaml
verl/trainer/config/algorithm/rollout_correction.yaml
verl/workers/comm_eff/state.py
verl/workers/comm_eff/activation_mask.py
verl/workers/comm_eff/__init__.py
verl/utils/fsdp_utils.py
verl/utils/rollout_trace.py
verl/utils/activation_offload.py
verl/workers/rollout/hf_rollout.py
verl/workers/rollout/llm_server.py
verl/workers/rollout/base.py
verl/workers/rollout/utils.py
verl/trainer/config/engine/fsdp.yaml
verl/utils/rollout_skip.py
tests/workers/rollout/rollout_vllm/test_vllm_abort.py
tests/workers/rollout/test_vllm_cli_args_on_cpu.py
verl/workers/engine/fsdp/utils.py
verl/workers/engine/fsdp/transformer_impl.py
verl/workers/engine/fsdp/__init__.py
tests/workers/rollout/perf/vllm_async_rollout.py
tests/workers/rollout/test_sglang_async_rollout_multimodal_delta.py
tests/workers/rollout/test_pd_disaggregation.py
verl/workers/rollout/tokenizer.py
verl/workers/rollout/schemas.py
verl/experimental/fully_async_policy/fully_async_rollouter.py
tests/special_npu/run_qwen3_06b_grpo_mindspeed.sh
tests/special_npu/run_qwen3_8b_grpo_mindspeedllm.sh
tests/special_npu/run_qwen3_30b_grpo_mindspeed.sh
verl/workers/rollout/naive/naive_rollout.py
verl/workers/rollout/naive/__init__.py
verl/workers/rollout/__init__.py
verl/workers/rollout/replica.py
tests/workers/rollout/rollout_trtllm/test_adapter.py
tests/workers/rollout/rollout_trtllm/test_inter_node_rollout.py
tests/workers/rollout/rollout_trtllm/__init__.py
tests/workers/rollout/rollout_trtllm/test_trtllm_abort.py
tests/workers/rollout/rollout_trtllm/test_trtllm_rollout_utils.py
tests/workers/rollout/rollout_trtllm/test_async_server.py
tests/workers/rollout/test_sglang_rollout_sharding_manager.py
verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py
verl/workers/rollout/vllm_rollout/vllm_async_server.py
verl/workers/rollout/vllm_rollout/utils.py
verl/workers/rollout/vllm_rollout/vllm_rollout.py
verl/workers/rollout/vllm_rollout/__init__.py
tests/special_npu/run_qwen3_30b_grpo_mindspeedllm.sh
verl/workers/rollout/sglang_rollout/sglang_rollout.py
verl/workers/rollout/sglang_rollout/utils.py
verl/workers/rollout/sglang_rollout/http_server_engine.py
verl/workers/rollout/sglang_rollout/sglang_pd_replica.py
verl/workers/rollout/sglang_rollout/__init__.py
verl/workers/rollout/sglang_rollout/async_sglang_server.py
verl/workers/rollout/trtllm_rollout/trtllm_async_rollout.md
verl/workers/rollout/trtllm_rollout/trtllm_async_server.py
verl/workers/rollout/trtllm_rollout/trtllm_rollout.py
verl/workers/rollout/trtllm_rollout/trtllm_worker_extension.py
verl/workers/rollout/trtllm_rollout/__init__.py
tests/special_npu/nightly_ci_ascend/run_grpo_qwen3_8b_mindspeedllm_npu.sh
tests/special_npu/nightly_ci_ascend/run_ppo_qwen3-8b_fsdp_npu.sh
tests/special_npu/run_qwen3_8b_grpo_profiling.sh
tests/models/test_fsdp_no_padding_on_gpu.py
tests/workers/rollout/rollout_sglang/test_lora_sleep_level.py
tests/workers/rollout/rollout_sglang/test_http_server_engine.py
tests/workers/rollout/test_hf_rollout.py
verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_64_64.sh
verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_8_8.sh
verl/experimental/fully_async_policy/shell/grpo_qwen35_35b_megatron_async.sh
verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_32_32.sh
verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_16_16.sh
verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_64_64_mis.sh
verl/experimental/fully_async_policy/shell/geo3k_qwen3vl_8b_fsdp2_16_16_npu.sh
verl/experimental/fully_async_policy/shell/dapo_30b_a3b_base_math_fsdp.sh
verl/experimental/fully_async_policy/shell/grpo_8b_base_math_megatron_4_4_trtllm.sh
verl/experimental/fully_async_policy/shell/grpo_30b_a3b_base_math_megatron_96_32.sh
verl/experimental/fully_async_policy/shell/geo3k_qwen3vl_30b_a3b_fsdp2_32_32.sh
verl/experimental/fully_async_policy/shell/grpo_qwen3_235b_megatron_npu.sh‎
verl/experimental/fully_async_policy/shell/grpo_30b_a3b_base_math_megatron_8_8_mis_trtllm.sh
verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh
verl/experimental/fully_async_policy/shell/dapo_30b_a3b_math_fsdp_npu.sh
verl/experimental/fully_async_policy/shell/grpo_30b_a3b_base_math_megatron_96_32_mis.sh
verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_12.sh
tests/experimental/agent_loop/test_standalone_rollout.py
tests/utils/test_fsdp_wrap_policy_on_cpu.py
tests/workers/comm_eff/test_activation_mask.py
tests/utils/test_fsdp_lora_merge.py
tests/utils/test_activation_offload.py
tests/utils/test_rollout_skip_on_cpu.py
tests/utils/test_fsdp2_peft_wrapping.py
tests/utils/test_rollout_trace_on_cpu.py
tests/workers/config/test_comm_eff_config.py
tests/special_distributed/test_fsdp_ckpt.py
tests/special_distributed/test_fsdp2_cpu_offload_state_dict.py

exec
/bin/zsh -lc 'rg -n "test_mask_hook_asserts_on_non_train_path|test_logprob_equal_mask_on_vs_off_when_tag_inactive|CommEff|comm_eff|path_tag|mask_applications|activation_mask|mask_hook" tests verl -S' in /Users/shamane/Documents/verl
 succeeded in 0ms:
verl/trainer/config/actor/actor.yaml:292:# The compressed path is selected ONLY by explicit comm_eff.* arguments.
verl/trainer/config/actor/actor.yaml:293:comm_eff:
verl/trainer/config/actor/actor.yaml:296:  _target_: verl.workers.config.CommEffConfig
verl/trainer/config/actor/actor.yaml:298:  # Master switch. false => every comm_eff hook is a no-op
verl/trainer/config/actor/actor.yaml:307:    _target_: verl.workers.config.CommEffMaskConfig
verl/trainer/config/actor/actor.yaml:309:    # Whether activation masking runs (still gated by the parent comm_eff.enabled)
verl/trainer/config/actor/actor.yaml:313:    # comm_eff/mask_ratio tracks this value. 0.0 means no masking.
verl/trainer/config/actor/actor.yaml:329:    _target_: verl.workers.config.CommEffAnchorConfig
verl/trainer/config/actor/actor.yaml:331:    # Whether the anchor circuit runs (gated by the parent comm_eff.enabled)
verl/trainer/config/actor/actor.yaml:344:    _target_: verl.workers.config.CommEffSpectralConfig
verl/trainer/config/actor/actor.yaml:346:    # Whether spectral correction runs (gated by the parent comm_eff.enabled)
verl/trainer/config/ppo_trainer.yaml:60:  # trainer/config/actor/actor.yaml (verl.workers.config.CommEffConfig) and is
verl/trainer/config/ppo_trainer.yaml:61:  # reachable as actor_rollout_ref.actor.comm_eff.* . Disabled by default
verl/trainer/config/ppo_trainer.yaml:62:  # (comm_eff.enabled=false) => the actor train path is a strict no-op: no
verl/trainer/config/ppo_trainer.yaml:65:  # actor_rollout_ref.actor.comm_eff.enabled=false works without a `+` prefix.
verl/workers/engine/base.py:112:    def _maybe_comm_eff_grad_correction(self) -> None:
verl/workers/engine/base.py:113:        """comm_eff spectral gradient-correction hook point (strict no-op when disabled).
verl/workers/engine/base.py:117:        gradients before ``optimizer_step``. When comm_eff is disabled (the
verl/workers/engine/base.py:118:        default: no ``_comm_eff_state`` is attached to the engine, or it is
verl/workers/engine/base.py:122:        attaches an enabled ``CommEffState`` to the engine.
verl/workers/engine/base.py:124:        state = getattr(self, "_comm_eff_state", None)
verl/workers/engine/base.py:145:        # comm_eff spectral gradient correction (no-op when disabled) runs after
verl/workers/engine/base.py:148:        self._maybe_comm_eff_grad_correction()
verl/workers/engine/fsdp/transformer_impl.py:611:    def _comm_eff_mask_active(self, forward_only: bool) -> bool:
verl/workers/engine/fsdp/transformer_impl.py:618:          * an enabled ``CommEffState`` is attached,
verl/workers/engine/fsdp/transformer_impl.py:625:        state = getattr(self, "_comm_eff_state", None)
verl/workers/engine/fsdp/transformer_impl.py:632:    def _comm_eff_register_mask_hooks(self) -> bool:
verl/workers/engine/fsdp/transformer_impl.py:642:        state = self._comm_eff_state
verl/workers/engine/fsdp/transformer_impl.py:645:        global_step = int(getattr(self, "_comm_eff_global_step", 0))
verl/workers/engine/fsdp/transformer_impl.py:660:        # comm_eff activation-mask hook lifecycle: register hooks on entry to the
verl/workers/engine/fsdp/transformer_impl.py:665:        _mask_hooks_live = False
verl/workers/engine/fsdp/transformer_impl.py:666:        if self._comm_eff_mask_active(forward_only=forward_only):
verl/workers/engine/fsdp/transformer_impl.py:667:            _mask_hooks_live = self._comm_eff_register_mask_hooks()
verl/workers/engine/fsdp/transformer_impl.py:671:            if _mask_hooks_live:
verl/workers/engine/fsdp/transformer_impl.py:672:                self._comm_eff_state.masker.unregister()
verl/workers/comm_eff/state.py:20:* ``maybe_build_comm_eff_state(config)`` returns ``None`` when
verl/workers/comm_eff/state.py:23:  registered. The actor therefore holds ``self._comm_eff_state = None`` for a
verl/workers/comm_eff/state.py:26:* Only when ``config.enabled`` is true is a ``CommEffState`` constructed; that
verl/workers/comm_eff/state.py:34:The instrumented counters (``mask_applications``, ``anchor_backwards``,
verl/workers/comm_eff/state.py:37:analyst treats as equivalent to ``== 0`` (no comm_eff op fired). When enabled
verl/workers/comm_eff/state.py:47:    from verl.workers.config.comm_eff import CommEffConfig
verl/workers/comm_eff/state.py:51:__all__ = ["CommEffState", "maybe_build_comm_eff_state", "comm_eff_metrics"]
verl/workers/comm_eff/state.py:55:    """Read the ``enabled`` flag from a comm_eff config that may be a dataclass,
verl/workers/comm_eff/state.py:64:class CommEffState:
verl/workers/comm_eff/state.py:67:    Constructed **only** when ``comm_eff.enabled=true``. Holds the operation
verl/workers/comm_eff/state.py:70:    never instantiates this class — see ``maybe_build_comm_eff_state``.
verl/workers/comm_eff/state.py:73:    def __init__(self, config: "CommEffConfig"):
verl/workers/comm_eff/state.py:77:            "CommEffState must not be constructed when comm_eff.enabled=false; "
verl/workers/comm_eff/state.py:78:            "go through maybe_build_comm_eff_state() so the disabled path stays a no-op."
verl/workers/comm_eff/state.py:84:        # Operation counters surfaced into training metrics under comm_eff/*.
verl/workers/comm_eff/state.py:85:        self.mask_applications = 0
verl/workers/comm_eff/state.py:107:        Idempotent. When ``comm_eff.mask.enabled`` is true this constructs an
verl/workers/comm_eff/state.py:118:            from verl.workers.comm_eff.activation_mask import ActivationMasker
verl/workers/comm_eff/state.py:131:        Surfaced as ``comm_eff/mask_ratio`` (mean across boundaries) plus a
verl/workers/comm_eff/state.py:138:        out = {"comm_eff/mask_ratio": mean_ratio}
verl/workers/comm_eff/state.py:140:            out[f"comm_eff/mask_ratio/layer_{idx}"] = r
verl/workers/comm_eff/state.py:144:        """Return the comm_eff operation counters for logging."""
verl/workers/comm_eff/state.py:146:            "comm_eff/mask_applications": self.mask_applications,
verl/workers/comm_eff/state.py:147:            "comm_eff/anchor_backwards": self.anchor_backwards,
verl/workers/comm_eff/state.py:148:            "comm_eff/spectral_corrections": self.spectral_corrections,
verl/workers/comm_eff/state.py:152:def maybe_build_comm_eff_state(config: Any) -> Optional[CommEffState]:
verl/workers/comm_eff/state.py:153:    """Construct a ``CommEffState`` iff comm_eff is enabled, else return ``None``.
verl/workers/comm_eff/state.py:158:    Callers store the result and guard every comm_eff op behind a ``None`` /
verl/workers/comm_eff/state.py:163:    state = CommEffState(config)
verl/workers/comm_eff/state.py:164:    logger.info("comm_eff: enabled — constructed CommEffState")
verl/workers/comm_eff/state.py:168:def comm_eff_metrics(state: Optional[CommEffState]) -> dict:
verl/workers/comm_eff/state.py:169:    """Return comm_eff counters for ``state``, or an empty dict when disabled.
verl/workers/comm_eff/state.py:172:    not each re-derive it. Includes the measured ``comm_eff/mask_ratio`` when a
verl/workers/comm_eff/activation_mask.py:23:``tests/workers/comm_eff/test_activation_mask.py``):
verl/workers/comm_eff/activation_mask.py:33:  matches the EXP-5 success criterion ``comm_eff/mask_ratio ≈ p ± 0.02``.
verl/workers/comm_eff/activation_mask.py:241:      * ``base_seed`` (``comm_eff.mask.seed``).
verl/workers/comm_eff/activation_mask.py:252:        self._state = state  # CommEffState, for the mask_applications counter
verl/workers/comm_eff/activation_mask.py:260:        # Last-measured masked fraction per boundary, surfaced as comm_eff/mask_ratio.
verl/workers/comm_eff/activation_mask.py:298:                masker._state.mask_applications += 1
verl/workers/comm_eff/activation_mask.py:317:                "comm_eff.activation_mask: could not locate decoder layers on %s; "
verl/workers/comm_eff/activation_mask.py:329:            "comm_eff.activation_mask: registered mask hooks on boundaries %s "
verl/workers/config/actor.py:25:from .comm_eff import CommEffConfig
verl/workers/config/actor.py:139:        comm_eff (CommEffConfig): Communication-efficient compression config. Disabled by
verl/workers/config/actor.py:191:    comm_eff: CommEffConfig = field(default_factory=CommEffConfig)
tests/workers/config/test_comm_eff_config.py:15:"""Unit tests for the comm_eff config group (EXP-4 M2 no-op scaffolding).
tests/workers/config/test_comm_eff_config.py:18:  1. comm_eff defaults to DISABLED (enabled=false) at every level — the
tests/workers/config/test_comm_eff_config.py:20:  2. The structured schema REJECTS unknown comm_eff.* keys (typos fail fast)
tests/workers/config/test_comm_eff_config.py:31:from verl.workers.comm_eff import maybe_build_comm_eff_state
tests/workers/config/test_comm_eff_config.py:34:    CommEffAnchorConfig,
tests/workers/config/test_comm_eff_config.py:35:    CommEffConfig,
tests/workers/config/test_comm_eff_config.py:36:    CommEffMaskConfig,
tests/workers/config/test_comm_eff_config.py:37:    CommEffSpectralConfig,
tests/workers/config/test_comm_eff_config.py:42:class TestCommEffConfigDefaults(unittest.TestCase):
tests/workers/config/test_comm_eff_config.py:43:    """comm_eff must default to disabled everywhere."""
tests/workers/config/test_comm_eff_config.py:46:        """A bare CommEffConfig is disabled and all circuits are off."""
tests/workers/config/test_comm_eff_config.py:47:        cfg = CommEffConfig()
tests/workers/config/test_comm_eff_config.py:51:        self.assertIsInstance(cfg.mask, CommEffMaskConfig)
tests/workers/config/test_comm_eff_config.py:52:        self.assertIsInstance(cfg.anchor, CommEffAnchorConfig)
tests/workers/config/test_comm_eff_config.py:53:        self.assertIsInstance(cfg.spectral, CommEffSpectralConfig)
tests/workers/config/test_comm_eff_config.py:55:    def test_actor_config_carries_disabled_comm_eff_by_default(self):
tests/workers/config/test_comm_eff_config.py:56:        """ActorConfig wires comm_eff and defaults it disabled."""
tests/workers/config/test_comm_eff_config.py:63:        self.assertIsInstance(config.comm_eff, CommEffConfig)
tests/workers/config/test_comm_eff_config.py:64:        self.assertFalse(config.comm_eff.enabled)
tests/workers/config/test_comm_eff_config.py:69:            {"_target_": "verl.workers.config.CommEffConfig"},
tests/workers/config/test_comm_eff_config.py:70:            dataclass_type=CommEffConfig,
tests/workers/config/test_comm_eff_config.py:72:        self.assertIsInstance(cfg, CommEffConfig)
tests/workers/config/test_comm_eff_config.py:76:        """enabled is honored both ways; this is the Run-A `comm_eff.enabled=false`
tests/workers/config/test_comm_eff_config.py:78:        disabled = omega_conf_to_dataclass({"enabled": False}, dataclass_type=CommEffConfig)
tests/workers/config/test_comm_eff_config.py:80:        enabled = omega_conf_to_dataclass({"enabled": True}, dataclass_type=CommEffConfig)
tests/workers/config/test_comm_eff_config.py:84:        """The composed actor YAML defaults comm_eff disabled (registered key,
tests/workers/config/test_comm_eff_config.py:85:        reachable as actor_rollout_ref.actor.comm_eff.*)."""
tests/workers/config/test_comm_eff_config.py:92:        self.assertIsInstance(config.comm_eff, CommEffConfig)
tests/workers/config/test_comm_eff_config.py:93:        self.assertFalse(config.comm_eff.enabled)
tests/workers/config/test_comm_eff_config.py:96:        """The plain (no `+`) override comm_eff.enabled=false composes — i.e. the
tests/workers/config/test_comm_eff_config.py:106:                    "comm_eff.enabled=false",
tests/workers/config/test_comm_eff_config.py:110:        self.assertFalse(config.comm_eff.enabled)
tests/workers/config/test_comm_eff_config.py:113:class TestCommEffConfigSchema(unittest.TestCase):
tests/workers/config/test_comm_eff_config.py:114:    """The structured schema must reject unknown comm_eff.* keys."""
tests/workers/config/test_comm_eff_config.py:117:        """An unknown comm_eff key (typo) must raise, not be silently dropped."""
tests/workers/config/test_comm_eff_config.py:121:                dataclass_type=CommEffConfig,
tests/workers/config/test_comm_eff_config.py:125:        """An unknown key under comm_eff.mask must raise."""
tests/workers/config/test_comm_eff_config.py:129:                dataclass_type=CommEffConfig,
tests/workers/config/test_comm_eff_config.py:133:        """An unknown key under comm_eff.spectral must raise."""
tests/workers/config/test_comm_eff_config.py:137:                dataclass_type=CommEffConfig,
tests/workers/config/test_comm_eff_config.py:143:            CommEffConfig(mask=CommEffMaskConfig(p=1.5))
tests/workers/config/test_comm_eff_config.py:145:            CommEffConfig(spectral=CommEffSpectralConfig(rank=0))
tests/workers/config/test_comm_eff_config.py:147:            CommEffConfig(anchor=CommEffAnchorConfig(ema_decay=2.0))
tests/workers/config/test_comm_eff_config.py:150:class TestCommEffStateInert(unittest.TestCase):
tests/workers/config/test_comm_eff_config.py:154:        """maybe_build_comm_eff_state returns None when disabled — no object,
tests/workers/config/test_comm_eff_config.py:156:        self.assertIsNone(maybe_build_comm_eff_state(CommEffConfig()))
tests/workers/config/test_comm_eff_config.py:157:        self.assertIsNone(maybe_build_comm_eff_state(CommEffConfig(enabled=False)))
tests/workers/config/test_comm_eff_config.py:158:        self.assertIsNone(maybe_build_comm_eff_state(None))
tests/workers/config/test_comm_eff_config.py:159:        self.assertIsNone(maybe_build_comm_eff_state({"enabled": False}))
tests/workers/config/test_comm_eff_config.py:163:        state = maybe_build_comm_eff_state(CommEffConfig(enabled=True))
tests/workers/config/test_comm_eff_config.py:167:        self.assertEqual(m["comm_eff/mask_applications"], 0)
tests/workers/config/test_comm_eff_config.py:168:        self.assertEqual(m["comm_eff/anchor_backwards"], 0)
tests/workers/config/test_comm_eff_config.py:169:        self.assertEqual(m["comm_eff/spectral_corrections"], 0)
tests/workers/config/test_comm_eff_config.py:172:        """comm_eff_metrics(None) is empty (disabled => counters absent)."""
tests/workers/config/test_comm_eff_config.py:173:        from verl.workers.comm_eff.state import comm_eff_metrics
tests/workers/config/test_comm_eff_config.py:175:        self.assertEqual(comm_eff_metrics(None), {})
verl/workers/comm_eff/__init__.py:19:``verl.workers.config.comm_eff.CommEffConfig``.
verl/workers/comm_eff/__init__.py:23:lazily by ``CommEffState.maybe_build`` and run **only** when
verl/workers/comm_eff/__init__.py:24:``comm_eff.enabled=true``; the disabled path never reaches them.
verl/workers/comm_eff/__init__.py:27:from .state import CommEffState, comm_eff_metrics, maybe_build_comm_eff_state
verl/workers/comm_eff/__init__.py:29:__all__ = ["CommEffState", "comm_eff_metrics", "maybe_build_comm_eff_state"]
verl/workers/config/__init__.py:15:from . import actor, comm_eff, critic, disaggregation, engine, model, optimizer, reward, rollout
verl/workers/config/__init__.py:17:from .comm_eff import *  # noqa: F401
verl/workers/config/__init__.py:29:    + comm_eff.__all__
verl/workers/engine_workers.py:46:from verl.workers.comm_eff import maybe_build_comm_eff_state
verl/workers/engine_workers.py:47:from verl.workers.comm_eff.state import comm_eff_metrics
verl/workers/engine_workers.py:643:    def _maybe_comm_eff_state(self):
verl/workers/engine_workers.py:644:        """Return this worker's comm_eff state, building it once on first use.
verl/workers/engine_workers.py:646:        Disabled is the strict no-op path: ``maybe_build_comm_eff_state`` returns
verl/workers/engine_workers.py:652:        state = getattr(self, "_comm_eff_state", None)
verl/workers/engine_workers.py:653:        if state is None and not getattr(self, "_comm_eff_state_built", False):
verl/workers/engine_workers.py:654:            comm_eff_cfg = self.config.actor.get("comm_eff", None)
verl/workers/engine_workers.py:655:            state = maybe_build_comm_eff_state(comm_eff_cfg)
verl/workers/engine_workers.py:658:            object.__setattr__(self, "_comm_eff_state", state)
verl/workers/engine_workers.py:659:            object.__setattr__(self, "_comm_eff_state_built", True)
verl/workers/engine_workers.py:660:            if state is None and not getattr(self, "_comm_eff_marker_logged", False):
verl/workers/engine_workers.py:661:                logger.info("comm_eff: disabled (no-op) — dense GRPO path unchanged")
verl/workers/engine_workers.py:662:                object.__setattr__(self, "_comm_eff_marker_logged", True)
verl/workers/engine_workers.py:673:                    object.__setattr__(engine, "_comm_eff_state", state)
verl/workers/engine_workers.py:674:                    logger.info("comm_eff: enabled — mask circuit attached to actor train engine")
verl/workers/engine_workers.py:675:        return getattr(self, "_comm_eff_state", None)
verl/workers/engine_workers.py:681:        # comm_eff guard. When disabled (default) this resolves to None with zero
verl/workers/engine_workers.py:684:        # comm_eff.enabled=true (later M2 work); the disabled path never touches
verl/workers/engine_workers.py:686:        comm_eff_state = self._maybe_comm_eff_state()
verl/workers/engine_workers.py:693:        if comm_eff_state is not None:
verl/workers/engine_workers.py:694:            comm_eff_state.mask_active = True
verl/workers/engine_workers.py:698:            if comm_eff_state is not None:
verl/workers/engine_workers.py:699:                comm_eff_state.mask_active = False
verl/workers/engine_workers.py:701:        # Surface the comm_eff operation counters into training metrics. When
verl/workers/engine_workers.py:702:        # disabled we emit explicit zeros (mask_applications / anchor_backwards /
verl/workers/engine_workers.py:708:            if comm_eff_state is None:
verl/workers/engine_workers.py:710:                    "comm_eff/mask_applications": 0,
verl/workers/engine_workers.py:711:                    "comm_eff/anchor_backwards": 0,
verl/workers/engine_workers.py:712:                    "comm_eff/spectral_corrections": 0,
verl/workers/engine_workers.py:715:                counters = comm_eff_metrics(comm_eff_state)
verl/workers/config/comm_eff.py:17:This module defines the ``comm_eff`` config group: the two-circuit compression
verl/workers/config/comm_eff.py:21:(see ``verl.workers.comm_eff.state`` and the guards in ``engine_workers``,
verl/workers/config/comm_eff.py:25:Hydra schema is validated up front (typos in ``comm_eff.mask.*`` etc. are
verl/workers/config/comm_eff.py:36:    "CommEffMaskConfig",
verl/workers/config/comm_eff.py:37:    "CommEffAnchorConfig",
verl/workers/config/comm_eff.py:38:    "CommEffSpectralConfig",
verl/workers/config/comm_eff.py:39:    "CommEffConfig",
verl/workers/config/comm_eff.py:44:class CommEffMaskConfig(BaseConfig):
verl/workers/config/comm_eff.py:45:    """Pipeline activation-masking sub-config (inert while ``comm_eff.enabled=false``).
verl/workers/config/comm_eff.py:50:    forward/backward. See ``verl.workers.comm_eff.activation_mask``.
verl/workers/config/comm_eff.py:54:            ``comm_eff.enabled`` regardless of this value (so the disabled path
verl/workers/config/comm_eff.py:56:            ``comm_eff.enabled=true`` activates masking without a second flag;
verl/workers/config/comm_eff.py:59:            element is zeroed (``mask=0``). The measured ``comm_eff/mask_ratio``
verl/workers/config/comm_eff.py:61:            ``comm_eff.enabled=true``.
verl/workers/config/comm_eff.py:63:            ranks and re-runs. Only drawn from when ``comm_eff.enabled=true`` —
verl/workers/config/comm_eff.py:81:class CommEffAnchorConfig(BaseConfig):
verl/workers/config/comm_eff.py:86:            ``comm_eff.enabled`` regardless of this value.
verl/workers/config/comm_eff.py:97:class CommEffSpectralConfig(BaseConfig):
verl/workers/config/comm_eff.py:102:            Gated by the parent ``comm_eff.enabled`` regardless of this value.
verl/workers/config/comm_eff.py:113:class CommEffConfig(BaseConfig):
verl/workers/config/comm_eff.py:118:    OmegaConf rejects unknown ``comm_eff.*`` keys at merge time (typos in the
verl/workers/config/comm_eff.py:123:    hooks short-circuit before importing any comm_eff machinery: no forward
verl/workers/config/comm_eff.py:131:        enabled (bool): Master switch. ``false`` (default) makes every comm_eff
verl/workers/config/comm_eff.py:134:        mask (CommEffMaskConfig): Pipeline activation-masking sub-config.
verl/workers/config/comm_eff.py:135:        anchor (CommEffAnchorConfig): Asynchronous anchor-circuit sub-config.
verl/workers/config/comm_eff.py:136:        spectral (CommEffSpectralConfig): Spectral-correction sub-config.
verl/workers/config/comm_eff.py:140:    mask: CommEffMaskConfig = field(default_factory=CommEffMaskConfig)
verl/workers/config/comm_eff.py:141:    anchor: CommEffAnchorConfig = field(default_factory=CommEffAnchorConfig)
verl/workers/config/comm_eff.py:142:    spectral: CommEffSpectralConfig = field(default_factory=CommEffSpectralConfig)
verl/workers/config/comm_eff.py:145:        """Validate comm_eff configuration parameters.
verl/workers/config/comm_eff.py:151:            raise ValueError(f"comm_eff.mask.p must be in [0, 1]; got {self.mask.p}")
verl/workers/config/comm_eff.py:153:            raise ValueError(f"comm_eff.mask.pp_size must be >= 1; got {self.mask.pp_size}")
verl/workers/config/comm_eff.py:155:            raise ValueError(f"comm_eff.spectral.rank must be >= 1; got {self.spectral.rank}")
verl/workers/config/comm_eff.py:157:            raise ValueError(f"comm_eff.anchor.ema_decay must be in [0, 1]; got {self.anchor.ema_decay}")
tests/workers/comm_eff/test_activation_mask.py:15:"""Unit tests for the comm_eff pipeline-boundary activation masker (EXP-5).
tests/workers/comm_eff/test_activation_mask.py:34:from verl.workers.comm_eff.activation_mask import (
tests/workers/comm_eff/test_activation_mask.py:245:def test_mask_applications_counter_increments():
tests/workers/comm_eff/test_activation_mask.py:246:    """When a CommEffState is attached, each hook fire bumps mask_applications."""
tests/workers/comm_eff/test_activation_mask.py:250:            self.mask_applications = 0
tests/workers/comm_eff/test_activation_mask.py:258:    assert state.mask_applications == 2

exec
/bin/zsh -lc 'git status --short && git branch --show-current && git log --oneline -5' in /Users/shamane/Documents/verl
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-DfRVIEZM' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-5odpFllz' (errno=Operation not permitted)
 M research/.claude/state/.last-sync
 M research/PROGRESS.md
 M research/runs/SUMMARY.md
?? .claude/worktrees/
?? research/findings/M2/SUMMARY.md
?? research/findings/M2/codex-review.md.partial
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-axyuN05x' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-HtQ1NA3t' (errno=Operation not permitted)
vast-ai-workload
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-AS88ViuV' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-TustBQua' (errno=Operation not permitted)
5b40bcf0 [autosave] research session eef4fb8b-7306-474d-9ce9-222433f63943 stop
060a0cf9 [autosave] research session eef4fb8b-7306-474d-9ce9-222433f63943 stop
cbaa68b8 [autosave] research session bb474a32-5b64-4120-b2df-6bd290929cf8 stop
fee8bfa9 [autosave] research session bb474a32-5b64-4120-b2df-6bd290929cf8 stop
b86cadbe [autosave] research session 62dc2878-e4f8-4fd4-867b-40d83b32ab5a stop

exec
/bin/zsh -lc "sed -n '220,330p' verl/workers/comm_eff/activation_mask.py" in /Users/shamane/Documents/verl
exec
/bin/zsh -lc "sed -n '600,720p' verl/workers/engine/fsdp/transformer_impl.py" in /Users/shamane/Documents/verl
 succeeded in 0ms:
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def get_model_parallel_group(self):
        raise NotImplementedError

    def get_context_parallel_group(self):
        raise NotImplementedError

    def _comm_eff_mask_active(self, forward_only: bool) -> bool:
        """True iff the activation-mask hooks should be live for this forward.

        Masking is confined to the actor-train forward/backward. This returns
        False (strict no-op) unless ALL of:
          * this is a train pass (``not forward_only`` — never on infer_batch /
            log-prob / ref / validation),
          * an enabled ``CommEffState`` is attached,
          * the worker has set ``state.mask_active`` (set only around
            ``update_actor``; cleared everywhere else),
          * a masker was constructed (mask sub-config enabled, ``p > 0``).
        """
        if forward_only:
            return False
        state = getattr(self, "_comm_eff_state", None)
        if state is None or not getattr(state, "enabled", False):
            return False
        if not getattr(state, "mask_active", False):
            return False
        return getattr(state, "masker", None) is not None

    def _comm_eff_register_mask_hooks(self) -> bool:
        """Register the activation-mask forward hooks for this train forward.

        Sets the PRF-key context (global step / optimizer-substep identity /
        sequence-shard id) and installs the hooks on the boundary decoder
        blocks. Returns True if hooks were registered (so the caller knows to
        unregister on exit). The substep counter advances per call so the same
        rollout batch reused across PPO mini-batches gets a distinct mask per
        substep.
        """
        state = self._comm_eff_state
        masker = state.masker
        # global optimizer step (best-effort; threaded by the trainer when set).
        global_step = int(getattr(self, "_comm_eff_global_step", 0))
        # sequence-shard identity when Ulysses SP is active (else 0).
        seq_shard = 0
        if getattr(self, "ulysses_sequence_parallel_size", 1) and self.ulysses_sequence_parallel_size > 1:
            try:
                seq_shard = self.get_data_parallel_rank()
            except Exception:
                seq_shard = 0
        masker.set_context(global_step=global_step, substep=state.substep, seq_shard=seq_shard)
        masker.register(self.module)
        # Advance the optimizer-substep identity for the next train forward.
        state.substep += 1
        return masker.is_registered

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
        # comm_eff activation-mask hook lifecycle: register hooks on entry to the
        # train forward/backward and remove them on exit, so a later log-prob /
        # infer / ref / validation forward on the same module is clean. When
        # disabled (default) or not on the actor-train path, nothing is registered
        # and no RNG is drawn, so the pass is byte-identical to dense GRPO.
        _mask_hooks_live = False
        if self._comm_eff_mask_active(forward_only=forward_only):
            _mask_hooks_live = self._comm_eff_register_mask_hooks()
        try:
            return self._forward_backward_batch_inner(data, loss_function, forward_only=forward_only)
        finally:
            if _mask_hooks_live:
                self._comm_eff_state.masker.unregister()

    def _forward_backward_batch_inner(
        self, data: TensorDict, loss_function: Callable, forward_only=False
    ) -> list[TensorDict]:
        # note that the global_batch_size should include data on all the dp
        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        # getattr fallback: some subclasses (e.g. VeOmniEngine) bypass FSDPEngine.__init__
        # and _build_fsdp_module, so self.scaler may not be set.
        scaler = getattr(self, "scaler", None)

        for micro_batch in micro_batches:
            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)

                if not forward_only:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            output_lst.append(meta_info)

        # postprocess and return
        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """
        Zero gradients and enforce FSDP grad-clipping logic.

 succeeded in 0ms:
    u = torch.rand(shape, generator=gen, dtype=torch.float32)
    keep = (u >= p).to(dtype=dtype)
    return keep.to(device=device, non_blocking=True)


class ActivationMasker:
    """Registers/clears in-graph activation-mask forward hooks on boundary blocks.

    One instance is owned by the engine. ``register(module)`` installs a forward
    hook on each boundary decoder block; ``unregister()`` removes them. The hooks
    must be live **only** during the actor train forward/backward — the engine
    registers on entry to ``forward_backward_batch`` (train) and unregisters on
    exit, so log-prob / ref / infer / validation / checkpoint forwards never see
    a mask.

    The PRF key per hook fire is composed from:
      * the boundary block index (stable per hook),
      * ``global_step`` (trainer optimizer step),
      * ``substep`` (optimizer-substep / microbatch identity within the step),
      * a sequence-shard id (0 when no SP; set by the engine when present),
      * ``hidden_size`` (last dim of the activation),
      * ``base_seed`` (``comm_eff.mask.seed``).

    ``global_step`` / ``substep`` / ``seq_shard`` are set by the engine via
    ``set_context(...)`` before each forward so the same rollout batch reused
    over multiple PPO mini-batches gets distinct masks per substep.
    """

    def __init__(self, *, p: float, base_seed: int, pp_size: int, state: Any = None):
        self.p = float(p)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        self._state = state  # CommEffState, for the mask_applications counter
        self._handles: list[Any] = []
        self._boundary_set: set[int] = set()
        self.boundary_indices: list[int] = []
        # Per-forward context, set by the engine before forward_backward.
        self._global_step = 0
        self._substep = 0
        self._seq_shard = 0
        # Last-measured masked fraction per boundary, surfaced as comm_eff/mask_ratio.
        self.last_mask_ratio: dict[int, float] = {}

    def set_context(self, *, global_step: int, substep: int, seq_shard: int = 0) -> None:
        """Set the PRF-key context for the next forward pass."""
        self._global_step = int(global_step)
        self._substep = int(substep)
        self._seq_shard = int(seq_shard)

    def _make_hook(self, layer_idx: int):
        masker = self

        def _hook(_mod: nn.Module, _inputs: tuple, output: Any):
            # HF decoder blocks return either a Tensor or a tuple whose first
            # element is the hidden state. Mask the hidden state in-graph.
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if not torch.is_tensor(h):
                return output
            hidden_size = h.shape[-1]
            key = (
                layer_idx,
                masker._global_step,
                masker._substep,
                masker._seq_shard,
                hidden_size,
                masker.base_seed,
            )
            mask = prf_mask(tuple(h.shape), key, masker.p, device=h.device, dtype=h.dtype)
            # h_tilde = h * mask, in-graph (no 1/(1-p) rescale). The multiply is
            # tracked by autograd so the masked gradient flows to the optimizer.
            h_tilde = h * mask
            # Instrumentation (does not affect the graph): measured masked fraction.
            with torch.no_grad():
                masker.last_mask_ratio[layer_idx] = float(1.0 - mask.mean().item())
            if masker._state is not None:
                masker._state.mask_applications += 1
            if isinstance(output, tuple):
                return (h_tilde,) + tuple(output[1:])
            return h_tilde

        return _hook

    def register(self, module: nn.Module) -> None:
        """Install forward hooks on the boundary decoder blocks of ``module``.

        Idempotent guard: if hooks are already registered this is a no-op (the
        engine pairs register/unregister, but a defensive guard avoids double
        registration leaking a mask onto a later pass).
        """
        if self._handles:
            return
        layers = find_decoder_layers(module)
        if layers is None:
            logger.warning(
                "comm_eff.activation_mask: could not locate decoder layers on %s; "
                "no mask hooks registered (masking is a no-op this pass)",
                type(module).__name__,
            )
            return
        num_layers = len(layers)
        self.boundary_indices = decoder_boundary_indices(num_layers, self.pp_size)
        self._boundary_set = set(self.boundary_indices)
        for idx in self.boundary_indices:
            handle = layers[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)
        logger.info(
            "comm_eff.activation_mask: registered mask hooks on boundaries %s "
            "(L=%d, pp_size=%d, p=%.4f)",

exec
/bin/zsh -lc "sed -n '620,725p' verl/workers/engine_workers.py" in /Users/shamane/Documents/verl
exec
/bin/zsh -lc "sed -n '1,340p' tests/workers/comm_eff/test_activation_mask.py" in /Users/shamane/Documents/verl
 succeeded in 0ms:
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None

    def _maybe_comm_eff_state(self):
        """Return this worker's comm_eff state, building it once on first use.

        Disabled is the strict no-op path: ``maybe_build_comm_eff_state`` returns
        ``None`` without drawing RNG, allocating buffers or registering hooks, so
        a dense GRPO run with the scaffolding merged is numerically identical to
        one without it. The result is cached so the per-substep ``update_actor``
        does not re-read the config each call.
        """
        state = getattr(self, "_comm_eff_state", None)
        if state is None and not getattr(self, "_comm_eff_state_built", False):
            comm_eff_cfg = self.config.actor.get("comm_eff", None)
            state = maybe_build_comm_eff_state(comm_eff_cfg)
            # object.__setattr__ avoids any frozen-config interplay; these are
            # plain worker attributes, not config fields.
            object.__setattr__(self, "_comm_eff_state", state)
            object.__setattr__(self, "_comm_eff_state_built", True)
            if state is None and not getattr(self, "_comm_eff_marker_logged", False):
                logger.info("comm_eff: disabled (no-op) — dense GRPO path unchanged")
                object.__setattr__(self, "_comm_eff_marker_logged", True)
            if state is not None:
                # Construct the masker (no hooks yet — the engine registers them
                # only inside the train forward/backward) and attach the state to
                # the underlying train engine so its forward-hook lifecycle and
                # grad-correction hook can see it. The state is the single object
                # shared between the worker (sets mask_active around update_actor)
                # and the engine (registers/clears hooks gated on mask_active).
                engine = getattr(getattr(self, "actor", None), "engine", None)
                if engine is not None:
                    state.build(getattr(engine, "module", None))
                    object.__setattr__(engine, "_comm_eff_state", state)
                    logger.info("comm_eff: enabled — mask circuit attached to actor train engine")
        return getattr(self, "_comm_eff_state", None)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        # comm_eff guard. When disabled (default) this resolves to None with zero
        # side effects (no hook, no buffer, no RNG) and the dense GRPO update runs
        # exactly as upstream. The compressed circuits are entered only when
        # comm_eff.enabled=true (later M2 work); the disabled path never touches
        # the gradient, so the no-op parity holds.
        comm_eff_state = self._maybe_comm_eff_state()

        # Mask-active flag scope: set ONLY around the actor-train forward/backward
        # so the masking forward-hooks fire exclusively on this path. The engine
        # registers hooks on entry to its train forward_backward_batch and removes
        # them on exit, gated on this flag; log_prob / infer / ref / validation /
        # checkpoint forwards never set it, so they stay byte-identical to dense.
        if comm_eff_state is not None:
            comm_eff_state.mask_active = True
        try:
            output = self.actor.train_mini_batch(data=data)
        finally:
            if comm_eff_state is not None:
                comm_eff_state.mask_active = False

        # Surface the comm_eff operation counters into training metrics. When
        # disabled we emit explicit zeros (mask_applications / anchor_backwards /
        # spectral_corrections == 0) so the no-op is machine-checkable; emitting a
        # constant metric is not a numerical side effect on training. `output` is
        # None on non-output ranks (train_mini_batch only populates metrics on the
        # mp-src rank), in which case there is nothing to annotate.
        if output is not None:
            if comm_eff_state is None:
                counters = {
                    "comm_eff/mask_applications": 0,
                    "comm_eff/anchor_backwards": 0,
                    "comm_eff/spectral_corrections": 0,
                }
            else:
                counters = comm_eff_metrics(comm_eff_state)
            metrics = tu.get(output, "metrics", default=None)
            if isinstance(metrics, dict):
                metrics.update(counters)

        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

 succeeded in 0ms:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the comm_eff pipeline-boundary activation masker (EXP-5).

These tests cover the masking-correctness properties the EXP-5 plan requires
codex-verify to gate on, none of which need a GPU:

* boundary indices == [1,3,5,7,9,11,13] for L=16 / pp_size=8, derived (not hardcoded);
* PRF determinism: same key -> same mask, across calls;
* value-independence: the mask depends only on the PRF key + shape, never on
  the activation values;
* measured mask ratio (zeroed fraction) tracks the configured p within tolerance;
* in-graph form h_tilde = h * mask with NO 1/(1-p) rescale;
* hook lifecycle: register installs hooks on boundaries only; unregister removes
  them so a later forward is clean.
"""

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import (
    ActivationMasker,
    decoder_boundary_indices,
    find_decoder_layers,
    prf_mask,
)


# --------------------------------------------------------------------------- #
# boundary partition
# --------------------------------------------------------------------------- #
def test_boundary_indices_L16_pp8():
    """The spec's canonical example: L=16 / pp_size=8 -> [1,3,5,7,9,11,13]."""
    assert decoder_boundary_indices(16, 8) == [1, 3, 5, 7, 9, 11, 13]


def test_boundary_indices_excludes_final_shard():
    """The final shard's last block (the model's last decoder block) is never masked."""
    idx = decoder_boundary_indices(16, 8)
    assert 15 not in idx  # final block excluded
    assert len(idx) == 7  # pp_size - 1 boundaries


def test_boundary_indices_uneven_partition():
    """Uneven L/pp_size: shards are near-even, larger shards come first."""
    # L=10, pp_size=4 -> shard lens [3,3,2,2] -> last idx [2,5,7,9] -> drop 9 -> [2,5,7]
    assert decoder_boundary_indices(10, 4) == [2, 5, 7]


def test_boundary_indices_pp_size_one_is_empty():
    assert decoder_boundary_indices(16, 1) == []


def test_boundary_indices_pp_capped_at_num_layers():
    # pp_size > L collapses to one block per shard; last shard dropped.
    assert decoder_boundary_indices(4, 8) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# PRF determinism + value-independence
# --------------------------------------------------------------------------- #
def test_prf_same_key_same_mask():
    shape = (2, 8, 32)
    key = (3, 1, 0, 0, 32, 7)
    m1 = prf_mask(shape, key, 0.9, device=torch.device("cpu"), dtype=torch.float32)
    m2 = prf_mask(shape, key, 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(m1, m2)


def test_prf_different_key_different_mask():
    shape = (2, 8, 32)
    a = prf_mask(shape, (3, 1, 0, 0, 32, 7), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    # different substep component -> different mask
    b = prf_mask(shape, (3, 2, 0, 0, 32, 7), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert not torch.equal(a, b)


def test_prf_mask_is_binary():
    m = prf_mask((4, 16, 64), (1, 0, 0, 0, 64, 0), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    uniq = set(m.unique().tolist())
    assert uniq.issubset({0.0, 1.0})


def test_mask_independent_of_activation_values():
    """The mask must depend only on the PRF key + shape, never on h's values."""
    masker = ActivationMasker(p=0.9, base_seed=7, pp_size=8)
    layer_idx = 3
    hook = masker._make_hook(layer_idx)

    shape = (2, 8, 32)
    h_zeros = torch.zeros(shape)
    h_rand = torch.randn(shape)

    masker.set_context(global_step=0, substep=0, seq_shard=0)
    out_zeros = hook(nn.Identity(), (), h_zeros)
    masker.set_context(global_step=0, substep=0, seq_shard=0)  # same key again
    out_rand = hook(nn.Identity(), (), h_rand)

    # Re-derive the mask directly from the key and confirm both inputs were
    # multiplied by the SAME mask (value-independence).
    key = (layer_idx, 0, 0, 0, 32, 7)
    mask = prf_mask(shape, key, 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(out_rand, h_rand * mask)
    assert torch.equal(out_zeros, h_zeros * mask)


# --------------------------------------------------------------------------- #
# measured mask ratio tracks p
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [0.90, 0.95])
def test_mask_ratio_tracks_p(p):
    # large tensor so the empirical zeroed fraction concentrates near p
    shape = (8, 64, 256)
    key = (5, 0, 0, 0, 256, 1)
    m = prf_mask(shape, key, p, device=torch.device("cpu"), dtype=torch.float32)
    measured_zero_fraction = float(1.0 - m.mean().item())
    assert abs(measured_zero_fraction - p) <= 0.02


# --------------------------------------------------------------------------- #
# in-graph form: h_tilde = h * mask, no 1/(1-p) rescale, autograd-tracked
# --------------------------------------------------------------------------- #
def test_no_forward_rescale():
    """Kept elements must equal h exactly (no 1/(1-p) scale-up)."""
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    h = torch.full((2, 4, 16), 2.0)
    out = hook(nn.Identity(), (), h)
    # every nonzero output element equals exactly the input (2.0), not 2.0/(1-p)
    nonzero = out[out != 0]
    assert torch.allclose(nonzero, torch.full_like(nonzero, 2.0))


def test_mask_is_in_graph():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    h = torch.randn(2, 4, 16, requires_grad=True)
    out = hook(nn.Identity(), (), h)
    out.sum().backward()
    assert h.grad is not None  # gradient flows through the masked multiply


def test_tuple_output_first_element_masked():
    """HF decoder blocks return tuples; only the hidden state (elem 0) is masked."""
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    h = torch.randn(2, 4, 16)
    extra = torch.randn(2, 4, 16)
    out = hook(nn.Identity(), (), (h, extra))
    assert isinstance(out, tuple)
    assert torch.equal(out[1], extra)  # second element untouched


# --------------------------------------------------------------------------- #
# decoder-layer discovery + hook lifecycle on a toy model
# --------------------------------------------------------------------------- #
class _ToyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


class _ToyDecoder(nn.Module):
    def __init__(self, num_layers=16, d=32):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(d) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_find_decoder_layers():
    model = _ToyDecoder(num_layers=16, d=32)
    layers = find_decoder_layers(model)
    assert layers is not None
    assert len(layers) == 16


def test_register_installs_hooks_on_boundaries_only():
    model = _ToyDecoder(num_layers=16, d=32)
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.register(model)
    assert masker.boundary_indices == [1, 3, 5, 7, 9, 11, 13]
    assert masker.is_registered
    # exactly the boundary blocks carry a forward hook
    for i, layer in enumerate(model.layers):
        has_hook = len(layer._forward_hooks) > 0
        assert has_hook == (i in masker.boundary_indices)
    masker.unregister()
    assert not masker.is_registered
    for layer in model.layers:
        assert len(layer._forward_hooks) == 0


def test_unregister_leaves_forward_clean():
    """After unregister, a forward sees no masking (every element preserved)."""
    torch.manual_seed(0)
    model = _ToyDecoder(num_layers=16, d=32)
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.set_context(global_step=0, substep=0, seq_shard=0)

    x = torch.randn(2, 4, 32)
    masker.register(model)
    out_masked = model(x)
    masker.unregister()
    out_clean = model(x)
    # the masked forward should differ from the clean forward (mask fired)
    assert not torch.allclose(out_masked, out_clean)
    # a second clean forward must reproduce the first clean forward exactly
    out_clean2 = model(x)
    assert torch.allclose(out_clean, out_clean2)


def test_register_is_idempotent():
    model = _ToyDecoder(num_layers=16, d=32)
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.register(model)
    n_handles = len(masker._handles)
    masker.register(model)  # second call must not double-register
    assert len(masker._handles) == n_handles
    masker.unregister()


def test_mask_applications_counter_increments():
    """When a CommEffState is attached, each hook fire bumps mask_applications."""

    class _FakeState:
        def __init__(self):
            self.mask_applications = 0

    state = _FakeState()
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8, state=state)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    hook(nn.Identity(), (), torch.randn(2, 4, 16))
    hook(nn.Identity(), (), torch.randn(2, 4, 16))
    assert state.mask_applications == 2

exec
/bin/zsh -lc "sed -n '340,760p' tests/workers/comm_eff/test_activation_mask.py" in /Users/shamane/Documents/verl
 succeeded in 0ms:
exec
/bin/zsh -lc "sed -n '1,220p' verl/workers/comm_eff/state.py" in /Users/shamane/Documents/verl
 succeeded in 0ms:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-worker state object for the communication-efficient compression method.

The integration contract is deliberately asymmetric so the disabled path is a
**strict no-op**:

* ``maybe_build_comm_eff_state(config)`` returns ``None`` when
  ``config.enabled`` is false (or the config is absent). No object is
  constructed, **no RNG is drawn**, no buffer is allocated, no forward hook is
  registered. The actor therefore holds ``self._comm_eff_state = None`` for a
  dense GRPO run, and every hook below short-circuits on the ``None`` check.

* Only when ``config.enabled`` is true is a ``CommEffState`` constructed; that
  is where mask RNG, anchor EMA buffers and the spectral workspace get
  allocated (lazily, by ``build()``, which later M2 work fills in).

Because construction is gated, a dense GRPO run with this scaffolding merged
consumes the exact same RNG sequence and issues the exact same collective ops
as one without it — the criterion-7 rel-tol-1e-4 parity check holds.

The instrumented counters (``mask_applications``, ``anchor_backwards``,
``spectral_corrections``) live on the state object. When disabled there is no
state object, so the counters are *absent* rather than zero — which the
analyst treats as equivalent to ``== 0`` (no comm_eff op fired). When enabled
they start at 0 and increment per fired op.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid an import cycle at runtime; only needed for type hints
    from verl.workers.config.comm_eff import CommEffConfig

logger = logging.getLogger(__name__)

__all__ = ["CommEffState", "maybe_build_comm_eff_state", "comm_eff_metrics"]


def _is_enabled(config: Any) -> bool:
    """Read the ``enabled`` flag from a comm_eff config that may be a dataclass,
    an OmegaConf node, a plain dict, or ``None``. Pure read — no side effects."""
    if config is None:
        return False
    if isinstance(config, dict):
        return bool(config.get("enabled", False))
    return bool(getattr(config, "enabled", False))


class CommEffState:
    """Per-worker communication-efficient compression state.

    Constructed **only** when ``comm_eff.enabled=true``. Holds the operation
    counters and (once ``build()`` is implemented by later M2 work) the mask
    RNG generator, anchor EMA buffers and spectral workspace. The disabled path
    never instantiates this class — see ``maybe_build_comm_eff_state``.
    """

    def __init__(self, config: "CommEffConfig"):
        # Invariant: never construct a disabled state. The factory enforces it;
        # this assert catches a future caller that forgets to go through it.
        assert _is_enabled(config), (
            "CommEffState must not be constructed when comm_eff.enabled=false; "
            "go through maybe_build_comm_eff_state() so the disabled path stays a no-op."
        )
        self.config = config
        self.enabled = True
        self._built = False

        # Operation counters surfaced into training metrics under comm_eff/*.
        self.mask_applications = 0
        self.anchor_backwards = 0
        self.spectral_corrections = 0

        # The activation masker (first circuit). Constructed in build(); None
        # when the mask sub-config is disabled.
        self.masker = None

        # Whether masking is currently active. Set True only on entry to the
        # actor-train forward/backward (around update_actor) and cleared on
        # exit, so log-prob / ref / infer / val / checkpoint forwards stay clean.
        self.mask_active = False

        # Monotonic optimizer-substep counter (microbatch identity for the PRF
        # key). A trainer step reuses one rollout batch over multiple PPO
        # mini-batches, so this advances per actor optimizer substep, giving
        # each substep a distinct mask even within the same trainer step.
        self.substep = 0

    def build(self, module: Any) -> None:
        """Construct the activation masker for the enabled mask circuit.

        Idempotent. When ``comm_eff.mask.enabled`` is true this constructs an
        ``ActivationMasker`` (no hooks registered yet — the engine registers
        them only on entry to the train forward and removes them on exit). Anchor
        / spectral workspace allocation is deferred to later M2 work.
        """
        if self._built:
            return
        mask_cfg = getattr(self.config, "mask", None)
        mask_enabled = bool(getattr(mask_cfg, "enabled", False)) if mask_cfg is not None else False
        if mask_enabled and float(getattr(mask_cfg, "p", 0.0)) > 0.0:
            # Imported lazily so the disabled path never pays the import cost.
            from verl.workers.comm_eff.activation_mask import ActivationMasker

            self.masker = ActivationMasker(
                p=float(mask_cfg.p),
                base_seed=int(getattr(mask_cfg, "seed", 0)),
                pp_size=int(getattr(mask_cfg, "pp_size", 8)),
                state=self,
            )
        self._built = True

    def mask_ratio_metrics(self) -> dict:
        """Return the most-recently-measured masked fraction per boundary layer.

        Surfaced as ``comm_eff/mask_ratio`` (mean across boundaries) plus a
        per-boundary breakdown. Empty when no mask fired this step.
        """
        if self.masker is None or not self.masker.last_mask_ratio:
            return {}
        ratios = self.masker.last_mask_ratio
        mean_ratio = sum(ratios.values()) / len(ratios)
        out = {"comm_eff/mask_ratio": mean_ratio}
        for idx, r in sorted(ratios.items()):
            out[f"comm_eff/mask_ratio/layer_{idx}"] = r
        return out

    def metrics(self) -> dict:
        """Return the comm_eff operation counters for logging."""
        return {
            "comm_eff/mask_applications": self.mask_applications,
            "comm_eff/anchor_backwards": self.anchor_backwards,
            "comm_eff/spectral_corrections": self.spectral_corrections,
        }


def maybe_build_comm_eff_state(config: Any) -> Optional[CommEffState]:
    """Construct a ``CommEffState`` iff comm_eff is enabled, else return ``None``.

    This is the single gate that guarantees the disabled path is inert: when
    ``config.enabled`` is false (or ``config`` is ``None``/absent) it returns
    ``None`` **without drawing RNG, allocating buffers or registering hooks**.
    Callers store the result and guard every comm_eff op behind a ``None`` /
    ``state.enabled`` check.
    """
    if not _is_enabled(config):
        return None
    state = CommEffState(config)
    logger.info("comm_eff: enabled — constructed CommEffState")
    return state


def comm_eff_metrics(state: Optional[CommEffState]) -> dict:
    """Return comm_eff counters for ``state``, or an empty dict when disabled.

    Centralises the "disabled means no counters" convention so call sites do
    not each re-derive it. Includes the measured ``comm_eff/mask_ratio`` when a
    mask fired this step.
    """
    if state is None:
        return {}
    out = state.metrics()
    out.update(state.mask_ratio_metrics())
    return out

exec
/bin/zsh -lc "sed -n '330,420p' verl/workers/comm_eff/activation_mask.py" in /Users/shamane/Documents/verl
 succeeded in 0ms:
            "(L=%d, pp_size=%d, p=%.4f)",
            self.boundary_indices,
            num_layers,
            self.pp_size,
            self.p,
        )

    def unregister(self) -> None:
        """Remove all mask hooks. Must be called on exit of the train forward."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def is_registered(self) -> bool:
        return bool(self._handles)

codex
One extra issue surfaced from the local tree: the files I can read show hook lifecycle gating via `forward_only` and `mask_active`, but I do not see the named path-tag tests or a tag enum in the checked-out code. I’ll phrase that as conditional, since the pasted summary may refer to a draft PR state not present in this worktree.
exec
/bin/zsh -lc 'rg -n "mask_active|forward_only|infer_batch|validation|compute_log_prob|compute_ref_log_prob|train_mini_batch|vllm|weight_sync|checkpoint" verl/workers verl/trainer tests/workers -S' in /Users/shamane/Documents/verl
 succeeded in 0ms:
tests/workers/test_engine_workers_lora_sync.py:45:    checkpoint_backend: str = "naive",
tests/workers/test_engine_workers_lora_sync.py:52:    # 0. early return for non-naive checkpoint backend
tests/workers/test_engine_workers_lora_sync.py:53:    if checkpoint_backend != "naive":
tests/workers/test_engine_workers_lora_sync.py:432:        """Non-naive checkpoint backend returns early, skips all LoRA logic."""
tests/workers/test_engine_workers_lora_sync.py:443:                checkpoint_backend="disaggregated",
verl/trainer/sft_trainer_ray.py:36:from verl.utils.checkpoint import CheckpointHandler, OrchestrationMode
verl/trainer/sft_trainer_ray.py:64:        self.resume_global_step = self.ckpt_handler.load_checkpoint()
verl/trainer/sft_trainer_ray.py:76:        self.ckpt_handler = CheckpointHandler(
verl/trainer/sft_trainer_ray.py:93:        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
verl/trainer/sft_trainer_ray.py:119:            checkpoint_config=self.checkpoint_config,
verl/trainer/sft_trainer_ray.py:356:                # early exit or validation step
verl/trainer/sft_trainer_ray.py:358:                    # Perform validation
verl/trainer/sft_trainer_ray.py:362:                        output = self.training_client.infer_batch(val_data)
verl/trainer/sft_trainer_ray.py:374:                    self.ckpt_handler.save_checkpoint(step=global_step)
verl/trainer/sft_trainer_ray.py:378:                    print(f"Final validation metrics: {last_valid_metric}")
verl/workers/engine/megatron/utils.py:32:    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
tests/workers/rollout/rollout_vllm/test_vllm_abort.py:15:Test vLLM abort functionality.
tests/workers/rollout/rollout_vllm/test_vllm_abort.py:18:    pytest tests/workers/rollout/rollout_vllm/test_vllm_abort.py -v -s
tests/workers/rollout/rollout_vllm/test_vllm_abort.py:20:    python tests/workers/rollout/rollout_vllm/test_vllm_abort.py
tests/workers/rollout/rollout_vllm/test_vllm_abort.py:29:def test_vllm_abort():
tests/workers/rollout/rollout_vllm/test_vllm_abort.py:34:    ROLLOUT_NAME = "vllm"
tests/workers/rollout/rollout_vllm/test_vllm_abort.py:38:    print("vLLM Abort Test")
tests/workers/rollout/rollout_vllm/test_vllm_abort.py:54:                "VLLM_LOGGING_LEVEL": "INFO",
tests/workers/rollout/rollout_vllm/test_vllm_abort.py:55:                "VLLM_USE_V1": "1",
tests/workers/rollout/rollout_vllm/test_vllm_abort.py:221:    test_vllm_abort()
tests/workers/rollout/test_vllm_cli_args_on_cpu.py:19:from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config
tests/workers/rollout/test_vllm_cli_args_on_cpu.py:68:        """Empty lists are skipped (vLLM nargs='+' requires at least one value)."""
verl/workers/engine/megatron/transformer_impl.py:30:from verl.trainer.config import CheckpointConfig
verl/workers/engine/megatron/transformer_impl.py:32:from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
verl/workers/engine/megatron/transformer_impl.py:50:from verl.utils.megatron_peft_utils import add_base_layer_suffix, build_peft_config_for_vllm
verl/workers/engine/megatron/transformer_impl.py:81:        checkpoint_config: CheckpointConfig,
verl/workers/engine/megatron/transformer_impl.py:88:        self.checkpoint_config = checkpoint_config
verl/workers/engine/megatron/transformer_impl.py:123:        # Apply checkpoint patch for MoE models
verl/workers/engine/megatron/transformer_impl.py:259:        if self.engine_config.forward_only:
verl/workers/engine/megatron/transformer_impl.py:287:        if self.engine_config.use_dist_checkpointing:
verl/workers/engine/megatron/transformer_impl.py:289:                module, self.engine_config.dist_checkpointing_path, is_value_model=self.is_value_model
verl/workers/engine/megatron/transformer_impl.py:366:        if self._qat_enabled and not self.engine_config.forward_only:
verl/workers/engine/megatron/transformer_impl.py:376:            self.engine_config.forward_only
verl/workers/engine/megatron/transformer_impl.py:384:        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
verl/workers/engine/megatron/transformer_impl.py:385:        if self.engine_config.forward_only:
verl/workers/engine/megatron/transformer_impl.py:389:            log_gpu_memory_usage("After offload model during init (forward_only)", logger=logger)
verl/workers/engine/megatron/transformer_impl.py:410:        self.checkpoint_mananager = MegatronCheckpointManager(
verl/workers/engine/megatron/transformer_impl.py:412:            checkpoint_config=self.checkpoint_config,
verl/workers/engine/megatron/transformer_impl.py:425:            use_checkpoint_opt_param_scheduler=self.optimizer_config.use_checkpoint_opt_param_scheduler,
verl/workers/engine/megatron/transformer_impl.py:429:            use_dist_checkpointing=self.engine_config.use_dist_checkpointing,
verl/workers/engine/megatron/transformer_impl.py:550:    def save_checkpoint(
verl/workers/engine/megatron/transformer_impl.py:559:        Save model, optimizer, and scheduler states to a checkpoint.
verl/workers/engine/megatron/transformer_impl.py:562:            local_path: Local filesystem path to save checkpoint.
verl/workers/engine/megatron/transformer_impl.py:563:            hdfs_path: Optional HDFS path to copy checkpoint.
verl/workers/engine/megatron/transformer_impl.py:565:            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
verl/workers/engine/megatron/transformer_impl.py:570:        self.checkpoint_mananager.save_checkpoint(
verl/workers/engine/megatron/transformer_impl.py:577:    def load_checkpoint(
verl/workers/engine/megatron/transformer_impl.py:581:        Load model, optimizer, and scheduler states from a checkpoint.
verl/workers/engine/megatron/transformer_impl.py:584:            local_path: Local filesystem path of the checkpoint.
verl/workers/engine/megatron/transformer_impl.py:585:            hdfs_path: Optional HDFS path where checkpoint is stored.
verl/workers/engine/megatron/transformer_impl.py:590:        self.checkpoint_mananager.load_checkpoint(
verl/workers/engine/megatron/transformer_impl.py:598:    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
verl/workers/engine/megatron/transformer_impl.py:639:            forward_only=forward_only,
verl/workers/engine/megatron/transformer_impl.py:656:            if forward_only and self.engine_config.router_replay.mode == "R2":
verl/workers/engine/megatron/transformer_impl.py:657:                # In R2 mode, forward_only calls (e.g., compute_log_probs) need to record routing information
verl/workers/engine/megatron/transformer_impl.py:672:            forward_only=forward_only,
verl/workers/engine/megatron/transformer_impl.py:719:            peft_config = build_peft_config_for_vllm(self.model_config.lora)
verl/workers/engine/megatron/transformer_impl.py:737:        # QAT: process weights through QATWeightExporter for quantized weight sync to vLLM
verl/workers/engine/megatron/transformer_impl.py:751:    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
verl/workers/engine/megatron/transformer_impl.py:973:        self, output, data: TensorDict, forward_only: bool, loss_function, local_cp_size=None
verl/workers/engine/megatron/transformer_impl.py:976:        # We move calculation of entropy to compute_log_probs, forward_only == True
verl/workers/engine/megatron/transformer_impl.py:988:            assert forward_only, "forward_only must be True when loss_function is None"
verl/trainer/ppo/core_algos.py:2296:        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
tests/workers/rollout/perf/vllm_async_rollout.py:15:Compare vLLM AsyncLLM backend: ExternalRayDistributedExecutor(remote call) vs RayDistributedExecutor(compiled graph)
tests/workers/rollout/perf/vllm_async_rollout.py:21:python3 tests/workers/rollout/perf/vllm_async_rollout.py >perf.log 2>&1
tests/workers/rollout/perf/vllm_async_rollout.py:26:- vllm==0.8.5
tests/workers/rollout/perf/vllm_async_rollout.py:80:        "VLLM_USE_V1": "1",
tests/workers/rollout/perf/vllm_async_rollout.py:81:        "VERL_VLLM_DISTRIBUTED_BACKEND": backend,
tests/workers/rollout/test_pd_disaggregation.py:60:def test_disaggregation_disabled_skips_validation():
tests/workers/rollout/test_pd_disaggregation.py:74:@pytest.mark.parametrize("name", ["vllm", "trtllm"])
tests/workers/rollout/test_pd_disaggregation.py:84:    for name in ("sglang", "vllm", "trtllm"):
tests/workers/rollout/test_pd_disaggregation.py:100:    """``sglang_pd``/``vllm_pd`` were dropped: PD is selected via the disaggregation flag."""
tests/workers/rollout/test_pd_disaggregation.py:104:    assert "vllm" in RolloutReplicaRegistry._registry
tests/workers/rollout/test_pd_disaggregation.py:106:    assert "vllm_pd" not in RolloutReplicaRegistry._registry
tests/workers/rollout/test_pd_disaggregation.py:130:        get_rollout_replica_class("vllm", disaggregation_enabled=True)
verl/workers/engine/automodel/utils.py:97:            activation_checkpointing=engine_config.activation_checkpointing,
verl/workers/engine/automodel/utils.py:103:            activation_checkpointing=engine_config.activation_checkpointing,
verl/workers/engine/automodel/utils.py:108:            activation_checkpointing=engine_config.activation_checkpointing,
verl/workers/engine/automodel/utils.py:189:        activation_checkpointing=engine_config.activation_checkpointing,
tests/workers/test_distillation_topk_symmetry_on_cpu.py:59:        entropy_checkpointing = False
verl/workers/engine/automodel/transformer_impl.py:18:LR scheduling, gradient clipping, and checkpointing to Automodel's
verl/workers/engine/automodel/transformer_impl.py:31:from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
verl/workers/engine/automodel/transformer_impl.py:42:from verl.trainer.config import CheckpointConfig
verl/workers/engine/automodel/transformer_impl.py:79:        checkpoint_config: CheckpointConfig,
verl/workers/engine/automodel/transformer_impl.py:87:        self.checkpoint_config = checkpoint_config
verl/workers/engine/automodel/transformer_impl.py:130:        """Build the model, optimizer, LR scheduler, and checkpointer using Automodel infrastructure."""
verl/workers/engine/automodel/transformer_impl.py:136:        if not self.engine_config.forward_only:
verl/workers/engine/automodel/transformer_impl.py:144:        self._build_checkpointer()
verl/workers/engine/automodel/transformer_impl.py:224:    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
verl/workers/engine/automodel/transformer_impl.py:237:        ctx = torch.no_grad() if forward_only else nullcontext()
verl/workers/engine/automodel/transformer_impl.py:239:        if not forward_only:
verl/workers/engine/automodel/transformer_impl.py:253:            if not forward_only and i == num_micro_batches - 1:
verl/workers/engine/automodel/transformer_impl.py:257:                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
verl/workers/engine/automodel/transformer_impl.py:258:                if not forward_only:
verl/workers/engine/automodel/transformer_impl.py:264:    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
verl/workers/engine/automodel/transformer_impl.py:337:        if self.engine_config.forward_only:
verl/workers/engine/automodel/transformer_impl.py:357:    def _build_checkpointer(self):
verl/workers/engine/automodel/transformer_impl.py:358:        ckpt_config = CheckpointingConfig(
verl/workers/engine/automodel/transformer_impl.py:360:            checkpoint_dir="checkpoints/",
verl/workers/engine/automodel/transformer_impl.py:367:        self.checkpointer = Checkpointer(
verl/workers/engine/automodel/transformer_impl.py:375:    def save_checkpoint(
verl/workers/engine/automodel/transformer_impl.py:383:        """Save model, optimizer, and LR scheduler using Automodel's Checkpointer."""
verl/workers/engine/automodel/transformer_impl.py:389:        self.checkpointer.save_model(self.module, local_path)
verl/workers/engine/automodel/transformer_impl.py:394:            self.checkpointer.save_optimizer(self.optimizer, self.module, local_path, scheduler=scheduler_list)
verl/workers/engine/automodel/transformer_impl.py:400:    def load_checkpoint(
verl/workers/engine/automodel/transformer_impl.py:403:        """Load model, optimizer, and LR scheduler using Automodel's Checkpointer."""
verl/workers/engine/automodel/transformer_impl.py:410:        self.checkpointer.load_model(self.module, model_path)
verl/workers/engine/automodel/transformer_impl.py:414:            self.checkpointer.load_optimizer(self.optimizer, self.module, local_path, scheduler=scheduler_list)
verl/workers/engine/automodel/transformer_impl.py:625:                    if not self.engine_config.entropy_checkpointing:
verl/workers/engine/automodel/transformer_impl.py:628:                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
verl/workers/engine/automodel/transformer_impl.py:655:                    if not self.engine_config.entropy_checkpointing:
verl/workers/engine/automodel/transformer_impl.py:658:                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
verl/workers/engine/automodel/transformer_impl.py:682:    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
verl/workers/engine/automodel/transformer_impl.py:703:                assert forward_only, "forward_only must be True when loss_function is None"
tests/workers/rollout/rollout_sglang/test_lora_sleep_level.py:53:# lora_as_adapter property tests (mirrors vllm_async_server pattern)
verl/trainer/main_ppo.py:65:        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
verl/trainer/main_ppo.py:261:        # Download the checkpoint from HDFS to the local machine.
verl/trainer/main_ppo.py:279:        # Create training and validation datasets.
verl/trainer/main_ppo.py:364:    # Use a sampler to facilitate checkpoint resumption.
tests/workers/config/test_engine_config_on_cpu.py:27:    def test_post_init_validation(self):
verl/trainer/constants_ppo.py:34:        "VLLM_LOGGING_LEVEL": "WARN",
verl/trainer/constants_ppo.py:35:        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
verl/trainer/constants_ppo.py:38:        # https://github.com/vllm-project/vllm/issues/31199
verl/trainer/constants_ppo.py:39:        "VLLM_DISABLE_COMPILE_CACHE": "1",
verl/trainer/ppo/ray_trainer.py:49:    process_validation_metrics,
verl/trainer/ppo/ray_trainer.py:61:from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
verl/trainer/ppo/ray_trainer.py:70:from verl.utils.tracking import ValidationGenerationsLogger
verl/trainer/ppo/ray_trainer.py:147:    The three inputs come from the rollout engine (vLLM request spec-decode
verl/trainer/ppo/ray_trainer.py:291:    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
verl/trainer/ppo/ray_trainer.py:322:            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
verl/trainer/ppo/ray_trainer.py:351:        self.validation_generations_logger = ValidationGenerationsLogger(
verl/trainer/ppo/ray_trainer.py:371:        self.checkpoint_manager = None
verl/trainer/ppo/ray_trainer.py:376:        Creates the train and validation dataloaders.
verl/trainer/ppo/ray_trainer.py:425:            shuffle=self.config.data.get("validation_shuffle", True),
verl/trainer/ppo/ray_trainer.py:431:        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
verl/trainer/ppo/ray_trainer.py:483:        """Dump rollout/validation samples as JSONL asynchronously."""
verl/trainer/ppo/ray_trainer.py:552:        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
verl/trainer/ppo/ray_trainer.py:573:        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)
verl/trainer/ppo/ray_trainer.py:633:                "recompute_log_prob": False,
verl/trainer/ppo/ray_trainer.py:648:                self.checkpoint_manager.sleep_replicas()
verl/trainer/ppo/ray_trainer.py:653:                self.checkpoint_manager.update_weights(self.global_steps)
verl/trainer/ppo/ray_trainer.py:658:            print("validation generation end")
verl/trainer/ppo/ray_trainer.py:699:        val_data_dir = self.config.trainer.get("validation_data_dir", None)
verl/trainer/ppo/ray_trainer.py:714:            print("_merge_validation_results validate result will be merged")
verl/trainer/ppo/ray_trainer.py:725:        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
verl/trainer/ppo/ray_trainer.py:751:    def _merge_validation_results(self, result_a, result_b):
verl/trainer/ppo/ray_trainer.py:821:                checkpoint_config=orig_critic_cfg.checkpoint,
verl/trainer/ppo/ray_trainer.py:892:        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
verl/trainer/ppo/ray_trainer.py:957:        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
verl/trainer/ppo/ray_trainer.py:958:        # Support custom CheckpointEngineManager via config
verl/trainer/ppo/ray_trainer.py:959:        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
verl/trainer/ppo/ray_trainer.py:960:        if checkpoint_manager_class_fqn:
verl/trainer/ppo/ray_trainer.py:961:            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
verl/trainer/ppo/ray_trainer.py:963:            from verl.checkpoint_engine import CheckpointEngineManager
verl/trainer/ppo/ray_trainer.py:964:        self.checkpoint_manager = CheckpointEngineManager(
verl/trainer/ppo/ray_trainer.py:965:            config=checkpoint_engine_config,
verl/trainer/ppo/ray_trainer.py:970:        # sleep all replicas to load checkpoint
verl/trainer/ppo/ray_trainer.py:971:        self.checkpoint_manager.sleep_replicas()
verl/trainer/ppo/ray_trainer.py:973:    def _save_checkpoint(self):
verl/trainer/ppo/ray_trainer.py:1003:        self.actor_rollout_wg.save_checkpoint(
verl/trainer/ppo/ray_trainer.py:1016:            self.critic_wg.save_checkpoint(
verl/trainer/ppo/ray_trainer.py:1026:        # latest checkpointed iteration tracker (for atomic usage)
verl/trainer/ppo/ray_trainer.py:1028:            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
verl/trainer/ppo/ray_trainer.py:1029:            and self.config.actor_rollout_ref.actor.checkpoint.async_save
verl/trainer/ppo/ray_trainer.py:1031:            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
verl/trainer/ppo/ray_trainer.py:1032:            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
verl/trainer/ppo/ray_trainer.py:1034:            print("skip write latest_checkpointed_iteration.txt when async_save is True")
verl/trainer/ppo/ray_trainer.py:1036:        local_latest_checkpointed_iteration = os.path.join(
verl/trainer/ppo/ray_trainer.py:1037:            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
verl/trainer/ppo/ray_trainer.py:1039:        with open(local_latest_checkpointed_iteration, "w") as f:
verl/trainer/ppo/ray_trainer.py:1042:    def _load_checkpoint(self):
verl/trainer/ppo/ray_trainer.py:1050:            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
verl/trainer/ppo/ray_trainer.py:1051:            if not os.path.isabs(checkpoint_folder):
verl/trainer/ppo/ray_trainer.py:1053:                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
verl/trainer/ppo/ray_trainer.py:1054:            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
verl/trainer/ppo/ray_trainer.py:1071:        print(f"Load from checkpoint folder: {global_step_folder}")
verl/trainer/ppo/ray_trainer.py:1081:        self.actor_rollout_wg.load_checkpoint(
verl/trainer/ppo/ray_trainer.py:1086:            self.critic_wg.load_checkpoint(
verl/trainer/ppo/ray_trainer.py:1223:        output = self.critic_wg.infer_batch(batch_td)
verl/trainer/ppo/ray_trainer.py:1231:    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
verl/trainer/ppo/ray_trainer.py:1242:            output = self.actor_rollout_wg.compute_log_prob(batch_td)
verl/trainer/ppo/ray_trainer.py:1244:            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
verl/trainer/ppo/ray_trainer.py:1269:        output = self.actor_rollout_wg.compute_log_prob(batch_td)
verl/trainer/ppo/ray_trainer.py:1352:        output = self.critic_wg.train_mini_batch(batch_td)
verl/trainer/ppo/ray_trainer.py:1384:        # load checkpoint and update weights before doing anything
verl/trainer/ppo/ray_trainer.py:1385:        self._load_checkpoint()
verl/trainer/ppo/ray_trainer.py:1386:        self.checkpoint_manager.update_weights(self.global_steps)
verl/trainer/ppo/ray_trainer.py:1390:        # perform validation before training
verl/trainer/ppo/ray_trainer.py:1391:        # currently, we only support validation using the reward_function.
verl/trainer/ppo/ray_trainer.py:1395:            pprint(f"Initial validation metrics: {val_metrics}")
verl/trainer/ppo/ray_trainer.py:1451:                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
verl/trainer/ppo/ray_trainer.py:1452:                    # rollout after replicas have been put to sleep, which can leave async vLLM
verl/trainer/ppo/ray_trainer.py:1470:                        self.checkpoint_manager.sleep_replicas()
verl/trainer/ppo/ray_trainer.py:1578:                            ref_log_prob = self._compute_ref_log_prob(batch)
verl/trainer/ppo/ray_trainer.py:1644:                        self.checkpoint_manager.update_weights(self.global_steps)
verl/trainer/ppo/ray_trainer.py:1655:                        # Check if the conditions for saving a checkpoint are met.
verl/trainer/ppo/ray_trainer.py:1668:                                print("Force saving checkpoint: ESI instance expiration approaching.")
verl/trainer/ppo/ray_trainer.py:1669:                            with marked_timer("save_checkpoint", timing_raw, color="green"):
verl/trainer/ppo/ray_trainer.py:1670:                                self._save_checkpoint()
verl/trainer/ppo/ray_trainer.py:1674:                            self.checkpoint_manager.update_weights(self.global_steps)
verl/trainer/ppo/ray_trainer.py:1758:                    pprint(f"Final validation metrics: {last_val_metrics}")
tests/workers/config/test_actor_config_on_cpu.py:171:    def test_actor_config_validation_exceptions(self):
tests/workers/config/test_actor_config_on_cpu.py:172:        """Test that ActorConfig.__post_init__ raises appropriate validation exceptions."""
tests/workers/config/test_actor_config_on_cpu.py:217:    def test_fsdp_actor_config_validation_exceptions(self):
tests/workers/config/test_actor_config_on_cpu.py:218:        """Test that FSDPActorConfig.validate() raises appropriate validation exceptions."""
tests/workers/config/test_actor_config_on_cpu.py:223:            use_dynamic_bsz=True,  # Skip batch size validation to focus on FSDP validation
tests/workers/config/test_actor_config_on_cpu.py:234:        """Test that ActorConfig.validate() raises appropriate validation exceptions."""
tests/workers/config/test_critic_config_on_cpu.py:183:    def test_profiler_config_type_validation(self):
tests/workers/config/test_critic_config_on_cpu.py:184:        """Test that profiler field has correct type and validation."""
tests/workers/config/test_critic_config_on_cpu.py:211:    def test_critic_config_validation_logic(self):
tests/workers/config/test_critic_config_on_cpu.py:212:        """Test the __post_init__ validation logic for CriticConfig."""
tests/workers/config/test_critic_config_on_cpu.py:254:    def test_micro_batch_size_divisibility_validation(self):
tests/workers/config/test_critic_config_on_cpu.py:255:        """Test micro batch size divisibility validation in __post_init__."""
tests/workers/config/test_critic_config_on_cpu.py:279:    def test_fsdp_sequence_parallelism_validation(self):
tests/workers/config/test_critic_config_on_cpu.py:280:        """Test FSDP sequence parallelism validation in FSDPCriticConfig.__post_init__."""
tests/workers/rollout/rollout_sglang/test_http_server_engine.py:227:    """Mock ServerArgs.__post_init__ to skip model path validation."""
tests/workers/config/test_model_config_on_cpu.py:47:        # This merge should NOT raise ValidationError
verl/trainer/main_generation_server.py:124:    ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}})
verl/trainer/ppo/metric_utils.py:554:def process_validation_metrics(
verl/trainer/ppo/metric_utils.py:558:    Process validation metrics into a structured format with statistical analysis.
verl/trainer/ppo/metric_utils.py:560:    This function organizes validation metrics by data source and prompt, then computes
verl/trainer/ppo/metric_utils.py:595:        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
verl/trainer/main_ppo_sync.py:55:from verl.checkpoint_engine import CheckpointEngineManager
verl/trainer/main_ppo_sync.py:80:    process_validation_metrics,
verl/trainer/main_ppo_sync.py:88:from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
verl/trainer/main_ppo_sync.py:101:from verl.utils.tracking import Tracking, ValidationGenerationsLogger
verl/trainer/main_ppo_sync.py:306:        # override sampling params for validation
verl/trainer/main_ppo_sync.py:470:            prompts (TensorDict): Input batch from train or validation dataset.
verl/trainer/main_ppo_sync.py:521:        # Download the checkpoint from HDFS to the local machine.
verl/trainer/main_ppo_sync.py:562:            shuffle=self.config.data.get("validation_shuffle", True),
verl/trainer/main_ppo_sync.py:619:                checkpoint_config=critic_cfg.checkpoint,
verl/trainer/main_ppo_sync.py:719:        # 10. initialize checkpoint engine manager
verl/trainer/main_ppo_sync.py:720:        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
verl/trainer/main_ppo_sync.py:721:        self.checkpoint_manager = CheckpointEngineManager(
verl/trainer/main_ppo_sync.py:722:            config=checkpoint_engine_config,
verl/trainer/main_ppo_sync.py:726:        logger.info("checkpoint engine manager initialized")
verl/trainer/main_ppo_sync.py:728:        # sleep all replicas to load checkpoint
verl/trainer/main_ppo_sync.py:729:        self.checkpoint_manager.sleep_replicas()
verl/trainer/main_ppo_sync.py:733:    def _load_checkpoint(self):
verl/trainer/main_ppo_sync.py:736:        # 1. find latest checkpoint folder
verl/trainer/main_ppo_sync.py:740:            checkpoint_folder = self.config.trainer.default_local_dir
verl/trainer/main_ppo_sync.py:741:            if not os.path.isabs(checkpoint_folder):
verl/trainer/main_ppo_sync.py:743:                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
verl/trainer/main_ppo_sync.py:744:            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
verl/trainer/main_ppo_sync.py:762:        # 2. load actor checkpoint
verl/trainer/main_ppo_sync.py:763:        self.actor_rollout_wg.load_checkpoint(
verl/trainer/main_ppo_sync.py:768:        # 3. load critic checkpoint
verl/trainer/main_ppo_sync.py:770:            self.critic_wg.load_checkpoint(
verl/trainer/main_ppo_sync.py:775:        # 4. load dataloader checkpoint
verl/trainer/main_ppo_sync.py:783:    def _save_checkpoint(self):
verl/trainer/main_ppo_sync.py:784:        """Save actor, critic, and dataloader checkpoints to local (and optionally remote) storage."""
verl/trainer/main_ppo_sync.py:790:        logger.info(f"Saving checkpoint to {local_global_step_folder}")
verl/trainer/main_ppo_sync.py:792:        # resolve max checkpoints to keep
verl/trainer/main_ppo_sync.py:813:        self.actor_rollout_wg.save_checkpoint(
verl/trainer/main_ppo_sync.py:827:            self.critic_wg.save_checkpoint(
verl/trainer/main_ppo_sync.py:836:        # write latest checkpointed iteration tracker for atomic resume
verl/trainer/main_ppo_sync.py:837:        actor_ckpt_cfg = self.config.actor_rollout_ref.actor.get("checkpoint", {})
verl/trainer/main_ppo_sync.py:839:            logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
verl/trainer/main_ppo_sync.py:841:        local_latest_checkpointed_iteration = os.path.join(
verl/trainer/main_ppo_sync.py:842:            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
verl/trainer/main_ppo_sync.py:844:        with open(local_latest_checkpointed_iteration, "w") as f:
verl/trainer/main_ppo_sync.py:877:                self.checkpoint_manager.sleep_replicas()
verl/trainer/main_ppo_sync.py:879:                self.checkpoint_manager.update_weights()
verl/trainer/main_ppo_sync.py:961:        val_data_dir = self.config.trainer.get("validation_data_dir", None)
verl/trainer/main_ppo_sync.py:996:        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
verl/trainer/main_ppo_sync.py:1013:        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)
verl/trainer/main_ppo_sync.py:1053:        """Dump rollout/validation samples as JSONL asynchronously."""
verl/trainer/main_ppo_sync.py:1133:        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
verl/trainer/main_ppo_sync.py:1315:        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
verl/trainer/main_ppo_sync.py:1352:    def _compute_ref_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
verl/trainer/main_ppo_sync.py:1364:            output = self.actor_rollout_wg.compute_log_prob(batch)
verl/trainer/main_ppo_sync.py:1366:            output = self.ref_policy_wg.compute_ref_log_prob(batch)
verl/trainer/main_ppo_sync.py:1381:        output = self.critic_wg.infer_batch(batch)
verl/trainer/main_ppo_sync.py:1470:        output: DataProtoFuture = self.critic_wg.train_mini_batch(batch)
verl/trainer/main_ppo_sync.py:1588:        self.validation_generations_logger = ValidationGenerationsLogger(
verl/trainer/main_ppo_sync.py:1593:        # load checkpoint and update weights before doing anything
verl/trainer/main_ppo_sync.py:1594:        self._load_checkpoint()
verl/trainer/main_ppo_sync.py:1595:        self.checkpoint_manager.update_weights()
verl/trainer/main_ppo_sync.py:1597:        # perform validation before training
verl/trainer/main_ppo_sync.py:1601:            pprint(f"Initial validation metrics: {val_metrics}")
verl/trainer/main_ppo_sync.py:1631:                    # 2. save checkpoint
verl/trainer/main_ppo_sync.py:1635:                        with marked_timer("save_checkpoint", timing_raw, color="green"):
verl/trainer/main_ppo_sync.py:1636:                            self._save_checkpoint()
verl/trainer/main_ppo_sync.py:1640:                        self.checkpoint_manager.update_weights()
verl/trainer/main_ppo_sync.py:1670:                    pprint(f"Final validation metrics: {last_val_metrics}")
verl/trainer/main_ppo_sync.py:1701:        self.checkpoint_manager.sleep_replicas()
verl/trainer/main_ppo_sync.py:1721:                batch = self._compute_ref_log_prob(batch, metrics=metrics)
verl/workers/engine/torchtitan/transformer_impl.py:29:from torch.distributed.checkpoint.state_dict import get_model_state_dict
verl/workers/engine/torchtitan/transformer_impl.py:31:from torchtitan.components.checkpoint import CheckpointManager
verl/workers/engine/torchtitan/transformer_impl.py:42:from verl.trainer.config import CheckpointConfig
verl/workers/engine/torchtitan/transformer_impl.py:86:        checkpoint_config: CheckpointConfig,
verl/workers/engine/torchtitan/transformer_impl.py:97:            checkpoint_config: Configuration for checkpointing.
verl/workers/engine/torchtitan/transformer_impl.py:104:        self.checkpoint_config = checkpoint_config
verl/workers/engine/torchtitan/transformer_impl.py:141:        checkpoint = CheckpointManager.Config(
verl/workers/engine/torchtitan/transformer_impl.py:151:        if self.engine_config.offload_policy or self.engine_config.forward_only:
verl/workers/engine/torchtitan/transformer_impl.py:163:            checkpoint=checkpoint,
verl/workers/engine/torchtitan/transformer_impl.py:234:        Sets up checkpoint manager.
verl/workers/engine/torchtitan/transformer_impl.py:237:        self.checkpointer = self.trainer.checkpointer
verl/workers/engine/torchtitan/transformer_impl.py:239:        self.checkpointer.load()
verl/workers/engine/torchtitan/transformer_impl.py:241:        if not self.engine_config.forward_only:
verl/workers/engine/torchtitan/transformer_impl.py:313:    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
verl/workers/engine/torchtitan/transformer_impl.py:333:        ctx = torch.no_grad() if forward_only else nullcontext()
verl/workers/engine/torchtitan/transformer_impl.py:337:                loss, output = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
verl/workers/engine/torchtitan/transformer_impl.py:338:                if not forward_only:
verl/workers/engine/torchtitan/transformer_impl.py:372:    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
verl/workers/engine/torchtitan/transformer_impl.py:407:        if self.engine_config.forward_only:
verl/workers/engine/torchtitan/transformer_impl.py:428:    def save_checkpoint(
verl/workers/engine/torchtitan/transformer_impl.py:436:        """Save checkpoint."""
verl/workers/engine/torchtitan/transformer_impl.py:443:        self.checkpointer.folder = parent_dir
verl/workers/engine/torchtitan/transformer_impl.py:446:            self.checkpointer.keep_latest_k = max_ckpt_to_keep
verl/workers/engine/torchtitan/transformer_impl.py:448:        self.checkpointer.save(curr_step=global_step)
verl/workers/engine/torchtitan/transformer_impl.py:455:    def load_checkpoint(
verl/workers/engine/torchtitan/transformer_impl.py:458:        """Load checkpoint."""
verl/workers/engine/torchtitan/transformer_impl.py:465:        self.checkpointer.folder = parent_dir
verl/workers/engine/torchtitan/transformer_impl.py:471:            self.checkpointer.load(step=step)
verl/workers/engine/torchtitan/transformer_impl.py:474:            self.checkpointer.load(step=-1)
verl/workers/engine/torchtitan/transformer_impl.py:498:        # Convert TorchTitan key names to HuggingFace key names (expected by vLLM)
verl/workers/engine/torchtitan/transformer_impl.py:499:        sd_adapter = self.checkpointer.sd_adapter
verl/workers/engine/torchtitan/transformer_impl.py:505:        # the torchtitan model). But vLLM needs lm_head.weight explicitly, so we
verl/workers/engine/torchtitan/transformer_impl.py:514:        # 128 with EP=8). vLLM needs ALL experts. We gather the missing experts
verl/workers/engine/torchtitan/transformer_impl.py:677:                if not self.engine_config.entropy_checkpointing:
verl/workers/engine/torchtitan/transformer_impl.py:680:                    entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
verl/workers/engine/torchtitan/transformer_impl.py:688:                if not self.engine_config.entropy_checkpointing:
verl/workers/engine/torchtitan/transformer_impl.py:691:                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
verl/workers/engine/torchtitan/transformer_impl.py:710:    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
verl/workers/engine/torchtitan/transformer_impl.py:725:                assert forward_only, "forward_only must be True when loss_function is None"
verl/trainer/config/_generated_ppo_trainer.yaml:44:      entropy_checkpointing: false
verl/trainer/config/_generated_ppo_trainer.yaml:45:      forward_only: false
verl/trainer/config/_generated_ppo_trainer.yaml:95:    checkpoint:
verl/trainer/config/_generated_ppo_trainer.yaml:96:      _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/_generated_ppo_trainer.yaml:155:    entropy_checkpointing: false
verl/trainer/config/_generated_ppo_trainer.yaml:218:      entropy_checkpointing: false
verl/trainer/config/_generated_ppo_trainer.yaml:219:      forward_only: true
verl/trainer/config/_generated_ppo_trainer.yaml:236:    entropy_checkpointing: false
verl/trainer/config/_generated_ppo_trainer.yaml:277:      vllm: {}
verl/trainer/config/_generated_ppo_trainer.yaml:311:    checkpoint_engine:
verl/trainer/config/_generated_ppo_trainer.yaml:312:      _target_: verl.workers.config.CheckpointEngineConfig
verl/trainer/config/_generated_ppo_trainer.yaml:384:    enable_gradient_checkpointing: true
verl/trainer/config/_generated_ppo_trainer.yaml:459:  validation_shuffle: false
verl/trainer/config/_generated_ppo_trainer.yaml:510:    entropy_checkpointing: false
verl/trainer/config/_generated_ppo_trainer.yaml:511:    forward_only: false
verl/trainer/config/_generated_ppo_trainer.yaml:540:  checkpoint:
verl/trainer/config/_generated_ppo_trainer.yaml:541:    _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/_generated_ppo_trainer.yaml:594:    enable_gradient_checkpointing: true
verl/trainer/config/_generated_ppo_trainer.yaml:842:  validation_data_dir: null
verl/trainer/config/_generated_ppo_trainer.yaml:855:  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
verl/trainer/config/optim/megatron.yaml:46:# use checkpoint optimizer parameter scheduler
verl/trainer/config/optim/megatron.yaml:47:use_checkpoint_opt_param_scheduler: False
verl/trainer/ppo/rollout_corr_helper.py:19:1. Policy mismatch between rollout and training implementations (e.g., vLLM BFloat16 vs FSDP FP32)
verl/trainer/ppo/rollout_corr_helper.py:20:2. Model update staleness (training on trajectories from older checkpoints)
verl/trainer/ppo/rollout_corr_helper.py:801:        rollout_log_prob: Log probabilities from the rollout policy (e.g., vLLM BF16),
verl/trainer/ppo/rollout_corr_helper.py:910:    - Policy mismatch (e.g., vLLM BF16 vs FSDP FP32)
verl/trainer/ppo/rollout_corr_helper.py:911:    - Model staleness (training on trajectories from older checkpoints)
verl/workers/rollout/hf_rollout.py:62:        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm
verl/workers/rollout/hf_rollout.py:75:                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
verl/workers/rollout/replica.py:80:    vLLM:
verl/workers/rollout/replica.py:82:    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 0 ...
verl/workers/rollout/replica.py:83:    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 8 ...
verl/workers/rollout/replica.py:230:        from verl.checkpoint_engine.base import CheckpointEngineWorker
verl/workers/rollout/replica.py:232:        rollout_worker_actor_cls = ray.remote(CheckpointEngineWorker)
verl/workers/rollout/replica.py:321:def _load_vllm():
verl/workers/rollout/replica.py:322:    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
verl/workers/rollout/replica.py:324:    return vLLMReplica
verl/workers/rollout/replica.py:331:        import vllm  # noqa: F401
verl/workers/rollout/replica.py:337:        mock_vllm = types.ModuleType("vllm")
verl/workers/rollout/replica.py:339:        mock_custom_ops = types.ModuleType("vllm._custom_ops")
verl/workers/rollout/replica.py:341:        mock_vllm._custom_ops = mock_custom_ops
verl/workers/rollout/replica.py:343:        mock_model_executor = types.ModuleType("vllm.model_executor")
verl/workers/rollout/replica.py:344:        mock_layers = types.ModuleType("vllm.model_executor.layers")
verl/workers/rollout/replica.py:345:        mock_activation = types.ModuleType("vllm.model_executor.layers.activation")
verl/workers/rollout/replica.py:357:        mock_vllm.model_executor = mock_model_executor
verl/workers/rollout/replica.py:359:        sys.modules["vllm"] = mock_vllm
verl/workers/rollout/replica.py:360:        sys.modules["vllm._custom_ops"] = mock_custom_ops
verl/workers/rollout/replica.py:361:        sys.modules["vllm.model_executor"] = mock_model_executor
verl/workers/rollout/replica.py:362:        sys.modules["vllm.model_executor.layers"] = mock_layers
verl/workers/rollout/replica.py:363:        sys.modules["vllm.model_executor.layers.activation"] = mock_activation
verl/workers/rollout/replica.py:378:RolloutReplicaRegistry.register("vllm", _load_vllm)
verl/workers/rollout/replica.py:389:    ``RolloutConfig.disaggregation.enabled``). Validation in
verl/workers/rollout/replica.py:396:        # _load_sglang side-effect: installs vllm mocks needed by SGLangPDReplica's
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:35:      entropy_checkpointing: false
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:48:      forward_only: false
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:86:    checkpoint:
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:87:      _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:171:      entropy_checkpointing: false
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:184:      forward_only: true
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:270:      vllm: {}
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:304:    checkpoint_engine:
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:305:      _target_: verl.workers.config.CheckpointEngineConfig
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:377:    enable_gradient_checkpointing: true
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:452:  validation_shuffle: false
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:494:    entropy_checkpointing: false
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:507:    forward_only: false
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:524:  checkpoint:
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:525:    _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:574:    enable_gradient_checkpointing: true
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:822:  validation_data_dir: null
verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml:835:  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
verl/workers/config/distillation.py:185:            case "vllm":
verl/workers/config/distillation.py:186:                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
verl/workers/config/distillation.py:187:                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
verl/workers/config/distillation.py:189:                    vllm_engine_kwargs["max_logprobs"] = topk
verl/workers/config/distillation.py:193:                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
verl/workers/config/distillation.py:196:                engine_kwargs["vllm"] = vllm_engine_kwargs
verl/workers/config/distillation.py:199:                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
verl/trainer/config/config.py:20:__all__ = ["CheckpointConfig", "ProfileConfig", "BaseModelConfig"]
verl/trainer/config/config.py:24:class CheckpointConfig(BaseConfig):
verl/trainer/config/config.py:25:    """Configuration for model checkpointing.
verl/trainer/config/config.py:30:        save_contents (list[str]): What to include in saved checkpoints.
verl/trainer/config/config.py:32:        load_contents (list[str]): Contents to load from checkpoint. Defaults to same as save_contents.
verl/trainer/config/config.py:33:        async_save (bool): Whether to save checkpoints asynchronously. Only implemented for Megatron as of now.
verl/trainer/config/config.py:34:        strict (bool): Whether to perform strict validation during weight export
verl/trainer/config/config.py:66:    Contains core settings for loading and initializing a pretrained model checkpoint.
verl/workers/rollout/base.py:84:    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
verl/workers/engine/fsdp/transformer_impl.py:35:from verl.trainer.config import CheckpointConfig
verl/workers/engine/fsdp/transformer_impl.py:38:from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
verl/workers/engine/fsdp/transformer_impl.py:97:        checkpoint_config: CheckpointConfig,
verl/workers/engine/fsdp/transformer_impl.py:112:        self.checkpoint_config = checkpoint_config
verl/workers/engine/fsdp/transformer_impl.py:182:        Sets up checkpoint manager and FLOPs counter.
verl/workers/engine/fsdp/transformer_impl.py:187:        self.checkpoint_manager = FSDPCheckpointManager(
verl/workers/engine/fsdp/transformer_impl.py:192:            checkpoint_config=self.checkpoint_config,
verl/workers/engine/fsdp/transformer_impl.py:232:            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16
verl/workers/engine/fsdp/transformer_impl.py:297:            if self.model_config.enable_gradient_checkpointing:
verl/workers/engine/fsdp/transformer_impl.py:298:                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
verl/workers/engine/fsdp/transformer_impl.py:381:            if self.engine_config.forward_only:
verl/workers/engine/fsdp/transformer_impl.py:408:            if self.engine_config.offload_policy or self.engine_config.forward_only:
verl/workers/engine/fsdp/transformer_impl.py:427:            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
verl/workers/engine/fsdp/transformer_impl.py:428:            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)
verl/workers/engine/fsdp/transformer_impl.py:507:        """Restore input_global_scale and input_amax from checkpoint for W4A4 mode."""
verl/workers/engine/fsdp/transformer_impl.py:547:        if self._qat_enabled and not self.engine_config.forward_only:
verl/workers/engine/fsdp/transformer_impl.py:561:        if not self.engine_config.forward_only:
verl/workers/engine/fsdp/transformer_impl.py:611:    def _comm_eff_mask_active(self, forward_only: bool) -> bool:
verl/workers/engine/fsdp/transformer_impl.py:616:          * this is a train pass (``not forward_only`` — never on infer_batch /
verl/workers/engine/fsdp/transformer_impl.py:617:            log-prob / ref / validation),
verl/workers/engine/fsdp/transformer_impl.py:619:          * the worker has set ``state.mask_active`` (set only around
verl/workers/engine/fsdp/transformer_impl.py:623:        if forward_only:
verl/workers/engine/fsdp/transformer_impl.py:628:        if not getattr(state, "mask_active", False):
verl/workers/engine/fsdp/transformer_impl.py:659:    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
verl/workers/engine/fsdp/transformer_impl.py:662:        # infer / ref / validation forward on the same module is clean. When
verl/workers/engine/fsdp/transformer_impl.py:666:        if self._comm_eff_mask_active(forward_only=forward_only):
verl/workers/engine/fsdp/transformer_impl.py:669:            return self._forward_backward_batch_inner(data, loss_function, forward_only=forward_only)
verl/workers/engine/fsdp/transformer_impl.py:675:        self, data: TensorDict, loss_function: Callable, forward_only=False
verl/workers/engine/fsdp/transformer_impl.py:694:        ctx = torch.no_grad() if forward_only else nullcontext()
verl/workers/engine/fsdp/transformer_impl.py:702:                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
verl/workers/engine/fsdp/transformer_impl.py:704:                if not forward_only:
verl/workers/engine/fsdp/transformer_impl.py:715:    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
verl/workers/engine/fsdp/transformer_impl.py:787:        if self.engine_config.forward_only:
verl/workers/engine/fsdp/transformer_impl.py:808:    def save_checkpoint(
verl/workers/engine/fsdp/transformer_impl.py:817:        Save FSDP checkpoint, handling parameter offload as needed.
verl/workers/engine/fsdp/transformer_impl.py:823:        self.checkpoint_manager.save_checkpoint(
verl/workers/engine/fsdp/transformer_impl.py:831:    def load_checkpoint(
verl/workers/engine/fsdp/transformer_impl.py:835:        Load FSDP checkpoint, restoring parameters and optimizer state.
verl/workers/engine/fsdp/transformer_impl.py:842:        self.checkpoint_manager.load_checkpoint(
verl/workers/engine/fsdp/transformer_impl.py:1170:                    if not self.engine_config.entropy_checkpointing:
verl/workers/engine/fsdp/transformer_impl.py:1173:                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
verl/workers/engine/fsdp/transformer_impl.py:1243:                    if not self.engine_config.entropy_checkpointing:
verl/workers/engine/fsdp/transformer_impl.py:1246:                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
verl/workers/engine/fsdp/transformer_impl.py:1297:    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
verl/workers/engine/fsdp/transformer_impl.py:1328:                assert forward_only, "forward_only must be True when loss_function is None"
verl/trainer/config/ppo_trainer.yaml:142:  # Number of generations to log during validation
verl/trainer/config/ppo_trainer.yaml:148:  # Directory for logging validation data; no dump if null
verl/trainer/config/ppo_trainer.yaml:149:  validation_data_dir: null
verl/trainer/config/ppo_trainer.yaml:157:  # Save frequency (by iteration) for model checkpoints
verl/trainer/config/ppo_trainer.yaml:162:  # To ensure a checkpoint is saved before ESI shuts down, the system will start saving a checkpoint in advance.
verl/trainer/config/ppo_trainer.yaml:163:  # The advance time is calculated as: Advance Time = Longest historical step duration + Checkpoint save duration + esi_redundant_time.
verl/trainer/config/ppo_trainer.yaml:168:  # "auto": resume from last checkpoint if available
verl/trainer/config/ppo_trainer.yaml:176:  # Whether to run validation before training begins
verl/trainer/config/ppo_trainer.yaml:179:  # Whether to run validation only
verl/trainer/config/ppo_trainer.yaml:182:  # Validation frequency (in training iterations)
verl/trainer/config/ppo_trainer.yaml:188:  # Default path to distributed filesystem for saving checkpoints
verl/trainer/config/ppo_trainer.yaml:191:  # Whether to delete local checkpoints after loading
verl/trainer/config/ppo_trainer.yaml:194:  # Default local directory for saving checkpoints
verl/trainer/config/ppo_trainer.yaml:195:  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
verl/trainer/config/ppo_trainer.yaml:197:  # Maximum number of actor checkpoints to keep
verl/trainer/config/ppo_trainer.yaml:200:  # Maximum number of critic checkpoints to keep
verl/trainer/config/ppo_trainer.yaml:312:      # Supported stages: actor_update, actor_compute_log_prob, ref_compute_log_prob,
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:41:      forward_only: false
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:89:    checkpoint:
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:90:      _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:206:      forward_only: true
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:259:      vllm: {}
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:293:    checkpoint_engine:
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:294:      _target_: verl.workers.config.CheckpointEngineConfig
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:366:    enable_gradient_checkpointing: true
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:441:  validation_shuffle: false
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:489:    forward_only: false
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:516:  checkpoint:
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:517:    _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:566:    enable_gradient_checkpointing: true
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:814:  validation_data_dir: null
verl/trainer/config/_generated_ppo_veomni_trainer.yaml:827:  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
verl/trainer/config/model/hf_model.yaml:33:# whether to enable gradient checkpointing. Only valid when we use hf model definition
verl/trainer/config/model/hf_model.yaml:34:enable_gradient_checkpointing: True
verl/trainer/config/model/hf_model.yaml:104:  # whether to sync weights / refit by either merging LoRA adapters into the base model weights before transferring to vLLM (for better inference speed but more refit time and potential precision loss). If this is False, it will load separate adapters.
verl/workers/config/critic.py:21:from verl.trainer.config import BaseModelConfig, CheckpointConfig
verl/workers/config/critic.py:66:        checkpoint (Dict[str, Any]): Checkpoint configuration.
verl/workers/config/critic.py:99:    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
verl/workers/config/critic.py:269:        enable_gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency.
verl/workers/config/critic.py:279:    enable_gradient_checkpointing: bool = True
verl/trainer/config/critic/critic.yaml:66:# checkpoint configs
verl/trainer/config/critic/critic.yaml:67:checkpoint:
verl/trainer/config/critic/critic.yaml:70:  _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/critic/critic.yaml:72:  # What to include in saved checkpoints
verl/trainer/config/critic/critic.yaml:73:  # with 'hf_model' you can save whole model as hf format, now only use sharded model checkpoint to save space
verl/trainer/config/critic/critic.yaml:76:  # What to include when loading checkpoints
verl/trainer/config/critic/critic.yaml:79:  # Whether to save checkpoints asynchronously. Only effective for Megatron as of now.
verl/trainer/config/critic/critic.yaml:176:      # Supported stages: actor_update, actor_compute_log_prob, ref_compute_log_prob,
verl/workers/rollout/utils.py:93:    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
verl/workers/rollout/utils.py:120:    server_addresses: vllm or sglang server addresses
verl/workers/rollout/utils.py:122:    rollout_name: name of the rollout backend (e.g., "vllm", "sglang")
verl/trainer/sft_trainer.py:35:from verl.utils.checkpoint import CheckpointHandler
verl/trainer/sft_trainer.py:73:        self.resume_global_step = self.ckpt_handler.load_checkpoint()
verl/trainer/sft_trainer.py:89:        self.ckpt_handler = CheckpointHandler(
verl/trainer/sft_trainer.py:110:                "LoRA is enabled but `model.lora_alpha` is not set; fallback to 0 in checkpoint metadata.",
verl/trainer/sft_trainer.py:144:        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
verl/trainer/sft_trainer.py:170:            checkpoint_config=self.checkpoint_config,
verl/trainer/sft_trainer.py:412:                # early exit or validation step
verl/trainer/sft_trainer.py:414:                    # Perform validation
verl/trainer/sft_trainer.py:418:                        output = self.training_client.infer_batch(val_data)
verl/trainer/sft_trainer.py:439:                    self.ckpt_handler.save_checkpoint(step=global_step)
verl/trainer/sft_trainer.py:444:                        print(f"Final validation metrics: {last_valid_metric}")
verl/trainer/config/actor/actor.yaml:123:# checkpoint configs
verl/trainer/config/actor/actor.yaml:124:checkpoint:
verl/trainer/config/actor/actor.yaml:127:  _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/actor/actor.yaml:129:  # What to include in saved checkpoints
verl/trainer/config/actor/actor.yaml:130:  # with 'hf_model' you can save whole model as hf format, now only use sharded model checkpoint to save space
verl/trainer/config/actor/actor.yaml:133:  # For more flexibility, you can specify the contents to load from the checkpoint.
verl/trainer/config/actor/actor.yaml:137:  # Whether to save checkpoints asynchronously. Only effective for Megatron as of now.
verl/trainer/config/actor/actor.yaml:144:  # speed up the checkpoint saving by 10x speed.
verl/trainer/config/actor/actor.yaml:147:  # Whether to perform strict validation during weight export
verl/trainer/config/actor/actor.yaml:263:      # Supported stages: actor_update, actor_compute_log_prob, ref_compute_log_prob,
verl/trainer/config/actor/actor.yaml:359:#   - Fast quantization is used when syncing weights to vLLM rollout
verl/trainer/config/actor/actor.yaml:384:  # Path to vLLM quantization config JSON file
verl/workers/engine/veomni/transformer_impl.py:33:from verl.trainer.config import CheckpointConfig
verl/workers/engine/veomni/transformer_impl.py:35:from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
verl/workers/engine/veomni/transformer_impl.py:93:        checkpoint_config: CheckpointConfig,
verl/workers/engine/veomni/transformer_impl.py:108:        self.checkpoint_config = checkpoint_config
verl/workers/engine/veomni/transformer_impl.py:171:        Sets up checkpoint manager and FLOPs counter.
verl/workers/engine/veomni/transformer_impl.py:175:        self.checkpoint_manager = FSDPCheckpointManager(
verl/workers/engine/veomni/transformer_impl.py:180:            checkpoint_config=self.checkpoint_config,
verl/workers/engine/veomni/transformer_impl.py:256:            enable_gradient_checkpointing=self.model_config.enable_gradient_checkpointing,
verl/workers/engine/veomni/transformer_impl.py:266:        if not self.engine_config.forward_only:
verl/workers/engine/veomni/transformer_impl.py:280:            self.model_config.enable_gradient_checkpointing,
verl/workers/engine/veomni/transformer_impl.py:304:    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
verl/workers/engine/veomni/transformer_impl.py:311:            forward_only: If True, perform only the forward pass. If False, perform forward and backward pass.
verl/workers/engine/veomni/transformer_impl.py:334:                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
verl/workers/engine/veomni/transformer_impl.py:335:            if not forward_only:
verl/workers/engine/veomni/transformer_impl.py:415:    def save_checkpoint(
verl/workers/engine/veomni/transformer_impl.py:424:        Save VeOmni checkpoint, handling parameter offload as needed.
verl/workers/engine/veomni/transformer_impl.py:430:        self.checkpoint_manager.save_checkpoint(
verl/workers/engine/veomni/transformer_impl.py:438:    def load_checkpoint(
verl/workers/engine/veomni/transformer_impl.py:442:        Load VeOmni checkpoint, restoring parameters and optimizer state.
verl/workers/engine/veomni/transformer_impl.py:447:        self.checkpoint_manager.load_checkpoint(
verl/workers/config/actor.py:21:from verl.trainer.config import CheckpointConfig, RolloutCorrectionConfig
verl/workers/config/actor.py:134:        checkpoint (CheckpointConfig): Configuration for checkpointing.
verl/workers/config/actor.py:183:    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
verl/workers/config/actor.py:272:        load_weight (bool): Whether to load model weights from checkpoint.
verl/workers/config/actor.py:301:        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
verl/workers/config/actor.py:310:    entropy_checkpointing: bool = False
verl/workers/config/actor.py:394:        load_weight (bool): Whether to load model weights from checkpoint.
verl/workers/rollout/trtllm_rollout/trtllm_rollout.py:461:        total_available_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) * 1024 * 1024
verl/trainer/config/rollout/rollout.yaml:4:# actor_rollout_ref.rollout.name: hf/vllm/sglang/trtllm. The default value will be removed in the future
verl/trainer/config/rollout/rollout.yaml:19:# Top-k sampling parameter. -1 for vLLM rollout, 0 for HF rollout.
verl/trainer/config/rollout/rollout.yaml:33:# for vllm rollout
verl/trainer/config/rollout/rollout.yaml:37:# Fraction of GPU memory used by vLLM/SGLang/TRTLLM for KV cache.
verl/trainer/config/rollout/rollout.yaml:49:# supported engines: vllm
verl/trainer/config/rollout/rollout.yaml:62:# For MoE models in vllm, EP=1 refers to ETP parallel in fused_moe with TP*DP weight splits,
verl/trainer/config/rollout/rollout.yaml:87:# scheduling policy for vllm rollout
verl/trainer/config/rollout/rollout.yaml:130:# Extra inference engine arguments (vllm, sglang, trtllm), please refer vllm/sglang/trtllm official doc for detail
verl/trainer/config/rollout/rollout.yaml:133:  # vllm engine config
verl/trainer/config/rollout/rollout.yaml:134:  vllm: {}
verl/trainer/config/rollout/rollout.yaml:142:# Sampling parameters used during validation.
verl/trainer/config/rollout/rollout.yaml:148:  # sampling parameters for validation
verl/trainer/config/rollout/rollout.yaml:149:  # Top-k sampling parameter. -1 for vLLM rollout, 0 for HF rollout.
verl/trainer/config/rollout/rollout.yaml:158:  # whether to repeat n times for validation
verl/trainer/config/rollout/rollout.yaml:260:    # Class name of the custom async server class (e.g. AsyncvLLMServer)
verl/trainer/config/rollout/rollout.yaml:263:# Checkpoint Engine config for update weights from trainer to rollout
verl/trainer/config/rollout/rollout.yaml:264:checkpoint_engine:
verl/trainer/config/rollout/rollout.yaml:266:  # Target class for checkpoint engine config
verl/trainer/config/rollout/rollout.yaml:267:  _target_: verl.workers.config.CheckpointEngineConfig
verl/trainer/config/rollout/rollout.yaml:269:  # Backend for checkpoint engine: naive, nccl, nixl, hccl
verl/trainer/config/rollout/rollout.yaml:286:  # Additional keyword arguments to pass to the checkpoint engine constructor
verl/trainer/config/rollout/rollout.yaml:291:  # in CheckpointEngineRegistry.
verl/trainer/config/rollout/rollout.yaml:421:      # Supported stages: actor_update, actor_compute_log_prob, ref_compute_log_prob,
verl/trainer/config/rollout/rollout.yaml:428:# prometheus configuration for vllm/sglang server mode
verl/trainer/config/rollout/rollout.yaml:446:# type of quantization in vllm, currently support fp8 and torchao
verl/trainer/config/ref/mindspeed_ref.yaml:32:  forward_only: True
verl/workers/config/comm_eff.py:147:        Validation only — no allocation, no RNG. When ``enabled=false`` this
verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py:17:Not recommended depending on vllm for this file.
verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py:44:# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
verl/workers/config/optimizer.py:142:        use_checkpoint_opt_param_scheduler (bool): Whether to use checkpoint optimizer parameter scheduler.
verl/workers/config/optimizer.py:153:    use_checkpoint_opt_param_scheduler: bool = False
verl/workers/engine/mindspeed/transformer_impl.py:23:from verl.trainer.config import CheckpointConfig
verl/workers/engine/mindspeed/transformer_impl.py:60:        checkpoint_config: CheckpointConfig,
verl/workers/engine/mindspeed/transformer_impl.py:62:        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
verl/workers/engine/mindspeed/transformer_impl.py:81:        checkpoint_config: CheckpointConfig,
verl/workers/engine/mindspeed/transformer_impl.py:83:        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
verl/workers/engine/mindspeed/transformer_impl.py:102:        checkpoint_config: CheckpointConfig,
verl/workers/engine/mindspeed/transformer_impl.py:104:        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
verl/workers/engine/mindspeed/transformer_impl.py:122:        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
verl/workers/engine/mindspeed/transformer_impl.py:123:        if self.engine_config.forward_only:
verl/workers/engine/base.py:98:    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
verl/workers/engine/base.py:105:            forward_only: If True, perform only the forward pass. If False, perform forward and backward pass.
verl/workers/engine/base.py:144:        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
verl/workers/engine/base.py:155:    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None) -> Any:
verl/workers/engine/base.py:169:            outputs = self.forward_backward_batch(data, loss_function, forward_only=True)
verl/workers/engine/base.py:204:    def save_checkpoint(
verl/workers/engine/base.py:213:        Save model, optimizer, and scheduler states to a checkpoint.
verl/workers/engine/base.py:216:            local_path: Local filesystem path to save checkpoint.
verl/workers/engine/base.py:217:            hdfs_path: Optional HDFS path to copy checkpoint.
verl/workers/engine/base.py:219:            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
verl/workers/engine/base.py:224:    def load_checkpoint(
verl/workers/engine/base.py:228:        Load model, optimizer, and scheduler states from a checkpoint.
verl/workers/engine/base.py:231:            local_path: Local filesystem path of the checkpoint.
verl/workers/engine/base.py:232:            hdfs_path: Optional HDFS path where checkpoint is stored.
verl/trainer/config/actor/dp_actor.yaml:40:entropy_checkpointing: False
verl/trainer/config/ref/torchtitan_ref.yaml:28:  forward_only: True
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:27:      use_checkpoint_opt_param_scheduler: false
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:42:      use_dist_checkpointing: false
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:43:      dist_checkpointing_path: null
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:46:      dist_checkpointing_prefix: ''
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:63:      forward_only: false
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:116:    checkpoint:
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:117:      _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:231:      use_dist_checkpointing: false
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:232:      dist_checkpointing_path: null
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:235:      dist_checkpointing_prefix: ''
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:246:      forward_only: true
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:305:      vllm: {}
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:339:    checkpoint_engine:
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:340:      _target_: verl.workers.config.CheckpointEngineConfig
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:412:    enable_gradient_checkpointing: true
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:487:  validation_shuffle: false
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:521:    use_checkpoint_opt_param_scheduler: false
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:536:    use_dist_checkpointing: false
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:537:    dist_checkpointing_path: null
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:540:    dist_checkpointing_prefix: ''
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:557:    forward_only: false
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:589:  checkpoint:
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:590:    _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:641:    enable_gradient_checkpointing: true
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:889:  validation_data_dir: null
verl/trainer/config/_generated_ppo_megatron_trainer.yaml:902:  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
verl/workers/config/rollout.py:34:    "CheckpointEngineConfig",
verl/workers/config/rollout.py:143:class CheckpointEngineConfig(BaseConfig):
verl/workers/config/rollout.py:145:    Configuration for checkpoint engine to update weights from trainer to rollout
verl/workers/config/rollout.py:148:    # Backend for checkpoint engine: naive, nccl, nixl, hccl
verl/workers/config/rollout.py:152:    # Additional keyword arguments for checkpoint engine
verl/workers/config/rollout.py:156:    # in CheckpointEngineRegistry.
verl/workers/config/rollout.py:242:    # Fully qualified class name for a custom CheckpointEngineManager. When set, the trainer
verl/workers/config/rollout.py:243:    # loads this class instead of the built-in CheckpointEngineManager.
verl/workers/config/rollout.py:244:    checkpoint_manager_class: Optional[str] = None
verl/workers/config/rollout.py:246:    # Checkpoint Engine config for update weights from trainer to rollout
verl/workers/config/rollout.py:247:    checkpoint_engine: CheckpointEngineConfig = field(default_factory=CheckpointEngineConfig)
verl/workers/config/rollout.py:326:            if self.name == "vllm" or self.name == "sglang" or self.name == "trtllm":
verl/workers/config/rollout.py:352:                f"rollout.name='sglang'; got {self.name!r}. (vLLM PD is a tracked follow-up.)"
verl/trainer/config/ref/megatron_ref.yaml:28:  forward_only: True
verl/trainer/config/sft_trainer_engine.yaml:49:# Checkpoint configuration
verl/trainer/config/sft_trainer_engine.yaml:50:checkpoint:
verl/trainer/config/sft_trainer_engine.yaml:51:  _target_: verl.trainer.config.CheckpointConfig
verl/trainer/config/sft_trainer_engine.yaml:52:  # What to include in saved checkpoints
verl/trainer/config/sft_trainer_engine.yaml:53:  # with 'hf_model' you can save whole model as hf format, now only use sharded model checkpoint to save space
verl/trainer/config/sft_trainer_engine.yaml:56:  # For more flexibility, you can specify the contents to load from the checkpoint.
verl/trainer/config/sft_trainer_engine.yaml:57:  load_contents: ${checkpoint.save_contents}
verl/trainer/config/sft_trainer_engine.yaml:62:  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
verl/trainer/config/sft_trainer_engine.yaml:72:  max_ckpt_to_keep: null  # Maximum number of checkpoints to keep, set to null to keep all
verl/trainer/config/sft_trainer_engine.yaml:76:  # "auto": resume from last checkpoint if available
verl/trainer/config/ref/ref.yaml:26:# profile the ref model in `compute_log_prob`
verl/trainer/config/ref/ref.yaml:116:      # Supported stages: actor_update, actor_compute_log_prob, ref_compute_log_prob,
verl/trainer/config/ref/veomni_ref.yaml:28:  forward_only: True
verl/workers/rollout/vllm_rollout/vllm_async_server.py:24:import vllm.entrypoints.cli.serve
verl/workers/rollout/vllm_rollout/vllm_async_server.py:27:from vllm import SamplingParams
verl/workers/rollout/vllm_rollout/vllm_async_server.py:28:from vllm.engine.arg_utils import AsyncEngineArgs
verl/workers/rollout/vllm_rollout/vllm_async_server.py:29:from vllm.entrypoints.cli.serve import run_headless
verl/workers/rollout/vllm_rollout/vllm_async_server.py:30:from vllm.entrypoints.openai.api_server import build_app, init_app_state
verl/workers/rollout/vllm_rollout/vllm_async_server.py:31:from vllm.inputs import TokensPrompt
verl/workers/rollout/vllm_rollout/vllm_async_server.py:32:from vllm.lora.request import LoRARequest
verl/workers/rollout/vllm_rollout/vllm_async_server.py:33:from vllm.outputs import RequestOutput
verl/workers/rollout/vllm_rollout/vllm_async_server.py:34:from vllm.usage.usage_lib import UsageContext
verl/workers/rollout/vllm_rollout/vllm_async_server.py:35:from vllm.v1.engine.async_llm import AsyncLLM
verl/workers/rollout/vllm_rollout/vllm_async_server.py:40:from verl.utils.profiler import DistProfiler, build_vllm_profiler_args
verl/workers/rollout/vllm_rollout/vllm_async_server.py:42:from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches
verl/workers/rollout/vllm_rollout/vllm_async_server.py:46:from verl.workers.rollout.vllm_rollout.utils import (
verl/workers/rollout/vllm_rollout/vllm_async_server.py:47:    VLLM_LORA_INT_ID,
verl/workers/rollout/vllm_rollout/vllm_async_server.py:48:    VLLM_LORA_NAME,
verl/workers/rollout/vllm_rollout/vllm_async_server.py:49:    VLLM_LORA_PATH,
verl/workers/rollout/vllm_rollout/vllm_async_server.py:53:    get_vllm_max_lora_rank,
verl/workers/rollout/vllm_rollout/vllm_async_server.py:56:_VLLM_VERSION = version.parse(vllm.__version__)
verl/workers/rollout/vllm_rollout/vllm_async_server.py:59:if _VLLM_VERSION > version.parse("0.11.0"):
verl/workers/rollout/vllm_rollout/vllm_async_server.py:60:    from vllm.utils.argparse_utils import FlexibleArgumentParser
verl/workers/rollout/vllm_rollout/vllm_async_server.py:62:    if _VLLM_VERSION == version.parse("0.12.0"):
verl/workers/rollout/vllm_rollout/vllm_async_server.py:63:        from vllm.entrypoints.harmony_utils import get_encoding
verl/workers/rollout/vllm_rollout/vllm_async_server.py:65:    elif _VLLM_VERSION >= version.parse("0.13.0"):
verl/workers/rollout/vllm_rollout/vllm_async_server.py:66:        from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
verl/workers/rollout/vllm_rollout/vllm_async_server.py:74:    from vllm.utils import FlexibleArgumentParser
verl/workers/rollout/vllm_rollout/vllm_async_server.py:81:class vLLMHttpServer:
verl/workers/rollout/vllm_rollout/vllm_async_server.py:82:    """vLLM http server in single node, this is equivalent to launch server with command line:
verl/workers/rollout/vllm_rollout/vllm_async_server.py:84:    vllm serve --tensor-parallel-size=8 ...
verl/workers/rollout/vllm_rollout/vllm_async_server.py:113:        # Forward the Ray job id into the vLLM worker subprocess so the
verl/workers/rollout/vllm_rollout/vllm_async_server.py:143:        # used for controlling vllm server profiler
verl/workers/rollout/vllm_rollout/vllm_async_server.py:211:        # 1. setup vllm serve cli args
verl/workers/rollout/vllm_rollout/vllm_async_server.py:278:        profiler_args = build_vllm_profiler_args(
verl/workers/rollout/vllm_rollout/vllm_async_server.py:281:        if _VLLM_VERSION >= version.parse("0.13.0"):
verl/workers/rollout/vllm_rollout/vllm_async_server.py:282:            # vLLM >= 0.13.0 supports profiler config via CLI args; env vars still work but will be deprecated
verl/workers/rollout/vllm_rollout/vllm_async_server.py:348:                "max_lora_rank": get_vllm_max_lora_rank(lora_rank),
verl/workers/rollout/vllm_rollout/vllm_async_server.py:385:        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
verl/workers/rollout/vllm_rollout/vllm_async_server.py:386:        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port
verl/workers/rollout/vllm_rollout/vllm_async_server.py:388:        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
verl/workers/rollout/vllm_rollout/vllm_async_server.py:395:        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)
verl/workers/rollout/vllm_rollout/vllm_async_server.py:409:        # vLLM >= 0.20.0 requires `model_config` to register pooling API routes
verl/workers/rollout/vllm_rollout/vllm_async_server.py:411:        # ``register_pooling_api_routers`` in vllm/entrypoints/pooling/factories.py
verl/workers/rollout/vllm_rollout/vllm_async_server.py:418:        if "vllm_config" in init_app_sig.parameters:
verl/workers/rollout/vllm_rollout/vllm_async_server.py:419:            await init_app_state(engine_client, vllm_config, app.state, args)
verl/workers/rollout/vllm_rollout/vllm_async_server.py:425:            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")
verl/workers/rollout/vllm_rollout/vllm_async_server.py:468:        # This serves as a safety upper bound. vLLM v0.20+ rejects `max_tokens < 1`
verl/workers/rollout/vllm_rollout/vllm_async_server.py:469:        # (see vllm.sampling_params.SamplingParams._verify_args), so we require at
verl/workers/rollout/vllm_rollout/vllm_async_server.py:494:        # is 1 because vLLM v0.20+ raises VLLMValidationError when max_tokens < 1.
verl/workers/rollout/vllm_rollout/vllm_async_server.py:524:            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
verl/workers/rollout/vllm_rollout/vllm_async_server.py:527:                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
verl/workers/rollout/vllm_rollout/vllm_async_server.py:587:                raise RuntimeError("vLLM MTP rollout requires request_spec_decode_stats; set disable_log_stats=False.")
verl/workers/rollout/vllm_rollout/vllm_async_server.py:668:        On vLLM >= 0.12.0, uses AsyncLLM.pause_generation() to abort in-flight
verl/workers/rollout/vllm_rollout/vllm_async_server.py:671:        validation).
verl/workers/rollout/vllm_rollout/vllm_async_server.py:673:        On vLLM < 0.12.0, manually aborts each request and resets prefix cache.
verl/workers/rollout/vllm_rollout/vllm_async_server.py:681:            if _VLLM_VERSION >= version.parse("0.12.0"):
verl/workers/rollout/vllm_rollout/vllm_async_server.py:695:                # Take an atomic snapshot to avoid race conditions with the vLLM engine thread
verl/workers/rollout/vllm_rollout/vllm_async_server.py:704:                from vllm.v1.engine import FinishReason
verl/workers/rollout/vllm_rollout/vllm_async_server.py:731:        Only effective on vLLM >= 0.12.0 where pause_generation is used.
verl/workers/rollout/vllm_rollout/vllm_async_server.py:736:        if _VLLM_VERSION >= version.parse("0.12.0"):
verl/workers/rollout/vllm_rollout/vllm_async_server.py:756:            from vllm.v1.engine import FinishReason
verl/workers/rollout/vllm_rollout/vllm_async_server.py:813:        """Return the key under config.engine_kwargs for this engine (e.g. 'vllm')."""
verl/workers/rollout/vllm_rollout/vllm_async_server.py:814:        return "vllm"
verl/workers/rollout/vllm_rollout/vllm_async_server.py:838:            from verl.utils.vllm.npu_vllm_patch import check_vllm_ascend_before_server_launch
verl/workers/rollout/vllm_rollout/vllm_async_server.py:840:            check_vllm_ascend_before_server_launch()
verl/workers/rollout/vllm_rollout/vllm_async_server.py:886:                # Apply vllm fp8 patches
verl/workers/rollout/vllm_rollout/vllm_async_server.py:887:                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
verl/workers/rollout/vllm_rollout/vllm_async_server.py:888:                apply_vllm_fp8_patches()
verl/workers/rollout/vllm_rollout/vllm_async_server.py:890:                os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"
verl/workers/rollout/vllm_rollout/vllm_async_server.py:899:        return "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
verl/workers/rollout/vllm_rollout/vllm_async_server.py:903:        return [vllm.entrypoints.cli.serve]
verl/workers/rollout/vllm_rollout/vllm_async_server.py:907:        return "vLLM CLI"
verl/workers/rollout/vllm_rollout/vllm_async_server.py:917:        # vllm_ascend not support sleep_level now. Enabling EP during training may lead to accuracy issues.
verl/workers/rollout/vllm_rollout/vllm_async_server.py:923:        if _VLLM_VERSION >= version.parse("0.17.0"):
verl/workers/rollout/vllm_rollout/vllm_async_server.py:927:class vLLMReplica(RolloutReplica):
verl/workers/rollout/vllm_rollout/vllm_async_server.py:941:        self.server_class = ray.remote(vLLMHttpServer)
verl/workers/rollout/vllm_rollout/vllm_async_server.py:986:                # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
verl/workers/rollout/vllm_rollout/vllm_async_server.py:987:                # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
verl/workers/rollout/vllm_rollout/vllm_async_server.py:1079:        # Drain all in-flight requests so that vLLM worker threads go idle
verl/workers/rollout/vllm_rollout/vllm_async_server.py:1091:        # For multi-node without DP (e.g TP=16), need vllm>=0.11.1, https://github.com/vllm-project/vllm/pull/23691
verl/workers/rollout/vllm_rollout/vllm_async_server.py:1093:            assert _VLLM_VERSION >= version.parse("0.11.1"), (
verl/workers/rollout/vllm_rollout/vllm_async_server.py:1094:                "For multi-node MP Executor, either (1) set data_parallel_size > 1 or (2) upgrade vLLM to >= 0.11.1"
verl/workers/rollout/vllm_rollout/vllm_async_server.py:1098:        """Return the Ray actor name prefix (e.g. 'vllm_')."""
verl/workers/rollout/vllm_rollout/vllm_async_server.py:1099:        return "vllm_"
verl/trainer/config/npu_profile/npu_profile.yaml:8:  # optional values: all, rollout_generate, actor_compute_log_prob, actor_update and ref_compute_log_prob.
verl/trainer/config/engine/automodel.yaml:25:# Whether to enable activation checkpointing
verl/trainer/config/engine/automodel.yaml:26:activation_checkpointing: false
verl/trainer/config/engine/automodel.yaml:73:forward_only: false
verl/trainer/config/engine/automodel.yaml:81:# Whether to use checkpointing for entropy computation
verl/trainer/config/engine/automodel.yaml:82:entropy_checkpointing: false
verl/workers/comm_eff/state.py:95:        # exit, so log-prob / ref / infer / val / checkpoint forwards stay clean.
verl/workers/comm_eff/state.py:96:        self.mask_active = False
verl/trainer/config/ref/dp_ref.yaml:20:  forward_only: True
verl/trainer/config/ref/dp_ref.yaml:30:entropy_checkpointing: False
verl/trainer/config/algorithm.py:66:    1. Policy mismatch: Rollout policy (e.g., vLLM BF16) vs Training policy (e.g., FSDP FP32)
verl/trainer/config/algorithm.py:67:    2. Model update staleness: Rollout data collected from older policy checkpoints
verl/trainer/config/algorithm.py:75:    - Type safety and validation
verl/workers/config/engine.py:22:from verl.trainer.config import CheckpointConfig
verl/workers/config/engine.py:86:        "forward_only",
verl/workers/config/engine.py:96:    forward_only: bool = False
verl/workers/config/engine.py:138:        quantization_config_path (Optional[str]): Path to quantization config JSON for vLLM
verl/workers/config/engine.py:170:        use_dist_checkpointing (bool): Whether to use distributed checkpointing.
verl/workers/config/engine.py:171:        dist_checkpointing_path (Optional[str]): Path for distributed checkpointing.
verl/workers/config/engine.py:172:        dist_ckpt_optim_fully_reshardable (bool): Use fully reshardable optimizer checkpoints.
verl/workers/config/engine.py:195:    use_dist_checkpointing: bool = False
verl/workers/config/engine.py:196:    dist_checkpointing_path: Optional[str] = None
verl/workers/config/engine.py:197:    dist_checkpointing_prefix: str = ""
verl/workers/config/engine.py:211:        """config validation logics go here"""
verl/workers/config/engine.py:259:    entropy_checkpointing: bool = False
verl/workers/config/engine.py:291:        enable_reentrant (bool): Use reentrant gradient checkpointing, default False
verl/workers/config/engine.py:345:    entropy_checkpointing: bool = False
verl/workers/config/engine.py:356:    load_checkpoint_path: Optional[str] = None
verl/workers/config/engine.py:429:    entropy_checkpointing: bool = False
verl/workers/config/engine.py:467:        activation_checkpointing (bool): Whether to enable activation checkpointing.
verl/workers/config/engine.py:500:                    checkpoint compatibility. Default: true.
verl/workers/config/engine.py:512:                ignore_router_for_ac (bool): Exclude router from activation checkpointing.
verl/workers/config/engine.py:529:        entropy_checkpointing (bool): Whether to use checkpointing for entropy computation.
verl/workers/config/engine.py:543:    activation_checkpointing: bool = False
verl/workers/config/engine.py:559:    entropy_checkpointing: bool = False
verl/workers/config/engine.py:586:        """config validation logics go here"""
verl/workers/config/engine.py:600:    checkpoint_config: CheckpointConfig = None
verl/trainer/config/data/legacy_data.yaml:13:# Validation parquet. Can be a list or a single file.
verl/trainer/config/data/legacy_data.yaml:44:# Batch size used during validation. Can be null.
verl/trainer/config/data/legacy_data.yaml:76:# Whether to shuffle the validation set.
verl/trainer/config/data/legacy_data.yaml:77:validation_shuffle: False
verl/trainer/config/engine/torchtitan.yaml:34:# Whether to use entropy checkpointing
verl/trainer/config/engine/torchtitan.yaml:35:entropy_checkpointing: false
verl/trainer/config/engine/torchtitan.yaml:74:forward_only: false
verl/workers/rollout/vllm_rollout/utils.py:25:from vllm.outputs import RequestOutput
verl/workers/rollout/vllm_rollout/utils.py:28:from verl.utils.vllm import TensorLoRARequest, VLLMHijack
verl/workers/rollout/vllm_rollout/utils.py:29:from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
verl/workers/rollout/vllm_rollout/utils.py:30:from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights
verl/workers/rollout/vllm_rollout/utils.py:36:VLLM_LORA_INT_ID = 123
verl/workers/rollout/vllm_rollout/utils.py:37:VLLM_LORA_NAME = "123"
verl/workers/rollout/vllm_rollout/utils.py:38:VLLM_LORA_PATH = "simon_lora_path"
verl/workers/rollout/vllm_rollout/utils.py:40:VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}
verl/workers/rollout/vllm_rollout/utils.py:54:    from vllm.platforms import current_platform
verl/workers/rollout/vllm_rollout/utils.py:68:def get_vllm_max_lora_rank(lora_rank: int):
verl/workers/rollout/vllm_rollout/utils.py:70:    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
verl/workers/rollout/vllm_rollout/utils.py:71:    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
verl/workers/rollout/vllm_rollout/utils.py:76:        from vllm.config.lora import MaxLoRARanks
verl/workers/rollout/vllm_rollout/utils.py:78:        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
verl/workers/rollout/vllm_rollout/utils.py:81:    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
verl/workers/rollout/vllm_rollout/utils.py:82:    if lora_rank > vllm_max_lora_ranks[-1]:
verl/workers/rollout/vllm_rollout/utils.py:83:        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")
verl/workers/rollout/vllm_rollout/utils.py:85:    for rank in vllm_max_lora_ranks:
verl/workers/rollout/vllm_rollout/utils.py:90:# https://github.com/vllm-project/vllm/issues/13175
verl/workers/rollout/vllm_rollout/utils.py:106:class vLLMColocateWorkerExtension:
verl/workers/rollout/vllm_rollout/utils.py:108:    The class for vLLM's worker to inherit from, in the colocate setting.
verl/workers/rollout/vllm_rollout/utils.py:111:    with both vLLM V0 and V1.
verl/workers/rollout/vllm_rollout/utils.py:124:        VLLMHijack.hijack()
verl/workers/rollout/vllm_rollout/utils.py:126:        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
verl/workers/rollout/vllm_rollout/utils.py:127:            apply_vllm_fp8_patches()
verl/workers/rollout/vllm_rollout/utils.py:129:        vllm_config = kwargs.get("vllm_config")
verl/workers/rollout/vllm_rollout/utils.py:130:        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
verl/workers/rollout/vllm_rollout/utils.py:137:            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
verl/workers/rollout/vllm_rollout/utils.py:142:            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")
verl/workers/rollout/vllm_rollout/utils.py:144:        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
verl/workers/rollout/vllm_rollout/utils.py:145:        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
verl/workers/rollout/vllm_rollout/utils.py:146:        # This is only a fix for vllm version < v0.13.0.
verl/workers/rollout/vllm_rollout/utils.py:148:            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
verl/workers/rollout/vllm_rollout/utils.py:150:                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]
verl/workers/rollout/vllm_rollout/utils.py:164:        spec = self.model_runner.vllm_config.speculative_config
verl/workers/rollout/vllm_rollout/utils.py:167:    def _use_mtp_drafter_weight_sync(self):
verl/workers/rollout/vllm_rollout/utils.py:168:        """Return whether the vLLM MTP drafter should receive actor weights."""
verl/workers/rollout/vllm_rollout/utils.py:169:        spec = self.model_runner.vllm_config.speculative_config
verl/workers/rollout/vllm_rollout/utils.py:175:        Only vLLM MTP drafter sync is supported for now. Independent non-MTP
verl/workers/rollout/vllm_rollout/utils.py:179:        if self._use_mtp_drafter_weight_sync():
verl/workers/rollout/vllm_rollout/utils.py:184:        yield self.model_runner.model, self.model_runner.vllm_config.model_config
verl/workers/rollout/vllm_rollout/utils.py:185:        if self._use_mtp_drafter_weight_sync():
verl/workers/rollout/vllm_rollout/utils.py:195:            patch_vllm_moe_model_weight_loader(model)
verl/workers/rollout/vllm_rollout/utils.py:199:        from vllm.platforms import current_platform
verl/workers/rollout/vllm_rollout/utils.py:201:        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver
verl/workers/rollout/vllm_rollout/utils.py:208:            self.remove_lora(VLLM_LORA_INT_ID)
verl/workers/rollout/vllm_rollout/utils.py:211:            self.model_runner.vllm_config
verl/workers/rollout/vllm_rollout/utils.py:222:            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload
verl/workers/rollout/vllm_rollout/utils.py:229:                patch_vllm_moe_model_weight_loader(model)
verl/workers/rollout/vllm_rollout/utils.py:251:            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading
verl/workers/rollout/vllm_rollout/utils.py:257:            from vllm.model_executor.model_loader.utils import process_weights_after_loading
verl/workers/rollout/vllm_rollout/utils.py:266:                lora_name=VLLM_LORA_NAME,
verl/workers/rollout/vllm_rollout/utils.py:267:                lora_int_id=VLLM_LORA_INT_ID,
verl/workers/rollout/vllm_rollout/utils.py:268:                lora_path=VLLM_LORA_PATH,
verl/workers/rollout/vllm_rollout/utils.py:273:            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
verl/workers/rollout/vllm_rollout/utils.py:277:            if is_fp8_model(self.model_runner.vllm_config):
verl/workers/rollout/vllm_rollout/utils.py:278:                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
verl/workers/rollout/vllm_rollout/utils.py:283:                if self._use_mtp_drafter_weight_sync():
verl/workers/rollout/vllm_rollout/utils.py:296:        job id is forwarded by the vLLMHttpServer actor as VERL_RAY_JOB_ID and
verl/workers/rollout/vllm_rollout/utils.py:297:        inherited by this vLLM worker subprocess.
verl/workers/rollout/vllm_rollout/utils.py:323:    Convert a config dictionary to CLI arguments for vLLM server.
verl/workers/rollout/vllm_rollout/utils.py:330:    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
verl/workers/rollout/vllm_rollout/utils.py:349:                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
verl/workers/engine_workers.py:30:from verl.checkpoint_engine import CheckpointEngineRegistry
verl/workers/engine_workers.py:98:        self.checkpoint_config = self.config.checkpoint_config
verl/workers/engine_workers.py:135:            checkpoint_config=self.checkpoint_config,
verl/workers/engine_workers.py:174:    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
verl/workers/engine_workers.py:227:            if forward_only:
verl/workers/engine_workers.py:236:    def train_mini_batch(self, data: TensorDict) -> TensorDict:
verl/workers/engine_workers.py:329:        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
verl/workers/engine_workers.py:373:                forward_only=False,
verl/workers/engine_workers.py:382:    def infer_batch(self, data: TensorDict) -> TensorDict:
verl/workers/engine_workers.py:411:                output = self.engine.infer_batch(data, loss_function=loss_function)
verl/workers/engine_workers.py:419:                forward_only=True,
verl/workers/engine_workers.py:428:    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
verl/workers/engine_workers.py:429:        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
verl/workers/engine_workers.py:432:    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
verl/workers/engine_workers.py:433:        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)
verl/workers/engine_workers.py:520:                checkpoint_config=ref_config.checkpoint,
verl/workers/engine_workers.py:548:                checkpoint_config=actor_config.checkpoint,
verl/workers/engine_workers.py:612:        # 4. build checkpoint engine
verl/workers/engine_workers.py:614:            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
verl/workers/engine_workers.py:615:            backend = checkpoint_engine_config.backend
verl/workers/engine_workers.py:616:            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
verl/workers/engine_workers.py:617:            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
verl/workers/engine_workers.py:619:            # in CheckpointEngineRegistry before the backend is instantiated.
verl/workers/engine_workers.py:620:            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
verl/workers/engine_workers.py:621:            self.checkpoint_engine = CheckpointEngineRegistry.new(
verl/workers/engine_workers.py:625:        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
verl/workers/engine_workers.py:629:    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
verl/workers/engine_workers.py:631:    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
verl/workers/engine_workers.py:632:        output = self.ref.infer_batch(data=data)
verl/workers/engine_workers.py:636:    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
verl/workers/engine_workers.py:638:    def compute_log_prob(self, data: TensorDict) -> TensorDict:
verl/workers/engine_workers.py:639:        output = self.actor.infer_batch(data)
verl/workers/engine_workers.py:668:                # shared between the worker (sets mask_active around update_actor)
verl/workers/engine_workers.py:669:                # and the engine (registers/clears hooks gated on mask_active).
verl/workers/engine_workers.py:691:        # them on exit, gated on this flag; log_prob / infer / ref / validation /
verl/workers/engine_workers.py:692:        # checkpoint forwards never set it, so they stay byte-identical to dense.
verl/workers/engine_workers.py:694:            comm_eff_state.mask_active = True
verl/workers/engine_workers.py:696:            output = self.actor.train_mini_batch(data=data)
verl/workers/engine_workers.py:699:                comm_eff_state.mask_active = False
verl/workers/engine_workers.py:705:        # None on non-output ranks (train_mini_batch only populates metrics on the
verl/workers/engine_workers.py:723:    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
verl/workers/engine_workers.py:724:        assert "actor" in self.role, "load_checkpoint only support actor role"
verl/workers/engine_workers.py:725:        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)
verl/workers/engine_workers.py:728:    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
verl/workers/engine_workers.py:729:        assert "actor" in self.role, "save_checkpoint only support actor role"
verl/workers/engine_workers.py:730:        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
verl/workers/engine_workers.py:739:        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.
verl/workers/engine_workers.py:749:                  ``config.rollout.checkpoint_engine.backend`` (default).
verl/workers/engine_workers.py:754:                  :meth:`checkpoint_engine.send_weights` for asynchronous weight
verl/workers/engine_workers.py:755:                  transfer via checkpoint engine, suitable for disaggregated
verl/workers/engine_workers.py:760:        effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend
verl/workers/engine_workers.py:765:            await self.checkpoint_engine.send_weights(per_tensor_param)
verl/workers/engine_workers.py:815:    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
verl/workers/engine_workers.py:816:        """Execute checkpoint engine method.
verl/workers/engine_workers.py:819:            method (str): Checkpoint engine method name.
verl/workers/engine_workers.py:824:        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
verl/trainer/config/engine/megatron.yaml:37:# Whether to use distributed checkpointing
verl/trainer/config/engine/megatron.yaml:38:use_dist_checkpointing: False
verl/trainer/config/engine/megatron.yaml:40:# distributed checkpointing path
verl/trainer/config/engine/megatron.yaml:41:dist_checkpointing_path: null
verl/trainer/config/engine/megatron.yaml:49:# distributed checkpointing prefix, e.g. Nemo2 will append prefix 'module.' to the state dict keys
verl/trainer/config/engine/megatron.yaml:50:dist_checkpointing_prefix: ''
verl/trainer/config/engine/megatron.yaml:52:# Make optimizer distributed checkpoint fully reshardable (TP/PP/EP/DP) as opposed to plain DP reshardability
verl/trainer/config/engine/megatron.yaml:78:  # 'uniform' divides the total number of transformer layers and checkpoints the input activation of each chunk
verl/trainer/config/engine/megatron.yaml:79:  # 'block' checkpoints the specified number of layers per pipeline stage at the specified granularity
verl/trainer/config/engine/megatron.yaml:82:  # 'full' will checkpoint the entire transformer layer and 'selective' only checkpoints memory intensive part of attention
verl/trainer/config/engine/megatron.yaml:103:forward_only: False
verl/trainer/config/engine/megatron.yaml:142:  # Path to quantization config JSON for vLLM weight export
verl/trainer/config/engine/veomni.yaml:41:forward_only: false
verl/workers/comm_eff/activation_mask.py:53:  ``infer_batch`` / validation / checkpoint forward on the same module sees no
verl/workers/comm_eff/activation_mask.py:232:    exit, so log-prob / ref / infer / validation / checkpoint forwards never see
verl/trainer/config/engine/fsdp.yaml:53:# Whether to use entropy checkpointing in fsdp.
verl/trainer/config/engine/fsdp.yaml:54:entropy_checkpointing: false
verl/trainer/config/engine/fsdp.yaml:57:forward_only: false
verl/trainer/config/engine/fsdp.yaml:90:  # Path to vLLM quantization config JSON file
verl/workers/config/model.py:43:    vLLM rollout parameters:
verl/workers/config/model.py:117:    enable_gradient_checkpointing: bool = True
verl/workers/rollout/tokenizer.py:15:The base tokenizer class, required for any hybrid engine based rollout or inference with vLLM.
verl/workers/rollout/tokenizer.py:27:    """the tokenizer property and function name should align with HF's to meet vllm requirement"""
verl/workers/rollout/sglang_rollout/sglang_rollout.py:36:from sglang.srt.weight_sync.utils import _preprocess_tensor_for_update_weights
verl/workers/rollout/sglang_rollout/sglang_rollout.py:37:from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
verl/workers/rollout/sglang_rollout/sglang_rollout.py:328:            update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
verl/workers/rollout/vllm_rollout/__init__.py:17:from .vllm_rollout import ServerAdapter  # noqa: F401
verl/workers/rollout/vllm_rollout/__init__.py:27:vllm_package_name = "vllm"
verl/workers/rollout/vllm_rollout/__init__.py:28:vllm_package_version = get_version(vllm_package_name)
verl/workers/rollout/vllm_rollout/__init__.py:29:if vllm_package_version is None:
verl/workers/rollout/vllm_rollout/__init__.py:31:        "To use vllm rollout, please ensure the 'vllm' package is properly installed. See "
verl/workers/rollout/vllm_rollout/__init__.py:38:    match = re.match(r"(\d+\.\d+\.?\d*)", vllm_package_version)
verl/workers/rollout/vllm_rollout/__init__.py:40:        vllm_package_version = match.group(1)
verl/workers/rollout/vllm_rollout/__init__.py:42:        raise ValueError(f"Warning: Could not parse version format: {vllm_package_version}")
verl/workers/rollout/sglang_rollout/async_sglang_server.py:67:    """Shape SGLang input-logprobs into the vLLM ``extract_prompt_logprobs`` contract.
verl/workers/rollout/sglang_rollout/async_sglang_server.py:70:    consumer in ``teacher_manager.AsyncTeacherLLMServerManager`` can treat vLLM and
verl/workers/rollout/sglang_rollout/async_sglang_server.py:73:    entry has ``logprob=None`` (no predicting context). That matches the vLLM
verl/workers/rollout/sglang_rollout/async_sglang_server.py:82:    # Entry 0 has logprob=None (no predicting context); skip it, matching vLLM.
verl/workers/rollout/sglang_rollout/async_sglang_server.py:91:            # 0 is the top-1 token, matching the vLLM extractor's rank-1 slot.
verl/workers/rollout/sglang_rollout/async_sglang_server.py:99:    # Trailing dummy row so total length == len(sequence_ids), matching vLLM.
verl/workers/rollout/sglang_rollout/async_sglang_server.py:460:        # Mirrors the vLLM sleep() pattern in vllm_async_server.py.
verl/workers/rollout/sglang_rollout/async_sglang_server.py:549:            # support vllm-style 'max_tokens' param
verl/workers/rollout/sglang_rollout/async_sglang_server.py:567:        # vLLM-style "prompt_logprobs=K" from the distillation teacher: request
verl/workers/rollout/vllm_rollout/vllm_rollout.py:15:The vllm_rollout that can be applied in different backend
verl/workers/rollout/vllm_rollout/vllm_rollout.py:18:- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
verl/workers/rollout/vllm_rollout/vllm_rollout.py:40:from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
verl/workers/rollout/vllm_rollout/vllm_rollout.py:44:from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
verl/workers/rollout/vllm_rollout/vllm_rollout.py:45:from verl.workers.rollout.vllm_rollout.utils import get_device_uuid
verl/workers/rollout/vllm_rollout/vllm_rollout.py:51:def _check_vllm_version_for_sleep_level():
verl/workers/rollout/vllm_rollout/vllm_rollout.py:52:    # https://github.com/vllm-project/vllm/issues/25171
verl/workers/rollout/vllm_rollout/vllm_rollout.py:54:    current_version = get_version("vllm")
verl/workers/rollout/vllm_rollout/vllm_rollout.py:56:        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
verl/workers/rollout/vllm_rollout/vllm_rollout.py:63:    vLLM server adapter used in native async mode, serve as a client to request vLLM server
verl/workers/rollout/vllm_rollout/vllm_rollout.py:91:        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
verl/workers/rollout/vllm_rollout/vllm_rollout.py:95:            self.sleep_level = VLLM_SLEEP_LEVEL
verl/workers/rollout/vllm_rollout/vllm_rollout.py:99:        # because CheckpointEngineWorker and vLLM worker may see different GPU UUIDs
verl/workers/rollout/vllm_rollout/vllm_rollout.py:101:        # Must use node-local rank (not rollout_rank) so it matches vLLM worker's
verl/workers/rollout/vllm_rollout/vllm_rollout.py:177:        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
verl/workers/rollout/vllm_rollout/vllm_rollout.py:198:        """Return the Ray actor name prefix matching the rollout type (e.g. 'vllm_')."""
verl/workers/rollout/vllm_rollout/vllm_rollout.py:199:        return f"{self.config.get('name', 'vllm')}_"
verl/workers/rollout/vllm_rollout/vllm_rollout.py:213:            "The vLLM SPMD mode was retired in PR #4411. For batch generation, "
verl/workers/rollout/vllm_rollout/vllm_rollout.py:214:            "please use the async server interface via vLLMReplica and LLMServerClient, "

exec
/bin/zsh -lc "rg -n \"path_tag|PathTag|train\\|rollout|old_logprob|old-logprob|ref_logprob|asserts_on_non_train|tag_inactive|mask.*wrong|wrong.*path\" . -S" in /Users/shamane/Documents/verl
 succeeded in 0ms:
./verl/trainer/ppo/core_algos.py:2126:def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
./verl/trainer/ppo/core_algos.py:2127:    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
./verl/trainer/ppo/core_algos.py:2133:        ref_logprob:
./verl/trainer/ppo/core_algos.py:2140:    forward_score = kl_penalty_forward(logprob, ref_logprob, base_kl_penalty)
./verl/trainer/ppo/core_algos.py:2149:    backward_score = 0.5 * (logprob - ref_logprob).square()
./verl/trainer/ppo/core_algos.py:2154:def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
./verl/trainer/ppo/core_algos.py:2155:    """Compute KL divergence given logprob and ref_logprob.
./verl/trainer/ppo/core_algos.py:2161:        ref_logprob:
./verl/trainer/ppo/core_algos.py:2167:        return logprob - ref_logprob
./verl/trainer/ppo/core_algos.py:2170:        return (logprob - ref_logprob).abs()
./verl/trainer/ppo/core_algos.py:2173:        return 0.5 * (logprob - ref_logprob).square()
./verl/trainer/ppo/core_algos.py:2178:        kl = ref_logprob - logprob
./verl/trainer/ppo/core_algos.py:2186:        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
./tests/trainer/ppo/test_core_algos_on_cpu.py:358:    ref_logprob = torch.randn(4, 8)
./tests/trainer/ppo/test_core_algos_on_cpu.py:360:    plus_value = kl_penalty(logprob, ref_logprob, name)
./tests/trainer/ppo/test_core_algos_on_cpu.py:361:    base_value = kl_penalty(logprob, ref_logprob, base)
./tests/trainer/ppo/test_core_algos_on_cpu.py:372:    ref_logprob = torch.randn(4, 8)
./tests/trainer/ppo/test_core_algos_on_cpu.py:374:    out_plus = kl_penalty(logprob, ref_logprob, "k3+").sum()
./tests/trainer/ppo/test_core_algos_on_cpu.py:378:    out_k2 = kl_penalty(logprob_k2, ref_logprob, "k2").sum()
./verl/trainer/distillation/losses.py:364:        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
./verl/workers/rollout/sglang_rollout/async_sglang_server.py:227:        LD_LIBRARY_PATH; wrong path ⇒ scheduler subprocess dies with SIGABRT."""
./verl/workers/utils/losses.py:135:        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
./research/runs/SUMMARY.md:11:| EXP-6 | M2 | Mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS: per-path counters train=28/all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean (0 leaked keys/4 shards), no NaN/Inf | PR #3 (draft) → `vast-ai-workload` |
./research/runs/SUMMARY.md:23:rollout/old_logprob/ref_logprob/val/infer/ckpt=0; old/ref log-prob bit-equal within 1e-6
./research/runs/EXP-6/verdict.md:7:- [x] Per-path mask counter `== 0` on rollout/old-logprob/ref-logprob/validation/checkpoint/infer paths (observed: key-prefix grep returns only `actor/comm_eff/mask_applications`; non-train-keyed falsifier grep is empty; unit tests `test_mask_hook_asserts_on_non_train_path[rollout|old_logprob|ref_logprob|val|infer|ckpt|None]` all PASSED)
./research/runs/EXP-6/verdict.md:9:- [x] `old_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: unit test `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — fixed-batch, deterministic; tag-inactive is the path old_log_prob recompute runs under)
./research/runs/EXP-6/verdict.md:10:- [x] `ref_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: same `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — ref_log_prob runs under the identical inactive-tag forward, mask asserts off per `test_mask_hook_asserts_on_non_train_path[ref_logprob]`)
./research/runs/EXP-6/verdict.md:36:- Mask confinement (the single most important falsifier per the plan) is fully and independently established: the key-prefix grep yields exactly one mask-applications key, `actor/comm_eff/mask_applications`, and the seven `test_mask_hook_asserts_on_non_train_path[...]` assertions turn any non-train activation into a hard failure rather than a silent counter.
./research/runs/EXP-6/verdict.md:37:- AUTO-RESUME caveat weighed and judged non-blocking: the mask_off cell reached `training/global_step:3` and shares the checkpoint dir + experiment_name (`m2-mask-invariants`) with mask_on, so verl almost certainly auto-resumed mask_off from mask_on's global_step_2 checkpoint. The live mask_off val number is therefore NOT a pristine masking-off-from-scratch reference, so criterion 5's val-PARITY sub-clause is only weakly supported by the live run (0.0508 vs 0.0485-0.0652, same ballpark on a 2-step smoke). It does NOT flip the verdict because: (a) criterion 5's first clause — "a validation pass ran and produced a score" — is hard-met; (b) the rigorous, deterministic log-prob equality at rel-tol 1e-6 is proven on a fixed batch by `test_logprob_equal_mask_on_vs_off_when_tag_inactive`, which is the exact mechanism that would make val differ if masking leaked into eval generation; (c) the EXP-6 hypothesis is mask CONFINEMENT, not val parity, and confinement is fully supported; (d) an auto-resume cannot manufacture a false confinement pass — it would, if anything, contaminate eval, yet the non-train counters are still 0. Recommend the next planner pin a unique `experiment_name`/checkpoint dir per cell on any future parity-sensitive rerun to harden the live val-parity evidence.
./research/findings/M2/codex-review.md.partial:24:- It leaves every RL-measurement path (rollout, old_logprob, ref_logprob, val, infer, ckpt) bit-unchanged (1e-6 log-prob equality).
./research/findings/M2/codex-review.md.partial:28:[Confinement] An explicit path-tag (train|rollout|old_logprob|ref_logprob|val|infer|ckpt) + an assert-on-wrong-path guard in the forward hook. Per-path counters: actor/comm_eff/mask_applications=28 on actor-train across two substeps; 0 (key absent) on every other path. Seven unit tests test_mask_hook_asserts_on_non_train_path[rollout|old_logprob|ref_logprob|val|infer|ckpt|None] PASSED — they assert that the hook RAISES if it ever fires under a non-train tag. Confinement at runtime is evidenced by a key-prefix grep on train.log: the only mask-applications metric emitted is actor/comm_eff/mask_applications; no non-train-keyed counter exists.
./research/findings/M2/codex-review.md.partial:29:[Measurement correctness] old_log_prob and ref_log_prob mask-on vs mask-off equal within rel-tol 1e-6, proven by ONE unit test test_logprob_equal_mask_on_vs_off_when_tag_inactive — described as "fixed-batch, deterministic". The argument is that old_logprob/ref_logprob recompute runs under the tag-INACTIVE forward, so the mask hook is inert there.
./research/findings/M2/codex-review.md.partial:31:[Checkpoint] live leakage scan loaded all 4 FSDP shards of global_step_2/actor and grepped keys for comm_eff|mask_applications|path_tag|anchor|spectral -> NONE. Plus 2 unit tests for a checkpoint guard.
./research/findings/M2/codex-review.md.partial:38:2. Is the assert-on-wrong-path guard a sufficient SILENT-FAILURE net? What failure modes would it NOT catch? (hook not installed on the eval model at all; eval running on a separate vLLM engine the hook never touches; tag set to train during an eval that is mislabeled; the assert being compiled out / swallowed under torch.compile or no_grad or inference_mode; the guard only checking tag value not whether mask was actually applied.)
./research/findings/M2/codex-review.md.partial:57:   altering the GRPO sequence (rollout → old_logprob → ref_logprob → reward →
./research/findings/M2/codex-review.md.partial:60:   (train|rollout|old_logprob|ref_logprob|val|infer|ckpt) + assert-on-wrong-path
./research/findings/M2/codex-review.md.partial:78:| EXP-6 | mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS — per-path counters train=28 / all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean, no NaN | #3 (draft) → vast-ai-workload |
./research/findings/M2/codex-review.md.partial:83:**EXP-6 headline:** per-path counters train=28, rollout/old_logprob/ref_logprob/val/infer/ckpt=0;
./research/findings/M2/codex-review.md.partial:154:   altering the GRPO sequence (rollout → old_logprob → ref_logprob → reward →
./research/findings/M2/codex-review.md.partial:157:   (train|rollout|old_logprob|ref_logprob|val|infer|ckpt) + assert-on-wrong-path
./research/findings/M2/codex-review.md.partial:175:| EXP-6 | mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS — per-path counters train=28 / all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean, no NaN | #3 (draft) → vast-ai-workload |
./research/findings/M2/codex-review.md.partial:180:**EXP-6 headline:** per-path counters train=28, rollout/old_logprob/ref_logprob/val/infer/ckpt=0;
./research/findings/M2/codex-review.md.partial:219:- [x] Per-path mask counter `== 0` on rollout/old-logprob/ref-logprob/validation/checkpoint/infer paths (observed: key-prefix grep returns only `actor/comm_eff/mask_applications`; non-train-keyed falsifier grep is empty; unit tests `test_mask_hook_asserts_on_non_train_path[rollout|old_logprob|ref_logprob|val|infer|ckpt|None]` all PASSED)
./research/findings/M2/codex-review.md.partial:221:- [x] `old_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: unit test `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — fixed-batch, deterministic; tag-inactive is the path old_log_prob recompute runs under)
./research/findings/M2/codex-review.md.partial:222:- [x] `ref_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: same `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — ref_log_prob runs under the identical inactive-tag forward, mask asserts off per `test_mask_hook_asserts_on_non_train_path[ref_logprob]`)
./research/findings/M2/codex-review.md.partial:248:- Mask confinement (the single most important falsifier per the plan) is fully and independently established: the key-prefix grep yields exactly one mask-applications key, `actor/comm_eff/mask_applications`, and the seven `test_mask_hook_asserts_on_non_train_path[...]` assertions turn any non-train activation into a hard failure rather than a silent counter.
./research/findings/M2/codex-review.md.partial:249:- AUTO-RESUME caveat weighed and judged non-blocking: the mask_off cell reached `training/global_step:3` and shares the checkpoint dir + experiment_name (`m2-mask-invariants`) with mask_on, so verl almost certainly auto-resumed mask_off from mask_on's global_step_2 checkpoint. The live mask_off val number is therefore NOT a pristine masking-off-from-scratch reference, so criterion 5's val-PARITY sub-clause is only weakly supported by the live run (0.0508 vs 0.0485-0.0652, same ballpark on a 2-step smoke). It does NOT flip the verdict because: (a) criterion 5's first clause — "a validation pass ran and produced a score" — is hard-met; (b) the rigorous, deterministic log-prob equality at rel-tol 1e-6 is proven on a fixed batch by `test_logprob_equal_mask_on_vs_off_when_tag_inactive`, which is the exact mechanism that would make val differ if masking leaked into eval generation; (c) the EXP-6 hypothesis is mask CONFINEMENT, not val parity, and confinement is fully supported; (d) an auto-resume cannot manufacture a false confinement pass — it would, if anything, contaminate eval, yet the non-train counters are still 0. Recommend the next planner pin a unique `experiment_name`/checkpoint dir per cell on any future parity-sensitive rerun to harden the live val-parity evidence.
./research/findings/M2/codex-review.md.partial:257:/bin/zsh -lc 'rg -n "mask_applications|path_tag|old_log_prob|ref_log_prob|auto-resume|resume|val-core|checkpoint|leak|1e-6|tag" research/findings/M2 -S' in /Users/shamane/Documents/verl
./research/findings/M2/codex-review.md.partial:259:research/findings/M2/codex-review.md.partial:24:- It leaves every RL-measurement path (rollout, old_logprob, ref_logprob, val, infer, ckpt) bit-unchanged (1e-6 log-prob equality).
./research/findings/M2/codex-review.md.partial:261:research/findings/M2/codex-review.md.partial:28:[Confinement] An explicit path-tag (train|rollout|old_logprob|ref_logprob|val|infer|ckpt) + an assert-on-wrong-path guard in the forward hook. Per-path counters: actor/comm_eff/mask_applications=28 on actor-train across two substeps; 0 (key absent) on every other path. Seven unit tests test_mask_hook_asserts_on_non_train_path[rollout|old_logprob|ref_logprob|val|infer|ckpt|None] PASSED — they assert that the hook RAISES if it ever fires under a non-train tag. Confinement at runtime is evidenced by a key-prefix grep on train.log: the only mask-applications metric emitted is actor/comm_eff/mask_applications; no non-train-keyed counter exists.
./research/findings/M2/codex-review.md.partial:262:research/findings/M2/codex-review.md.partial:29:[Measurement correctness] old_log_prob and ref_log_prob mask-on vs mask-off equal within rel-tol 1e-6, proven by ONE unit test test_logprob_equal_mask_on_vs_off_when_tag_inactive — described as "fixed-batch, deterministic". The argument is that old_logprob/ref_logprob recompute runs under the tag-INACTIVE forward, so the mask hook is inert there.
./research/findings/M2/codex-review.md.partial:264:research/findings/M2/codex-review.md.partial:31:[Checkpoint] live leakage scan loaded all 4 FSDP shards of global_step_2/actor and grepped keys for comm_eff|mask_applications|path_tag|anchor|spectral -> NONE. Plus 2 unit tests for a checkpoint guard.
./research/findings/M2/codex-review.md.partial:267:research/findings/M2/codex-review.md.partial:38:2. Is the assert-on-wrong-path guard a sufficient SILENT-FAILURE net? What failure modes would it NOT catch? (hook not installed on the eval model at all; eval running on a separate vLLM engine the hook never touches; tag set to train during an eval that is mislabeled; the assert being compiled out / swallowed under torch.compile or no_grad or inference_mode; the guard only checking tag value not whether mask was actually applied.)
./research/findings/M2/codex-review.md.partial:277:research/findings/M2/codex-review.md.partial:78:| EXP-6 | mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS — per-path counters train=28 / all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean, no NaN | #3 (draft) → vast-ai-workload |
./research/findings/M2/codex-review.md.partial:286:research/findings/M2/EXP-6.md:7:- [x] Per-path mask counter `== 0` on rollout/old-logprob/ref-logprob/validation/checkpoint/infer paths (observed: key-prefix grep returns only `actor/comm_eff/mask_applications`; non-train-keyed falsifier grep is empty; unit tests `test_mask_hook_asserts_on_non_train_path[rollout|old_logprob|ref_logprob|val|infer|ckpt|None]` all PASSED)
./research/findings/M2/codex-review.md.partial:288:research/findings/M2/EXP-6.md:9:- [x] `old_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: unit test `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — fixed-batch, deterministic; tag-inactive is the path old_log_prob recompute runs under)
./research/findings/M2/codex-review.md.partial:289:research/findings/M2/EXP-6.md:10:- [x] `ref_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: same `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — ref_log_prob runs under the identical inactive-tag forward, mask asserts off per `test_mask_hook_asserts_on_non_train_path[ref_logprob]`)
./research/findings/M2/codex-review.md.partial:299:research/findings/M2/EXP-6.md:36:- Mask confinement (the single most important falsifier per the plan) is fully and independently established: the key-prefix grep yields exactly one mask-applications key, `actor/comm_eff/mask_applications`, and the seven `test_mask_hook_asserts_on_non_train_path[...]` assertions turn any non-train activation into a hard failure rather than a silent counter.
./research/findings/M2/codex-review.md.partial:300:research/findings/M2/EXP-6.md:37:- AUTO-RESUME caveat weighed and judged non-blocking: the mask_off cell reached `training/global_step:3` and shares the checkpoint dir + experiment_name (`m2-mask-invariants`) with mask_on, so verl almost certainly auto-resumed mask_off from mask_on's global_step_2 checkpoint. The live mask_off val number is therefore NOT a pristine masking-off-from-scratch reference, so criterion 5's val-PARITY sub-clause is only weakly supported by the live run (0.0508 vs 0.0485-0.0652, same ballpark on a 2-step smoke). It does NOT flip the verdict because: (a) criterion 5's first clause — "a validation pass ran and produced a score" — is hard-met; (b) the rigorous, deterministic log-prob equality at rel-tol 1e-6 is proven on a fixed batch by `test_logprob_equal_mask_on_vs_off_when_tag_inactive`, which is the exact mechanism that would make val differ if masking leaked into eval generation; (c) the EXP-6 hypothesis is mask CONFINEMENT, not val parity, and confinement is fully supported; (d) an auto-resume cannot manufacture a false confinement pass — it would, if anything, contaminate eval, yet the non-train counters are still 0. Recommend the next planner pin a unique `experiment_name`/checkpoint dir per cell on any future parity-sensitive rerun to harden the live val-parity evidence.
./research/findings/M2/codex-review.md.partial:309:research/findings/M2/SUMMARY.md:34:| EXP-6 | mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS — per-path counters train=28 / all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean, no NaN | #3 (draft) → vast-ai-workload |
./research/findings/M2/codex-review.md.partial:319:research/findings/M2/EXP-5.md:9:- [x] `comm_eff/mask_applications` >0 on actor-train, ==0 on rollout/old-logprob/ref-logprob/validation/checkpoint/infer_batch (observed: the ONLY mask-applications metric emitted is `actor/comm_eff/mask_applications` = 14/28 on masked cells, 0 on disabled; zero non-actor-path-keyed mask counters exist in either masked log)
./research/findings/M2/codex-review.md.partial:322:research/findings/M2/EXP-5.md:25:`diff_against_baseline.py runs/EXP-5 --baseline EXP-3` wrote runs/EXP-5/baseline_diff.md. Per the plan, this entry point is used here only to confirm the disabled cell still tracks the dense path (EXP-4 no-op contract regression), not as a performance comparison — p95/staleness/communication metrics are n/a for this actor-only masking integration smoke. The disabled cell reproduces dense GRPO: all comm_eff counters 0, finite low grad_norm (1.13, 0.37, consistent with an un-masked KL-regularized actor vs the masked cells' inflated grad_norm), and reaches step 2. No fork of the GRPO algorithm: same rollout → old-logprob → ref-logprob → reward → advantage → update_actor → weight-sync sequence in all three cells.
./research/findings/M2/codex-review.md.partial:443:/bin/zsh -lc 'rg -n "test_mask_hook_asserts_on_non_train_path|test_logprob_equal_mask_on_vs_off_when_tag_inactive|CommEff|comm_eff|path_tag|mask_applications|activation_mask|mask_hook" tests verl -S' in /Users/shamane/Documents/verl
./research/LOG.md:4:M2 — mask contamination guard: invariants for rollout / old-logprob / ref-logprob / validation / checkpoint / infer_batch
./research/LOG.md:5:- hypothesis: per-path mask counter strictly 0 on rollout, old-logprob, ref-logprob, validation, checkpoint, infer paths; strictly > 0 only on actor-train forward/backward; old/ref log-probs equal within 1e-6 regardless of mask config; validation unchanged vs masking-off; checkpoints contain no comm_eff/mask state
./research/findings/M2/SUMMARY.md:13:   altering the GRPO sequence (rollout → old_logprob → ref_logprob → reward →
./research/findings/M2/SUMMARY.md:16:   (train|rollout|old_logprob|ref_logprob|val|infer|ckpt) + assert-on-wrong-path
./research/findings/M2/SUMMARY.md:34:| EXP-6 | mask contamination guard: explicit path-tag + assert-on-wrong-path + per-path counters + checkpoint mask-free | PASS — per-path counters train=28 / all-RL-paths=0; 35 unit tests incl 1e-6 logprob equality + checkpoint guard; live 2-step smoke val ran, ckpt leak-scan clean, no NaN | #3 (draft) → vast-ai-workload |
./research/findings/M2/SUMMARY.md:39:**EXP-6 headline:** per-path counters train=28, rollout/old_logprob/ref_logprob/val/infer/ckpt=0;
./research/findings/M2/EXP-6.md:7:- [x] Per-path mask counter `== 0` on rollout/old-logprob/ref-logprob/validation/checkpoint/infer paths (observed: key-prefix grep returns only `actor/comm_eff/mask_applications`; non-train-keyed falsifier grep is empty; unit tests `test_mask_hook_asserts_on_non_train_path[rollout|old_logprob|ref_logprob|val|infer|ckpt|None]` all PASSED)
./research/findings/M2/EXP-6.md:9:- [x] `old_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: unit test `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — fixed-batch, deterministic; tag-inactive is the path old_log_prob recompute runs under)
./research/findings/M2/EXP-6.md:10:- [x] `ref_log_prob` mask-on vs mask-off equal within rel-tol `1e-6` (observed: same `test_logprob_equal_mask_on_vs_off_when_tag_inactive` PASSED — ref_log_prob runs under the identical inactive-tag forward, mask asserts off per `test_mask_hook_asserts_on_non_train_path[ref_logprob]`)
./research/findings/M2/EXP-6.md:36:- Mask confinement (the single most important falsifier per the plan) is fully and independently established: the key-prefix grep yields exactly one mask-applications key, `actor/comm_eff/mask_applications`, and the seven `test_mask_hook_asserts_on_non_train_path[...]` assertions turn any non-train activation into a hard failure rather than a silent counter.
./research/findings/M2/EXP-6.md:37:- AUTO-RESUME caveat weighed and judged non-blocking: the mask_off cell reached `training/global_step:3` and shares the checkpoint dir + experiment_name (`m2-mask-invariants`) with mask_on, so verl almost certainly auto-resumed mask_off from mask_on's global_step_2 checkpoint. The live mask_off val number is therefore NOT a pristine masking-off-from-scratch reference, so criterion 5's val-PARITY sub-clause is only weakly supported by the live run (0.0508 vs 0.0485-0.0652, same ballpark on a 2-step smoke). It does NOT flip the verdict because: (a) criterion 5's first clause — "a validation pass ran and produced a score" — is hard-met; (b) the rigorous, deterministic log-prob equality at rel-tol 1e-6 is proven on a fixed batch by `test_logprob_equal_mask_on_vs_off_when_tag_inactive`, which is the exact mechanism that would make val differ if masking leaked into eval generation; (c) the EXP-6 hypothesis is mask CONFINEMENT, not val parity, and confinement is fully supported; (d) an auto-resume cannot manufacture a false confinement pass — it would, if anything, contaminate eval, yet the non-train counters are still 0. Recommend the next planner pin a unique `experiment_name`/checkpoint dir per cell on any future parity-sensitive rerun to harden the live val-parity evidence.
./research/findings/M2/EXP-5.md:9:- [x] `comm_eff/mask_applications` >0 on actor-train, ==0 on rollout/old-logprob/ref-logprob/validation/checkpoint/infer_batch (observed: the ONLY mask-applications metric emitted is `actor/comm_eff/mask_applications` = 14/28 on masked cells, 0 on disabled; zero non-actor-path-keyed mask counters exist in either masked log)
./research/findings/M2/EXP-5.md:25:`diff_against_baseline.py runs/EXP-5 --baseline EXP-3` wrote runs/EXP-5/baseline_diff.md. Per the plan, this entry point is used here only to confirm the disabled cell still tracks the dense path (EXP-4 no-op contract regression), not as a performance comparison — p95/staleness/communication metrics are n/a for this actor-only masking integration smoke. The disabled cell reproduces dense GRPO: all comm_eff counters 0, finite low grad_norm (1.13, 0.37, consistent with an un-masked KL-regularized actor vs the masked cells' inflated grad_norm), and reaches step 2. No fork of the GRPO algorithm: same rollout → old-logprob → ref-logprob → reward → advantage → update_actor → weight-sync sequence in all three cells.
./research/runs/EXP-6/verify/checkpoint_leakage_scan.md:7:and searched every state_dict key for `comm_eff | mask_applications | path_tag |

exec
/bin/zsh -lc "nl -ba verl/workers/engine_workers.py | sed -n '643,716p'" in /Users/shamane/Documents/verl
exec
/bin/zsh -lc "nl -ba tests/workers/comm_eff/test_activation_mask.py | sed -n '1,285p'" in /Users/shamane/Documents/verl
exec
/bin/zsh -lc "nl -ba verl/workers/engine/fsdp/transformer_impl.py | sed -n '611,675p'" in /Users/shamane/Documents/verl
 succeeded in 0ms:
   611	    def _comm_eff_mask_active(self, forward_only: bool) -> bool:
   612	        """True iff the activation-mask hooks should be live for this forward.
   613	
   614	        Masking is confined to the actor-train forward/backward. This returns
   615	        False (strict no-op) unless ALL of:
   616	          * this is a train pass (``not forward_only`` — never on infer_batch /
   617	            log-prob / ref / validation),
   618	          * an enabled ``CommEffState`` is attached,
   619	          * the worker has set ``state.mask_active`` (set only around
   620	            ``update_actor``; cleared everywhere else),
   621	          * a masker was constructed (mask sub-config enabled, ``p > 0``).
   622	        """
   623	        if forward_only:
   624	            return False
   625	        state = getattr(self, "_comm_eff_state", None)
   626	        if state is None or not getattr(state, "enabled", False):
   627	            return False
   628	        if not getattr(state, "mask_active", False):
   629	            return False
   630	        return getattr(state, "masker", None) is not None
   631	
   632	    def _comm_eff_register_mask_hooks(self) -> bool:
   633	        """Register the activation-mask forward hooks for this train forward.
   634	
   635	        Sets the PRF-key context (global step / optimizer-substep identity /
   636	        sequence-shard id) and installs the hooks on the boundary decoder
   637	        blocks. Returns True if hooks were registered (so the caller knows to
   638	        unregister on exit). The substep counter advances per call so the same
   639	        rollout batch reused across PPO mini-batches gets a distinct mask per
   640	        substep.
   641	        """
   642	        state = self._comm_eff_state
   643	        masker = state.masker
   644	        # global optimizer step (best-effort; threaded by the trainer when set).
   645	        global_step = int(getattr(self, "_comm_eff_global_step", 0))
   646	        # sequence-shard identity when Ulysses SP is active (else 0).
   647	        seq_shard = 0
   648	        if getattr(self, "ulysses_sequence_parallel_size", 1) and self.ulysses_sequence_parallel_size > 1:
   649	            try:
   650	                seq_shard = self.get_data_parallel_rank()
   651	            except Exception:
   652	                seq_shard = 0
   653	        masker.set_context(global_step=global_step, substep=state.substep, seq_shard=seq_shard)
   654	        masker.register(self.module)
   655	        # Advance the optimizer-substep identity for the next train forward.
   656	        state.substep += 1
   657	        return masker.is_registered
   658	
   659	    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
   660	        # comm_eff activation-mask hook lifecycle: register hooks on entry to the
   661	        # train forward/backward and remove them on exit, so a later log-prob /
   662	        # infer / ref / validation forward on the same module is clean. When
   663	        # disabled (default) or not on the actor-train path, nothing is registered
   664	        # and no RNG is drawn, so the pass is byte-identical to dense GRPO.
   665	        _mask_hooks_live = False
   666	        if self._comm_eff_mask_active(forward_only=forward_only):
   667	            _mask_hooks_live = self._comm_eff_register_mask_hooks()
   668	        try:
   669	            return self._forward_backward_batch_inner(data, loss_function, forward_only=forward_only)
   670	        finally:
   671	            if _mask_hooks_live:
   672	                self._comm_eff_state.masker.unregister()
   673	
   674	    def _forward_backward_batch_inner(
   675	        self, data: TensorDict, loss_function: Callable, forward_only=False

 succeeded in 0ms:
   643	    def _maybe_comm_eff_state(self):
   644	        """Return this worker's comm_eff state, building it once on first use.
   645	
   646	        Disabled is the strict no-op path: ``maybe_build_comm_eff_state`` returns
   647	        ``None`` without drawing RNG, allocating buffers or registering hooks, so
   648	        a dense GRPO run with the scaffolding merged is numerically identical to
   649	        one without it. The result is cached so the per-substep ``update_actor``
   650	        does not re-read the config each call.
   651	        """
   652	        state = getattr(self, "_comm_eff_state", None)
   653	        if state is None and not getattr(self, "_comm_eff_state_built", False):
   654	            comm_eff_cfg = self.config.actor.get("comm_eff", None)
   655	            state = maybe_build_comm_eff_state(comm_eff_cfg)
   656	            # object.__setattr__ avoids any frozen-config interplay; these are
   657	            # plain worker attributes, not config fields.
   658	            object.__setattr__(self, "_comm_eff_state", state)
   659	            object.__setattr__(self, "_comm_eff_state_built", True)
   660	            if state is None and not getattr(self, "_comm_eff_marker_logged", False):
   661	                logger.info("comm_eff: disabled (no-op) — dense GRPO path unchanged")
   662	                object.__setattr__(self, "_comm_eff_marker_logged", True)
   663	            if state is not None:
   664	                # Construct the masker (no hooks yet — the engine registers them
   665	                # only inside the train forward/backward) and attach the state to
   666	                # the underlying train engine so its forward-hook lifecycle and
   667	                # grad-correction hook can see it. The state is the single object
   668	                # shared between the worker (sets mask_active around update_actor)
   669	                # and the engine (registers/clears hooks gated on mask_active).
   670	                engine = getattr(getattr(self, "actor", None), "engine", None)
   671	                if engine is not None:
   672	                    state.build(getattr(engine, "module", None))
   673	                    object.__setattr__(engine, "_comm_eff_state", state)
   674	                    logger.info("comm_eff: enabled — mask circuit attached to actor train engine")
   675	        return getattr(self, "_comm_eff_state", None)
   676	
   677	    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
   678	    @DistProfiler.annotate(color="red", role="actor_update")
   679	    @_with_routing_replay_flag(enabled=True)
   680	    def update_actor(self, data: TensorDict) -> TensorDict:
   681	        # comm_eff guard. When disabled (default) this resolves to None with zero
   682	        # side effects (no hook, no buffer, no RNG) and the dense GRPO update runs
   683	        # exactly as upstream. The compressed circuits are entered only when
   684	        # comm_eff.enabled=true (later M2 work); the disabled path never touches
   685	        # the gradient, so the no-op parity holds.
   686	        comm_eff_state = self._maybe_comm_eff_state()
   687	
   688	        # Mask-active flag scope: set ONLY around the actor-train forward/backward
   689	        # so the masking forward-hooks fire exclusively on this path. The engine
   690	        # registers hooks on entry to its train forward_backward_batch and removes
   691	        # them on exit, gated on this flag; log_prob / infer / ref / validation /
   692	        # checkpoint forwards never set it, so they stay byte-identical to dense.
   693	        if comm_eff_state is not None:
   694	            comm_eff_state.mask_active = True
   695	        try:
   696	            output = self.actor.train_mini_batch(data=data)
   697	        finally:
   698	            if comm_eff_state is not None:
   699	                comm_eff_state.mask_active = False
   700	
   701	        # Surface the comm_eff operation counters into training metrics. When
   702	        # disabled we emit explicit zeros (mask_applications / anchor_backwards /
   703	        # spectral_corrections == 0) so the no-op is machine-checkable; emitting a
   704	        # constant metric is not a numerical side effect on training. `output` is
   705	        # None on non-output ranks (train_mini_batch only populates metrics on the
   706	        # mp-src rank), in which case there is nothing to annotate.
   707	        if output is not None:
   708	            if comm_eff_state is None:
   709	                counters = {
   710	                    "comm_eff/mask_applications": 0,
   711	                    "comm_eff/anchor_backwards": 0,
   712	                    "comm_eff/spectral_corrections": 0,
   713	                }
   714	            else:
   715	                counters = comm_eff_metrics(comm_eff_state)
   716	            metrics = tu.get(output, "metrics", default=None)

 succeeded in 0ms:
     1	# Copyright 2024 Bytedance Ltd. and/or its affiliates
     2	#
     3	# Licensed under the Apache License, Version 2.0 (the "License");
     4	# you may not use this file except in compliance with the License.
     5	# You may obtain a copy of the License at
     6	#
     7	#     http://www.apache.org/licenses/LICENSE-2.0
     8	#
     9	# Unless required by applicable law or agreed to in writing, software
    10	# distributed under the License is distributed on an "AS IS" BASIS,
    11	# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    12	# See the License for the specific language governing permissions and
    13	# limitations under the License.
    14	
    15	"""Unit tests for the comm_eff pipeline-boundary activation masker (EXP-5).
    16	
    17	These tests cover the masking-correctness properties the EXP-5 plan requires
    18	codex-verify to gate on, none of which need a GPU:
    19	
    20	* boundary indices == [1,3,5,7,9,11,13] for L=16 / pp_size=8, derived (not hardcoded);
    21	* PRF determinism: same key -> same mask, across calls;
    22	* value-independence: the mask depends only on the PRF key + shape, never on
    23	  the activation values;
    24	* measured mask ratio (zeroed fraction) tracks the configured p within tolerance;
    25	* in-graph form h_tilde = h * mask with NO 1/(1-p) rescale;
    26	* hook lifecycle: register installs hooks on boundaries only; unregister removes
    27	  them so a later forward is clean.
    28	"""
    29	
    30	import pytest
    31	import torch
    32	import torch.nn as nn
    33	
    34	from verl.workers.comm_eff.activation_mask import (
    35	    ActivationMasker,
    36	    decoder_boundary_indices,
    37	    find_decoder_layers,
    38	    prf_mask,
    39	)
    40	
    41	
    42	# --------------------------------------------------------------------------- #
    43	# boundary partition
    44	# --------------------------------------------------------------------------- #
    45	def test_boundary_indices_L16_pp8():
    46	    """The spec's canonical example: L=16 / pp_size=8 -> [1,3,5,7,9,11,13]."""
    47	    assert decoder_boundary_indices(16, 8) == [1, 3, 5, 7, 9, 11, 13]
    48	
    49	
    50	def test_boundary_indices_excludes_final_shard():
    51	    """The final shard's last block (the model's last decoder block) is never masked."""
    52	    idx = decoder_boundary_indices(16, 8)
    53	    assert 15 not in idx  # final block excluded
    54	    assert len(idx) == 7  # pp_size - 1 boundaries
    55	
    56	
    57	def test_boundary_indices_uneven_partition():
    58	    """Uneven L/pp_size: shards are near-even, larger shards come first."""
    59	    # L=10, pp_size=4 -> shard lens [3,3,2,2] -> last idx [2,5,7,9] -> drop 9 -> [2,5,7]
    60	    assert decoder_boundary_indices(10, 4) == [2, 5, 7]
    61	
    62	
    63	def test_boundary_indices_pp_size_one_is_empty():
    64	    assert decoder_boundary_indices(16, 1) == []
    65	
    66	
    67	def test_boundary_indices_pp_capped_at_num_layers():
    68	    # pp_size > L collapses to one block per shard; last shard dropped.
    69	    assert decoder_boundary_indices(4, 8) == [0, 1, 2]
    70	
    71	
    72	# --------------------------------------------------------------------------- #
    73	# PRF determinism + value-independence
    74	# --------------------------------------------------------------------------- #
    75	def test_prf_same_key_same_mask():
    76	    shape = (2, 8, 32)
    77	    key = (3, 1, 0, 0, 32, 7)
    78	    m1 = prf_mask(shape, key, 0.9, device=torch.device("cpu"), dtype=torch.float32)
    79	    m2 = prf_mask(shape, key, 0.9, device=torch.device("cpu"), dtype=torch.float32)
    80	    assert torch.equal(m1, m2)
    81	
    82	
    83	def test_prf_different_key_different_mask():
    84	    shape = (2, 8, 32)
    85	    a = prf_mask(shape, (3, 1, 0, 0, 32, 7), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    86	    # different substep component -> different mask
    87	    b = prf_mask(shape, (3, 2, 0, 0, 32, 7), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    88	    assert not torch.equal(a, b)
    89	
    90	
    91	def test_prf_mask_is_binary():
    92	    m = prf_mask((4, 16, 64), (1, 0, 0, 0, 64, 0), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    93	    uniq = set(m.unique().tolist())
    94	    assert uniq.issubset({0.0, 1.0})
    95	
    96	
    97	def test_mask_independent_of_activation_values():
    98	    """The mask must depend only on the PRF key + shape, never on h's values."""
    99	    masker = ActivationMasker(p=0.9, base_seed=7, pp_size=8)
   100	    layer_idx = 3
   101	    hook = masker._make_hook(layer_idx)
   102	
   103	    shape = (2, 8, 32)
   104	    h_zeros = torch.zeros(shape)
   105	    h_rand = torch.randn(shape)
   106	
   107	    masker.set_context(global_step=0, substep=0, seq_shard=0)
   108	    out_zeros = hook(nn.Identity(), (), h_zeros)
   109	    masker.set_context(global_step=0, substep=0, seq_shard=0)  # same key again
   110	    out_rand = hook(nn.Identity(), (), h_rand)
   111	
   112	    # Re-derive the mask directly from the key and confirm both inputs were
   113	    # multiplied by the SAME mask (value-independence).
   114	    key = (layer_idx, 0, 0, 0, 32, 7)
   115	    mask = prf_mask(shape, key, 0.9, device=torch.device("cpu"), dtype=torch.float32)
   116	    assert torch.equal(out_rand, h_rand * mask)
   117	    assert torch.equal(out_zeros, h_zeros * mask)
   118	
   119	
   120	# --------------------------------------------------------------------------- #
   121	# measured mask ratio tracks p
   122	# --------------------------------------------------------------------------- #
   123	@pytest.mark.parametrize("p", [0.90, 0.95])
   124	def test_mask_ratio_tracks_p(p):
   125	    # large tensor so the empirical zeroed fraction concentrates near p
   126	    shape = (8, 64, 256)
   127	    key = (5, 0, 0, 0, 256, 1)
   128	    m = prf_mask(shape, key, p, device=torch.device("cpu"), dtype=torch.float32)
   129	    measured_zero_fraction = float(1.0 - m.mean().item())
   130	    assert abs(measured_zero_fraction - p) <= 0.02
   131	
   132	
   133	# --------------------------------------------------------------------------- #
   134	# in-graph form: h_tilde = h * mask, no 1/(1-p) rescale, autograd-tracked
   135	# --------------------------------------------------------------------------- #
   136	def test_no_forward_rescale():
   137	    """Kept elements must equal h exactly (no 1/(1-p) scale-up)."""
   138	    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
   139	    masker.set_context(global_step=0, substep=0, seq_shard=0)
   140	    hook = masker._make_hook(3)
   141	    h = torch.full((2, 4, 16), 2.0)
   142	    out = hook(nn.Identity(), (), h)
   143	    # every nonzero output element equals exactly the input (2.0), not 2.0/(1-p)
   144	    nonzero = out[out != 0]
   145	    assert torch.allclose(nonzero, torch.full_like(nonzero, 2.0))
   146	
   147	
   148	def test_mask_is_in_graph():
   149	    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
   150	    masker.set_context(global_step=0, substep=0, seq_shard=0)
   151	    hook = masker._make_hook(3)
   152	    h = torch.randn(2, 4, 16, requires_grad=True)
   153	    out = hook(nn.Identity(), (), h)
   154	    out.sum().backward()
   155	    assert h.grad is not None  # gradient flows through the masked multiply
   156	
   157	
   158	def test_tuple_output_first_element_masked():
   159	    """HF decoder blocks return tuples; only the hidden state (elem 0) is masked."""
   160	    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
   161	    masker.set_context(global_step=0, substep=0, seq_shard=0)
   162	    hook = masker._make_hook(3)
   163	    h = torch.randn(2, 4, 16)
   164	    extra = torch.randn(2, 4, 16)
   165	    out = hook(nn.Identity(), (), (h, extra))
   166	    assert isinstance(out, tuple)
   167	    assert torch.equal(out[1], extra)  # second element untouched
   168	
   169	
   170	# --------------------------------------------------------------------------- #
   171	# decoder-layer discovery + hook lifecycle on a toy model
   172	# --------------------------------------------------------------------------- #
   173	class _ToyBlock(nn.Module):
   174	    def __init__(self, d):
   175	        super().__init__()
   176	        self.lin = nn.Linear(d, d)
   177	
   178	    def forward(self, x):
   179	        return self.lin(x)
   180	
   181	
   182	class _ToyDecoder(nn.Module):
   183	    def __init__(self, num_layers=16, d=32):
   184	        super().__init__()
   185	        self.layers = nn.ModuleList([_ToyBlock(d) for _ in range(num_layers)])
   186	
   187	    def forward(self, x):
   188	        for layer in self.layers:
   189	            x = layer(x)
   190	        return x
   191	
   192	
   193	def test_find_decoder_layers():
   194	    model = _ToyDecoder(num_layers=16, d=32)
   195	    layers = find_decoder_layers(model)
   196	    assert layers is not None
   197	    assert len(layers) == 16
   198	
   199	
   200	def test_register_installs_hooks_on_boundaries_only():
   201	    model = _ToyDecoder(num_layers=16, d=32)
   202	    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
   203	    masker.register(model)
   204	    assert masker.boundary_indices == [1, 3, 5, 7, 9, 11, 13]
   205	    assert masker.is_registered
   206	    # exactly the boundary blocks carry a forward hook
   207	    for i, layer in enumerate(model.layers):
   208	        has_hook = len(layer._forward_hooks) > 0
   209	        assert has_hook == (i in masker.boundary_indices)
   210	    masker.unregister()
   211	    assert not masker.is_registered
   212	    for layer in model.layers:
   213	        assert len(layer._forward_hooks) == 0
   214	
   215	
   216	def test_unregister_leaves_forward_clean():
   217	    """After unregister, a forward sees no masking (every element preserved)."""
   218	    torch.manual_seed(0)
   219	    model = _ToyDecoder(num_layers=16, d=32)
   220	    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
   221	    masker.set_context(global_step=0, substep=0, seq_shard=0)
   222	
   223	    x = torch.randn(2, 4, 32)
   224	    masker.register(model)
   225	    out_masked = model(x)
   226	    masker.unregister()
   227	    out_clean = model(x)
   228	    # the masked forward should differ from the clean forward (mask fired)
   229	    assert not torch.allclose(out_masked, out_clean)
   230	    # a second clean forward must reproduce the first clean forward exactly
   231	    out_clean2 = model(x)
   232	    assert torch.allclose(out_clean, out_clean2)
   233	
   234	
   235	def test_register_is_idempotent():
   236	    model = _ToyDecoder(num_layers=16, d=32)
   237	    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
   238	    masker.register(model)
   239	    n_handles = len(masker._handles)
   240	    masker.register(model)  # second call must not double-register
   241	    assert len(masker._handles) == n_handles
   242	    masker.unregister()
   243	
   244	
   245	def test_mask_applications_counter_increments():
   246	    """When a CommEffState is attached, each hook fire bumps mask_applications."""
   247	
   248	    class _FakeState:
   249	        def __init__(self):
   250	            self.mask_applications = 0
   251	
   252	    state = _FakeState()
   253	    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8, state=state)
   254	    masker.set_context(global_step=0, substep=0, seq_shard=0)
   255	    hook = masker._make_hook(3)
   256	    hook(nn.Identity(), (), torch.randn(2, 4, 16))
   257	    hook(nn.Identity(), (), torch.randn(2, 4, 16))
   258	    assert state.mask_applications == 2

exec
/bin/zsh -lc "nl -ba verl/workers/comm_eff/activation_mask.py | sed -n '220,330p'" in /Users/shamane/Documents/verl
 succeeded in 0ms:
   220	    u = torch.rand(shape, generator=gen, dtype=torch.float32)
   221	    keep = (u >= p).to(dtype=dtype)
   222	    return keep.to(device=device, non_blocking=True)
   223	
   224	
   225	class ActivationMasker:
   226	    """Registers/clears in-graph activation-mask forward hooks on boundary blocks.
   227	
   228	    One instance is owned by the engine. ``register(module)`` installs a forward
   229	    hook on each boundary decoder block; ``unregister()`` removes them. The hooks
   230	    must be live **only** during the actor train forward/backward — the engine
   231	    registers on entry to ``forward_backward_batch`` (train) and unregisters on
   232	    exit, so log-prob / ref / infer / validation / checkpoint forwards never see
   233	    a mask.
   234	
   235	    The PRF key per hook fire is composed from:
   236	      * the boundary block index (stable per hook),
   237	      * ``global_step`` (trainer optimizer step),
   238	      * ``substep`` (optimizer-substep / microbatch identity within the step),
   239	      * a sequence-shard id (0 when no SP; set by the engine when present),
   240	      * ``hidden_size`` (last dim of the activation),
   241	      * ``base_seed`` (``comm_eff.mask.seed``).
   242	
   243	    ``global_step`` / ``substep`` / ``seq_shard`` are set by the engine via
   244	    ``set_context(...)`` before each forward so the same rollout batch reused
   245	    over multiple PPO mini-batches gets distinct masks per substep.
   246	    """
   247	
   248	    def __init__(self, *, p: float, base_seed: int, pp_size: int, state: Any = None):
   249	        self.p = float(p)
   250	        self.base_seed = int(base_seed)
   251	        self.pp_size = int(pp_size)
   252	        self._state = state  # CommEffState, for the mask_applications counter
   253	        self._handles: list[Any] = []
   254	        self._boundary_set: set[int] = set()
   255	        self.boundary_indices: list[int] = []
   256	        # Per-forward context, set by the engine before forward_backward.
   257	        self._global_step = 0
   258	        self._substep = 0
   259	        self._seq_shard = 0
   260	        # Last-measured masked fraction per boundary, surfaced as comm_eff/mask_ratio.
   261	        self.last_mask_ratio: dict[int, float] = {}
   262	
   263	    def set_context(self, *, global_step: int, substep: int, seq_shard: int = 0) -> None:
   264	        """Set the PRF-key context for the next forward pass."""
   265	        self._global_step = int(global_step)
   266	        self._substep = int(substep)
   267	        self._seq_shard = int(seq_shard)
   268	
   269	    def _make_hook(self, layer_idx: int):
   270	        masker = self
   271	
   272	        def _hook(_mod: nn.Module, _inputs: tuple, output: Any):
   273	            # HF decoder blocks return either a Tensor or a tuple whose first
   274	            # element is the hidden state. Mask the hidden state in-graph.
   275	            if isinstance(output, tuple):
   276	                h = output[0]
   277	            else:
   278	                h = output
   279	            if not torch.is_tensor(h):
   280	                return output
   281	            hidden_size = h.shape[-1]
   282	            key = (
   283	                layer_idx,
   284	                masker._global_step,
   285	                masker._substep,
   286	                masker._seq_shard,
   287	                hidden_size,
   288	                masker.base_seed,
   289	            )
   290	            mask = prf_mask(tuple(h.shape), key, masker.p, device=h.device, dtype=h.dtype)
   291	            # h_tilde = h * mask, in-graph (no 1/(1-p) rescale). The multiply is
   292	            # tracked by autograd so the masked gradient flows to the optimizer.
   293	            h_tilde = h * mask
   294	            # Instrumentation (does not affect the graph): measured masked fraction.
   295	            with torch.no_grad():
   296	                masker.last_mask_ratio[layer_idx] = float(1.0 - mask.mean().item())
   297	            if masker._state is not None:
   298	                masker._state.mask_applications += 1
   299	            if isinstance(output, tuple):
   300	                return (h_tilde,) + tuple(output[1:])
   301	            return h_tilde
   302	
   303	        return _hook
   304	
   305	    def register(self, module: nn.Module) -> None:
   306	        """Install forward hooks on the boundary decoder blocks of ``module``.
   307	
   308	        Idempotent guard: if hooks are already registered this is a no-op (the
   309	        engine pairs register/unregister, but a defensive guard avoids double
   310	        registration leaking a mask onto a later pass).
   311	        """
   312	        if self._handles:
   313	            return
   314	        layers = find_decoder_layers(module)
   315	        if layers is None:
   316	            logger.warning(
   317	                "comm_eff.activation_mask: could not locate decoder layers on %s; "
   318	                "no mask hooks registered (masking is a no-op this pass)",
   319	                type(module).__name__,
   320	            )
   321	            return
   322	        num_layers = len(layers)
   323	        self.boundary_indices = decoder_boundary_indices(num_layers, self.pp_size)
   324	        self._boundary_set = set(self.boundary_indices)
   325	        for idx in self.boundary_indices:
   326	            handle = layers[idx].register_forward_hook(self._make_hook(idx))
   327	            self._handles.append(handle)
   328	        logger.info(
   329	            "comm_eff.activation_mask: registered mask hooks on boundaries %s "
   330	            "(L=%d, pp_size=%d, p=%.4f)",

codex
ADVERSARIAL: CONTESTED

The evidence supports a narrow integration smoke, not the stated claims of “provably confined” and “every RL-measurement path bit-unchanged.” As written, the milestone overclaims.

**Highest-Severity Holes**
- The confinement proof is partly circular. The main runtime falsifier is the same instrumentation whose correctness is under test. A missing non-train `mask_applications` key proves no logged non-train counter, not necessarily no non-train perturbation.
- The local code I can inspect does not show the advertised path-tag/assert guard in the hook. The hook masks and increments the counter in [activation_mask.py](/Users/shamane/Documents/verl/verl/workers/comm_eff/activation_mask.py:272); gating appears to come from `forward_only` and `mask_active` in [transformer_impl.py](/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:611) and [engine_workers.py](/Users/shamane/Documents/verl/verl/workers/engine_workers.py:688). If the draft PR has additional code not in this worktree, this is a traceability gap; if not, the summary is inaccurate.
- The counter can miss real failures: hook installed on an unlogged rank, eval mislabeled as train, stale `mask_active=True`, hook registration failure, separate rollout/vLLM engine not covered, metric emission dropped, or non-output ranks not contributing counters.
- The assert-on-wrong-path net only catches “hook fires while tag says non-train.” It does not catch missing hooks, missing guards, wrong tags, stale train tags, eval running through a train-labeled path, vLLM/rollout engines that never see the hook, or masking that occurs before/after the guarded region.

**Measurement-Invariance Weaknesses**
- The 1e-6 log-prob test is under-powered and close to tautological. If the hook is gated inactive, “mask-on config with inactive tag == mask-off” mostly tests the gate condition, not the real old/ref measurement pipeline.
- One deterministic fixed batch does not represent GSM8K lengths, padding/rmpad behavior, bf16, FSDP sharding, sequence parallelism, real Qwen boundary layers `[3,7,11,15,18,21,24]`, rollout sync, validation generation, value model paths, or checkpoint/reload behavior.
- “1e-6 equality” is not “bit-unchanged.” It is a tolerance check on old/ref log-probs only. It does not cover rollout, val, infer, value, reward inputs, KL/advantage downstream tensors, or live distributed paths.
- A test could pass if masking is never installed, if the tested fixture is too small, or if it bypasses the production actor/ref/rollout split.

**Baseline And Control Problems**
- The `mask_off` auto-resume undermines more than val parity. It means the “mask_off” cell is not an independent dense control; it is a continuation from masked-trained model, optimizer, RNG, scheduler, and checkpoint state.
- Therefore its `0` mask counter and finite grads only show “no new mask counter after resume under disabled config,” not “dense GRPO control is unperturbed.”
- The claim that auto-resume “cannot manufacture a false confinement pass” is too strong. It can hide the absence of an independent control and can make downstream eval look dense while inheriting masked-training effects.
- “Same ballpark” GSM8K validation on a 2-step smoke is not statistical evidence. No confidence interval, no fixed fresh paired seed, no predeclared tolerance, and the compared cells are at different histories/steps.

**P-Hacking Risks**
- The val criterion appears post-hoc softened to “same ballpark.”
- The plan grep missed the actual metric name, then the emitted key was substituted. That may be reasonable operationally, but it weakens preregistration.
- Absence of metric keys is treated as zero, despite known log-format pitfalls from EXP-5.
- Multiple weak signals are combined into PASS: 35 unit tests, finite grad norm, no NaN grep, 2-step completion, key grep clean. Most do not independently test measurement invariance.

**Checkpoint Claim**
- The checkpoint scan supports “no obvious comm_eff-named tensors in actor FSDP model shards.” It does not prove all checkpoint artifacts are mask-free unless optimizer, scheduler, RNG, trainer metadata, latest-step files, and alternate names are scanned too.
- “Mask-free checkpoint” should not be conflated with “free of masking influence.” The trained weights are expected to contain effects of masked actor updates.

**Unsupported Wording To Fix**
- Replace “provably confined” with “not observed to fire outside actor-train in the tested smoke and unit fixtures.”
- Replace “every RL-measurement path bit-unchanged” with “one fixed-batch old/ref log-prob fixture matched within 1e-6 when the mask gate was inactive.”
- Replace “val parity within noise” with “live val produced scores, but parity evidence is contaminated by auto-resume.”

Minimum hardening: fresh no-resume paired mask_on/mask_off runs; end-to-end old/ref/val/infer equality on production FSDP/bf16 batches; rank-aggregated per-path counters; a real path-tag guard visible in the hook; negative-control tests that deliberately mislabel paths; and full checkpoint artifact scans.

ADVERSARIAL: CONTESTED
