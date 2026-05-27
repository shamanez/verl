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
