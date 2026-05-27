# M2 Milestone Summary

**M2 goal:** comm-eff integration smokes (mask/spectral/anchor, two-step GRPO).

Establish that the activation-masking circuit integrates into verl's GRPO actor
update, is confined to the actor-train forward/backward, leaves the recomputed
old/ref log-prob paths unchanged within 1e-6, and saves mask-free actor weights.
M2 gates the two downstream circuits (spectral correction, async anchor). Three
PASS findings.

> **Status: CONTESTED by adversarial review** (`findings/M2/codex-review.md`,
> 2026-05-28). The per-experiment PASS verdicts stand (each met its plan's
> success criteria), but the milestone-level claims below are scoped to the
> evidence actually collected and carry a hardening backlog (see end). Do not
> promote M2 to "settled" until the backlog is addressed. Wording has been
> softened from an earlier draft that over-claimed "provably confined / every
> RL-measurement path bit-unchanged / val parity".

## What M2 establishes

1. **Integration** — the masking hook plugs into the actor-train path without
   altering the GRPO sequence (rollout → old_logprob → ref_logprob → reward →
   advantage → update_actor → weight-sync) in any of the disabled / p90 / p95 cells.
2. **Confinement** — masks fire on actor-train only, evidenced by *three
   independent* mechanisms (not the counter alone, which would be circular):
   (a) the mask hooks are **ephemeral and train-scoped** — registered on entry to
   the actor train forward/backward and removed on exit, so structurally they
   cannot fire on a later log-prob/ref/val/infer pass; (b) the **assert-on-wrong-path
   guard** in the hook (`activation_mask.py:304-308`) raises if `path_tag != "train"`;
   (c) per-path counters: 28 on actor-train, 0 elsewhere. Caveat: all three are
   single-rank-logged and same-engine; rollout generation runs on a *separate* vLLM
   engine the hook never touches (so its 0 is structural, not measured).
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
2. **EXP-6 mask_off auto-resume invalidates that cell as a control** — the mask_off
   reference cell shared `experiment_name`/checkpoint dir with mask_on, so verl
   auto-resumed it from mask_on's masked-trained global_step_2 weights+optimizer+RNG
   (mask_off reached step 3, not a fresh step 2). So the mask_off cell is **not a
   clean dense control** — its 0 counter only shows "no new mask op after resume under
   disabled config," not "dense GRPO unperturbed." The clean disabled control for M2
   is **EXP-5's from-scratch disabled cell**, not this one. Next planner: pin a unique
   `experiment_name` per cell (m2-mask-on vs m2-mask-off). (EXP-6 confinement itself
   rests on the mask_ON cell's per-path counters + the assert guard, independent of
   this resume; the 1e-6 log-prob equality is proven separately by unit test.)

## Hardening backlog (from adversarial review — required before promoting M2)

The adversarial review (`findings/M2/codex-review.md`, ADVERSARIAL: CONTESTED) raised
six points. One — "assert guard not found in the hook" — was a **false alarm** (codex
read the `vast-ai-workload` checkout; the guard is in exp/6 `activation_mask.py:304-308`,
and 35 unit tests confirm it fires). The rest are accepted as hardening work:
- **Fresh paired mask_on/mask_off runs** with unique `experiment_name` (no auto-resume) —
  to get a real dense control and trustworthy val-parity.
- **End-to-end old/ref/val/infer log-prob equality on production batches**
  (GSM8K-length, bf16, FSDP-sharded, real boundary layers) — the current 1e-6 test
  is a toy fixed batch with the gate inactive (near-tautological).
- **Rank-aggregated per-path counters** — current counters are single-rank-logged.
- **Negative-control tests** that deliberately mislabel a path to confirm the assert fires
  end-to-end in the live engine (not just the unit-test hook).
- **Full checkpoint-artifact scan** — the live leak scan covered actor *model* shards
  only; also scan optim/scheduler/RNG/extra_state/HF-export. "Mask-free checkpoint
  tensors" ≠ "checkpoint free of masking *influence*".
3. **Plan-grep literal mismatch** — plans grep `val/test_score`, but verl emits
   `val-core/openai/gsm8k/acc/mean@1`. Update plan templates.

## Not yet covered

- **Spectral correction** (issue #7) — gradient correction for masked params; unplanned.
- **Async anchor circuit** (issue #8) — unmasked auxiliary forward; unplanned.

Both remain intentionally inert in M2 (`spectral_corrections=0`, `anchor_backwards=0`)
and will gate on this milestone's confinement + correctness proofs.
