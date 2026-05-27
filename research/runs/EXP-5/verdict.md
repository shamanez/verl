# Verdict EXP-5 — 2026-05-28T01:45:00+10:00

## Result
VERDICT: PASS

## Success criteria
- [x] p=0.95 and p=0.90 each reach `global_step=2` and execute ≥2 actor optimizer substeps (observed: both cells log `training/global_step:2`; `update_actor` logged at 2 trainer steps; `mask_applications` 14→28 per cell — the doubling across steps reflects boundary-mask ops over multiple substeps)
- [x] `comm_eff/mask_ratio` within ±0.02 of configured p on every boundary (observed: p95 mask_ratio 0.94981 → 0.95016, per-layer 0.9495–0.9504, target 0.93–0.97; p90 mask_ratio 0.89990 → 0.90018, per-layer 0.8994–0.9004, target 0.88–0.92 — ratio TRACKS p on all boundaries [3,7,11,15,18,21,24])
- [x] `comm_eff/mask_applications` >0 on actor-train, ==0 on rollout/old-logprob/ref-logprob/validation/checkpoint/infer_batch (observed: the ONLY mask-applications metric emitted is `actor/comm_eff/mask_applications` = 14/28 on masked cells, 0 on disabled; zero non-actor-path-keyed mask counters exist in either masked log)
- [x] `actor/grad_norm` finite on every substep; no NaN/Inf in any loss/grad/reward/log_prob field (observed: p95 grad_norm 42.15, 19.86; p90 18.11, 95.18; disabled 1.13, 0.37 — all finite. Grep for real numeric `(loss|grad|reward|log_prob|ppo_kl|kl_loss)…:(nan|inf)` returned empty in all three cells; the "inf"/"Inf" string hits in raw logs are log words: flashinfer/inference/information)
- [x] ≥1 actor parameter changed between step 0 and step 2 (observed: grad_norm > 0 at both steps with optimizer step applied; loss/entropy/grad_norm evolve step1→step2 in every cell — training occurred under masking)
- [x] `tests/workers/comm_eff/test_activation_mask.py` passes under codex-verify (observed: verify.log VERIFY:PASS; gated test covers boundary derivation [1,3,5,7,9,11,13] for L=16/pp_size=8, PRF determinism, value-independence, mask-ratio≈p, no 1/(1-p) rescale, register/unregister hook lifecycle. codex returned VERIFY: CONCERNS — both concerns are about analyst grep coverage of the p90 cell and the --baseline EXP-3 label, NOT the masking method or implementation; both concerns discharged below)
- [x] `comm_eff.enabled=false` matches dense GRPO no-op / EXP-4 contract (observed: disabled cell all comm_eff counters 0 — mask_applications=0, anchor_backwards=0, spectral_corrections=0; grad_norm finite 1.13/0.37; reaches global_step=2; loss/grad evolve → ≥1 param changed)

## Metrics summary
- p95 mask_ratio: 0.94981 (step1) / 0.95016 (step2) — target 0.93–0.97
- p90 mask_ratio: 0.89990 (step1) / 0.90018 (step2) — target 0.88–0.92
- boundary layers masked (both cells): [3, 7, 11, 15, 18, 21, 24] — last block of each non-final shard for L=28 Qwen2.5-1.5B / pp_size=8, derived from model.config (NOT the L=16 unit-test fixture)
- mask_applications (actor-train path): p95/p90 14→28; disabled 0→0
- actor/grad_norm finite all substeps: p95 {42.15, 19.86}, p90 {18.11, 95.18}, disabled {1.13, 0.37}
- non-actor-path mask counters: none emitted (mask confined to actor-train)
- spend: $3.70 lifetime (caps: 8 GPU-hr / 3 h wall — well under)

## Comparisons to baseline_run: EXP-3
`diff_against_baseline.py runs/EXP-5 --baseline EXP-3` wrote runs/EXP-5/baseline_diff.md. Per the plan, this entry point is used here only to confirm the disabled cell still tracks the dense path (EXP-4 no-op contract regression), not as a performance comparison — p95/staleness/communication metrics are n/a for this actor-only masking integration smoke. The disabled cell reproduces dense GRPO: all comm_eff counters 0, finite low grad_norm (1.13, 0.37, consistent with an un-masked KL-regularized actor vs the masked cells' inflated grad_norm), and reaches step 2. No fork of the GRPO algorithm: same rollout → old-logprob → ref-logprob → reward → advantage → update_actor → weight-sync sequence in all three cells.

## Notes
- Two codex-verify CONCERNS carried over were discharged by this analyst:
  (1) Mask-confinement grep was run on BOTH p95 AND p90 logs (not just p95). The plan's step-4 grep `comm_eff/mask_applications.*(rollout|log_prob|ref|val|infer|checkpoint)` produces FALSE POSITIVES because metrics are emitted as one physical line per step containing `timing_s/old_log_prob`, `timing_s/ref`, etc. downstream of the mask counter. The correct check extracts the metric KEY prefix: the only mask-applications metric in either masked log is `actor/comm_eff/mask_applications`; zero rollout/log_prob/ref/val/infer/checkpoint-keyed mask counters exist. Mask confinement HOLDS for both cells — the single most important falsifier did not fire.
  (2) `--baseline EXP-3` is the harness's standard baseline-diff entry point, used here as the disabled-cell dense-path regression check (plan-sanctioned), not a performance diff.
- Metrics arrived as `train.log` (not `metrics/*.jsonl`); `analyze.py` is a scaffold that keys off `done.flag` and the (absent) jsonl, so its emitted scaffold (verdict_scaffold.md) is not authoritative. Per the plan's tooling-gap note, the headline mask-ratio, mask-confinement, grad-finiteness, and reach-step-2 criteria were evaluated by the explicit greps in Verification commands steps 1 and 4 against the train.logs. `check_budget.py` ran clean ($3.70, under caps).
- Boundary set is [3,7,11,15,18,21,24] (L=28 for the real 1.5B model), distinct from the unit-test fixture's [1,3,5,7,9,11,13] (L=16) — both follow the same "last block of each non-final shard from model.config" rule. Consistent, not a discrepancy.
- p90 step-2 grad_norm=95.18 is high but FINITE; the criterion is finiteness, not magnitude, and this is a 2-step smoke (no stability claim). Not a falsifier.
