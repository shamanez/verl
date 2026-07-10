# Verdict — 64-middle-block-freeze-grpo (issue #64)

VERDICT: PASS

8-cell replication matrix {frozen (train ONLY decoder block L11-15), dense (full-param)}
x {gsm8k, bigmath} x {data.seed 42, 7}, comm-eff OFF on all, 75 steps, on branch
exp/64-dense-wandbfix (freeze-hook + #65 wandb final-step fix). Plus base-model val_only
on both datasets for S_base. Judged against run.json `success_criteria` + `pass_rule`
("PASS iff all gates green; a clean symmetric negative — C<0.80 with gates green — is PASS").

## C(block) = (S_frozen - S_base) / (S_dense - S_base)

| dataset | seed | S_frozen | S_dense | S_base | C(block) | source (S_frozen / S_dense) |
|---------|------|----------|---------|--------|----------|------------------------------|
| gsm8k   | 42   | 0.766490 | 0.777104 | 0.077331 | **0.985** | frozen-gsm8k-s42.valcore / dense-gsm8k-s42.valcore (step:75) |
| gsm8k   | 7    | 0.744503 | 0.780895 | 0.077331 | **0.948** | frozen-gsm8k-s7.valcore / dense-gsm8k-s7.valcore (step:75) |
| bigmath | 42   | 0.580    | 0.606   | 0.538  | **0.618** | frozen-bigmath-s42.valcore / dense-bigmath-s42.valcore (step:75) |
| bigmath | 7    | 0.580    | 0.614   | 0.538  | **0.553** | frozen-bigmath-s7.valcore / dense-bigmath-s7 (incoming.log:6749, step:75) |

C values reproduce metrics/results_summary.json exactly (0.985 / 0.948 / 0.618 / 0.553).
S_base: gsm8k 0.0773313115996967 (results_summary.json; raw val_only in wandb 64-base-gsm8k-v2),
bigmath 0.538 (greppable at incoming.log step:0, and results_summary.json).

Seed spread:
- gsm8k: frozen 0.766490 vs 0.744503 = 0.0220 abs; C spread 0.985 vs 0.948 = 0.0366.
- bigmath: frozen 0.580 vs 0.580 = 0.000 abs (identical); C spread 0.618 vs 0.553 = 0.0650
  (spread is driven entirely by the dense denominator: S_dense 0.606 s42 vs 0.614 s7, since frozen is identical).

## Success criteria (gates)

- [x] **All 8 cells reach step 75, step-75 val lands.** 7/8 have a step:75 line in
  metrics/*.valcore.txt; dense-bigmath-s7 step:75=0.614 is in metrics/incoming.log:6749
  ("Training Progress: 100%|...| 75/75", training/global_step:75). All 8 have a valid
  step-75 val. Source: the 10 *.valcore.txt files + incoming.log:6749.
- [x] **No NaN / non-finite grads.** NaN/inf scan over incoming.log = 0 hits; every
  actor/grad_norm finite and small (range ~0.064–0.166 across all logged steps; step-75
  grad_norm 0.162 on dense-bigmath-s7). Gate green.
- [x] **Freeze-correctness (frozen cells).** launch_matrix.sh wires TRAIN_LAYERS=11-15 on
  all 4 frozen cells (lines 91-99, 120-124); dense cells leave it unset and correctly emit
  the "[TRAIN_LAYERS] unset/empty ... DENSE" guard (6 occurrences in incoming.log) — its
  absence on frozen cells corroborates freeze active. Structural proof (per operator brief,
  now off-box): optimizer ckpt = 234M / 15.2% params, 60 tensors. The freeze-ACTIVE
  logger.info marker is a known un-captured launcher gap, not a failure.
- [x] **C(block) per dataset per seed, 2-seed error bar.** Computed above; PASS predicate
  is dataset-split (see conclusion). Both splits are gates-green clean results.
- [x] **S_dense = fresh dense-<dataset>-s<seed> step-75 val** (measured natively, not the
  old 0.7657 fixed ref). Old cross-checks match: freeze-block-l11-15-gsm8k step75=0.7627
  (run.json cross_check 0.7627), freeze-block-l11-15-bigmath step75=0.568 (cross_check 0.568).

## Metrics summary

- comm-eff OFF verified structurally: all runtime actor/comm_eff/* counters = 0.0
  (mask_applications, anchor_backwards, spectral_corrections, powersgd_applications) at
  step 75; config trace has actor_rollout_ref.actor.comm_eff.enabled=false.
- Vanilla GRPO confirmed (resolved override trace, incoming.log:6126, last-wins):
  algorithm.adv_estimator=grpo, use_kl_in_reward=False, actor.use_kl_loss=False,
  entropy_coeff=0 — no-KL / no-entropy, as required by the fixed control.
- gsm8k learns the reward format fast: frozen step-25 val already ~0.727–0.735 vs S_base 0.077.
- bigmath advantages non-degenerate at step 75 (critic/advantages spans ±2.47, reward mean 0.52).

## Baseline comparison

Dense (full-param, comm-eff OFF) is the in-matrix apples-to-apples baseline (baseline_run:
"baseline"). Frozen-vs-dense at step 75: gsm8k frozen recovers 94.8–98.5% of the dense RL
gain over base; bigmath frozen recovers 55.3–61.8%. Old fixed-ref cross-checks
(S_full 0.7657 gsm8k; old frozen 0.7627 / 0.568) are consistent with the fresh natives.

## Resolved-params excerpt / provenance

See resolved_params.txt + resolved_cmd.txt. RESOLVED_CONFIG_MISSING: the canonical
`python3 -m verl.trainer.main_ppo` set -x line was not synced locally (no train.log; box
44376214 torn down), so capture_resolved_config.py produced nothing — params were
reconstructed from the trainer's own Hydra override dump (incoming.log:6126) + launch_matrix.sh.
No plan-vs-ran divergence found: adv_estimator=grpo, comm_eff.enabled=false, TRAIN_LAYERS=11-15
(frozen), model Qwen2.5-1.5B-Instruct, 75 steps, seeds {42,7} — all match the run.json cells[] spec.

## Notes

- RESOLVED_CONFIG_MISSING (see above) — provenance recovered from log dumps + launcher;
  ground truth is consistent, but the set -x main_ppo trace itself is off-box.
- Benign, NOT failures (per operator brief, independently corroborated in-log):
  1. All 8 cells' fail_*.flag (rc=1) are a wandb/DataLoader atexit teardown race AFTER
     step 75 + val; every step-75 val is present, so the gate is judged on vals not flags.
     (The flags were on the remote box; not present locally.)
  2. First dense-bigmath-s7 attempt (PID 125495) crashed in _save_checkpoint with
     "RuntimeError: basic_ios::clear: iostream error" + Ray "No space left on device"
     (incoming.log:6126-6404) — a disk-full infra crash on checkpoint write, NOT numerical
     divergence. The 8 cells filled the 200G disk; checkpoints were deleted and the affected
     cells re-run on a clean disk. The dense-bigmath-s7 re-run (PID 159834) is clean at step 75.
  3. base-gsm8k S_base was likewise re-measured post-crash (wandb 64-base-gsm8k-v2); clean.
- S_base format-artifact caveat: GSM8K S_base=0.077 is low because base Qwen2.5-1.5B-Instruct
  does not emit the `####` answer format the GSM8K reward extractor requires; RL learns the
  format within ~25 steps (step-25 val ~0.73). So GSM8K C~0.98 does NOT mean "the block ~=
  full model capability" — it means the block recovers ~all of the (largely format-driven) RL
  gain. Big-Math base already emits \boxed (S_base=0.538), so its C reflects genuine reasoning
  gain. C is internally consistent per dataset (identical eval config for base / frozen / dense).

## Scientific conclusion

Training only the middle decoder block (L11-15) of Qwen2.5-1.5B-Instruct recovers essentially
all of the full-parameter GRPO gain on GSM8K (C = 0.985 / 0.948 at seeds 42 / 7; frozen seed
spread ~0.022 in raw val), but only about half of it on Big-Math (C = 0.618 / 0.553), a clean,
gates-green negative on the harder split. The result is therefore dataset-dependent: the middle
block is sufficient to carry the (format-dominated) RL improvement on GSM8K, but for genuine
multi-step math reasoning (Big-Math) roughly 40-45% of the dense gain requires parameters
outside L11-15. Both splits pass all gates (all cells reach step 75, no NaN/non-finite grads,
freeze-correctness holds), so per the plan's predicate — a clean symmetric negative with gates
green is a PASS — the experiment cleanly answered its question. VERDICT: PASS.
