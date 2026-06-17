# Verdict EXP-34 — 2026-06-18T01:10:02+10:00

## Result
VERDICT: REVISE

The headline predicate is met **on the letter** (best completed cell val@50 =
0.7635 > the +0.024 bar of 0.7511), and all config-provenance / correctness /
account criteria pass. But with `seed_replicates=1` and best-of-3 max-selection,
the +0.0364 margin rests on a single noisy draw whose within-cell draw spread
(0.0228, b0p25 val@50 vs val@55) is nearly as wide as the margin over the bar.
Two consistent positive draws (b0p25 +0.0341, b0p50 +0.0364) are a genuine
*signal* that β_anc>0 lifts signed_ema — contrary to EXP-33's flat β curve on
delayed_ef — but promoting signed_ema β=0.50 over the β=0 reference on a single
best-of-3 draw is premature. A replicate at β=0.50 cleanly converts signal →
promotable result (or collapses it to a noise-bounded tie ⇒ STOP/closure). Per
the plan's analyst note + the prompt's explicit authorization, this is a REVISE
for a replicate, not a PASS.

## Success criteria
- [x] (config provenance) all 3 cells resolve `spectral.correction_mode=signed_ema` AND `spectral.signed_ema_alpha=0.5` (observed: both present as trailing Hydra overrides in every cell's `resolved_params.txt`; signed_ema_alpha appears ONLY as the explicit 0.5 override, never the dataclass default 0.0)
- [x] (one-knob) only differing comm_eff knob across cells is `spectral.beta_anc ∈ {0.25, 0.50, 0.75}` (observed: `diff` of the three `resolved_params.txt` comm_eff lines yields ONLY `beta_anc` + the expected `experiment_name` label; powersgd r=77, anchor owns_q/cadence=5/delay_K=5/replay/cpu-snapshot, clean_cadence=0, max_targets=-1, ema_device=cpu, λ=1.0 identical across all three)
- [x] (W&B routing) all 3 cells `trainer.project_name=verl_compression_research_beta_sweep_signed_ema` (observed: in resolved_params + the TaskRunner config dump `'project_name': 'verl_compression_research_beta_sweep_signed_ema'`)
- [x] (validation cadence) val rows at step 25 + 50 (`val_before_train=False`, `test_freq=25`); cells 1-2 also have an unplanned step-55 row (total=55), informational only; cell 3 has 25+50 ONLY (torn down at val@50 by design)
- [x] (completion) all 3 cells reached step 50 with val@50 captured (observed: 0.7612 / 0.7635 / 0.7225); no early_stop@25 needed
- [x] (correctness) no NaN/Inf in any loss/grad row (0 across all cells); response_length mean stayed 152–303 tok (no length ignition, far below 16384 cap); grad_norm spikes only at warmup steps 1-2 then single-digit through step 51 (no collapse); bytes_ratio ≈ 0.0504 (PowerSGD r=77 active)
- [x] (account) handle `runs/EXP-34/handles/41292294.json` stamps `vast_account=team` (instance 41292294, 4×H200, team-key teardown)
- [x] (no code) no PR opened; `code_change=false`, `promote_launcher_as=none`
- [~] (headline / hypothesis) `best(val@50) − 0.7271 = 0.7635 − 0.7271 = +0.0364`, which is **> +0.024 on the letter**. BUT this is a single best-of-3 draw (`seed_replicates=1`); the plan's noise discipline + the prompt direct a REVISE-for-replicate rather than a PASS when the letter is met but single-seed + selection noise has not been ruled out before promotion. Treated as not-yet-checked pending one replicate.
- [x] (context) β → val@50 curve tabulated below vs the EXP-32 β=0 reference (0.7271) and the B2 delayed_ef context point (0.7528)

## Metrics summary
All values grepped verbatim from each cell's `train.log` row
`val-core/openai/gsm8k/acc/mean@1` (per-cell `metrics/*.jsonl` were not synced;
the train.log console rows are the authoritative durable record per the plan).

| cell | β_anc | val@25 | val@50 (headline) | val@55 (informational) | Δ(val@50 − 0.7271) |
|---|---|---|---|---|---|
| signed_ema_b0p25 | 0.25 | 0.7271 | 0.7612 | 0.7384 | +0.0341 |
| signed_ema_b0p50 | 0.50 | 0.7430 | **0.7635** | 0.7665 | **+0.0364** |
| signed_ema_b0p75 | 0.75 | 0.7028 | 0.7225 | (none — torn down at val@50) | −0.0046 |

- best_cell_val@50 = 0.7635 (b0p50) ; bar = 0.7271 + 0.024 = 0.7511 → clears by +0.0124
- bytes_ratio ≈ 0.0504 (PowerSGD r=77) ; no NaN/Inf ; resp_len mean 152–303 tok (no ignition)
- budget: lifetime_spent_usd 97.07 (cap 1500), running_count 0 (box torn down) — well under all caps

## Comparisons to baseline_run: EXP-32
`diff_against_baseline.py` could NOT resolve a local `runs/EXP-32` (run dirs are
intentionally cleared — DIFF_EXIT=2, expected). Per the plan's documented
fallback I use the recorded reference directly: **EXP-32 signed_ema α=0.5 β=0
val@25 = 0.7278, val@50 = 0.7271**.

The β-curve is **non-flat and peaks at β=0.50**: both β=0.25 (0.7612) and β=0.50
(0.7635) sit ~+0.034/+0.036 above the β=0 reference (0.7271) and ABOVE B2
delayed_ef SOTA (0.7528); β=0.75 regresses to 0.7225 (≈ tie with β=0). This
contrasts with EXP-33, where β_anc was a *flat* free-averaging tie on the
delayed_ef merger — on signed_ema, mild anchor averaging appears to genuinely
help in the 0.25–0.50 region and decays by 0.75. The magnitude (~0.76) lands at
**B2/dense parity** (B2 0.7528, dense band ≈ 0.75–0.78, apples-to-apples draw
0.7839), i.e. a likely-real lift over the β=0 *signed_ema* reference but NOT a
clean surpass-of-dense. Because `seed_replicates=1` + best-of-3 selection,
the exact magnitude is not yet promotable without a replicate.

## Resolved parameters (ground truth)
Source: `runs/EXP-34/<cell>/resolved_params.txt` (extracted from each cell's
train.log `set -x` main_ppo trace, Hydra last-wins; NOT the plan). Comm-eff +
headline knobs, verbatim — identical across all three cells EXCEPT `beta_anc`:

```
actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema
actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.5
actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.25 | 0.50 | 0.75   # the ONLY differing comm_eff knob
actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0
actor_rollout_ref.actor.comm_eff.spectral.max_targets=-1
actor_rollout_ref.actor.comm_eff.spectral.ema_device=cpu
actor_rollout_ref.actor.comm_eff.compression_type=powersgd
actor_rollout_ref.actor.comm_eff.powersgd.rank=77
actor_rollout_ref.actor.comm_eff.powersgd.q_basis=act
actor_rollout_ref.actor.comm_eff.powersgd.sync_basis=true
actor_rollout_ref.actor.comm_eff.clean_cadence=0
actor_rollout_ref.actor.comm_eff.anchor.enabled=true
actor_rollout_ref.actor.comm_eff.anchor.owns_q=true
actor_rollout_ref.actor.comm_eff.anchor.cadence=5
actor_rollout_ref.actor.comm_eff.anchor.delay_K=5
actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch=true
actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=cpu
trainer.project_name=verl_compression_research_beta_sweep_signed_ema
trainer.val_before_train=False
trainer.test_freq=25
trainer.total_training_steps=55
+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true
```

**Divergence from plan/issue — NONE material.** The B2 wrapper hard-exports
`correction_mode=delayed_ef`, `beta_anc=0.0`, `val_before_train=True` early in
the command; these are correctly OVERRIDDEN by the trailing Hydra passthrough
(`correction_mode=signed_ema`, `signed_ema_alpha=0.5`, per-cell `beta_anc`,
`val_before_train=False`) — Hydra last-wins resolves to the intended signed_ema
config in every cell, confirmed by `resolved_params.txt`. The
`+...vllm.disable_custom_all_reduce=true` override is the documented box-compat
break-glass (NCCL all-reduce; greedy-val-neutral) — a controlled variable
present in all three cells, not an off-axis violation. `total_training_steps=55`
(not 50) is the locked 50+5-flush surface; the headline metric is val@50, and
the unplanned val@55 rows (cells 1-2) are informational only.

## next_actions (REVISE only)
- knob: seed/replicate
  from: 1 draw at beta_anc=0.50 (val@50 = 0.7635, single best-of-3 selection)
  to: 2 additional independent draws at beta_anc=0.50 (fresh comm_eff/rollout seed; signed_ema alpha=0.5, B2 substrate otherwise; 50 steps, val@25/50), then take the MEAN val@50 across all 3 draws and compare to 0.7271 + 0.024
  rationale: the headline clears the +0.024 bar on the letter, but with seed_replicates=1 + best-of-3 max-selection the +0.0364 margin rests on one noisy draw whose own within-cell spread (val@50 0.7612 vs val@55 0.7384 on b0p25 = 0.0228) is nearly as wide as the margin over the bar. A 3-draw mean cleanly separates "real lift over signed_ema beta=0" from a high single draw and decides PASS (promotable) vs STOP (noise-bounded tie => beta=0 stays the signed_ema reference). Only beta=0.50 needs replicating (it is the curve peak; beta=0.25 is corroborating, beta=0.75 already regresses).

## Notes
- **Cell 3 (β=0.75) has no val@55 by design** — the box was torn down the instant
  its val@50 was captured (operator's no-val@55 directive; FIXED_CONTROL_SURFACE
  `no-end-of-training-val55`). This is a **valid terminal state, not an infra
  failure**: the completion criterion (val@50 captured) is satisfied. Cells 1-2
  ran the full total=55 and so carry an informational val@55, ignored for the
  headline (the comparison metric is val@50). No `done.flag` was written (box
  torn down before the script reached it) — the auto-generated PENDING stub's
  "done.flag not yet present" warning is moot; completion is established from the
  captured val@50 train.log rows per the plan's completion criterion.
- **Signal vs promotion.** The β-curve being non-flat + peaked (β=0.50 best,
  β=0.75 regresses) is itself a finding: on the *signed_ema* merger β_anc DOES
  lift (unlike the flat delayed_ef curve in EXP-33). But the lift reaches only
  B2/dense *parity* (~0.76), not a clean surpass-of-dense — consistent with the
  converged thesis that no anchor-usage/β lever credibly beats dense. So even if
  the replicate confirms, signed_ema β=0.50 ≈ B2 parity, NOT a new SOTA over B2
  delayed_ef (0.7528). `promote_launcher_as` stays `none` regardless.
- **This is the 1st REVISE on the EXP-34 lineage** (iterations cap = 3). The
  replicate is the only proposed next-action; do not re-run β=0 (EXP-32 provides
  it) and do not add new β values.
- **Provenance fully recovered.** `capture_resolved_config.py` extracted 118
  resolved params from each of the 3 per-cell train.logs (1 main_ppo invocation
  each) → `resolved_params.txt` written per cell. No RESOLVED_CONFIG_MISSING.
- `diff_against_baseline.py` returned EXIT=2 (baseline EXP-32 dir cleared) — the
  documented fallback path, NOT a measurement failure; the EXP-32 reference
  (0.7271 / 0.7278) is used directly per the plan. All headline numbers come
  from grep-able train.log rows.
- analysis.log: `runs/EXP-34/analysis.log` (analyze.py PENDING stub overwritten
  by this verdict; check_budget healthy; diff fallback as above).
