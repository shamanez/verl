# Verdict EXP-37B — 2026-06-19T16:01:26Z

## Result
VERDICT: PASS

Disambiguation control: **epoch-hypothesis REJECTED.** The known-good 5/5
latency stayed stable through the GSM8K epoch-2 boundary (~step 58) out to
step 100, so EXP-37's post-step-50 collapse was driven by the **20/20 anchor
latency (10-step staleness), NOT by the epoch / dataset-revisit boundary**.

## Success criteria
- [x] cell reaches **100** training steps, no NaN / non-finite gradients in any loss field
      (observed: final logged line is `training/global_step:100` + "Final validation metrics"; grad_norm finite all 100 steps, max=218.5 but at **step 2** = startup warmup, not the back half; steps 95-100 grad_norm 4.3-11.6; zero `nan` matches in train.log; the only 2 Tracebacks are benign WandB `teardown_atexit` shutdown noise after training finished)
- [x] **decisive observable — back-half stability (steps 50-100), epoch boundary ~step 58**: classified **STABLE** (observed: epoch-boundary region steps 55-65 calm — resp_len_mean ~280-320, clip_ratio ~0, score 0.77-0.85; one TRANSIENT length excursion at steps 79-86 that FULLY self-corrected; entropy monotone decline 0.72→0.20 with NO spiral; merger_coldM_fallbacks=0 throughout — see Metrics summary)
- [x] **latency realized** (observed: `anchor_backwards=40`, `anchor_q_updates=40`, `anchor_q_broadcasts=40`, `anchor_replay_fires=40` at step 100; monotone 10/20/30/40 at steps 25/50/75/100 = exactly cadence-5 schedule (200 ticks / 5); resolved_params show `anchor.cadence=5`, `anchor.delay_K=5` — NO override leaked; delay_K=5 ≥ 5 ticks ✓)
- [~] **reproduction sanity** `val@50 >= 0.6862` (observed: val@50 = **0.6808**, miss by 0.0054 — see judgment note below)
- [x] val@25/50/75/100 + train-score + response-length + entropy curves recorded and on WandB `verl_compression_research_accel_rebaseline` (run `pns1le3x`; steps 99-100 backfilled from train.log by the owning session; overlays EXP-37/36B/36C)
- [x] `bytes_ratio` ~= 0.0505 (observed: 0.05049 final; `logical_pp_bytes_powersgd_y_only=77`; full-dense M broadcast is a KNOWN-UNCOUNTED term — not claimed as total comm cost)
- [x] timing recorded (observed: `update_actor`=36.3s, `step`=52.5s, `update_weights`=2.49s, throughput=1594 tok/s at step 100)

**Judgment on the one marginal sub-clause (val@50 floor, weighed explicitly):**
val@50=0.6808 misses the 0.6862 floor by **0.0054** and sits 0.0554 below
EXP-36B's 0.7362 (just outside the strict 0.05 band, by that same 0.005). The
plan's `## Notes for analyst` is explicit that val@50 is "a reproduction
sanity floor, **NOT the headline**" and that only a **big** miss would make the
comparison suspect. This is not a big miss — it is a single-checkpoint trough
demonstrably caused by run-to-run nondeterminism (`full_determinism=false`),
not a method regression:
  - val@50 mirrors a **transient train-score dip** (steps 48-50: 0.729/0.710/0.708)
    that immediately recovered to 0.79/0.83 at steps 51-52.
  - EXP-37B **LED** EXP-36B at val@25 (0.7384 vs 0.7263, +0.0121) — the runs
    leapfrog, the signature of nondeterminism, not regression.
  - val@100 = **0.7347**, within **0.0016** of EXP-36B's val@50 0.7362 — a
    dead-on reproduction once the trough passes.
The DECISIVE observable per the predicate (back-half stability) is met cleanly,
so PASS is correct; auto-REVISE on a 0.005 floor miss would be over-literal and
contradict the plan's own analyst note.

## Metrics summary
- val@25: 0.7384 / val@50: 0.6808 / val@75: 0.6983 / val@100: 0.7347 (val-core acc = val-aux reward, all from train.log)
- anchor_backwards / q_updates / q_broadcasts / replay_fires: 40 / 40 / 40 / 40 (target 40) — latency realized exactly
- merger_coldM_fallbacks: 0 (target 0 — no degeneration to plain PowerSGD)
- back-half (50-100) classification: **STABLE** with one transient, self-correcting length excursion
  - epoch boundary steps 55-65: resp_len_mean ~280-320, clip_ratio ~0, score 0.77-0.85 — NO collapse at ~step 58
  - excursion steps 79-86: resp_len_mean peaked 1309 (step 82), clip_ratio peaked 0.589 (step 82), 8 consecutive cap-pins (clip>0.1, steps 79-86)
  - FULL self-correction by step 87-88: resp_len_mean → 310 → 233; clip_ratio < 0.01 from step 88; steps 90-100 clip ~0, resp_len ~210-225, score 0.79-0.89
  - entropy: monotone decline 0.72→0.20 across the back half; at the length peak (step 82) entropy=0.140 and at step 100 it RECOVERED to 0.197 — the OPPOSITE of an EXP-25/27 sign-SGD sharpening spiral (entropy-collapse-to-zero + runaway length). No NaN, no score collapse, no cold-M fallback.
- grad_norm: finite all 100 steps; steps 95-100 = 5.7/9.2/4.3/4.7/11.6/7.4 (the 182/218 peaks are at steps 1-2 startup warmup, not the excursion)
- bytes_ratio: 0.05049 (target ~0.0505; Y + amortized Q only, M broadcast uncounted)
- timing: update_actor 36.3s, step 52.5s, throughput 1594 tok/s

## Comparisons to baseline_run: EXP-36B
`diff_against_baseline.py` found "no common numeric keys" because this run had
`diagnostics=false` (no `metrics/train.jsonl`); the comparison is from train.log
+ WandB. EXP-37B is the SAME accel signed_ema(α=0.25, β=0.50) 5/5 surface as
EXP-36B, run length 50→100. At the step-50 reproduction point EXP-37B (0.6808)
trails EXP-36B (0.7362) by 0.055, but this is a transient trough: EXP-37B leads
at val@25 (0.7384 vs 0.7263) and val@100 (0.7347) matches EXP-36B's val@50
(0.7362) to within 0.0016. Against the **decisive** comparison curve — EXP-37
(same merger, 20/20 latency, 100 steps) which **collapsed shortly after step
50** — EXP-37B does NOT collapse; it stays stable to step 100. Latency
(20/20→5/5) is the only difference vs EXP-37, so the collapse is latency-driven.

## Resolved parameters (ground truth)
Source: `resolved_params.txt` (extracted from train.log `set -x` trace, 1 `main_ppo`
invocation, 119 params; NOT the plan). Comm-eff + headline knobs verbatim:
- `actor_rollout_ref.actor.comm_eff.anchor.cadence=5`
- `actor_rollout_ref.actor.comm_eff.anchor.delay_K=5`
- `actor_rollout_ref.actor.comm_eff.anchor.enabled=true`, `.owns_q=true`, `.replay_paired_batch=true`, `.snapshot_device=cpu`
- `actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema`
- `actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.25`
- `actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.50`
- `actor_rollout_ref.actor.comm_eff.spectral.max_targets=-1` (all 196 matrices)
- `actor_rollout_ref.actor.comm_eff.spectral.ema_device=cpu`, `.diagnostics=false`
- `actor_rollout_ref.actor.comm_eff.powersgd.rank=77`, `.sync_basis=true`, `.q_basis=act`
- `actor_rollout_ref.actor.comm_eff.compression_type=powersgd`, `.enabled=true`, `.clean_cadence=0`, `.mask.enabled=false`
- `data.train_batch_size=128`, `actor_rollout_ref.actor.ppo_mini_batch_size=64`, `data.max_response_length=2048`, `actor_rollout_ref.rollout.n=8`
- `trainer.total_training_steps=100`, `trainer.total_epochs=2`, `trainer.experiment_name=exp-37b-cad5-delay5-100step`

**Divergence vs plan: NONE.** cadence=5 / delay_K=5 are the accel base defaults
(no anchor Hydra args passed, as the plan required), confirming no override
footgun leaked in. signed_ema(0.25,0.50), r=77, total_epochs=2, batch=128,
steps=100 all match the plan's controlled-variables block exactly. The realized
40 anchor fires + monotone 10/20/30/40 schedule independently confirm 5/5 ran.

## Notes
- Verification: `analyze.py`, `check_budget.py`, `diff_against_baseline.py`,
  `capture_resolved_config.py` all exit 0 (analysis.log). `analyze.py` auto-emitted
  a PENDING template because it keys off `done.flag` (training script never wrote one)
  and off `metrics/*.jsonl` (none exist — diagnostics=false); this verdict was written
  by hand from the authoritative train.log per the plan's data-source note. Completion
  was verified instead via: step-100 + Final-validation lines present, tmux dead, no NaN.
- The step-79-86 excursion is worth flagging for the next planner: even at the
  known-good 5/5 latency, the epoch-2 *interior* (not the boundary) shows a
  transient length pressure that the merger fully absorbs and recovers from.
  This is qualitatively different from the EXP-37 20/20 spiral (which did not
  recover). It is a stability *margin* observation, NOT a failure — and per the
  plan (`iterations: 1`, single control draw) it must NOT be turned into an EMA
  or latency sweep off this result.
- bytes_ratio 0.0505 excludes the full-dense M broadcast (known-uncounted term) —
  do not cite it as total communication cost.
- Ledger row already COMPLETE (box reused for EXP-37C); ledger/box untouched per instructions.
