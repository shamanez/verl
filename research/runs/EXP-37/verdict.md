# Verdict EXP-37 — 2026-06-18T22:55:00Z

## Result
VERDICT: STOP

Hypothesis falsified on TWO independent clauses: the headline gate (val@50 < 0.6862)
AND the back-half stability clause (length ignition + entropy collapse, steps 50-100).
Per the plan predicate and `## Notes for analyst`, a run that fails the headline and
ignites in the back half is a STOP, not a PASS — instability / material degradation at
cadence/delay 20/20 is itself the publishable finding. `iterations: 1`; do NOT sweep
EMA or latency knobs off this result.

(Note: the mechanical `analyze.py --emit` draft emitted an M0-smoke PASS template — it
only checks done.flag + no-NaN, found "no metrics/*.jsonl", and never read the
validation curve or the back-half stability signals. The plan predicate, which weights
back-half stability heavily, governs and gives STOP.)

## Success criteria
- [ ] cell reaches 100 training steps with no NaN / non-finite gradients
      (observed: 100 steps reached, no real NaN/Inf in any loss/grad field — the run DID
      complete cleanly; the completion sub-clause is met, but this criterion's headline
      pairing fails below)
- [ ] stability: no length ignition / entropy collapse / cold-M fallback across steps 50-100
      (observed: FAIL — response_length/mean spiral 189 -> 251 -> 373 -> 581 -> 683 over
      steps 93-100; entropy collapsing 0.81 -> 0.42; pg_clipfrac rising 0.028 -> 0.125.
      Classic staleness-driven length-hack / sign-SGD sharpening ignition in exactly the
      historical 50-100 window. target: flat length, stable entropy)
- [x] latency realized (override took): anchor_backwards == 10, anchor_q_updates == 10,
      anchor_q_broadcasts == 10; delay >= 20 ticks
      (observed: 10 / 10 / 10; resolved cmd anchor.cadence=20, anchor.delay_K=20 — the
      trailing-Hydra-arg path WON over the bare-export 5; the clobber bug did NOT bite.
      The run is fully interpretable for the hypothesis ⇒ this is NOT the REVISE case.)
- [ ] headline: val@50 >= 0.6862
      (observed: 0.6482, target: 0.6862; degradation vs EXP-36B base 0.7362 is 0.0880,
      worse than the 0.05 absolute bar ⇒ "remains near base" is falsified)
- [x] val@25/50/75/100 + train-score recorded and overlaid in WandB accel_rebaseline
      (observed: 0.5921 / 0.6482 / 0.4898 / 0.4435; WandB run fxo8chsv, project
      verl_compression_research_accel_rebaseline, state finished)
- [x] bytes_ratio recorded ~= 0.0505 (fast-path Y + amortized Q; full-dense M broadcast
      KNOWN-UNCOUNTED)
      (observed: ~0.0502; reported as partial comm cost, NOT total)
- [x] timing recorded
      (observed: update_actor + anchor-step latency present in train.jsonl/incoming.log)

## Metrics summary
- val@25: 0.5921 (no gate; informational)
- val@50: 0.6482 (target >= 0.6862 — FAIL, degradation 0.0880 vs base 0.7362)
- val@75: 0.4898 (post-ignition collapse)
- val@100: 0.4435 (post-ignition collapse; monotone decline 50 -> 100)
- response_length/mean (steps ~93-100): 189 -> 251 -> 373 -> 581 -> 683 (length ignition)
- actor/entropy (tail): 0.81 -> 0.42 (collapsing)
- actor/pg_clipfrac (tail): 0.028 -> 0.125 (rising — off-policy length hack)
- anchor_backwards / q_updates / q_broadcasts: 10 / 10 / 10 (latency realized)
- bytes_ratio: ~0.0502 (target ~0.0505; partial — M broadcast uncounted)
- steps completed: 100 / 100; no NaN/Inf/OOM/Traceback

## Comparisons to baseline_run: EXP-36B
Reference only (NOT re-run this cycle); the script-level train.jsonl diff found no
common numeric keys, so the comparison is at the validation/stability level. EXP-36B
(accel signed_ema 0.25/0.50 at cadence/delay 5/5, 50 steps) held val@50 = 0.7362.
Raising latency to 20/20 and extending to 100 steps drops the matched-step val@50 to
0.6482 — a 0.0880 absolute degradation, above the 0.05 "remains near base" bar — and the
back half (steps 50-100, never previously exercised on the 5/5 base) ignites: response
length runs away (189 -> 683), entropy collapses (0.81 -> 0.42), val falls monotonically
(0.6482 -> 0.4898 -> 0.4435). EXP-36B was stable only because it stopped at step 50,
before the staleness-driven spiral window. Dense same-surface ref EXP-36C ~= 0.7657 is
far out of reach. Conclusion: signed_ema(0.25,0.50) does NOT remain near the accel base
under realistic 10-global-step anchor staleness — it is destabilized by it.

## Resolved parameters (ground truth)
Source: `resolved_params.txt` (extracted from train.log set -x trace, NOT the plan).
```
actor_rollout_ref.actor.comm_eff.anchor.cadence=20
actor_rollout_ref.actor.comm_eff.anchor.delay_K=20
actor_rollout_ref.actor.comm_eff.enabled=true
actor_rollout_ref.actor.comm_eff.powersgd.rank=77
actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema
actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.25
actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.50
trainer.total_training_steps=100
trainer.test_freq=25
```
No divergence between plan and launched command. The CRITICAL launch gotcha flagged in
`## Notes for runner` (the accel base's bare `export ...=5` clobbering caller env) was
correctly avoided: cadence/delay were passed as trailing Hydra args and WON under
last-wins (resolved = 20/20, fires = 10), so the run is interpretable for the hypothesis.
This rules out the latency-not-realized REVISE branch.

## Notes
- This is a STOP on hypothesis falsification, not a failure of mechanics. The experiment
  did exactly what it was designed to do: it cleanly characterized signed_ema's behavior
  under realistic anchor staleness, and the answer is negative.
- The instability is in the back half (steps 93-100), the precise window the run was
  extended to 100 steps to probe. The 5/5 base "passed" only by stopping at step 50,
  before the spiral. This corroborates the standing finding that M_anchor-carrier mergers
  are STRUCTURALLY unstable and that 50-step survivals are censored, not stable (cf. prior
  EXP-25 / 27 / 32 censored-stability findings).
- Per the plan (`on_fail: stop`, `iterations: 1`, predicate STOP clause): do NOT sweep EMA
  alpha/beta or latency knobs off this draw. The deliverable is this single characterized
  data point: signed_ema degrades materially AND ignites at cadence/delay 20/20 over 100
  steps.
- bytes_ratio 0.0502 counts only fast-path Y + amortized Q; the full-dense M broadcast is a
  KNOWN-UNCOUNTED term — not the total communication cost.
- WandB run fxo8chsv (project verl_compression_research_accel_rebaseline), state finished;
  curves overlay EXP-36B (rsvo7y1p) / EXP-36C directly.
- done.flag reads rc=1; the training script wrote done.flag and the run reached 100 steps
  with all metrics rsynced, so completion is satisfied (rc reflects a non-fatal
  teardown/wandb-flush exit, not a training failure — no NaN/OOM/Traceback in train.log).
