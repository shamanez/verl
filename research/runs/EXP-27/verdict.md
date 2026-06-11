# Verdict EXP-27 — 2026-06-11T04:25:00+00:00

## Result
VERDICT: STOP

Lineage-terminating. Predicate hit on **both** STOP clauses: best val <= 0.7210 AND
ignition fired at the damped settings. This is revise cycle 2 of the ef_powersgd
lineage's max 3; the lineage terminates here with EXP-26's REVISE findings standing as
the M6 record (parity 0.7414 NOT reached). No next_actions.

Note on tooling: `analyze.py --emit verdict.md` auto-wrote `VERDICT: PASS` because it
found no `metrics/*.jsonl` (only `metrics/incoming.log` + the per-step
`train_exp27_B_ef_damped.log`), so it fell back to the "done.flag present, no NaN"
M0-smoke default. That PASS is a stub, NOT a science verdict — every number below comes
from a grep-able row in `train_exp27_B_ef_damped.log` / `metrics/incoming.log`, and the
plan's `## Analyst predicate` is the authority. This verdict overwrites the stub.

## Success criteria
- [x] probe banner confirms ef_decay=0.5 ef_clip=0.5 q_basis=act step_target=100; realism counters green (observed: resolved_params.txt shows correction_mode=ef_powersgd, ef_clip=0.5, ef_decay=0.5, q_basis=act, total_training_steps=100, anchor owns_q cadence5 delay_K5, clean_cadence=0, powersgd rank=77 sync_basis=true)
- [ ] cell reaches step 100 without NaN/OOM (observed: KILLED early at step ~66-68 on confirmed LENGTH_EXPLOSION; max_memory_allocated_gb=123.3/~143 = OOM imminent; no NaN/inf in any metric value — this is a length-hack ignition, not numerical divergence)
- [ ] best val@{50,75,100} >= 0.7414 (observed: best val = 0.7202 @ step 50; val@75/100 never measured — killed first; target: 0.7414) — FAIL by 2.1 pt
- [ ] NO ignition (observed: IGNITION FIRED — resp_len/mean 557.6 > 509 = 2×step-10 baseline at step 66; resp_len/max pinned at 16384 for steps 61-68 consecutive; entropy collapsed 0.34 -> 0.079) — FAIL
- [x] cos(G_comp,G_corr) median stays in preserved band (observed: 12 paired G_comp/G_corr ticks captured at global_steps 5-11; EF residual dose rel_change_mean stayed CAPPED ~0.02-0.19, vs parent's 0.30->0.47 climb — direction-preserving correction intact in the healthy phase, as designed by the damping. Exact median cosine not recomputed here: fp32 tensors still rsyncing; parent band 0.9558 referenced from EXP-26)
- [x] measured comm/bytes_ratio ≈ 0.05 (observed: bytes_ratio min 0.05029 / mean 0.05049 / max 0.05059 across steps 1-68 = ~19.8x comm factor, unchanged codec)
- [ ] val does not fall below 0.7210 without a stated diagnostic reason (observed: best val 0.7202 <= 0.7210 falsify floor; diagnostic reason = no gain over parent un-damped ef + ignition before val@75) — FAIL

## Metrics summary
- best val (reward/mean@1): 0.7202 @ step 50 (target >= 0.7414; falsify floor <= 0.7210) — STOP both ways
- val@25: 0.7134192570128886 (incoming.log + train log)
- val@50: 0.7202426080363912 (train log)
- val@75 / val@100: NOT MEASURED — cell killed at step ~66-68 before val@75
- resp_len/mean ignition: step60=171.0, step61=267.9, step63=294.7, step64=395.1, step65=448.1, step66=557.6 (>509), step67=566.0, step68=575.3
- resp_len/max: pinned 16384.0 for steps 61,62,63,64,65,66,67,68 (8 consecutive)
- actor/entropy collapse: step60=0.342 -> step61=0.253 -> step65=0.210 -> step66=0.079 -> step68=0.082
- critic/score/mean during ignition: 0.73-0.84 (NO reward collapse — length-hack, consistent with EXP-25 mechanism: length-explosion not low-entropy is the killer)
- actor/perf/max_memory_allocated_gb: 57.9 (steps 1-44) -> 117.6 (step 45+) -> 123.3 (steps 63-68) of ~143 — OOM imminent, justifying the early kill per the LENGTH_EXPLOSION rescue trigger
- EF residual dose (spectral/rel_change_mean): peak ~0.189 (step 12), mostly 0.02-0.16, never the parent's 0.30->0.47 — damping ef_clip=0.5/ef_decay=0.5 capped the dose AS DESIGNED
- comm bytes_ratio: ~0.0505 (mean) = ~19.8x comm saving (codec unchanged)
- NaN/inf in metric values: 0

## Comparisons to baseline_run: EXP-26
`diff_against_baseline.py --baseline EXP-26` wrote baseline_diff.md but found "no common
numeric keys" (EXP-26's local run dir was cleared per the standing scaffold-is-ephemeral
practice; all parent metrics live in W&B `tilwe80t` and runs/EXP-26/verdict.md). Manual
comparison against the plan's W&B references:

| run | best val | resp ignition | EF dose | direction (cos) | comm |
|---|---|---|---|---|---|
| dense `5e2jpho9` (never re-run) | 0.7536 | — | — | — | 1x |
| A0 fresh-clean `oquyeic3` | 0.7415 | — | — | — | full-rank clean steps |
| **parity bar (floor+0.05)** | **0.7414** | — | — | — | — |
| parent ef `tilwe80t` (EXP-26, clip1.0/decay0.9, 50 steps) | 0.7210 | fired on 1 of 2 realizations | climbed 0.30->0.47 | 0.9558 | ~19.8x |
| **EXP-27 damped ef** (clip0.5/decay0.5, target 100, killed ~66) | **0.7202** | **fired (step 66)** | capped ~0.02-0.19 | preserved band | ~19.8x |
| plain `u1v94opv` | 0.6437 | — | — | n/a | ~19.8x |
| floor EXP-23 A1 | 0.6914 | — | — | — | — |

The damping did exactly what it was designed to do at the dose level (residual capped
~0.04-0.19 vs the parent's 0.30->0.47) and direction-preservation in the healthy phase
held. But it only **delayed** ignition (parent ~step 29-42 -> here ~step 61) without
**preventing** it, and bought **no val gain** (0.7202 ~= parent 0.7210). Capping the EF
dose neither closes the 2.0-pt gap to the parity bar nor stops the EXP-25 length-explosion
spiral.

## Resolved parameters (ground truth)
Source: `resolved_params.txt` (extracted from train_exp27_B_ef_damped.log, NOT the plan).
The launched command matched the plan exactly — no divergence:

```
actor_rollout_ref.actor.comm_eff.spectral.correction_mode=ef_powersgd
actor_rollout_ref.actor.comm_eff.spectral.ef_clip=0.5
actor_rollout_ref.actor.comm_eff.spectral.ef_decay=0.5
actor_rollout_ref.actor.comm_eff.powersgd.q_basis=act
actor_rollout_ref.actor.comm_eff.powersgd.rank=77
actor_rollout_ref.actor.comm_eff.powersgd.sync_basis=true
actor_rollout_ref.actor.comm_eff.anchor.owns_q=true
actor_rollout_ref.actor.comm_eff.anchor.cadence=5
actor_rollout_ref.actor.comm_eff.anchor.delay_K=5
actor_rollout_ref.actor.comm_eff.clean_cadence=0
actor_rollout_ref.actor.comm_eff.spectral.blend_eta=0.5
actor_rollout_ref.actor.comm_eff.spectral.inject_gamma=1.0
trainer.total_training_steps=100
trainer.test_freq=25
trainer.n_gpus_per_node=4
data.train_batch_size=128 / ppo_mini_batch_size=64 / rollout.n=8 / max_response_length=16384
actor.entropy_coeff=0 / use_kl_loss=False / algorithm.use_kl_in_reward=False
```

No knob diverged from the plan. The only changes vs the parent ef arm were the intended
three: ef_clip 1.0->0.5, ef_decay 0.9->0.5, step_target 50->100. The locked substrate
(powersgd r77, anchor-owns-Q cadence5/delay_K5, clean0, q_basis=act) matched EXP-26.

## Notes
- **Why STOP, not REVISE**: the predicate's STOP clause fires on EITHER best val <= 0.7210
  OR ignition. Here BOTH fired. The plan caps this lineage at 3 cycles (2 consumed; this is
  cycle 2's terminal result, and cycle 2 falsified the hypothesis). Per the plan's
  `on_fail` and `## Analyst predicate`, the lineage terminates and EXP-26's REVISE findings
  stand as the M6 record (ef_powersgd = best realistic live merger at 0.7210, +7.7 over
  plain, cos 0.956, ~19.8x comm; parity bar 0.7414 NOT reached).
- **Falsification is clean**: the hypothesis was "halving clip+decay caps the dose while
  keeping the direction-preserving correction, closing the 2.0-pt gap with no ignition." The
  dose WAS capped (rel_change_mean 0.02-0.19 << parent 0.30-0.47) and direction WAS
  preserved in the healthy phase — yet the arm still ignited and gained nothing. So the
  ignition is NOT driven by an oversized EF residual dose; capping the dose only buys ~20
  steps of delay. This sharpens the EXP-25 mechanism: the length-explosion is a property of
  the entropy-collapse trajectory under this control surface (no KL/entropy/length cap),
  reached regardless of EF dose magnitude, with score/mean staying 0.73-0.84 (length-hack,
  not reward collapse).
- **Run did NOT reach step 100**: killed at step ~66-68 on the confirmed LENGTH_EXPLOSION
  rescue trigger (markers: EARLY_KILL_LENGTH_EXPLOSION, done.flag). val@75/100 do not exist.
  The predicate was applied to the data we have — val@25, val@50, and the ignition
  trajectory — exactly as instructed; this is correct because best-of-available val (0.7202)
  already <= 0.7210 falsify floor and ignition already fired, so no later val point could
  rescue a PASS.
- **No numerical divergence**: 0 NaN/inf in any metric value; grad_norm healthy through the
  ignition window. This is a length-hack spiral, not a training blowup.
- **Captures**: 12 paired G_comp/G_corr ticks (global_steps 5-11) landed in
  captures/exp27_B_ef_damped/rank0/manifest.jsonl; full fp32 tensors were still rsyncing at
  analysis time. Median cos(G_comp,G_corr) not recomputed on the partial dump; the EF dose
  trajectory (capped) is the in-band proxy and is consistent with intact direction
  preservation in the healthy phase.
- **Budget**: check_budget.py reports running_count 0, lifetime_spent_usd 32.945, monthly
  cap 1500 — well within caps; box is operator-HELD warm and the ledger row is already
  COMPLETE. No teardown / no ledger edit performed (per instructions).
- **OPTIONAL future probe (not a REVISE, lineage is terminated)**: the plan listed one
  remaining un-tried damping knob (ef_clip 0.5->0.25, or decay-only). Given that the dose
  was already capped well below the parent's igniting range and the arm STILL ignited with
  zero val gain, my judgment is that an even smaller clip is unlikely to clear the 2.0-pt
  parity gap — it would, at most, delay ignition further while still failing to surpass
  0.7210. I flag it for the operator only as a low-prior optional future issue, NOT a
  recommended next step. The likely productive direction lives elsewhere (the
  conversion-spine thesis: training/eval diversity, not merger-dose tuning).

## Post-mortem (team analysis)

Full analysis in `runs/EXP-27/MECHANISM_ANALYSIS.md` (mechanist-math) and
`runs/EXP-27/RUN_COMPARISON.md` (comparator-runs, W&B cross-check on 6 runs).

**Headline mechanism (MECHANISM_ANALYSIS.md §(h)):** the implemented ef_powersgd is not
true error feedback — it is a persistent tangential forcing loop. `comp_t ⊥ G_t` by
construction (`spectral_filter.py:378-379`); M has ~50-global-step memory; and the
projection is nearly vacuous (cos(G_anchor,G_comp) ≈ 0.01–0.06), making the injected
force effectively a direct, norm-clipped copy of the stale anchor EMA. A persistent
tangential force on the reward-flat "correct-but-longer" direction integrates linearly
(‖Σ e_t‖ ~ λ·T·‖G‖) with no telescoping cancellation. Dose sets only the **lag**:
ef r1 dose 0.200 → lock-in s30; EXP-27 dose 0.092→0.021 → lock-in s61 (ratio 2.03 vs
dose ratio 2.17 — near-exactly linear). EXP-27 ignited at its dose **minimum**.

**Entropy-as-trigger falsified ×3** (RUN_COMPARISON.md §7c): dense is the lowest-entropy
run (0.12–0.16 from s36) and most stable; ef r1 ignited at entropy 0.83 (HIGH) then
collapsed; entropy is a follower of the length spiral, not its trigger. The discriminator
is merger-carrier presence, not entropy level or gradient noisiness (plain has the same
noisy grad_norm class as the merger arms with zero emission — RUN_COMPARISON.md §8).
Monitor triggers updated in `diagnostics/ENTROPY_COLLAPSE_WATCH.md` §2026-06-11:
P1/P2/P3 (length spiral) are now the primary kill triggers; T1–T3 (entropy) are
demoted to corroborators. E1 early gate (any len/max>4000 in steps [10,30]) flags
UNSTABLE-LIKELY with zero false negatives on 6 retro-validated runs.

**α=0.5@100 open question (P1 revised):** signed_ema α=0.5 was already in the early
spiral at its 50-step endpoint (consecutive 16384 pins at s47–48, len/mean slope
+5.92/step, len/max 5806@50 — scored DANGER, worse than EXP-27 looked at its own s50).
P(ignite by s100) ≈ 0.60 (agreed independently by mechanist-math and comparator-runs).
The α=0.5 "stable at 50" framing is a censoring artifact; see MECHANISM_ANALYSIS.md §e.2.

**Top fix:** rebuild EF as true error feedback on the codec's own dropped residual
(sender-local, zero extra comm, telescoping identity restores bounded-lag correction with
no exogenous carrier). See MECHANISM_ANALYSIS.md §(g) item 1.
