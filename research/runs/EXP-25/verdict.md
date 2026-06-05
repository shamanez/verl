# Verdict EXP-25 — 2026-06-06T00:00:00Z

## Result
VERDICT: STOP

The hypothesis is FALSIFIED. Best-α = 0.5 reaches `val@50 = 0.7066`, which is
**below the plan's explicit falsification line** `val@50 <= 0.7114 = floor+0.02`
(floor = no-refresh PowerSGD r=77 = 0.6914). Per the `## Analyst predicate`, a
best-α `val@50 <= 0.7114` is an automatic STOP: the anchor-default signed_ema
substrate is no better than no-refresh PowerSGD, so it does NOT recover the
comm-efficiency gap and does NOT unblock #24.

This is a **borderline STOP** (0.7066 is ~4.8 pts below the STOP line but ~15 pts
above the raw floor). It is the *first* sweep on this lineage (0 of `iterations:3`
REVISE cycles consumed) and a concrete continuation knob exists (α>0.5, see
next_actions). The STOP is nonetheless correct and is NOT overridden to REVISE
because: (1) the predicate's falsification clause is a hard machine-checkable
condition that fires here; (2) the dose-response is **monotonic** — every increase
in correction strength (α: 0.5→0.3→0.0) makes val *worse*, with the best result
sitting at the α→1 (least-correction) edge of the swept grid, which is the signal
that the signed correction primitive itself is net-harmful, not merely mistuned.
The next_actions below are recorded as an operator menu, not as a passed-through
REVISE; a child experiment is a new lineage decision.

## Success criteria
- [x] (gate, id 0) anchor-M probe hard gates pass — `anchor-load loaded 338/338`, `coverage anchor_targets=196 merger_expected=196 set_equal=True missing=[] extra=[]`, `dp-reduce MEAN ||G||_post/||G||_pre_mean≈0.71–0.79` (mean not sum), `M-dp-identical cross_rank_max_rel_dev=0.000e+00`, `||dM_anchor||_mean=1.41e-03 > 0`, `anchor_ratio=1.0 anchor_optimizer_steps=0 anchor_grad_corrected=0 anchor_mask_applications=0` (logs/exp25_id0_anchorM.trainlog)
- [x] (gate, id 1) all-flags-ON probe hard gates pass — `[bcast] Q updated=True boundaries=7 changed=5/4 cross_rank_max_rel_dev=0.0`, `[bcast] M broadcast targets=196`, `[merger] corrected=196`, `powersgd_basis_updates=0` (fast net never updates Q); no NaN/OOM/single-GPU (logs/exp25_id1_R2R3.trainlog)
- [x] (off-path parity) `q_cross_rank_max_rel_dev == 0.0`; `reconstruction_rel_error = 0.02399` (within 1e-3 of 0.024) on the PowerSGD path
- [x] every α arm reached 50 steps without NaN or non-finite gradients (observed: 50/50 step rows each; 0 NaN in any loss field; end-of-log Traceback lines are post-training DataLoader/UnixTransport teardown noise at lines 3173+ AFTER step 50)
- [x] (comparison) controlled variables hold equal across α arms (observed: every codec/anchor/surface knob bit-identical in all three banners; ONLY `signed_ema_alpha` differs = 0.0/0.3/0.5 — see Resolved parameters)
- [ ] best-α `val@50 >= 0.7315` (observed: 0.7066, target: 0.7315)
- [ ] best-α `val@50 >= floor+0.05 = 0.7414` (observed: 0.7066, target: 0.7414)
- [ ] best-α does NOT regress below floor AND train-reward curve trends above the no-refresh reference (observed: val 0.7066 > floor 0.6914 ✓ on val, BUT below the +0.02 falsification line; and the α<0.5 arms' `critic/score/mean` curves PEAK then CRASH — α=0 0.787@s28→0.478@s50, α=0.3 0.773@s28→0.621@s50 — degrading, not trending above reference)

## Metrics summary
- best-α (α=0.5) val@25: 0.7051 / val@50: **0.7066** (target ≥0.7414; FALSIFY line ≤0.7114)
- α=0.3 val@25: 0.6937 / val@50: 0.6164 (delayed collapse)
- α=0.0 val@25: 0.7180 / val@50: 0.3541 (catastrophic collapse)
- α=0.5 final train-reward `critic/score/mean`: 0.814@s50 (peak 0.814@s41) — stable, no crash
- α=0.5 response_length: 276→170 (max 288), `clip_ratio` ~0 the whole run — NO length explosion
- α=0.0 response_length: 278→5863 (max 8634@s47), `clip_ratio` 0→0.46 — length explosion
- α=0.3 response_length: 282→15786 (saturates the 16384 cap), `clip_ratio` 0→0.909 — length explosion
- entropy (all three): 5.69→{0.086 (α=0), 0.334 (α=0.3), 0.371 (α=0.5)} — declines in ALL arms incl. the non-collapsing one
- warm-step `rel_change` median (α=0): 1.416 ≈ √2 (n=1379, max 1.889) ⇒ stale-anchor sign disagrees with the live grad on ~50% of magnitude-weighted coords every step
- merger_coldM_fallbacks: 196→196→0 (steps 1,2,3) in all arms — the cold-M guard fires correctly then the merger fully engages from step 3 (= the collapse onset for α<0.5)

## Comparisons to baseline_run: EXP-20

`diff_against_baseline.py --baseline EXP-20` returned rc=2 (`baseline not found:
runs/EXP-20`) — **expected, not a failure**: the EXP-20/EXP-23 run dirs were
cleared locally in the issue-#25 clean-slate commit f15c702dc (plan §Background).
The references survive in W&B and are read from there, per the plan. Reference
values (read, never re-run): dense ceiling **0.7536** (W&B 5e2jpho9), A0 PowerSGD
r=77 + fresh-clean@5 **0.7415** (oquyeic3), no-refresh PowerSGD r=77 floor
**0.6914** (EXP-23 A1).

| run | val@50 | vs floor (0.6914) | vs A0 (0.7415) | vs dense (0.7536) |
|---|---|---|---|---|
| dense (5e2jpho9) | 0.7536 | +0.062 | +0.012 | — |
| A0 r77+clean@5 (oquyeic3) | 0.7415 | +0.050 | — | −0.012 |
| no-refresh floor (EXP-23 A1) | 0.6914 | — | −0.050 | −0.062 |
| **EXP-25 α=0.5 (1wulaelw)** | **0.7066** | **+0.015** | **−0.035** | **−0.047** |
| EXP-25 α=0.3 (r8kc702g) | 0.6164 | −0.075 | — | — |
| EXP-25 α=0.0 (uyrpaftw) | 0.3541 | −0.337 | — | — |

Best-α (0.7066) clears the raw floor by +0.015 but FAILS the +0.02 falsification
margin (needs ≥0.7114) and the +0.05 success target (needs ≥0.7414). The
anchor-default substrate does NOT close the gap to A0 fresh-clean; the signed_ema
correction makes things monotonically worse as it is dialled up.

## Resolved parameters (ground truth)
Source: `resolved_params.txt` (extracted from train.log `set -x` trace by
`capture_resolved_config.py`, NOT the plan; 85 params, last-write-wins Hydra
semantics). The captured `train.log` == `exp25_alpha_0p5.fulltrain.log` (md5
336d4b0b…), so resolved_params is the α=0.5 arm. The other two arms were
cross-checked directly from their banners (logs/exp25_alpha_0p{0,3}.fulltrain.log).

Comm-eff + headline knobs (verbatim, α=0.5 arm):
```
actor_rollout_ref.actor.comm_eff.enabled=true
actor_rollout_ref.actor.comm_eff.compression_type=powersgd
actor_rollout_ref.actor.comm_eff.powersgd.rank=77
actor_rollout_ref.actor.comm_eff.powersgd.sync_basis=true
actor_rollout_ref.actor.comm_eff.mask.enabled=false
actor_rollout_ref.actor.comm_eff.clean_cadence=0
actor_rollout_ref.actor.comm_eff.anchor.enabled=true
actor_rollout_ref.actor.comm_eff.anchor.cadence=5
actor_rollout_ref.actor.comm_eff.anchor.delay_K=5
actor_rollout_ref.actor.comm_eff.anchor.owns_q=true
actor_rollout_ref.actor.comm_eff.spectral.enabled=true
actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema
actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.5   # ← the ONLY varied knob (0.0/0.3/0.5)
actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95
actor_rollout_ref.actor.comm_eff.spectral.max_targets=-1
actor_rollout_ref.actor.comm_eff.spectral.cadence=1
actor_rollout_ref.actor.comm_eff.spectral.ema_device=cpu
actor_rollout_ref.actor.entropy_coeff=0
actor_rollout_ref.actor.use_kl_loss=False
algorithm.use_kl_in_reward=False
data.train_batch_size=128
actor_rollout_ref.actor.ppo_mini_batch_size=64
actor_rollout_ref.rollout.n=8
data.max_response_length=16384
trainer.total_training_steps=50
trainer.test_freq=25
```

Controlled-variable assertion: ALL of the above are bit-identical across the three
arm banners EXCEPT `signed_ema_alpha` (0.0 / 0.3 / 0.5). The α-effect is therefore
unconfounded — this is a clean dose-response.

**Divergence callouts (plan/issue vs what actually ran):**
1. `use_kl_loss` — the launcher's ACTOR array sets `use_kl_loss=True kl_loss_coef=0.001`
   EARLY in the command, but the **last-wins resolved value is `use_kl_loss=False`**
   (a later override in the same command + the TaskRunner config dump at
   `exp25_alpha_0p5.fulltrain.log:218` shows `'use_kl_loss': False`). So the run is
   vanilla GRPO no-KL/no-entropy as the FIXED control surface requires — the early
   `True` is dead (overridden). Launcher hygiene smell (a True that never takes
   effect), NOT a control-surface violation; flagged so the operator can clean it.
2. `anchor.cadence=5 / delay_K=5` count OPTIMIZER/MINI-BATCH TICKS, not global
   steps. With train_batch=128 / ppo_mini=64 = 2 ticks/global-step, the EFFECTIVE
   refresh + staleness is **~2.5 global steps, not 5** (confirmed live:
   `anchor_q_updates=14` at global_step=37; `anchor_backwards=19` at step 49). HELD
   FIXED across all arms so it does NOT confound the α-sweep, but the realistic
   comm-amortization is over ~2.5 steps (≈2× more anchor traffic than a 5-step
   assumption). A REVISE re-run should re-pin to global-step units.

## next_actions (operator menu — STOP, not a passed-through REVISE)
Recorded for the operator because a concrete continuation exists, but the verdict
is STOP (hypothesis falsified). A child experiment is a fresh lineage decision.
- knob: signed_ema_alpha
  from: 0.5
  to: "{0.7, 0.85, 1.0}"
  rationale: "Dose-response is MONOTONIC — best result is at the LEAST-correction edge (α=0.5) and α→1 = plain PowerSGD (no signed correction). Sweeping toward 1.0 tests whether ANY signed_ema correction helps; the prediction (from monotonicity) is that 1.0 ties or beats 0.5, i.e. the correction is net-harmful and should be abandoned. Single highest-information next point."
- knob: regularizer
  from: "none (no-KL/no-entropy/no-length-cap)"
  to: "entropy floor (entropy_coeff>0) OR KL penalty OR a hard response-length cap/penalty"
  rationale: "The proximate val-killer is the RESPONSE-LENGTH EXPLOSION (clip_ratio→0.46/0.91 in α<0.5) under no-KL/no-entropy, NOT low entropy per se (the dense ref and α=0.5 run at LOW entropy but bounded length and high val). A length cap or KL brake removes the reward-hack channel the persistent-sign steps exploit, decoupling 'correction strength' from 'degeneration'. Test on α=0.3 (has signal but collapses)."
- knob: correction_primitive
  from: "signed_ema (sign from stale-anchor M, magnitude from G_noisy)"
  to: "magnitude/direction-preserving correction (error-feedback on the PowerSGD residual, #24) — abandon sign-replacement"
  rationale: "signed_ema replaces the live update DIRECTION with a stale sign on every coord, destroying the per-coordinate sign-cancellation that regularizes the true PG step (warm rel_change median ≈ √2 ⇒ ~50% of coords get a wrong full-magnitude sign every step). Error-feedback (#24) corrects the COMPRESSION residual without overriding direction; #24 was gated on #25 — this STOP is the signal to redesign the primitive before #24 spends compute."

## Notes
- COMPLETION verified: no `done.flag` written, but `alpha_sweep_ALL.done` exists
  ("all 3 arms done"), all three `exp25_alpha_*.arm-done` markers present, the box
  is torn down, and each `*.fulltrain.log` has 50/50 step rows — completion is
  unambiguous.
- The id-0 and id-1 PROBE GATES ARE GREEN (all hard invariants verified from the
  on-box trainlogs). The α-sweep results are therefore interpretable — this STOP
  is a genuine TRAINING-DYNAMICS result, not a broken-implementation artifact.
- The merger CODE IS CORRECT (`spectral_filter.py:307`
  `g_corr = alpha*gm + (1-alpha)*gm.abs()*torch.sign(anc)`; matrix-level cold-M
  fallback `if anc_norm<=eps: return g_mask` at :296). The deep-audit verdict
  (PROGRESS.md history) stands: this falsification is about the signed_ema
  PRIMITIVE, not a bug. No RESCUE_REQUEST warranted.
- `diff_against_baseline.py` rc=2 (baseline dir cleared) is the documented
  condition, not a measurement failure — references read from W&B per plan.
- `resolved_params.txt` WAS captured (the `RESOLVED_CONFIG_MISSING` path did not
  trigger). The run is reproducible from it.
- Cross-issue: #24 (`depends_on: #25`) is BLOCKED — #25 did not reach PASS, so #24
  must NOT launch until the correction primitive is redesigned (next_actions[2]).
- Deep scientific writeup: `runs/EXP-25/DEEP_FINDINGS.md` (full dose-response,
  cross-run isolation table, mechanism, ranked improvements).
