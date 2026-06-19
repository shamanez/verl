# Verdict EXP-37C — 2026-06-20T02:59:36+10:00

## Result
VERDICT: STOP

(Operator-added sibling of issue #37. The plan's `## Analyst predicate` was written
for EXP-37B — the 5/5-latency control. EXP-37C is a *different draw* — signed_ema,
beta_anc=0.0, anchor latency 20/20 — so it is judged as an **instability
characterization** at the high-latency operating point, NOT against the 5/5 PASS/STOP
disambiguation predicate. This verdict does not touch issue #37's `status:pass`
label, which belongs to EXP-37B.)

## Success criteria
The plan's success checkboxes target EXP-37B (5/5). EXP-37C is re-scoped to the
high-latency-instability question. Mapped against the operative checks for THIS draw:

- [x] cell reaches **100** training steps, no NaN / non-finite gradients in any loss field (observed: `training/global_step:100`; grep for nan/inf in loss/grad fields = empty; the 3 Tracebacks are benign WandB `teardown_atexit` + dataloader `_shutdown_workers` noise post-training, exit_code=0)
- [x] **latency realized** — confirms the 20/20 draw actually ran (observed: `anchor_backwards=10`, `anchor_q_updates=10`, `anchor_q_broadcasts=10`, `anchor_replay_fires=10`; target at 20/20 over 200 ticks = 200/20 = **10** fires — matches. This is NOT the 5/5 plan's `==40`; the trailing Hydra override `anchor.cadence=20 delay_K=20` is the whole point of EXP-37C)
- [ ] **back-half (50-100) stability** (observed: **OSCILLATING — collapse→recover→collapse**, NOT stable; see Metrics summary. Two distinct length-ignition events: steps 35-41 and steps 88-100. This fails any "stable" classification and is the decisive finding)
- [ ] **val@50 reproduction floor `>= 0.6862`** (observed: **0.5368** — well below floor, because step 50 sits in the recovery tail of the first ignition; this floor was an EXP-37B/5/5 sanity check and is NOT meaningful for the 20/20 draw, but it is recorded as failed-by-design)
- [x] val@25 / val@50 / val@75 / val@100 + train-score + response-length + entropy recorded (observed: val 0.6808 / 0.5368 / 0.7013 / 0.3457; full per-step scalar trajectory in train.log; WandB `u16ui4vx`, project `verl_compression_research_accel_rebaseline`, backfilled to step 100)
- [x] `bytes_ratio` recorded ~= 0.0505 (observed: 0.0502 across steps 25/50/75/100; fast-path Y + amortized Q only — full-dense M broadcast is a KNOWN-UNCOUNTED term, do not present 0.0502 as total comm cost)
- [x] timing recorded (observed at step 100: `update_actor=41.6s`, `step=59.6s`; anchor-step ticks inflate `update_actor` on the 10 fire ticks)

## Metrics summary
Per-step trajectory (train.log, authoritative — diagnostics=false so no metrics/*.jsonl):

| phase | steps | score_mean | resp_len_mean | resp_max | len_clip | grad_norm | entropy |
|---|---|---|---|---|---|---|---|
| warmup | 1-10 | 0.11-0.14 | ~270-290 | <2048 | ~0 | 120-340 | 4.7-6.0 |
| **stable learning** | 11-34 | 0.14 -> 0.61 | falls to ~145 | mixed | ~0 | 1.5-7.4 | 2.4 -> 1.58 |
| **COLLAPSE #1 (length ignition)** | 35-41 | 0.50 -> 0.36 | **159 -> 779** | 2048 pinned | **0.004 -> 0.32** | 5 -> **50** | 1.58 -> **0.53** |
| **self-recovery** | 42-50 | 0.36 -> 0.56 | 779 -> 135 | 2048->1400 | 0.32 -> 0 | 50 -> 3.7 | 0.53 -> 1.13 |
| **FULL recovery (peak)** | 51-87 | climbs to ~0.76-0.80 | ~120-200 | mostly <1000 | ~0 | 1.4-8 | 1.4 -> 0.80 |
| **COLLAPSE #2 (re-ignition)** | 88-100 | **0.80 -> 0.32** | **173 -> 305** | 2048 reappears | 0 -> **0.034** | 4 -> **14** | 0.80 -> **0.49** |

Headline observables:
- val@25=0.6808, val@50=**0.5368** (in recovery tail of collapse #1), val@75=**0.7013** (recovered peak), val@100=**0.3457** (in collapse #2)
- Step 99/100: resp_len 314/305, len_clip 0.032/0.034, grad_norm 13.8/14.2, pg_clipfrac 0.074/0.151, score 0.31/0.32, entropy 0.49/0.49
- **Entropy declines monotonically across the entire back half** (1.46 @step60 -> 0.49 @step100) — a sign-SGD sharpening signature consistent with signed_ema |G|.sign(M) at beta_anc=0 (no EMA history; M = the single fresh, maximally-stale anchor gradient at 20/20). It does NOT recover with the score during steps 51-87, so the second collapse re-ignites from an already-sharpened, low-entropy policy.
- merger_coldM_fallbacks=0, residual_reset_on_shape_mismatch=0 across the run — collapse is NOT a cold-M / merger-degeneration fallback; it is a length-ignition / entropy-sharpening spiral.
- bytes_ratio ~= 0.0502 (Y + Q only; M broadcast uncounted).

## Comparisons to baseline_run: EXP-37 (20/20, beta_anc=0.50) and EXP-37B (5/5, beta_anc=0.50)

`diff_against_baseline.py` reports "no common numeric keys" for both EXP-37 and
EXP-37B (diagnostics=false on all three -> no train.jsonl to diff). The comparison
is therefore made on the recorded WandB/train.log curves, holding everything fixed
except the two named knobs. The scientific contrast is a 2x2 on (latency, beta_anc):

| run | latency (cad/delay) | beta_anc | back-half behavior | val@100 |
|---|---|---|---|---|
| **EXP-37** | 20/20 | 0.50 (EMA smoothing) | single **TERMINAL monotonic** collapse ~step 61, **no recovery** | collapsed |
| **EXP-37C** (this) | 20/20 | **0.00** (no history; M=fresh anchor grad) | **OSCILLATING** — collapse 35-41 -> full recovery (val@75=0.70) -> re-collapse 88-100 | **0.3457** |
| **EXP-37B** | **5/5** | 0.50 | **STABLE** through step 100 (PASS) | 0.7346 |

**Conclusion — beta trades onset-time for recoverability at high latency, and neither
beta is stable at 20/20.** Holding latency at the collapsing 20/20:
- beta_anc=0.50 (EXP-37): the EMA *delayed* the onset (~step 61 vs ~step 35 here) but
  made the collapse **terminal** — the smoothed, accumulated stale signal pins the
  failure once it starts.
- beta_anc=0.00 (EXP-37C): no EMA history means the merger tracks only the latest
  (maximally stale) anchor gradient, so collapses ignite **earlier** (step 35) but are
  **recoverable** (full bounce-back to val@75=0.70) — yet recovery is not durable; a
  second ignition follows by step 100.

The only stable cell in the 2x2 is **EXP-37B's 5/5 latency**. This corroborates
EXP-37B's finding that the EXP-37 post-step-50 collapse was **latency-driven, not
epoch-driven**: at 5/5 the run sails through the GSM8K epoch-2 boundary (~step 58)
unharmed, whereas BOTH 20/20 draws (either beta) destabilize. **20/20 latency is not a
viable operating point regardless of beta_anc** — hence STOP. (This is instability
characterization, NOT a surpass-dense claim.)

## Resolved parameters (ground truth)
Source: `resolved_params.txt` (extracted from train.log `set -x` trace, NOT the plan;
119 params, 1 main_ppo invocation, Hydra last-wins on duplicate keys).

Comm-eff + headline knobs that define the EXP-37C draw (verbatim, post-last-wins):
```
actor_rollout_ref.actor.comm_eff.enabled=true
actor_rollout_ref.actor.comm_eff.compression_type=powersgd
actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema
actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.25
actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.0        # last-wins; first-occurrence was 0.50
actor_rollout_ref.actor.comm_eff.anchor.cadence=20            # last-wins; bare-export default was 5
actor_rollout_ref.actor.comm_eff.anchor.delay_K=20            # last-wins; bare-export default was 5
actor_rollout_ref.actor.comm_eff.powersgd.rank=77
actor_rollout_ref.actor.comm_eff.anchor.owns_q=true
actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch=true
actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=cpu
actor_rollout_ref.actor.comm_eff.spectral.ema_device=cpu
actor_rollout_ref.actor.comm_eff.spectral.max_targets=-1
actor_rollout_ref.actor.comm_eff.spectral.diagnostics=false
trainer.total_training_steps=100
trainer.experiment_name=exp-37c-cad20-delay20-beta0-100step
```
Plan-vs-launched divergence: **expected and intentional.** The plan file `37.md`
documents EXP-37B (5/5, beta_anc=0.50). EXP-37C is an operator-added sibling whose
trailing Hydra overrides flip three knobs vs that plan: `beta_anc 0.50 -> 0.0`,
`cadence 5 -> 20`, `delay_K 5 -> 20`. The launcher's bare exports
(`COMM_EFF_ANCHOR_CADENCE=5`/`DELAY_K=5`, `beta_anc=0.50`) appear as the *first*
occurrences and are correctly clobbered by the trailing args (Hydra last-wins) — this
is the known accel-base banner footgun; the resolved cmd, not the banner, is ground
truth. The 20/20 latency is realized in the counters (10 fires, exactly 200 ticks /
cadence 20). No unintended drift.

## Notes
- **Completion verification**: no `done.flag`, but the run is complete — tmux/main_ppo
  process is dead (no live process), train.log is non-empty and reached
  `training/global_step:100` with all 4 validations, and WandB finished `exit_code=0`.
  Contract's OR-branch (session dead AND train.log non-empty) is satisfied.
- **`analyze.py` emitted a PENDING scaffold** which this hand-written verdict replaces;
  `baseline_diff.md` is "no common numeric keys" for both baselines because
  diagnostics=false produced no train.jsonl on any of the three runs. All numbers above
  are grepped from train.log scalar rows (steps 1-100) — none invented. analysis.log
  captures all four script invocations.
- **Deliverable classification (per the back-half observable)**: OSCILLATING /
  recurring-but-recoverable instability. Two length-ignition events (35-41, 88-100)
  bracketing a full recovery (val@75=0.70). This is distinct from EXP-37's single
  terminal collapse and from EXP-37B's clean stability.
- **No EMA / latency sweep is recommended off this result** (`iterations: 1` on the
  lineage; the high-latency operating point is now characterized for both beta values
  and is non-viable). The viable operating point remains 5/5 (EXP-37B PASS).
- **Issue label untouched**: #37 stays `status:pass` (EXP-37B). EXP-37C is recorded
  here + in PROGRESS only; it is a sibling exploration, not a downgrade of #37.
- **Ledger/box untouched** per instructions: EXP-37C row is already COMPLETE and the
  box is reused for EXP-37D.
- **Mechanism note for the planner/theorist** (consistent with the entropy-collapse
  memory line): at 20/20 the anchor gradient is maximally stale (10-global-step lag);
  signed_ema's `|G|.sign(M)` then applies the sign of a stale gradient to current
  magnitudes. beta_anc=0 = no averaging -> the sign flips track the freshest-but-stale
  M, giving recoverable oscillation; beta_anc=0.50 = EMA -> the accumulated stale sign
  locks in, giving a later but terminal collapse. The shared root cause is **anchor
  staleness at 20/20**, which the merger amplifies into a length/entropy-sharpening
  spiral regardless of beta. This matches the pre-existing finding that the
  M_anchor-carrier instability is structural at high latency.
