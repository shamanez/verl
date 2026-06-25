# Verdict EXP-41 — 2026-06-25T18:05:00Z

## Result
VERDICT: STOP

The look-ahead cell (B, fixed_linear 20/20) **collapsed by step 100** per the plan's own
operationalized collapse test, and `val@100` fell far below cell A's 5/5 reference band. The
fixed-linear-look-ahead-at-20/20-reaches-the-5/5-band hypothesis is **falsified on this surface**.
Cell B's `on_fail` ("If B ignites/collapses or fails a hard invariant, STOP — a learned projector
initialised at the same point will not rescue an unstable integration") and the Analyst-predicate
STOP clause ("any look-ahead cell ignites/collapses by step 100") both fire. Cell C (learned) is
OFF the table — it is conditional ONLY on B being clean-but-underlifted, and B was NOT clean.

Note the genuine tension and why it does not flip to REVISE: the anti-damping alignment lift IS
present (+0.027, 6/8 true fires positive, peak +0.131), so the *other* STOP-trigger ("alive but
no lift / damped toward the dead floor") is NOT met, and the historic catastrophic ignition did
NOT recur (no NaN, entropy never ran away). But the plan operationalizes "no collapse" as
`response_length/mean` not exceeding 2× its first-25-step mean — and cell B breaches that line at
8 distinct steps (peak 552 vs 496 threshold) while val crashes 0.498@50 → 0.115@75 → 0.048@100.
Collapse is a hard, machine-checkable STOP trigger in this plan; it does not become REVISE merely
because lineage `iterations` remain. The beta_anc re-examination the lift+collapse pattern suggests
is recorded for the operator/planner below (the plan itself defers beta_anc to a REVISE-path
follow-up), but the verdict per the verbatim predicate is STOP.

## Success criteria
- [x] (code_change) all 10 `hard`-gate boxes pass the fire-forcing pre-run probe (cited from `runs/EXP-41/probe-invariants.md`: fixed-linear identity, no-leakage, anchor isolation, cross-rank determinism, bounded NEW ring, backend integration, source-snapshot canary, LayerNorm/embed exclusion, alignment telemetry emitted, disabled-path parity) — NOT re-litigated
- [x] every launched cell reaches >= 100 steps with no NaN / non-finite gradients (A: 100 steps, B: 100 steps; no NaN/Inf in any pg_loss/loss/grad_norm; B grad_norm max 410 at warmup step 8, finite)
- [x] training cells use `max_response_length=1024` (resolved_params_A.txt:9 and resolved_params_B.txt:9 — NOT 2048/16K)
- [x] cell A (5/5 reference) reaches 100 steps clean, records val@100 band, AND emits `anchor_align_cos` raw-stale baseline (40 fires, mean +0.006277 — staging NOT gated behind the look-ahead flag, confirmed)
- [ ] **no collapse by step 100** in cell B (observed: `response_length/mean` breaches 2× its first-25-step mean = 496.37 at 8 steps {54,57,58,59,61,62,91,93}, peak 552.3 @ step 59; val crashes 0.498→0.115→0.048; target: resp_mean stays <= 496.37 AND val holds the band) — **FAILS: length-explosion-driven performance collapse**
- [x] **anti-damping alignment lift (load-bearing):** cell B true-lookahead `anchor_align_cos` mean = +0.0329 vs cell A raw-stale baseline +0.0063 ⇒ lift +0.0267 (6/8 true fires positive, peak +0.131 @ tick 120 / gs60); the "alive but no lift" STOP-trigger is therefore NOT met — **LIFT IS PRESENT**
- [~] **off-diagonal lift (reported diagnostic, NOT a hard gate):** `diag_bound` was NOT emitted by the run ⇒ **off-diagonal: not measured**. Does not flip the verdict (must-fix #7).
- [ ] cell B `val@100` reaches cell A's 5/5 reference band (observed: 0.0478; target band: [0.7066, 0.7255]) — **FAILS by ~0.66 absolute**
- [x] look-ahead fires at cadence 20 and per-fire log emits `lookahead_source_ticks`, no source tick `>= t` (machine-checked: ticks 60→[40,20], 80→[60,40], …, 200→[180,160]; spacing = cadence 20; newest source always < current ⇒ no leakage)
- [x] anchor invariant counters clean across the run: `anchor_optimizer_steps` / `anchor_rollouts_generated` / `anchor_rewards_recomputed` / `anchor_mask_applications` all **0.0** at every logged step (both cells)
- [x] controlled variables IDENTICAL across A/B (resolved_params diff: only `anchor.cadence`/`delay_K` 5/5 vs 20/20 and `lookahead_anchor`/`lookahead_mode` false/disabled vs true/fixed_linear; max_response_length, signed_ema 0.25/0.50, PowerSGD r=77, clean_cadence=0, lr=1e-6, batch=128/mini=64, rollout.n=8, total=100 all identical)
- [x] communication substrate unchanged from locked PowerSGD r=77 anchor path (rank=77, model, GRPO no-KL no-entropy, GSM8K, resp=1024 all unchanged)

3 hard boxes unchecked: **no-collapse** and **val@100-band** both FAIL; off-diagonal is a not-flipping
diagnostic. PASS requires ALL hard boxes ⇒ NOT a PASS. The two failing boxes are exactly the collapse
+ band falsification ⇒ STOP (not REVISE), per the cell-B `on_fail` and the predicate's collapse clause.

## Metrics summary
(every value greppable in `runs/EXP-41/verl_train_{A,B}.log`)
- Cell A val-core/openai/gsm8k/acc/mean@1 @25/50/75/100: 0.6998 / 0.7255 / 0.7233 / 0.7066 → 5/5 reference band ≈ [0.7066, 0.7255]
- Cell B val-core/openai/gsm8k/acc/mean@1 @25/50/75/100: 0.3616 / 0.4981 / 0.1145 / 0.0478 (target: A's band)
- Cell A `anchor_align_cos` (raw stale cos(g(θ[t-5]),g_live)): 40 fires, mean **+0.006277** (logged-step snapshots 0.0462 / 0.0140 / 0.00215 / 0.0521)
- Cell B `anchor_align_cos`: 2 warmup raw-stale fires (ticks 20/40: +0.0202, -0.0102) then 8 true-lookahead fires cos(g(θ̂),g_live) at ticks 60/80/100/120/140/160/180/200 = +0.0325 / -0.0741 / +0.0538 / **+0.1310** / +0.0528 / +0.0785 / +0.0176 / -0.0288; **true-fire mean +0.0329, 6/8 positive, peak +0.1310**
- **Alignment lift = +0.0329 − 0.0063 = +0.0267** (load-bearing box PASSES)
- Cell B `response_length/mean`: first-25-step mean 248.19; 2× collapse threshold 496.37; **peak 552.31 @ step 59**; 8 breach steps (two oscillations: ~step 54-62 and ~step 90-93)
- Cell B entropy: warmup 5.76 → declines to 0.78 @ step 100 (never runs away; NOT historic catastrophic ignition)
- Cell B grad_norm: finite throughout, max 410 at warmup step 8, settles 1–14 in steady state (no explosion)
- Anchor isolation counters (both cells): anchor_optimizer_steps/rollouts_generated/rewards_recomputed/mask_applications = 0.0 at every step
- `lookahead_peak_retained_snapshots` = 2 (bounded NEW ring, no leak); `lookahead_excluded_count` = 142 (LayerNorm/embed excluded, 196 decoder matrices extrapolated)
- comm bytes_ratio ≈ 0.0505 (PowerSGD r=77 substrate intact); powersgd_reconstruction_rel_error ≈ 0.033

## Comparisons to baseline_run: current_ce_baseline_20_20 (not re-run; documented prior)
`diff_against_baseline.py --baseline current_ce_baseline_20_20` returned "baseline not found" — by
design: the plan states the 20/20 raw-anchor k-collapse baseline is the **documented prior, NOT
re-run here** (it ignites ~step 61). This is NOT a missing-measurement STOP: the real, in-plan PASS
bar is cell A (the on-surface 5/5 reference), which IS present in this run dir. Against that bar,
fixed-linear look-ahead at 20/20 did NOT reach the band (0.048 vs ~0.71). The relevant contrast vs
the documented raw 20/20 prior: the look-ahead patch DID move the failure mode — it removed the
catastrophic entropy ignition (entropy stayed bounded, no NaN) and DID lift anchor alignment
(+0.027), but a softer length-explosion / val collapse still occurred by step 100. Better aligned
anchor + bounded-but-still-collapsing training is consistent with the merger (signed_ema β_anc=0.50,
tuned for a STALE anchor) over-amplifying the now-fresher, better-aligned projected gradient.

## Resolved parameters (ground truth)
Source: `resolved_params_A.txt` / `resolved_params_B.txt` (rebuilt from each cell's authoritative
`python3 -m verl.trainer.main_ppo` `set -x` trace, one arg per line; last-wins Hydra). NOTE: the
`capture_resolved_config.py` helper expects a single `train.log` and could not run here (this
launcher writes per-cell `train_A.log`/`train_B.log`); the operator-rebuilt per-cell files are the
authoritative provenance and were used directly. No RESOLVED_CONFIG_MISSING flag is warranted — the
real launched parameters ARE recovered.

Comm-eff + headline knobs (verbatim, effective last-wins values):
```
# CELL A (5/5 reference, lookahead DISABLED)
data.max_response_length=1024
actor_rollout_ref.actor.comm_eff.anchor.cadence=5            # last-wins over bare-export 20
actor_rollout_ref.actor.comm_eff.anchor.delay_K=5            # last-wins over bare-export 20
actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor=false
actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode=disabled
actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema
actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.25
actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.50
actor_rollout_ref.actor.comm_eff.powersgd.rank=77
actor_rollout_ref.actor.comm_eff.clean_cadence=0
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=64
data.train_batch_size=128
actor_rollout_ref.rollout.n=8
trainer.total_training_steps=100

# CELL B (fixed_linear 20/20)
data.max_response_length=1024
actor_rollout_ref.actor.comm_eff.anchor.cadence=20
actor_rollout_ref.actor.comm_eff.anchor.delay_K=20
actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor=true
actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode=fixed_linear
# (all other comm-eff + RL knobs IDENTICAL to cell A: signed_ema 0.25/0.50, r=77, clean=0,
#  lr=1e-6, mini=64, batch=128, n=8, total=100)
```
**Divergence check (plan/issue specified vs launched):** NONE. Every controlled variable the plan
pins matches the launched command exactly, and the only two axes the plan permits to vary
(anchor latency A vs B; lookahead_mode) are exactly the two that differ. β_anc=0.50 is the
plan-specified FIXED value (no confound). No plan-vs-launch divergence is itself a finding here.

## Notes
- **The pre-run correctness gate (10/10 hard invariants) passed** — cited from
  `runs/EXP-41/probe-invariants.md`, not re-litigated. The look-ahead implementation is *correct*
  (fixed-linear identity θ̂=2θ[t-20]-θ[t-40] verified, no leakage, anchor-isolated, bounded ring,
  cross-rank-deterministic, LN/embed-excluded, telemetry emitted). The STOP is a *scientific*
  falsification of the hypothesis on this surface, NOT a broken-patch STOP. Do NOT re-enter the
  commit-hotfix loop — the code is sound; the method-at-this-merger is what failed.
- **Telemetry is NET-NEW and present + finite** in both cells (`anchor_align_cos`,
  `lookahead_source_ticks`, `lookahead_fires`, `lookahead_excluded_count`,
  `lookahead_peak_retained_snapshots`). No telemetry-absence STOP applies; this is a measured,
  scorable run.
- **Why STOP and not REVISE (verbatim-contract reasoning):** cell B `on_fail` states collapse ⇒
  STOP and explicitly forecloses cell C ("a learned projector initialised at the same point will
  not rescue an unstable integration"). The Analyst predicate lists "any look-ahead cell
  ignites/collapses by step 100" as a STOP clause. The collapse is machine-confirmed (8 resp_mean
  breaches of the 2× line + val crash to 0.048). Lineage `iterations` remaining does NOT convert a
  collapse-STOP into a REVISE under this plan's contract. The off-diagonal box was not measured and
  (per must-fix #7) does not flip the verdict.
- **For the operator / next-lineage planner (deferred, NOT a next_action of this STOP verdict):**
  the lift-present + still-collapsing pattern is exactly what the plan's REVISE space anticipated —
  a fresher, better-aligned projected anchor gradient may want LESS EMA than the stale-tuned
  signed_ema β_anc=0.50, which now appears to over-amplify the improved anchor and drive the
  length-explosion. If the operator chooses to open a *new* M4 lineage rather than abandon, the
  single highest-value axis is β_anc ∈ {0.50 → 0.10–0.25} (and/or signed_ema_alpha) with
  fixed-linear look-ahead held on at 20/20 — the merger, not the projector, is now the suspect. A
  shorter look-ahead window or coefficient regularization are secondary. This is recorded as a
  research direction for human review; it is NOT emitted as a `next_actions:` block because the
  verdict is STOP.
- Cell A is a clean, reusable on-surface 5/5 reference (val 0.6998/0.7255/0.7233/0.7066, raw-stale
  cos baseline +0.006277) for any future look-ahead lineage on this 1K surface.
- `done.flag` ("A+B complete"), `done_A.flag`, `done_B.flag` all present; both cells train_rc=0;
  box already torn down (external team box). Budget check clean (running_count=0, no live spend).
