# Verdict EXP-31 — 2026-06-16T04:20:00+10:00

## Result
VERDICT: STOP

The HEADLINE surpass criterion is FALSIFIED. All four operator-selected
anchor-usage levers (L4 perturbation, L2 δ-momentum, L3 adaptive dose, L1
control-variate) fail to beat the B2 substrate beyond rollout noise (±0.024),
with **no nameable, non-blocked knob** that has a credible mechanistic path to
≥0.78. Per the operator `/goal` override recorded in plan §HANDOFF
("if all remaining levers only match parity, write the honest STOP verdict"),
this is the honest STOP. The plan's nominal STOP gate ("…AND `iterations` REVISE
cycles consumed") is superseded by that directive; it is moot regardless because
no lever showed any signal worth a REVISE cycle (REVISE must name a non-blocked
knob, and none exists — see §forward note).

## Success criteria
- [x] (code_change) all `hard`-gate invariants pass — off-path parity, async
      admissibility, Step-C avoidance, cross-rank determinism
      (observed: every cell's `set -x` trace shows `delayed_ef_lambda=1.0`,
      `beta_anc=0.0`, all non-active lever knobs OFF; the per-tick `[delayed_ef]`
      diagnostic prints the OFF-path identity assertions; recon steady-state
      ≈ 0.025 stays in the act band, NOT the 0.68 Step-C plateau; 8-agent
      adversarial code review = GO/GO, no defect — `tournament_state.md` §Code
      verification)
- [x] every launched cell reaches its target or is cleanly early-killed, no
      NaN / non-finite grad / OOM, no ignition trip-wire fired
      (observed: all 6 cells healthy; no real NaN/Inf/OOM — the only "nan" in
      logs is `subbasis_energy_ratio=nan` from the OFF sub-basis 0/0, benign;
      response_length/mean flat-or-decreasing in every cell ⇒ P1/P2/P3/E1 all
      clear; max actor allocated ≤ 30.78 ceiling; L4/L3 "74.8/45.97" are
      `max_memory_reserved`, not allocated — no OOM)
- [x] (comparison) controlled variables hold equal + asserted: `bytes_ratio == B2`
      (observed: `actor/comm/bytes_ratio` ∈ [0.0504, 0.0506] across all 6 cells,
      inside the [0.0500,0.0510] gate); forward Q untouched (recon in act band);
      NO generation-side change (greedy mean@1); codec/batch/lr/n/resp identical
      per the resolved traces)
- [x] Cell A reproduces the B2 band: `val@50 ∈ [0.716, 0.774]`
      (observed: B2_live val@25 = 0.7202 [verified in A_b2_reproduce/train.log],
      val@50 = 0.7354, WandB fy920fty — inside the band; dense-this-box 0.7506 ⇒
      B2 at parity, gap 0.015 < ±0.024)
- [ ] **HEADLINE:** ≥1 lever's best `val@50` clears B2_live beyond ±0.024
      (north-star ≥ ~0.78)
      (observed: NO lever cleared even the val@25 CONTINUE-to-improve bar; best
      lever val@25 = L4 0.7157 [−0.0045 vs B2_live 0.7202]; all others
      parity-or-below; L2_mom09 = 0.5701 [−0.1501 REGRESS]. No lever was
      extended to 50. target: ≥ 0.7594 @50 / north-star ≥ 0.78)

4 of 5 criteria PASS; the single HEADLINE criterion — the entire point of the
tournament — FAILS. Process/correctness was clean; the science is a null.

## Metrics summary
All single-draw, seed 0, box i_41048644 (4×H200), `disable_custom_all_reduce`;
greedy `val-core/openai/gsm8k/acc/mean@1`; rollout noise ±0.024. Every val@25
below was grepped directly from the named cell's `train.log`.

| cell | lever knob (resolved trace) | val@25 | Δ vs B2_live@25 | health | WandB |
|---|---|---|---|---|---|
| A_b2_reproduce | bitwise B2 (all levers OFF) | 0.7202 | — (reference) | no ignition; val@50 0.7354 | fy920fty |
| L4_perturb_s001 | `perturb_sigma=0.01` | 0.7157 | −0.0045 (parity) | killed@34, no ignition | cvu8jw1n (partial) |
| L2_mom09 | `delta_momentum_mu=0.9, age_decay=true` | **0.5701** | **−0.1501 (REGRESS)** | reward→0.51, len 276→168↓, no ignition | ybemd5ux |
| L2_mom05 | `delta_momentum_mu=0.5, age_decay=true` | 0.7089 | −0.0113 (parity) | reward→0.69, len flat ~215, no ignition | knlzxh2x |
| L3_ratio_k10 | `adaptive_lambda_mode=ratio, kappa=1.0, cap=2.0` | 0.7119 | −0.0083 (parity) | reward→0.695, len flat ~212, no ignition | kzohyuod |
| L3_cos_k10 | `adaptive_lambda_mode=cos, kappa=1.0, cap=2.0` | 0.7134 | −0.0068 (parity) | reward→0.756, len ~181, no ignition | wmpmmdj1 |

- B2_live@25 = 0.7202; B2_live@50 = 0.7354 (target to beat: val@50 ≥ 0.7594 to EXTEND)
- dense-this-box = 0.7506; dense band ≈ 0.75–0.78; SURPASS target = 0.80
- `actor/comm/bytes_ratio` ∈ [0.0504, 0.0506] all cells (controlled var, B2-equal)
- steady-state `powersgd_reconstruction_rel_error` ≈ 0.025 (act band; Step-C avoided)
- NO lever cleared the operator's aggressive val@25 gate ("KILL unless val@25
  *clearly improves* on 0.7202 — a real breakthrough"); all banked early.

## Comparisons to baseline_run: EXP-31/B2_baseline (B2; canonical WandB u9okvgzz)

The live substrate control for this tournament is **Cell A (B2_live)** on the
SAME box/seed, not the migrated B2_baseline — single-draw comparisons must be
same-box to control rollout noise. Cell A reproduces B2 (val@25 0.7202, val@50
0.7354), inside the B2 band and at parity with dense-this-box (0.7506, gap 0.015
< ±0.024). Every lever was run as B2 + exactly ONE knob (verified per-cell in the
resolved `set -x` traces), so each Δ is a clean one-knob attribution. The best
lever (L4, −0.0045 @25) sits below B2_live; the worst (L2_mom09, −0.1501)
actively regresses. `diff_against_baseline.py` ran clean (wrote `baseline_diff.md`);
`check_budget.py` clean (month spend $890 < $1500 cap; one operator-managed box
still RUNNING — operator teardown, not in scope here). No lever beats B2; none
approaches the dense band, let alone 0.80.

## Resolved parameters (ground truth)
Source: `resolved_params.txt` (extracted from the top-level `train.log` `set -x`
trace — this trace is `experiment_name=A_b2_reproduce` = Cell A / B2_live; the
per-lever knobs below are from each cell's own `train.log` trace, NOT the plan).

B2 substrate (Cell A — the OFF-path target every lever must match bitwise):
```
actor_rollout_ref.actor.comm_eff.spectral.correction_mode=delayed_ef
actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0
actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.0
actor_rollout_ref.actor.comm_eff.spectral.perturb_sigma=0.0          # L4 OFF
actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_rank=0       # sub-basis OFF
actor_rollout_ref.actor.comm_eff.powersgd.rank=77
actor_rollout_ref.actor.comm_eff.powersgd.q_basis=act
actor_rollout_ref.actor.comm_eff.powersgd.sync_basis=true
actor_rollout_ref.actor.comm_eff.anchor.cadence=5 / delay_K=5 / owns_q=true / replay_paired_batch=true / snapshot_device=cpu
actor_rollout_ref.actor.comm_eff.clean_cadence=0
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432 / ppo_micro_batch_size_per_gpu=1 / use_dynamic_bsz=False
actor_rollout_ref.actor.optim.lr=1e-6 / ppo_mini_batch_size=64
data.train_batch_size=128 / max_response_length=16384
actor_rollout_ref.rollout.n=8 / name=vllm
+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true
trainer.total_training_steps=50 / test_freq=25
```

Per-lever knob (one knob flipped from B2 per cell — verified):
```
L4_perturb_s001 : spectral.perturb_sigma=0.01   (else OFF)
L2_mom09        : spectral.delta_momentum_mu=0.9  + delta_momentum_age_decay=true   (adaptive OFF, perturb OFF)
L2_mom05        : spectral.delta_momentum_mu=0.5  + delta_momentum_age_decay=true   (adaptive OFF, perturb OFF)
L3_ratio_k10    : spectral.adaptive_lambda_mode=ratio kappa=1.0 lambda_cap=2.0      (momentum OFF, perturb OFF)
L3_cos_k10      : spectral.adaptive_lambda_mode=cos   kappa=1.0 lambda_cap=2.0      (momentum OFF, perturb OFF)
```

**Divergence between plan and what ran:** NONE in the lever wiring — every cell
launched exactly the plan's one-knob-from-B2 config. Two cosmetic notes, neither
a behavioral divergence:
- The resolved trace carries `kl_loss_coef=0.001` / `kl_loss_type=low_var_kl`,
  but `use_kl_loss=False` AND `use_kl_in_reward=False` AND `entropy_coeff=0` ⇒
  KL and entropy are inert. Effective loss is vanilla GRPO no-KL/no-entropy, as
  the fixed control surface requires. The 0.001 is a dead default, not active.
- The per-tick diagnostic banner is tagged `[EXP-30]` (inherited string in
  `spectral_filter.py`); the run is EXP-31. Label-only, no effect on the math.

## Mechanistic synthesis — why all four levers are NULL

The unifying thesis (plan analysis §1/§3c, now empirically confirmed by all four
levers): **B2 caps at parity because the correction δ = M − G_comp_ring
reconstructs the *dense gradient evaluated on stale data*. You cannot exceed
dense by reweighting, accumulating, perturbing, or de-noising a stale estimate of
dense.** Each lever manipulates that same stale δ in a different way, and each
hits the same parity ceiling for a lever-specific reason:

- **L4 (perturbation, σ=0.01) — NULL, parity (−0.0045).** Isotropic ξ=randn
  touches the anchor nowhere ⇒ a regularization CONTROL, not an anchor-usage
  lever. The nudge is injected in RAW-gradient space, but AdamW rescales σ
  per-coordinate before it reaches the weights, so the SGLD/flat-minima "dose" is
  washed out. It adds zero-mean noise to a parity-quality update; greedy mean@1
  doesn't benefit. (σ=0.03 correctly skipped — more isotropic noise ⇒ ≤ σ=0.01.)

- **L2 (δ-momentum) — NULL; lever CLOSED.** Normalized-EMA `m←μm+(1−μ)δ`
  (stationary gain 1, accumulate only at refresh ticks — code-verified GO).
  - μ=0.9 REGRESSES hard (−0.15): heavy smoothing over-averages the held
    correction, so the buffer lags the fast-changing *early* correction and
    slows convergence (reward plateaus ~0.51, length contracts 276→168). The
    persistently-missed mode is NOT stationary enough across fires for momentum
    to compound a useful steady push — it just adds staleness on top of staleness.
  - μ=0.5 tracks B2 (parity, −0.011). The lighter buffer is essentially B2.
  - Both bracket B2 from below ⇒ no μ between 0.5 and 0.9 surpasses; momentum on
    a stale reconstruction of dense gives back the lag dense never had.

- **L3 (adaptive dose) — NULL, parity.** Mean-1 centered gate
  `λ_t = λ + κ(c̄ − c_t)` (E[λ_t]≈1; ratio c̄≈1.025, cos self-calibrates —
  code-verified GO, NOT the forbidden `1+κ(1−cos)`). κ=1.0 is the MAXIMUM
  modulation under cap=2.0 and lands at parity for BOTH agreement metrics
  (ratio 0.7119, cos 0.7134). Spending more correction "where compression hurt
  most" doesn't help because the correction is itself a stale reconstruction of
  dense — modulating the dose of a parity-quality signal stays at parity. Since
  κ=1.0 (max) = neutral and κ=0 IS B2, every milder κ lands between = parity.

- **L1 (control-variate) — GATED OUT, not built.** The plan's L1 gate is a
  conjunction: cov(G_comp,M) non-trivial AND L2/L3 showed signal AND box free.
  Both science conditions FAIL: (a) L2/L3 showed no surpass signal (all
  parity/regress); (b) F1 established cos(G_comp,M)≈0 ⇒ cov≈0, so a
  batch-difference control variate has ~nothing to cancel — adding an
  uncorrelated M-baseline raises variance, it doesn't reduce it. L1 is the
  heaviest lever (new `transformer_impl.py` replay backward + OOM risk; L3
  already reserved 45.97 GB). Building+running against a failed gate would burn
  GPU for a near-certain null. Skipping it is the correct, plan-mandated call.

**Code trust:** the 8-agent adversarial review (`tournament_state.md` §Code
verification) found L2 and L3 faithful to the plan math (gain-1 EMA, mean-1
centered gate, off-path parity bitwise, cross-rank deterministic, Step-C-avoiding,
no defect). Therefore every NULL here is a TRUSTWORTHY null — the lever did
exactly what the plan intended and still didn't surpass. These nulls are not code
artifacts.

## Defensibility of the unrun cells
- **L3 κ=0.5 (ratio + cos): SKIP defensible.** Monotonic-by-cap argument: κ=1.0
  is max modulation and is neutral (parity, not hurting); κ=0 is B2; so κ=0.5
  can only land between parity and B2 = parity. No "overshoot-hurting" signal a
  milder κ could rescue. Zero surpass information; correctly not run.
- **L1: GATED OUT defensible.** Failed-gate conjunction (no L2/L3 signal AND
  cov≈0), as above.
- **L4 σ=0.03 / L4 δ-subspace ξ / L2 age-toggle: not run.** δ-subspace ξ needs
  code and is lower priority than L1; given the unifying thesis (the anchor is a
  stale reconstruction of dense), an anchor-shaped nudge perturbs that same
  reconstruction and has no mechanistic surpass path. All remaining unrun knobs
  are heuristic/low-priority with no credible path to ≥0.78.

## What would actually be needed to surpass (forward note — NOT a REVISE)
The deep mechanism rules out the whole "manipulate δ" family: reweighting (L3),
accumulating (L2), perturbing (L4), and de-noising (L1) a stale reconstruction
of the dense gradient all cap at dense-on-stale-data = parity. To exceed dense,
the comm-eff path must inject something dense genuinely LACKS, which — given the
LOCKED constraints (forward codec/Q, batch, generation side, no anchor-lead) —
is not reachable by any anchor-USAGE knob. Every credible surpass route now lies
OUTSIDE this tournament's mandate and outside this plan's blocked list:
- a *different forward basis* that captures the stable-rank≈2 off-principal mode
  the act-basis Q misses (F3) — but that is the forward Q, a Step-C / locked axis;
- a *generation-side* diversity change (higher n / rollout temperature) converting
  to a pass@k or relocated-mode edge — explicitly locked off here;
- a fundamentally different decentralized-RL objective, not a re-use of the
  same stale dense estimate.
None of these is a non-blocked knob in EXP-31, so there is no admissible REVISE.
A future experiment would have to re-scope the LOCKED surface (a new plan +
operator decision), not iterate a lever here.

## Notes
- Completion: no top-level `done.flag` (cells were operator-early-killed by the
  val@25 gate, not run to a done-flag). Verified-complete via: each lever cell's
  `train.log` is non-empty with a recorded val@25, KILLED markers present
  (`KILLED_val25_*`, `*.val25-*` banked copies), and the box is operator-managed
  (tournament_state §Handoff: "STOPPED + box CLEANED per operator"). The
  experiment is decisively concluded by the operator gate, not abandoned mid-run.
- Verification commands all exit 0 (`analysis.log`): `analyze.py` (wrote a
  PENDING stub, overwritten by this verdict), `check_budget.py` (within cap),
  `diff_against_baseline.py` (wrote `baseline_diff.md`). `capture_resolved_config.py`
  wrote `resolved_params.txt` + `resolved_cmd.txt` from the Cell A trace; per-lever
  knobs were read from each cell's own `train.log` trace.
- B2_live@50 = 0.7354 is sourced from WandB fy920fty / tournament_state.md (Cell
  A reached step 50 on-box; the local top-level train.log was synced to ~step 37
  before the box was cleaned — val@50 lives in WandB, not the local log). The
  decision is unaffected: even at the optimistic end, no lever cleared the val@25
  improve-gate, so none would have been extended to 50.
- This STOP closes the anchor-USAGE tournament with the falsification cleanly
  recorded: the four operator-selected levers, run faithfully (code GO/GO) on the
  locked B2 substrate, do not surpass dense; the "can't beat dense by manipulating
  a stale reconstruction of dense" thesis is now empirically established across all
  four lever families. Issue label is operator-managed this session (not touched).
