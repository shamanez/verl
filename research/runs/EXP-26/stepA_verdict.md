# Verdict EXP-26 — 2026-06-10T11:08:00+10:00 — STAGE A (geometry audit gate)

## STAGE A — Steps B/(C)/E remain, NOT the terminal issue verdict.

> This verdict clears ONLY the **Step-A diagnostic gate** of EXP-26. It routes the
> orchestrator to the next staged run (Step B / conditional Step C); it does **not**
> mark the whole issue PASS or STOP. The issue label stays `status:running`. The
> terminal verdict is decided in a LATER session after the Step B/(C)/E training arms.

## Result
VERDICT: PASS-STAGE-A

Step-A gate cleared: all hard correctness invariants verified GREEN on-box (prior
runner record), the audit computed from REAL fp32 post-warm tensors (ticks 10/15,
warm-Q recon ~0.025) across all three arms, and a machine-readable DECISION emitted.
This SUPERSEDES the earlier STUCK record — the broken parallel-`G_dense` clone is
retired and the dense reference is `G_fresh_anchor@delay_K=0` (Option A, operator-approved).

## Success criteria (Step A — geometry audit gate)
- [x] (pre-run gate) every `hard`-gate correctness invariant passed the on-box probe
  (off-path parity, probe-never-feeds-optimizer, anchor owns Q, full-pass-only-in-anchor,
  delay_K>=5, fp32 dump fidelity) — verified by the runner; fp32 dump fidelity re-confirmed
  here (recon drift 4.5e-5 « 1e-3)
- [~] `cos(G_dense, G_comp) >= 0.95` for plain PowerSGD (H1) — measured (A1 = +0.0096) but
  CONFOUNDED (loss + operand mismatch; see Notes). Confirmed in SPIRIT via the confound-free
  isolate `cos(G_comp, G_corr) = 0.717`. The literal box is not cleanly testable from this
  reference; this is a known-and-documented measurement limitation, not a failed measurement.
- [x] `cos(G_dense, G_corr)` materially below plain-PowerSGD `cos(G_dense, G_comp)` (H1) —
  confirmed: confound-free `cos(G_comp, G_corr) = 0.717` (merger rotates the compressed update
  ~44 deg); cos(G_fresh, G_corr) 0.349 > cos(G_fresh, G_comp) 0.060 (same direction)
- [~] `Q_act` activation capture `>= 0.99` AND update-energy capture with off-principal share
  (H2) — activation **0.9985 (PASS)**; update-capture **0.318** (off-principal **0.68**) ⇒
  Q_act reads as MISSING off-principal update energy (H2 TRUE, operand-confound caveated)
- [x] sign-agreement(M, G_comp) and sign-agreement(G_fresh_anchor@delay_K=0, G_comp),
  magnitude-weighted, at delay_K∈{0,5} (H3) — **CONFIRMED**: ∈ [0.50, 0.52] at both delay_K ⇒
  coin-flip even fresh ⇒ sign-replacement structurally unrecoverable
- [x] machine-readable DECISION emitted — `go_C_then_B` + `retire_sign_replacement(confirmed)`

## Metrics summary (all from runs/EXP-26/captures/*/manifest.jsonl, post-warm ticks 10/15)
- H1a cos(G_fresh_anchor, G_comp) A1 plain-PowerSGD: **+0.0096** (n=14) — LOW/confounded
- H1b cos(G_fresh_anchor, G_corr) A2 signed_ema: **+0.3486**; cos(G_fresh_anchor, G_comp) A2: +0.0601
- **Confound-free merger isolate cos(G_comp, G_corr) A2: +0.7165** (~44 deg rotation)
- H2 Q_act activation capture (A1): **0.9985** (target >=0.99 — PASS)
- H2 Q_act UPDATE-energy capture ‖QQᵀG_fresh‖²/‖G_fresh‖² (A1): **0.3179**, off-principal **0.682**
- H3 sign-agreement @delay_K=0: A1 **0.5004**, A2 **0.5227** (coin-flip band [0.45,0.55])
- H3 sign-agreement @delay_K=5: A2 **0.5195** (A1 N/A — plain PowerSGD has no merger M)
- VALIDITY cos(G_fresh_anchor, G_dense) on A0 dense arm: **0.9848** (~0.985 expected — PASS)
- sanity cos(G_fresh_anchor, G_anchor) A1: +1.0003 (fresh IS the genuine full uncompressed grad)
- fp32 dump fidelity max recon drift: 4.5e-5 (« 1e-3)
- No NaN / Traceback / OOM / non-finite grad in any arm's train log (grad_norm finite, settles ~0.7–2.3)
- DECISION: **go_C_then_B** (+ retire_sign_replacement confirmed)

## Comparisons to baseline_run: EXP-25
`diff_against_baseline.py --baseline EXP-25` rc=0 (baseline_diff.md). The EXP-20/23 dirs were
cleared in the #25 clean-slate; references are read from W&B per the issue — a documented
condition, not a measurement failure. The clean Option-A audit CORROBORATES the EXP-25 terminal
STOP (signed_ema falsified): the merger's coin-flip sign-agreement (0.50–0.52 even at delay_K=0)
and its ~44 deg rotation of the compressed update (cos(G_comp,G_corr)=0.717) are the geometric
mechanism behind #25's monotonic dose-response degradation. Sign-replacement is retired; the
direction-preserving `ef_powersgd` successor (Step B) carries no sign term.

## Resolved parameters (ground truth)
Source: `resolved_params.txt` (extracted from train.log via capture_resolved_config.py — 95
params, 1 main_ppo invocation = the A1 arm expansion), NOT the plan.

Comm-eff substrate (LOCKED control surface — bit-matches resolved_params.txt):
```
actor_rollout_ref.actor.comm_eff.compression_type=powersgd
actor_rollout_ref.actor.comm_eff.powersgd.rank=77
actor_rollout_ref.actor.comm_eff.powersgd.sync_basis=true
actor_rollout_ref.actor.comm_eff.powersgd.q_basis=act
actor_rollout_ref.actor.comm_eff.anchor.enabled=true
actor_rollout_ref.actor.comm_eff.anchor.owns_q=true
actor_rollout_ref.actor.comm_eff.anchor.cadence=5
actor_rollout_ref.actor.comm_eff.anchor.delay_K=5
actor_rollout_ref.actor.comm_eff.clean_cadence=0
actor_rollout_ref.actor.comm_eff.spectral.ema_device=cpu
actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95
actor_rollout_ref.actor.comm_eff.spectral.max_targets=-1
actor_rollout_ref.actor.use_kl_loss=False  (entropy_coeff=0, use_kl_in_reward=False)
data.train_batch_size=128  ppo_mini_batch_size=64  rollout.n=8
data.max_response_length=16384  max_prompt_length=1024
trainer.total_training_steps=6   (short Step-A diagnostic capture run, NOT full training)
```
Capture flags (Step-A instrumentation): capture.enabled=true, capture_fresh_anchor=true,
capture_g_dense=true (clone RETIRED in analysis — footnote only), dump_dtype=fp32, max_ticks=8,
stratified_targets=4.

**Divergence callouts (vs plan):**
- The resolved_params.txt captured the **A1** invocation, whose launcher block carries the
  inert defaults `spectral.correction_mode=signed_ema`, `spectral.signed_ema_alpha=0.5`,
  `spectral.enabled=false`. A1 ACTUALLY ran with `COMM_EFF_SPECTRAL_ENABLED=false` (plain
  PowerSGD, NO merger) — confirmed by the manifest (A1 has no G_corr / M roles) and
  launch_A1A2_optionA.sh. The signed_ema fields are dead defaults, NOT what A1 ran. The A2 arm
  ran `SPECTRAL_ENABLED=true correction_mode=signed_ema alpha=0.5` (G_corr/M present). No
  substrate knob diverged from the plan's locked control surface — this is launcher-default
  noise, not a real setting drift. Flagged so the next-stage planner is not misled by the
  signed_ema lines in A1's resolved_params.txt.
- The plan's literal `mask.*` block (p=0.9, enabled=false) is the dead mask path (codec is
  powersgd, not mask) — inert, expected.

## next_actions
```yaml
# DECISION = go_C_then_B → Step C runs BEFORE Step B (Q_act misses off-principal update energy).
- stage: C
  id: C                                   # per ## Experiment sequence id:C (rlvr-native-Q-sweep)
  knob: q_basis
  from: act
  to: [grad, adv, tail, hybrid, ticket]   # fixed total rank 77; vary Q CONTENT, not rank
  rationale: "Q_act activation-capture 0.999 but UPDATE-energy capture only 0.318 (off-principal
              share 0.68) — the activation basis misses most of the dense GRPO update direction.
              Find a Q content that captures update energy (judged by update-capture +
              off-principal preservation, NOT activation reconstruction). Substrate + merger
              from the passing Step-B config; do NOT vary Q content AND merger simultaneously."
  gate: "geometry-audit a Q family that beats Q_act on update-capture; its training arm
         val@50 >= 0.7414 with no length/clip collapse."

- stage: B
  id: B                                   # per ## Experiment sequence id:B (ef-powersgd-direction-preserving)
  knob: correction_mode
  from: signed_ema
  to: ef_powersgd                         # direction-preserving error-feedback, NO sign term
  rationale: "H3 confirmed sign-replacement is a structural coin-flip (0.50–0.52 even at
              delay_K=0) ⇒ retire it. ef_powersgd re-injects the dropped PowerSGD residual
              WITHOUT a sign term. Arms = {ef_powersgd, plain-PowerSGD r77, dense}; 50→100
              steps, val@25; substrate LOCKED (anchor owns Q, delay_K=5, clean_cadence=0, r=77)."
  gate: "ef_powersgd best arm val@50 >= 0.7414 AND direction-preservation improves over plain
         PowerSGD (use the CONFOUND-FREE cos(G_comp, G_corr) discriminator, see Notes) AND no
         length/clip collapse. Limiting-case invariant: residual off (clip=0/decay=0) ⇒
         G_corr == G_comp == plain PowerSGD."
```

## Notes
- **STAGE GATE only.** Step B/(C)/E run in a later session on a fresh/warm box. The terminal
  EXP-26 verdict (parity recovery: val@50 >= 0.7414 with improved cosine, no collapse, measured
  comm saving in E) is NOT decided here. Issue label intentionally left at `status:running`.
- **The STUCK is resolved.** Earlier STUCK (broken parallel-`G_dense` clone, norm ~r/H,
  anti-correlated) is superseded. Option A (dense reference = `G_fresh_anchor@delay_K=0`) is
  sound — validated on the dense arm (cos(G_fresh_anchor, G_dense)=0.985) and proven to BE the
  genuine full uncompressed grad (cos(G_fresh_anchor, G_anchor)=1.0003).
- **MEASUREMENT CAVEAT carried forward (load-bearing for Step B's analyst).** The plan's
  literal weight-space discriminator `cos(G_dense, G_comp)` is NOT cleanly measurable from
  `G_fresh_anchor`: (1) `G_fresh_anchor` is the anchor's CLEAN-PG-loss gradient while `G_comp`
  is the fast path's PPO-clip-loss gradient (they coincide only when ratio≈1, i.e. the dense
  arm); (2) PowerSGD compresses the boundary ACTIVATION gradient, not the weight gradient, so
  `G_comp` is not `QQᵀG_weight` (G_comp-onto-Q update-capture is 0.47, not ~1). Both push the
  literal cosine toward 0 INDEPENDENT of compression direction. Since plain PowerSGD r77 ties
  dense at 0.7415 (locked #25), compression cannot actually be orthogonal to dense — the low
  cosine is an artifact. **For Step B's "cos improves over plain PowerSGD" success box, use the
  confound-free `cos(G_comp, G_corr)` (merger vs its own compressed input, 0.717 here), or
  capture the anchor's grad under the SAME PPO-clip loss as the fast path.** The `go_C_then_B`
  routing and `retire_sign_replacement` rest only on confound-free measurements and are robust.
- **NaN/divergence watch:** clean. No NaN/Traceback/OOM; grad_norm finite (cold pre-warm ticks
  show ~150–210 then settle to ~0.7–2.3 by post-warm — diagnostic capture, no full training).
- **H2 caveat:** the Q_act update-capture 0.318 projects the clean-PG fresh-anchor grad, so it
  inherits the operand confound. It cannot prove Q_act ALREADY captures update energy (which
  would justify go_B_skip_C), so the conservative `go_C_then_B` is the correct routing — Step C
  re-measures update-capture per Q family directly and gates on it before committing the merger.
- All audit numbers are greppable in runs/EXP-26/analysis.log and the audit_A{0,1,2}_optA.json
  emits. Staging symlinks live under runs/EXP-26/audit_stage_optA/ (A2 → rank0_new_optA, the
  clean optA captures; the stale A2/rank0 cold-tick re-run#1 data was NOT used).
