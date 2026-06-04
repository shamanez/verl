# Verdict EXP-23 — 2026-06-04T18:36:00+00:00

## Result
VERDICT: STOP

Hypothesis FALSIFIED per the plan's pre-registered predicate. A stale (delay_K=5)
full-gradient re-anchor — via additive `inject` (A2, γ=1.0) AND convex `blend`
(A3, η=0.5), fresh clean step OFF — does NOT recover the fresh-clean benefit for
PowerSGD r=77. Falsification line: `max(val@50(A2), val@50(A3)) <= val@50(A1)+0.02`.
Observed: `max(0.6967, 0.6861) = 0.6967 <= 0.6914 + 0.02 = 0.7114` → TRUE.

This is a CLEAN, well-instrumented falsification, not a failed run. The integration
worked exactly as designed: every correctness invariant passed, the anchor+spectral
circuit fired on the PowerSGD path (anchor_backwards=20, spectral_corrections=80 per
A2/A3), codec health stayed green in all arms, and there were zero NaN/OOM/single-GPU
events. The experiment did its job — a decisive negative result with an
evidence-backed next lever.

## Success criteria
- [x] (code_change) every `hard`-gate box in `## Correctness invariants` passed the smoke (step 0) — gated the arms; A2/A3 ran with the circuit live (anchor_backwards=20, spectral_corrections=80) (observed: smoke.done.flag + smoke_fire.done.flag present; circuits fired in arms)
- [x] (code_change) resolved_params for A2 = `correction_mode=inject` + `anchor.delay_K=5` + `anchor.enabled=true` + `clean_cadence=0`; A3 = `correction_mode=blend` (+ same anchor/clean) — launcher wired the knobs, NO silent reweight/delay_K=20 fallback (observed: A2 inject/delay_K=5/enabled=true/clean_cadence=0; A3 blend/delay_K=5/enabled=true/clean_cadence=0 — verbatim from resolved_params__*.txt)
- [x] each of A1, A2, A3 reached `global_step>=50`, `clean_steps==0`, zero NaN/non-finite grad, zero OOM, zero single-GPU fallback (observed: all three reached step 50; clean_steps=0 all arms; 0 NaN/Inf; 0 OOM; 0 single-GPU fallback; world_size=4 confirmed in grad-repr discovery)
- [x] (fairness) codec block identical across A1/A2/A3 AND equal to EXP-20 r=77 (observed: compression_type=powersgd, powersgd.rank=77, sync_basis=true, update_cadence=1, warm_start=true, compress_recompute=true, qr_dtype=fp32, seed=0 — identical in all three resolved_params + matches EXP-20 r=77)
- [x] (fairness) training surface identical across A1/A2/A3 (observed: train_batch=128, ppo_mini=64, micro=1, rollout.n=8, max_response=16384, lr=1e-6, total_training_steps=50, total_epochs=2, GSM8K, use_kl_loss=False, use_kl_in_reward=False, entropy_coeff=0 — identical; declared `ppo_max_token_len_per_gpu` exception A1=36864 vs A2/A3=18432 recorded, NOT a confound — see Notes)
- [x] codec health holds in all arms (observed: q_cond 1.0000002–1.0000004 every step; reconstruction_rel_error < 1.0 every step [steady ~0.019–0.030, matching EXP-20 ~0.024; the only values >0.5 are the 2 PowerSGD warm-start transient steps, 0.976→0.695]; q_cross_rank_max_rel_dev == 0.0 — the sole distinct value — every step, all arms)
- [x] (A2/A3 only) `spectral_corrections>0` AND `anchor_backwards>0` on every anchor-cadence step, finite cos per fire (observed: A2/A3 anchor_backwards=20, spectral_corrections=80; n=320 finite cos diagnostics per arm; A1 correctly 0/0)
- [ ] **HEADLINE (success):** `max(val@50(A2),val@50(A3)) >= 0.7315 AND >= val@50(A1)+0.05` (observed: max=0.6967; bar1 0.7315 missed by −0.0348; bar2 0.7414 missed by −0.0447 — BOTH fail; FALSIFICATION bar 0.7114 met: 0.6967 ≤ 0.7114)
- [ ] **corroboration:** winning A2/A3 train-reward climbs ABOVE A1 and tracks A0's shape (observed: A1 last-5-step critic/score/mean=0.6402, A2=0.6496, A3=0.6531 — A2/A3 are within ±0.013 of A1, statistically indistinguishable from the floor; the mechanism does NOT visibly move the trajectory — consistent with the val falsification)

## Metrics summary
- val@50 A1 (no-refresh floor): 0.6914 (`val-core/openai/gsm8k/acc/mean@1`, last of 6 checkpoints)
- val@50 A2 (stale inject, γ=1.0): 0.6967 (+0.0053 vs floor)
- val@50 A3 (stale blend, η=0.5): 0.6861 (−0.0053 vs floor)
- max(A2,A3): 0.6967 — PASS bar 0.7315 (−0.0348), floor+0.05 bar 0.7414 (−0.0447), FALSIFY bar floor+0.02=0.7114 (MET)
- train-reward (critic/score/mean, last-5-step mean): A1=0.6402, A2=0.6496, A3=0.6531 (all within noise of the floor)
- codec health (all arms): q_cond ∈ [1.0000002, 1.0000004]; reconstruction_rel_error steady ~0.019–0.030 (<1.0 every step); q_cross_rank_max_rel_dev = 0.0 every step
- circuit fires: A1 anchor_backwards=0 / spectral_corrections=0; A2/A3 anchor_backwards=20 / spectral_corrections=80; clean_steps=0 all arms
- safety: 0 NaN/Inf, 0 OOM, 0 single-GPU fallback (world_size=4) in all three arms

### Geometry suite (measure-then-decide deliverable)
Source: the `[comm_eff][EXP-18][inject|blend]` diagnostics in `exp-23-A2/A3-*.train.log`,
n=320 per-target fires per arm.
- cos(G_powersgd, M_anchor) — A2 inject: abs(cos) ∈ [0.00000, 0.00480], mean=0.00110.
  A3 blend: abs(cos) ∈ [0.00000, 0.00400], mean=0.00102. NEAR-ORTHOGONAL — an order
  of magnitude MORE orthogonal than the mask codec's cos≈0.5 in EXP-21 (where reweight
  was inert). Persists across all 50 steps of both arms (not a warm-start artifact).
- complement fraction `‖M − proj_G(M)‖/‖M‖ = √(1−cos²)` ≈ 0.999999 (A2 and A3) — i.e.
  essentially ALL of the stale full gradient is missing from G's span. The "missing
  direction" inject is meant to add IS effectively the entire (scale-matched) M.
- norm ratios — A2 inject: `‖inj‖/‖G‖ = 1.0000` exactly (unit-normed complement), but
  `scale = ‖G‖/‖M‖` ∈ [0.0002, 0.2533], mean=0.0338 (‖M‖ ≫ ‖G‖). NET inject correction
  = γ·scale·complement ⇒ a TINY (~0.03·‖G‖) orthogonal nudge: the injection is
  unit-normed but scaled DOWN to a small orthogonal noise vector. A3 blend:
  `‖G_corr‖/‖G‖` ∈ [0.7059, 0.7085], mean=0.7071 = √0.5 (the exact η=0.5 orthogonal
  bound — blend merely shrinks the step to 0.71×, swapping half of live G for a stale
  orthogonal direction).
- staleness rotation cos(G_t, G_{t−1}) — NOT captured (the plan marked this the one
  genuinely-new log and explicitly DEFERRED it to the follow-up if not cheap+clean to
  add; it was deferred, correctly, to avoid perturbing the run).
- clean-calibration cos(G, G_clean_fresh) — NOT captured this run (no clean step;
  clean_cadence=0). Per the plan, EXP-20's clean@K fresh-full-grad traces remain the
  reference; not re-run.

**Coverage caveat (4/196) — the analyst respects the `## Notes for analyst` interpretation
note.** Only 4 distinct targets were corrected — `model.layers.0.{q,k,v,o}_proj` — because
the correction loop breaks at `spectral.max_targets=4` and iterates in `named_parameters`
order, always stopping at layer-0 attention. That is 4 of ~196 candidate 2D boundary
matrices (28 decoder layers × ~7 target types). So the measured orthogonality is
LAYER-0-ATTENTION ONLY; whether MLP (`gate/up/down_proj`) or deeper layers have a less
orthogonal cos(G,M) is UNMEASURED. Per the plan's caveat, cos(G,M)/complement measure the
RELATIONSHIP between our two estimators (lossy-current G and stale-full M), NOT either's
individual error vs a ground-truth current full gradient — only a (sparse, absent here)
clean step grounds that. Raw matrices were not dumped (scalars only). The falsification
verdict is on the headline val + train-reward predicate, which is full-coverage; the
geometry is the mechanism explanation, and its 4/196 scope is a confidence caveat on the
"orthogonality is model-wide" claim, NOT on the verdict.

**Mechanism (why both modes are inert/near-inert):** PowerSGD r=77 throws away exactly
the directions M lives in, so G ⊥ M is the expected (and measured) outcome. (i) `inject`
re-adds M's orthogonal complement but scale-matched DOWN by ‖G‖/‖M‖≈0.03 ⇒ the net
correction is a tiny orthogonal noise vector, too small to move the trajectory. (ii)
`blend` mixes in a scale-matched but orthogonal M at η=0.5 ⇒ it only shrinks the step to
0.71× while swapping out half the live on-policy signal for a stale orthogonal direction
that is not a current descent direction. Neither operator can supply the missing
descent component because the stale anchor, after scale-matching, simply does not point
where G needs help.

## Comparisons to baseline_run: EXP-20
`diff_against_baseline.py runs/EXP-23 --baseline EXP-20` found NO common numeric keys
(EXP-23 has no per-arm `metrics/train.jsonl` — only raw `exp-23-*.train.log`; EXP-20's
metrics live likewise). As the plan explicitly authorizes, the head-to-head deltas were
computed directly by grepping the per-arm train logs (greps recorded in
`runs/EXP-23/analysis.log`). EXP-20 supplies the two reference constants:
A0 (fresh-clean@5, r=77) = 0.7415 and dense ceiling = 0.7536.

| arm | refresh mechanism | val@50 | Δ vs A1 floor | Δ vs A0 (0.7415) | Δ vs dense (0.7536) |
|---|---|---|---|---|---|
| A1 | none (floor) | 0.6914 | — | −0.0501 | −0.0622 |
| A2 | stale inject γ=1.0 | 0.6967 | +0.0053 | −0.0448 | −0.0569 |
| A3 | stale blend η=0.5 | 0.6861 | −0.0053 | −0.0554 | −0.0675 |
| A0 | fresh-clean@5 (EXP-20 ref) | 0.7415 | +0.0501 | — | −0.0121 |
| dense | ceiling (EXP-20 ref) | 0.7536 | — | +0.0121 | — |

**The A1-floor finding for the parent direction (load-bearing).** A1 (PowerSGD r=77,
NO refresh of any kind) = 0.6914, which is 0.0501 BELOW A0's fresh-clean 0.7415. So the
"stronger-PASS" variant the plan named — A1 landing within 0.01 of A0, i.e. PowerSGD
needing no refresh at all — does NOT hold for r=77: a FRESH full-gradient refresh
genuinely matters (it buys ~+0.05 / ~5 pts and closes most of the gap to the dense
ceiling). The falsification is precisely that a STALE re-anchor (inject OR blend at
delay_K=5) does NOT recover that +0.05 — both arms sit on the no-refresh floor. The
parent question resolves cleanly: refresh helps, but it has to be the fresh full
gradient, not a 5-step-stale orthogonal one.

## Resolved parameters (ground truth)
Source: `resolved_params__{A1,A2,A3}.txt` (extracted from each arm's train.log `set -x`
trace, NOT the plan). The headline refresh-axis + codec block, verbatim:

Refresh axis (the ONLY intended variable) — confirmed wired, no silent fallback:
- A1: `comm_eff.anchor.enabled=false`, `comm_eff.spectral.enabled=false`, `comm_eff.clean_cadence=0` (anchor.delay_K=20 / correction_mode=reweight present but INERT — circuit OFF; anchor_backwards=0, spectral_corrections=0)
- A2: `comm_eff.anchor.enabled=true`, `comm_eff.anchor.delay_K=5`, `comm_eff.anchor.cadence=5`, `comm_eff.spectral.enabled=true`, `comm_eff.spectral.correction_mode=inject`, `comm_eff.spectral.inject_gamma=1.0`, `comm_eff.spectral.cadence=5`, `comm_eff.spectral.ema_device=cpu`, `comm_eff.clean_cadence=0`
- A3: as A2 but `comm_eff.spectral.correction_mode=blend`, `comm_eff.spectral.blend_eta=0.5`

Codec block (FIXED, identical across A1/A2/A3 AND equal to EXP-20 r=77):
`comm_eff.compression_type=powersgd`, `powersgd.rank=77`, `powersgd.sync_basis=true`,
`powersgd.update_cadence=1`, `powersgd.warm_start=true`, `powersgd.compress_recompute=true`,
`powersgd.qr_dtype=fp32`, `powersgd.seed=0`, `powersgd.pp_size=8`, `powersgd.reortho_eps=1e-6`.

Training surface (FIXED, identical across arms): `lr=1e-6`, `train_batch_size=128`,
`ppo_mini_batch_size=64`, `ppo_micro_batch_size_per_gpu=1`, `rollout.n=8`,
`max_response_length=16384`, `total_training_steps=50`, `total_epochs=2`,
`use_kl_loss=False`, `use_kl_in_reward=False`, `entropy_coeff=0`, GSM8K.

**Divergence between plan and what actually ran:** NONE that affects the science. The
ONE declared exception — `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` = 36864 (A1)
vs 18432 (A2/A3) — ran exactly as the plan declared (the anchor's second backward OOM
guard). This is a memory-packing/microbatch-tiling knob, not an optimization
hyperparameter (same global batch 128, mini 64, per-gpu micro 1), so it is a recorded
controlled-variable exception, not a confound. (The `ppo_max_token_len_per_gpu=3000`
token that also appears in resolved_cmd is the launcher's earlier dynamic-bsz draft
value; the LAST-wins actor value — Hydra semantics — is 36864/18432, the one that ran.)

## Next lever (MANDATORY on falsification — per `## Falsification contingency`)
Source (BINDING): `runs/EXP-23/stale_gradient_research/STALE_GRADIENT_ALTERNATIVES.md`
(the `exp23-stale-grad` agent team report; §8 "A2 AND A3 both fail" branch is now the
binding next-lever readout, three independent lines — measured geometry, async-RL
theory, cited literature — converging).

**delay_K is NOT the lever.** The measured cos(G_powersgd, M_anchor) ≈ 0.001 (an order
of magnitude more orthogonal than EXP-21's mask cos≈0.5) means the failure is the
COMBINE, not the AGE. A smaller delay_K cannot fix orthogonality: PowerSGD r=77 discards
exactly the directions M lives in, so G ⊥ M would hold at K=1 as well. The async-RL
literature corroborates K=5 is deep inside the safe staleness band (AReaL η≤8 lossless;
2601.04537 shows the RLVR drift is linear with cos>0.9 across K-windows, so a 5-step-stale
full gradient is information-rich, not stale junk). The problem is that inject scales the
orthogonal complement down to ~0.03·‖G‖ (inert), and blend at η=0.5 only shrinks the step.

**Top recommended next lever (report Rank 1 + Rank 2):**
1. **Error-feedback on the PowerSGD residual** — the standing #21 top lever, and the
   named fix in arxiv 2602.03839 (PULSELoCo Alg. 2 FP32 EF buffer). Maintain a per-matrix
   FP32 buffer `e` accumulating `G_full − G_compressed`, apply `G_step = G_compressed +
   decay·e`, and flush/re-ground `e` against the stale full-rank anchor M every K steps.
   This compresses the RESIDUAL so the correction aligns with what G actually misses —
   the direct attack on orthogonality — instead of adding a scale-killed orthogonal impulse.
2. **Staleness-aware blend η ∝ 1/K (≈0.2 at K=5)** — a basis-/drift-aligned anchor used
   as a low-variance drift estimate folded in via a small-λ, spectrum-preserving convex
   blend (A-3PO α=1/d, SA-SGD lr∝1/τ, Gap-Aware ‖Δθ‖ all prescribe a staleness-shrinking
   weight; 2511.08567 argues small λ to preserve the spectrum). This is the "basis-aligned
   anchor" direction — fold M's REAL direction back in, masked to the live active
   subnetwork (Rank 3), rather than adding it orthogonally.

**Recommended follow-up issue (do NOT bolt onto EXP-23 — additive, flag-gated, OFF by
default per the Prime Directive):** the report seeds **EXP-24** — "Error-feedback PowerSGD
residual + staleness-aware blend (η∝1/K) — the named #21 top lever, gated and OFF by
default." Arms B1 = A1 floor (byte-parity), B2 = EF-only, B3 = EF + blend(η=1/K); gate on
B2 ≥ A1 floor and B3 ≥ B2. Carry an OFF-by-default `spectral.debug_dump` (to capture
per-target SVD spectra) and a `max_targets`/MLP/deep-layer COVERAGE axis as a separate
secondary lever — the 4/196 coverage gap means orthogonality should be re-measured on
MLP/deep layers before paying to correct them. The orchestrator should open EXP-24
carrying this recommendation.

## Notes
- Why this is a STOP and not a REVISE: the falsification predicate
  (`max(val@50(A2),val@50(A3)) <= val@50(A1)+0.02`) is met, and the plan's analyst
  predicate routes a met falsification line to STOP. The next lever is NOT a knob tweak
  on EXP-23's existing modes (a delay_K or γ/η sweep) — the cos≈0.001 orthogonality shows
  both additive modes are structurally inert/near-inert on the PowerSGD codec — it is a
  new mechanism (error-feedback codec) that requires its own code_change + plan + review.
  Hence STOP + follow-up issue, not REVISE.
- The integration WORKED: this run de-risks the central engineering question (the
  codec-agnostic grad-correction hook `_maybe_comm_eff_grad_correction` composes with the
  PowerSGD recompute path) — anchor fired, M_anchor populated, correction applied to the
  post-PowerSGD p.grad, codec health intact, world_size=4 held. That carries forward to
  EXP-24 (the EF path reuses the same hook).
- Budget: `check_budget.py` reports lifetime/month spend $100.66, monthly cap $1500,
  running_count=0 (box already torn down). No budget exhaustion; the STOP is on the
  science (falsification), not the cap. iterations cap = 3 (none consumed; this is the
  first run on the lineage).
- `analyze.py` emitted a stub `verdict=PASS` only because its `_is_m0_smoke` heuristic
  misfired (the handle JSON gpu_name is H200, not the "h100" it checks for) — that is a
  scaffold default, NOT a success-criteria evaluation. This verdict overrides it by
  applying the plan's predicate to the criteria. `diff_against_baseline.py` found no
  common jsonl keys (no per-arm train.jsonl); deltas computed by direct grep per the plan.
- The `ppo_max_token_len_per_gpu` exception (A1=36864 vs A2/A3=18432) did not bias the
  comparison: A2/A3 (the lower-token-len arms) are within noise of A1 (the higher one) on
  BOTH val and train-reward, so the OOM-guard tiling difference is not masking a real
  effect. Re-running A1 at 18432 to tighten it is unnecessary given the −0.035 to −0.045
  margin to the success bars (the gap is far larger than any tiling artifact).
- staleness-rotation cos(G_t,G_{t-1}) was correctly DEFERRED (the plan permitted deferral
  if not cheap+clean to add as a non-perturbing debug diag); EXP-24's debug_dump proposal
  is the place to capture it. The forward-looking research report is filed regardless and
  is binding here as the next-lever readout.
