# EXP-30 Verdict

**VERDICT: PASS**

- plan: `.claude/plans/30.md` · issue #30 · milestone M6 · code_change: true (branch `exp/30-valid-m-geometry@c56c13bbf`)
- written by: orchestrator applying the §Analyst predicate verbatim (analyst subagent stalled twice on infra
  watchdog — see PROGRESS 2026-06-13; every number below is recomputed from local artifacts, not inherited)
- cells run: `stepA_geometry_probe` (20 steps, probe-only) → gate eval → `B2_delayed_ef_valid_residual`
  (50 steps, production). **B1 never launched** (GATE-B1 CLOSED — pre-registered rule, not a failure).
- budget: ~9.2 of 24 GPU-hr (box i_40697545, 4×H200, 22:11→00:28 +10). Single REVISE re-roll NOT spent.

## Pre-registered gate outcomes (thresholds untouched; recompute: `runs/EXP-30/eval_stepA_gate.py` → `stepA_gate.md`)

| gate | rule (verbatim) | measured | outcome |
|---|---|---|---|
| GATE-B1 | med-over-fires m1 ≥ 0.10 AND m1 ≥ 2×m2 paired ≥80% of fires | med m1 = **0.0121**; med m2 = 0.0036; paired-frac = **0.57** | **CLOSED** |
| GATE-B2 | med ‖δ‖/‖G_comp_ring‖ ∈ [0.1, 1.5] AND loss-mismatch ≤ 0.02 nats | med = **1.0528**; max mismatch = **0.0103** | **OPEN** |

7 post-warmup fires (8 total), complete m1–m7 over all 196 targets, `metrics/stepA_fires.jsonl`.

## Success criteria — all unconditional boxes green

- [x] hard correctness invariants: CPU suite 230 green pre-provision; on-box probe gates: canary 16/16
      `match=True` (0 mismatch), mean step time 85.01 ≤ 86.3 s, bytes_ratio 0.0504 ∈ [0.0500, 0.0510],
      max mem 27.92 ≤ 30.77 GB; `anchor_grad_corrected=0` and `anchor_optimizer_steps=0` across all of Step A
- [x] Step A: 20 steps, 7 post-warmup fires, complete m1–m7, no NaN/Inf/OOM (done.flag rc=1 is
      post-checkpoint shutdown noise — DataLoader SIGKILL + WandB telemetry in `__del__`; training completed
      20/20 and checkpoints saved BEFORE the tracebacks, log lines 1407–1410 vs 1413+)
- [x] `stepA_gate.md` computed by the verbatim rules
- [x] m1–m7 table posted to #28 (the can-never-be-wasted deliverable)
- [x] B2 emission: **ZERO post-warmup emission** — no step in [10, 50] with `response_length/max` > 4000;
      no P1 (the only 16384 pin in the whole run is a single pre-injection rollout at step 2, 1/1024,
      non-consecutive, outside the window). Step-50 state: len/mean 203.9, len/max 784, clip_ratio 0.
- [x] B2 val: 0.0864@0 / 0.7036@25 / **0.7528@50** ⇒ best val@50 = **0.7528 > 0.7210** floor.
      **Parity aspiration 0.7414: REACHED** (dense ceiling 0.7536 missed by 0.0008).
- [x] bytes_ratio in band every cell (A: 0.0504; B2: 0.05052) — codec untouched, GOAL-3
- [x] controlled-variables: resolved_params diff A→B2 exactly {correction_mode none→delayed_ef, probe
      flag/posture, total_training_steps 20→50, experiment_name}; substrate byte-identical; merger hygiene
      verified (ef_decay/clip 0.0, signed_ema not live, β_anc=0, max_targets=−1, replay=true, snapshot=cpu)
- [x] spend 9.2 ≤ 24 GPU-hr

PASS clause satisfied: ≥1 launched gated cell (B2) emission-free with best val@50 > 0.7210.

## The sharp questions, answered

**Q1 — validity artifact (H_validity) or batch decorrelation (H_decorr)?** Neither, cleanly — and the
distinction matters. The valid generator-consistent gradient is *just as orthogonal* to the live compressed
gradient as the old invalid M was (m1 ≈ 0.012 vs m2 ≈ 0.004, both ≈ 0; m3 = 0.59–0.76 high — the two anchor
feeds largely agree with *each other*). So H_validity is falsified for the linear-blend route: generator
match does not open the blend geometry, and GATE-B1's closure retires blend-on-valid-M without spending a
training cell. **But the pure H_decorr signature did NOT appear either**: it requires m4 ≈ 0 at j ≥ 4, and
the fast gradient's lag-autocorrelation is decidedly nonzero (medians j1 0.086, j2 0.200, j3 0.115,
**j4 0.295, j5 0.169**). Cross-batch PG cosines are generically tiny (~0.01) regardless of estimator
validity, yet within-circuit self-correlation survives ≥5 ticks — so K-delayed signals are NOT uniformly
dead; only *cross-gradient blending* is. That is precisely the crack B2 drove through.

**Q2 — does a valid short-memory blend convert?** Not tested (gate closed) — and the gate existing is the
point: the EXP-23 lesson (tuning an inert primitive measures zero at any dose) was enforced for ~4 GPU-hr
instead of re-learned for ~10.

**Q3 — is the codec's weight-gradient error recoverable by a K-delayed exact residual? YES — the headline.**
m5 said the residual is identifiable (loss-mismatch ≤ 0.0103 ≈ EXP-29's relevance band, so δ is codec error,
not objective mismatch) and bounded (med ratio 1.0528). Cell B2 (`G_corr = G_comp + δ`, λ=1, β_anc=0)
converted: **0.7528 vs the 0.7210 best-realistic floor (+0.0318), past 0.7414 parity, 0.0008 under dense** —
emission-free, with the per-fire `delta_ratio` bounded and *declining* (1.37 → 1.03 over the run, no
monotone climb). δ lifecycle behaved exactly as designed: cold-fallback (= plain PowerSGD) ticks 1–9,
first valid pair at tick 10, refresh-at-fire/hold-between thereafter; `coldM_fallbacks=0` post-warmup;
20 anchor fires over 100 ticks.

## Findings (F1–F5)

- **F1 — The within-pair geometry discovery (decision-grade for #28).** At IDENTICAL (batch, θ),
  cos(δ, G_comp_ring) ≈ −0.92…−0.98 with ‖δ‖/‖G_comp_ring‖ ≈ 1.05, which algebraically forces
  **cos(G_anc_rep, G_comp_ring) ≈ 0 and ‖G_anc_rep‖ ≈ 0.33·‖G_comp_ring‖**: the compressed fast gradient is
  ~3× larger than the true gradient on the same data and points almost entirely *off* the true direction.
  The codec error in weight space is the dominant component of the fast gradient, not a perturbation — the
  weight-space confirmation of EXP-26's activation-proxy 0.318 (act-basis captures ~1/3 of update energy).
  B2 works because δ = G_anc_rep − G_comp_ring simultaneously *cancels the stale codec artifact* and
  *injects the true direction* — a blend can only add a near-orthogonal partner (B1 closed). Interpretation
  of how plain PowerSGD still trains decently (0.6437–0.7415 family) given this — Adam's per-coordinate
  normalization, cross-step averaging of the artifact — is open; flagged for the path-forward team.
- **F2 — m6 persistence risk (carrier-law clause, REQUIRED statement).** Cross-fire autocorrelation of
  M_rep: median ≈ **0.62** on real cross-pair fires (fires 3–8: 0.617, 0.586, 0.622, 0.628, 0.622, 0.751;
  fire-2's 0.9999 is the shared-tick-5-pair artifact). The valid anchor signal itself carries moderate-high
  persistence — β_anc=0 does NOT make the carrier memoryless, it only stops *compounding* it. Per the
  carrier law (ignition needs autocorrelation time ≫ cadence), persistence risk is partly intrinsic and the
  50-step emission-free result must be read as CENSORED (EXP-27's damped-EF ignited at step ~61). A
  small-β_anc EMA successor is NOT cleared by this run; m6 ≈ 0.62 is exactly the number a successor proposal
  must reckon with. **This is why the 100-step extension is the binding next measurement.**
- **F3 — m7: the team's RLVR-gradient premise, finally measured on a valid PG gradient.** Stable rank
  ‖G‖²_F/‖G‖²₂ ≈ **1.8–2.05** (vs ambient 1536) and top-1% coordinate mass ≈ **0.58–0.61**: replay
  gradients are *extremely* low-rank and heavily concentrated. A rank-77 codec has abundant capacity for a
  stable-rank-2 object — the failure is basis MISMATCH (act-basis Q does not contain the gradient
  directions), not capacity. (Constraint: EXP-26 Step C falsified update-energy/hybrid Q — any basis
  redesign must avoid that exact corner.)
- **F4 — Emission discriminator sharpened.** Every prior merger carried the spiral; B2 (short-memory, valid,
  residual-form) is the first correction-carrying cell with zero post-warmup emission AND val conversion.
  Combined with F2, the working hypothesis becomes: ignition requires persistent *exogenous* direction —
  the δ-residual is endogenous (it cancels the circuit's own artifact) and so does not pump length. 50-step
  censoring caveat stands until the extension runs.
- **F5 — Program integrity.** The geometry-gate-before-training contract did its job in both directions in
  one experiment: closed B1 cheaply (saved a doomed cell), opened B2 with quantified priors (m5), and the
  probe perturbed nothing (canary bitwise, step-time and bytes gates green).

## Bars (W&B read-only constants)

| reference | val@50 | B2 relative |
|---|---|---|
| dense ceiling `5e2jpho9` | 0.7536 | −0.0008 |
| A0 fresh-clean@5 `oquyeic3` | 0.7415 | +0.0113 |
| parity bar | 0.7414 | **reached** |
| ef r2 floor `tilwe80t` | 0.7210 | +0.0318 |
| signed_ema α0.5 `1wulaelw` | 0.7066 | +0.0462 |
| no-refresh floor (EXP-23 A1) | 0.6914 | +0.0614 |
| plain-on-substrate `u1v94opv` | 0.6437 | +0.1091 (single-knob read: the δ-correction is the only delta, modulo the replay knob postdating that run) |

## Disposition

- Verdict **PASS** → log-writer: LOG.md entry + draft PR `exp/30-valid-m-geometry` → `vast-ai-workload`.
- m1–m7 table + gate outcomes posted to #28 as priors (B2's conversion is *additive evidence* for #28's EF
  mechanism — K-delayed telescoping EF works in production; #28's current-step variant and plain@100 control
  remain valuable, the latter now ALSO as the no-carrier control for any 100-step B2 extension).
- Stability claims in this verdict are 50-step CENSORED statistics (EXP-27 lesson) — de-censoring via the
  operator-authorized 100-step B2 extension is the immediate next measurement.
- Step-A val (0.244@20) is a probe diagnostic, NOT a deliverable — never quote it as science.

---

## Addendum (2026-06-13): ext100 de-censoring outcome + B1 paper run

**ext100 (`exp30_B2_ext100`, W&B `b59ncque`) — operator-authorized 100-step extension of B2, identical
settings.** Outcome: **de-censored for seed 0 — no ignition through step 100.**

- Traversed the EXP-27 ignition band (51–66) cleanly; the registered forecast (no ignition through ~61)
  held; the P(ignite 61–100) ≈ 20–35% event did NOT occur.
- Vals: 0.7278@25 / **0.7536@50 (= the dense ceiling value)** / 0.7475@75 / 0.7400@100. Mild late decay
  past step 50; @75 still above the 0.7414 parity bar, @100 slightly below it but well above the 0.7210
  floor. No 100-step dense reference exists (dense never re-run) — plain@100 (#28 Cell B) is the right
  comparator for whether the decay is mechanism-specific or substrate/epoch-generic.
- Emission, reported honestly: steps 10–93 fully clean; two **isolated single-rollout cap-pins** at
  steps 94 and 99 (clip_ratio exactly 1/1024 each; len/mean flat 190–227 with zero slope; entropy
  1.3–1.8 healthy; max reverts to ≤1413 immediately). No P1 (non-consecutive), no P2/P3. Same benign
  signature as B2's pre-injection step-2 pin — consistent with the stochastic outlier-rollout base rate
  (3 single-rollout pins in ~150 observed steps across cells), NOT the ignition mechanism (EXP-27's
  signature was mean climbing 171→575 with entropy → 0.08).
- Health: max_mem 30.75 GB (ceiling 30.77 — note: 100-step headroom is ~zero; future ≥100-step cells
  should re-derive the ceiling), bytes_ratio 0.0505, 40 anchor fires, coldM 0, `delta_ratio` settled
  ≈ 1.001 (perfectly bounded, declining all run).
- Stability claims now censored at 100 steps instead of 50 for seed 0 (the honest restatement).

**B1 blend-on-valid-M paper run (operator-directed, post-verdict):** queued/in-flight on a fresh box —
completes the controlled combination-operator row (same valid anchor signal, blend η=0.3 vs residual
λ=1). GATE-B1's measured prediction: inert (~0.64–0.70, no conversion). Outcome will be appended here.

**Incident note:** box i_40697545 was auto-destroyed by the teardown hook (`no-heartbeat-30min`) minutes
after ext100 completed — the EXP-30-EXT ledger row's heartbeat path (`runs/EXP-30-EXT/metrics/`) never
existed because metrics sync to `runs/EXP-30/`. Zero science lost (artifacts pulled pre-teardown); the
initial B1 launch died in init and was re-provisioned. Lesson recorded: extension rows must reuse the
run-dir ID or materialize their heartbeat path.
