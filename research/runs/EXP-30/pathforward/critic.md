# EXP-30 PASS — Adversarial validity review (task #2)

**Author:** critic (team exp30-pathforward) · **Date:** 2026-06-13
**Scope:** attack what the EXP-30 numbers *can and cannot support* before the program builds on
them. Pre-registered thresholds are NOT relitigated (they were applied verbatim — settled). All
numbers below recomputed from local artifacts: `runs/EXP-30/{verdict.md, stepA_gate.md,
resolved_params_B2.txt}`, `metrics/stepA_fires{,_targets}.jsonl`,
`train_B2_delayed_ef_valid_residual.log`.

**Bottom line up front.** The PASS is *procedurally clean and the headline F1 geometry is the most
robust thing in the experiment* (verified per-target, not a median artifact — see T5). But the
**"parity reached" framing is statistically overclaimed** (T2, the highest-severity threat): a
single 1319-problem greedy val point cannot distinguish 0.7528 from the 0.7536 dense reference, and
cannot even cleanly separate B2 from the 0.7210 floor at conventional significance. The **stability
claim is censored exactly as the verdict admits** (T1) and the running ext100 only *partially*
de-censors it. Two confounds (T3 replay-knob, T6 comm-budget) are real but already disclosed; the
load-bearing one the program must not forget is that **B2's win is a single seed at a single step**.

What would settle the most for the least money: **one additional seed of B2 to 50 steps (~5 GPU-hr)**
buys more than the 100-step extension does, because the binding uncertainty is sampling/seed
variance, not the (separately important) censoring horizon.

---

## T1 — Censoring: 50-step emission-free is the exact statistic EXP-27 proved censored
**Severity: HIGH** (but fully disclosed; the verdict itself flags it as F2/F4 and the binding next
measurement).

**The threat.** The program's own carrier law says ignition needs autocorrelation-time ≫ cadence,
and EXP-27's damped-EF ignited at **step ~61** — *beyond* the 50-step horizon. B2 ran 50 steps. So
"zero post-warmup emission" is a censored survival statistic: it observed no ignition in [10,50] but
says nothing about [51,∞). The verdict is honest about this (F2, F4, Disposition all say
"50-step CENSORED").

**What the data actually shows (and it is reassuring, within the window).** I reconstructed the full
per-step emission trace for B2:

- `response_length/max` over steps 1–50: stays in **~550–1350** for the entire post-warmup window,
  with three transient excursions — **3580 @ step 25, 2104 @ step 41** (both *well under* the 4000
  threshold), and the often-cited **16384 @ step 2**.
- The 16384 pin is genuinely benign: `clip_ratio = 0.0009765625 = 1/1024` (one rollout of 1024),
  `entropy 5.63` (pre-warmup, model still near-init), `grad_norm 244.6` (warmup transient), and it is
  **non-consecutive** and **pre-injection** (delayed_ef's first valid δ fires at tick 10 = step 5).
  The verdict's reading is correct.
- `response_length/mean` *declines* 274 → ~204 over the run; entropy settles to a flat **~2.0–2.2**;
  grad_norm settles to **~2–5**. No spiral signature is present in-window.

**What ext100 CAN de-censor:** it pushes the observation horizon to step 100, which *covers* EXP-27's
~61 ignition point with ~40 steps of margin. If B2-ext100 reaches step 100 emission-free, the
specific EXP-27 failure mode (ignition in the 51–66 band) is **ruled out for this carrier**.

**What ext100 CANNOT do:** (a) it cannot prove stability beyond 100 steps — the carrier law gives no
finite horizon, so any fixed T is still censored at T; (b) it is the **same seed** continuing the same
trajectory, so it cannot tell you whether a *different* seed of B2 would have ignited earlier — seed
and censoring are confounded in a single run; (c) F2 measured M_rep cross-fire autocorrelation
**m6 ≈ 0.62** (medians on real cross-pair fires 3–8: 0.617/0.586/0.622/0.628/0.622/0.751). That is
*moderate-high* persistence in the valid anchor signal itself. β_anc=0 stops M_rep from *compounding*
across fires but does not make the per-fire signal memoryless — m6≈0.62 means a small-β_anc EMA
successor inherits a carrier with non-trivial autocorrelation time, and ext100 on the β_anc=0 cell
does **not** clear that successor.

**Cheapest resolving measurement:** ext100 is already running and is the right ~5 GPU-hr spend for the
EXP-27-band question. Do not over-read a clean ext100 as "stable" — read it as "EXP-27's specific
ignition window is cleared for seed 0." For the successor-EMA persistence question (m6), the cheap
measurement is a **2–3 fire micro-probe of M_rep autocorrelation at the candidate β_anc** before any
training cell, not a full run.

---

## T2 — Single seed, single 50-step val point: is "parity reached" overclaimed? (quantified)
**Severity: HIGH. This is the threat most likely to mislead the program.**

GSM8K val = **exactly 1319 problems** (confirmed: `data.val_files=.../gsm8k/test.parquet`, val
telemetry `num_turns` over 1319). Greedy `mean@1` ⇒ each problem is one Bernoulli trial ⇒ the val
accuracy has an exact binomial standard error.

| quantity | acc | binomial SE = √(p(1−p)/1319) | 95% CI (±1.96 SE) |
|---|---|---|---|
| B2 @50 | 0.7528 | **0.0119** | ±0.0233 |
| dense ceiling `5e2jpho9` | 0.7536 | 0.0119 | ±0.0233 |
| ef r2 floor `tilwe80t` | 0.7210 | 0.0123 | ±0.0242 |
| parity bar | 0.7414 | 0.0121 | ±0.0236 |

**The arithmetic that should temper the celebration:**

- **B2 vs dense:** diff = **−0.0008**. That is **~1.06 problems out of 1319** (one flipped answer =
  ±0.00076 acc). Two-sample z ≈ **−0.05**. The claim "0.0008 under dense" / "parity reached" is
  **inside the noise of a single problem** — B2 and dense are statistically indistinguishable at this
  N. This cuts both ways: it is *not* evidence B2 underperforms dense, but it is equally *not*
  evidence of parity. The honest statement is **"B2@50 is consistent with dense within sampling
  noise; one seed cannot establish parity."** The verdict's checkbox "Parity aspiration 0.7414:
  REACHED" leans on a point estimate whose CI (±0.023) spans from below the floor to above dense.
- **B2 vs the 0.7210 floor** (the *actual* pre-registered success bar): diff = +0.0318 ≈ **42
  problems**, two-sample z ≈ **+1.86**. This clears the pre-registered bar by the plan's verbatim rule
  (point estimate > 0.7210), but note it is only **~1.9σ** as a difference-of-proportions — i.e. it
  would be marginal at p<0.05 if you demanded statistical separation rather than a point-estimate
  threshold. The pre-registered rule asks only for the point estimate, so the PASS is correct *by the
  rule*; the criticism is that the *rule's discriminating power* at N=1319, single seed, is lower than
  the +0.0318 headline suggests.
- **Compounding factor — single val point per run.** B2 has exactly **3 val points** (step 0:
  0.0864, step 25: 0.7036, step 50: 0.7528). "best val@50" is the max of steps 25 and 50 = the step-50
  number. There is no within-run replication of the 0.7528, no step-45/55 neighbours to show it is not
  a lucky validation draw on top of a noisier underlying curve.

**Severity justification:** every downstream decision ("B2 converts," "the residual route beats the
merger routes," "M6 has its first converting cell") rests on this one number being meaningfully above
the floor and at parity with dense. The first is true-by-rule but ~1.9σ; the second is false as a
statistical claim.

**Cheapest resolving measurement:** **one additional B2 seed (seed 1) to 50 steps, ~5 GPU-hr.** Two
seeds let you (a) report a mean ± range instead of a point, (b) detect whether 0.7528 is reproducible
or a favourable draw, and (c) — combined with seed 0's ext100 — partially decouple seed from
censoring. This is a *higher-value* spend than further extending the single seed. If the program will
spend on only one thing, spend it here, not on ext150.

---

## T3 — The B2 − plain single-knob comparison confound (u1v94opv predates the replay knob)
**Severity: MED (disclosed in the verdict's own bars table).**

The Disposition bars read B2 as **+0.1091 over plain-on-substrate `u1v94opv` 0.6437**, captioned
"single-knob read: the δ-correction is the only delta, **modulo the replay knob postdating that
run**." That caveat is doing a lot of work and should not be dropped downstream.

**The confound.** `u1v94opv` is plain PowerSGD on the substrate, but it was run **before** EXP-29
introduced `replay_paired_batch` + `snapshot_device=cpu`. B2 runs *with* `replay_paired_batch=true`
(confirmed in `resolved_params_B2.txt`). So B2 vs u1v94opv differs by **two** things: (1) the
delayed_ef δ-correction, and (2) the entire replay machinery (paired-batch snapshotting, fire-aware
ring, anchor feed construction). The "δ-correction is the only delta" framing is **not strictly
true** — it is the only *correction-mode* delta, but the substrate underneath is not byte-identical to
u1v94opv.

In practice the replay machinery is *supposed* to be inert when no correction consumes it (it only
constructs the anchor feed; with correction_mode=none the optimizer never sees it — Step A asserted
`anchor_grad_corrected=0`). So the confound is *probably* small. But "probably inert" is exactly the
kind of assumption this program has been burned by (cf. the anchor-clone-on-random-weights bug). The
+0.1091 number should be quoted as **"B2 vs the pre-replay plain baseline"** with the two-delta caveat,
never as a clean isolation of the δ-correction's value.

**Cheapest resolving measurement:** the true single-knob control is **plain PowerSGD *on the EXP-29
replay substrate* with correction_mode=none, 50 steps** (i.e. Step A's exact config but run as a
production val arm rather than a probe). Issue #28's "plain@100 control" partially covers this; a
50-step replay-substrate-plain val arm (~5 GPU-hr) would make the B2−plain delta a genuine one-knob
read. Until then, treat +0.1091 as an upper bound on the correction's contribution.

---

## T4 — Step-A gate operationalization: were any judgment calls load-bearing?
**Severity: LOW. The gate logic is mechanical and the outcomes are not near-threshold in the
direction that matters.**

I re-read `eval_stepA_gate.py` against the plan's verbatim rules. The operationalization choices the
plan left to the executor (statistic only, thresholds fixed) were: "matrix-median" = median over the
196 per-matrix cosines at one fire; gate statistic = median-over-post-warmup-fires of that
matrix-median; paired clause = per-fire (m1 ≥ 2·m2) in ≥80% of fires. The script implements exactly
this. Two observations:

- **GATE-B1 CLOSED is robust to the choices.** med(m1) = **0.0121** vs the 0.10 threshold — it misses
  by **~8×**, not by a hair. No reasonable alternative aggregation (mean instead of median, different
  fire subset) rescues a distribution whose per-target median sits at ~0.01 and straddles zero (T5
  shows frac_neg 0.22–0.53). The paired-fraction (0.57 vs 0.80) is also a clear miss. B1 was never
  going to open; the operationalization is not load-bearing here.
- **GATE-B2 OPEN: the one mildly load-bearing call is the m5_ratio aggregation.** med-over-fires =
  **1.0528**, inside [0.1, 1.5]. The *median* sits comfortably mid-band, but the **per-fire upper tails
  are not trivial**: at tick 10, **43% of the 196 matrices have ‖δ‖/‖G‖ > 1.5** (max 4.596); the band
  excursion shrinks as the run settles (11% @ tick 15, ≤1% @ ticks 20–40). Because the gate uses the
  median-of-medians, the early-fire heavy upper tail is invisible to it. This did not change the
  OPEN/CLOSED outcome (the median is mid-band at every fire), but it means **"δ is bounded in [0.1,1.5]"
  is a statement about the median matrix, not about all matrices** — at the first valid fire nearly
  half the matrices carry a δ larger than 1.5× their fast gradient. The loss-mismatch clause (max
  0.0103 ≤ 0.02) is a clean pass with margin.

**Cheapest resolving measurement:** none needed for the verdict (the call wasn't outcome-changing).
But the path-forward team should know that **δ's magnitude is heavy-tailed across matrices early in
training**; a λ=1 injection at the first few fires is adding a δ that exceeds the fast gradient on
~40% of matrices. That the run survived it is informative for F1, but a successor that injects earlier
or with less warmup should expect a rougher first few steps.

---

## T5 — The m5_cos ≈ −0.95 reading: is it a real geometry or a measurement artifact?
**Severity: LOW for "is it real" (it is strongly real); MED for "is it being interpreted correctly."**

This is the headline F1 (cos(δ, G_comp_ring) ≈ −0.92…−0.98, ‖δ‖/‖G‖ ≈ 1.05 ⇒ the codec error
*dominates* the fast gradient). I attacked the four alternative explanations the task names, using the
per-target sidecar (`stepA_fires_targets.jsonl`, 196 matrices/fire):

- **"Per-matrix medians hide a sign-mixed distribution" — FALSIFIED.** The cos is per-target uniform,
  not a median artifact. At the converged fires the fraction of 196 matrices with cos > −0.5 is:
  tick 20 **0.01**, tick 25 **0.00**, tick 30 **0.00**, tick 35 **0.00**, tick 40 **0.00**; min cos
  reaches **−1.000**. There is essentially no positive-cos mass — the −0.95 is the whole distribution,
  not its midpoint. (Only the *first* valid fire, tick 10, is softer: 26% of matrices have cos > −0.5,
  med −0.722 — consistent with the codec still cold/warming.) This is the *opposite* of the m1
  distribution (T5 below), which genuinely straddles zero — so the probe is clearly capable of
  resolving sign-mixed vs sign-coherent, and δ is sign-coherent.
- **"DP-reduction / normalization mismatch despite the unit test" — unlikely.** The plan added a hard
  invariant (delta scale-consistency, #25 mean-vs-sum trap) with a synthetic multi-rank unit test, and
  ‖δ‖/‖G‖ ≈ 1.0 is *exactly* what you'd see if the two gradients are correctly co-normalized (a
  world-size mismatch would show a ratio off by a factor of 4, not 1.05). The ratio sitting at ~1.0 is
  itself evidence the normalization matches. **However** — a residual cos ≈ −1 with ratio ≈ 1 is also
  the algebraic signature of **G_anc_rep ≈ 0** (if δ = G_anc_rep − G_comp and G_anc_rep→0 then
  δ→−G_comp, giving cos exactly −1 and ratio exactly 1). The verdict's F1 acknowledges this: it derives
  ‖G_anc_rep‖ ≈ 0.33·‖G_comp‖ and cos(G_anc_rep, G_comp) ≈ 0. So the geometry is "G_anc_rep is small
  AND nearly orthogonal to G_comp," which is *consistent* but means **the −0.95 cos is mostly telling
  you G_comp dominates δ, not that G_anc_rep carries rich structure.** The interpretation "δ cancels
  the codec artifact and injects the true direction" is plausible but rests on G_anc_rep (the
  ~0.33-magnitude, ~orthogonal piece) actually being the *true* gradient — which is asserted from the
  loss-mismatch ≤ 0.01 nats relevance probe, not directly verified as "this is the dense direction."
- **"bf16 staging" — not supported by evidence.** QR is fp32 (`qr_dtype=fp32`), the dumps are fp32
  (`dump_dtype=fp32`), and powersgd reconstruction_rel_error logs at ~0.02–0.05 (sane). No bf16
  rounding signature.

**The interpretive caveat worth carrying forward:** F1 is robust *as geometry* (the codec error is
the dominant component of the compressed fast gradient — a strong, decision-grade fact). But "B2 works
*because* δ injects the true direction" is one step beyond what's measured. An equally consistent
reading is "δ ≈ −G_comp + (small valid correction), so G_corr = G_comp + δ ≈ the small valid
correction" — i.e. B2 may be working by **largely cancelling the (biased, dominant) compressed fast
gradient and stepping on the small residual valid signal**, which is a *different* mechanism story
than "telescoping EF recovers full information" and has different implications for how it scales
(it would predict the effective learning signal is small/slow, which is consistent with the modest
val and the need for the full 50 steps). This is exactly the F1 question flagged for mechanist (task
#1); I raise it here only to mark that **the cos number does not by itself adjudicate which mechanism
is operating.**

**Cheapest resolving measurement:** the mechanist's task. If they want a number: at one fire, log
cos(G_anc_rep, g_dense_fresh) on a handful of stratified matrices (a tier-3 extra-backward probe, so
only on a dedicated ≤10-step diagnostic, never a production arm) to confirm G_anc_rep points the dense
way rather than just "away from G_comp."

### T5b — m1 (GATE-B1) is genuinely near-zero AND sign-mixed (the gate closure is real)
For symmetry I checked the *other* side. m1 per-target: med +0.012, **frac_neg 0.22–0.53**, min/max
spanning roughly [−0.5, +0.5] at every fire. So GATE-B1's closure is not a median artifact either —
the valid anchor gradient really is ~orthogonal-and-sign-incoherent with the live compressed
gradient. H_validity is cleanly falsified for the *blend* route, as the verdict says. Good.

---

## T6 — Does B2's δ-injection violate the comm budget honestly? (GOAL-3 / anchor-side traffic)
**Severity: MED. The bytes_ratio is honest *as defined*, but the definition omits the cost δ relies
on, and downstream "0.0505 = 20× savings" claims must keep the anchor caveat attached.**

`comm/bytes_ratio` in B2 ranges **0.05037–0.05056** (band [0.0500, 0.0510] — in-band, GOAL-3 box green).
But what that ratio measures is **only the PowerSGD-compressed fast-circuit boundary traffic**
(`bytes_compressed / bytes_dense_equiv`, the rank-77 y-only logical PP bytes). It does **not** count:

- the **anchor's full-gradient transfer**: the anchor maintains a `delay_K=5`-stale, full-coverage,
  DP-reduced gradient M, refreshed every cadence-5 ticks. That is full-H traffic, just low-frequency.
  The FIXED_CONTROL_SURFACE and GOAL.md both treat this as real inter-stage cost; the plan's GOAL
  alignment map explicitly says M/δ traffic is "anchor-side, cadence-amortized, **counted in the honest
  caveat as before**." So the honest savings number is *not* 0.0505 — it is 0.0505 for the fast circuit
  plus an amortized anchor term (the standing estimate elsewhere in the program is "amortized comm ~4×,
  not 20×" — see the clean-step-realism memory).
- **B2 adds no *new* traffic beyond the substrate's anchor.** This is the one genuinely reassuring
  part: δ = G_anc_rep − G_comp_ring is built entirely from quantities the anchor circuit *already*
  computes and transfers (the replay fire's gradient + the ring's stored compressed gradient). The
  injection is a local arithmetic combination at the consumer, not a new collective. So **B2's comm
  budget is identical to the substrate's** — whatever the honest anchor-inclusive number is for plain
  PowerSGD-on-anchor, it is the same for B2. The δ-correction is "free" in comm terms *given* the
  anchor is already paid for.

**The threat to honesty is therefore not B2-specific** — it is the program-wide habit of quoting
0.0505 (the fast-circuit ratio) as "the savings" while the anchor's full-gradient traffic lives only
in a prose caveat. For a *verdict* that's acceptable (the box is defined as the fast-circuit ratio and
it's in-band). For the **path-forward narrative and any external comms**, GOAL-3 says savings must be
"reported as a concrete number" — and the concrete number that includes the mandatory anchor is **not
20×**. The program should commit to one anchor-inclusive savings figure rather than letting 0.0505
travel unqualified.

**Cheapest resolving measurement:** no GPU needed — it's an accounting decision. Compute the
anchor-inclusive effective ratio analytically: fast-circuit 0.0505 + (anchor full-H bytes per refresh
× refresh frequency / dense-equiv per step). The byte counters to do this are already logged
(`logical_pp_bytes_powersgd_y_only`, `bytes_dense_equiv`, anchor cadence). Publish that single number
as "the" savings and retire the bare 0.0505 from comparison tables.

---

## Cross-cutting: things that are clean (so the program doesn't over-correct)

To keep this review honest about what survives scrutiny:

- **Merger hygiene held.** `resolved_params_B2.txt` confirms the only active correction is
  delayed_ef (λ=1.0); the inherited `signed_ema_alpha=0.5`, `blend_eta=0.3`, `inject_gamma=1.0`,
  `ef_clip/ef_decay=0.0` are present-but-inert leftover defaults, and `anchor_grad_corrected`/
  `anchor_optimizer_steps`/`merger_coldM_fallbacks` post-warmup behave as designed. The
  EXP-29-lineage merger-hygiene trap (ef_powersgd/signed_ema leaking in) did **not** bite.
- **No-KL/no-entropy surface held.** I specifically checked the KL knobs because the raw command
  echoes `kl_loss_coef=0.001` — but the GROUND-TRUTH resolved params (last-write-wins) are
  `use_kl_loss=False`, `entropy_coeff=0`, `use_kl_in_reward=False`. The 0.001 coef is an inert
  default. Vanilla GRPO is intact; this is **not** a controlled-variable violation. (Flagging that I
  looked, so nobody re-raises it.)
- **The controlled-variable diff A→B2 is exactly {correction_mode, experiment_name,
  total_training_steps}** — verified by diff of the two resolved_params files. Substrate
  byte-identical as claimed.
- **In-window dynamics are genuinely healthy**, not just non-emitting: declining response length,
  flat entropy ~2.0, grad_norm ~2–5. If a spiral were latent it is not yet expressing at step 50.
- **delta_ratio declines monotonically-ish** (1.369 → 1.015 over the fires), matching the verdict's
  "bounded and declining, no monotone climb" — confirmed from the per-fire `delta_ratio_median` log
  scalar, independent of the Step-A probe.

---

## Severity-ranked summary + cheapest resolving measurement

| # | threat | severity | evidence now | cheapest resolution |
|---|---|---|---|---|
| T2 | "parity reached" overclaimed; single seed/single val point | **HIGH** | B2−dense = −0.0008 ≈ 1 problem, z≈−0.05; B2−floor z≈+1.86; SE=0.0119 at N=1319 | **+1 seed of B2@50 (~5 GPU-hr)** — higher value than any extension |
| T1 | 50-step emission-free is censored (EXP-27 @~61) | **HIGH** | clean in-window trace; m6≈0.62 persistence intrinsic | ext100 (running) clears the EXP-27 band for seed 0 only; m6 micro-probe for EMA successors |
| T3 | B2−plain is a 2-delta read (replay knob postdates u1v94opv) | MED | +0.1091 carries the verdict's own "modulo replay" caveat | plain-on-replay-substrate@50 val arm (~5 GPU-hr); #28 plain@100 partially covers |
| T6 | bytes_ratio 0.0505 omits anchor full-grad traffic (GOAL-3) | MED | B2 adds no new traffic; but anchor cost is prose-only | analytic anchor-inclusive savings number (no GPU); retire bare 0.0505 |
| T5 | m5_cos≈−0.95 interpretation (mechanism, not artifact) | LOW(real)/MED(interp) | per-target frac(cos>−0.5)≈0 at converged fires — robust; but cos≈−1+ratio≈1 also = G_anc_rep small | tier-3 cos(G_anc_rep, g_dense) probe on a ≤10-step diagnostic (mechanist) |
| T4 | Step-A gate operationalization judgment calls | LOW | B1 misses 0.10 by ~8×; B2 median mid-band; only early m5 upper-tail hidden | none for the verdict; note heavy-tailed early δ for successors |

**One-line verdict on the verdict:** the PASS is correct *by the pre-registered rules*, the F1
geometry is the real prize and survives the hardest probe I could throw at it (per-target sign
coherence), but the program should internalize that **B2's val win is one seed at one step, parity
with dense is statistically unestablished (not refuted), and the savings headline owes an
anchor-inclusive number.** Spend the next ~5 GPU-hr on a second B2 seed, not on a longer single one.
