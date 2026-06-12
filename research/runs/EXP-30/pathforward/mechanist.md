# EXP-30 — Mechanist deep-dive (task #1, F1 priority)

Author: mechanist (team exp30-pathforward) · 2026-06-13
Sources: `runs/EXP-30/verdict.md`, `stepA_gate.md`, `.claude/plans/30.md`,
`metrics/stepA_fires.jsonl` (per-fire matrix-medians), `metrics/stepA_fires_targets.jsonl`
(per-target sidecar: 196 matrices × {m1,m2,m3,m5_ratio,m5_cos,m6,m7,g_comp_norm,rep_norm,old_norm}),
`train_B2_delayed_ef_valid_residual.log` (50-step trajectory), branch `exp/30-valid-m-geometry`
(`verl/workers/comm_eff/anchor.py` for the exact metric definitions).

Every number below is recomputed from the local artifacts. Where I derive algebra I show it; where I
forecast I label it speculation. I do not relitigate any gate threshold.

## 0. Definitions I had to pin before trusting F1 (load-bearing)

The sidecar nearly tricked me, so this matters for everyone downstream. From `anchor.py`:

- `g_comp_norm` (sidecar) = `‖G_comp(t)‖`, the **live** fast gradient at tick `t`. It is the denominator
  for m1/m2/m4 only.
- m5 is computed against a **different** vector: `δ(t) = G_anc_rep(t) − G_comp_ring(t−K)`, and
  `m5_ratio = ‖δ‖/‖G_comp_ring(t−K)‖`, `m5_cos = cos(δ, G_comp_ring(t−K))`. The ring norm
  `‖G_comp_ring(t−K)‖` is **not** stored per-target in the sidecar.
- `rep_norm = ‖G_anc_rep(t)‖`, `old_norm = ‖G_anc_old(t)‖`.

Consequence: you **cannot** reconstruct F1's geometry by combining `m5_cos`/`m5_ratio` with the
sidecar's `g_comp_norm` — that mixes the live grad with the ring grad and gives a per-matrix
reconstruction error of 0.3–1.0 (I checked; it is not a bug, it is two different vectors). The correct
F1 derivation uses **only** the self-consistent `(m5_ratio, m5_cos)` pair, both defined on the ring.
This is what I do in §1.

## 1. F1 — the within-pair geometry: confirmed, and sharper than the verdict stated

### 1a. The algebra (explicit)

Let `C = G_comp_ring(t−K)` (compressed fast grad), `A = G_anc_rep(t)` (true/valid PG grad on the
identical (batch, θ)), `δ = A − C`. Step A measures `r = ‖δ‖/‖C‖` (m5_ratio) and `c = cos(δ, C)`
(m5_cos). Then, with no further assumptions:

```
δ·C   = c · ‖δ‖ · ‖C‖ = c·r·‖C‖²
A·C   = (C+δ)·C = ‖C‖²(1 + c·r)
‖A‖²  = ‖C‖² + ‖δ‖² + 2 δ·C = ‖C‖²(1 + r² + 2cr)
⇒  ‖A‖/‖C‖     = √(1 + r² + 2cr)
⇒  cos(A,C)    = (1 + c·r) / √(1 + r² + 2cr)
```

### 1b. The numbers (recomputed per-matrix, then pooled over 196 × 7 post-warmup fires)

Per-fire matrix medians of the derived quantities:

| fire | step | med c=cos(δ,C) | med r=‖δ‖/‖C‖ | med ‖A‖/‖C‖ | med cos(A,C) | q10..q90 cos(A,C) |
|---|---|---|---|---|---|---|
| 2 | 5 | −0.722 | 1.385 | 0.964 | −0.008 | −0.052 .. +0.034 |
| 3 | 8 | −0.880 | 1.137 | 0.542 | +0.004 | −0.040 .. +0.104 |
| 4 | 10 | −0.962 | 1.037 | 0.288 | +0.016 | −0.057 .. +0.109 |
| 5 | 13 | −0.981 | 1.018 | 0.199 | +0.006 | −0.055 .. +0.084 |
| 6 | 15 | −0.950 | 1.053 | 0.325 | +0.015 | −0.037 .. +0.098 |
| 7 | 18 | −0.924 | 1.074 | 0.415 | +0.028 | −0.024 .. +0.103 |
| 8 | 20 | −0.973 | 1.031 | 0.237 | +0.001 | −0.100 .. +0.109 |

**Pooled over all 7 post-warmup fires (1372 matrix observations):** median `cos(A, C) = +0.007`,
q10..q90 = −0.055 .. +0.098, and only **6.9%** of matrices have |cos(A,C)| > 0.2. The true gradient is
statistically orthogonal to the compressed ring gradient — not "small positive," genuinely ⊥.

**Pooled over the settled fires 4–8** (after the δ_ratio settles to ~1.03): median
`‖A‖/‖C‖ = 0.29` (IQR 0.16–0.49). So the verdict's headline "‖G_anc_rep‖ ≈ 0.33·‖G_comp_ring‖"
is correct in direction and within rounding of the settled median — I would state it as **the true
gradient is ~3.4× smaller in norm than the compressed gradient on the same data** (0.29×), with the
0.33 figure being the looser early-fire estimate.

### 1c. What F1 means mechanically

The compressed fast gradient `C` decomposes, on identical data, into a vector `A` that is the true
gradient and a vector `(C − A) = −δ` that is the codec error. The measurement says:

- `−δ` (the codec error) **dominates** `C`: ‖δ‖ ≈ ‖C‖ (r ≈ 1.03–1.05 settled), while ‖A‖ ≈ 0.29‖C‖.
  Squaring: of `‖C‖²` energy, the true direction carries `‖A‖²/‖C‖² ≈ 0.084` and the codec error
  carries the rest, with the cross-term `2A·(−δ)` accounting for the geometry. The true signal is
  **~8% of the compressed gradient's energy** — the weight-space confirmation of EXP-26's
  activation-proxy number (act-basis Q captured 0.318 of update energy; here, in weight space on the
  *valid* gradient, the recoverable-true fraction is even smaller because the act-basis residual that
  EXP-26 left off-principal lands here as pure error).
- Because cos(A, C) ≈ 0, the codec error is **not** a rescaling or a same-direction perturbation of the
  true gradient — it is a near-orthogonal injection. This is exactly why **the residual converts and a
  blend cannot** (§3).

This is the decision-grade fact for #28: PowerSGD on the PP boundary, at r=77 in the act-basis, does
not return a noisy version of the true gradient — it returns a vector whose dominant component is
orthogonal codec error, with the true gradient buried at ~8% energy and statistically ⊥ to the output.

### 1d. Reconciling with "plain PowerSGD still trains decently" (the verdict's open question)

Plain PowerSGD on this substrate scores 0.6437 (`u1v94opv`), and psgd-clean reaches 0.7415. If C is
~92% orthogonal codec error, why does stepping on C train at all? Four mechanisms, ranked by how much
of the gap I think each explains. **All four are quantitative hypotheses, not yet isolated — I flag
the discriminating experiment for each.**

1. **The 8% true component is a *consistent* descent direction; the 92% error is high-variance and
   averages toward its own fixed point, not toward damage.** A·C > 0 in expectation is not required for
   training — what is required is that the *expected* update has positive overlap with the true
   gradient. Adam accumulates `C` across steps; if the codec error `−δ` has lower step-to-step
   autocorrelation than the true component `A`, the error partially cancels in the running first moment
   while the true component reinforces. **This is the most likely dominant mechanism and it is directly
   measurable but was NOT measured**: m4 gives `cos(C(t), C(t−j))` for the *whole* compressed grad
   (median 0.09–0.30, see §2), but we never separated the autocorrelation of the `A` part from the
   `−δ` part. *Discriminator:* a probe logging `cos(δ(t), δ(t−j))` vs `cos(A(t), A(t−j))` — if
   `A` autocorrelates substantially more than `δ`, this mechanism is confirmed.

2. **Adam's per-coordinate normalization rescales away the error's magnitude advantage.** Adam divides
   each coordinate by `√v̂`. The codec error is rank-77-structured (it lives in the *complement* of the
   act-basis subspace, by construction) while m7 says the true gradient is stable-rank ~1.9 and
   top-1%-mass ~0.60 (§4) — extremely concentrated. On the handful of coordinates where the true
   gradient is large, `A` is large *and* persistent, so `v̂` on those coordinates is dominated by the
   persistent true signal, and the sign of the Adam step there tracks `sign(A)`, not `sign(C)`, even
   though ‖A‖ ≪ ‖C‖ globally. The orthogonality is a *global* (all-1536²-coordinate) statistic; Adam
   acts per-coordinate, so a globally-orthogonal C can still produce a per-coordinate step aligned with
   A on the concentrated coordinates that actually matter. **This is the cleanest reconciliation of
   "global cos ≈ 0" with "training works."** *Discriminator:* compute cos(A, C) restricted to the
   top-1% coordinates by |A| — I predict it is materially > 0 there even though the global value is 0.
   (The artifact does not carry coordinate-resolved cosines, so this is a probe for a follow-on, not a
   recompute.)

3. **Cross-step artifact averaging at the optimizer.** Independently of Adam normalization, the
   *first* moment itself averages C over the optimizer's effective window. If the codec error rotates
   (Q is refreshed; the act-basis tracks activations which drift), `−δ` is non-stationary in direction
   while `A` tracks the loss landscape and is more stationary near a descent path. The moment estimate
   therefore concentrates on the stationary `A` component. This overlaps with (1) but is about the
   optimizer's accumulation rather than the raw signal's autocorrelation.

4. **Per-matrix vs global-cos distinction.** F1's cos(A,C) ≈ 0 is a *median over 196 matrices*. Some
   matrices may carry positive overlap. From §1b the per-fire q90 of cos(A,C) is +0.08..+0.11, and
   6.9% of matrices exceed |cos|>0.2 — so a minority of matrices do retain alignment, and if those are
   the high-leverage matrices (late-layer, output-adjacent) the decent-training is carried by them.
   This is the weakest of the four (the effect is small) but it is real and free to check against the
   per-layer breakdown in the sidecar.

My ranked belief: **(2) Adam per-coordinate normalization on the concentrated true coordinates is the
primary reason plain PowerSGD trains, with (1) running-moment cancellation of the rotating error
second.** Both are consistent with F1 and with m7. The path-forward team should treat the
top-1%-coordinate cos(A,C) probe as the single highest-value cheap measurement to close this — it is
also the measurement that would tell us whether a *cheaper* codec (a coordinate-sparse residual instead
of a full delta) could capture most of the conversion.

## 2. F1/Q1 — why m1 ≈ 0.012 cross-batch but m4 j4 ≈ 0.295 self-lag: what survives K-delay

This is the cleanest discriminator in the experiment and the verdict states it correctly; I add the
distributional shape and the mechanism.

- **m1** `cos(G_comp(t), G_anc_rep(t))` — same tick, **different batches** (t vs t−K data), valid
  estimator both sides. Pooled median +0.012, q10..q90 ≈ −0.06..+0.10, 13–19% of matrices |m1|>0.1.
  This is the *cross-batch policy-gradient cosine* and it is generically ~0: two PG estimates on
  disjoint GSM8K minibatches at nearby θ are nearly uncorrelated in weight space. **m1 ≈ m2 ≈ 0**
  (m2 median 0.004) is the falsification of H_validity for the blend route: making the estimator valid
  (generator-matched) does **not** raise the cross-batch cosine, because the limiting factor is *batch
  decorrelation*, not estimator validity. Generator match buys you nothing for a blend.

- **m4** `cos(G_comp(t), G_comp(t−j))` — **same circuit, same estimator, lagged batches.** Medians
  j1 0.086, j2 0.200, j3 0.115, **j4 0.295**, j5 0.169. Decidedly nonzero out to j=5 (and noisy in j —
  the ordering j4>j2>j5>j3>j1 is not monotone, consistent with small-sample matrix-median noise over
  7 fires, but the *level* ~0.1–0.3 is robust).

The reconciliation: **cross-batch decorrelation (m1) and within-circuit lag-autocorrelation (m4) are
different objects.** m1 compares two *independently sampled* gradients; m4 compares the same compressed
circuit's output to its own recent past. The compressed circuit's output `C(t)` is dominated by the
codec error `−δ` (F1), and **the codec error is autocorrelated** — Q (the act-basis) is refreshed only
every `update_cadence=1` tick but warm-started, and the activation statistics that define the basis
drift slowly, so the projection residual `−δ(t)` shares structure with `−δ(t−j)` for several ticks.
That shared codec-error structure is what survives K-delay and shows up as m4 ~ 0.1–0.3, **not** any
shared *true-gradient* signal (which would also appear in m1, and does not).

**Why this is the crack B2 drove through:** H_decorr in its pure form predicted m4 ≈ 0 at j≥4, which
would have killed *all* K-delayed operators including the residual. m4 is *not* zero at j≥4, so a
K-delayed signal is not uniformly dead — but what survives is the *codec-error* autocorrelation, not a
true-gradient correlation. A residual `δ = A − C` is precisely the operator that *uses* the surviving
codec-error structure (to cancel it) while *not* depending on cross-batch true-gradient correlation
(which is absent). A blend, by contrast, would need the *true-gradient* cross-correlation (m1) to be
nonzero to add a useful partner — and it is zero. **m4-survives + m1-dead is the exact signature that
selects residual-over-blend**, and it is the mechanistic content of GATE-B1-closed / GATE-B2-open.

## 3. F1/Q3 — why the residual converts while blends are inert: cancellation vs addition

The two operators (plan §combination operator):

```
Blend  (B1):   G_corr = (1−η)·G_comp + η·(‖G_comp‖/‖M_rep‖)·M_rep
Residual (B2): G_corr = G_comp + λ·δ,   δ = G_anc_rep − G_comp_ring(t−K),   λ=1
```

With λ=1 and the ring pair, the residual telescopes **exactly**:

```
G_corr(t) = G_comp(t) + [G_anc_rep(t) − G_comp_ring(t−K)]
```

When the fast circuit is near-stationary over K ticks (`G_comp(t) ≈ G_comp_ring(t−K)`, which m4's
nonzero lag-autocorrelation makes *approximately* true — that is the very assumption m4 licenses), the
two compressed terms cancel and `G_corr(t) ≈ G_anc_rep(t)` — the true gradient. The residual is an
**exact codec-error subtraction**: it removes the ~92%-energy orthogonal artifact (F1) and leaves the
true direction. This is K-delayed error feedback in the telescoping-EF sense (plan §theory):
`G_corr` = compressed step + the dropped residual, re-injected one period late.

The blend cannot do this. A blend *adds* a scaled `M_rep` to `G_comp`. From F1, `M_rep = G_anc_rep` is
**orthogonal** to `G_comp` (cos ≈ 0) and ~3.4× smaller. Adding an orthogonal, smaller vector to
`G_comp` does not cancel the codec error — it leaves the full `−δ` artifact in place and tilts the step
by a small orthogonal nudge. The resulting `‖G_corr‖² = ‖G_comp‖²[(1−η)² + η²·1 + 2η(1−η)·c]` with
`c = cos(G_comp, M_rep) ≈ 0` reduces to `‖G_comp‖²[(1−η)² + η²·(‖M_rep‖/‖G_comp‖)²·(...)]` — i.e. you
down-weight the (mostly-error) compressed step by (1−η) and add back η× of an orthogonal partner that
is itself only ~0.29 the norm. At no η does this *subtract* the artifact. That is why a blend is inert
at any dose — the EXP-23 lesson — and why GATE-B1 closing it without a training cell was correct.

**The compact statement:** the codec error is near-orthogonal to the true gradient and dominates the
compressed gradient (F1). Orthogonal contamination of that magnitude can only be removed by
**subtraction of the same artifact** (residual), never by **addition of a near-orthogonal partner**
(blend). The cancellation algebra is the mechanism; F1's cos ≈ 0 is what makes addition impossible and
subtraction necessary.

A subtlety worth flagging for the path-forward team: the cancellation is only *approximate* because
`G_comp(t) ≠ G_comp_ring(t−K)` exactly (m4 ≈ 0.1–0.3, not 1.0). So B2 injects
`G_anc_rep(t) + [G_comp(t) − G_comp_ring(t−K)]` — the true gradient plus a *fast-circuit drift term*
over K ticks. That drift term is small (the circuit is autocorrelated) and, importantly, it is
**endogenous** to the fast circuit, which §5 argues is why it does not pump length. The cleaner the
fast circuit's K-step stationarity, the closer B2 is to pure true-gradient descent — which suggests
shrinking K (delay_K) is a lever the path-forward team should consider (cheaper staleness, tighter
cancellation), bounded by the cost of more frequent anchor fires.

## 4. F3/m7 — where the gradient actually lives, vs the act-basis Q

m7 on the **valid** PG gradient `G_anc_rep`, recomputed per-matrix:

- **Stable rank** `‖G‖²_F/‖G‖²₂`: per-fire medians 1.77–2.05 (q10 ~1.3, q90 ~3.0), against ambient
  dimension 1536. The valid RLVR gradient is **rank ~2** in the stable-rank sense — its energy is
  concentrated in ~2 effective singular directions.
- **Top-1% coordinate energy mass**: per-fire medians 0.58–0.61 (q10 ~0.34, q90 ~0.80). ~60% of the
  squared gradient lives in the largest-magnitude 1% of coordinates. Heavily concentrated, consistent
  with the (never-before-measured-on-a-valid-grad) "RLVR gradients are sparse/low-rank" premise.

The implication for the codec is the F3 headline and it is **not** a capacity problem: a rank-77 codec
has enormous spare capacity for a stable-rank-~2 object (77 ≫ 2). The reason the act-basis codec
nonetheless leaves ~92% orthogonal error (F1) is **basis mismatch**: the act-basis Q is built from
*activation* second moments at the boundary, and the directions that diagonalize activation covariance
are **not** the ~2 directions where the weight-space PG gradient concentrates. The codec faithfully
captures the rank-77 act-principal subspace and discards everything outside it — and the true gradient's
~2 dominant directions largely lie *outside* that subspace (that is the geometric content of "off-principal
0.68" from EXP-26, now confirmed in weight space as cos(A,C) ≈ 0).

**Hard constraint for any basis redesign (from the program):** EXP-26 Step C already falsified the
update-energy / hybrid-Q corner (`hybrid_act_cols`/`hybrid_grad_cols`, here both −1 = off). So "just
build Q from the gradient instead of the activations" is *the exact corner that was tried and failed*.
The live finding is narrower and more useful: the *target* subspace is only stable-rank ~2, so the
question for the path-forward team is not "a better full-rank Q" but "**can a stable-rank-2 residual
be transmitted in far fewer than r=77 columns**" — i.e. the residual δ is itself a low-rank object
(it inherits A's concentration), so the K-delayed residual could plausibly be compressed *again* to a
handful of columns without losing the conversion. That is a GOAL-3 (bytes) lever that F3 + F1 jointly
open and that nothing in the program has yet falsified.

## 5. F2/m6 — carrier persistence, ignition risk at 100 steps, and a falsifiable prediction

### 5a. The number

m6 = `cos(M_rep(t), M_rep(t−5 ticks))` across fires. Excluding fire 2 (shares the tick-5 replay pair
with fire 1 — a structural 0.9999 artifact, correctly flagged in the gate doc), the **real cross-pair**
fires 3–8 are: 0.617, 0.586, 0.622, 0.628, 0.622, 0.751 → median **0.622**, with a clear upward drift
(0.62 → 0.75 by step 20). Distributionally tight: per-fire q10 ~0.39, q90 ~0.78.

So β_anc=0 does **not** make the carrier memoryless. The valid anchor gradient `M_rep = G_anc_rep`
*itself* carries moderate-high persistence (it is the policy gradient on a slowly-drifting policy, so
consecutive fires' true gradients are correlated). β_anc=0 only prevents *compounding* that
persistence into an unbounded EMA; it does not remove the intrinsic autocorrelation of the signal.

### 5b. The carrier-law arithmetic (explicit)

Treat the M_rep series as AR(1) across the 5-tick refresh interval: `ρ₅ = m6 ≈ 0.622`, so per-tick
`ρ₁ = 0.622^(1/5) ≈ 0.909`, giving an **autocorrelation time `τ = −1/ln(ρ₁) ≈ 10.5 ticks ≈ 5.3 global
steps`** (range across fires: τ ≈ 9.4–17.5 ticks, i.e. 4.7–8.7 steps for the m6 = 0.586–0.751 spread).

Compare to the carrier that *did* ignite, EXP-27's β=0.95 EMA: `τ_β = −1/ln(0.95) ≈ 19.5 ticks`, and
critically that one **compounded** (the EMA integrates, so effective memory grows without the refresh
reset). The carrier law (memory): ignition needs `τ ≫ cadence`. Here cadence = 5 ticks and
τ ≈ 10.5 ticks, so **τ/cadence ≈ 2 — marginal, neither memoryless (τ ≈ cadence) nor the deep
≫-cadence regime of the ignited EMA carrier (τ/cadence ≈ 4, plus compounding).** B2 sits in between,
on the safer side, but not safely off the axis.

### 5c. Why 50 steps is censored

B2 ran 50 global steps = 100 optimizer ticks. EXP-27's damped-EF ignited at step ~61 ≈ 122 ticks. So
B2 was observed to **0.82×** of the EXP-27 ignition horizon and stopped *before* the horizon where the
only comparable carrier-bearing cell ignited. The 50-step emission-free result is therefore a **censored
statistic** in the precise EXP-27 sense — exactly as the verdict states. The clean B2 trajectory
(length mean 274→204 *declining*, clip_ratio ≡ 0, entropy stable ~2.0–2.2, the only 16384 max a single
pre-injection step-2 rollout, val 0.086→0.704@25→0.753@50 monotone) is real and encouraging but does
not de-censor: EXP-27 also looked clean through step ~50.

### 5d. Falsifiable prediction for the 100-step extension (EXP-30-EXT)

EXP-30-EXT is live and at step 4 as of this writing (same config: cadence 5, delay_K 5, λ=1, β_anc=0,
total 100 steps = 200 ticks). My predictions, stated to be falsified:

1. **No ignition through step ~61 (the EXP-27 horizon).** Because B2's carrier is refresh-reset
   (β_anc=0, no compounding) and the *injected* signal is endogenous (§5e), I predict the EXP-30-EXT
   length-mean does **not** cross the early-gate `len/max > 4000` for any consecutive window through
   step ~61, and clip_ratio stays ≈ 0. If it ignites at ~61 like EXP-27, my endogenous-carrier
   hypothesis (§5e) is **falsified** and the τ/cadence ≈ 2 marginal reading was the operative risk.
2. **Elevated but bounded risk in the step 61–100 window.** Given τ/cadence ≈ 2 (not ≫1), I put the
   probability of an ignition (P1 consecutive cap-pins OR P2 sustained len-mean slope > 0 with
   len/max > 4k) in [61, 100] at **roughly 20–35%** — materially above zero, well below EXP-27's
   compounding carrier. *This is the binding measurement; my point estimate is the thing to grade.*
3. **m6 itself should keep drifting up** (it went 0.62→0.75 over steps 8–20). If it climbs past ~0.85
   in the extension, τ crosses into the ≫-cadence regime and ignition probability rises sharply — m6
   logged per fire in the extension is the **leading indicator**, earlier than length. I would treat
   m6 crossing 0.85 as the trip-wire.
4. **val should hold or improve past 0.7528**, not regress, *conditional on no ignition* — the residual
   is converting, and absent a length spiral there is no mechanism for regression in [50,100].

If all four hold, the small-β_anc EMA successor that F2 says is "not cleared" becomes a defensible
proposal *with a measured persistence budget* (it must keep effective τ < ~2× cadence). If (1) or (2)
fail, the residual-at-λ=1 form is the ceiling and EMA variants are off the table.

### 5e. The discriminator hypothesis (why B2 might be genuinely safer, not just luckier)

F4's working hypothesis, which I endorse and sharpen: **ignition requires a persistent *exogenous*
direction pumped into the policy; the δ-residual is *endogenous*.** δ = A − C cancels the circuit's
*own* codec artifact and re-injects the *true* gradient of the *current* objective — it adds no
direction the honest dense optimizer would not also follow. EXP-27's EMA carrier, by contrast, injected
a *stale, compounded* direction (β=0.95 over ~20-tick memory) that is exogenous to the current
objective — a self-reinforcing length-reward-hacking direction. The mechanistic claim: **endogenous
correction (residual) does not pump length; exogenous accumulation (EMA) does.** Prediction (1) above
is the test — if the endogenous residual stays emission-free past the EXP-27 horizon, this is the first
positive evidence for the endogenous/exogenous discriminator, which would be the most reusable
mechanism finding in the program for designing *any* future carrier-bearing correction.

## 6. One thing the verdict did not flag (for the validity-review teammate)

The B2 launch command in `train_B2_delayed_ef_valid_residual.log` carries
`actor_rollout_ref.actor.use_kl_loss=True kl_loss_coef=0.001 kl_loss_type=low_var_kl` and
`actor/entropy` is logged (entropy_coeff=0, so no entropy *bonus*, but a small KL-to-ref loss is on).
CLAUDE.md's fixed control variable is "vanilla GRPO, no-KL no-entropy." This appears to be a
launcher-template default rather than an EXP-30 choice, and it is *held identical* across the A→B2 diff
and against the dense ceiling and the floor bars (all on the same substrate), so it does **not**
threaten the single-knob δ-correction read or the byte-identity contract. But it does mean the absolute
val numbers (0.7528, 0.7536 dense) are a *KL-regularized* GRPO ladder, not the literally-vanilla one in
the GOAL doc. I am flagging it as a known-and-controlled deviation for the validity review to confirm
it is template-wide and not an EXP-30 artifact; it does not change any F1–F5 mechanism conclusion.

## 7. Summary for the team

- **F1 confirmed and sharpened:** on identical (batch, θ), cos(G_anc_rep, G_comp_ring) = +0.007 pooled
  (statistically ⊥, 6.9% of matrices |cos|>0.2), ‖G_anc_rep‖/‖G_comp_ring‖ = 0.29 settled. The true
  gradient is ~3.4× smaller and orthogonal; codec error is ~92% of the compressed gradient's energy.
  Weight-space confirmation of EXP-26's 0.318.
- **Plain-PowerSGD-still-trains reconciliation (open question answered):** ranked — (2) Adam
  per-coordinate normalization on the concentrated true coordinates is primary, (1) running-moment
  cancellation of the rotating codec error second. Single best cheap probe to close it: top-1%-coordinate
  cos(A,C).
- **m1≈0 vs m4≈0.1–0.3 reconciled:** m1 is cross-batch true-gradient correlation (dead); m4 is
  within-circuit *codec-error* autocorrelation (alive). Residual uses the alive structure (cancels it);
  blend needs the dead structure. This is the residual-over-blend selector.
- **Residual converts, blend inert:** λ=1 ring telescoping is exact codec-error subtraction; orthogonal
  contamination of ~92% energy can only be subtracted, never offset by adding an orthogonal partner.
  Consider shrinking delay_K (tighter cancellation) as a lever.
- **m7:** valid gradient is stable-rank ~1.9, top-1%-mass ~0.60. r=77 is over-provisioned in capacity;
  the defect is basis mismatch (act-basis ≠ the ~2 gradient directions). Hybrid-Q corner is already
  falsified (EXP-26 Step C); the live lever is compressing the *residual* (itself low-rank) to ≪77
  columns for GOAL-3 bytes.
- **F2/m6 = 0.62:** β_anc=0 is not memoryless; injected-carrier τ ≈ 10.5 ticks ≈ 2× cadence (marginal,
  below EXP-27's compounding τ ≈ 19.5 + integration). 50 steps = 0.82× the EXP-27 ignition horizon =
  censored. **Falsifiable forecast for EXP-30-EXT: no ignition through step ~61; P(ignite in 61–100)
  ≈ 20–35%; m6 crossing 0.85 is the trip-wire; val holds/improves absent ignition.** The
  endogenous-vs-exogenous carrier discriminator (§5e) is the most reusable mechanism this experiment can
  yield, and EXT prediction (1) is its test.
