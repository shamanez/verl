# EXP-25 — Gradient-Flow Analysis of the α=0 Collapse (fast circuit ↔ anchor circuit ↔ merger)

**Scope.** A mechanistic, quantitative account of WHY the EXP-25 `signed_ema` α=0 arm
collapsed (val@50 = 0.354), traced through the GRADIENT FLOW: the fast PowerSGD circuit
that produces `G_noisy`, the anchor circuit that produces `M` (the stale clean-gradient EMA),
and the merger that combines them. This is the `mechanist` deliverable for the `surpass-dense`
team and the direct input to `strategist`'s "where does compression noise help vs collapse"
question.

**Builds on (does not re-derive):** the `(2α−1)` knee and the √2 disagreement signature
(`runs/EXP-25/DEEP_FINDINGS.md`, `ENTROPY_COLLAPSE_FINDINGS.md`), the standing watch
(`research/diagnostics/ENTROPY_COLLAPSE_WATCH.md`). **Adds (new):** (1) the three-way
decomposition of WHY disagreement is ~50% — staleness vs compression vs EMA — settled with
data; (2) the compression-noise BIAS characterization from the projector math + the
PowerSGD-only control; (3) the reversal→length chain isolated against entropy level; (4) the
per-layer uniformity result; (5) the fast↔anchor object-mismatch thesis.

**Verdict it supports:** the collapse is NOT a compression-noise effect. Compression
(`G_noisy`) is a faithful, low-bias, low-variance estimator of the dense gradient (the
PowerSGD-only control ties dense at val 0.741). The entire pathology lives in the `signed_ema`
MERGER, which overrides the live update DIRECTION with a stale clean sign on a ~50% coin-flip
basis, destroying the implicit step-size regularizer and igniting a length-degeneration
reward-hack under the no-KL/no-entropy surface.

**Provenance.** All numbers from the local fulltrain logs
(`runs/EXP-25/logs/exp25_alpha_0p{0,3,5}.fulltrain.log`) and W&B (entity `shamanework-pl`,
project `verl_compression_research`). Code cites are `file:line`. W&B run ids: α=0 `uyrpaftw` ·
α=0.3 `r8kc702g` · α=0.5 `1wulaelw` · dense `5e2jpho9` · PowerSGD-only r77+clean@5 `oquyeic3`.

---

## 0. The three circuits and where the gradient flows (orientation)

```
                 ┌─────────────────────────  FAST CIRCUIT (every step)  ─────────────────────────┐
   activations A → boundary fwd hook:  Â = (A@Q)@Qᵀ = A·P      (powersgd_activation.py:381-382)
                   P = QQᵀ  rank-r=77 ORTHOGONAL PROJECTOR, Q detached, M in-graph (no STE)
                   backward through Â ⇒ weight grad  G_noisy   (FRESH, COMPRESSED)
                 └──────────────────────────────────────────────────────────────────────────────┘
                 ┌────────────────  ANCHOR CIRCUIT (every cadence=5 mini-ticks ≈ 2.5 global steps)  ┐
   θ_{t−delay_K} → build_anchor_module deep-clone (anchor.py:349), load K-stale weights
                   UNMASKED clean-PG fwd/bwd (anchor.py:115 anchor_pg_loss, ratio≡1, no clip)
                   raw G_anchor → DP-mean-reduce → M ← β·M + (1−β)·G_anchor   (β=0.95, EMA on CPU)
                   (M = STALE, CLEAN, UNCOMPRESSED, β-SMOOTHED)              (spectral_filter.py:181)
                 └──────────────────────────────────────────────────────────────────────────────┘
                 ┌──────────────────────────────  MERGER (every step, all 196 matrices)  ─────────┐
   G_corr = α·G_noisy + (1−α)·|G_noisy|·sign(M)               (spectral_filter.py:307)
            └ magnitude from FRESH+NOISY+COMPRESSED ┘ └ sign from STALE+CLEAN+SMOOTHED ┘
                 └──────────────────────────────────────────────────────────────────────────────┘
                                              ↓
                              Adam + grad-clip → optimizer step
```

The object the optimizer actually consumes (`G_corr`) takes its MAGNITUDE from one gradient
estimator and its SIGN from a DIFFERENT gradient estimator. §5 argues this object-mismatch is
the core pathology.

---

## 1. The sign-disagreement decomposition (the crux) — it is STRUCTURAL, not staleness/EMA

The merger only acts on coordinates where `sign(M)` disagrees with `sign(G_noisy)` (on agreeing
coordinates `G_corr = |G_noisy|·sign(G_noisy) = G_noisy` for all α). The per-matrix metric
`rel_change = ‖G_corr − G_noisy‖/‖G_noisy‖` (`spectral_filter.py:310`) measures this. For α=0 a
disagreeing coordinate contributes `−2·G_noisy_i`, so `rel_change² = 4·(disagree energy)/(total
energy)` ⇒ **disagree fraction = (rel_change/2)²**. Median rel_change = **1.416 ≈ √2** ⇒ ~50%.

The plan posited three candidate causes for the ~50%: (a) STALENESS (M from θ_{t−K}), (b)
COMPRESSION NOISE (G_noisy ≠ dense), (c) EMA SMOOTHING (M is a β=0.95 average). **The data rule
out all three as the *driver*. The 50% is intrinsic to comparing two different estimators of a
near-zero-mean per-coordinate GRPO gradient.**

### 1.1 It is already 50% at the FIRST warm step — kills staleness-accumulation and EMA-depth

The cold-M guard (`spectral_filter.py:296`) no-ops the merger while `‖M‖≤eps`:
`merger_coldM_fallbacks = 196` (all matrices) on global steps 1–2, then **drops to 0 the moment
the first anchor refresh fires** (`exp25_alpha_0p0.fulltrain.log:1049,1154`; first refresh
`step=5` mini-tick, `realized_delay=4 warmup_fallback=True`). At that first warm correction sweep
`M = 0.95·0 + 0.05·G_anchor` from a SINGLE refresh — i.e. `sign(M) = sign(one near-fresh anchor
gradient)`, with essentially **no EMA history** and **minimal staleness (delay 4)**.

Disagreement fraction (= (median rel_change / 2)²), walking warm samples in order:

| window | rel_change median | disagree fraction |
|---|---|---|
| first 196 warm samples (≈step 3, M ≈ 1 fresh anchor grad) | 1.4200 | **0.504** |
| samples 196–392 (next sweep) | 1.4134 | 0.499 |
| last 196 warm samples (≈step 49, deep collapse) | 1.4229 | 0.506 |

It is **already 50.4% at the first warm comparison** and **flat at ~50% for the entire run**
(per-step medians all in 1.38–1.43; no growth within an anchor refresh cycle, no drift over 50
steps). If staleness-accumulation drove it, disagreement would RISE across the steps 6,7,8,9
between refreshes and reset at the next refresh — it does not. If deep EMA averaging drove it,
disagreement would BUILD as M's effective memory (~1/(1−β)=20 ticks) fills — it does not.
**Both are present but second-order.**

### 1.2 The dominant cause: GRPO per-coordinate signs are near-coin-flip, and M vs G_noisy are two *different* estimators of that near-zero-mean gradient

The true minibatch policy gradient at a coordinate is `g_i = Σ_b A_b · ∂logπ/∂θ_i` — a sum of
SIGNED score-function terms times group-normalized GRPO advantages. Across a GRPO group these
partially cancel, so for most coordinates `|g_i|` is small relative to `Σ_b|A_b ∂logπ|` and `g_i`
sits NEAR ZERO, where its sign is essentially a coin flip determined by residual sampling noise.
Two *independent noisy estimates of the same near-zero-mean quantity* agree in sign ~50% of the
time. `M` (stale + clean + uncompressed + β-smoothed) and `G_noisy` (fresh + compressed) are
exactly two such different estimators of the same underlying per-coordinate gradient. Hence the
~50% disagreement is the EXPECTED, STEADY-STATE signature of the GRPO gradient geometry — not a
defect of any one of the three candidate causes.

This is corroborated by §4 (the disagreement is UNIFORM across all matrix types and all 28
layers, whereas if compression caused it, it would concentrate at the 7 compressed boundary
layers — it does not) and by `‖dM_anchor‖_mean` rising 8.46e-4 → 1.36e-3 → … over the run
(`exp25_alpha_0p0.fulltrain.log:1081,1188`): M is a moving target, never a stable sign oracle.

**OPEN (runtime check that would settle the residual contribution of each cause):** directly log,
per matrix per step, the cosine `cos(G_noisy, M)` and the cosine `cos(G_noisy, G_dense)` (need a
parallel uncompressed fast backward to get `G_dense`). The decomposition above is inferred from
the timing/uniformity invariants, not from a direct three-way ablation of sign sources. A cheap
settle: a one-step diagnostic logging `sign-agreement(M, G_noisy)` vs `sign-agreement(G_anchor_fresh,
G_noisy)` at delay_K=0 — if both are ~50%, staleness contributes ~0, confirming §1.2.

---

## 2. Compression-noise characterization: LOW-MAGNITUDE, LOW-VARIANCE, but STRUCTURALLY BIASED (the key input to strategy)

This is the question `strategist` needs answered. **`G_noisy − G_dense` is NOT zero-mean
exploration noise; it is a deterministic, structured BIAS — but a small, direction-faithful one.**

### 2.1 From the projector math: the residual is a fixed off-subspace projection, not random noise

The fast hook reconstructs the boundary activation as `Â = (A@Q)@Qᵀ = A·P`, `P = QQᵀ` a rank-r
ORTHOGONAL PROJECTOR (`powersgd_activation.py:381-382`). Q is detached and M stays in-graph, so the
autograd backward is the EXACT self-adjoint projector `dL/dA = (dL/dÂ)·QQᵀ` (no STE,
`powersgd_activation.py:376-382`). The dropped residual `A − Â = A·(I−P)` is therefore the
**off-subspace component of the activation** — the SAME (H−r) directions are dropped on EVERY
forward (Q is frozen for the whole global step, `maybe_update_basis` runs only end-of-step). A
quantity that is deterministically removed every step is a BIAS by definition, not a fresh
zero-mean perturbation.

### 2.2 But the bias is small and aligned, because the kept subspace is the DOMINANT one

`r/H = 77/1536 = 0.050` — PowerSGD keeps only 5% of the activation dimensions and drops 95% — yet
`reconstruction_rel_error` (= ‖A−Â‖/‖A‖) is only **median 0.025** (range 0.023–0.034 across the
whole run; W&B `uyrpaftw` per-step, layer_11/layer_3 logged). The activation energy is
concentrated in a low-rank dominant subspace, and the block-power-iteration `Q ← orth(V)`,
`V = Σ Mᵀ(MQ)` (`powersgd_activation.py:413-424,581-583`) deliberately tracks exactly those
dominant directions. So the bias preferentially keeps the high-energy directions and drops the
low-energy tail ⇒ `G_noisy` is a slightly *shrunk-but-aligned* version of `G_dense`, with low bias
in DIRECTION. This is why the PowerSGD-only control byte-matched dense in EXP-20 (r=77, 0.7415) and
re-confirms here (§3).

### 2.3 The reconstruction error is STATIONARY — compression does not destabilize during collapse

`reconstruction_rel_error` is flat at ~0.023–0.034 for the entire α=0 run, drifting up only mildly
(0.023→0.034) as the policy degenerates into long sequences (W&B `uyrpaftw`). It does NOT spike at
the collapse onset (step ~30) the way entropy/length do. So the compression channel is bounded,
low, and constant THROUGHOUT — it is not the thing that changes when the run collapses.

### 2.4 Magnitude: compression slightly inflates grad norm; the MERGER inflates it catastrophically

Warm-step (4–28) median `actor/grad_norm`:

| run | mechanism | grad_norm median | grad_norm mean (max) |
|---|---|---|---|
| dense `5e2jpho9` | true gradient | **0.387** | 0.385 (0.4) |
| PowerSGD-only `oquyeic3` | `G_noisy`, no merger | 1.645 | 1.576 (3.3) |
| α=0.5 `1wulaelw` | merger zeroes disagreers | 2.676 | 3.847 (26) |
| **α=0 `uyrpaftw`** | merger reverses disagreers | **3.346** | **11.3 (115)** |

The dense PG is small (0.387) BECAUSE of per-coordinate sign cancellation (§1.2). `G_noisy` alone is
~4× larger but stable. The α=0 merger is **~8.6× the dense norm at the median and wildly erratic
(mean 11, max 115)** — because `|G_noisy|·sign(M)` puts the full compressed magnitude on every
coordinate with a fixed stale sign, DESTROYING the cancellation. (Note: `mask.rescale=true` in the
resolved config is INERT here — the codec is pure PowerSGD, `mask_applications=0`, no
inject/blend path — so the inflation is purely the merger, not the mask rescale referenced in older
EXP-16/18 docs.)

### 2.5 Net answer to strategist

> PowerSGD activation compression yields a **low-magnitude, low-variance, structurally-biased
> (fixed off-subspace-dropped), but direction-faithful** gradient. It is the **opposite** of
> zero-mean exploration noise: there is no fresh randomness injected per step, and the bias is
> aligned with the dominant signal directions, so it behaves like dense-grade signal (val 0.741 ≈
> dense 0.754). Therefore the collapse is **NOT** a "compression noise hurts" story, and equally a
> "compression noise helps exploration" story has no zero-mean perturbation to lean on. Any
> surpass-dense path must treat `G_noisy` as ≈dense signal and look elsewhere for an edge.

---

## 3. The PowerSGD-only control: compression alone is BENIGN — the merger is the entire pathology

The cleanest isolation. `oquyeic3` is PowerSGD r=77 with a fresh clean step every 5, **NO
anchor/merger** (`anchor_backwards=0`). Same model/data/no-KL/no-entropy surface, same codec, same
reconstruction error (~0.022):

| step | entropy | score | resp_len | recon_rel_err |
|---|---|---|---|---|
| 10 | 0.379 | 0.305 | 250 | 0.0247 |
| 25 | 0.335 | 0.688 | 223 | 0.0203 |
| 40 | 0.278 | 0.734 | 213 | 0.0239 |
| 48 | (val tick) | 0.780 | 198 | 0.0238 |

`val@50 = 0.7414` — ties dense (0.7536) within ~1pt. Entropy settles to ~0.27–0.40 (LOWER than the
non-collapsing α=0.5's 0.39) yet length stays bounded (198–295) and reward rises monotonically. So
**identical compression, identical low entropy, identical reconstruction error, but NO merger ⇒ no
collapse.** The only experimental difference between healthy `oquyeic3` (0.741) and catastrophic
α=0 (0.354) is the `signed_ema` merger. Compression is exonerated; the merger is the cause.

---

## 4. Per-layer / per-matrix: the disagreement is UNIFORM, NOT concentrated at compression boundaries

Warm-step median rel_change for α=0, by matrix type and by layer (`exp25_alpha_0p0.fulltrain.log`):

| matrix type | n | median rel_change |
|---|---|---|
| k_proj | 190 | 1.427 |
| q_proj | 195 | 1.422 |
| gate_proj | 200 | 1.421 |
| v_proj | 196 | 1.420 |
| o_proj | 206 | 1.416 |
| up_proj | 198 | 1.414 |
| down_proj | 194 | 1.413 |

By layer index: layers 0–16 all median ~1.38–1.44; the final logged layer 27 is slightly lower
(1.29). **The sign-disagreement is uniform across all 7 matrix types and all layers.** The PowerSGD
codec compresses activations at only 7 boundary layers (`decoder_boundary_indices`, e.g.
3/7/11/15/18/21/24), but the merger corrects all 196 matrices and the disagreement does NOT track
the compressed boundaries. This is direct evidence that the ~50% disagreement is a property of the
GRPO gradient geometry across the WHOLE model (§1.2), **not** of the compression boundaries —
de-linking the disagreement from compression a second way. The mild drop at layer 27 (nearest the
output head, where the PG signal is strongest and least cancelled) is consistent with the coin-flip
story: where `|g_i|` is genuinely large the sign is better-determined and agreement rises.

---

## 5. The fast↔anchor MISMATCH is the core pathology, and the (2α−1) knee neutralizes it

### 5.1 The object-mismatch thesis

`G_corr` takes its **magnitude from `G_noisy`** (FRESH, COMPRESSED, full-magnitude per coordinate)
and its **sign from `M`** (STALE θ_{t−K}, CLEAN/uncompressed, β=0.95-SMOOTHED). These are two
*different gradient objects* evaluated at *different weights* through *different forward paths*. The
merger glues the magnitude of one onto the sign of the other. On the ~50% of coordinates where they
disagree (§1), the result is `(2α−1)·|G_noisy|·sign(G_noisy)`:

- **α=0 → −1.0·|G_noisy|**: full-magnitude REVERSAL (ascent on the live objective).
- **α=0.3 → −0.4·|G_noisy|**: 40% reversal (still net wrong-direction).
- **α=0.5 → 0**: disagreeing coordinates ZEROED — the mismatch is NEUTRALIZED (the knee).
- **α→1 → +1.0·|G_noisy|**: no merger = plain PowerSGD (the §3 benign control).

The mismatch is maximally harmful precisely because (a) the stale clean sign is a *worse* estimator
of the live update direction than the live (compressed) gradient's own sign — `G_noisy` is
direction-faithful to `G_dense` (§2.2), while `M` is a smoothed average over a non-stationary
boundary geometry evaluated K steps ago; and (b) the merger keeps the FULL live magnitude on the
reversed coordinate, so a wrong-signed step is taken at full strength. There is no information in
the merged object that improves on `G_noisy` — it strictly degrades the direction on half the mass
while keeping the magnitude.

### 5.2 Why α=0.5 escapes and α=0/0.3 do not — same mechanism, dose-graded

At α=0.5 the merger is `G_noisy` on the agreeing half and 0 on the disagreeing half: a strict
CONTRACTION of the live gradient (projection onto the sign-agreement set), never an ascent
direction. It throws away ~half the gradient (which is why it only reaches 0.707, not dense 0.754)
but cannot drive a runaway. α=0/0.3 keep a wrong-signed full-magnitude component, which is a
persistent ascent force.

---

## 6. The reversal → length-degeneration chain (why LENGTH is the channel)

### 6.1 Entropy level is NOT the trigger — sign-reversal is

The cleanest dissociation in the whole experiment. Length-ignition (first step where
`response_length/mean > 2× its step-10 baseline) vs the minimum entropy each run reaches:

| run | mechanism | length-ignites? (step, entropy-there) | min entropy | final resp_len |
|---|---|---|---|---|
| dense `5e2jpho9` | true grad | **never** | **0.122** | 193 |
| PowerSGD-only `oquyeic3` | G_noisy, no merger | **never** | 0.266 | 198 |
| α=0.5 `1wulaelw` | merger, disagreers→0 | **never** | 0.392 | 165 |
| α=0.3 `r8kc702g` | merger, −0.4 reversal | **YES (step 33, ent 0.61)** | 0.308 | 15580 |
| α=0 `uyrpaftw` | merger, −1.0 reversal | **YES (step 30, ent 0.47)** | 0.059 | 6383 |

**Dense trains down to entropy 0.122 — LOWER than any non-collapsing comm-eff arm — with bounded
length 193.** Low entropy is therefore NOT what ignites length: a confident GRPO policy on GSM8K is
correct, not collapsed. Length ignites ONLY in the two reversal arms, and ONLY once entropy has
fallen far enough (~0.5–0.6) that the policy is committed. The reversal is a persistent
wrong-direction force; once the policy is sharp, that force pushes it OFF the solution manifold into
the nearest reward-correlated degenerate basin.

### 6.2 Why that basin is LENGTH specifically, under no-KL/no-entropy

GRPO's reward on GSM8K is lenient (answer-string match). Under `use_kl_loss=False`,
`use_kl_in_reward=False`, `entropy_coeff=0` (resolved config) there is NO anchor to a reference
policy and NO entropy bonus — the ONLY shaping signal is the reward. A persistent wrong-direction
full-magnitude step (§5.1) needs *somewhere* to go that the reward tolerates. Emitting longer
outputs raises the per-sequence chance of stumbling onto the answer substring under the lenient
match, so reward briefly RISES even as the policy degenerates (α=0 score peaks 0.787 @ step 28,
while length is already starting to climb). Past the useful regime the responses run away to the
16K cap: α=0 length 282→593→977→1406→…→8634, `response_length/clip_ratio` 0.00→0.46; α=0.3 saturates
the cap at 15639, clip_ratio 0.905. The runaway is self-reinforcing (the H3 loop): low entropy ⇒
repetitive non-EOS continuations ⇒ longer ⇒ wrong/garbled ⇒ reward falls ⇒ within-group reward
variance shrinks ⇒ GRPO advantages degrade ⇒ noisier signal ⇒ further collapse. The
train↔rollout IS gap (`rollout_probs_diff_mean`) collapses 0.84→0.07 (α=0) / 0.61→0.20 (α=0.3) as
the policy goes near-deterministic on its own degenerate rollout, while α=0.5/dense/PowerSGD hold
~0.62–0.64. `ppo_kl`≈0 and `pg_clipfrac` small throughout (rollout correction OFF ⇒ `old_log_prob`
recomputed by the training policy ⇒ ratio≈1) means PPO clipping provides NO brake on the drift —
shared with the non-collapsing references, so it is a permissive amplifier, not the cause.

### 6.3 α=0.3 vs α=0.5: the bifurcation is sharp and dose-located

α=0.3 and α=0.5 are IDENTICAL through step ~28 (both: entropy ~1.0, length ~150–180, score
~0.77–0.79). Then at step 32–33 α=0.3's length ignites (440→3682→12681) while α=0.5's stays flat
(116→111→126). The −0.4 reversal in α=0.3 accumulates as a slow wrong-direction drift that crosses
the ignition threshold ~10 steps later than α=0's −1.0; α=0.5's zeroing never crosses it at all. The
collapse is a dose-graded bifurcation in the reversal coefficient `(2α−1)`, with the phase boundary
exactly at α=0.5.

---

## 7. The pending KL diagnostic — prediction to fold in

A KL-divergence run (α=0 signed_ema + `use_kl_loss=true`, `kl_loss_coef=0.001` to a frozen
reference policy; `runs/EXP-25/exp25_a0_kl001.sh`) is running on a separate box.

**Prediction (mechanism-grounded):** the KL brake should SUBSTANTIALLY ARREST the length explosion.
The reference policy assigns very low probability to runaway-length non-EOS continuations, so a KL
penalty to it directly opposes the §6 degenerate basin — supplying exactly the brake the no-KL
surface lacks (§6.2). If so, it ISOLATES "the merger injects a persistent wrong-direction force"
(still present, unchanged — `rel_change` should stay √2) from "nothing stops the resulting runaway"
(now braked). Expected outcome: entropy still declines and the merger bias still degrades the
DIRECTION, so val should recover from 0.354 toward the α=0.5/PowerSGD band BUT is unlikely to beat
the PowerSGD-only control (0.741) — because the KL brake removes the catastrophic length channel
without making the stale-sign correction *helpful* (the monotonic α-dose-response, DEEP_FINDINGS §c,
says correction is net-harmful regardless of the brake). A clean falsifier of my account would be:
KL does NOT arrest the length explosion (would implicate something beyond the reward-only degenerate
channel) OR `rel_change` departs from √2 under KL (would mean KL changes the gradient geometry, not
just the brake). Update this section when the result lands.

---

## 8. Summary — the gradient-flow account in five claims

1. **Compression is benign.** `G_noisy` is a low-magnitude, low-variance, structurally-biased
   (fixed off-subspace-dropped) but direction-faithful estimator of `G_dense`
   (`reconstruction_rel_error` ~0.025, stationary; PowerSGD-only control ties dense at 0.741). It is
   the OPPOSITE of zero-mean exploration noise. [§2, §3]
2. **The ~50% sign-disagreement is structural, not staleness/compression/EMA.** It is already 50.4%
   at the first warm comparison and flat across the whole run; it is uniform across all matrix types
   and layers (not concentrated at the 7 compressed boundaries). It is the coin-flip signature of two
   different estimators of a near-zero-mean per-coordinate GRPO gradient. [§1, §4]
3. **The merger is the entire pathology.** `G_corr = α·G_noisy + (1−α)·|G_noisy|·sign(M)` glues a
   fresh-noisy magnitude onto a stale-clean sign; on the disagreeing half it yields
   `(2α−1)·|G_noisy|`, a full-magnitude reversal at α=0 that destroys the implicit sign-cancellation
   step-size regularizer (grad-norm inflates from dense's 0.387 to α=0's 3.3–11). [§2.4, §5]
4. **The proximate killer is a length-degeneration reward-hack, not low entropy.** Dense trains at
   LOWER entropy (0.122) than any non-collapsing comm-eff arm yet never ignites length. Length
   ignites ONLY under sign-reversal (α<0.5), once entropy is low enough that the persistent
   wrong-direction force pushes the committed policy into the reward-only length basin (no KL/entropy
   brake to stop it). [§6]
5. **The fast↔anchor object-mismatch is the core defect, neutralized exactly at α=0.5.**
   clean-stale-sign × noisy-fresh-magnitude is strictly worse than `G_noisy`'s own (direction-faithful)
   sign; α=0.5 zeroes the mismatch (contraction, survives but throws away half the gradient), α→1
   removes the merger (the benign PowerSGD control). The monotonic dose-response says the correction
   is net-harmful at every dose. [§5, §3]

**Strategic implication (for `strategist`):** treat PowerSGD-compressed gradient as dense-grade
signal. Do NOT use a stale signal to OVERRIDE the live update direction (the falsified primitive).
If a stale full/clean signal is to be used at all, it must be ADDED in a direction-preserving way
(error-feedback on the compression residual, or a confidence-weighted preconditioner) — never as a
sign replacement. There is no compression-noise exploration edge to exploit here; the edge, if any,
must come from the stale clean signal CORRECTING the (small, off-subspace) compression BIAS without
touching direction.

### 8.1 The parity-vs-surpass ceiling (load-bearing for the surpass-dense plan)

A blunt quantitative caution on how much head-room a direction-preserving correction actually has.
The compression BIAS that such a correction (error-feedback / preconditioner) can recover is the
off-subspace component `(I−P)·g`. Its relative size is bounded by the dropped activation energy =
`reconstruction_rel_error² = 0.025² ≈ 0.0006` — i.e. **~0.06% of the activation energy is dropped**;
the boundary weight gradient is genuinely LOW-RANK (the EXP-20 #20/#21 finding: recon →~2% flat
across r∈[77,102], compressed steps book **57–95%** of the reward gain, the full-rank clean step only
**4.8–19.6%**). So the bias a direction-preserving correction can recover is single-digit-% of the
update at most. That makes the realistic ceiling of "anchor corrects compression bias"
**PARITY with dense** (it can let you drop the clean step ⇒ pure comm savings), **not surpass**.

To SURPASS dense, the >dense signal has to come from an information channel dense does not already
use — and the stale CLEAN UNCOMPRESSED anchor gradient is NOT such a channel, because dense already
sees the full uncompressed activations every step. The anchor only re-supplies, stale and smoothed,
information the dense run has fresh. So a surpass-dense claim built on "the anchor adds back what
compression drops" is mechanically a parity-recovery argument, not a surpass argument. If
`strategist` has a surpass mechanism, it must identify where the extra-dense signal originates
(e.g. the EMA acting as a variance-reducing momentum/look-ahead that dense's single-step gradient
lacks — a *different* and testable claim from "correct the compression bias"). This is the crux to
converge on. **OPEN:** the one number that would settle whether even parity is reachable is the
dense-vs-compressed update COSINE (predicted ≳0.98 post-warmup, never logged — EXP-20 `verdict.md`
success criterion); log it on the next run.
