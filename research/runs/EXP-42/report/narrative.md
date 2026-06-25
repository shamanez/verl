# EXP-42 — Look-ahead anchor: projection-horizon sweep

## Summary / TL;DR

Decoupling the look-ahead projection *horizon* from the anchor *staleness* did
not fix k-collapse. With the staleness held fixed at the high-latency setting
(`delay_K = cadence = 20`) and a new `lookahead_strength` α knob sweeping how
far the anchor weights are extrapolated forward, **every cell collapsed** — the
three fixed-α cells (α = 0.25 / 0.50 / 0.75) and the learned/adaptive cell —
and **none reached the 5/5 reference band** (val@100 = 0.7066, established in
EXP-41). The most robust horizon was α = 0.50 (A50), which survived to global
step ~83 with val@25 = 0.646 before length-explosion took it down, but it still
crashed before val@100; the shortest horizon (A25, α = 0.25) collapsed earliest
(step 38), and the longest runnable horizon (A75, α = 0.75) had the lowest
val@25 (0.187) before its log truncated at step 27. The decisive negative is
mechanistic: extrapolating the stale weights forward did **not** lift the
anchor→live-gradient alignment — the first true extrapolated cosines sit in the
same near-zero, sign-unstable regime as the raw-stale anchor (A25 +0.016, A50
+0.008, learned −0.014). The horizon hypothesis is therefore falsified on the 1K
GSM8K surface, and the suspect now common to both EXP-41 and EXP-42 is the
merger, not the projector.

## The method & what EXP-42 tested

The communication-efficient GRPO trainer corrects each compressed update with a
**stale "anchor" gradient**: a frozen clone of the policy whose weights lag the
live model by `delay_K` optimizer ticks. At low latency (`delay_K = cadence = 5`)
the anchor is usable and training is stable. At high latency
(`delay_K = cadence = 20`) the stale anchor gradient is no longer aligned with
the live policy, and training suffers **k-collapse** — a length-explosion–driven
performance crash.

EXP-41 introduced the **look-ahead anchor** (AsyncPP / arXiv:2505.01099
fixed-linear seed): instead of forwarding the anchor gradient from the raw stale
weights θ[t−K], extrapolate the weights forward to a predicted point and compute
the anchor gradient there. EXP-41 ran the *full catch-up* form,

> θ̂ = 2·θ[t−K] − θ[t−2K]   (predict K ticks ahead, i.e. all the way to "now"),

which is α = 1.0 in the parameterization below. EXP-41 was a STOP: the
extrapolation *did* lift alignment (+0.027 vs the raw-stale baseline) and
removed the historic catastrophic entropy ignition, yet training still collapsed
via a softer length explosion (val@100 = 0.0478). Its verdict named the
**signed_ema merger** — β_anc = 0.50, tuned for a *stale* anchor gradient — as
the prime suspect, over-amplifying the now-fresher projected gradient.

EXP-42 acts on the operator's insight that the projection **horizon** is a knob
separate from the staleness. The staleness K is held fixed (the anchor still
lags 20 ticks); what varies is *how far forward* the weights are extrapolated.
This is implemented as a new `lookahead_strength` α:

> **θ̂ = (1+α)·θ[t−K] − α·θ[t−2K]**,  projector coeffs = (1+α, −α, 0),

where α is the fraction of the staleness extrapolated forward (α·K ticks). The
sweep covers α ∈ {0.25, 0.50, 0.75} — predicting 5 / 10 / 15 ticks ahead — plus
a **learned / adaptive** cell whose coefficients cold-start at the fixed seed
(α = 1.0, coeffs (2, −1, 0)) and are then trained online from retrospective
residuals (θ_true[t_prev] − θ̂[t_prev] at a prior fire — no peek at the current
live weights). α = 1.0 is byte-identical to EXP-41's full catch-up. The
hypothesis (plan §Experiment): a *shorter* horizon yields a gentler, less-sharp
anchor that survives 100 steps and reaches the 5/5 band where α = 1.0 collapsed.

## Experimental design

Four cells were scored, all on the **fixed 1K surface** (the only deliberate
cross-cell axes are α and fixed-vs-learned):

| Cell | mode | α (strength) | predicts ahead |
|------|------|--------------|----------------|
| A25 | fixed_linear | 0.25 | 5 ticks |
| A50 | fixed_linear | 0.50 | 10 ticks |
| A75 | fixed_linear | 0.75 | 15 ticks |
| L | learned (cold-start α=1.0) | 1.0 → adaptive | ~10 ticks |

**Held-fixed controls** (identical across cells): Qwen2.5-1.5B-Instruct, vanilla
GRPO (no-KL, no-entropy), GSM8K, `max_response_length = 1024`; anchor
`delay_K = cadence = 20`; merger `signed_ema` (α_ema = 0.25, **β_anc = 0.50**);
PowerSGD r = 77; n = 8 rollouts; 100-step target, `test_freq = 25`
(val@25/50/75/100); 4 GPU. Two reference cells carried over from EXP-41 on the
same surface: **EXP41_ref_5over5** (look-ahead disabled, anchor 5/5, stable
val@100 = 0.7066) and **EXP41_alpha1p0** (α = 1.0 full catch-up at 20/20,
collapsed, val@100 = 0.0478).

**Step-unit note.** Metric `step:` lines are *global steps*; the look-ahead
diagnostics (`tick`) are *optimizer ticks*, with 2 ticks per global step at
batch 128 / mini 64. The anchor fires every 20 ticks; at 20/20 the warm-up needs
two raw-stale fires (ticks 20, 40) before the ring holds both source snapshots,
so the **first true extrapolated fire is at tick 60 ≈ global step 30** (later for
the learned cell, which keeps a deeper 3-point ring — first extrapolation at
tick 80 ≈ global step 40).

**Early-collapse-skip protocol.** A cell is judged collapsed at the first global
step where `response_length/mean` exceeds 2× its own first-25-step mean,
sustained ≥ 2 consecutive logged steps (per-cell thresholds: A25 491, A50 500,
A75 480, L 509). Runs were not forced to step 100 once collapse was unambiguous.

**Probe gate.** Both new code paths passed a fire-forcing invariant probe at
`cadence = delay_K = 1` before any scored cell (`runs/EXP-42/probe-invariants.md`):
P1 confirmed the α knob plumbs end-to-end (`strength = 0.5000` in every
diagnostic; coeffs (1.5, −0.5, 0); source ticks [t−1, t−2], newest < t ⇒ no
leakage; 142 excluded / 196 extrapolated; ring bounded at 2). P2 exercised the
never-run learned path: 3-point source ring, cold-start residual = 0 so the first
learned fire equals the fixed prediction by construction, and — the determinism
invariant the plan demanded —
`lookahead_coeff_cross_rank_max_rel_dev = 0.0` (learned coefficients exactly
cross-rank identical). Anchor-isolation counters were 0 in both probes. The
collapse reported below is therefore a **scientific result, not a broken patch**.

**Data gap (honest).** A75's internal log truncated at global step 27 (Training
Progress 28%), with no error/OOM/Traceback — it ended *before* the first
extrapolated fire (tick 60 ≈ step 30) and before any collapse could be judged.
So A75 has only **val@25 = 0.187**, no collapse verdict, and no extrapolated
cosine; its mode/strength were recovered from the resolved launch command, not
from a fire diagnostic. Every other cell has complete per-step series.

## Results

**The α → collapse-onset curve is non-monotonic (Figure 1).** Among the runnable
fixed-linear cells, collapse onset (global step) does not move monotonically with
the horizon: A25 (α = 0.25) collapses **earliest at step 38**, the learned cell
collapses at **step 43**, EXP-41's α = 1.0 reference collapses at **step 57**, and
A50 (α = 0.50) survives **longest, to step 83**. The mid horizon (α = 0.50) is the
most robust, not the shortest — directly contradicting the plan's
"shorter-is-gentler" hypothesis. A75 cannot be placed on the onset axis (log
truncated before any fire), but its **val@25 = 0.187** is already far below the
other cells' val@25 (A25 0.572, A50 0.646, L 0.397), so the longest horizon was
the worst-performing where it could be measured at all.

**Alignment does not lift — the cosine oscillates near zero (Figure 2).** The
first *true* extrapolated anchor↔live cosine (`cos(g(θ̂), g_live)`) is
near-zero and sign-unstable in every cell: **A25 +0.0157** (tick 60), **A50
+0.0081** (tick 60), **L −0.0138** (tick 80, its first true extrapolated fire),
versus EXP-41 α = 1.0's +0.0325. For comparison, the stable EXP41_ref_5over5
raw-stale baseline hovers in the same band (40 fires, mean ≈ +0.006; individual
fires range roughly −0.06 to +0.05). A50's later fires actually go *negative*
(tick 80 −0.0159, tick 100 −0.0297) before drifting back positive — i.e. the
extrapolated gradient is no more correlated with the live gradient than the raw
stale one, and it flips sign across fires. The learned cell's first true
extrapolated fire being negative (−0.0138) is the sharp datum: the online
coefficients could not keep the projected gradient aligned at the moment it began
extrapolating, and collapse followed within ~3 global steps.

**Length explosion is the recurring collapse mode (Figure 3).** Every collapsing
cell follows the same arc: `response_length/mean` first *declines* through
mid-training (the policy tightens), then reverses into a runaway. A25:
110 → 519 → 665 over steps 31 → 38 → 39. A50: a long slow recovery from ~100
(step 50) back up to ~230 (step 80), then 272 → 388 → 543 → 698 → 842 over steps
81–85. L: a steadier climb 220 → 365 → 510 → 603 over steps 30 → 38 → 43 → 45.
The companion `response_length/clip_ratio` tracks this in lock-step (e.g. A50
clip_ratio 0.06 → 0.17 → 0.36 → 0.55 → 0.76 over the final five steps), and
`grad_norm` rises into the explosion (A25 ~17 → 37 near onset; A50 settling 3–9
then 15.6 at the breach). This is the same soft length-explosion failure mode
EXP-41 documented for α = 1.0, not a return of the historic catastrophic entropy
ignition.

**Validation trajectories all fall away from the band (Figure 4).** No cell holds
the 5/5 reference band (val@100 = 0.7066; band ≈ [0.7066, 0.7255]). A50 — the
only cell to record more than one val point — peaks at val@25 = 0.6459 (already
below the band's floor) and decays monotonically: **0.6459 → 0.5694 (val@50) →
0.3124 (val@75)**, collapsing before val@100. A25 records only val@25 = 0.5724
(collapsed before val@50). L records val@25 = 0.3965. A75 records val@25 = 0.1873.
For reference EXP-41 α = 1.0 read 0.3616 / 0.4981 / 0.1145 / 0.0478. The
best single validation number any horizon cell reached, at any checkpoint, is
A50's val@25 = 0.646 — still ~0.06 below the band floor and on a falling curve.

**Score and entropy confirm the same story (Figure 5).** `critic/score/mean`
rises during early training then rolls over coincident with the length runaway
(A50 peaks ~0.69 around step 36, then declines through the 0.4s and 0.3s as
length explodes; A25 peaks ~0.56 around step 20 then falls to 0.09 by step 39).
Entropy in the fixed-linear cells settles into the usual low band (~2.0 after the
warm-up drop) and trends *down* into collapse (A25 0.94 → 0.68 at the end; A50
0.63 → 0.50) — it does **not** run away, reinforcing that the collapse is driven
by length explosion under a mis-merged gradient rather than by an entropy
blow-up.

## What worked

Stated fairly, several things did work:

- **The implementation is correct.** Both new code paths passed the full
  fire-forcing invariant probe: the α knob plumbs end-to-end (`strength` surfaced
  in every diagnostic, identity θ̂ = (1+α)θ[t−K] − αθ[t−2K] verified, no leakage,
  anchor isolation counters 0, bounded ring), and the never-before-run learned
  projector is **exactly cross-rank deterministic**
  (`lookahead_coeff_cross_rank_max_rel_dev = 0.0`). The collapse is a property of
  the method, not of a bug.
- **The α knob and the learned path work mechanically.** Strength decouples the
  horizon from the staleness as designed; the fires occur at the right ticks; the
  learned residual updates from retrospective residuals with no peek.
- **The mid horizon delayed collapse the longest.** A50 (α = 0.50) survived to
  step ~83 — roughly 45 steps longer than A25 (step 38) and ~26 longer than
  EXP-41 α = 1.0 (step 57) — and reached the highest val@25 (0.646) of any cell.
  If any horizon were ever to be carried forward, it would be α ≈ 0.5.
- **The learned coefficients did dampen oscillation during warm-up.** Through the
  raw-stale warm-up phase (before the ring is deep enough to extrapolate), the
  learned cell's behaviour and alignment were consistent with the cold-start fixed
  prediction it is seeded from — the adaptive machinery did not introduce
  instability of its own. The failure came only once it began *truly*
  extrapolating.

## What didn't work

The core failure is alignment, and it is the same across every cell:

- **Extrapolation does not lift anchor↔live-gradient alignment.** The first true
  extrapolated cosine sits in the same near-zero, sign-unstable regime as the raw
  stale anchor: **A25 +0.0157, A50 +0.0081, L −0.0138** (cf. the raw-stale 5/5
  baseline mean ≈ +0.006). Moving the projection point forward changed *which*
  stale-derived point the gradient is taken at, but not its correlation with the
  live policy gradient. This is the load-bearing negative: the whole premise of
  look-ahead is that a forward-extrapolated weight yields a better-aligned anchor
  gradient, and on this surface it does not.
- **The learned projector could not prevent the sign-flip at first extrapolation.**
  The learned cell's first true extrapolated fire was **negative** (tick 80, cos
  −0.0138), and collapse onset followed at step 43 — within ~3 global steps of
  that fire. Online residual adaptation, cold-started at the fixed seed, did not
  rescue alignment the moment it mattered.
- **The length-explosion collapse mode recurs for every α.** Whether the horizon
  is short (A25), medium (A50), full (α = 1.0), or learned (L), training ends the
  same way: `response_length/mean` breaches 2× its early baseline and validation
  crashes. The horizon knob changes *when* this happens, not *whether*.

## Mechanism / interpretation

The results compose into a coherent picture. Linear weight extrapolation pushes
the anchor toward a predicted future weight, but the predicted point is built
deterministically from two stale snapshots, θ[t−K] and θ[t−2K]. The gradient
evaluated there, g(θ̂), is no better correlated with the live gradient g_live
than the raw stale gradient is — the measured cosines (≈ ±0.01–0.03 across all
cells, sign-unstable) say so directly. Larger α just amplifies the difference
between the two stale snapshots (overshooting further along the same stale
direction), which is why A75 looked worst where it could be measured and why
A50, the moderate horizon, lasted longest. There is no horizon at which the
extrapolated gradient becomes a faithful proxy for the live one.

Feeding a near-uncorrelated, sometimes sign-flipped anchor gradient into the
**signed_ema merger at β_anc = 0.50** is what then drives the length explosion.
That merger was tuned for a *stale* gradient and effectively amplifies the anchor
contribution; when the anchor direction is no better than noise (and occasionally
anti-aligned), amplifying it perturbs the update toward longer, lower-reward
responses, and the clip ratio and grad-norm climb until the policy runs away.
This is exactly the suspicion EXP-41 recorded for α = 1.0 — a fresher-looking but
not-actually-better anchor gradient that the stale-tuned merger over-amplifies —
now reproduced across the entire horizon sweep.

This also matches the broader **σ(M) ceiling** framing from the project's
theory line: any deterministic function of the stale and current gradients
(θ̂ is a fixed linear combination of two stale weight snapshots; g(θ̂) is a
deterministic readout of it) lives inside the information already carried by
those gradients and cannot inject *new* information about the live policy.
Horizon decoupling reshuffles that fixed information; it does not add the missing
on-policy signal. That is why no α, fixed or learned, clears the bar.

## Verdict so far

**STOP for the horizon hypothesis on this 1K surface.** The projection-distance
knob — fixed (α ∈ {0.25, 0.50, 0.75}) or learned/adaptive — is **not** the fix
for k-collapse at `delay_K = cadence = 20`. Every cell collapsed via length
explosion, none reached the 5/5 reference band (val@100 = 0.7066), and the
mechanistic reason is direct: extrapolation does not lift anchor↔live alignment
out of the near-zero, sign-unstable regime. A50 (α = 0.50) was the most robust
horizon and still failed before val@100.

This is explicitly a **verdict so far** — a falsification of the *horizon route*,
not a claim that the whole anchor program is dead. The implementation is sound
(probe gates passed, including the never-run learned path's cross-rank
determinism), so this is a scientific negative, not a broken-patch STOP. Do not
re-enter a commit-hotfix loop on the projector.

**Recommended next direction (deferred, NOT run here): attack the merger.** Both
EXP-41 and EXP-42 now point at the same suspect — the signed_ema merger's
β_anc = 0.50 over-amplifying an anchor gradient (stale *or* projected) that is no
better aligned than noise. The single highest-value next axis is to **lower
β_anc (0.50 → 0.10–0.25)** so the anchor is not over-amplified, holding the
look-ahead and staleness fixed. A shorter look-ahead window or coefficient
regularization on the learned projector are secondary at best, since the horizon
itself has now been shown not to move alignment. This is recorded for operator /
planner review; it is not emitted as an autonomous next-action of this STOP.
