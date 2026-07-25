# Pre-read: a6 at step 51 of 200. One attribution is already safe, two are NOT.

Written 2026-07-25T21:00Z. The registered bar is scored at 100-120 and cannot be
evaluated yet. a5b taught this document's main lesson twice, so the framing below
separates what is structurally decided from what only looks decided.

## The three-way, matched windows

| DRIFT `actor/kl_loss` | 2-20 | 21-40 | 41-51 | slope 21-51 |
|---|---|---|---|---|
| incumbent PRF, no TIS | 0.03052 | 0.03290 | 0.04084 | +0.000427 |
| a5b FRLR+TIS+bnorm | 0.16312 | 0.00564 | 0.01882 | +0.000745 |
| **a6 PRF+TIS+bnorm** | **0.03100** | **0.03304** | **0.03926** | **+0.000340** |

| GAP `rollout_corr/kl` | 2-20 | 21-40 | 41-51 | slope 21-51 |
|---|---|---|---|---|
| incumbent PRF, no TIS | 13.74950 | 13.84502 | 14.01668 | +0.010750 |
| a5b FRLR+TIS+bnorm | 5.68152 | 4.54744 | 4.72780 | +0.009988 |
| **a6 PRF+TIS+bnorm** | **13.75121** | **13.82510** | **13.96315** | **+0.009713** |

| SCORE | 2-20 | 21-40 | 41-51 |
|---|---|---|---|
| incumbent PRF, no TIS | 0.36025 | 0.39204 | 0.49343 |
| a5b FRLR+TIS+bnorm | 0.35752 | 0.35244 | 0.36044 |
| **a6 PRF+TIS+bnorm** | **0.35244** | **0.36851** | **0.39666** |

## SAFE NOW: the gap win belongs entirely to FRLR, not to the weighting

a6 carries the same token-IS and the same batch normalisation as a5b, on the
incumbent's codec, and its gap is **13.96 against the incumbent's 14.02**. That is
a difference of 0.4 percent. a5b's gap is **4.73**.

This attribution is safe at step 51, unlike the two below, and the reason is
structural rather than statistical: **the gap difference is a step-1 property.** At
window 2-20, before meaningful training, a5b already read 5.68 against the
incumbent's 13.75, while a6 reads 13.75, matching the incumbent to four
significant figures. A codec's contribution to the measured gap is an offset
present from the first forward pass, so it does not need 200 steps to establish.

**Consequence.** Token-IS with normalisation buys **nothing** on the gap. The whole
3.2x gap improvement that made the a5 line interesting is the **FRLR codec**, and
it is available without any importance weighting at all. That is a genuinely useful
result: it says the interesting object is the codec, and the weighting can be
dropped from future FRLR cells unless it earns its place on some other axis.

## NOT SAFE YET: the drift attribution, which is the whole point of this cell

a6's drift currently sits **on top of the incumbent's** at every window (0.0310 vs
0.0305, 0.03304 vs 0.03290, 0.03926 vs 0.04084) with a slightly **lower** slope
(+0.000340 vs +0.000427). The tempting conclusion is that the weighting is
harmless and FRLR owns the drift failure.

**That conclusion is not available yet, and a5b is the reason.** At this same window
a5b's drift was **0.01882, less than half the incumbent's 0.04084**. It looked not
merely benign but better. It then went 0.0683, 0.1497, 0.2681, 0.727, and 2.226 by
step 200. Early drift windows do not predict late drift on these arms; a5b's
divergence only became visible after about step 60.

So a6 matching the incumbent at step 51 carries almost no information about step
200. The informative window is **100-120**, and the decisive one is a6's own
terminal drift against a5b's 2.226. Until then the drift attribution stays open.

## NOT SAFE YET: the learning comparison

a6 is at 0.39666 against the incumbent's 0.49343 at 41-51, which is 0.80x and would
suggest token-IS costs learning even on a codec that learns well. **a5b showed
0.73x at this window and converged to 0.98x by 141-152**, so a shortfall here is
consistent with the onset delay that both TIS arms appear to share, not with a
ceiling. Judging learning before step 100 is the specific error this program has now
made once and does not need to repeat.

If a6 also converges, the onset delay itself becomes the attributable cost of the
weighting, which would be a real finding and cheap to state.

## A mechanism confirmation that IS solid: the two codecs' view offsets move in opposite directions

`probe/kl_gain` is the ratio of codec-view drift to codec-free drift.

| arm | trajectory | direction |
|---|---|---|
| a6, PRF exact-k | 134.6x, 120.4x, 121.5x, 137.8x, 126.3x, 126.8x, 100.5x, 81.0x, 58.1x, 44.0x at steps 5-50 | **falling** |
| a5b, FRLR | 13.8x, 33.2x, 34.7x, 49.3x, 66.1x, 92.6x at steps 25-150 | **rising** |

These are opposite, and the mechanism explains both. PRF exact-k draws a fresh mask
each step from a **fixed, stationary** distribution (p=0.95), so its view offset is
roughly constant in absolute terms; as true drift grows, a constant offset divided
by a growing denominator gives a **falling** ratio. FRLR **refreshes its basis Q
every step** (`frlr_q_cadence=1`), so its offset adapts and grows, giving a
**rising** ratio.

This is the prediction that motivated adding the dense probe, and it now holds in
both directions on two codecs rather than being asserted for one.

Note the practical implication: a6's codec-view drift is inflated **44x to 135x**
over its codec-free value, so a6's `actor/kl_loss` readings are even less usable as
behaviour than a5b's were. The codec-free channel is doing real work here.

Finally, the codec-free drift itself is nearly **identical** across the two arms at
matched steps (a6 0.000246 and 0.000921 at steps 25 and 50; a5b 0.000252 and
0.000752), which is a further reason not to read anything into the codec-view
difference at this stage.
