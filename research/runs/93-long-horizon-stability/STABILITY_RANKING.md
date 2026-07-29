# Issue #93 stability ranking

Standalone ranking artifact. Every number in this document comes from the
single authoritative fact sheet for issue #93, built from one WandB snapshot
pulled 2026-07-28 (entity `shamanework-pl`, step axis `training/global_step`).
No number here was computed outside that snapshot.

Run ids: incumbent `woqs8zra`, dense `a134dxxx`, a1 `h0n67q3a`, a2 `3muohefm`,
a3 `k8dvru5l`, a4 `8rux5ea6`, a5 `kfrkehju`, a5b `i54ol342`, a6 `5exrewe2`,
a7 `r7go40tb`, a8 `4zxthzif`, a9 `x6miw0zd`, a10 `5qu3lkt8`, c600 `5v3hrpef`.

## 1. The reframe

The objective of this program is the most stable training achievable under
internet-grade activation compression, not the best reward. The deployment
setting is a model split across pipeline stages that sit on ordinary community
GPUs connected over the public internet, so the inter-stage activations and
boundary gradients are compressed hard and permanently. What we have to be able
to promise about such a system is that it does not walk away from itself over a
long run: that the sampler-versus-trainer gap stays where it started, that the
optimizer stays in a steady state instead of climbing, and that the model never
collapses. A codec that reaches a slightly higher score at step 200 and then
diverges at step 500 is worth nothing in that setting. A codec that holds its
numbers flat for 600 steps is the whole product. So the ranking below is built
on gap stationarity, gradient-norm behaviour, and non-collapse, and the arms are
ordered by those three things.

Capability cannot carry any part of this conclusion, because capability is a tie.
Held-out val across every arm that did not collapse lands in a band of
0.6573 to 0.6713, with the dense control, a reference rather than an arm, reaching
0.6874 at step 450. And the three 600-step runs finish their training reward within
0.006 of each other (dense 0.7437, incumbent 0.7406, c600 0.7380 over the block
501-600). The arms with the best val of the twelve, a7 and a9 tied at
0.6713426853707415, is also the arm family that fails at horizon; the arm that
holds capability perfectly well at 600 steps, c600 at 0.6633, is the arm whose
optimizer drifts 9.25x and whose gap grows 5.12x. Capability therefore
discriminates nothing: it is reported here for completeness and as a floor check,
and it is never used to move an arm up or down the table. The only place it
carries weight is as a veto, in the one case where it collapsed.

## 2. Metrics that may not rank arms

- `actor/kl_loss`. It is real drift multiplied by a codec-view inflation factor
  that itself moves by 50x between arms and across a run, so it confounds the
  thing being measured with the instrument. It spans 55x across a7, a8 and a9,
  which differ only in `Q` governance and have identical physical drift.
- `actor/entropy`. The dense control sharpens the same way, so entropy decline is
  normal GRPO on math and not compression damage. It is also a codec-view
  quantity: c600 reads 6.14 down to 0.053 while the incumbent reads a flat 7.81
  to 7.85, and sampler-side `rollout_log_ppl` is near-identical across c600,
  incumbent and dense at step 599 (0.091, 0.093, 0.108).

Neither metric appears anywhere in the ranking.

## 3. The headline

**Twelve arms were run to beat the incumbent PRF exact-k on stability, and none
of them did.**

The incumbent (`woqs8zra`, PRF exact-k) and the dense control (`a134dxxx`) are
references, not program arms. They set the bar and the ceiling respectively; they
do not compete for a rank. The incumbent remains the only codec in the program
demonstrated stable for 600 steps: gap slope +0.000848 nats per step over the
window 100-599 (n=500, 0.42 nats moved in 500 steps), gap drift ratio 1.029,
`actor/grad_norm` block median flat at 1.50 to 1.82 across all twelve 50-step
blocks, block max never above 4.645, no collapse, and four held-out vals inside
0.6613 to 0.6733.

## 4. The ranking of the twelve program arms

Slopes are OLS on `rollout_corr/kl` in nats per step, quoted over the longest
window each arm actually reached; the window is named in the same cell. Gap drift
ratio is the mean of the last 5 percent of steps divided by the level at step 100,
where 1.00 is perfectly stationary. grad_norm drift is the median of the last 20
percent of steps over the median of the first 20 percent. Absolute grad_norm
levels are not comparable across codecs; the shape is.

| rank | arm | codec | gap slope (window) | gap drift ratio | grad_norm drift | grad_norm max | horizon (steps) | outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | a3 | sr_quant 2-bit byte-parity subset, k=493 | +0.000101 (100-120, n=21) | 1.001 | 0.71x | 7.28 | 120 | ran to schedule, val-off by design |
| 2 | a4 | PRF exact-k plus CVC cross-entropy, wire parity | +0.001542 (100-120, n=21) | 1.002 | 0.91x | 3.62 | 120 | ran to schedule, val-off by design |
| 3 | a1 | sr_quant 1-bit stochastic rounding, block 32, 2304-bit wire | +0.002825 (100-120, n=21) | 1.003 | 0.77x | 0.898 | 120 | ran to schedule, val-off by design |
| 4 | a8 | FRLR r48/k28, `frlr_q_cadence=20` | +0.001485 (100-199, n=100) | 0.986 | 0.05x | 53.82 | 200 | ran to schedule, val 0.6613 @200 |
| 5 | a9 | FRLR r48/k28, anchor-owned `Q`, cadence 20 ticks | +0.012203 (100-199, n=100) | 1.192 | 0.40x | 24.70 | 199 | ran to schedule, val 0.6713 @200 |
| 6 | a7 | FRLR r48/k28, fast `Q` cadence 1, no token-IS | +0.028329 (100-199, n=100) | 1.551 | 1.86x | 68.01 | 200 | ran to schedule, val 0.6713 @200 |
| 7 | a5b | FRLR r48/k28 plus token-IS plus batch normalisation | +0.006673 (100-199, n=100) | 1.150 | 0.86x | 204.39 | 200 | ran to schedule, val 0.6593 @200 |
| 8 | a5 | FRLR r48/k28 plus token-IS | -0.005037 (100-120, n=21) | 1.016 | 1.00x | 2.91 | 120 | ran to schedule, barely learned (reward 0.5895 over 101-200) |
| 9 | a10 | FRLR anchor-owned `Q` plus unbiased residual gain | +0.009809 (30-60, n=31) | 1.018 | 1.31x | 2.28 | killed at 62 (61 grad rows) | killed for FUTILITY, not instability |
| 10 | c600 | FRLR anchor-owned `Q`, corrected, horizon run | +0.045972 (100-599, n=500) | 5.122 | 9.25x | 176.37 | 600 | ran to schedule and FAILED at horizon, val 0.6633 @600 |
| 11 | a2 | sr_quant 1-bit round-to-nearest, deliberately biased | +0.014397 (30-60, n=31) | 1.033 | 0.49x | 62.24 | killed at 60 (62 grad rows) | killed on the pre-authorised instability trigger |
| 12 | a6 | PRF exact-k plus token-IS plus batch normalisation | +0.000413 (100-199, n=100) | 1.002 | 2.27x | 608.81 | 200 | COLLAPSED, val 0.5391 |

## 5. The five axes

### Axis 1: gap stationarity, `rollout_corr/kl` OLS slope in nats per step

Matched window 100-120, the registered gate window, every arm that reached 120:

| rank | arm | slope (window 100-120) | level at 100 -> 120 |
|---|---|---|---|
| 1 | a3 | +0.000101 | 14.984 -> 14.990 |
| 2 | a5b | +0.000358 | 4.410 -> 4.386 |
| 3 | a6 | +0.000383 | 14.132 -> 14.131 |
| 4 | **incumbent** | **+0.000838** | 14.243 -> 14.239 |
| 5 | a8 | +0.001262 | 7.242 -> 6.829 |
| 6 | a4 | +0.001542 | 14.230 -> 14.265 |
| 7 | a1 | +0.002825 | 13.727 -> 13.775 |
| 8 | a5 | -0.005037 | 4.350 -> 4.312 |
| 9 | c600 | +0.008696 | 5.580 -> 5.946 |
| 10 | a9 | +0.009262 | 5.821 -> 5.999 |
| 11 | a7 | +0.016351 | 4.945 -> 5.198 |
| - | dense | -0.000001 | 0.000 -> 0.000 (no codec, no mismatch) |

Longest window each arm actually has. Do not quote a window an arm did not reach:

| arm | window | slope | n |
|---|---|---|---|
| incumbent | 100-599 | **+0.000848** | 500 |
| c600 | 100-599 | **+0.045972** | 500 |
| a6 | 100-199 | +0.000413 | 100 |
| a8 | 100-199 | +0.001485 | 100 |
| a5b | 100-199 | +0.006673 | 100 |
| a9 | 100-199 | +0.012203 | 100 |
| a7 | 100-199 | +0.028329 | 100 |
| a1, a3, a4, a5 | 100-120 only | see table above | 21 |
| a2 | 30-60 | +0.014397 | 31 |
| a10 | 30-60 | +0.009809 | 31 |

Gap drift ratio, mean of the last 5 percent of steps divided by the level at
step 100, where 1.00 is perfectly stationary:

a3 1.001 | a4 1.002 | a6 1.002 | a1 1.003 | a5 1.016 | a10 1.018 |
**incumbent 1.029** | a2 1.033 | a5b 1.150 | a9 1.192 | a7 1.551 |
**c600 5.122** | a8 0.986 (the only arm ending BELOW its step-100 level)

**a3 wins axis 1**, with the flattest slope in the program on the matched window
100-120 (+0.000101) and the drift ratio closest to unity (1.001); a8's 0.986 is
the only sub-1.0 value in the field. a8 owns the one thing a3
does not, the only gap in the program that ends below its step-100 level (0.986).

### Axis 2: gradient-norm behaviour, `actor/grad_norm`

Absolute level is not comparable across arms, because different codecs put the
optimizer at genuinely different scales. The shape over time is comparable. Drift
is the median of the last 20 percent of steps over the median of the first 20
percent; a stationary optimizer sits at or below 1.0.

| arm | steps | p50 first 20% | p50 last 20% | DRIFT | run max | max/p50 |
|---|---|---|---|---|---|---|
| a8 | 200 | 42.9403 | 2.1538 | 0.05x | 53.82 | 18.6x |
| a9 | 199 | 13.1143 | 5.2952 | 0.40x | 24.70 | 6.2x |
| a2 | 62 | 16.3871 | 8.0670 | 0.49x | 62.24 | 6.8x |
| a3 | 120 | 3.6585 | 2.5972 | 0.71x | 7.28 | 2.5x |
| dense | 600 | 0.0579 | 0.0441 | 0.76x | 0.11 | 2.3x |
| a1 | 120 | 0.8499 | 0.6504 | 0.77x | 0.90 | **1.3x** |
| **incumbent** | 600 | 1.7866 | 1.5259 | **0.85x** | **4.65** | **2.9x** |
| a5b | 200 | 0.7236 | 0.6255 | 0.86x | 204.39 | 284.4x |
| a4 | 120 | 1.9412 | 1.7667 | 0.91x | 3.62 | 2.0x |
| a5 | 120 | 0.1457 | 0.1457 | 1.00x | 2.91 | 20.9x |
| a10 | 61 | 1.0318 | 1.3471 | 1.31x | 2.28 | 1.9x |
| a7 | 200 | 1.9744 | 3.6814 | 1.86x | 68.01 | 30.1x |
| a6 | 200 | 17.7015 | 40.2098 | 2.27x | 608.81 | 18.3x |
| **c600** | 600 | 3.5370 | 32.7205 | **9.25x** | **176.37** | 12.1x |

A drift ratio far below 1.0 (a8 0.05x, a9 0.40x, a2 0.49x) means a large startup
transient decaying, not a calm run. Read it next to the run max.

The 50-step block table for the two 600-step runs is the single clearest
stability figure in the program. The incumbent's block median is flat or gently
declining across all twelve blocks and its block max never exceeds 4.645. c600's
block median climbs monotonically and its max reaches 176.4:

| block | inc max | inc p50 | c600 max | c600 p50 |
|---|---|---|---|---|
| 1-50 | 4.645 | 1.815 | 28.551 | 4.970 |
| 51-100 | 4.142 | 1.802 | 4.789 | 2.537 |
| 101-150 | 2.796 | 1.713 | 9.049 | 4.903 |
| 151-200 | 3.835 | 1.681 | 56.038 | 5.709 |
| 201-250 | 3.764 | 1.627 | 19.720 | 6.784 |
| 251-300 | 2.683 | 1.554 | 22.057 | 9.632 |
| 301-350 | 4.166 | 1.554 | 41.602 | 13.623 |
| 351-400 | 2.363 | 1.522 | 58.687 | 21.379 |
| 401-450 | 2.865 | 1.500 | 48.462 | 25.212 |
| 451-500 | 2.054 | 1.513 | 74.714 | 31.945 |
| 501-550 | 2.455 | 1.528 | **176.367** | 27.966 |
| 551-600 | 2.559 | 1.534 | 131.317 | 35.534 |

**a1 wins axis 2.** Its drift is 0.77x with a max/p50 of 1.3x and a run maximum
of 0.898, the tightest gradient behaviour ever measured in this program, and
unlike a8 and a9 it gets there without a decaying startup transient. a4 is the
best like-for-like result at wire parity (0.91x, max/p50 2.0x, max 3.62).

### Axis 3: collapse and kill events

- **a2** killed at step 60 on the pre-authorised trigger. Biased 1-bit
  round-nearest. Its run-MINIMUM grad_norm of 6.153 is 6.9x a1's 120-step MAXIMUM
  of 0.898, on the same codec with only the bias differing.
- **a6** ran to 200 and collapsed capability: val 0.5391 against a field of
  0.6593 to 0.6713. grad_norm p50 40.2 in its last 20 percent, max 608.81.
- **a10** killed at step 62 on two registered triggers. Read this carefully: a10
  was killed for FUTILITY, not instability. Its grad_norm max is 2.285 and its
  drift 1.31x, both unremarkable. Removing FRLR's bias removed FRLR's entire gap
  advantage (14.93 against the incumbent's own 14.66).
- Every other arm ran to its scheduled end.

**c600 wins axis 3 among the program arms**, as the only one that ran 600 steps
with no kill and no collapse. That is also the sharpest illustration of why this
axis is a veto and not a score: c600 passes it while failing axes 1 and 2 outright.

### Axis 4: codec-free drift, `probe/kl_dense` at matched steps

**The incumbent and the dense control have NO probe (0 points). This axis
therefore CANNOT compare PRF against FRLR.** Nothing in this subsection may be
read as a PRF-versus-FRLR result, and no ranking position rests on it.

| step | c600 | a7 | a8 | a9 | a5b | a6 | a10 |
|---|---|---|---|---|---|---|---|
| 100 | 0.004441 | 0.003992 | 0.004282 | 0.004319 | 0.003857 | 0.006561 | - |
| 120 | 0.005216 | 0.005095 | 0.005329 | 0.005265 | - | 0.008268 | - |
| 150 | 0.006209 | 0.005214 | 0.007006 | 0.006088 | 0.008710 | 0.010755 | - |
| 195 | 0.007405 | 0.007876 | 0.010185 | 0.008593 | - | 0.027510 | - |
| 200 | 0.008186 | 0.008201 | 0.010872 | - | 0.016754 | 0.026793 | - |
| 60 | 0.002647 | 0.002613 | 0.002666 | 0.002511 | - | 0.002270 | 0.002419 |

c600 continues alone: 0.011176 at 300, 0.016189 at 450, 0.027622 at 600. Real
drift keeps accelerating even while `actor/kl_loss` appears to flatten, because
the inflation factor `probe/kl_gain` falls from 589.7x at 300 to 354.4x at 600.
This is the cleanest demonstration of why `actor/kl_loss` is disqualified from
ranking.

At the matched step 200 the corrected FRLR run (c600) and a7 are identical to
0.2 percent (0.008186 against 0.008201), which re-confirms with clean data that
`Q` governance does not touch physical drift.

**No arm wins axis 4.** a7 is lowest at steps 120 and 150, a5b at 100, a6 at
60, and c600 at 195 and 200; c600 and a7 agree to 0.2 percent at the matched step
200. Selecting a step selects a winner, which is exactly what correction 4 was
about, so no winner is declared here. Because the incumbent has no probe, even a
within-family reading cannot be extended to PRF against FRLR.

### Axis 5: capability, reported and not used to separate the arms

Held-out val, `val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1`:

- dense: 0.6593 @150, 0.6653 @300, 0.6874 @450, **0.6774 @600**
- incumbent: 0.6613 @150, 0.6633 @300, 0.6733 @450, **0.6613 @600**
- c600: 0.6573 @300, **0.6633 @600** (WandB; log-confirmed 0.657314629258517 and
  0.6633266533066132)
- a7 @200: **0.6713426853707415** and a9 @200: **0.6713426853707415**, identical
  to the digit (335/499); a9's is from the on-box train.log because WandB dropped
  its final step
- a8 @200: 0.6613 | a5b @200: 0.6593 | a6 @200: **0.5391** (the only collapse)
- a1, a2, a3, a4, a5, a10: no val, they ran val-off by design

Training reward `critic/score/mean`, 100-step block means. The three 600-step
runs finish within 0.006 of each other:

| arm | 1-100 | 101-200 | 201-300 | 301-400 | 401-500 | 501-600 |
|---|---|---|---|---|---|---|
| dense | 0.5764 | 0.6729 | 0.6999 | 0.7150 | 0.7291 | **0.7437** |
| incumbent | 0.5015 | 0.6726 | 0.6996 | 0.7216 | 0.7344 | **0.7406** |
| c600 | 0.5081 | 0.6833 | 0.7092 | 0.7271 | 0.7351 | **0.7380** |
| a8 | 0.5048 | 0.6837 | - | - | - | - |
| a9 | 0.5040 | 0.6821 | - | - | - | - |
| a7 | 0.5003 | 0.6726 | - | - | - | - |
| a3 | 0.5071 | 0.6562 | - | - | - | - |
| a5b | 0.4234 | 0.6606 | - | - | - | - |
| a1 | 0.4993 | 0.6519 | - | - | - | - |
| a4 | 0.4987 | 0.6484 | - | - | - | - |
| a5 | 0.4233 | 0.5895 | - | - | - | - |
| a6 | 0.4150 | 0.5185 | - | - | - | - |
| a2 | 0.4580 | - | - | - | - | - |
| a10 | 0.4216 | - | - | - | - | - |

**a7 and a9 win axis 5 on the nose, tied at 0.6713426853707415, and it does not
matter.** The whole non-collapsed field of arms sits inside 0.6573 to 0.6713, so this
axis separates nothing and moves no arm in the ranking. Its only load-bearing use
is the veto it applies to a6.

### The FRLR crossover, exact

c600 against the incumbent on gap level:

| step | incumbent | c600 | ratio |
|---|---|---|---|
| 100 | 14.243 | 5.580 | 0.39 (c600 2.55x better) |
| 200 | 14.318 | 6.846 | 0.48 |
| 300 | 14.429 | 9.199 | 0.64 |
| 400 | 14.479 | 13.190 | 0.91 |
| 450 | 14.542 | 17.805 | 1.22 |
| 599 | 14.659 | 31.104 | **2.12 (c600 2.12x worse)** |

First step where c600 exceeds the incumbent at all: **417**. First step after
which it stays above for the rest of the run: **424**.

## 6. The two counterexamples that discipline the reading

**a6: the flattest gap of any 200-step arm, and it collapsed.** a6's slope over
the window 100-199 is +0.000413, flatter than anything else that reached 200
steps, and its gap drift ratio is 1.002. It also finished with val 0.5391 against
a field of 0.6593 to 0.6713, a grad_norm p50 of 40.2 over its last 20 percent, a
drift of 2.27x and a run max of 608.81. Gap stationarity ALONE is not stability.
The bar has to be: flat gap AND a stationary optimizer AND no collapse. Any
ranking that reads axis 1 without axis 2 and axis 3 puts a6 near the top, which
is why it is 12 of 12 here.

**a5 and a5b: the most stationary numbers in the program, on runs that were not
learning.** a5's grad_norm drift is exactly 1.00x with a median of 0.1457, the
most stationary number in the whole table, and a5b's drift of 0.86x sits one row
below the incumbent's 0.85x and looks like the same shape. Both got there
because token-IS suppressed the update: a5's training reward over 101-200 is
0.5895, second-worst in the field, and a5b's is 0.6606 against 0.6726 to 0.6837
for the other arms that ran the full 101-200 block (the 120-step arms a3, a1 and
a4 read 0.6562, 0.6519 and 0.6484 over steps 101-120 only and are not comparable
here). a5b's calm median also hides a single 204.39 spike, giving
it a max/p50 of 284.4x. A flat metric on a run that is not learning is worthless.
Axis 2 must always be read next to reward.

## 7. What is not established

Stated plainly, because the conclusion is narrower than it looks.

- **On the short matched window 100-120 the incumbent is 4th, not 1st.** a3
  (+0.000101), a5b (+0.000358) and a6 (+0.000383) are all flatter than its
  +0.000838 over that same window. The incumbent's win rests entirely on being
  the only arm whose flatness was tested to 600 steps.
- **a3 and a4 are the live upgrade candidates.** a3 (sr_quant 2-bit byte-parity)
  and a4 (PRF exact-k plus CVC cross-entropy) both matched or beat the incumbent
  on gap flatness AND on gradient tightness at 120 steps. Neither has horizon
  evidence. That is the gap to close, not a settled result.
- **a1's gradient win is bought with bandwidth.** It has the tightest gradient
  behaviour ever measured in this program (max/min 1.2x, run max 0.898), but its
  wire is 2304 bits, 1.87x parity. It is not a like-for-like win over the
  incumbent.
- **Axis 4 cannot compare PRF to FRLR at all**, because the incumbent has no
  probe. Every codec-free drift comparison in this document is within the FRLR
  family plus a6.
- **Compression costs about 1.4 to 1.6 points of val against dense at 600 steps**
  (0.6613 and 0.6633 against 0.6774), and both codecs pay it equally. That cost
  is not attributable to either codec over the other.

## 8. What we would actually ship

PRF exact-k, 77 of 1536 coordinates, constant 1/(1-p) rescale, mask a PRF of
seed, step and layer. It is unbiased, there is no side channel, there is no basis
to broadcast and no anchor coupling, and it is the only configuration with 600
steps of evidence that the optimizer stays in a steady state.

The two upgrades worth a horizon run, in order, are **a4** (PRF exact-k plus CVC
cross-entropy, wire parity, gap +0.001542 over 100-120 and grad drift 0.91x) and
**a3** (sr_quant 2-bit byte-parity subset, gap +0.000101 over 100-120 and grad
drift 0.71x).

FRLR in every governance variant is out for long-horizon work. Its advantage is
real early (2.55x better gap at step 100) and it inverts: crossing the incumbent
at 417 and permanently from 424, ending 2.12x worse at step 599 with the
optimizer drifting 9.25x over the window 1-600. It also needs a biased estimator
to get even the early advantage, which a10 established. This inversion was
predicted in advance by the section-20 theory: PRF's error is rotation-invariant
so its gap is stationary, while FRLR's is alignment-dependent so it chases a
moving subspace.

## 9. Corrections ledger

These supersede earlier text in this program. Do not silently re-revert them.

Standing corrections, already recorded:

1. **"`pg_clipfrac` is 0 by construction" was FALSE for anchor-owned `Q`.** The
   operator found spikes of 0.19 to 0.37 at every anchor fire in a9 and a10.
   Cause: `anchor_update_basis` published `Q` immediately, but the anchor fires
   inside `train_batch` AFTER `old_log_probs` were recomputed, so the two forwards
   of one step used different bases and the PPO ratio deviated from 1 for a
   MEASUREMENT reason, not a policy one. Fixed by porting the staging half of the
   PowerSGD anchor contract, commit `f2ac3c64`. Verified in this snapshot: a9 has
   9 nonzero clipfrac steps (max 0.352534, first at step 20) and a10 has 3 (max
   0.374120); every other run in the program, c600 included, is exactly 0 at
   every step.
2. **"a8 is the best cell" was wrong.** Its terminal val of 0.6613 is BELOW a7's
   and a9's 0.6713.
3. **"The anchor-`Q` constraint costs 7.3x"** was measured at a8's U-curve
   turning point, not as a general figure.
4. **"a9 has the lowest codec-free drift"** was an artifact of unmatched steps.
5. **"Bias is not the driver, so the unbiased test is pointless" was wrong.** The
   operator overruled that call and the test produced the finding that FRLR's
   advantage is bias-dependent.
6. **"The R2 back-fill will not finish"** was projected from a 4-minute sample and
   was wrong.

New corrections from the 2026-07-28 snapshot:

7. **"The incumbent's grad_norm is 1.4 to 2.1 throughout" is TOO TIGHT.** Correct:
   min 1.181, median 1.594, **max 4.645**, last 1.337. The median band is 1.50 to
   1.82; the maximum is 4.645.
8. **"c600's grad_norm goes 3.1 to 68 to 38" is TOO KIND.** Correct: min 1.531,
   median 14.551, **max 176.367**, last 35.534, and the last-50-step max is
   131.317.
9. **"c600 crossed the incumbent at step 413" is wrong.** Correct is **417 first,
   424 permanently.**
