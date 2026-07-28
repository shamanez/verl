# Verdict: a10-frlr-anchorq-unbiased-200. KILLED at step 62 on two registered triggers. The codec's bias is what buys the entire gap advantage, and removing it barely touches real drift.

## STABILITY VERDICT (2026-07-28 re-scoring)

> **Re-scored against stability, not reward.** The body below this section was
> written against a bar that leads with capability. Capability turned out to be
> a tie across the field, so it cannot carry a conclusion. What follows
> supersedes the original ranking claims; the original text is kept in place.

**Stability rank: 9 of 12.** a10 is a9 plus `frlr_unbiased`, killed at step 62 on two registered triggers for FUTILITY and not for instability: it did nothing bad to the training dynamics, it simply removed the only reason FRLR was on the table.

| axis | this arm | reference | read |
|---|---|---|---|
| gap slope | +0.009809 over its only window, 30-60, n=31 | incumbent +0.000848 over 100-599 (n=500), +0.000838 over 100-120 | no matched window exists: a10 stopped at 62 and the fact sheet records no incumbent 30-60 slope, so this is 31 rows of early-run trend against 500 rows of horizon evidence, not a comparison |
| gap drift ratio | 1.018 over 61 steps | incumbent 1.029 over 600 steps | nominally flatter than the incumbent and it means almost nothing: 61 steps is not enough elapsed training for a gap to drift |
| grad_norm drift | 1.31x, p50 1.0318 in the first 20 percent to 1.3471 in the last 20 percent, 61 steps | incumbent 0.85x over 600 steps | mildly rising, unremarkable, and measured over one tenth of the incumbent's horizon |
| grad_norm max | 2.285, max/p50 1.9x | incumbent 4.645, max/p50 2.9x | the tightest gradient envelope of the whole FRLR family and tighter than the incumbent's, with no spike anywhere in 61 steps |
| collapse / kill | killed at step 62 on two registered triggers, both gap-level triggers | incumbent none in 600 | this is a futility kill, not an instability kill. Nothing in axis 2 was firing |
| capability | no val, ran val-off by design. Training reward 0.4216 over its 1-100 block, which covers only 62 steps | incumbent 0.6613 @600 val, reward 0.5015 over 1-100 | does not separate the arms |

What a10 proves is a negative about the codec, not about the optimizer. Removing the bias from FRLR's residual gain removes FRLR's entire gap advantage: a10 sits at 14.93 against the incumbent's own 14.66. That is the incumbent's operating point, reached by a codec that additionally needs a basis, an anchor coupling and a `Q` to govern, none of which PRF exact-k requires. Under a stability bar this reads harder than it did under a capability bar. The only argument for carrying FRLR's extra machinery was a lower mismatch gap. Unbiased, there is no lower gap, so there is nothing left to defend, and the horizon evidence that landed later (c600, gap drift ratio 5.122 and grad_norm drift 9.25x with max 176.367) closes the family out anyway.

What a10 does NOT prove is that unbiased FRLR is unstable. Its gradient behaviour is genuinely clean: run max 2.285 and max/p50 1.9x are better than the incumbent's 4.645 and 2.9x, and second only to a1's 1.3x in the whole program. Its gap drift ratio of 1.018 is nominally flatter than the incumbent's 1.029. Those numbers look good for a bad reason: they come from 61 steps. Absence of instability at step 62 is worth very little as evidence about step 600, and this program has two direct demonstrations of that. a7 looked ordinary early and reached grad_norm drift 1.86x with max 68.01 by 200. c600 is a9's configuration, which is a10's configuration minus the unbiased flag, carried to horizon, and it drifts 9.25x. There is a second reason to discount a10's calm: its 1-100 training reward block mean is 0.4216, the second-lowest in the field above only a6's 0.4150, and that block covers only 62 steps so it is dominated by the startup ramp. The same discipline that disqualified a5's perfect 1.00x drift ratio applies here, read axis 2 next to reward and do not credit a quiet optimizer on a run that had barely started learning.

Two corrections from the earlier reading are preserved and must not be re-reverted. First, a10 does NOT overturn the a1/a2 bias result. a2 was killed at step 60 with a run-minimum grad_norm of 6.153, 6.9x a1's 120-step maximum of 0.898, on the same codec with only the estimator bias differing, and that bias is per-coordinate rounding in a 1-bit quantizer. a10's bias is a single detached per-token norm scalar on a low-rank residual, a structurally different object. Bias matters for some codecs and not for this one. Second, running the arm was correct: the earlier claim that the unbiased test was pointless because bias was not the driver is recorded as a corrected claim, and the test is what established that FRLR's advantage is bias-dependent, which is a real and previously unstated property of the method.

Two of the body's channels below need labels. The comparison table quoting `actor/kl_loss` at 0.0637 for a9 against 0.1775 for a10 may not be used to rank either arm: that metric is real drift multiplied by a codec-view inflation factor that itself moves by 50x between arms and across a run, and the very same table shows `probe/kl_gain` moving 106.5x to 322.7x alongside it, so the row is measuring the instrument tripling, not the drift tripling. The codec-free probe is the channel that survives, and at the matched step 60 the field is tightly bunched: a10 0.002419, a9 0.002511, a7 0.002613, c600 0.002647, a8 0.002666, a6 0.002270. The incumbent has no probe at all, so axis 4 cannot compare a10 to PRF. Separately, a10 is one of only two runs in the entire program with nonzero `pg_clipfrac`, 3 steps with a maximum of 0.374120 against exactly 0 everywhere else including c600, and that too was instrument rather than policy: `anchor_update_basis` published `Q` before the anchor fired inside `train_batch`, so the two forwards of one step used different bases. Fixed in commit `f2ac3c64`. So even a10's single anomalous PPO signal is not a stability finding against it.

Killed 2026-07-26T23:40Z at step 62 of 200 under the pre-registered early-kill
triggers in `PREREG_a10.md`, per the operator's instruction not to wait for 200 steps
when the signal is clear. Cell: a9's exact configuration plus
`frlr_unbiased=true`, so the residual gain becomes the constant `H/k` and
`E[h_hat | h, Q] = h` exactly. One environment variable from a9.

## The triggers that fired

| trigger | measured 41-60 | threshold | result |
|---|---|---|---|
| a10-specific gap level | **14.8751** | <= 8.7078 (1.5x a9's 5.8052) | **KILL, 2.56x a9** |
| gap at step 60 | **14.9201** | <= 12.0 | **KILL** |
| score level | 0.5179 | >= 0.40 | pass |
| gap slope | +0.008671 | <= +0.016 | pass |

The gap ceiling was registered as a **ratio** before a9 existed and pinned to an
absolute the moment a9's window closed, so it was not chosen after seeing a10.

## The result: bias buys the gap, and its removal buys almost no drift

**a10's gap is 14.8751, which is the INCUMBENT's operating point** (PRF exact-k sits
at 14.2458). The entire 2.4x gap advantage that FRLR has over the incumbent comes
from the biased, capped, per-token norm-matching gain. Make the codec unbiased and
FRLR is no better than the mask it was built to beat.

And the thing the arm existed to test:

| `probe/kl_dense`, codec-FREE | a9 **biased** | a10 **unbiased** | ratio |
|---|---|---|---|
| step 25 | 0.000285 | 0.000256 | 0.90x |
| step 50 | 0.001498 | 0.001471 | 0.98x |
| step 60 | 0.002511 | 0.002419 | 0.96x |

Removing the bias lowers real drift by **2 to 10 percent**. That is consistently in
the predicted direction at all three probe points, so the effect is probably real,
and it is the **same order as the arm-to-arm noise** already established across
a7/a8/a9 (4 to 8 percent at matched steps). Against a gap penalty of **2.56x**, the
trade is not close.

| at 41-60 | a9 biased | a10 unbiased |
|---|---|---|
| gap | **5.8052** | 14.8751 |
| score | 0.5385 | 0.5179 |
| `actor/kl_loss` | 0.0637 | 0.1775 |
| `probe/kl_gain` | 106.5x | 322.7x |
| wire | 1232 bits | **1216 bits** |

a10 is the only arm in the program **below** the wire parity budget, because the
unbiased gain makes the per-token norm scalar unnecessary (76 coordinates rather than
77, verified from the runtime `mask_ratio` of 0.9505208 = 1 - 76/1536, not merely from
the config). It bought 16 bits and paid 9 nats.

## Prediction scorecard

| # | prediction | outcome |
|---|---|---|
| P1 | gap level WORSE than a9's | **CONFIRMED**, and far beyond the registered margin: 2.56x, not the 1.5x kill line |
| P2 | gap slope within 2x of a9's either way | **CONFIRMED**: +0.008671 vs +0.009262, a ratio of 0.94 |
| P3 | codec-free drift at 200 at or below a9's | **trending CONFIRMED** at 0.90-0.96x through step 60; not scored at 200 because the cell was killed |
| P4 | terminal val within the reference's 0.0120 spread of a9's | **not scored**, cell killed |

P1 was registered against my own interest in the arm and it is confirmed. The
mechanism is the one stated in the pre-registration: the constant `H/k` gain is
unbiased in expectation but has **higher per-token variance**, and the gap is a
per-token KL, so it is exactly the quantity that variance inflates. The capped gamma
trades a little bias for a lot of variance reduction, and for this observable that is
a very favourable trade.

## What this settles, and what it does NOT

**Settles, for FRLR:** directional bias in the residual gain is **not** what drives
reference drift. The unbiased arm drifts 2-10 percent less while its mismatch is
2.56x worse. If bias were the mechanism behind divergence, this arm should have shown
a large drift improvement, and it did not.

**Does NOT overturn the a1/a2 factorial.** That result (biased round-to-nearest killed
at step 60 with 6.9x worse drift at z=+15, unbiased stochastic rounding surviving) was
measured on **sr_quant, a 1-bit quantizer**, where the bias is a rounding bias on every
coordinate. a10's bias is a **norm-matching scalar on a low-rank residual**, a
structurally different object: it is a single detached per-token multiplier, not a
per-coordinate systematic error. So the honest statement is that **bias matters for
some codecs and not for this one**, and the a1/a2 caveat still stands on its own terms
(it was measured on the discredited `actor/kl_loss` channel with no probe on either
arm).

**The operator was right to insist this be tested.** I had demoted it on the wrong
reasoning, that a8 flattening while biased showed bias was irrelevant. The correct
reason to run it was that the question was open and cheap, and the answer turned out
to be worth having: it is now established that FRLR's advantage over the incumbent
rests entirely on a **biased** estimator, which is a real and previously unstated
property of the method being proposed.

## Cost and consequence

Killed at step 62 of 200, so about **2 GPU-h spent of 6.5 budgeted**, and the 4.5 h
saved went straight into run 3. The pre-registered selection rule for run 3 was
"flattest gap slope at 100-120 among {a9, a10} at G1-passing capability"; a10 is
disqualified, so **a9 wins by the rule**, and a9 also satisfies the operator's
architectural constraint.

`c600-a9-anchorq-val600` launched at 23:45:53Z: a9's configuration at **600 steps**,
val at 300 and 600, probe cadence 5, **R2 sink deliberately OFF** because a9 showed
the in-training sink idles the GPU at teardown (three saves would cost about 2.5 h).
GPU idle across the a10-to-run-3 handoff was about **6 minutes**.
