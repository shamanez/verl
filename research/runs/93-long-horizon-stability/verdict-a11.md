# Verdict: a11-prf-exactk-dense50-1000. The periodic full-fidelity step is a NULL on its own hypothesis, and it produced the best held-out accuracy in the programme.

**VERDICT: NULL on the registered primary test. No change to the default.**

Scored 2026-07-29 against `PREREG_a11.md`, registered at step 5 before any
step-50 data existed and amended at step 1 of the 1000-step relaunch. Run of
record **WandB `agwwys1z`** (two dead runs share its name; select by ID).

The run reached **step 905 of 1000** and then died when the box vanished (see
"How the run ended"). **The registered scoring window 100-599 is complete**, so
every registered threshold is scorable and none of them is affected. Only steps
906-1000 are missing.

## The registered scorecard

| check | measured | registered bar | call |
|---|---|---|---|
| **M1** gap on a bypassed step | 0.000316 / 0.000213 / 0.000214 at steps 50/100/150 | < 0.01 | **PASS**, and it lands on the dense control's own 0.000279 / 0.000237 |
| **M2** `mask_applications` flat | key not logged under the expected name | flat across a bypass | **not evaluable** |
| **M3** log markers present | `DENSE STEP` count = **0** | both markers | **FAIL**, see below |
| **S2-LONG** gap slope 100-599 | **+0.000725** +/- 0.000006 | success <= +0.000424 | **NULL** (null band is +0.000636 to +0.001060) |
| **S3-LONG** gap @600 | 14.5758 | <= 14.50 | **FAIL**, though 0.083 below the control |
| **S4** grad_norm max | 4.788 (median 1.466) | fail if > 9.29 | **PASS**, and tighter median than the control's 1.594 |
| **S6** matched val, veto | 0.6493 / 0.6794 / 0.6733 / 0.6914 | fail if any > 0.03 below control | **PASS**, three of four ABOVE the control |
| **S7** compound or decay | 0.72x over 100-300, **0.96x** over 300-599 | benefit should hold or grow | **DECAYS** |
| **S8** per-injection delta | **-0.0024** nats mean over 11 injections | consistently negative | **no effect** |
| **S5** `probe/kl_dense` | 0.0270 @600, 0.0505 @900 | descriptive only | reported, not scored |

## The primary result

| gap slope 100-599, bypassed steps excluded from both fits | value |
|---|---|
| a11 | **+0.000725** +/- 0.000006 (n=490) |
| incumbent `90-prf-exactk-600` | +0.000848 +/- 0.000005 (n=490) |
| difference | -0.000122 +/- 0.000008, **14.9 sigma**, ratio **0.86x** |

The creep is 14 percent slower and the difference is overwhelming
statistically. It is still inside the pre-registered null band, so the
registered call is NULL. Significance is not effect size, and the bar was set
before the data existed and is not being moved.

## The hypothesis is refuted, not merely unsupported

The operator's mechanism was that one clean unbiased gradient partially cancels
accumulated coherent codec error and drags the gap level back. Two independent
measurements say that is not what happens.

**No per-injection pull-back.** For each of the eleven injections inside 50..550,
the mean gap over the ten steps after minus the ten steps before, with the
control's own change over the identical window subtracted, averages **-0.0024
nats** (range -0.0185 to +0.0103). There is no kick at the injection. The level
snaps straight back, exactly as the pre-registration predicted it would, because
the gap is dominated by a **static codec artifact present at step 1** (13.88
nats before any training) that no optimizer lever can touch.

**The small benefit that does exist decays.** 0.72x over 100-300 falls to
**0.96x over 300-599**. Whatever the mechanism is, it is diffuse and it washes
out. Over the uncontrolled 600-900 region a11 reads +0.000631, with no control
to compare against.

## Why the lever is weak, which the run answered by accident

On a bypassed step the gradient norm reads **0.037 to 0.057**, while the
compressed steps either side read **1.4 to 2.1**. The clean gradient is
**30 to 35x smaller** than the compressed gradients it is supposed to correct.
That is the same ratio seen between the dense control's median 0.046 and the
incumbent's 1.594: **the codec inflates gradient magnitude about 35x.**

Under Adam the update size roughly tracks the gradient, so one clean step in
fifty moves the weights far less than each of the forty-nine compressed steps
around it. A 14 percent effect from a lever with 1/35th the magnitude at 1/50th
the frequency is arguably the expected size rather than a disappointment. It also
says the way to test the mechanism properly is a **comparably sized** clean
gradient: several consecutive dense steps, or a raised learning rate on the dense
step. That is the a12 proposal, not a rerun of this one.

## The unexpected result: best held-out accuracy in the programme

| val step | a11 | incumbent | delta |
|---|---|---|---|
| 150 | 0.6493 | 0.6613 | -0.0120 |
| 300 | 0.6794 | 0.6633 | +0.0160 |
| 450 | 0.6733 | 0.6733 | 0.0000 |
| **600** | **0.6914** | 0.6613 | **+0.0301** |
| 750 | 0.6934 | no control | |
| 900 | 0.6774 | no control | |

**0.6914 at step 600 is the highest held-out accuracy anywhere in issue #93**,
above the dense control's 0.6774 at 600 and its 0.6874 at 450. And a11's
training reward is *lower* than the control's over the same window (0.7241
against 0.7375 over 401-600), which reads as less overfitting rather than more
learning.

**This does not promote the arm.** Capability is registered as a veto-only axis
and the programme's central finding is that it does not discriminate reliably:
+0.0301 is about 15 problems on a 499-problem set, and this is one seed. But it
is the first arm in the programme to beat the dense control on validation, and it
deserves a deliberate replicate rather than a footnote.

## Corrections to my own earlier reporting

1. **At step 204 I reported the creep as 0.69x, "31 percent flatter", at 5.5
   sigma.** Over the full registered window the answer is **0.86x, 14 percent**.
   The early window flattered the arm and I led with the flattering number. This
   is the **third** time an early window has misled this programme (a8's
   registered window sat on the descending arm of a U-curve; c600's slope
   accelerated 4x between windows and crossed the incumbent at step 424). The
   rule already on the books, no trend claim from a short window, needs to apply
   to *ratios against a control* and not only to raw slopes.
2. **M3 as registered failed and the mechanism is still confirmed.** The
   `DENSE STEP` and `anchor refresh SUPPRESSED` markers never reach the log file,
   because those are `logger.info` calls from a Ray worker. M3 was a convenience
   check; the mechanism is established far more strongly by the gap collapsing
   67,000x on every bypassed step and by the dense-step gradient norm landing
   inside the dense control's own 0.033-0.107 band. **Do not read M3's failure as
   a mechanism failure**, and do not add log-marker checks to future
   pre-registrations without first confirming the logging path.

## How the run ended, and what was lost

The run **crashed at step 905** and the box (vast `45725398`) no longer exists.
I did not tear it down. The harness cron reaper could not have: every invocation
in `/tmp/teardown.cron.log` fails with `Operation not permitted`. The ledger row
reads `TORN_DOWN` but carries none of the `teardown_attempts` /
`teardown_last_at` metadata that a harness teardown writes, so the row was
marked rather than acted on. The most consistent explanation is that the provider
terminated the instance, with the step-905 crash and the disappearance being the
same event. This is not established and is recorded as unresolved.

**Lost:** steps 906-1000 (5 percent of the run) and **every a11 checkpoint**. The
in-training R2 sink was deliberately off, because it is asynchronous to training
but not to process exit and would have idled the GPU; the plan was to push
`global_step_800` and `global_step_1000` to R2 after the run finished. The box
vanished first. The step-600 weights that scored 0.6914 had already been rotated
away by verl's retention policy, which also defeated part of the reason for the
step-9 relaunch.

**Survived:** the complete WandB history through step 905 including all six
validations, so the entire registered scorecard above. All twelve earlier cells'
checkpoints remain byte-verified in R2 at 199.46 GiB.

**Lesson, and it is a repeat.** Durability was deferred to the end of a
thirty-hour run on a machine I do not control. `global_step_800` existed for
hours and could have been pushed the moment it appeared. The next long run pushes
each checkpoint to R2 as soon as it lands, verifies it, and only then considers
deleting the local copy.

## Consequences

- **No change to the default.** PRF exact-k stays the default activation codec on
  the strength of the issue #93 stability ranking. a11 does not beat it and does
  not need to: a11 *is* PRF exact-k plus one extra mechanism, and the mechanism
  did not earn its place.
- **`mask.dense_every` lands anyway**, defaulting to `0` and therefore inert. It
  is the first step-number gate on the codec, it is unit-tested including the
  assertion that a bypassed step's backward is bit-identical to a codec-free
  model, and it is the instrument any follow-up on this idea needs.
- **a12, if pursued:** a comparably sized clean gradient (consecutive dense steps
  or a raised dense-step learning rate) to test the mechanism at a magnitude
  where it could plausibly act, and a replicate of the val@600 result.
- **Ranked against the twelve:** a11 does not enter the stability ranking, which
  closed with the twelve arms of the original matrix. On the ranking's own axes it
  would sit near a4: gap drift comparable to the incumbent, gradients slightly
  tighter, no collapse, and 600 steps of controlled evidence.
