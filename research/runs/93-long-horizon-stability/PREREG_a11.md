# Pre-registration: a11-prf-exactk-dense50-1000

**AMENDED 2026-07-28 at step 1 of the relaunch: the operator raised the horizon
from 100 steps to 1000.** The 100-step run (`6zk556im`) was killed at step 3 and
is void; ignore it. Everything below is registered against the new run, still
before any step-50 data exists.

Original registration written at step 5 of the 100-step run. Every threshold is
hardcoded from the incumbent's own finished history and from the dense control.
Nothing here is to be rescored on a window chosen after the fact.

WandB `93-long-horizon-stability/a11-prf-exactk-dense50-1000`, id `d3h0fgg5`.
Code: `93-mismatch-control-kit` at `0d7886ce`.

## What the longer horizon changes

**Twenty injections instead of two.** At `dense_every=50` over 1000 steps the
codec is bypassed at steps 50, 100, 150, ... 1000. Nineteen of those are
followed by at least 50 more steps, so the creep test that rested on a single
event at 100 steps now rests on nineteen, eleven of them inside the region where
a matched control exists.

**The control runs out at step 600.** The incumbent `90-prf-exactk-600` is 600
steps long. Steps 1-600 are controlled and are where every registered threshold
lives. **Steps 601-1000 have no control at all** and are descriptive only: they
can show whether an effect compounds or decays, but they cannot be scored
against anything, and no claim of the form "better than the incumbent at step
800" is admissible.

**Capability becomes measurable.** Validation now runs every 150 steps, so
150/300/450/600 are **matched** against the incumbent's own 0.6613 / 0.6633 /
0.6733 / 0.6613, and 750/900 are uncontrolled.

**Cost.** About 30 to 45 hours at the observed 109 to 163 s/step, so roughly
$100 to $150, taking the ledger past its 100 GPU-h cap to about 105-120. The
operator asked for the horizon directly; the cap is a harness backstop, not a
spend authorisation, and it is flagged rather than silently exceeded.

## The primary readout at 1000 steps

The headline stability number of issue #93 is the incumbent's gap slope over
**100-599: +0.000848 nats/step**. That is the number a11 has to beat, and it is
now directly comparable because a11 covers the same window.

| control quantity (90-prf-exactk-600) | value |
|---|---|
| gap slope 100-599 | **+0.000848** /step |
| gap slope 100-300 | +0.000924 /step |
| gap slope 300-599 | +0.000814 /step |
| gap @100 / @300 / @600 | 14.2427 / 14.4294 / **14.6583** |
| grad_norm 1-600 | min 1.181, p50 1.594, max 4.645 |
| val @150/300/450/600 | 0.6613 / 0.6633 / 0.6733 / 0.6613 |

**S2-LONG (the primary test, supersedes S2 as the headline).** a11's gap slope
over **100-599** against the control's +0.000848.
- **Success** iff **<= +0.000424** (half the control).
- **Null** iff inside +/-25 percent, so +0.000636 to +0.001060.
- **Backfire** iff **>= +0.001272**.

**S3-LONG.** a11 gap@600 against the control's **14.6583**. Success iff
**<= 14.50**, which is 0.16 nats below, more than 3x the ordinary 0.05-nat
step-to-step wander.

**S6 (new, capability).** Matched val at 150/300/450/600 against 0.6613 /
0.6633 / 0.6733 / 0.6613. Registered fail iff any matched val comes in more than
**0.03 below** its control point, which is about 15 problems on the 499-problem
set and well outside the incumbent's own 0.0120 checkpoint spread. Capability is
a **veto, not a score**: it cannot promote this arm, only sink it. That is the
issue #93 lesson and it is not being unlearned here.

**S7 (new, and this is the one 1000 steps is really for).** Does the effect
**compound or decay**? Fit the gap slope separately over 100-300 and 300-599 and
compare each against the control's +0.000924 and +0.000814. If the injections
cancel accumulated coherent error, the benefit should be at least as large in
the second window as the first. A benefit that appears early and washes out by
step 600 is a different and weaker finding than one that holds, and the two must
not be reported as the same thing.

**S8 (new, per-injection).** For each injection at step `i` inside 50..550,
compute `gap(i+1..i+10) mean` minus `gap(i-10..i-1) mean`. Under the snap-back
prior this is about zero every time. A consistently negative value across eleven
injections is the cleanest possible evidence for the operator's mechanism, and
with eleven independent events it does not depend on any single one.

## The question

The incumbent PRF exact-k codec is the stability winner of issue #93, but its
train-inference gap still rises monotonically: 13.879 at step 1 to 14.659 at
step 600. Round A established that drift is gated by **coherent,
constant-direction** error rather than by error magnitude, so a permanently
compressed run accumulates one direction step after step.

**Operator's hypothesis:** injecting a periodic uncompressed step gives the fast
circuit one clean unbiased gradient that partially cancels the accumulated
direction, and drags the slow rise back. The duty cycle between "always
compressed" (every arm so far) and "never compressed" (the dense control) has
never been run.

## The cell

The #90 incumbent config with exactly one thing changed: `mask.dense_every=50`.
On every step where `global_step % 50 == 0` the codec is bypassed on every path,
so the forward AND backward are uncompressed and the anchor refresh is
suppressed. 100 steps, so the injections land at **step 50 and step 100**.
Validation off, no checkpoints, `probe/kl_dense` at cadence 5 for measurement
only (controller off).

Verified from WandB, not from the log: `dense_every=50`, `p=0.95`,
`exact_k=true`, `rescale_mode=constant`, `mask_recompute=true`,
`mask_reference=true`, `frlr=false`, anchor enabled 20/20 `owns_q=false`
`rollout_batch`, spectral on. Identical to the incumbent on every one of those.

**The incumbent's own steps 1-100 are the matched control.** No separate control
run is needed and none will be launched.

## Control values, all from finished runs

| quantity | incumbent (90-prf-exactk-600) | dense control (90-dense-600) |
|---|---|---|
| gap @49 | 14.0614 | - |
| gap @50 | 14.0124 | **0.000279** |
| gap @51 | 14.0650 | - |
| gap @100 | 14.2427 | 0.000237 |
| gap mean 45-49 | 14.0299 | - |
| gap mean 51-55 | 14.0945 | - |
| gap slope 51-100 | **+0.002909** /step | - |
| gap slope 2-100 | +0.006296 /step | - |
| grad_norm 1-100 | min 1.376, p50 1.804, max 4.645 | - |

Ordinary step-to-step wander in the incumbent's gap around step 50 is about
**0.05 nats** (14.0614 at 49, 14.0124 at 50, 14.0650 at 51). Every threshold
below is set well outside that.

## MECHANISM checks. If any fails the run is void, not negative.

**M1. The gap must collapse on the bypassed step.** On step 50 and step 100 the
trainer's forward carries no codec, so `rollout_corr/kl` must fall to roughly
the dense control's level. **Registered: gap@50 < 0.01**, against about 14 on
the neighbouring steps. That is a 1000x drop and it is unmissable. If gap@50 is
anywhere near 14, the bypass did not fire and nothing else in this document
means anything.

**M2. `comm_eff/mask_applications/train` must be FLAT across step 50**, because
the hook never reaches the counter on a bypassed step. Registered: the counter
at 50 equals the counter at 49.

**M3. The log must carry both markers**: `comm_eff: DENSE STEP 50` and
`[comm_eff][dense-step] anchor refresh SUPPRESSED`. Step 100 is also an anchor
fire step (cadence 20 with one optimizer tick per global step), so the
suppression marker at 100 is the one that proves the anchor interlock.

## SCIENCE. The registered predictions, and my prior on each.

**S1. Snap-back. Does the correction persist at all?**
Compare a11's gap mean over 51-55 against the incumbent's **14.0945**.
- **Lasting effect** iff a11's 51-55 mean is at or below **13.94**, i.e. at
  least 0.15 nats below the control, which is 3x the ordinary wander and more
  than 2x the incumbent's own 45-49 to 51-55 rise of 0.065.
- **Snap-back** iff it is within 0.05 of 14.0945.

**My prior: snap-back, and I will say so now rather than after.** The #93 report
decomposed this gap into a **13.9-nat static codec artifact present at step 1,
before any training**, plus a +0.28-nat creep over 600 steps. A static artifact
is a property of the codec being on, not of accumulated weight error, so no
optimizer lever can touch it and it must return the instant compression resumes.
If S1 shows a lasting drop, that decomposition is wrong and that is a bigger
result than the one we are looking for.

**S2. The creep. This is the actual target and the only prediction I am unsure
about.**
Compare a11's gap slope over **51-100** against the incumbent's **+0.002909**.
- **Success** iff a11's slope over 51-100 is **<= +0.001455**, half the control.
- **Null** iff it is inside +/-25 percent of the control, so +0.002182 to
  +0.003636.
- **Backfire** iff it is **>= +0.004364**, 1.5x the control.

S2 is the honest test of the operator's mechanism, because the creep is the part
of the gap that accumulates and therefore the only part a periodic clean
gradient could plausibly cancel.

**S3. Level at the end.** a11 gap@100 against the incumbent's **14.2427**.
Success iff **<= 14.10**. This is partly redundant with S2 and is registered
because it needs no slope fit.

**S4. No damage.** `actor/grad_norm` over 1-100 must stay in family with the
incumbent's min 1.376, p50 1.804, max 4.645. **Registered fail iff a11's max
over 1-100 exceeds 9.29**, twice the control's max. A dense step injects a
gradient of a different magnitude than the compressed steps around it, so a
transient at 50 and 100 is expected and is not by itself a failure; a sustained
elevation is.

**S5. Real drift is DESCRIPTIVE ONLY.** `probe/kl_dense` runs at cadence 5, but
**the incumbent carries no probe**, so there is no matched control for codec-free
drift and no threshold is registered on it. It will be reported as a trajectory,
never as a comparison against another arm.

## What this design cannot answer, stated in advance

- **No control past step 600.** Steps 601-1000 are descriptive. They can show
  compounding or decay in a11's own trajectory and nothing more.
- **Cadence is unexplored.** N=50 is the operator's choice. If S2-LONG succeeds,
  N is the obvious next axis; if it nulls, N=25 is a cheaper retry than N=100.
- **One arm, one seed.** Every threshold treats the incumbent's single run as
  the control, so a difference smaller than run-to-run variation cannot be
  distinguished from one. The 0.05-nat wander figure is the only variance
  estimate available and it is within-run, not between-run. Thresholds are set
  at 3x that or more for exactly this reason.
- **The dense steps are inside the measured series.** Steps 50, 100, 150 and so
  on will read a near-zero gap by construction, so every slope fit must
  EXCLUDE the bypassed steps or it will be dragged down by the mechanism rather
  than by the effect. Registered: all slope fits drop steps where
  `step % 50 == 0`. This is a real way to fool ourselves and it is being closed
  in advance.
- **Wire cost is not settled.** A dense step sends 1536 numbers per token rather
  than 77. At N=50 that is an average of about 1699 bits/token/boundary against
  the incumbent's 1232, a 1.38x increase, and it only counts if the dense pass
  crosses the constrained link rather than running in the central mesh. That
  placement decision is not made here.

## Standing constraint noted, and knowingly set aside by the operator

`CLAUDE.md` records: "Fast circuit should never run a dense forward and backward
pass and it is too expensive." This cell deliberately does exactly that, at the
operator's explicit instruction on 2026-07-28, as a **diagnostic probe of the
mechanism** rather than a shippable configuration. If it works, where the dense
pass should live is the follow-up question, not this one.
