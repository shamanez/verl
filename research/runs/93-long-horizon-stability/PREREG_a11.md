# Pre-registration: a11-prf-exactk-dense50

Written 2026-07-28 while the run was at **step 5**, before any step-50 data
existed. Every threshold below is hardcoded from the incumbent's own history and
from the dense control, both of which are finished runs. Nothing here is to be
rescored on a window chosen after the fact.

WandB `93-long-horizon-stability/a11-prf-exactk-dense50`, id `6zk556im`.
Code: `93-mismatch-control-kit` at `0d7886ce`.

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

- **One injection.** Only step 50 lies inside the measurable window, since step
  100 is the last step and nothing follows it. So S2 rests on a single event.
- **Short horizon.** The control's own slope over 51-100 is +0.002909, which is
  0.145 nats across the whole window. Detecting a halving means resolving about
  0.07 nats. A 50-point OLS averages the 0.05-nat wander down enough for that,
  but it is not a comfortable margin, and a null result at 100 steps does not
  license the claim that the idea fails at 600.
- **Cadence is unexplored.** N=50 is the operator's choice. If S2 succeeds, N is
  the obvious next axis; if it nulls, N=25 is a cheaper retry than N=100.
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
