# The next runs, relocked 2026-07-26T13:35Z

Supersedes the 11:30Z two-run version. Two changes: the operator's instruction
that `Q` must move only inside the anchor is now implemented rather than
described, and **the unbiased-FRLR test is reinstated** after the operator pushed
back on my demotion of it. Three runs, each changing exactly one thing.

`a8-frlr-qcad20-200` is still running (step 144/200 at 13:20Z, lands ~14:56Z).
The chain `a8 -> a9 -> a10` is armed on the box in tmux `chain-93c`, one
sequential process, so there is no idle GPU at either handoff and no possibility
of two watchers firing on the same signal.

---

## 1. `a9-frlr-anchorq-200` — anchor-owned FRLR. RUNNING NEXT, ~6.5 GPU-h.

**What moves.** a7's exact codec, with the fast path removed as a `Q` writer
entirely. The basis is harvested from the anchor's clean stale-weight forward and
refreshed only when the anchor fires (cadence 20 optimizer ticks). The operator's
instruction of 2026-07-26 verbatim, and the governance PowerSGD has always used.

**Why it is not a re-run of a8.** a8 slowed the *fast* refresh, so `Q` still
tracked the policy it was compressing, only less often. Here `Q` is fitted to the
**slow, stale-weight net** and cannot chase the policy at all. And the broadcast
moves onto the slow circuit, which this program does not charge, so FRLR regains
**exact 1232-bit parity** instead of 1233.4.

**It needed code, not a flag.** Landed in `1ff5e775` + `f0f4a167`:
`comm_eff.py`'s veto rested on the false premise that the codec "has no PowerSGD
basis Q" (true of the plain mask, false of FRLR); `activation_mask.py` had no
ownership plumbing at all. 25 new tests, 159 pass, ruff clean.

Bar, predictions and early-kill triggers: `PREREG_a9.md`.

## 2. `a10-frlr-anchorq-unbiased-200` — does the bias matter? ~6.5 GPU-h.

**What moves.** a9 plus `frlr_unbiased=true`: the residual gain becomes the
constant `H/k`, so `E[h_hat|h,Q] = h` exactly, at *negative* wire cost (the
per-token norm scalar stops being sent, 76 numbers rather than 77). One env var
from a9.

**Why it is back.** I demoted it on the grounds that a8 flattened while still
biased. That reasoning was wrong twice over. a8 shows estimator variance is
**sufficient** to explain much of the gap trend, not that bias is **excluded**;
and the operator was asking about **divergence**, not the gap trend, where the
program's strongest evidence is the a1/a2 factorial: the biased round-to-nearest
arm killed at step 60 with **6.9x worse drift at z=+15**, the unbiased
stochastic-rounding arm survived, one env var apart. FRLR is biased, PRF is not,
and it has never been isolated within FRLR. Open, not settled: that a1/a2 evidence
sits on the `actor/kl_loss` channel this program has discredited and neither arm
has a probe. Cheapness is what settles it, ahead of a 20-h commitment.

Bar, predictions and the fixed outcome table: `PREREG_a10.md`.

## 3. 600 steps of whichever of a9/a10 wins. ~20 GPU-h. NOT yet authorized.

**Why.** PRF is the only arm proven at 600 steps; everything else in this program
is a 200-step result. Durability is the single remaining question against the
incumbent and it cannot be answered at 200.

**Config.** 600 steps, val 0/300/600, probe cadence 5, `SAVE_FREQ=200`, R2 sink
on. **Kill** if the gap crosses the incumbent's 14.3, or val@300 is below 0.65
(under every incumbent checkpoint).

**Selection rule, fixed now.** Flattest gap slope at 100-120 among {a9, a10} at
G1-passing capability. Tie inside 2x on slope goes to the lower codec-free drift
at 200; still tied goes to a9, the simpler arm.

This is new spend on top of a9+a10 and needs the operator's word.

---

## Explicitly dropped, with reasons

| dropped | why |
|---|---|
| further Q-cadence sweeps | operator instruction; a8 gave the cadence-20 datapoint and a9 is the limit case |
| incumbent + cadence-5 probe | would explain *why* but capability already decides *which*; not decision-critical |
| round B controller | nothing in the evidence points at needing an adaptive KL coefficient |
| periodic dense forward+backward | operator rejected |
| error feedback | structurally inapplicable to activation compression: GRPO draws fresh rollouts each step, so no persistent object carries a residual. It belongs on gradients |

## Disk and R2

**Correction to my own correction.** I first recorded ~19G per cell, then
"corrected" it to 37G on seeing three 37G directories, then said the disk would
overflow. The precise figure is **~19G per checkpoint**: a5b/a6/a7 are 37G because
they saved twice (steps 100 and 200), and a8 is **19G** because it saved once. So
a9 and a10 add ~19G each and the disk (132G of 200G now) would have finished around
170G. **It was never going to overflow, and the back-fill is not urgent for space.**
What it IS for is not losing the checkpoints with the box, which is pre-teardown
step 1. The R2 back-fill is running now in tmux `r2-backfill`:
per cell it syncs, verifies **byte-exact** against the remote listing, and only
then deletes the local copy. A cell that fails verification is left on disk. a9
and a10 run with the sink ON, so they upload as they save. That retires
pre-teardown step 1 early instead of racing it at teardown.

## Teardown

**No standing authorization. Ask explicitly, every time.** The trigger is a10
finishing with run 3 unapproved. Ledger headroom is not permission.
