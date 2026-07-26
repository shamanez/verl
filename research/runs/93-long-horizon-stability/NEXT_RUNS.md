# The next runs, relocked 2026-07-26T13:35Z

Supersedes the 11:30Z two-run version. Two changes: the operator's instruction
that `Q` must move only inside the anchor is now implemented rather than
described, and **the unbiased-FRLR test is reinstated** after the operator pushed
back on my demotion of it. Three runs, each changing exactly one thing.

**Status 15:05Z.** a8 finished 200/200 at 14:56Z (terminal val 0.6613, scored in
`verdict-a8.md`) and the chain launched **a9 within the same minute**, so the
handoff cost no measurable GPU time. a9 is training; `'owns_q': True` is confirmed
in its resolved Hydra config with zero errors, so the validator relaxation and the
engine wiring work end to end. a10 is chained behind it.

---

## 1. `a9-frlr-anchorq-200` — anchor-owned FRLR. RUNNING NOW since 14:56Z, ~6.5 GPU-h.

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
step 1.

**The first back-fill attempt FAILED and the guard did its job.** `aws s3 sync`
returned **InvalidPart** on `CompleteMultipartUpload` for exactly the four large
files per cell (11.5G optimizer, 6.62G model); every small file landed. At
aws-cli's default 8MB chunk an 11.5G file is ~1437 parts, and R2 rejects that many
at the default concurrency of 10. The per-cell byte-exact check caught it and
**kept the local copies**, which is precisely what it was for. Fix:
`max_concurrent_requests 1` + `multipart_chunksize 256MB` (~46 parts), and per-file
`aws s3 cp` (the `r2_sink.py` path #90 already proved) instead of `sync`.
`chain/r2_backfill2.sh` retries with model files before optimizer state, so if it
fails part-way the half that post-hoc geometry and OOD eval need is already safe.
a9 and a10 run with the sink ON.

**The measured ceiling changes the scope, so state it plainly.** The fix works
(part size lands at exactly 268435456), but throughput is **~2.2 MB/s**, and
concurrency is not the lever: 1 and 4 both gave ~0.5 parts/min, so **the box uplink
is the ceiling**. All 130G is therefore ~16 h, longer than a9 and a10 combined
(~13 h), so the back-fill **will not finish**. The model-first ordering is what
makes that acceptable rather than a problem:

| tier | size | time at 2.2 MB/s | needed for |
|---|---|---|---|
| model + config + tokenizer | ~46G | ~6 h | post-hoc geometry, OOD eval, weight diffs |
| `optim_*.pt` optimizer state | ~84G | ~10 h more | RESUMING training only |

So the model tier lands comfortably and the optimizer state is **explicitly
best-effort**. Losing it costs the ability to resume a 200-step gate cell, which we
would re-run rather than resume. Recorded now so that a partial upload at teardown
reads as the planned outcome rather than a failure.

## Teardown

**No standing authorization. Ask explicitly, every time.** The trigger is a10
finishing with run 3 unapproved. Ledger headroom is not permission.
