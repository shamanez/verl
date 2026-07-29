# Pre-registration: cell a7, `a7-frlr-r48k28-notis-200`

Written and committed **before the cell exists**, 2026-07-26T01:15Z, while a6 is at
step 162 of 200. Thresholds below may not move once the run starts.

## Why this cell, and why it is the best one available

It is the **never-run corner of the 2x2** that `PREREG_a6.md` drew, and the two
cells run since then have made it the obvious candidate rather than a gap-filler.

| | IS off | IS on + bnorm |
|---|---|---|
| **PRF exact-k** | `90-prf-exactk-600` incumbent | **a6**: gap 14.14, learns 0.78x, ESS 0.0006, grad_norm 57 |
| **FRLR r48 k28** | **a7, this cell** | **a5b**: gap 4.45, learns 0.98x, true drift 0.0168 |

What the two finished cells established:

- the **3.2x gap reduction is FRLR's alone**, since a6 carries the same weighting on
  the incumbent codec and reproduces the incumbent's gap to 0.8 percent
- the **weighting is only viable because the gap is low**: at 14 nats it degenerates
  to ESS 0.0006 and a 33x gradient explosion

Neither says the weighting is *necessary*. Nobody has run FRLR on its own.

## Two questions in one cell

1. **Does FRLR alone deliver the gap win with no IS pathology?** If yes, a7 is the
   best configuration this program has produced: the gap benefit without an
   importance-sampling estimator at all, and therefore without ESS, truncation bias,
   normalisation amplification, or the onset delay both TIS arms showed.
2. **Does token-IS add TRUE, codec-free drift?** a7 versus a5b is the same codec with
   the weighting on and off, and **both carry a dense probe**. That attribution was
   otherwise impossible, because the incumbent has no probe at all, which
   `FINDING_drift_metric_invalid.md` named as the one structural hole left open.

Question 2 is why a7 beats the alternative candidate I considered, an
incumbent-plus-probe reference cell. That would have supplied a reference number but
tested no new configuration; a7 supplies an attribution **and** tests the most
promising setting.

## Exact config

```
ARM=a7 EXPERIMENT_NAME=a7-frlr-r48k28-notis-200 \
TOTAL_STEPS=200 TEST_FREQ=200 SAVE_FREQ=100 \
COMM_EFF_PROBE_EVERY=5 COMM_EFF_PROBE_CTRL_ENABLED=false \
bash examples/grpo_trainer/run_93_cell.sh
```

`ROLLOUT_IS` is deliberately unset; the engine default is `null`, correction off.
The codec was verified **byte-identical to a5/a5b** by diffing the resolved
`DRY_RUN` config, so the only difference from a5b is the weighting. Probe cadence 5
matches a6 and is a superset of a5b's 25. Terminal val at 200 matches both.

Cost: 200 steps at roughly 100-120 s/step plus 40 probes at 20 s, about **6.5 GPU-h**,
roughly $22. Ledger reads about 31.5 h of 100 when a6 lands.

## The bar (fixed now, same thresholds as a5b and a6)

Primary window **100-120** per `PREREG_a6.md` amendment 1; secondary 61-120 also
reported.

| id | criterion | threshold | source |
|---|---|---|---|
| **G1** learning | `critic/score/mean` level >= **0.6248** | unchanged registered floor |
| **G2** gap | `rollout_corr/kl` level < **14.2458** strictly | the incumbent's value |
| **G2** gap slope | <= **+5.0e-4** | registered, and known fragile, see below |
| **G3** drift | `actor/kl_loss` slope <= **3.264e-3** | 1.5x the incumbent's 100-120 slope |
| **G4** wire | = **1232** bits | 48+28+1 = 77 coords x 16, automatic |

Two gates are reported but **explicitly discounted in advance**, on evidence already
in hand rather than after seeing a7's numbers:

- **G2's slope clause is not well posed.** On a5b one additional sample flipped it
  from FAIL at 2.8x to PASS, because the gap oscillates about 0.3 nats inside a
  20-step window while the bar targets 0.01 nats. It will be reported and not banked
  in either direction.
- **G3 is codec-view and not comparable across codecs.** Per
  `FINDING_drift_metric_invalid.md`, PRF's inflation falls 134.6x to 14.3x while
  FRLR's rises 13.8x to 132.9x, and the two channels ranked a6 and a5b in opposite
  orders. **`probe/kl_dense` is the drift number that matters for a7**, and the
  headline comparison is a7 versus a5b on that channel at matched steps.

## Pre-registered predictions, so the read is falsifiable

1. **Gap**: a7 lands at **4.2 to 5.0** nats at 100-120, matching a5b's 4.45 within
   noise. Falsified if it exceeds 6.0, which would mean the weighting contributed to
   the gap after all and the a6 attribution was wrong.
2. **Learning**: a7 reaches **>= 0.6248** and does so **without a5b's onset delay**,
   so its score at 41-60 should exceed a5b's 0.3728. Falsified if a7 shows the same
   delay, which would make the delay a property of FRLR rather than of the weighting.
3. **Codec-free drift**: a7's `probe/kl_dense` at step 200 comes in **at or below**
   a5b's 0.016754. Falsified if a7 drifts more, which would mean token-IS was
   *reducing* true drift and the a5b drift story needs rewriting.
4. **IS health**: not applicable, no weights. `rollout_corr/rollout_is_*` should be
   absent, which is itself the check that the arm is configured as intended.

## Decision rules

- **G1 passes and gap matches a5b** -> a7 is the program's best configuration and the
  round-B/C candidate, with the weighting dropped entirely. This is the outcome the
  evidence points at.
- **G1 passes and gap regresses to about 14** -> prediction 1 falsified, the a6
  attribution was wrong, and the gap win requires the weighting after all.
- **G1 fails** -> FRLR itself cannot learn at this budget, a5b's learning came from
  the weighting, and the FRLR line closes.
- **a7 codec-free drift below a5b's** -> token-IS adds true drift, which finally
  attributes the drift cost and closes the hole in the finding doc.
