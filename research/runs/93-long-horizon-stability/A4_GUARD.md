# a4 uniformization guard, operationalized and committed BLIND (a3 at step 25, a4 not launched)

The registration specifies a4's KILL condition qualitatively: "KILL if rollout ppl, reward slope, or a val proxy degrade (uniformization guard)". That is not executable as written, and a4 launches in about 3.5 hours. Fixing the thresholds now, before a4 produces a single step, for the same reason the gap amendment was posted blind: after the data exists, any threshold I choose is suspect.

This **operationalizes** the three registered signals onto measured quantities. It does not add new criteria.

## The failure mode being guarded

a4 is the incumbent's codec (PRF exact-k p=0.95) plus CVC in cross-entropy mode, lambda 0.003, warmup 20. CVC penalises disagreement between the codec view and the true view. The cheapest way for a policy to satisfy that penalty is to become **flatter**, since a near-uniform distribution looks the same through any codec. That would improve the gap while destroying the sharpness that carries capability. So "the gap got better" is not by itself evidence a4 worked; it is also the signature of the failure.

Because a4 runs with validation off and no checkpoints, the observable proxies are entropy (for rollout perplexity), `critic/score/mean` (the val proxy), and reward slope.

## Calibration, measured on the two relevant comparators over the identical steps 21 to 120 window

Post-warmup window, so CVC is active throughout. a4's correct comparator is the **incumbent**, since a4 is the incumbent's codec plus CVC.

| | entropy level | entropy slope | gap slope | reward slope (gate window) |
|---|---|---|---|---|
| incumbent `90-prf-exactk-600` | 7.812 | **+0.00007/step** | +0.00113 | +0.00310 |
| a1 `a1-srq-b1-sr` | 7.921 | **+0.00036/step** | +0.00658 | +0.00312 |

Both are essentially flat. That is what a non-uniformizing run looks like.

## The committed guard

Evaluated over **steps 21 to 120** (post-warmup). a4 is KILLED if any of U1, U2, U4 fires, and U3 is the diagnostic that distinguishes "CVC worked" from "CVC cheated".

> **U1, entropy veto: KILL if the OLS slope of `actor/entropy` over steps 21 to 120 exceeds +0.0015/step.**
> That is about 21x the incumbent's +0.00007 and about 4x a1's +0.00036, so it cannot fire on noise or on ordinary drift; it fires only on a qualitatively different regime. Over the 100-step window it corresponds to about +0.15 nats of added entropy, which is a visible flattening.
>
> **U2, val-proxy veto: KILL if `critic/score/mean` level at steps 100 to 120 falls below 0.6248** (0.95x the incumbent's 0.65769, the same V3 bar already committed in the A-to-B amendment).
>
> **U3, the uniformization signature (not a veto on its own, a reading): if the gap slope improves to at or below S-bar (+5.0e-4) WHILE U1 or U2 fires, then CVC bought the gap with capability and the arm is disqualified rather than promoted.** This is the row that matters: it prevents a4 being crowned for the exact behaviour the guard exists to catch. Conversely, gap slope at or below S-bar with entropy slope at or below +0.0015 and score level at or above 0.6248 is a genuine PASS and triggers row 1 of the decision ladder.
>
> **U4, degeneracy veto: KILL if `response_length/mean` at steps 100 to 120 falls below 60 percent of its own steps 21 to 40 mean.** Calibration: a1 drifted 748 to 687 tokens, about -1.19 tok/step, which is 8 percent and nowhere near this bar. This is a tripwire for collapse-style truncation, not a fine-grained health score.

If a4 trips U1, U2 or U4, the decision ladder's rows 3 and 4 already specify what happens next (CVC route closed, fall through to a5, or switch I4 to DC mode), so no new decision is created by this addendum.

## Tooling caveat found while calibrating

`gate93.py` returns `ERROR: no history rows` for `90-dense-600`. That is **not** missing data. The script requests the `actor/comm_eff/*` confinement counters in its key list, and `scan_history` only returns rows in which the requested keys exist, so any comm-eff-DISABLED run yields zero rows. The dense reference is read from its on-box `train.log` throughout this program (which is also where the controller setpoint table came from), so nothing downstream is affected. Recording it so a future session does not conclude the dense WandB run is empty.
