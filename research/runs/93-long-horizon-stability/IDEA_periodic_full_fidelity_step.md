# Operator idea, noted for later: a periodic full-fidelity forward AND backward every 50 to 100 steps

Recorded 2026-07-25 at the operator's request. **Not implemented, not scheduled.** Noted only so it can be checked later.

## The idea, as the operator stated it

Keep the two-circuit setup with the PRF mask exactly as it is. The #90 PRF exact-k run already **worked properly for 600 steps with no collapse**, but its train-inference gap was **still rising** (13.88 to 14.66, and its reference KL reached 0.91 nats by step 600).

So: every **50 or 100 steps**, do a **proper full forward and backward pass** with no compression. The intent is corrective rather than diagnostic. A clean full-fidelity gradient every N steps should pull the accumulated difference back a little and **prevent the run from overshooting**, instead of letting the gap and the drift creep monotonically upward for 600 steps.

## Why it is worth checking

- It attacks the mechanism round A actually established. The a1/a2 factorial showed drift is gated by **coherent, constant-direction** error, not by error magnitude. A compressed run accumulates that direction step after step, which is why a5's drift grew as roughly t^2.65. One clean unbiased gradient every N steps partially cancels the accumulated direction.
- The two endpoints are the only things ever tested: always compressed (the incumbent and all five round-A arms) and never compressed (the dense control). **The duty cycle in between has never been run.**
- It is simpler than every mechanism in the round-A matrix: no snapshots, no RELEX, no signed EMA, no controller, no extra hyperparameters beyond N.
- It is distinct from the anchor. The anchor replays a batch densely to compute a **correction** to the compressed gradient. This idea just takes a normal uncompressed optimizer step.

## What it would cost, and the open question

Charged against the per-step wire budget, a dense pass sends 1536 numbers instead of 77:

| dense every | average bits/token/boundary | vs the incumbent's 1232 |
|---|---|---|
| 50 steps | 1699 | 1.38x |
| 100 steps | 1465 | 1.19x |

The deployment premise in `CLAUDE.md` does permit "a periodic slow sync ... dense weight sync or dense passes, for example in a central GPU mesh", and if the pass runs in the central mesh it never crosses the constrained internet link. We already apply exactly that accounting to the anchor, which does a full dense replay every 20 optimizer ticks and is never charged to the wire budget. **Whether this idea is free or costs 19 to 38 percent depends on that placement decision, and that should be settled before it is run, not after.**

Standing constraint from the operator: **the fast circuit should never run a dense forward and backward, it is too expensive.** So if this is implemented, the full-fidelity step belongs in the slow or central-mesh circuit, not in the fast path.

## Implementation note

There is currently **no per-step on/off gate for the codec anywhere in the code.** The masker fires on `path_tag`, i.e. which forward pass it is, and never on the step number. Every existing cadence knob gates something else: `anchor.cadence`/`delay_K` 20/20 (paired dense replay for the correction machinery), `frlr_q_cadence` (basis refresh), `spectral.cadence` (signed EMA), `probe_every` (measurement only). So this needs a new step-cadence condition on the codec firing gate, which carries assertions confining masking to specific forward paths and therefore needs care rather than a one-line change.

## Suggested test if it is ever taken up

Incumbent PRF exact-k config, 600 steps, identical to #90, with the full-fidelity step at N = 50 and N = 100 as two arms. Read against #90's own curve: does the gap stop rising (13.88 to 14.66 becomes flat) and does reference KL come in below 0.91 at step 600, at unchanged validation accuracy. That is a direct A/B against a run we already have in full.
