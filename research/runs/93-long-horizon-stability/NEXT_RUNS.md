# The next two runs, locked 2026-07-26T11:30Z

Decided after a8's registered window came in. Only two, and the second is
conditional on the first. Everything else is explicitly dropped below.

## Run 1: anchor-owned FRLR, 200 steps, val at 200

**Why this one.** It is the configuration we would actually ship, per the operator's
constraint that Q methods must live in the anchor. It is also the limit case of what
a8 just validated: refresh Q rarely, from a large accumulated sketch. a8 flattened
the gap slope **13x** (+0.016351 to +0.001262) at equal-or-better learning, and its
gap is still falling monotonically (11.7151 to 8.0304 to 6.8293).

**Second, independent benefit.** If Q lives in the anchor its broadcast rides the
slow circuit, which this program already does not charge to the wire budget (the
anchor's dense replay is not charged either). That erases the 1233.4-against-1232
discrepancy from the wire-budget correction, so FRLR regains exact parity with PRF.

**This needs a code change**, not a flag. Two parts, both mirroring an existing
pattern:

1. `verl/workers/config/comm_eff.py:738` blanket-rejects `anchor.owns_q` for
   `compression_type='prf_mask'`, on the stated grounds that "the PRF activation mask
   has no PowerSGD basis Q for the anchor to own". **That premise is false when
   `mask.frlr=true`**, which does carry a basis. Relax the check to reject only when
   FRLR is off.
2. `verl/workers/comm_eff/activation_mask.py` has no ownership plumbing at all; only
   `powersgd_activation.py`, `activation_quant.py` and `state.py` read `owns_q`. Port
   PowerSGD's `_should_accumulate_sketch` logic into the FRLR sketch path: accumulate
   only inside the anchor's stale-weight forward, never on the fast path, and refresh
   the basis at anchor fires.

Config otherwise identical to a7: FRLR r48/k28, no token-IS, probe cadence 5,
`TEST_FREQ=200`, `SAVE_FREQ=200`. Anchor stays at cadence/delay 20/20.

**Early-kill triggers**, per the operator's instruction not to wait for 200 steps:

- kill if score at 41-60 is below **0.40** (a6's failure signature)
- kill if the gap exceeds **12** at step 60, since it then cannot beat PRF on level
- kill if the gap slope at 61-80 exceeds **+0.016** (a7's failing value), since the
  whole point is a flatter trend

## Run 2, conditional: 600 steps of the winner

**Why.** PRF is the only arm proven at 600 steps. Everything else in this program is
a 200-step result. The single remaining question against the incumbent is durability,
and it cannot be answered at 200.

**Which arm.** Whichever of a7, a8 or run 1 has the flattest gap trend at
equal-or-better learning. On current evidence that is a8 or run 1, not a7.

**Config.** 600 steps, val at 0/300/600, probe cadence 5, `SAVE_FREQ=200`, R2 sink
ON (checkpoints from a 20-hour run must not be local-only).

**Early-kill triggers.**

- kill if the gap crosses the incumbent's **14.3**, since the codec has then lost its
  only advantage
- kill if val at 300 is below **0.65**, which is below every incumbent checkpoint

Cost about 20 GPU-h. This is the run that decides what goes in the paper.

## Explicitly dropped, with reasons

| dropped | why |
|---|---|
| further Q-cadence sweeps | operator instruction; a8 already gave the cadence-20 datapoint and run 1 tests the limit case |
| FRLR unbiased mode | demoted from run 1 to unscheduled. a8 is **still the biased variant** and its trend flattened anyway, so bias is not the main driver. Revisit only if run 1's trend still rises |
| incumbent + cadence-5 probe | would explain *why* but capability already decides *which*. Not decision-critical |
| round B controller | nothing in the evidence points at needing an adaptive KL coefficient |
| periodic dense forward+backward | operator rejected; note only |
| error feedback | structurally inapplicable to activation compression: GRPO draws fresh rollouts each step so there is no persistent object to carry a residual on |

## Teardown

**No standing authorization. Ask explicitly.** The trigger to ask is run 1 finishing
with run 2 not approved, or run 2 finishing. Before any teardown, push checkpoints to
R2: they are currently local-only for a5b, a6, a7 and a8 at
`/workspace/verl/checkpoints/93-long-horizon-stability/<cell>/global_step_*`, about
19 GB each, and are lost with the box. Also capture each cell's step-N metrics from
the on-box log, because WandB drops the final step.
