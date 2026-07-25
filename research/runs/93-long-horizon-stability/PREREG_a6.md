# Pre-registration: cell a6, `a6-prf-exactk-tis-bnorm-200`

Written and committed **before the cell exists**, on 2026-07-25 (box clock),
while `a5b-frlr-bnorm-200` is still training. None of the numbers below may be
moved after the run starts.

Corrected after the fact: the first version of this line said "step 118 of 200".
That reading came from `grep -o "step:[0-9]*"`, which also matches
`timing_s/step:118.86`, so 118 was the per-step wall time in seconds. a5b was at
`global_step:26` at the time of writing. The correction changes no threshold in
this document; it is recorded because the pre-registration record must be
accurate about when it was written relative to the data.

## Why this cell exists

Arm a5 changed **two** things at once versus the #90 incumbent:

| | IS off | IS on + batch_normalize |
|---|---|---|
| **PRF exact-k** (incumbent codec) | `90-prf-exactk-600`: 600/600, no collapse, gap 13.88 -> 14.66 still rising, ref-KL 0.91 at 600 | **a6, this cell** |
| **FRLR r48 k28** | never run | `a5b-frlr-bnorm-200`, running now |

So no result from a5 or a5b is attributable: a win could belong to the codec or
to the weighting. This cell moves **only the weighting**. It is the cheapest
cell in the program and the only one that makes the 2x2 readable.

Confirmed from the launchers before writing this: the engine default is
`ROLLOUT_IS=null` (correction off, `vast_comm_eff_engine_grpo.sh:188`);
`ROLLOUT_IS=token` appears in exactly one arm block in the whole program (a5);
`run_prf_exactk_600.sh` sets no `ROLLOUT_IS` at all; and
`ROLLOUT_IS_BATCH_NORMALIZE` defaults false, so no run in this program's history
has ever had it true except `a5b`. The cell is genuinely untested.

The code comment that used to justify a5 being the sole IS arm claimed token-IS
was "measured dead on the PRF/sr_quant views". That claim is from #88 and was
**refuted** at #88's close (run `clvaf683` cut KL 40 to 78 percent and removed a
collapse with token-IS on), and every earlier token-IS test ran with
`batch_normalize=false`, the setting now suspected of being the actual problem.
The comment has been corrected in `run_93_cell.sh` rather than left standing.

## Exact config

```
ARM=a6 \
EXPERIMENT_NAME=a6-prf-exactk-tis-bnorm-200 \
TOTAL_STEPS=200 TEST_FREQ=200 SAVE_FREQ=100 \
COMM_EFF_PROBE_EVERY=25 COMM_EFF_PROBE_CTRL_ENABLED=false \
ROLLOUT_IS_BATCH_NORMALIZE=true \
bash examples/grpo_trainer/run_93_cell.sh
```

Instrumentation is **matched to `a5b` exactly** (same step count, same probe
cadence, same checkpoint cadence, same terminal val) so the two cells are
directly comparable. Codec env verified by `ARM=a6 DRY_RUN=1`: `prf_mask
enabled=true p=0.95 rescale_mode=constant exact_k=true`, `frlr: false`,
`cvc: ce_lambda=0.0`, `rollout_is: token threshold=2.0`. That is the #90
incumbent codec byte for byte plus the weighting.

Cost: 200 steps at 117 s/step, about **6.5 GPU-h** (roughly $22). Ledger
`93-long-horizon-stability` sits near 27 h of 100 when `a5b` lands.

## The bar (fixed now)

Every threshold below is an already-fixed number carried from round A. Nothing
is derived from `a5b`, so `a5b`'s outcome cannot contaminate this bar.

| id | criterion | window | threshold | source |
|---|---|---|---|---|
| **G1** learning not damaged | `critic/score/mean` level | 100-120 | **>= 0.6248** | the same registered floor `a5b` carries |
| **G2** gap improved | `rollout_corr/kl` level | 100-120 | **< 14.2458** strictly | the incumbent's own value |
| **G3** drift not worsened | `actor/kl_loss` slope | 61-120 | **<= 3.264e-3** | 1.5x the incumbent's 0.002176, the matrix-wide V1 bar |
| **G4** wire | bits/token/boundary | n/a | **= 1232** | automatic: 77 kept coordinates x 16 bits, identical to the incumbent |

G4 is guaranteed by construction and carries no information; it is listed so the
scoring table stays the same shape as every other cell's.

Windows 100-120 and 61-120 are used even though the run is 200 steps, so that
the comparison against round A and against `a5b` is at matched steps. The
100-200 extension is reported as **secondary** and may not be substituted for
the registered window.

`ESS` and `rollout_corr/ratio_fraction_low|high` are reported but not gated. If
a6's ESS lands >= 0.5 while `a5b`'s was < 0.5, the two cells differ in estimator
health as well as codec, and that must be stated rather than papered over.

## Decision rules (registered)

- **G1 and G2 and G3 all pass** -> token-IS with batch normalization is a free
  improvement on the incumbent. It becomes the round-B/C candidate, and it
  resolves the a5 confound in favour of the weighting rather than FRLR.
- **G1 and G3 pass, G2 fails** -> the weighting does nothing for the gap on a
  PRF view. If `a5b` passed G2, the gap win belongs to **FRLR**, and FRLR plus
  IS is the round-B/C candidate.
- **G1 fails** -> token-IS costs learning even with the shrinkage removed, on a
  codec that is known to learn fine. That indicts the **weighting**, not FRLR,
  and closes token-IS as a line for this program.
- **G3 fails** -> the weighting itself adds drift. Same conclusion as above,
  held more strongly.
- **a6 and `a5b` land on the same side of every gate** -> the codec choice is
  irrelevant at this budget and the program's answer concerns the weighting
  alone. This is a real possible outcome and it is publishable.

## Standing hazards for the read

- `actor/ppo_kl` and `pg_clipfrac` will be exactly 0. That is correct by
  construction (`train_batch` 128 == `ppo_mini` 128, one inner update, PPO ratio
  identically 1) and is not evidence about the IS weighting, which is a
  different ratio entirely (trainer view over sampler view).
- `rollout_corr/kl` and the codec-view entropy are **instrument readings**, not
  behaviour. Round A established that the codec inflates the entropy reading
  about 43x. No level comparison across two different codecs is admissible
  without a sampler-side cross-check, and a6 vs `a5b` is exactly such a
  cross-codec comparison. Score, response length and the dense probe are the
  codec-free channels.

## Naming note (both cells carry token-IS)

`a5b-frlr-bnorm-200` and `a6-prf-exactk-tis-bnorm-200` **both** run
`rollout_is=token`, `rollout_is_threshold=2.0` and
`rollout_is_batch_normalize=true`. Only a6's run name says `tis`, because a5b's
`EXPERIMENT_NAME` override dropped it from the a5 arm's own slug
(`frlr-r48k28-tis`). The names are asymmetric; the configs are not. Verified in
a5b's live config dump: `'rollout_is': 'token'`,
`'rollout_is_batch_normalize': True`, `'rollout_is_threshold': 2.0`.

The single difference between the two cells is therefore the **codec**, which is
what makes the pair a clean single-knob comparison in both directions:

- a6 vs the #90 incumbent isolates the **weighting** at fixed codec (PRF exact-k).
- a5b vs a6 isolates the **codec** at fixed weighting (token-IS + normalize).

## Chain preflight (fixed 2026-07-25T14:45Z)

`run_93_cell.sh` resolves the ARM at line 41 but re-syncs the checkout only at
line 243, so an arm that landed on the branch after the box last synced dies
instantly, before the fetch that would have taught it. The box was at `800feef8`
and a6 landed in `9546ae25`, so the original chain would have failed in under a
second and idled the GPU. `launch_a6.sh` now fetches and resets first, then
proves `ARM=a6 DRY_RUN=1` resolves, with a copy staged at
`/workspace/run_93_cell.a6.sh` as the fallback if the fetch cannot reach origin,
and it tees its preflight into a6's `train.log` so a failure is visible to both
the chain's bring-up watch and the laptop monitor.

## AMENDMENT 1, logged 2026-07-25T15:10Z, BEFORE a6 has any data

While proving the scoring pipeline end to end I found a provenance error in G3 of
this document. It is amended here, openly, while a6 does not yet exist.

**The error.** G3's threshold `3.264e-3` is 1.5x the incumbent's drift slope of
`0.002176`. Searching the incumbent's actual history shows `0.002176` is its
slope over window **100-120**, matching to six decimal places:

| window | incumbent `actor/kl_loss` slope |
|---|---|
| 2-120 | +0.001565 |
| 2-60 | +0.000383 |
| 61-120 | +0.002344 |
| **100-120** | **+0.002176** (the registered figure) |

This document registered that threshold against the **61-120** window, where the
incumbent's own slope is 0.002344 and a 1.5x bar would have been 3.516e-3. Round
A applied the same V1 bar at the terminal 20-step window, so 100-120 is also the
precedent.

**The amendment.** The threshold value does **not** move: it stays `3.264e-3`,
exactly as registered. The primary window for G3 and G2 is corrected to
**100-120**, which is the window the threshold was derived from and the one round
A used. The 61-120 figures will also be reported for both cells so nothing is
concealed by the choice.

**Why this is not bar-shopping.** For an arm whose drift is accelerating (a5's
slope ran +0.00164, +0.00298, +0.00462 across 61-80, 81-100, 101-120) the
terminal 20-step slope is LARGER than the 60-step slope, so testing at 100-120
against a fixed threshold is **harder**, not easier. The correction moves the bar
in the conservative direction, and both windows get reported either way.

## Tooling defects found and fixed before scoring, 2026-07-25T15:10Z

Three real defects, all of which would have corrupted the a5b and a6 reads. All
were found by running the pipeline against live partial data rather than waiting
for termination.

1. **`gate93.py` fitted the reference-KL and gap slopes over the FULL RUN**, but
   every registered bar is a windowed slope. These cells open with a large
   step-1/2 codec transient, so the full-run fit measures that transient decaying
   and can flip sign: on a5b at 20-45 the full-run fit is **-0.00649/step** while
   the matched-window fit is **+0.000512/step**. Windowed slopes are now computed
   and printed, labelled as the ones to use, with the full-run figures marked.
2. **`gate93.py` used `run.history(keys=[...])`, which both samples AND drops any
   row missing any key.** The sampling returned 13 rows for the incumbent's
   21-step gate window, enough to move a slope against a fixed threshold. The
   all-keys behaviour meant the script returned zero rows for the incumbent at
   all, because a run with the rollout correction off logs no
   `rollout_corr/rollout_is_*`. Now each key is pulled separately via
   `scan_history` (unsampled) and merged on `global_step`; the incumbent's gate
   window now yields the correct 21 rows and reproduces gap 14.246.
3. **`slope_compare93.py` took a single `--project` for both runs**, but the
   engine derives the WandB project from `WANDB_RUN_GROUP`, so the incumbent
   lives in project `90-prf-exactk-600` while every #93 cell lives in
   `93-long-horizon-stability`. Any incumbent-versus-cell comparison would have
   failed with "run not found". Added `--ref-project` and `--test-project`, each
   defaulting to `--project`.
