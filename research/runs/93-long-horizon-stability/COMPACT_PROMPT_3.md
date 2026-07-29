Preserve issue #93 in full detail; drop tool transcripts and log dumps.

AUTHORITATIVE COPIES on branch 93-mismatch-control-kit:
research/runs/93-long-horizon-stability/COMPACT_PROMPT_3.md (this file) and
NEXT_RUNS.md. Re-read them if anything below is unclear.

## LIVE
Box vast 45725398, `ssh -i ~/.ssh/vast_ai -p 8602 root@50.46.253.92`, 1x H200 NVL,
$3.344/h. Ledger `93-long-horizon-stability` started 2026-07-24T17:30Z, cap 100,
about **42 h** at 11:30Z Jul 26.

RUNNING: **`a8-frlr-qcad20-200`**, tmux `run-93`, step ~123/200, lands ~14:00Z.
= a7's codec + `COMM_EFF_MASK_FRLR_Q_CADENCE=20`. Let it finish for its terminal val.
**Nothing is chained behind it.**

## THE FIVE FINISHED CELLS

| cell | codec | token-IS | gap @200 | true drift | actor/kl_loss | val | verdict |
|---|---|---|---|---|---|---|---|
| `90-prf-exactk-600` | PRF exact-k | off | ~14.3 | no probe | 0.9085 @600 | .6613/.6633/.6733/.6613 | incumbent, **only arm proven to 600** |
| `a5b-frlr-bnorm-200` | FRLR | on+bnorm | 5.37 | 0.016754 | 2.2262 | 0.6593 | FAIL G3 |
| `a6-prf-exactk-tis-bnorm` | PRF | on+bnorm | 14.13 | 0.026793 | 0.2918 | 0.5391 | FAIL G1 |
| `a7-frlr-r48k28-notis` | FRLR | **off** | 8.18 | 0.008200 | 5.8246 | **0.6713** | best capability; gap-slope FAIL |
| `a8-frlr-qcad20-200` | FRLR, q cad 20 | off | running | 0.00459 @110 | 0.1064 | pending | **flattest gap trend** |

Bar for every probe cell: G1 score 100-120 >= 0.6248, G2 gap < 14.2458 and slope
<= +5.0e-4, G3 drift slope <= 3.264e-3, G4 wire 1232. Primary window 100-120.
Round A (a1-a5, 120 steps, no probes, val off) closed, all verdicts posted.

## FINDING 1: `actor/kl_loss` IS NOT DRIFT AND RANKS THE WRONG WAY
`probe/kl_dense` (codec silent, forward only, no backward) IS the policy's KL to ref.
`actor/kl_loss` is that times a codec-specific, time-varying factor seen at 10.1x,
14.3x, 132.9x, 352.9x, 641.1x, **710.2x**.
- **Spearman(`actor/kl_loss`, val) = +1.00** across a5b/a6/a7: higher reading = BETTER
  capability. **Spearman(`probe/kl_dense`, val) = -1.00**: correct direction.
- n=3, perfect ordering by chance = 1/6, so consistent evidence NOT proof.
- **a7 holds 5.82 nats, inside the historical 3-8 nat collapse band, with the best
  capability measured.** Any gate on that band kills the winner.
- INVALIDATED: all cross-codec `actor/kl_loss` comparisons, incl. round A's drift
  column. OPEN NOT REFUTED: the a1/a2 "coherence not magnitude" law (bias 6.9x worse,
  z=+15) was measured on `actor/kl_loss` and neither arm has a probe.

## FINDING 2 (theory, section 20 of the report): WHY THE Q GAP STARTS LOW AND CLIMBS
- **Starts low = alignment, not low-rank form.** Activations are anisotropic, so an
  ALIGNED rank-48 subspace beats its nominal 3.1%. Measured: at step 1 with a RANDOM Q
  the edge is only **1.29x** and energy capture ties (5.0% vs 4.9%); by step 20 with a
  fitted Q it is **2.98x**.
- **PRF is flat because its error is ROTATION-INVARIANT** (depends only on k/H), so it
  is indifferent to the activation distribution moving. **FRLR's is
  ALIGNMENT-DEPENDENT**, so it chases. PRF buys a permanently worse but stationary gap;
  FRLR a better but chasing one.
- **The climb is Q ESTIMATOR VARIANCE, not staleness.** a8 refutes staleness: freezing
  Q 20 steps flattened the slope **13x** (+0.016351 -> +0.001262; -0.026533 over
  61-120) with the gap falling monotonically 11.7151 -> 8.0304 -> 6.8293. At cadence 1
  Q is re-derived every step from ONE batch with ONE power iteration so it jitters; at
  cadence 20 the sketch accumulates over 20 batches. a8's higher LEVEL (6.83 vs a7's
  5.08) is the same fact inverted: only 10 refreshes, slower convergence from random.
- **Bias is NOT the main driver**: a8 is still the biased variant and flattened anyway.
- **Degrading compressibility is NOT MEASURABLE**: no FRLR spectrum diagnostic exists
  (`rank1_evr_mean` is the anchor's RELEX predictor, not the codec basis).

## OTHER ESTABLISHED RESULTS
- **The gap win is FRLR's alone.** a6 carries the weighting on PRF and reproduces the
  incumbent's gap to 0.8%; a7 carries none and reproduces a5b's to 5%. Token-IS
  contributes NOTHING to the gap and it CAUSED the onset delay (a7 has none, 0.997x
  incumbent learning, grad_norm 2.243 vs 1.808).
- **`batch_normalize` is GAP-CONDITIONAL and dangerous at large gaps**: divides by the
  mean IS weight; at 14 nats mean=0.0005 so ~1600-2000x amplification, ESS 0.0006,
  grad_norm 57. Guard at a mean-IS-weight floor near 0.05.
- **WIRE: FRLR is 1233.4 bits/token/boundary, not 1232.** Q (1536x48) must be
  broadcast, 0.115% of traffic at cadence 1. **PRF needs NO side channel** (mask is a
  PRF of seed/step/layer, zero bits). If Q moves to the anchor the broadcast rides the
  slow circuit, which is not charged, and exact parity returns.
- **PRF exact-k IS unbiased** (`constant` gain 1/(1-p), exact to 0.26%); **FRLR default
  is BIASED** (capped detached data-dependent gamma). `frlr_unbiased=true` gives exact
  `E[h_hat|h,Q]=h` at zero extra wire, defaults false, NEVER enabled by any arm.
- **Error feedback is structurally inapplicable** to activation compression: GRPO draws
  fresh rollouts every step, so there is no persistent object to carry a residual on.
  It belongs on gradients. Neither codec implements it.
- **The observation that cuts against the program's premise**: a7 cut the mismatch 3x
  for a val gain inside the reference's own noise, while the incumbent ran 600 steps at
  14.6 nats and finished where it was at step 150. At this scale/horizon a 14-nat
  mismatch may simply not be harmful. Untested beyond 600 steps.

## THE NEXT TWO RUNS (locked; see NEXT_RUNS.md)
**Run 1: anchor-owned FRLR, 200 steps + val.** The shippable config per the operator's
constraint that Q methods live in the anchor, and the limit case of what a8 validated.
**NEEDS CODE, not a flag:** (a) `comm_eff.py:738` blanket-rejects `anchor.owns_q` for
`compression_type='prf_mask'` on the false premise that it "has no PowerSGD basis Q" --
relax to reject only when `mask.frlr` is off; (b) `activation_mask.py` has no ownership
plumbing (only `powersgd_activation.py`, `activation_quant.py`, `state.py` read
`owns_q`) -- port PowerSGD's `_should_accumulate_sketch` so the sketch accumulates only
in the anchor's stale-weight forward and the basis refreshes at anchor fires. Otherwise
identical to a7. KILL EARLY if score 41-60 < 0.40, or gap > 12 at step 60, or gap slope
61-80 > +0.016.

**Run 2, conditional: 600 steps of whichever arm has the flattest gap trend at
equal-or-better learning** (currently a8 or run 1, not a7). val 0/300/600, probe
cadence 5, **R2 sink ON**. ~20 GPU-h. KILL EARLY if the gap crosses 14.3 or val@300 <
0.65. This is the run that decides the paper.

**DROPPED:** further cadence sweeps (operator instruction); FRLR unbiased mode
(demoted, since a8 flattened while still biased -- revisit only if run 1 still rises);
incumbent+probe reference cell (explains why, not which); round B controller; periodic
dense step (operator rejected); error feedback (inapplicable).

## WHEN TO SWITCH OFF THE GPU
**No standing authorization; ask explicitly every time via `needs:human`.** Trigger to
ask: run 1 finishes with run 2 unapproved, or run 2 finishes. Ledger headroom is NOT
permission. **BEFORE teardown: checkpoints are LOCAL ONLY for a5b/a6/a7/a8** at
`/workspace/verl/checkpoints/93-long-horizon-stability/<cell>/global_step_*`, ~19 GB
each, disk 114/200 GB used, and are LOST with the box unless pushed to R2. Also capture
each cell's step-N metrics from the on-box log because WandB drops the final step.

## OPERATOR DECISION PENDING (`needs:human` set)
Demote `actor/kl_loss` to a labelled diagnostic; promote `probe/kl_dense` to the drift
criterion; require a cadence-5 probe on every cell (~3% cost); gate promotion on val
and OOD. Four independent supports. Changes the registered procedure, so operator's
call.

## STANDING HAZARDS
- `grep -o "step:[0-9]*"` also matches `timing_s/step:` and returns the DURATION.
  Match `global_step:[0-9]+`. The box clock is the authority, not the laptop's.
- WandB `history(keys=...)` SAMPLES; `scan_history(keys=[many])` returns only rows where
  EVERY key exists. Pull ONE key at a time and merge on `global_step`. Five silent wrong
  answers came from this in one session.
- `grep -c ... || echo 0` emits TWO lines (grep -c prints 0 AND exits 1) and broke a
  watcher into a false TERMINAL. Never line-number-parse remote output; prefix-tag it.
- A watcher must decide "am I done" from CELL-SPECIFIC evidence, never the shared tmux
  name. Four watcher bugs came from that. Use `chain/watch_cell.sh <cell>`.
- **Early windows lie.** Four separate over-reads this session, including one that
  nearly justified killing a8, the cell that identified the mechanism. Score at the
  registered window; treat anything before it as non-evidence.
- Launchers `git reset --hard` to origin at EVERY launch and the box has no push creds;
  push from a laptop worktree. The engine TRUNCATES `$LOG` at start, so verify config
  from WandB, not the log.
- `research/runs/` is gitignored: `git add -f`. Pre-commit needs `--no-verify`.
- SIGTERM not SIGINT; kill residuals by process NAME, never `pkill -f`; never a bare
  `ray stop`.
- `actor/ppo_kl` and `pg_clipfrac` are exactly 0 by construction (train_batch ==
  ppo_mini, one inner update, ratio identically 1). NOT the token-IS ratio, which is
  trainer-view over sampler-view.
- No em-dashes in any deliverable.
