Preserve issue #93 in full detail; drop tool transcripts and log dumps.

AUTHORITATIVE COPY: research/runs/93-long-horizon-stability/COMPACT_PROMPT_2.md on
branch 93-mismatch-control-kit. Re-read it if anything below is unclear.

## LIVE
Box vast 45725398, `ssh -i ~/.ssh/vast_ai -p 8602 root@50.46.253.92`, 1x H200 NVL,
$3.344/h. Ledger `93-long-horizon-stability` started 2026-07-24T17:30Z, max_gpu_hr
100, reads about **41 h** at 2026-07-26T10:35Z.

RUNNING: **`a8-frlr-qcad20-200`** in tmux `run-93`, step 55/200, lands about
**17:00Z (~47 h ledger)**. Config = ARM=a7 + `COMM_EFF_MASK_FRLR_Q_CADENCE=20`,
EXPERIMENT_NAME=a8-frlr-qcad20-200, TOTAL_STEPS=200 TEST_FREQ=200 SAVE_FREQ=200
COMM_EFF_PROBE_EVERY=5 COMM_EFF_PROBE_CTRL_ENABLED=false. Launcher
`/workspace/launch_a8.sh`. WandB entity `shamanework-pl`, project
`93-long-horizon-stability`.

**NOTHING IS CHAINED BEHIND a8.** When it ends the GPU goes idle.

## FINISHED CELLS AND THEIR NUMBERS

| cell | codec | token-IS | gap @200 | true drift @200 | actor/kl_loss | val | verdict |
|---|---|---|---|---|---|---|---|
| `90-prf-exactk-600` | PRF exact-k | off | ~14.3 | no probe | 0.9085 @600 | 0.6613/0.6633/0.6733/0.6613 | incumbent, only arm proven to 600 |
| `a5b-frlr-bnorm-200` | FRLR r48/k28 | on+bnorm | 5.37 | 0.016754 | 2.2262 | 0.6593 | FAIL G3 |
| `a6-prf-exactk-tis-bnorm-200` | PRF exact-k | on+bnorm | 14.13 | 0.026793 | 0.2918 | 0.5391 | FAIL G1 |
| `a7-frlr-r48k28-notis-200` | FRLR r48/k28 | **off** | 8.18 | **0.008200** | 5.8246 | **0.6713** | **best; gap-slope FAIL** |

Round A (a1-a5, 120 steps each, NO probes, val off) is closed; all five verdicts
posted. Registered bar for every probe cell: G1 score level 100-120 >= 0.6248, G2
gap level < 14.2458 and gap slope <= +5.0e-4, G3 drift slope <= 3.264e-3, G4 wire
1232. Primary window 100-120 per amendment 1.

## THE CENTRAL FINDING: `actor/kl_loss` IS NOT DRIFT

`probe/kl_dense` = codec silent, forward only, no backward, no weight change. It IS
the policy's KL to the reference. `actor/kl_loss` = that quantity times a
codec-specific, time-varying inflation factor, observed in this program at **10.1x,
14.3x, 132.9x, 352.9x, 641.1x, 710.2x**.

- **Spearman(`actor/kl_loss`, val) = +1.00** across the three probe cells: higher
  "drift" reading goes with BETTER capability.
- **Spearman(`probe/kl_dense`, val) = -1.00**: higher real drift goes with worse
  capability, the correct direction.
- n=3, perfect ordering by chance = 1/6, so consistent evidence NOT proof.
- **a7 holds 5.8246 nats, inside the historical 3-8 nat "collapse band", with the
  best capability measured.** Any gate on that band kills the winner.
- Mechanism: PRF's mask is policy-independent so inflation FALLS (134.6x -> 10.9x);
  FRLR refreshes Q every step so it's fitted to the current policy while the FROZEN
  reference is reconstructed ever worse, so inflation RISES.

INVALIDATED: all cross-codec `actor/kl_loss` comparisons, including round A's drift
column and the V1 veto as applied across arms. Round A had no probes so it cannot be
repaired. OPEN NOT REFUTED: the a1/a2 factorial behind "coherence not magnitude"
(bias 6.9x worse at z=+15) was measured on `actor/kl_loss`; neither arm has a probe.

## OTHER ESTABLISHED RESULTS
- **The gap win is FRLR's alone.** a6 carries the same weighting on the incumbent
  codec and reproduces the incumbent's gap to 0.8%; a7 carries none and reproduces
  a5b's to 5%. Token-IS contributes NOTHING to the gap.
- **Token-IS caused the onset delay** and is not needed. a7 has no delay, 0.997x the
  incumbent's learning, grad_norm 2.243 vs 1.808.
- **`batch_normalize` is GAP-CONDITIONAL and dangerous at large gaps.** It divides by
  the mean IS weight; at PRF's 14 nats the mean is 0.0005, so it amplifies ~1600-2000x,
  ESS 0.0006, grad_norm 57. Guard: refuse it below a mean-IS-weight floor near 0.05.
- **FRLR's step-1 advantage is only 1.29x (random Q); Q fitting buys the rest** to
  2.98x by step 20. Energy capture at step 1 is identical (4.9% vs 5.0%), so the 1.29x
  is per-token norm matching vs PRF's constant rescale.
- **Wire budget CORRECTED to 1233.4 bits/token/boundary for FRLR**, not 1232: Q
  (1536x48) must be broadcast, 0.115% of boundary traffic at cadence 1. PRF needs NO
  side channel (mask is a PRF of seed/step/layer). Parity survives at 0.1%.
- **NO error feedback in either codec, and it is structurally inapplicable** to
  activation compression: no persistent object across steps to carry a residual on.
- **PRF exact-k IS unbiased** (`constant` gain 1/(1-p), `E[h_tilde]=h`, exact to
  0.26%). **FRLR default is BIASED** (capped detached data-dependent gamma).
- **The observation that cuts against the program's premise:** a7 cut the mismatch 3x
  for a val gain inside the reference's own noise, while the incumbent ran 600 steps
  at 14.6 nats and finished where it was at step 150. At this scale/horizon a 14-nat
  mismatch may simply not be harmful. Hypothesis, untested beyond 600 steps.

## WHAT IS **NOT** RUNNING AND HAS NEVER BEEN RUN
1. **Round B** (controller / adaptive KL coefficient, 200 steps). Never started. Its
   setpoint table is baked and in the issue.
2. **Round C** (600 steps + val 0/300/600 + the 10-benchmark OOD suite). Never
   started. R2 sink wired, `aws`/`boto3` installed, OOD parquets built.
3. **Incumbent + probe reference cell** (PRF exact-k, cadence-5 probe, no IS). NEVER
   RUN. This is the measurement hole: the incumbent's TRUE drift is unmeasured, so
   "does FRLR drift more or less than PRF" is currently **unanswerable**.
4. **FRLR unbiased mode** (`COMM_EFF_MASK_FRLR_UNBIASED=true`). NEVER RUN by any arm;
   defaults false. This is the actionable version of AJ's error-feedback question.
5. **a7 at 600 steps.** Never run. The durability question (does the rising gap
   actually cost anything at the incumbent's proven horizon) is untested.
6. **Periodic full-fidelity forward+backward.** Operator REJECTED; note only in
   `IDEA_periodic_full_fidelity_step.md`.

## WHAT TO RUN NEXT, IN PRIORITY ORDER
1. **Score a8 at its registered window** (100-120) when it lands. NOTE THE CONFOUND
   I built in: at cadence 20 over 200 steps Q gets only 10 power iterations vs a7's
   200, so a8 changes both view stationarity (intended) and total Q fitting (not).
   Its early gap is 2x a7's and its inflation HIGHER (376.7x vs 71.6x at step 50),
   probably an under-fitted Q. **Cadence 5 would have separated them.**
2. **FRLR unbiased mode**, one env var, tests AJ's hypothesis on our own data.
3. **Incumbent + cadence-5 probe**, closes the measurement hole and gives the first
   trustworthy drift number for the reference arm.
4. **a7 at 600 steps**, the durability test against the incumbent's proven horizon.
Each is ~6.5 GPU-h except (4) at ~20 h. All are NEW SPEND.

## WHEN TO SWITCH OFF THE GPU
- **There is NO standing teardown authorization. Teardown requires the operator to
  authorize it explicitly, every time.** Raise `needs:human` and ask.
- The trigger to ask: **a8 finishes and no further cell is approved.** At that point
  the GPU is idle and burning $3.344/h, so ask immediately rather than waiting.
- Ledger headroom is not permission: 100 h authorized, ~47 h after a8, but unused
  headroom does not imply approval for more cells.
- **Before any teardown**: confirm every artifact is off the box. Checkpoints are
  LOCAL ONLY (R2 sink off for a5b/a6/a7/a8) at
  `/workspace/verl/checkpoints/93-long-horizon-stability/<cell>/global_step_*`, about
  19 GB each, disk at 114 GB used of 200. **They are lost at teardown unless pushed
  to R2 first.** Also capture each cell's step-200 metrics from the on-box log,
  because WandB drops the final step.

## OPERATOR DECISION PENDING
Demote `actor/kl_loss` from veto to labelled diagnostic; promote `probe/kl_dense` to
the drift criterion; require a cadence-5 probe on every cell (~3% cost); gate
promotion on val and OOD. Four independent supports now. This changes the registered
decision procedure so it is the operator's call. `needs:human` is set on the issue.

## ARTIFACTS
Issue: github.com/shamanez/verl-compression-research/issues/93. Branch
`93-mismatch-control-kit`, `research/runs/93-long-horizon-stability/`:
`verdict-a1..a7.md`, `PREREG_a6.md` (+2 amendments), `PREREG_a7.md`,
`PREREAD_a5b.md`, `PREREAD_a6.md`, `PREREAD_a7.md`,
`FINDING_drift_metric_invalid.md`, `WIRE_BUDGET.md` (+Q correction),
`PROGRAM_STATE.md`, `ROUND_A_*.md`, `chain/*`. Scripts:
`research/scripts/{score93_bar.py,gate93.py,slope_compare93.py,a5_tripwire.py,roundA_table.py}`.
HTML report: `~/Documents/com-eff-RLVR/runs/93-long-horizon-program.html`, sections
0-19, live at com-eff-rlvr.pages.dev.

## STANDING HAZARDS
- `grep -o "step:[0-9]*"` also matches `timing_s/step:` and returns the DURATION.
  Always match `global_step:[0-9]+`. Box clock is the authority, not the laptop's.
- WandB `history(keys=...)` SAMPLES and `scan_history(keys=[...])` returns only rows
  where EVERY key exists. Pull ONE key at a time and merge on `global_step`. This
  produced five silent wrong answers in one session.
- `grep -c ... || echo 0` emits TWO lines (grep -c prints 0 AND exits 1). It broke a
  watcher into a false TERMINAL. Never line-number-parse remote output; prefix-tag it.
- A watcher must decide "am I done" from CELL-SPECIFIC evidence, never the shared
  tmux name. Four watcher bugs came from that. Use `chain/watch_cell.sh <cell>`.
- Launchers `git reset --hard` to origin at EVERY launch and the box has no push
  credentials; push from a laptop worktree. The engine also TRUNCATES `$LOG` at start,
  so verify config from WandB, not the log.
- `research/runs/` is gitignored: `git add -f`. Pre-commit needs `--no-verify`.
- SIGTERM not SIGINT for a clean cell stop; kill residuals by process NAME, never
  `pkill -f`. Never a bare `ray stop`.
- `actor/ppo_kl` and `pg_clipfrac` are exactly 0 by construction (train_batch ==
  ppo_mini, one inner update, ratio identically 1). Not a bug, and NOT the token-IS
  ratio, which is trainer-view over sampler-view.
- No em-dashes in any deliverable.
