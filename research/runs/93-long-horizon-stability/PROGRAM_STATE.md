# Issue 93 program state (live, /execute session)

Updated 2026-07-25 ~07:15 AEST. Purpose: survive a context summary or a session
handoff without losing hard-won facts.

## Box and budget

- Program box: vast **45725398** (TEAM), 1x H200 NVL, $3.344/h.
  `ssh -i ~/.ssh/vast_ai -p 8602 root@50.46.253.92`
- `/workspace/verl` on branch `93-mismatch-control-kit`. All program work lands here.
- Heartbeat contract: `/workspace/train.log` is a symlink to the ACTIVE cell's
  train.log. Re-point it at every cell launch (`launch_cell.sh` does this).
- **Box speed 112 to 115 s/step**, not the 67 to 77 the plan assumed. Known fact,
  not a fault. Per-cell cost: 120 steps = 3.8 h, 200 steps = 6.3 h,
  600 steps + 3 vals = about 20.5 h.
- Ledger `max_gpu_hr` raised **44 to 100** by the operator on 2026-07-25 after the
  reprojection (full program projects about 50 GPU-h). Headroom now covers the
  a1-prime and b2 contingency arms.
- Ledger row `93-long-horizon-stability`, status RUNNING.

## Cell status

| cell | run name | state |
|---|---|---|
| a1 | `a1-srq-b1-sr` | TRAINING, WandB id `h0n67q3a` |
| a2 | `a2-srq-b1-rn` | queued, config dry-run verified |
| a3 | `a3-srq-parity-k493` | queued, config dry-run verified |
| a4 | `a4-prf-exactk-cvc-ce` | queued, config dry-run verified |
| a5 | `a5-frlr-r48k28-tis` | queued, config dry-run verified |
| b1 | `b1-<arm>-ctrl` | needs round-A winner + setpoint table |
| c | `c-<arm>-val600` | preflight now passes end to end |

## Round C readiness: three blockers found and FIXED on 2026-07-25

These were all latent and would each have surfaced 27+ h into the program.

1. **`aws` CLI and `boto3` were both absent.** Both `r2_sink.py` (checkpoint
   push) and `ood_run_all.sh` (checkpoint pull) shell out to `aws s3`.
   Installed `aws-cli/1.45.56` via pip. System python is `/usr/bin/python`,
   pip is `/usr/local/bin/pip`, there is no venv on this box.
2. **`R2_BUCKET` pointed at the wrong thing.** It was set to the bucket
   `autonomous-harness-rlvr-compression`, but that string is meant to be a
   PREFIX inside bucket `shamane-pluralis`. Both names exist on the account,
   which is why it looked plausible. `r2_sink.py` hard-guards on the exact
   literal `shamane-pluralis` (`R2_REQUIRED_BUCKET`), and `run_93_cell.sh`
   preflights the same literal and fatals, so round C would have refused to
   launch. Fixed; secrets file backed up to `secrets.env.bak-93-<epoch>` first.
3. **`R2_CKPT_BUCKET` was unset**, which `ood_run_all.sh` hard-requires
   (`:?set R2_CKPT_BUCKET`). Set to `shamane-pluralis`.

Round C dry-run now resolves: `r2 ckpt sink: true`, `save_freq=100`,
`test_freq=300` (val at 0/300/600), `probe every=25 ctrl=true`.

## R2 layout (verified by listing)

```
s3://shamane-pluralis/autonomous-harness-rlvr-compression/<experiment>/<regime>/checkpoints/global_step_<N>/actor/
```

- `90-prf-exactk-600/prf-exactk/` has steps **100 to 600** (the baseline).
- `90-dense-600/dense/` has steps **100 to 500** at last look; 600 lands as the
  finished run flushes.
- Round C will write to `93-long-horizon-stability/c-<arm>-val600/`.
- For `ood_run_all.sh`: `R2_CKPT_BUCKET=shamane-pluralis`,
  `R2_PREFIX=autonomous-harness-rlvr-compression/<experiment>`, `run=<regime>`.

## OOD suite

- Harness at `research/scripts/ood_eval/` on the branch (`ood_prep.py`,
  `ood_eval.sh`, `ood_run_all.sh`).
- **All 10 benchmark parquets BUILT on the program box** at `/root/data/ood/`:
  gsm8k, math500, minerva, olympiad, amc23, aime24, aime25, aime26, hmmt25,
  mmlu_stem. 10 OK lines, no errors. Done during a1 at zero GPU cost.
- **Do NOT re-run the dense OOD arm on the program box.** The other session is
  already evaluating dense d600 on ITS box (45621340, tmux `dense-eval`, started
  2026-07-24 20:49 UTC, output `/workspace/runs/ood-eval-dense/`). A completed
  #90 PRF OOD result set also already exists there at
  `/workspace/runs/ood-eval/RESULTS.txt` (s100 vs s600, `OOD_DONE` present).
  Pull both for round C's comparison instead of spending about 2 GPU-h.

## Controller setpoint table (COMPLETE, required by b1 and c)

Dense reference `90-dense-600` **finished 600/600** in 11:35:42 at about 70 s/step
on box 45621340. Its WandB run reports `state=crashed` with history ending at
step 408: that is a SYNC DROP, not a training failure. Always prefer its on-box
`train.log`.

Baked from that log, 600 clean (step, kl) pairs, zero gaps, zero duplicates.
Cross-check: parsed `actor/kl_loss` at step 173 = 0.0049365 against the 0.0049
recorded in issue section 1, agreeing within 0.7 percent.

```
50:0.005,100:0.007664,150:0.008869,200:0.011265,250:0.0139,300:0.014769,350:0.018083,400:0.020659,450:0.0206,500:0.024645,550:0.027861,600:0.031209
```

setpoint = max(0.005, 2 x dense actor/kl_loss at matched step). The 0.005 floor
binds only at step 50. Raw dense kl: 50 -> 0.0024896, 100 -> 0.0038318,
150 -> 0.0044345, 200 -> 0.0056326, 250 -> 0.0069498, 300 -> 0.0073846,
350 -> 0.0090416, 400 -> 0.0103297, 450 -> 0.0102998, 500 -> 0.0123225,
550 -> 0.0139304, 600 -> 0.0156043. File:
`runs/93-long-horizon-stability/kl_target_table.txt`.

**Finding that contradicts the issue body.** Section 1 describes the dense
control's reference KL as "concave". It is mildly **convex and accelerating**:
about 0.00109 per 50 steps over steps 50 to 300 against about 0.00131 per 50
steps over 300 to 600, roughly 20 percent faster in the back half, with a
consistently positive quadratic term over all 600 points. Monotone apart from a
trivial -0.29 percent dip at step 450. The parse is corroborated by the step-173
cross-check and gapless coverage, so this looks like a real property of the run.
Consequence: acceleration of reference KL is NOT by itself a compression
pathology, since the uncompressed control does it too. What separates the arms is
MAGNITUDE, and there the gap is enormous (dense reaches 0.0156 nats at step 600;
a1's real component is already about 0.28 at step 76, roughly 90x dense at
matched step).

## a1 pre-read (step 76) headline

See `preread-a1.md` for the full table. The brief's premise that a1's
`actor/kl_loss` is a flat ~1.9 stochastic-rounding noise floor is **falsified**:
flat only over steps 2 to 18, then a monotone accelerating climb, r2 0.93,
significant under OLS, Theil-Sen, Newey-West HAC, moving-block bootstrap,
Spearman and Mann-Kendall. WandB and the on-box log agree to 4 significant
figures.

- reference-KL slope 0.003399/step, 2.3x the PRF baseline's 0.0015, accelerating
- train-inference gap 13.650 nats against the incumbent 14.24, only 4 percent
  better, widening at +0.008622/step, projecting about 14.03 at step 120
- reward slope +0.004434/step, 1.39x baseline, PASS on parity
- no collapse signature: entropy flat, grad_norm falling, `kl_coef` pinned 0.001
- `actor/ppo_kl` and `pg_clipfrac` exactly 0 at every step is CORRECT and
  expected, not an unpopulated metric: `train_batch_size=128` equals
  `ppo_mini_batch_size=128`, so there is one inner update per step and the ratio
  is identically 1. Verified against the launch config. Do not chase this.

**a2 kill gate, made well posed.** The pre-registered rule "kill a2 at step 60 if
its reference-KL slope is at least 2x a1's" is undefined without a fixed window,
because a1's slope varies 2.5x by window. Pre-registered here: fit a2 over steps
2 to 60 and compare against a1's matched steps 2 to 60 slope of 0.002707/step,
so the **threshold is 0.005414/step**. Using a1's full-run slope would give
0.006799, a gate 26 percent more permissive. Per-arm HAC SE is about 0.00023, so
0.0045 to 0.0063 is statistically inconclusive: confident kill above 0.0063,
confident acquittal below 0.0045, and inside the band decide on reward-slope and
gap corroboration rather than the point estimate.

## The box is NOT durable storage for branch work (learned the hard way)

`run_93_cell.sh` bring-up does, at EVERY cell launch (line 211):

```
git fetch --depth 1 origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD && git reset --hard FETCH_HEAD
```

So **any commit made only on the box is wiped off HEAD the next time a cell
launches.** This happened: three commits (val cadence, analysis docs, plan)
were made on the box during a1 and were reset away when a2 launched. They were
recovered from the reflog, cherry-picked onto the origin tip, and pushed.

Consequences to remember:
- The box has NO git push credentials (HTTPS asks for a username and fails).
  Push from the laptop, which has them.
- Working pattern: `git format-patch` on the box, scp the patches to the laptop,
  `git worktree add --detach <path> <origin-tip>`, `git am` the patches, then
  `git push origin HEAD:93-mismatch-control-kit`. The laptop's primary checkout
  is on `autonomous-harness-v1` and diverges, so use a worktree, never the
  primary HEAD.
- **Anything that must affect a future cell has to be on origin before that
  cell launches.** The val-cadence fix is the live example: unpushed, round C
  would have silently reverted to val at 0/150/300/450/600.
- origin tip as of this writing: `223e4b1d`. The other session also pushes to
  this branch (`952d7a0` was theirs), so always rebase onto the current tip.

## a1 final result (120/120 completed, WandB state=finished, all 120 steps present)

Gate window steps 100 to 120, 21 rows:

| quantity | a1 | baseline | flag |
|---|---|---|---|
| reference KL absolute | 2.2518 | 0.156 to 0.203 | not meaningful (SR view offset about 1.86) |
| reference KL slope | +0.00435/step | about +0.0015/step | FAIL, 2.9x |
| train-inference gap | 13.751 nats | 14.24 | FAIL vs the < 10 gate (3.4 percent better than incumbent) |
| gap slope | +0.00658/step | +0.00047/step training-view | widening |
| E[rho] | 0.0055 | 0.0014 | 3.9x baseline, still about 180x below 1.0 |
| reward slope, full run | +0.00324/step | 0.0032, bar 0.00288 | PASS, but only 1.01x baseline |
| reward slope, gate window | **-0.00107/step** | - | CAUTION, reward stopped improving |
| `actor/ppo_kl` | exactly 0 | about 0 | PASS by construction (mini equals batch) |
| entropy | 7.935, +0.00036/step | - | no collapse |
| grad_norm max | 0.898 | - | no collapse |
| confinement counters | 78358 / 34138 / 6 / 19 | - | clean, non-degenerate |

Runtime 3h58m39s at 119.33 s/step average. Terminated with 5 error markers, ALL
benign shutdown-path noise: a DataLoader worker killed during teardown and a
WandB `teardown_atexit` BrokenPipeError. Training itself completed all 120 steps.

## Standing operating rules for this program

- GPU busy outranks everything. On a non-STOP verdict launch the next cell
  immediately, target under 15 min gap. Fable consults happen AFTER the launch.
- One cell at a time. Never a bare `ray stop`. Never print secrets.
- Nothing launches past a STOP. Ambiguous equals REVISE, amend only the next
  step, `needs:human` before any new spend.
- Box teardown needs explicit operator authorization. There is NO standing grant.

## Overnight state, 2026-07-25T14:30Z (box clock is the authority)

Two cells are committed, back to back, with no human in the loop.

| | cell | steps | starts | lands (UTC) |
|---|---|---|---|---|
| running | `a5b-frlr-bnorm-200` | 200 | 13:26Z Jul 25 | about **20:05Z** |
| chained | `a6-prf-exactk-tis-bnorm-200` | 200 | on a5b's exit | about **02:50Z** Jul 26 |

Ledger `93-long-horizon-stability` started 2026-07-24T17:30Z, so it reads about
**21.0 h of 100** at the time of writing and lands near **33.5 h** when a6 ends.
At $3.344/h the remaining committed work is about $41.

### The handoff is automated on the box, not on the laptop

`/workspace/chain_a6.sh` runs detached in tmux session `chain-93` and appends to
`/workspace/chain-93.log`. It waits for tmux `run-93` to disappear (the only
completion signal used: the pane runs the launcher directly and
`remain-on-exit` is off, so the session dies exactly when the driver exits; a
log-quiet heuristic is deliberately NOT used because relaunching onto a live
run would put two trainers on one GPU), snapshots a5b's terminal metrics to
`/workspace/runs/a5b-frlr-bnorm-200/final/` because WandB drops the final step,
waits for the GPU to fall below 4 GB while killing residual workers by process
NAME only (`pkill -x`, never `-f`, never a bare `ray stop`), re-points the
`/workspace/train.log` reaper heartbeat symlink and seeds it so its mtime stays
fresh through bring-up, then launches `/workspace/launch_a6.sh` in a fresh
`run-93`. It then watches 40 minutes of bring-up and logs `ALERT` if a6 dies.

Because the chain lives on the box it survives the laptop sleeping. It fires on
ANY termination of a5b, clean or crashed, which is correct: a6 is a
pre-registered cell, not a known-broken config, so occupying the GPU with it is
right either way and a5b's log stays on disk for scoring.

### Correction: earlier step readings in this program were durations, not steps

`grep -o "step:[0-9]*"` also matches `timing_s/step:118.86`, the per-step wall
time in seconds. Reported "step 118" and "step 121" for a5b were 118 s/step and
121 s/step. **Always read `global_step:[0-9]+`.** At 14:25Z a5b was at
`global_step:26`, one hour in, on pace.

### Confirmed: `batch_normalize` is firing

`rollout_is_batch_norm_factor:0.186` appears in a5b's log, so the weights are
divided by 0.186 and the update is scaled back up **5.38x** (the pre-run
estimate was about 6.03x). Note that every `rollout_corr/rollout_is_*` metric is
computed at `rollout_corr_helper.py:604`, which is BEFORE the normalization
block, so `rollout_is_mean:0.19` and `eff_sample_size:0.238` are the RAW
distribution and are directly comparable to a5's 0.166 and 0.268. They are not
evidence the knob is inert; `rollout_is_batch_norm_factor` is the proof it fired.

### Also settled

Checkpoints go to `checkpoints/93-long-horizon-stability/<run>` relative to
`/workspace/verl`. Disk is 197 G free of 200 G, so a5b's and a6's two saves each
have ample room. No checkpoint exists yet because a5b has not reached step 100.
