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

## Standing operating rules for this program

- GPU busy outranks everything. On a non-STOP verdict launch the next cell
  immediately, target under 15 min gap. Fable consults happen AFTER the launch.
- One cell at a time. Never a bare `ray stop`. Never print secrets.
- Nothing launches past a STOP. Ambiguous equals REVISE, amend only the next
  step, `needs:human` before any new spend.
- Box teardown needs explicit operator authorization. There is NO standing grant.
