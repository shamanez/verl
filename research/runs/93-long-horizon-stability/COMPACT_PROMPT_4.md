Preserve issue #93 in full detail; drop tool transcripts and log dumps.

AUTHORITATIVE COPIES on branch `93-mismatch-control-kit`, under
`research/runs/93-long-horizon-stability/`: this file, `NEXT_RUNS.md`, and
`verdict-a7.md` through `verdict-a10.md`. Re-read them if anything below is unclear.

## LIVE
Box vast 45725398, `ssh -i ~/.ssh/vast_ai -p 8602 root@50.46.253.92`, 1x H200 NVL,
$3.344/h. Ledger `93-long-horizon-stability`, cap 100 GPU-h, about **52 h** used.

**RUNNING: `c600-a9-anchorq-val600`**, tmux `run-93`, launched 2026-07-26T23:45:53Z,
lands about **19:00Z on 2026-07-27**. a9's config (anchor-owned FRLR r48/k28, biased
norm-matching, no token-IS) at **600 steps**, val at 300/600, probe cadence 5,
`SAVE_FREQ=200`, **R2 sink deliberately OFF**. Launcher `/workspace/launch_c600_a9.sh`.
Kill triggers: gap crossing the incumbent's **14.3**, or **val@300 below 0.65**.
**NOTHING is chained behind it**; `chain-93c` has retired.

## THE SEVEN CELLS

| cell | codec | gap | codec-free drift | `actor/kl_loss` | terminal val | verdict |
|---|---|---|---|---|---|---|
| `90-prf-exactk-600` | PRF exact-k | ~14.3 | no probe | 0.9085 @600 | .6613/.6633/.6733/.6613 | incumbent, only arm proven to 600 |
| `a5b` | FRLR + IS + bnorm | 5.37 | 0.016754 | 2.2262 | 0.6593 | FAIL G3 |
| `a6` | PRF + IS + bnorm | 14.13 | 0.026793 | 0.2918 | 0.5391 | FAIL G1 |
| `a7` | FRLR, fast Q cad 1 | 7.7618 @199 | 0.008200 | 5.8246 | **0.6713** | best-tied capability; gap slope FAIL |
| `a8` | FRLR, fast Q cad 20 | 7.2249 @199 | 0.007006 @150 | 0.1064 | 0.6613 | wins the REGISTERED slope, but see U-curve |
| `a9` | FRLR, **anchor-owned Q** | **7.0031 @199** | 0.008594 | 4.1330 | **0.6713** | flattest LATE slope, lowest terminal gap |
| `a10` | a9 + **unbiased** | 14.88 @41-60 | 0.002419 @60 | 0.1775 | n/a | **KILLED @62**, two triggers |

Bar for probe cells: G1 score 100-120 >= 0.6248, G2 gap < 14.2458 and slope <=
+5.0e-4, G3 drift slope <= 3.264e-3, G4 wire 1232. Round A (a1-a5) closed.

## THE FOUR FINDINGS

**1. `actor/kl_loss` CANNOT RANK ARMS.** a7/a8/a9 differ ONLY in how `Q` is governed
and their codec-free `probe/kl_dense` is IDENTICAL at matched steps (step 120: 0.005095
/ 0.005329 / 0.005265, within 4%; within 8% everywhere; holds to step 200), while
`actor/kl_loss` spans **55x**. a7 and a9 have terminal vals identical to the digit
(**0.6713426853707415** = 335/499). So Q governance moves the gap and the codec view
and moves NEITHER capability NOR real drift. `probe/kl_dense` is the physical channel.

**2. FRLR'S ADVANTAGE REQUIRES A BIASED ESTIMATOR** (new 2026-07-26, from a10). With
`frlr_unbiased=true` the gap is **14.8751**, i.e. the incumbent's own 14.2458. The
entire 2.4x advantage comes from the biased capped per-token norm-matching gain.
Removing the bias lowers real drift only **2-10%** (0.90x/0.98x/0.96x at steps
25/50/60), the same order as arm-to-arm noise, against a **2.56x** gap penalty.
Mechanism: constant H/k is unbiased in expectation but higher per-token VARIANCE, and
the gap is a per-token KL. a10 is the only arm below wire parity (**1216 bits**, 76
coords, verified from runtime `mask_ratio` 0.9505208 = 1-76/1536). It bought 16 bits
and paid 9 nats. **This does NOT overturn the a1/a2 factorial** (sr_quant 1-bit, where
bias is a per-coordinate rounding error, versus a10's single detached per-token scalar
on a low-rank residual): bias matters for some codecs, not this one.

**3. NO FRLR ARM SETTLES.** Late-window (150-199) gap slopes: a7 +0.038535, a8
+0.018366, a9 +0.012172. All accelerate. The registered success criterion asks for a
SETTLING gap and nothing achieves it at 200 steps.

**4. a8's REGISTERED-WINDOW SLOPE IS A TURNING POINT.** a8 kept FALLING to a minimum
of 6.1173 at step **143**, so its 100-120 window sits on the descending arm of a U and
its 121-150 slope is NEGATIVE (-0.007288). Its +0.001262 is real for that window and
also the bottom of a curve that then rises at +0.018366. **Do NOT rescore on the late
window** (the registered criterion is 100-120 and a8 wins it); the point is that a8's
number extrapolates badly. Full slope table in `verdict-a9.md`.

## CORRECTIONS I MADE TO MY OWN CLAIMS (do not re-derive the wrong versions)
- "a8 is the best cell / best learning in the program": its terminal val (0.6613) is
  BELOW a7's and a9's 0.6713. Window score does NOT predict held-out val.
- "The anchor-Q constraint costs 7.3x": measured at a8's turning point. a9 is at least
  tied best on late slope, terminal gap, terminal val and drift, so **the constraint's
  cost is NOT established and may be zero.**
- "a9 has the lowest codec-free drift in the program": artifact of comparing a9@120 to
  a8@150 to a7@200. **Always compare at MATCHED steps.**
- "Bias is not the driver, so the unbiased test is pointless": confused "sufficient"
  with "exclusive". The operator overruled me and the test produced finding 2.
- "The R2 back-fill will not finish": projected from a 4-minute sample at 2.2 MB/s;
  the real rate is 5.8-12.8 MB/s.

## R2 STATE: pre-teardown step 1 is DONE
**145.06 GiB verified, ZERO byte mismatches**: a5b 21 objects / 36.27 GiB (local
DELETED, independently audited against a6's local file sizes since its own reference
was gone), a6 21 / 36.27, a7 21 / 36.27, a8 11 / 18.13, a9 12 / 18.13. Local copies
kept for a6/a7/a8/a9. **c600 needs a back-fill afterwards** via
`chain/r2_backfill3.sh` with `DELETE_AFTER_VERIFY=no UPLOAD_TIMEOUT=5400`.

## R2 OPERATIONAL HAZARDS (all cost time today)
- `aws s3 sync` FAILS on the large files with **InvalidPart**. Use per-file `aws s3 cp`
  with `multipart_chunksize=256MB` (already set in `/root/.aws/config`, which also
  covers the in-training sink since `r2_sink.py` shells out to the same binary).
- **A part can SILENTLY go missing.** One upload had 46 parts where the file needs 47.
- **`aws s3 cp` has NO TIMEOUT**; one upload hung 78 min. Wrap in `timeout`, sized from
  the SLOWEST measured rate (uplink varied 2.2 to 12.8 MB/s).
- **NEVER hand-complete a multipart.** I did, and produced a corrupt object of
  12,081,588,677 bytes for a 12,350,024,133-byte file. Only the size check caught it.
- **Hang vs slow:** sample `list-parts` count AND `ps -o time= -C aws` twice. Hang =
  both flat. Slow = both climbing at ~1s CPU per 256MB part.
- **The in-training R2 sink IDLES THE GPU at teardown.** "async" is async to TRAINING,
  not to process EXIT. a9 sat at step 200 with GPU at 0% until its uploads were killed
  (11 min instead of a projected 55). A 600-step run with three saves would idle ~2.5h.
  Hence the sink is OFF for c600.

## AWAITING THE OPERATOR
1. **Teardown**: NO standing authorization, ask explicitly every time. Ledger headroom
   is not permission.
2. **Was launching run 3 acceptable?** I launched it (~20 GPU-h) on the instruction
   "do not wait for my decision, work on all the training runs, don't let the box sit
   idle", overriding a needs:human gate I had set for myself earlier the same day. It
   is killable at any time.
3. **Registered-procedure change** (still pending from earlier): demote `actor/kl_loss`
   to a labelled diagnostic, promote `probe/kl_dense`, require a cadence-5 probe on
   every cell (~3%), gate promotion on val and OOD.
4. **The obvious untested cell**, one variable from a9 and NOT scheduled: anchor-owned
   `Q` accumulating its sketch over SEVERAL anchor fires before orthonormalising. a9
   beats a8 on gap level from ONE minibatch of sketch against a8's 20 steps' worth, so
   alignment buys the level and sample size buys the slope. Nothing has both.

## STANDING HAZARDS
- `grep -o "step:[0-9]*"` also matches `timing_s/step:` and returns the DURATION.
  Match `global_step:[0-9]+`. The BOX clock is authority.
- The WandB step axis is **`training/global_step`**, NOT `global_step`; the bare name
  silently returns ZERO rows even on a finished run. And `history()` SAMPLES while
  `scan_history(keys=[a,b])` returns only rows where EVERY key exists: pull ONE key at
  a time and merge.
- **NEVER `pgrep -f` / `pkill -f` a pattern that appears in your own command line.** I
  ran `pgrep -f main_ppo` inside an ssh command containing "main_ppo", killed my own
  remote shell, and the relaunch silently never happened. Run such logic from a script
  FILE and filter by `/proc/<pid>/cmdline`. SIGTERM, never SIGINT (the ray driver
  swallows SIGINT).
- **Six watcher bugs so far**, all from deciding "am I done" off shared or mutable
  state. Latest: I fixed a watcher to key on its log's done line instead of a tmux
  name, then renamed that line. Key on evidence of your OWN completion.
- **Early windows lie.** Four short-window over-reads today. Rule now mechanical: no
  rate or trend claim from under 15 minutes of wall clock or fewer than ~10 samples,
  and no gate read before its registered window. `research/scripts/earlykill93.py`
  and `score93_bar.py` both REFUSE incomplete windows; trust that refusal.
- Launchers `git reset --hard` to origin at EVERY launch and the box has no push
  creds: push from a laptop worktree. The engine TRUNCATES `$LOG` at start, so verify
  config from WandB, not the log.
- `research/runs/` is gitignored: `git add -f`. Pre-commit needs `--no-verify`.
- `actor/ppo_kl` and `pg_clipfrac` are exactly 0 by construction (train_batch ==
  ppo_mini, one inner update, ratio identically 1). NOT the token-IS ratio.
- `a9`'s config shows `frlr_q_cadence=1` and that is **INERT** under anchor ownership;
  the fast-path branch is skipped. The disambiguator is the refresh counter (7 per
  anchor fire; cadence-1 fast-path writing would read ~700 by step 100).
- No em-dashes in any deliverable.
