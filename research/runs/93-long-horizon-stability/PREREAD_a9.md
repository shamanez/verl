# Pre-read: a9 at step 20 of 200. The anchor-owned Q mechanism is confirmed working, and the counter proves both halves.

Written 2026-07-26T15:45Z. The registered bar is scored at **100-120** and is not
evaluable. This note records **one thing only**: whether the code does what it was
written to do. No trend claims; see the rule at the bottom.

## The single line that validates the change

```
[comm_eff][frlr-anchor-q] refreshed global_step=20 anchor_step=20 boundaries=7 refreshes=7
```

Zero `Traceback` / `AssertionError` / `FATAL` in the log.

### Why `refreshes=7` is decisive rather than merely encouraging

`frlr_q_refreshes` is incremented in **two** places: the fast path's cadence branch
in `_frlr_ensure_basis`, and `anchor_update_basis`. Under `anchor_owns_q` the fast
branch is skipped (`elif self.anchor_owns_q: pass`), so the counter can only move
inside the anchor.

- The model has 7 masked boundaries at `pp_size=8`.
- One anchor fire refreshes all 7, so the counter reads **7**.
- **Had the fast path still been writing Q at its cadence of 1, the counter would
  read about 7 x 20 = 140 by now.** It reads 7.

So the arithmetic proves both halves of the operator's instruction at once: **the
anchor IS the Q writer, and the fast path is NOT.** That is a stronger check than
"it did not crash", and it is the check that could have silently failed.

### And the fail-closed assert not tripping is itself information

The engine asserts a **non-empty** sketch before refreshing. That assert passing
means the whole harvest chain executed: the validator accepted `prf_mask` with
`owns_q`, `state.py` plumbed the flag onto the masker, the engine registered the
mask codec on the **anchor clone**, the hook's harvest branch ran ahead of the
confinement assert with `path_tag=None`, and `_should_accumulate_frlr_sketch`
returned True. That last one is where the bug a unit test caught would have bitten:
the gate read `is_grad_enabled()` from **inside** a `no_grad` block, where it is
False by construction, so the sketch would have stayed empty and the assert would
have fired here at step 20 instead of at test time.

Timing was as predicted too: `train_batch` 128 equals `ppo_mini` 128, so one
optimizer tick per step, and an anchor cadence of 20 ticks puts the first fire at
**exactly step 20**.

## Numbers, recorded WITHOUT interpretation

| at step ~20 | a9 | for context only |
|---|---|---|
| `rollout_corr/kl` (gap) | 10.71 to 10.82 | a8 was 11.7151 over its first window; a7 was ~5.65 |
| `critic/score/mean` | 0.369 to 0.390 | incumbent 0.360 over 2-20, a7 0.365 |

Learning is proceeding at the normal early rate. The gap sits nearer a8's early
value than a7's, which is **consistent with** an under-fitted basis after a single
refresh, since a9 gets 10 refreshes across the whole run where a7 got 200. I am
recording that as a consistency observation, not as support for anything.

## The rule I am now applying mechanically

Three times this session I extrapolated from a window far too short to carry the
claim: a9 read as 1.7x slow from four step times (it is not, its per-step times
match a8's), R2 concurrency judged on four minutes of part counts, and R2
throughput projected at 2.2 MB/s where 20.6 minutes of data gives 5.78. Each was
corrected within the hour, and one of the same class nearly justified killing a8,
the cell that identified the gap mechanism.

The program had already written down that early windows lie. Writing it down is
evidently not the same as applying it, so: **no rate or trend claim from under 15
minutes of wall clock or fewer than about 10 samples, and no gate read before its
registered window.** a9's five predictions are scored at 100-120 and nowhere else.

## Config verified from WandB, and one reading trap to flag now

The engine truncates `$LOG` at start, so the config was read off WandB
(`a9-frlr-anchorq-200`, id `x6miw0zd`):

| key | value | as registered? |
|---|---|---|
| `comm_eff.anchor.owns_q` | **True** | yes, the point of the arm |
| `comm_eff.mask.frlr` | True | yes |
| `comm_eff.mask.frlr_unbiased` | **False** | yes, a9 is the biased variant; a10 flips it |
| `comm_eff.anchor.cadence` | 20 | yes, so Q moves every 20 optimizer ticks |
| `comm_eff.compression_type` | prf_mask | yes |
| `trainer.checkpoint_r2_enabled` | **True** | yes, so a9 uploads its own step-200 save |
| `trainer.save_freq` / `test_freq` | 200 / 200 | yes |
| `comm_eff.mask.frlr_q_cadence` | **1** | **inert here, see below** |

**The trap: a9's config says `frlr_q_cadence = 1`, and that does NOT mean Q
refreshes every step.** Under `anchor_owns_q` the fast-path cadence branch is
skipped entirely, so the knob has no effect. Anyone reading a9's config next to
a7's would see the same `frlr_q_cadence=1` and could conclude the two arms are
identical in Q handling, which is the exact opposite of the truth. The
disambiguator is the counter: a9 reads `refreshes=7` after 20 steps where a7 would
read about 140. Flagged here because a future reader of this program (or of these
notes after a context compaction) is the most likely person to make that mistake.

**Also worth recording:** the `aws configure` fix that repaired the R2 back-fill
lives in `/root/.aws/config` and therefore applies to the **in-training** sink as
well, since `r2_sink.py` shells out to the same `aws` binary as root. Without it,
a9's and a10's automatic step-200 uploads would have hit the same `InvalidPart`
failure on their 6.62G model files. The checkpoint-tree upload is exception-guarded
(WARN, keep local, continue), so it would not have crashed the run, but both cells'
checkpoints would have silently stayed local while the log claimed a sink was on.
