# Verdict: issue 93 round A, cell `a2-srq-b1-rn` (the bias/coherence control)

VERDICT: REVISE

Cell `a2-srq-b1-rn` (WandB `3muohefm`, `state=finished`, history through step 62) was
killed at step 60 under the operator's pre-authorized rule, and it was the right kill by a
wide margin: its reference-KL slope over the authorized steps 2 to 60 window is
**+0.018559/step against a1's +0.002707/step, a ratio of 6.856x** where the threshold was
2x, a margin of +0.013145 on a combined standard error of 0.000875, i.e. **z = +15.02**.
The call is not close on any window: 11.778x at steps 2 to 38, 6.632x over a2's whole life
(steps 2 to 61). Even the most adverse pairing of the two moving-block bootstrap intervals,
a2's 2.5th percentile against a1's 97.5th, gives **4.93x**, still 2.5x above the gate.

The cell did exactly the job its plan row assigned it ("informative for the A1-vs-A2
mechanism read; STOPs at step 60 if reference-KL slope >= 2x A1"), and it delivered a clean,
decisive, symmetric mechanism answer at about 2 GPU-h. It did not clear the section-1 gate
(2 of 6 gate flags) and it is structurally ineligible for that gate, since it never reached
the steps 100 to 120 read window. **REVISE**, not PASS and not STOP:

- **Not PASS.** There is an argument for PASS, because the plan's cell row declares the
  step-60 stop as expected behaviour and the mechanism finding is exactly the falsifiable
  negative the row asked for. I decline it for one concrete reason: the stage-1 gate is
  "at least one arm clears the section-1 gate at steps 100 to 120". a2 must be counted in
  that tally as a **non-clearing** arm, and a PASS label would put the opposite reading into
  the round-A gate table. Never PASS what was not measured at the gate.
- **Not STOP.** No program-level stop condition fired. The stage-1 gate is judged at the end
  of round A across all five arms; a3 (`a3-srq-parity-k493`) is already training on the box
  (`metrics/incoming.log` line 7240, `TaskRunner pid=139095`); budget is about 6.1 of 100
  GPU-h; revise depth 1 of `iterations` 7. The elevated gradient norm is 11x but **declining**
  (-0.197155/step over steps 2 to 61) with **zero NaN and zero Inf across all 61 synced rows
  and every logged key**, so it is not the exploding-grad-norm divergence that STOP is
  reserved for, and the arm is dead regardless.
- **REVISE** is the label that honestly reads "informative control, killed as pre-authorized,
  program continues", and it is the one that carries next_actions into a3, a4 and a5.

Judged against: issue 93 section 1 gate card and sections 6/8 decision tree (plan resolved
live from the issue body), cross-checked against `run.json` `success_criteria[0]` and
`cells[1]`. All three agree, including on the early-stop authorization.

---

## 1. The kill gate

Authorized rule (`run.json` `cells[1]`, issue section 6 and 8, window fixed in
`PROGRAM_STATE.md` before a2 finished): kill at step 60 if a2's `actor/kl_loss` slope is at
least 2x a1's, both arms fitted over the identical steps 2 to 60 window. Step 1 is excluded
because `actor/kl_loss` is exactly 0.0 there by construction in both arms.

All values from `runs/93-long-horizon-stability/analysis-a2.log` sections 1 and 2, with the
raw fits in `metrics/a2_killgate_2_60.json` and `metrics/a2_killgate_2_38.json`.

| quantity | a1 (REF) `h0n67q3a` | a2 (TEST) `3muohefm` |
|---|---|---|
| `actor/kl_loss` across the window | 1.909554 to 2.077836 | **0.639383 to 1.542281** |
| OLS slope, steps 2 to 60 | **+0.002707/step** | **+0.018559/step** |
| standard error, iid | 0.000121 | 0.000484 |
| standard error, Newey-West L=3 | 0.000214 | 0.000764 |
| moving-block bootstrap 95 percent, 20000 draws | [+0.001958, +0.003536] | [+0.017432, +0.023351] |
| residual sd, lag-1 ACF | 0.015851, +0.666 | 0.063259, +0.850 |

| gate arithmetic | value |
|---|---|
| ratio test/ref | **6.856x** |
| threshold, 2.0x ref | +0.005414/step |
| test minus threshold | **+0.013145** |
| combined standard error (HAC, both arms) | 0.000875 |
| **z** | **+15.02** |
| **CALL** | **KILL**, unambiguous |

Robustness of the call, all from the same log:

| window | a1 slope | a2 slope | ratio | z |
|---|---|---|---|---|
| 2 to 38 (earlier read) | +0.001632 | +0.019221 | **11.778x** | +8.10 |
| 2 to 60 (authorized) | +0.002707 | +0.018559 | **6.856x** | **+15.02** |
| 2 to 61 (a2's whole life, offline fit from `incoming.log`) | +0.002776 | +0.018410 | 6.632x | n/a |

**On not acting at step 38.** The verdict was already decisive at step 38 at z = +8.10 and
was correctly not acted on: the authorization named step 60, and at that point about 1.8
GPU-h of headroom made waiting free while removing any argument that the gate was met on a
short, noisy window. That is the correct handling of a pre-registered threshold. It cost
about 40 minutes of box time and bought an 8-sigma result becoming a 15-sigma result.

---

## 2. Metric comparison versus a1, with the window confound removed

The dispatch's comparison put a2's steps 40 to 60 window against a1's steps 100 to 120
window. Those are different points in training, and three of the quantities involved move
substantially with step, so the a1 column is reported **twice**: once at a2's matched window
and once at a1's own gate window. Matched-window a1 numbers come from
`analysis-a2.log` sections 5, 7, 8 and 9 (`metrics/a1_matched_40_60.json`); a1's gate-window
numbers from `analysis-a1.log` and `verdict-a1.md`.

| quantity | a2 @ 40-60 (n=21) | a1 @ MATCHED 40-60 (n=21) | a1 @ its gate 100-120 | baseline #90 |
|---|---|---|---|---|
| reference KL `actor/kl_loss`, absolute | 1.4521 | 2.0179 | 2.2518 | 0.156 to 0.203 (not comparable, see section 5) |
| reference KL slope, matched 2-60 fit | **+0.018559** | **+0.002707** | +0.003887 at gate | about +0.0015 |
| train-inference gap `rollout_corr/kl` | **12.1851** | 13.4700 | 13.7511 | 14.24 |
| gap slope, matched 2-60 fit | **+0.017478** | +0.008805 | +0.002825 at gate | +0.0005 creep |
| E[rho] = `k3_kl - kl + 1` | 0.0325 | **0.0269** | 0.0055 | 0.0014 |
| reward slope, matched horizon 2-61 | **+0.005243** (se 0.000302) | **+0.004390** (se 0.000298) | full run +0.003235 | 0.0032, bar 0.00288 |
| reward slope, in-window 40-60 | +0.00484 | **+0.00680** | -0.001074 | - |
| `critic/rewards/mean` level at step 61 | 0.5869 | 0.5605 | 0.6529 at gate | - |
| `actor/grad_norm`, window mean | **7.5738** | **0.6741** | 0.6490 | - |
| `actor/grad_norm`, run max | **62.2378 at step 1** | 0.8982 at step 11 | - | - |
| `actor/grad_norm`, run min | **6.1533** | 0.6081 | - | - |
| `actor/entropy`, level and 2-61 slope | 7.9494, +0.001194 | 7.9125, +0.000328 | 7.9348, +0.000309 | - |
| `actor/ppo_kl`, max absolute | exactly 0.0 | exactly 0.0 | exactly 0.0 | about 0 |
| `actor/kl_coef` | 0.001 constant | 0.001 constant | 0.001 constant | 0.001 |
| wire budget, `logical_pp_bits_sr_quant` | **2304** | **2304** | 2304 | 1232 |
| non-finite values, all keys | **0** | 0 | 0 | - |

a2's section-1 gate flags, informational, from `analysis-a2.log` section 4: `ref_kl_le_baseline`
False, `gap_lt_10` False, `gap_lt_3_target` False, `reward_slope_parity` **True**,
`ppo_kl_zero` **True**, `e_rho_gt_0p05` False. **2 of 6**, read at 40-60 rather than at the
gate, so not gate-comparable in any case.

**The two arms were the same circuit.** At matched step 60 the confinement counters are
bit-identical on 6 of 7 keys (`analysis-a2.log` section 9): `spectral_corrections` 13858 =
13858, `anchor_replay_fires` 3 = 3 (exactly `floor(60/20)` in both), `rank1_bypass_ticks`
19 = 19, `rank1_fires` 2 = 2, `merger_coldM_fallbacks` 0 = 0, `rank1_m_ready` 1 = 1. Only
`mask_applications` differs, 40376 versus 40950 (-1.4 percent), which tracks a2's slightly
longer responses (779.6 tokens at step 1 versus a1's 748.4) and therefore its microbatch
count, not codec behaviour. Combined with the identical 2304-bit wire budget and the single
shared launcher case arm (section 7), the causal attribution is as tight as this program can
make it: **one env var changed, and nothing else in the pipeline behaved differently.**

---

## 3. The tension resolved: was killing it correct?

Yes. Killing it was correct, and the "a2 is better on three of five" reading does not
survive contact with matched windows and with what the gap actually measures.

### 3.1 Two of the three claimed wins mostly dissolve

**E[rho] (claimed 0.0325 versus 0.0055, 5.9x).** E[rho] decays monotonically through a run
(the incumbent went 0.052 to 0.0003 over 600 steps). At **matched** steps 40 to 60 it is
0.0325 versus **0.0269**, a **1.21x** difference, and both sit about 20x below the 0.05 that
the importance-sampling safety net needs to be live. This is not a differentiator; it is a
window artifact.

**Reward slope (claimed +0.00519 versus +0.00324, 1.6x).** That compares a2's 62 steps
against a1's 120 steps, and a1's full run includes its saturation half (a1's own steps 61 to
120 slope is +0.001077). At the **truly matched horizon**, steps 2 to 61 for both arms:
a2 **+0.005243** (se 0.000302) versus a1 **+0.004390** (se 0.000298). The advantage is
+0.000853/step, **1.19x**, z = +2.01 on iid standard errors. These series are strongly
autocorrelated (measured residual lag-1 ACF +0.67 to +0.85 on `kl_loss`, where the iid to HAC
inflation is 1.6x to 1.8x), so the iid z overstates and the honest read is **a marginal
19 percent edge**. In the 40 to 60 window itself a2's reward slope (+0.00484) is **below**
a1's (+0.00680). "a2 learns much faster" is not supported.

### 3.2 The gap win is real but smaller, mechanically expected, and anti-settling

At matched steps the advantage is **1.285 nats** (12.1851 versus 13.4700), 9.5 percent, not
the 1.566 nats and 11.4 percent that the unmatched comparison suggests. Three things about it:

1. **`rollout_corr/kl` is itself a view metric.** It is computed from training-view logprobs,
   which pass through the codec. Round-to-nearest is by construction the MSE-minimising
   rounding at fixed bits; stochastic rounding is the same quantiser **plus** a zero-mean
   dither. So RN's training-view logprobs are less perturbed and the measured
   sampler-versus-trainer divergence is mechanically smaller. A lower measured gap here is a
   **smaller measurement distortion**, not a better-aligned policy. This is the same class of
   artifact as the reference-KL view offset in section 5, on a different metric.
2. **It is not settling; it is closing.** a2's gap slope over the matched 2 to 60 window is
   **+0.017478/step against a1's +0.008805/step, 1.985x** (`analysis-a2.log` section 3). At
   those rates the 1.285-nat advantage is consumed in about **148 steps**, i.e. the two curves
   cross around step 200, inside the round-C horizon. Program success criterion 1 asks the gap
   to **settle** (slope toward 0) or sit below 3 nats. a2 does neither, and it moves away from
   settling twice as fast as a1.
3. It buys nothing on wire budget: **2304 bits/token/boundary, identical to a1**, 1.8701x the
   incumbent's 1232. a2 fails issue section 7 item 4 exactly as a1 does.

### 3.3 The two losses survive every reframing

**Reference-KL drift: 6.856x (z +15.02), 11.778x at 2 to 38, 6.632x at 2 to 61.** No choice
of window makes this small, and it is the program's capability-preservation metric. A linear
extrapolation, which is charitable given the run's own convexity, projects a2 near **11 nats
at step 600** against the incumbent's 0.899.

**Gradient norm: this is not a spike.** a2's maximum, 62.2378, is at **step 1**, and its
**minimum over 61 steps is 6.1533**, which is **6.85x a1's maximum over 120 steps (0.8982)**.
The two distributions are disjoint. At matched steps 40 to 60 the means are 7.5738 versus
0.6741, **11.23x**. It declines (-0.197155/step) and plateaus near 8, so it is not runaway,
but it is a **permanently 11x elevated gradient scale at an identical LR of 1e-6, an
identical loss, and a demonstrably identical anchor circuit**. A gradient norm 11x larger
with only a 1.19x reward-slope edge is not a better optimizer; it is extra motion with an
uncontrolled direction.

### 3.4 So yes: a2 is learning fast while sprinting away from the base model

That framing is right, and the exchange rate makes the decision trivial: the sprint is
**6.9x** and the learning edge is **1.19x and marginal**. The mechanism explains both
symptoms at once. **A deterministic quantiser's error is coherent**: the same activation
pattern produces the same error every step, so the error does not average out across the
thousands of micro-updates but accumulates directionally. That is a phantom gradient with a
fixed direction, which shows up as (a) sustained extra gradient magnitude that is not task
signal and (b) monotone drift away from the base policy. It also explains the marginal reward
edge: a coherent push is still a push, and at step 60 on MATH that direction happens to have
positive projection on reward. There is no reason it stays aligned for another 540 steps, and
the anchor circuit, the only thing correcting boundary error, had fired exactly 3 times by
step 60. Under SR the residual it must correct is zero-mean between fires; under RN it is a
systematic offset that grows monotonically between fires.

The project's cardinal rule and issue section 7 both rank capability preservation above
reward. On that metric a2 is the worst arm this program has run. Kill upheld.

### 3.5 The standalone-objective lesson (program level, not just this cell)

**The train-inference gap must never be used as a standalone objective.** a2 is the existence
proof: any change that reduces the codec's distortion of the training-view logprobs reduces
the measured gap whether or not it reduces real policy mismatch, and a biased quantiser does
exactly that while making the real drift 6.9x worse. Gap and reference-KL drift moved in
**opposite** directions here at matched bytes and matched circuit. Every gap number in this
program must be reported with the reference-KL slope beside it, and any ranking on the gap
must carry the reference-KL clause as a hard filter. See section 6.

---

## 4. Mechanism conclusion: SR wins, and a1-prime does NOT fire

**SR wins, decisively.** The pre-registered A1-versus-A2 discriminator (unbiasedness versus
raw MSE) returns unbiasedness by 6.856x on reference-KL slope at z = +15.02, plus an 11.23x
sustained gradient-norm penalty for RN, at matched wire budget (2304 bits both), matched
codec structure (6 of 7 confinement counters bit-identical at step 60), and matched protocol.
RN's only durable advantage is a 1.285-nat lower **measured** gap whose slope is 1.985x worse.

**Therefore, per the pre-registered decision tree:** "If SR wins, precision allocation is the
axis: carry the better of {A1, A3} forward." The contingency arm
`a1-prime-srquant-hadamard` has the trigger "**ONLY if RN beats SR in A1-vs-A2**". That
condition is **not met**. **a1-prime does NOT fire.** It is stood down, saving about
3.8 GPU-h at the box's measured 112 to 119 s/step (the plan's 2.5 h estimate assumed
67 s/step). The round-A matrix continues unchanged: a3 running, then a4, then a5.

**What "SR wins" does and does not license.** It answers the mechanism question and selects
the axis label. It does **not** say the axis reaches the gate. a1's verdict measured the
noise-energy elasticity at 0.494 nats of gap per e-fold and projected about 4.7 bits per
coordinate, roughly 6.5x the incumbent's wire budget, to reach the sub-10-nat gap gate, which
contradicts the deployment premise. So the tree's prescription reduces to a **within-family
choice between a1 and a3**, among arms that may all miss the gate. a3, at byte parity, is the
arm that decides whether the precision family has any path at all.

**One idea worth re-filing rather than discarding.** a1-prime's Hadamard pre-rotation is a
**decorrelation** move, and the combined a1-plus-a2 picture (section 5.2) says decorrelation
of the error is the lever that matters. Its trigger did not fire and it must not spend round-A
GPU. Record it as a phase-2 candidate on merit, not as a round-A contingency.

---

## 5. Two corroborations worth recording

### 5.1 a2 independently confirms a1's view-offset interpretation

At step 1 both arms log `actor/kl_loss` **exactly 0.0** (structural). At step 2 each has
taken exactly **one** optimizer step from the same base checkpoint at the same LR of 1e-6
under the same protocol, so the real KL to the reference cannot be of order 1 nat: the dense
control reaches only **0.0049 nats after 173 steps** (issue section 1, cross-checked in
`PROGRAM_STATE.md`). Measured at step 2:

| arm | rounding | `actor/kl_loss` at step 2 |
|---|---|---|
| a1 | stochastic (dither on) | **1.909554** |
| a2 | round-to-nearest (dither off) | **0.639383** |
| difference attributable to the dither | | **1.270171 nats** |

That is **66.5 percent** of a1's step-2 level and **67.8 percent** of the 1.87260 quadratic
intercept that `verdict-a1.md` attributed to the SR view offset. a1's verdict inferred that
offset from within-run structure alone (constant detrended noise amplitude while the level
moved, plus the quadratic intercept). a2 removes the dither and the measured floor collapses
by exactly the predicted order, from a second and independent direction. The residual 0.639
is RN's own, smaller, view offset, so the correct reading is that **essentially all of the
approximately 1.9 level is measurement and the real drift is the slope only**, exactly as
a1's verdict scored it.

**Consequence for the gate, now confirmed twice.** The section-1 clause "reference KL at
matched step at or below baseline" is **unscorable on the absolute value for any sr_quant
arm** (`gate93.py` prints this caveat itself). It must be read on the slope. a3 is also
sr_quant, at 2 bits, so it will carry a **different** offset again, since the offset scales
with the quantiser. Do not compare a3's absolute reference KL to a1's, a2's, or the
baseline's.

### 5.2 Reconciling a1's "unbiasedness does not discriminate" with a2's "bias matters enormously"

These are not in tension. The two cells form a factorial in which **only one factor loads**.

| comparison | bias | variance / noise energy | result |
|---|---|---|---|
| incumbent PRF exact-k versus a1 (1-bit SR) | **pinned at zero for both** (both unbiased) | varied 2.7x, about 19 to about 7 ‖h‖², tails heavy to bounded | gap moved 3.4 percent (14.24 to 13.751); drift got **2.6x worse** for the LOWER-variance arm |
| a1 (1-bit SR) versus a2 (1-bit RN) | **flipped, 0 to nonzero** | **pinned**: same bits, block, coordinates, 2304-bit wire; RN's noise energy is if anything LOWER, since SR is RN plus a zero-mean dither | drift **6.86x worse**, grad_norm **11.23x higher** |

Within the unbiased family, unbiasedness cannot discriminate for the same reason a binary
condition cannot rank items that all satisfy it: once bias is zero, the residual differences
between unbiased codecs (magnitude, tail shape, mask coherence) do not order the drift, and
a1 showed the variance axis is not even monotone. Cross the biased/unbiased boundary at
constant bytes and constant noise energy and drift multiplies by about 7.

**Combined picture: what drives reference-KL drift under activation compression is whether the
codec's error has a persistent DIRECTION, not how large the error is.** This is the third
independent confirmation of the standing finding that bias times coherence, not magnitude,
gates reference KL (the dropout-versus-codec mechanism, the #89 coherence wall, and now the
a1/a2 pair at matched bytes and matched circuit). Operationally: **stop buying precision and
start buying decorrelation of the error across steps and across coordinates.** That is
precisely why the remaining round-A arms (a4's CVC correction and a5's FRLR plus token-IS)
are the interesting half of the matrix, and why the precision ladder (a1, a3) is now the
control half.

---

## 6. What a2 implies for the A-to-B winner rule

The rule (issue sections 6 and 8): *lowest train-inference gap **subject to** reference KL at
or below baseline and reward-slope parity; tie-break on higher E[rho]*.

**The ordering is doing the right thing, and a2 is the proof.** a2 has the **lowest measured
gap of anything this program has run** (12.1851 at 40-60, versus a1's 13.4700 matched and the
incumbent's 14.24, i.e. 14.4 percent below the incumbent), it **passes** reward-slope parity
(+0.005243 against the 0.00288 bar), and it has the **highest E[rho]** (0.0325). Under a
"lowest gap, then tie-break on E[rho]" reading with a weak or absolute-value constraint
clause, **a2 wins round A**. It is simultaneously the worst arm on the program's cardinal
metric. The constraint-first ordering is therefore not decoration; it is the only thing
standing between the winner rule and promoting an unstable arm into a 600-step round-C run
with checkpoints, validation and OOD spend. Keep it, and harden it on four points:

1. **Make the reference-KL clause a hard filter on the SLOPE, evaluated over a fixed matched
   window, applied BEFORE any gap ranking.** On the absolute value every sr_quant arm is
   unscorable (section 5.1); on the slope a2 is out at 6.856x and a1 is out at 1.8x to 2.6x
   depending on window.
2. **Add a settling requirement to the gap clause.** The winner rule says "lowest gap";
   program success criterion 1 says the gap must **settle**. a2 has a lower gap level and a
   1.985x faster-rising gap slope. A level-only ranking prefers the arm whose gap is racing
   upward. Read it as "lowest gap at the gate window **and** gap slope no worse than the
   incumbent's creep".
3. **Declare early-killed arms INELIGIBLE for the winner contest.** a2's numbers come from
   steps 40 to 60 and can never be compared at the 100 to 120 gate window. Treating them as
   candidate winner numbers is exactly the window error that produced the "better on three of
   five" reading. An arm that does not reach step 100 is a mechanism datapoint, not a winner
   candidate.
4. **Read the E[rho] tie-break at matched steps.** 0.0325 versus 0.0269 (1.21x), not 0.0325
   versus 0.0055 (5.9x). At these magnitudes, all about 20x below 0.05, the tie-break is
   nearly inert and must not be allowed to decide anything.

This does not touch the stage-1 threshold question that `verdict-a1.md` section 6 escalates
(the sub-10-nat gate that the incumbent itself fails). That remains an operator decision at
the A-to-B boundary, and nothing here reinterprets it in either direction.

---

## 7. Resolved params and plan-versus-ran

Full file: `runs/93-long-horizon-stability/resolved_params-a2.txt` (written alongside, not
over, a1's `resolved_params.txt`). `capture_resolved_config.py` still cannot run:

```
$ python3 scripts/capture_resolved_config.py runs/93-long-horizon-stability
capture_resolved_config: no train.log at runs/93-long-horizon-stability/train.log
rc=1
```

Flag: **`RESOLVED_CONFIG_MISSING`**. There is no launcher `set -x` trace in the synced log, so
no expanded `main_ppo` command line exists to recover. **No plan-versus-ran divergence was
found in anything verifiable**, and a2's load-bearing knob is better pinned than a1's, from
three independent directions:

1. **Source.** `origin/93-mismatch-control-kit:examples/grpo_trainer/run_93_cell.sh` has a
   single shared case arm `a1|a2)` setting `COMM_EFF_COMPRESSION_TYPE=sr_quant`,
   `QUANT_BITS=1`, `QUANT_BLOCK_SIZE=32`, `QUANT_SUBSET_K=0`, `MASK_PP_SIZE=8`,
   `ANCHOR_OWNS_Q=false`, with the **only** divergence being
   `COMM_EFF_QUANT_ROUNDING=rn` versus `sr`. The file is byte-identical between 223e4b1d (the
   origin tip when a2 launched, per `PROGRAM_STATE.md`) and the current tip 3f5e1996:
   `git log --oneline 223e4b1d..3f5e1996 -- examples/grpo_trainer/run_93_cell.sh` is empty.
   The same launcher hard-asserts `TRAIN_BATCH_SIZE=128`, `PPO_MINI_BATCH_SIZE=128` and
   `MAX_RESPONSE_LENGTH=2048` with fatal-on-miss.
2. **Runtime wire budget.** `logical_pp_bits_sr_quant = 2304` in both arms, every row.
3. **Runtime circuit behaviour.** 6 of 7 confinement counters bit-identical at matched step
   60 (section 2), the seventh off 1.4 percent on token count.

Not verifiable from the run dir (same synced-tail gap as a1): `pp_size=8`,
`train_batch_size=128`, `ppo_mini_batch_size=128`, `max_prompt_length=1024`, anchor
cadence/delay 20/20, `beta_anc=0.25`, `alpha=0.25`, rank-1 RELEX W2, LR 1e-6. Circumstantial
support: `anchor_replay_fires = 3` at step 60 is exactly `floor(60/20)`, which requires both
cadence 20 and one optimizer tick per global step, i.e. `train_batch_size` equal to
`ppo_mini_batch_size`; and `actor/ppo_kl` is exactly 0.0 at every step, the same signature.

The only deviation from the cell plan is the intended one: `run.json` `cells[1]` says
"early stop at 60 if kl slope >= 2x a1", and the cell stopped with WandB history at step 62.

---

## 8. next_actions

```yaml
next_actions:
  - knob: a3 read (a3-srq-parity-k493, running now)
    from: "judge a3 on gap level at 100-120 versus a1's 13.751 and versus the incumbent's 14.24"
    to: "judge a3 on (i) gap level AND gap slope at 100-120, (ii) reference-KL SLOPE fitted over the SAME steps 2-60 window as the a1/a2 pair, against a1's +0.002707, (iii) grad_norm max and window mean against a1's 0.898 / 0.674, and (iv) its own absolute reference-KL offset reported but NOT compared to a1's or the baseline's"
    rationale: "a2 proved the arm ranking flips depending on which quantity is read and on which window. a3 is 2-bit SR, i.e. unbiased with lower noise energy, so it is the third point on the variance axis at bias = 0 and a1's 0.494 nats-per-e-fold elasticity predicts about 13.6 nats. Its offset differs from a1's because the offset scales with the quantiser, so absolute reference KL is unscorable across sr_quant arms."
  - knob: a4 CVC success test (a4-prf-exactk-cvc-ce)
    from: "success = training-view gap slope goes negative and stabilizes with reward slope intact"
    to: "success = gap slope negative AND reference-KL slope at or below a1's +0.002707 on the matched 2-60 fit AND grad_norm max within 2x a1's 0.898; a negative gap slope with a worse reference-KL slope is scored as a MEASUREMENT win and a capability loss, not as CVC working"
    rationale: "a2 showed a codec can sit 1.285 nats lower on the measured gap while drifting 6.9x faster, because rollout_corr/kl is a view metric that any reduction in codec distortion lowers mechanically. CVC applies explicit uniformization pressure on that same view, so it is the arm most able to reproduce a2's pathology in a form that looks like success. Also note a1's untreated codec already decelerates from +0.008805 to +0.002825/step on its own, so merely-decelerating is not CVC working."
  - knob: round-A instability flag and its read point
    from: "no explicit grad_norm criterion in the section-1 gate; instability judged after the fact"
    to: "add 'grad_norm max > 5x a1's 0.898, or grad_norm window mean > 3x a1's 0.674' as a logged instability flag in every round-A gate table, evaluated at the first anchor window (about step 25) and reported for a4 and a5 alongside E[rho] in [0.2, 2]"
    rationale: "a2's grad_norm was 62.238 at step 1 and never fell below 6.153 in 61 steps, against a1's 120-step maximum of 0.898: the distributions are disjoint and the signal was fully available at step 1, about 2 GPU-h before the reference-KL slope gate resolved. a5 changes both the codec and the importance-sampling path, the two things that move gradient scale, so it is the arm most likely to need this flag."
```

---

## Notes

1. **Completion evidence.** No `done.flag` in the run dir, and the box was deliberately not
   touched (a3 is training on it). Terminal state established from artifacts: WandB reports
   `state=finished` with history through step **62** (`analysis-a2.log` sections 1 to 4), the
   local sync `metrics/incoming.log` carries a2 rows for steps 1 to 61 with **zero gaps**, and
   a3 (`TaskRunner pid=139095`, `experiment_name: a3-srq-parity-k493`) is present in the same
   log, which on a strictly one-cell-at-a-time box requires a2 to have gone terminal. The
   graceful SIGTERM produced a clean `finished` state rather than `crashed`, so the usual
   final-step WandB drop hazard did not bite here. The dispatch's report that the run touched
   step 63 during teardown is an on-box observation that is **not** present in the run dir; the
   verified last step is 62 in WandB history and 61 in the local sync, and nothing in this
   verdict depends on the difference.
2. **Verification commands.** The plan's per-cell command is "pull history, compute the gate
   table" (the `analyze.py` pass is round C only, explicitly "not run per stage"). Executed as
   `slope_compare93.py` (the kill gate, three metrics/windows) plus `gate93.py` (the section-1
   table), then two offline addenda parsed from `metrics/incoming.log` for the grad-norm
   series, the non-finite check, the matched-horizon fits and the matched-step counters. Full
   stdout in `runs/93-long-horizon-stability/analysis-a2.log`; machine-readable fits in
   `metrics/a2_killgate_2_60.json`, `metrics/a2_killgate_2_38.json`,
   `metrics/a2_gapslope_2_60.json`, `metrics/a2_gate.json`, `metrics/a1_matched_40_60.json`.
   GPU-free reads only. **No GPU box was touched and no new experiment was run.**
3. **Small numeric differences from the dispatch, all resolved in favour of the log.** The
   dispatch's a2 reference-KL slope +0.01944 and gap slope +0.01718 are `gate93.py`'s full-run
   fits including the step-1 structural zero; the log reports +0.01910 and +0.01690 for those,
   and the honest matched-window fits are +0.018559 and +0.017478. Reward slope +0.00519
   against the log's +0.00522 full run and +0.00484 in window. Confinement counters
   40376/13858/3/19 are the step-60 read (matching the dispatch exactly); `gate93.py`'s
   last-observed values at step 62 are 41636/14534/3/19. None of these differences move any
   conclusion; every number in this verdict is the one in `analysis-a2.log`.
4. **The one comparison the dispatch got structurally wrong, restated plainly.** Comparing
   a2's steps 40 to 60 against a1's steps 100 to 120 inflated a2's apparent advantages on all
   three of the quantities where it "wins". Corrected at matched steps: gap advantage 1.285
   nats not 1.566; E[rho] 1.21x not 5.9x; reward slope 1.19x and marginal, not 1.6x, and
   negative-signed inside the 40 to 60 window itself. This did not change the kill call, which
   was computed on matched windows from the start, but it materially changes how the tension
   should be read, and it is the same window error that item 3 of the winner-rule amendments
   is meant to prevent.
5. **Budget and depth.** Ledger `93-long-horizon-stability` has `max_gpu_hr` 100 (operator
   raised from 44). a1 consumed 3h58m39s; a2's 62 steps at the box's measured 112 to 119
   s/step is about **1.9 to 2.05 h**, so cumulative round-A spend is about **6.1 of 100 GPU-h**.
   Killing at 60 rather than running to 120 saved about 1.85 h and about $6.19 at $3.344/h.
   Standing a1-prime down saves about a further 3.8 h. `run.json` `iterations` is 7 with no
   recorded `revise_depth`, so this is depth 1 of 7. REVISE is comfortably in budget on both
   axes.
6. **Metrics-log hygiene, still unfixed and now confirmed to matter.** `metrics/incoming.log`
   is a tail-resync of the shared `/workspace/train.log` heartbeat symlink and now carries
   rows from **three** cells: a1 (`pid=6382`, 120 steps), a2 (`pid=93602`, 61 steps) and a3
   (`pid=139095`, starting). Every a2 number parsed offline in this verdict is pid-filtered.
   `verdict-a1.md` note 2 recommended per-cell metric sync targets from a3 onward; that has
   not happened, and with three cells interleaved a parse keyed on `global_step` alone will now
   silently mix arms. `metrics/sync-errors.log` also shows a repeating benign rsync failure on
   a `hotfix-patches` directory that does not exist on the box.
