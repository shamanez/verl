# EXP-27 Run Comparison — dense vs signed_ema α0.5 vs damped-EF

**Author:** comparator-runs (team exp27-postmortem, task #2)
**Date:** 2026-06-11
**Source:** W&B entity `shamanework-pl`, project `verl_compression_research`. Full
scalar histories pulled via `comparison_metrics/pull_wandb.py` (`run.scan_history`,
every logged step — not sampled). Run 3 (`qa6sll3h`) cross-checked line-for-line
against the local ground-truth log `runs/EXP-27/train_exp27_B_ef_damped.log`
(entropy/len/clip at steps 50,60,64,65,66,67 match exactly).

Raw per-run CSVs and aligned-metric CSVs live in `comparison_metrics/`.

---

## TL;DR (answers to the operator's question)

> **"Would α=0.5 (or dense) length-explode if run to 100 steps?"**

- **dense**: **No evidence it would, and good evidence it would not.** Dense is the
  *lowest-entropy* run of all five (0.12–0.16 from step 36 on) yet has a **dead-flat
  len/mean (~195–205) with len/mean slope −0.009/step at step 50** and only **one
  isolated** len/max=16384 spike in the entire run (step 6, zero consequence). It is
  the most stable trajectory by every precursor metric. **BUT** dense trained only
  **50 steps** — we have *zero* observations of dense at 51–100, so we cannot *prove*
  it stays stable; we can only say it shows none of the carrier signature and is the
  least likely of the three to ignite. **Honest limit: any claim about dense beyond
  step 50 is extrapolation.**

- **signed_ema α=0.5**: **Higher ignition risk than the optimistic read suggests —
  its step-50 state is NOT "safe", it is on the leading edge.** At step 50, α0.5 was
  flagged **DANGER** while exp27 was flagged **CLEAN** by the same scorecard. α0.5 @50
  had: len/mean slope **+5.9/step (rising, ~60× exp27's +0.10)**, a **consecutive pair**
  of len/max=16384 spikes at steps 47–48 (+nonzero clip both steps), and len/max=5806
  still elevated at 50. This is the *same precursor signature* exp27 showed at its
  steps 58–61 right before ignition locked in. **Verdict: α=0.5 had a materially
  elevated chance of igniting by step ~65–80 if extended.** It also ran only 50 steps,
  so this is a projection — but a worse-looking one than exp27's was at 50.

**Implication for the "merger = carrier" mechanism (challenge to mechanist-math
below):** if the merger drove ignition, α0.5 — which carries a 0.5-damped signed-EMA
merger — should look *at least as carrier-loaded* as exp27 at the same step, and it
does (worse, on len/mean slope and spike-clustering). Consistent with the merger
being the carrier. **However**, the absolute-entropy threshold (`entropy<0.4`) is
**falsified** as a cause: dense sits at 0.12–0.16 forever and never explodes. Entropy
*level* is not the trigger; entropy is a *follower* of the length spiral, not its seed.

---

## 1. How many steps did each run actually train?

| run | W&B id | trained steps (max) | val points logged | best val | final val |
|---|---|---|---|---|---|
| **dense control** | `5e2jpho9` | **50** (last train metric at step 48; val@50) | 0:0.085, 10:0.7324, 20:0.7377, 30:0.7415, 40:0.7483, **50:0.7536** | **0.7536** | 0.7536 |
| **signed_ema α=0.5** | `1wulaelw` | **50** | 25:0.7051, **50:0.7066** | 0.7066 | 0.7066 |
| **exp27 damped-EF** | `qa6sll3h` | **67** (killed ~step 67/68 on length-explosion) | 25:0.7134, **50:0.7202** | 0.7202 | 0.7202 |
| ef parent r2 (ctx) | `tilwe80t` | 50 | 25:0.6740, **50:0.7210** | 0.7210 | 0.7210 |
| plain (ctx) | `u1v94opv` | 50 | 25:0.4094, **50:0.6437** | 0.6437 | 0.6437 |

**Dense ran 50 steps only — this bounds every dense claim to ≤ step 50.** The operator's
"reported best val 0.7536" is confirmed and it is the step-50 value (dense val climbs
monotonically 10→50; it had not plateaued, so dense-at-100 is genuinely unknown).

exp27's "killed ~68" is confirmed: last logged step is 67, and the kill marker
(`done.flag`) cites step ~66, len/mean 557.6, max pinned 16384, entropy 0.079 — all
matched in the W&B history (step 66: entropy 0.079, len/mean 557.6, len/max 16384).

---

## 2. Aligned-step tables (key metrics)

Steps 10–50 are present for all 50-step runs; 55–67 are exp27-only (its ignition zone).
Full tables for every metric are in `comparison_metrics/aligned_<metric>.csv`.

### actor/entropy

| step | dense | a0p5 | exp27 | ef_r2 | plain |
|---|---|---|---|---|---|
| 10 | 0.348 | 2.303 | 2.118 | 2.363 | 2.252 |
| 20 | 0.263 | 1.541 | 1.404 | 1.268 | 1.506 |
| 30 | 0.194 | 0.788 | 0.882 | 0.880 | 0.902 |
| 40 | 0.163 | 0.493 | 0.640 | 0.549 | 0.715 |
| 50 | — | 0.371 | 0.401 | 0.396 | 0.478 |
| 60 | — | — | 0.342 | — | — |
| 64 | — | — | 0.220 | — | — |
| 66 | — | — | **0.079** | — | — |

**Note the regime difference.** dense's entropy is already ~0.35 at step 10 and decays
to 0.12–0.16; the comm-eff runs *start* near 2.3 (much higher exploration) and decay
toward ~0.4 by 50. So at step 50 the comm-eff runs have **HIGHER** entropy than dense
ever had after step 10. Low entropy is therefore not what separates the exploder from
the stable runs (see scorecard discussion).

### response_length/mean

| step | dense | a0p5 | exp27 | ef_r2 | plain |
|---|---|---|---|---|---|
| 10 | 211.8 | 244.0 | 254.4 | 251.7 | 253.6 |
| 30 | 200.9 | 119.3 | 180.1 | 148.8 | 223.0 |
| 40 | 204.2 | 125.5 | 176.5 | 147.4 | 221.0 |
| 50 | — | 170.4 | 165.8 | 145.7 | 197.0 |
| 60 | — | — | 171.0 | — | — |
| 64 | — | — | **395.1** | — | — |
| 66 | — | — | **557.6** | — | — |

### response_length/max  (16384 = max_response_length pin)

| step | dense | a0p5 | exp27 |
|---|---|---|---|
| 40 | 863 | 704 | 1035 |
| 47 | 839 | **16384** | — |
| 48 | 1014 | **16384** | — |
| 50 | — | **5806** | 594 |
| 59 | — | — | **16384** |
| 61–67 | — | — | **16384 (sustained)** |

### val-core/openai/gsm8k/acc/mean@1

| step | dense | a0p5 | exp27 | ef_r2 | plain |
|---|---|---|---|---|---|
| 25 | — | 0.7051 | 0.7134 | 0.6740 | 0.4094 |
| 50 | **0.7536** | 0.7066 | 0.7202 | 0.7210 | 0.6437 |

dense > ef_r2 ≈ exp27 > a0p5 > plain at step 50. **All three comm-eff variants
(exp27 0.7202, ef_r2 0.7210, a0p5 0.7066) sit ~3.3–4.7 pts below dense's 0.7536.**
None surpasses dense; exp27's val@50 (0.7202) is essentially at the ef_r2 floor
(0.7210) it was meant to beat — the falsify floor was missed.

### actor/grad_norm (ignition co-signal)

| step | dense | a0p5 | exp27 |
|---|---|---|---|
| 50 | — | 3.05 | 4.45 |
| 60 | — | — | 5.57 |
| 64 | — | — | **23.2** |
| 67 | — | — | **39.7** |

dense grad_norm is ~0.35 the entire back half (40× smaller than the comm-eff runs at
50). The comm-eff runs carry persistently noisy, large gradients even when "healthy";
ignition pushes exp27's grad_norm to 20–60.

---

## 3. exp27 ignition window (ground truth, steps 55–67)

| step | entropy | len_mean | len_max | clip_ratio | score | grad_norm |
|---|---|---|---|---|---|---|
| 55 | 0.325 | 174.8 | 866 | 0.000 | 0.772 | 6.5 |
| 58 | 0.353 | 185.1 | 4015 | 0.000 | 0.782 | 5.7 |
| 59 | 0.325 | 201.3 | **16384** | 0.0010 | 0.708 | 43.7 |
| 60 | 0.342 | 171.0 | 597 | 0.000 | 0.780 | 5.6 |
| **61** | 0.253 | 267.9 | **16384** | 0.0059 | 0.829 | 58.9 |
| 62 | 0.257 | 224.7 | **16384** | 0.0020 | 0.807 | 58.5 |
| 63 | 0.263 | 294.7 | **16384** | 0.0068 | 0.729 | 6.4 |
| 64 | 0.220 | 395.1 | **16384** | 0.0117 | 0.766 | 23.2 |
| 65 | 0.210 | 448.1 | **16384** | 0.0156 | 0.748 | 20.6 |
| 66 | **0.079** | 557.6 | **16384** | 0.0215 | 0.802 | 14.2 |
| 67 | 0.083 | 566.0 | **16384** | 0.0195 | 0.760 | 39.7 |

**Ignition anatomy:** isolated 16384 spikes at 45/53/59 do NOT ignite on their own
(len/mean recovers each time, e.g. 59→60 drops 201→171). The lock-in is at **step 61**:
the first time a 16384 spike *fails to recover* and the next step also pins → from 61
on it is monotone (len/mean 268→225→295→395→448→558→566, clip 0.006→0.020). Entropy
*follows* the length spiral down (0.34 at 60 → 0.08 at 66), it does not lead it.
**Score stays 0.73–0.84 throughout the explosion** — the policy is still answering
correctly while bloating, so reward gives no warning. This is a length-hack /
runaway-CoT, not a low-entropy reward collapse.

---

## 4. Ignition-precursor scorecard @ each run's step 50

Window = trailing 10 steps ending at the run's step 50 (its last common checkpoint).
`len_mean slope` and `entropy slope` are per-step OLS over that window.

| run | entropy@50 | ent slope(41–50) | len_mean@50 | **len_mean slope(41–50)** | len_max@50 | #(len_max≥16384) in 41–50 | #(clip>0) in 41–50 | VERDICT |
|---|---|---|---|---|---|---|---|---|
| **dense** (@48 last) | 0.135 | −0.004 | 193 | **−0.009** | 1014 | **0/8** | 0/8 | **CLEAN** |
| **a0p5** | 0.371 | −0.014 | 170 | **+5.92** | **5806** | **2/10** (47,48 — consecutive) | **2/10** | **DANGER** |
| **exp27** | 0.401 | −0.022 | 166 | **+0.10** | 594 | 1/10 (47, isolated) | 1/10 | **CLEAN** |
| ef_r2 | 0.396 | −0.020 | 146 | +0.80 | (n/a) | 0/9 | 0/9 | WATCH (ent slope) |
| plain | 0.478 | −0.024 | 197 | −1.46 | (n/a) | 0/8 | 0/8 | CLEAN |

**Precursor verdict logic (revised after dense falsified the entropy threshold):**
the discriminating signals are **(i) len/mean trailing slope > ~2/step, (ii) ≥2
len/max=16384 spikes, ESPECIALLY consecutive, in trailing-10, (iii) nonzero clip_ratio
recurring.** The absolute `entropy<0.4` flag is dropped as a *cause* — dense (0.12–0.16)
violates it permanently and is the most stable run.

### The decision-relevant comparison: α0.5@50 vs exp27@50

| signal | α0.5 @ step 50 | exp27 @ step 50 | who looks worse? |
|---|---|---|---|
| entropy level | 0.371 | 0.401 | ~tie (both ≈0.4) |
| len/mean trailing slope | **+5.92/step** | +0.10/step | **α0.5 (60× steeper)** |
| len/max=16384 spikes (41–50) | **2, CONSECUTIVE (47,48)** | 1, isolated (47) | **α0.5** |
| len/max at step 50 | **5806 (elevated)** | 594 (clean) | **α0.5** |
| clip>0 events (41–50) | 2 | 1 | **α0.5** |

**exp27 ignited 14–16 steps after its step-50 checkpoint (lock-in at 61).** Yet at the
matched step 50, **α0.5 looked WORSE than exp27 on every carrier metric except entropy
level (a tie).** α0.5's len/mean slope (+5.92) at step 50 was even steeper than exp27's
*pre-ignition* step-60 slope (+1.97, window 51–60). The consecutive 47–48 spikes are
the α0.5 analogue of exp27's first clustering at 59→61.

**So the answer to "is α0.5@50 safer/same/worse than exp27@50?": WORSE.** Not safe.
α0.5 was likely in the early phase of the same ignition; had it run to ~65–80 it had a
materially elevated chance of the same explosion. The reason it "looked stable" to the
operator is that the explosion lags the precursor by 10–16 steps and val@50 was still
fine — exactly the trap exp27 fell into (val@50 0.7202 healthy, dead by 67).

---

## 5. Why dense does NOT contradict this (the entropy red herring)

Dense's full trajectory: entropy crosses below 0.4 at **step ~5** and sits at
**0.12–0.16 from step 36 to the end**, with len/mean pinned at 195–205, len/mean slope
≈ 0, grad_norm ≈ 0.35, pg_loss ≈ 0, and exactly **one** isolated 16384 spike (step 6,
no recovery problem). Dense is simultaneously the *lowest-entropy* and the *most
stable* run.

This proves the trigger is not "entropy got low." The trigger is a **positive-feedback
length spiral** that the comm-eff runs are susceptible to and dense is not. Candidate
distinguishing factor (for mechanist-math): dense has tiny, clean gradients
(grad_norm ~0.35); the comm-eff runs carry **persistently large, noisy boundary
gradients (grad_norm 3–7 when healthy, 20–60 in ignition)** from the
compression/EF/merger path. A length-favouring direction in that noisy gradient, once
it pins a few samples at max length, gets amplified by GRPO's advantage normalization
(longer correct answers keep nonzero reward → reinforced) with no dense full-rank step
to flush it. Dense's clean gradient never lets the spiral start.

---

## 6. Verdict + uncertainty

**Operator question — "would α=0.5 or dense ignite by 100 steps?"**

- **dense: ~unlikely to ignite, but unproven beyond 50.** Probability of dense
  length-exploding by step 100 ≈ **low (subjective ~10–20%)**. It shows zero carrier
  signature (flat len/mean, 1 isolated spike, clean small gradients). Caveat: we have
  *no* data past step 50, dense val was still climbing (not converged), and one cannot
  rule out a late instability. We can claim "dense shows no precursor at 50 and is the
  least ignition-prone of the three"; we **cannot** claim "dense is proven stable to
  100."

- **signed_ema α=0.5: materially elevated ignition risk; my estimate it would have
  ignited by ~step 100 ≈ 55–70% (subjective).** Its step-50 state already carries the
  precursor (rising len/mean +5.9/step, consecutive 47–48 max-pins, elevated tail at
  50) and looks *worse* than exp27 did at 50, while exp27 ignited 14–16 steps later.
  The main uncertainty: α0.5's 47–48 cluster could have been a transient that decayed
  (exp27's 45/53/59 isolated spikes did decay before the 61 lock-in), so it is not
  certain — but "stable" is the wrong label for it. **The operator's hypothesis that
  α0.5 "looked stable and might be fine at 100" is not supported; the data lean toward
  it being on the same ignition path, just earlier in it and lucky to be killed at 50.**

**One-line for the team:** the merger-carried length spiral is the killer (entropy is a
follower, not the cause); dense's clean small gradients immunize it; α0.5 is not a safe
fallback — at step 50 it already looked worse than the run we killed for exploding.

---

## 7. Cross-run ef-class evidence (added for mechanist-math, task #1)

mechanist-math's mechanism: the implemented "EF" injects a force ~orthogonal to the
fresh gradient (measured cos(G_comp,G_corr)=0.956), so it transports the policy along
reward-flat "correct-but-longer" directions; **dose sets the LAG to ignition, not
whether**. Two cross-run pulls test this.

### 7a. ef r1 (`c7fa7kjv` = exp26_B_ef, full-dose) — IDENTICAL precursor sequence, ignites EARLIER

42 steps. Lock-in at **step 29-30** (vs exp27's 61). Exact same anatomy:

| phase | ef_r1 (c7fa7kjv) | exp27 (qa6sll3h, damped) |
|---|---|---|
| isolated early spike (recovers) | 5782 @ s12 | 16384 @ s2,45,53 (recover) |
| lock-in onset (pin fails to recover) | **s29→30** (13347→16384, first clip>0) | **s61** |
| sustained 16384 from | s30 (30-36, 38-42) | s61 |
| len/mean creep follows | flat ~160 to s37, then 143→203→222→301→328 | flat ~170 to s60, then 268→395→558→566 |
| entropy follows DOWN | 0.58@36 → 0.16@41 → 0.13@42 | 0.34@60 → 0.08@66 |
| score during ignition | 0.73–0.78 (no warning) | 0.73–0.84 (no warning) |
| grad_norm spikes | 22.9@31, 33.4@39 | 23@64, 40@67 |

**This is the strongest single confirmation:** ef r1 (full dose) ignited at ~30; exp27
(damped clip 0.5/decay 0.5, ~half the dose) ignited at ~61. Damping roughly **doubled
the lag (30→61) but did not prevent ignition** — exactly "dose sets lag, not whether."

**Dose magnitude correction (cross-check of mechanist's numbers):** from W&B,
`spectral_rel_change_mean` median over s18-27 = **ef_r1 0.200, ef_r2 0.250, exp27
0.092**. mechanist's quoted exp27 "0.19-0.30 @ s20-25" actually matches the *ef parents*,
not exp27 — exp27's damping had already cut the dose to ~0.09 (≈half) from early on.
exp27 dose decays 0.092 (s18-27) → 0.021 (s45-67); ignition s61-66 is near the dose
**minimum**, confirming ignition is not a dose spike. (Direction of mechanist's claim
holds; the absolute s20 number was a run mix-up.)

### 7b. Spike-clustering taxonomy across all six runs — the real discriminator

Every run hits len/max=16384 at least once. What separates exploders from survivors is
whether the pins ever become **consecutive**:

| run | dose class | 16384-pin steps | max consecutive run | outcome by its last step |
|---|---|---|---|---|
| dense | none | {6} | 1 | STABLE (50) |
| plain | none (no codec force) | {1} | 1 | STABLE (50) |
| ef_r2 | full, lucky | {5, 27} + 4061@47 | 1 | STABLE (50) — *late isolated spike present* |
| **a0p5** | signed-EMA δ0.5 | {17, **47, 48**} + 5806@50 | **2** | killed @50 (DANGER, pre-ignition) |
| **exp27** | EF damped 0.5 | {2,45,53,59, **61..67**} | **7** | EXPLODED (lock-in 61) |
| **ef_r1** | EF full | {12(=5782),29, **30..36, 38..42**} | **7** | EXPLODED (lock-in 30) |

**Mechanist Ask-A result, partly contradicting his prediction:** he predicted α0.5 max
"stays <~1k". It does NOT — α0.5 hits 16384 at s17,47,48 and 5806 at 50. So α0.5 emits
the same long-tail spikes as the ef class; it is **not** spike-free. ef_r2 confirms his
other prediction: it *does* show late isolated spikes (4061@47, 16384@27) despite
finishing clean — drift-toward-long-tail is an ef-class property and **r2 was
first-passage-lucky** (its spikes never landed consecutively). a0p5's consecutive 47-48
pair is the only thing separating it from a confirmed exploder — it is the *onset* of
the clustering, caught one step before lock-in.

### 7c. Ask-B (entropy regime) — confirmed

`entropy` first crosses below 0.4 at: **dense s1, a0p5 s45, ef_r2 s50, exp27 s51, plain
never (min 0.478)**. So the merger/ef arms sit in the high-entropy (>0.4)
"seed-emitting" window for ~45-50 steps; dense leaves it at step 1. entropy@50: dense
0.135 (@48), a0p5 0.371, ef_r2 0.396, exp27 0.401, plain 0.478. Confirms mechanist's
"merger arms sit in the high-entropy window ~3× longer than dense" — but note this is
*correlation*: dense's low entropy co-occurs with its clean small gradient, and it is
the gradient cleanliness (not the low entropy) that prevents the spiral (§5).

---

## Files

- `comparison_metrics/pull_wandb.py` — W&B fetch (scan_history, all 5 runs).
- `comparison_metrics/<label>.csv` — raw per-step history per run.
- `comparison_metrics/aligned_<metric>.csv` — step×run aligned tables.
- `comparison_metrics/scorecard.csv` — the precursor scorecard.
- `comparison_metrics/analyze.py` — table/scorecard/slope generator.
- Ground truth for run 3: `runs/EXP-27/train_exp27_B_ef_damped.log`.
