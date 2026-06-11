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
length spiral** that some arms are susceptible to and dense is not.

**Correction (2026-06-11, conceded to mechanist-math): the distinguishing factor is NOT
gradient size/noisiness — it is merger-carrier presence (defer to §8).** An earlier
draft of this section proposed that dense's tiny clean gradients (grad_norm ~0.35) vs the
comm-eff arms' large noisy gradients (3–7 healthy, 20–60 igniting) were the immunizing
factor. **§8 falsifies that:** plain (no merger, SURVIVOR, zero emission) has grad_norm
median 3.4 / endpoint-mean 7.4 / max 10.5 — the *same noisy-boundary class* as ef_r2
(carrier, full-run median 4.9 / mean 9.3 / max 13.5, which DID emit four long-tail
spikes). plain is if anything slightly *less* noisy than ef_r2 yet emits nothing.
EXP-25's psgd-only control (grad_norm median ~1.6, no merger) was likewise clean (0.7415).
So gradient size/noisiness does **not** separate exploders from survivors;
**merger-carrier presence does** (§8). Dense's small clean grad is a *consequence* of
being merger-free and converged, not an independent immunizing mechanism. The operative
statement is §8's: the spectral merger (folding stale M into the fast gradient) is the
length-spiral carrier; the bare codec+anchor substrate (plain) is not.

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
*correlation*: the spiral-preventing factor is merger-absence (§8), not entropy and not
gradient size (§5 correction). dense's low entropy and small grad are both *consequences*
of being merger-free and converged.

**Ignition is NOT gated by a fixed entropy level — it's a joint (dose × sharpness)
boundary** (mechanist-math, confirmed from ef_r1): ef_r1 (full dose ~0.20) reached
**lock-in at steps 29–30 while entropy was still 0.83–0.81** (high), then entropy
*collapsed* 0.81→0.13 over steps 30–42 (a consequence of the spiral, not its gate). By
contrast low-dose exp27 (dose 0.02–0.09) did not ignite until entropy had sharpened to
~0.34 at step ~61. So higher dose ignites at higher entropy; lower dose needs more
sharpening first. This is a clean second confirmation that entropy is a follower/co-variate
(§5), not the trigger — ef_r1 ignited *before* its entropy collapsed.

---

## 8. Addendum: carrier-vs-substrate attribution (plain vs ef-parent r2)

**Question (task #5):** is the length-spiral a property of (ii) the bare comm-eff
*substrate* (PowerSGD codec + stale-anchor refresh) — which every comm-eff arm shares —
or (iii) specifically the *merger/carrier* (the spectral correction that folds
M_anchor into the fast gradient)? The decisive control is **plain `u1v94opv`
(`exp26_B_plain`)**, because its config differs from **ef-parent r2 `tilwe80t`
(`exp26_B_ef_r2`)** in **exactly one knob**:

| knob (from W&B config) | plain | ef_r2 | dense |
|---|---|---|---|
| `comm_eff.enabled` | True | True | — |
| `compression_type` | powersgd | powersgd | — |
| `powersgd.q_basis` | act | act | — |
| `anchor.enabled` / cadence / delay_K | True / 5 / 5 | True / 5 / 5 | — |
| `clean_cadence` | 0 | 0 | — |
| **`spectral.enabled` (the merger)** | **False** | **True** | — |
| `spectral.ef_clip` / `ef_decay` | 0 / 0 | 1 / 0.9 | — |

So plain = **substrate only** (PowerSGD + stale anchor refresh, merger OFF); ef_r2 =
substrate **+ merger**. Everything else identical. This isolates the merger.

### Scorecard at the step 40–50 endpoint

| run | merger? | entropy@end | len_mean@end | len_mean slope(41–50) | len_max range (back-half) | max consecutive 16384-pin | #16384 (41–50) | #clip (41–50) | grad_norm (mean/max, 40–50) | VERDICT |
|---|---|---|---|---|---|---|---|---|---|---|
| **plain** (`u1v94opv`) | **NO** | 0.478 | 197 | **−1.46 (declining)** | **≤1217 (no spike ≥4k), max ≤826 after s30** | **1** (lone @s1) | **0** | **0** | 7.4 / 10.5 | **CLEAN — no carrier signature at all** |
| **ef_r2** (`tilwe80t`) | YES | 0.396 | 146 | +0.80 (mild) | repeated 7817@32, 2020@35, 2648@38, 4061@47 | 1 (never clusters) | 0 (none ≥16384 in 41–50; spikes were @27/32/35/38/47) | 0 | 9.3 / 13.5 | WATCH — carrier present (isolated long-tail spikes), first-passage-lucky |

(Full attribution scorecard with these + the consecutive-pin and grad_norm columns:
`comparison_metrics/scorecard_addendum.csv`; the original 5-run scorecard is
`comparison_metrics/scorecard.csv`.)

### Attribution verdict — implicates the MERGER (hypothesis iii), not the substrate

> **ONE-LINE ATTRIBUTION:** plain (anchor refresh, *NO* merger — M_anchor never folded
> into the fast gradient) ends CLEAN (max consecutive cap-pin streak = 1, len/mean slope
> −1.46 = no spiral), while *every* M_anchor-folding merger arm (ef_r2, a0p5, exp27,
> ef_r1) shows the spiral signature ⇒ **within the 50-step censoring limit, the
> MERGER/CARRIER is implicated, not the bare PowerSGD+anchor substrate.** Honest caveat:
> plain is itself a *censored 50-step observation* (it never ran to 100), so this is
> "clean within 50 steps," not "proven stable" — the airtight version is a plain@100 run.

**plain (merger OFF) shows ZERO long-tail emission**: its len/max never exceeds ~826 in
the entire back half, len/mean is *declining* (slope −1.46), it has zero 16384 pins
after step 1, and its entropy hasn't even collapsed (still 0.48–0.55 at the end — the
*highest* of the comm-eff family). The PowerSGD codec + stale-anchor refresh substrate,
by itself, produces a perfectly stable training curve.

**ef_r2 (same substrate + merger ON) immediately reintroduces the carrier signature**:
repeated isolated long-tail spikes (7817, 2020, 2648, 4061) that plain never produces.
ef_r2 only survived because those spikes never landed *consecutively* (first-passage
luck) — but the *emission* is present, and ef_r1 (full-dose merger) and exp27
(damped merger) show that once they cluster, ignition follows.

**Therefore the length-spiral is carrier-specific, not substrate-generic.** Turning on
the spectral merger (folding the stale-anchor memory M into the fast gradient) is what
injects the reward-flat "longer" force; the bare codec+anchor substrate does not. This
matches mechanist-math's tangential-forcing mechanism: the carrier *is* the spectral
correction term, and plain has no such term (`spectral.enabled=False` ⇒ no M folded in).

**Caveat / honest limit:** plain's val is much lower (0.6437 vs ef_r2 0.7210) and its
entropy is still high at step 50 — plain is a *weaker, less-converged* policy, so one
could argue it simply hadn't entered the danger regime yet (it leaves the high-entropy
window latest of all). But that cuts the same way: plain spends the *longest* in the
high-entropy "seed-emitting" window and *still* emits no long-tail spikes, whereas ef_r2
emits them despite being further along. The merger, not exposure-time-in-high-entropy,
is the emission source. The clean falsification would be a "plain extended to 100 steps"
run; absent that, plain@50 is strong but not airtight evidence. **What we can claim:
with the merger OFF, the substrate produced no precursor in 50 steps; with the merger
ON (ef_r2/ef_r1/exp27), every run emitted the carrier spikes.**

---

## 9. Early-warning gate @ step ≤30 (operator ask, task #6)

**Question:** can we predict ignition susceptibility by step ~25–30 — *pretending we
cannot see past step 30* — instead of waiting for the 50+-step mark? Retro-tested on all
six runs over the visible window **[10, 30]** (steps ≤9 excluded as codec/optimizer
warmup; see the warmup caveat below). Outcomes: igniters = ef_r1 (lock-in 30), exp27
(lock-in 61), a0p5 (consecutive-pin onset 47–48); survivors-within-50 = dense, plain;
ef_r2 = censored-survivor (finished 50 clean but emitted back-half spikes → scored BOTH
ways). Full per-run numbers in `comparison_metrics/scorecard_early.csv`.

### The decisive feasibility test: is exp27's ≤30 window clean? NO.

This is the make-or-break question — exp27 ignited at step 61, so if its ≤30 window were
clean, a ≤30 gate could never catch a late-seeding run. **It is not clean:** exp27 emits
`len_max` = **3220 @ step 19** and **9764 @ step 23**, ~38 steps before lock-in. So the
*carrier emission* (isolated long-tail spikes) is visible at ≤30 even when *ignition*
(spike clustering) is 30+ steps away. The gate is feasible.

### Per-run early signals (window [10,30])

| run | outcome | #(len_max>2000) | #(>4000) | #(>8000) | p90 len_max | max len_max | len_mean slope@30 | grad_norm max | entropy slope |
|---|---|---|---|---|---|---|---|---|---|
| dense | SURVIVOR | **0** | 0 | 0 | 1147 | 1306 | +0.62 | 0.41 | −0.0073 |
| plain | SURVIVOR | **0** | 0 | 0 | 1089 | 1217 | −3.19 | 6.65 | −0.0592 |
| ef_r2 | CENSORED? | 1 | 1 | 1 | 1146 | 16384 | −3.41 | 6.43 | −0.0654 |
| a0p5 | IGNITE | 2 | 2 | 1 | 1270 | 16384 | −7.19 | 26.0 | −0.0766 |
| exp27 | IGNITE | 2 | 1 | 1 | 1296 | 9764 | −3.29 | 8.29 | −0.0603 |
| ef_r1 | IGNITE | 3 | 3 | 2 | 5782 | 16384 | −0.41 | 11.83 | −0.0598 |

### Best rule — long-tail spike-RATE gate

> **RULE (early-warning, evaluate at step 30 over window [10,30]):**
> **IF `response_length/max > 4000` on ≥1 step in [10,30] → flag UNSTABLE-LIKELY**
> (extend-watch + arm the P1 2-consecutive-pin kill; do NOT yet hard-kill, since the
> spike alone is not ignition). The threshold is robust anywhere in **2000–8000** —
> all four emitters have a spike exceeding even 8000 in the window, all survivors stay
> ≤1306 — so 4000 is a safe operating point well clear of both sides.

**Retro confusion matrix** (6 runs):

| ef_r2 treated as | TP | FP | TN | FN | sensitivity | specificity |
|---|---|---|---|---|---|---|
| IGNITE (it emitted) | 4 | 0 | 2 | 0 | **1.00** | **1.00** |
| SURVIVOR (conservative) | 3 | 1 | 2 | 0 | **1.00** | 0.67 |

Either way **sensitivity = 1.00** (catches all confirmed igniters, including exp27 at
step ~23 — 38 steps early — and a0p5 at step 17). The lone "false positive" under the
conservative scoring is ef_r2, which *did* emit a 16384 spike at step 27 and is itself a
censored 50-step run — i.e. a defensible "extend-watch" flag, not a true miss. Within
this 6-run sample the gate has **zero false negatives**.

### What FAILED as an early signal (so the telemetry spec doesn't waste them)

- **len/mean trailing-10 slope @ ≤30 — USELESS early.** Every comm-eff arm has a
  *negative* slope at step 25/30 (−3 to −9; the policy is still *shortening*). The
  len/mean *rise* is a post-ignition phenomenon only (exp27's slope is +0.10 at step 50,
  +1.97 at 60, +50 in ignition). P2 (mean-slope) is a *confirmation/kill* signal, **not**
  an early-warning one — do not expect it to fire before ~step 50.
- **p90 of len_max — USELESS.** Spikes are too rare to move the 90th percentile in a
  21-step window (all runs 1089–1296 except ef_r1). Use spike **count/rate**, not a
  percentile of the level.
- **entropy decline rate @ ≤30 (conditional, comm-eff only) — adds NOTHING.** All four
  comm-eff arms have near-identical entropy slope (−0.059 to −0.077) regardless of
  outcome; it does not separate igniters from survivors even *within* the merger class.
  Confirms the §5 finding (entropy is a follower) extends to the *rate* at ≤30, not just
  the level.
- **grad_norm character @ ≤30 — weak/secondary.** `grad_norm max` does separate dense
  (0.41) from everything else, and the igniter a0p5 is high (26.0), but plain (survivor,
  6.65) and ef_r2 (6.43) overlap exp27 (8.29). A `grad_norm max > 15` sub-rule would
  catch only a0p5 and miss exp27/ef_r1 — strictly dominated by the spike gate. Keep
  grad_norm as a *cross-run health baseline* (which arm is nearest dense-quality
  gradients), not an early trigger.

### Honest limits of a ≤30-step gate

1. **It catches carrier *emission*, not the *clustering* that actually ignites.** The
   gate fires on the first isolated long-tail spike; clustering (the P1 kill condition)
   can still be 30+ steps later (exp27) or never (ef_r2 survived 50). So the gate's
   correct action is **extend-watch + arm P1**, not hard-kill — it raises suspicion
   early, it does not call ignition early.
2. **It cannot catch a run whose first spike lands after step 30.** None of our six do
   (latest first-in-window emitter is ef_r2 at step 27), but with n=6 we cannot rule out
   a slow-seeding arm that stays sub-2000 through step 30 and ignites later. For such a
   run the gate would (wrongly) pass at 30; only the running P1/P2 monitor would catch
   it later. The gate **shortens** the watch for the common case; it does not **replace**
   the full-horizon monitor.
3. **n = 6, classes 2-clean / 4-emitting (one censored).** Perfect separation on six
   runs is encouraging but not statistically airtight. The gate's *floor* is solid (the
   two true merger-OFF survivors emit nothing ≤30; every merger-ON arm emits), which is
   the mechanistically expected pattern (§8) — but the operating threshold should be
   revisited as more runs accrue.
4. **Warmup exclusion is load-bearing.** dense emits one isolated 1306-max at step 6 and
   plain a >2000 at step 1 — both in the ≤9 warmup zone the gate excludes (consistent
   with ENTROPY_COLLAPSE_WATCH's codec-warmup ≤4 caveat). If the window were [1,30]
   instead of [10,30], dense/plain would NOT cleanly pass. The window start matters; 10
   is a safe choice (codec basis + optimizer have settled by then on these runs).

**Bottom line for the operator:** YES — a step-≤30 gate is feasible and would have
flagged all four merger arms (incl. the latest-igniting exp27, 38 steps early) with zero
false negatives on this sample, using a single signal: **any `response_length/max>4000`
in steps 10–30**. It buys ~20–35 steps of early warning for the common (early-seeding)
case, with the honest caveat that it raises suspicion (extend-watch) rather than calling
ignition, and cannot guarantee catching a hypothetical late-seeding run.

---

## Files

- `comparison_metrics/pull_wandb.py` — W&B fetch (scan_history, all 5 runs).
- `comparison_metrics/<label>.csv` — raw per-step history per run.
- `comparison_metrics/aligned_<metric>.csv` — step×run aligned tables.
- `comparison_metrics/scorecard.csv` — the 5-run precursor scorecard (task #2).
- `comparison_metrics/scorecard_addendum.csv` — carrier-vs-substrate scorecard
  (task #5): adds merger flag, max-consecutive-cap-pin, back-half len_max, grad_norm
  character; covers all six runs.
- `comparison_metrics/ef_r1_c7fa7kjv.csv` — ef r1 (exp26_B_ef) full history (§7a).
- `comparison_metrics/scorecard_early.csv` — ≤30-step early-warning signals, all 6 runs (§9).
- `comparison_metrics/early_gate.py` — early-gate retro-test + confusion matrix (§9).
- `comparison_metrics/analyze.py` — table/scorecard/slope generator (task #2).
- `comparison_metrics/addendum_scorecard.py` — carrier-vs-substrate generator (task #5).
- Ground truth for run 3: `runs/EXP-27/train_exp27_B_ef_damped.log`.
