# EXP-25 — Deep Findings: the signed-EMA α-sweep falsifies the anchor-default substrate

**Scope.** This is the experiment-level scientific writeup for the EXP-25 α-sweep
(issue #25). It builds on the single-arm root-cause already written in
`runs/EXP-25/ENTROPY_COLLAPSE_FINDINGS.md` (α=0 only) and the standing watch
`research/diagnostics/ENTROPY_COLLAPSE_WATCH.md` (T1–T7). What is NEW here:
(a) the full three-arm dose-response with a cross-run isolation table against the
four anchor-OFF references; (b) a rigorous RL account of WHY the collapse happens
and why α=0.5 escapes; (c) the deeper implication that the dose-response is
MONOTONIC ⇒ signed_ema is the wrong primitive; (d) a ranked menu of improvements.

**Verdict it supports:** STOP (hypothesis falsified). Best-α=0.5 `val@50 = 0.7066
<= 0.7114` (the plan's falsification line). See `runs/EXP-25/verdict.md`.

**Provenance.** All numbers below are from the local fulltrain logs
(`runs/EXP-25/logs/exp25_alpha_0p{0,3,5}.fulltrain.log`) + W&B. The gates are from
the probe trainlogs (`exp25_id0_anchorM.trainlog`, `exp25_id1_R2R3.trainlog`).
Resolved config is `runs/EXP-25/resolved_params.txt`. The merger code is correct
(`verl/workers/comm_eff/spectral_filter.py:307` + cold-M guard at :296) — this is a
training-dynamics result, NOT a bug.

W&B run ids (entity shamanework-pl, project verl_compression_research):
α=0.0 `uyrpaftw` · α=0.3 `r8kc702g` · α=0.5 `1wulaelw` ·
dense `5e2jpho9` · A0 r77+clean@5 `oquyeic3` · r102+clean@5 `kqozxfr0` ·
mask+clean@5 `3yxzzwn3` · no-refresh floor = EXP-23 A1 (0.6914).

---

## (a) The dose-response and the cross-run isolation table

### A1. The three-arm headline

| α | merger on a disagreeing coord | val@25 | val@50 | peak train-reward | final train-reward | entropy s1→s50 | resp_len s1→s50 (max) | clip_ratio s50 | collapse? |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 (`uyrpaftw`) | `(2·0−1)·\|G\| = −1.0·\|G\|` (full reversal) | 0.7180 | **0.3541** | 0.787 @s28 | 0.478 | 5.69→0.086 | 278→5863 (8634) | 0.290 | **YES, catastrophic** |
| 0.3 (`r8kc702g`) | `(2·0.3−1)·\|G\| = −0.4·\|G\|` (40% reversal) | 0.6937 | **0.6164** | 0.773 @s28 | 0.621 | 5.70→0.334 | 282→15786 (15786, hits 16K cap) | 0.909 | **YES, delayed ~10 steps** |
| 0.5 (`1wulaelw`) | `(2·0.5−1)·\|G\| = 0` (disagreeing coords ZEROED) | 0.7051 | **0.7066** | 0.814 @s41 | 0.814 | 5.69→0.371 | 276→170 (288) | 0.000 | **NO** |

The ordering is strict and monotonic in α: **val@50 improves monotonically as α
rises** (0.354 → 0.616 → 0.707), i.e. *less* signed correction is *always* better.
The best arm sits at the high-α (least-correction) edge of the swept grid.

### A2. The KEY isolation — length explosion, not low entropy, is the val-killer

The headline scientific result is a clean dissociation. Compare the collapsing
arms, the surviving arm, and the four anchor-OFF references on three axes —
entropy, response length, and val:

| run | mechanism | final entropy | final resp_len | final clip_ratio | val@50 | length-explodes? | val-healthy? |
|---|---|---|---|---|---|---|---|
| dense (`5e2jpho9`) | no comm-eff | ~0.13 (low) | ~280 (bounded) | ~0 | 0.7536 | NO | YES |
| A0 r77+clean@5 (`oquyeic3`) | PowerSGD + fresh clean step | low | bounded | ~0 | 0.7415 | NO | YES |
| no-refresh floor (EXP-23 A1) | PowerSGD, no anchor | low | bounded | ~0 | 0.6914 | NO | mostly (floor) |
| mask+clean@5 (`3yxzzwn3`) | prf_mask + clean step | low | bounded | ~0 | (ref) | NO | YES-ish |
| **EXP-25 α=0.5 (`1wulaelw`)** | signed_ema, disagreeing→0 | 0.371 (declining) | 170 (bounded) | 0.000 | 0.7066 | **NO** | partial |
| **EXP-25 α=0.3 (`r8kc702g`)** | signed_ema, −0.4·\|G\| | 0.334 | **15786 (16K cap)** | **0.909** | 0.6164 | **YES** | **NO** |
| **EXP-25 α=0.0 (`uyrpaftw`)** | signed_ema, −1.0·\|G\| | 0.086 | **5863 (peak 8634)** | **0.290** | 0.3541 | **YES** | **NO** |

Read the table by columns:

1. **Entropy declines in EVERY arm, including the healthy ones.** The dense
   reference trains at ~0.13 entropy and gets val=0.75. α=0.5 ends at 0.37 entropy
   and gets 0.71. So *low entropy alone is not pathological* — a confident policy
   on GSM8K is correct, not collapsed. **Entropy is NOT the discriminator.**

2. **Response-length explosion + high clip_ratio co-occur with bad val, and ONLY
   in the sign-reversal arms (α<0.5).** Every anchor-OFF reference AND α=0.5 keep
   length ~170–290 with `clip_ratio ≈ 0`. The two collapsing arms run away to
   thousands of tokens (α=0.3 saturates the 16384 cap; `clip_ratio` 0.91), and
   that is exactly where val falls off a cliff. **Length explosion (driven by
   `response_length/clip_ratio`) is the discriminator.**

3. **The train-inference gap confirms it.** `training/rollout_probs_diff_mean`
   collapses to 0.072 (α=0) and 0.261 (α=0.3) by s50 but stays 0.620 (α=0.5) — the
   reversal arms drive the trained policy into near-deterministic agreement with
   its own degenerate long-output rollout, while α=0.5 keeps the healthy train↔
   rollout gap that the dense and PowerSGD references also keep.

**Conclusion (isolation):** the proximate cause of the val collapse is a
**response-length degeneration reward-hack**, not entropy collapse per se. Entropy
collapse and length explosion *co-occur only when the merger reverses signs*
(α<0.5); when the merger merely zeroes disagreeing coords (α=0.5), entropy still
declines but length stays bounded and val survives.

---

## (b) Why the collapse happens — rigorous RL account

### B1. What the merger does to the optimizer

`spectral_filter.py:307`:
```python
g_corr = alpha * gm + (1.0 - alpha) * gm.abs() * torch.sign(anc)
```
`gm = G_noisy` is the fast (PowerSGD-compressed, activation-rescaled) live
gradient; `sign(anc) = sign(M_anchor)` is the sign of a β=0.95 EMA of the
K-stale anchor gradient. At α=0 this is pure **magnitude-preserving sign-SGD**:
the update keeps the live per-coordinate *magnitude* but takes its entire
*direction* from the stale anchor sign vector.

### B2. The (2α−1) knee — exact, on a sign-disagreeing coordinate

On a coordinate where `sign(M_anchor)` disagrees with `sign(G_noisy)`, write
`G_noisy = |G|·sign(G)` and `sign(M) = −sign(G)`. Then:
```
g_corr = α·|G|·sign(G) + (1−α)·|G|·(−sign(G)) = (2α−1)·|G|·sign(G)
```
So the correction acts on disagreeing coordinates with coefficient **(2α−1)**:
- α=0.0 → −1.0·|G| : the step is **fully reversed** at full magnitude (ascent).
- α=0.3 → −0.4·|G| : the step is **40% reversed** (still net wrong-direction).
- α=0.5 → 0·|G| : the disagreeing coordinate is **zeroed** (the knee — no reversal,
  no progress, just dropped). This is why α=0.5 is the phase boundary.
- α→1.0 → +1.0·|G| : no correction = plain PowerSGD `G_noisy`.

On *agreeing* coordinates the formula gives `+1·|G|` for all α (no harm). So the
entire effect of the merger is concentrated on the disagreeing fraction.

### B3. How big is the disagreeing fraction? — the √2 signature

The merger logs `rel_change = ||G_corr − G_noisy|| / ||G_noisy||` per matrix every
step. For α=0, on the warm steps (after M warms at step 3), the **median
rel_change = 1.416 ≈ √2** (n=1379 per-matrix-step samples, max 1.889;
`exp25_alpha_0p0.fulltrain.log`). For α=0 a coordinate that agrees contributes 0
to `G_corr − G_noisy` and one that disagrees contributes `−2·|G|`. So
`||G_corr − G_noisy||² = 4·Σ_{disagree}|G|²` and
`rel_change² = 4·(disagree energy)/(total energy)`. `rel_change = √2` ⇒
`disagree energy / total energy = 1/2`: **~50% of the magnitude-weighted gradient
energy is on coordinates whose stale-anchor sign disagrees with the live gradient,
EVERY step.** The stale (≈2.5-global-step, β=0.95-smoothed) anchor sign is wrong
on half the gradient's mass at all times — this is not a transient warm-up
artifact, it is the steady state.

### B4. Why sign-reversal → length-degeneration under no-KL/no-entropy

Two compounding effects:

1. **Destroyed sign-cancellation = no implicit regularizer.** The true minibatch
   PG at a coordinate is a *sum of signed* per-sample score-function terms times
   group-normalized advantages; across a GRPO group these partially cancel, so the
   true step is small on most coordinates and near-zero (sign ill-defined) on
   many. That partial cancellation is the implicit step-size regularizer. The
   merger replaces it with `(2α−1)·|G|·sign(M)` — **full activation-rescaled
   magnitude on every covered coordinate with a fixed stale sign**. There is no
   cancellation left, so each step is large and biased.

2. **No brake on the degenerate direction.** With `use_kl_loss=False`,
   `use_kl_in_reward=False`, `entropy_coeff=0` (confirmed last-wins in
   `resolved_params.txt` despite an overridden early `use_kl_loss=True` in the
   launcher), there is no KL anchor to a reference policy and no entropy bonus. The
   only thing shaping the policy is the GSM8K reward. A persistently
   wrong-direction, full-magnitude step pushes the policy toward the nearest
   reward-correlated degenerate mode it can reach: **emit longer and longer
   outputs.** Longer outputs raise the per-sequence chance of stumbling onto the
   answer string under the lenient reward, so reward briefly RISES (peak 0.787@s28
   for α=0) even as the policy is degenerating — then the responses blow past the
   useful regime, `response_length/clip_ratio` climbs (truncations), and val
   collapses (0.354). α=0.3 is the same mechanism with a weaker push (−0.4 vs
   −1.0), so it takes ~10 more steps and saturates the 16K cap before val falls to
   0.616.

### B5. Why α=0.5 escapes

At the (2α−1)=0 knee the merger *zeroes* disagreeing coordinates instead of
reversing them. So the update is `G_noisy` on the agreeing half and 0 on the
disagreeing half — a **masked but never wrong-signed** step. This is a strict
contraction of the live gradient (a projection onto the sign-agreement set), not
an ascent direction, so it cannot drive the runaway. Entropy still declines
(the policy still sharpens on the agreeing-coordinate signal), but length stays
bounded (170, clip~0) and val survives (0.707). α=0.5 is essentially "trust the
live magnitude only where the stale sign agrees, otherwise do nothing" — which is
the least-harmful point on the family but still throws away ~half the gradient,
which is why it only reaches 0.707, below the floor+0.02 line.

---

## (c) The deeper implication — signed_ema is the wrong primitive

The dose-response is **monotonic in α**: val@50 = 0.354 (α=0) < 0.616 (α=0.3) <
0.707 (α=0.5), and by extrapolation the best member of the family is the α→1 limit
— which is *plain PowerSGD with no signed correction at all*. There is no interior
optimum. This is the decisive scientific finding of EXP-25:

> Every unit of signed correction the merger applies makes RL training worse. The
> correction term `(1−α)·|G_noisy|·sign(M_anchor)` is **net-harmful** for this
> problem; its optimum is to be turned off.

Why this is principled, not a tuning miss:
- The correction's only effect is on sign-disagreeing coordinates (B2), and the
  stale anchor disagrees on ~50% of the gradient mass at all times (B3).
- A stale (≈2.5-step, β=0.95) sign is a *worse* estimator of the live update
  direction than the live (compressed) gradient's own sign — the EMA averages over
  a non-stationary boundary geometry (the #21/#23 finding that the boundary
  gradient drifts). So overriding the live direction with the stale sign injects
  bias precisely where it hurts.
- Magnitude-preserving sign-SGD with a *persistent* (non-cancelling) sign is a
  known degeneration driver; under no-KL/no-entropy there is nothing to stop it.

So #25's structural inversion (full-coverage DP-reduced stale M, anchor-owned Q,
the signed_ema merger) is *correctly implemented* (all id-0/id-1 gates green) but
the merger PRIMITIVE it was built to serve does not recover the comm-efficiency
gap — it makes the gap worse the more it is used. The right conclusion is to keep
the verified anchor substrate (M is now provably global/full-coverage/real-weight/
evolving; Q is provably anchor-owned and broadcast) and **replace the
sign-replacement correction with a primitive that does not override the live update
direction.**

---

## (d) Improvements, ranked

Ranked by expected information-per-GPU-dollar and by how directly each addresses
the falsified mechanism.

1. **Abandon sign-replacement; switch to error-feedback on the PowerSGD residual
   (issue #24).** This is the principled successor and #24 was already gated on
   #25. Error-feedback accumulates the compression residual `G_noisy − decompress(
   compress(G))` and adds it back next step — it corrects what compression DROPS
   without ever overriding the live update DIRECTION (the exact failure of
   signed_ema). Combined with the basis-aligned anchor that #25 already built and
   verified (Q owned + broadcast from the slow net), this keeps the comm savings
   while removing the net-harmful sign term. **This STOP is the green light to
   redesign the primitive before #24 spends compute.** Highest priority.

2. **Add a regularizer that closes the length-degeneration channel, then re-test
   the correction.** The proximate killer is the length explosion, not the
   correction per se. Re-run α=0.3 (has signal, collapses) with ONE of: an entropy
   floor (`entropy_coeff>0`), a KL penalty to the reference policy (`use_kl_loss=
   True`, small `kl_loss_coef`), or a hard response-length cap / length penalty in
   the reward. If a brake lets even α=0.3 hold its s28 peak (0.773) to s50, that
   isolates "correction bias" from "degeneration" and may make a weaker correction
   viable. NOTE this changes the FIXED control surface (no-KL/no-entropy) — it is a
   deliberate axis change, run it as a labelled new lineage, not a silent drift.
   Second priority; cheap (one 50-step arm).

3. **Sweep α∈{0.7, 0.85, 1.0} to confirm the monotonicity / quantify the α→1
   limit.** The prediction from (c) is that 1.0 (plain PowerSGD) ties or beats 0.5.
   One sweep settles whether ANY signed_ema correction is ever worth it, and gives
   the clean "correction is net-harmful" datapoint for the writeup. Cheapest
   confirmatory experiment; lower priority than (1)/(2) because the expected answer
   is "turn it off."

4. **Magnitude-and-direction-preserving correction variants.** If a stale signal
   is to be used at all, use it to *scale* or *precondition* the live gradient
   (e.g. trust-weight by per-coordinate sign-agreement confidence) rather than to
   *replace* the sign. This preserves the live direction (which B3 shows is the
   better estimator) while still injecting anchor information. More design work;
   only after (1) is scoped.

5. **Re-pin anchor cadence/delay_K to global-step units.** Currently
   `cadence=5/delay_K=5` are mini-batch ticks ⇒ effective ~2.5 global steps
   (confirmed `anchor_q_updates=14`@step37). A clean re-run at cadence=10/delay_K=10
   ticks would give the intended 5-global-step staleness and tighten the
   comm-amortization accounting (currently ~2× more anchor traffic than the plan
   assumed). This is a hygiene fix, not a fix for the falsification; bundle it into
   whichever of (1)–(4) runs next.

### What a viable next experiment would test
Per (1)+(2): on the verified #25 anchor substrate (keep the green M/Q machinery),
replace `correction_mode=signed_ema` with an **error-feedback** correction on the
PowerSGD residual, and run it BOTH under the locked no-KL surface AND with a small
KL/entropy/length brake. Success criterion unchanged: `val@50 ≥ 0.7414` (within
~1pt of A0 fresh-clean 0.7415) with NO length-explosion alert (the EXP-25
discriminator). If error-feedback also cannot beat 0.7114, the comm-efficient-RL
approach for this surface should pivot away from per-step gradient correction
entirely toward the amortized clean-step / staleness-tolerant regime (#22/#23).

---

## Appendix — gate evidence (the results are interpretable)

The α-sweep is only meaningful because both probe gates were green. From the probe
trainlogs:
- **id-0 (anchor M):** `anchor-load loaded 338/338` (real stale weights);
  `coverage anchor_targets=196 merger_expected=196 set_equal=True missing=[]
  extra=[]` (full coverage, set-equal to the merger set); `dp-reduce MEAN
  ||G||_post/||G||_pre_mean=0.79` (mean not sum — a SUM bug would be ~4×);
  `M-dp-identical cross_rank_max_rel_dev=0.000e+00` (global gradient);
  `||dM_anchor||_mean=1.41e-03 > 0` (M evolves); `anchor_ratio=1.0
  anchor_optimizer_steps=0 anchor_grad_corrected=0 anchor_mask_applications=0`
  (clean, isolated anchor).
- **id-1 (R2+R3):** `[bcast] Q updated=True boundaries=7 changed=5/4
  cross_rank_max_rel_dev=0.0` (anchor-owned Q, broadcast lands, identical across
  ranks); `[bcast] M broadcast targets=196` (M receipt); `[merger] corrected=196
  merger_coldM_fallbacks=0` after warm-up (merger fires on all targets);
  `powersgd_basis_updates=0` (fast net never updates Q — the core inversion holds).
- **off-path parity:** `q_cross_rank_max_rel_dev=0.0`,
  `reconstruction_rel_error=0.02399` (within 1e-3 of the EXP-20 reference 0.024).
- **NaN-free, 50/50 steps each arm.** End-of-log Tracebacks (lines 3173+) are
  post-training DataLoader/UnixTransport teardown noise, AFTER step 50.
- **cold-M guard correct:** `merger_coldM_fallbacks` 196→196→0 (steps 1,2,3) in
  all arms — the matrix-level guard (`spectral_filter.py:296`) returns G_noisy
  unchanged while M is cold, then the merger engages fully from step 3 (= the
  collapse onset). The α=0 SFT-validated default would have silently zeroed the
  gradient on steps 1–2 WITHOUT this guard; it did not.
