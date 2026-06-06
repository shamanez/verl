# SURPASS-DENSE STRATEGY — A Ranked, Sequenced Program to Convert Compression's Exploration Surplus into a >Dense Reward Edge

**Status: CONVERGED (strategist + critic, 3 rounds).** This is the `surpass-strategy`
team deliverable. It goes *beyond* the prior team's single gating experiment
(`PATH_TO_SURPASS_DENSE.md`'s EXP-SURPASS-1) into a **ranked, sequenced program** whose
spine is the one real affirmative finding established by the previous team and mechanist:

> **Communication-efficient training sustains a genuinely more-diffuse policy than dense
> — corroborated on the UNCOMPRESSED rollout generator (rollout_ppl 1.40 vs dense 1.24
> @s25) — but this exploration surplus currently FAILS TO CONVERT into reward.** Every
> comm-eff arm sits at or below dense despite more diversity.

So the surpass-dense problem is precisely: **how do we CONVERT the demonstrated exploration
surplus into a >dense val edge?** That conversion question is the spine of everything below.

**Builds on (does NOT re-derive):**
- `runs/EXP-25/COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` (mechanist): compression is dense-grade
  PARITY (PowerSGD-only `oquyeic3` 0.741 ≈ dense 0.754); the residual is a structured BIAS
  (SNR ~42:1, 0.06% energy dropped), NOT zero-mean noise; the §8.1 parity-vs-surpass ceiling;
  the §8.0B real-but-unconverted diversity fingerprint; §6 length-hack-not-low-entropy.
- `runs/EXP-25/PATH_TO_SURPASS_DENSE.md` (prior strategist): the C1–C4 RL≠SFT channels; the
  bias/fluctuation decomposition; the mask-p sweep + Gaussian probe + T0 gating experiment.
- `runs/EXP-25/DEEP_FINDINGS.md`, `ENTROPY_COLLAPSE_FINDINGS.md`, `verdict.md`.

**Two decisive findings this team ADDED (round-1/round-2 with critic), which reshaped the
whole program away from the prior plan:**

1. **The activation mask is TRAIN-ONLY** (verified: `verl/workers/comm_eff/state.py:76`
   `TRAIN_TAG`; `activation_mask.py:301` eligibility assert hard-crashes off the train path;
   rollouts are vLLM with no hooks). So the mask — like PowerSGD and a Gaussian probe —
   perturbs the **gradient/training forward**, NOT the sampling distribution. It is the SAME
   channel class as PowerSGD, which already proved that a train-side perturbation makes real
   diffuse rollouts that **don't convert**. ⇒ "generate more diversity via the mask" is the
   WEAK family; the mask is demoted out of the surpass ranking (it stays as the parity codec).

2. **The val metric is GREEDY, single-sample argmax** (verified: `train.log:478-483` —
   `val_kwargs: do_sample=False, n=1, temperature=0`). A more-diffuse policy spreads
   probability mass *away* from the mode; greedy `mean@1` reads ONLY the mode. So the
   diversity compression produces is **structurally invisible to the bar** unless it
   relocates the trained policy's *argmax* to a better sequence. This is the keystone: it
   splits "conversion" into two mechanistically distinct routes (§2) and proves *why* the
   credit-assignment + temperature levers are necessary rather than optional.

---

## 0. The bar, the prize, and the honest prior

| reference | val@50 (greedy mean@1) | what it is | W&B |
|---|---|---|---|
| **dense** | **0.7536** | no comm-eff — the bar to BEAT | `5e2jpho9` |
| PowerSGD r77 + fresh-clean@5 | 0.7415 | best comm-eff so far (−0.012) | `oquyeic3` |
| signed_ema α=0.5 (knee) | 0.7066 | least-harmful merger | `1wulaelw` |
| signed_ema α=0.3 | 0.6164 | delayed collapse | `r8kc702g` |
| signed_ema α=0.0 | 0.3541 | catastrophic collapse | `uyrpaftw` |

**The bar (pinned, two-sided).** SURPASS = greedy `val@50 = val-core/openai/gsm8k/acc/mean@1`
**> 0.765** (dense 0.7536 + 2σ, σ ≈ 0.0075 from the dense/PowerSGD spread), confirmed on
**2 seeds**. Anything within ±2σ of dense is PARITY, not surpass.

**The honest prior: < 20%.** Every perturbation tried so far has only LOST ground (the table
above is monotone-down from dense). A surpass claim requires a *conversion* mechanism that
relocates the greedy mode above a confident-correct policy's mode — a strong, non-obvious
claim (a confident correct policy *is* the best mode; that is exactly why dense at entropy
0.122 wins greedy-val, COLLAPSE_ANALYSIS §6.1). The burden of proof is on the conversion
mechanism, and this program is designed so that **every outcome — including the likely null —
is interpretable** rather than ambiguous.

**What is NOT a surpass mechanism (settled, do not relitigate):**
- *PowerSGD-noise-as-exploration* — the residual is a deterministic structured bias, not
  zero-mean noise (COLLAPSE_ANALYSIS §2.2).
- *Anchor-corrects-compression-bias* — mechanically PARITY (recovers ≤0.06% off-subspace
  energy; the stale clean anchor is info dense already has fresh; Adam already supplies fresh
  β1=0.9 momentum) (COLLAPSE_ANALYSIS §8.1).
- *signed_ema / blend / inject* — net-harmful (§5) or inert by orthogonality (cos≈0.001).
- *Generating more train-side diversity (mask / entropy-bonus)* — the mask is train-only
  (Finding 1); we are NOT generation-limited, we are CONVERSION-limited.

---

## 1. The spine — we are CONVERSION-limited, not generation-limited

The data say diversity already EXISTS and dense lacks it; it just doesn't pay. At matched
warmed steps (COLLAPSE_ANALYSIS §8.0B), PowerSGD-r77 vs dense:

| metric | psgd r77 | dense | reading |
|---|---|---|---|
| `actor/entropy` @s25 | 0.335 | 0.222 | psgd more diffuse |
| **uncompressed `rollout_ppl` @s25** | **1.401** | **1.238** | **diffuse on the hook-free generator — REAL, not a measurement artifact** |
| `score` @s25 | 0.688 | 0.786 | **diversity does NOT convert — psgd lags** |
| val@50 (greedy) | 0.741 | 0.7536 | ties-not-beats |

The diversity is genuine (corroborated by the uncompressed vLLM generator, which has no
compression hooks) and it is dense's structural deficit — but it is **lost, not harnessed.**
The surpass question is therefore not "make more diversity" (we have it) but "make the
diversity we have PAY in greedy reward." Everything below ranks mechanisms by how directly
and how compression-specifically they attack that conversion.

---

## 2. The two conversion routes (the keystone — derived from greedy eval)

Because val is **greedy argmax mean@1** (Finding 2) but training rollouts are **sampled**
(`do_sample=True`, `train.log:368`), "convert diversity → reward" splits into two
mechanistically distinct routes. Separating them is this document's central contribution and
the source of its interpretability.

**ROUTE A — eval-time diversity → pass@k ONLY.** Sampling more diffusely *at validation* can
raise `pass@k`/`mean@k` (k>1, temperature>0) because diversity buys coverage of the answer
manifold across attempts — the standard reason higher-entropy policies win pass@k. But it can
**never** raise greedy `mean@1`, because greedy reads only the mode. *Honest caveat:* pass@k
is a **different, more-lenient bar** than the operator's stated `val@50=0.7536` (mean@1). We
do **not** let pass@k silently become the goal — that is moving the goalposts. pass@k is a
**logged SECONDARY** (a legitimate compression-specific result if it lands), never the
headline.

**ROUTE B — training-time diversity → relocates the trained policy's MODE → greedy mean@1
surpass.** This is the ONLY path to the actual bar. Mechanism, end to end:

```
diffuse TRAINING rollouts (compression already supplies this surplus, sampled at T>0)
   → BEST-OF-GROUP surfaces a higher-reward completion the confident dense policy misses
   → GRPO credit assignment (group-relative advantage) reinforces it
   → the trained weights' ARGMAX relocates to that better sequence
   → greedy val mean@1 rises above dense.
```

The keystone consequence: **the greedy metric is what makes credit-assignment + temperature
NECESSARY, not optional.** Eval-side diversity is a dead end for the real bar; only
training-side diversity that *relocates the mode* can win mean@1 — and that requires the
conversion step (credit assignment) to not wash out the better completion. The proven leak
(COLLAPSE_ANALYSIS:314, run *backward*): "within-group reward variance shrinks ⇒ GRPO
advantages degrade." Inverted: a diffuse policy has **higher** within-group reward variance ⇒
more informative advantages — **but only if n is large enough** to (a) sample the rare-good
completion and (b) give a stable baseline so its positive advantage isn't averaged away across
the group. That `n` dependence is the conversion bottleneck.

**Why this is compression-SPECIFIC (the load-bearing claim).** Dense's policy is sharper
(rollout_ppl 1.24) ⇒ low within-group reward variance ⇒ extra `n` mostly duplicates the same
completions ⇒ marginal value of `n` is LOW for dense. The diffuse compressed policy
(rollout_ppl 1.40) has high within-group variance ⇒ extra `n` surfaces genuinely different
completions, some better ⇒ marginal value of `n` is HIGH. So **d(val)/dn is predicted steeper
for compressed than dense** — that asymmetry is an edge dense cannot replicate by spending the
same `n`, and it is exactly what the dense control in §3 falsifies.

---

## 3. The ranked, sequenced program

Ranked by `P(beats dense on greedy mean@1) × compression-specificity / cost`, converged with
critic. Every arm: fixed surface (Qwen2.5-1.5B-Instruct, GSM8K, GRPO, batch 128 / mini 64, lr
1e-6, 50 steps, 4–8 GPU); **KL(0.001) + length-cap (1024–2048) on every arm** (COLLAPSE_ANALYSIS
§7 confirmed KL is a GUARDRAIL/enabler that bounds the length-explosion reward-hack, NOT a source
of edge — it makes any perturbation safe to push); the §4 conversion-metric set logged on every
arm. **Note:** `rollout.n=8` and the no-KL/no-entropy surface are LOCKED in the fixed control
surface (`runs/FIXED_CONTROL_SURFACE.md`, operator directive 2026-06-04); BET 1/2/4 below
deliberately change them and MUST be run as **labelled new lineages**, with the matched dense
control on the *same* changed surface so comparability is preserved.

### BET 1 (CO-LEAD) — Credit-conversion via raised `n`, compounded with rollout temperature `T`

The Route-B surpass test. Promoted to lead because it is the only bet that (i) attacks the
*proven* conversion bottleneck (advantage-washing, COLLAPSE_ANALYSIS:314) directly and (ii) is
**compression-specific** (§2: d(val)/dn steeper for the diffuse compressed policy).

- **Mechanism.** Raise `n` (rollouts/prompt) so best-of-group better completions get a clean,
  non-washed group-relative advantage and relocate the greedy mode (Route B). Compound with
  rollout temperature `T>1` (the ONLY knob that perturbs the *sampling* distribution directly —
  `rollout.py:58/177`, default 1.0): `T` GENERATES the training-rollout diversity that is the
  Route-B precondition; `n` CONVERTS it. Compression supplies a third, free, train-side diffusion
  source on top.
- **Why it can beat dense.** The compound `compression-diffusion + T-diffusion + n-credit` is a
  channel dense cannot fully match: dense starts less diffuse, so it gains less from extra `n`
  and from `T` (its argmax is already its confident mode). If the diffuse policy's surfaced-then-
  reinforced completion is genuinely better, its relocated greedy mode beats dense's.
- **Decisive experiment.** `compressed × {n=8, n=16} × {T=1.0, T=1.2}` vs **`dense × {n=8, n=16}
  × {T=1.0, T=1.2}`** (the dense control on the *same* changed surface is MANDATORY — it is what
  separates a compression-specific edge from a generic more-samples/more-temperature win).
  - **SURPASS** = a compressed cell beats dense (greedy mean@1) AND beats the matched
    `dense×{T,n}` cell by > 2σ (≥ 0.765), 2-seeded.
  - **Generic (not comm-eff)** = `dense×{T,n}` catches up to or beats the compressed cell — the
    win was "more samples/temperature," available to dense, not a compression edge. Reported
    honestly as "the surface is exploration/credit-limited; compression adds nothing extra."
  - **Null** = no cell beats dense → conversion via `n`/`T` is insufficient; go to BET 1b.
- **The decisive DISCRIMINATOR is the pass@k COVERAGE CURVE, not the scalar `n16 > dense-n16`.**
  `n16 > dense-n16` is necessary but NOT sufficient — it cannot separate "compression has a
  coverage edge that scales with k" from "compression happened to sit at a better operating
  point." The clean test: log per-prompt `pass@k` (P(≥1 correct in k samples)) for both arms and
  compare how the **compressed-minus-dense pass@k ADVANTAGE grows with k**. Compression-specific
  ⟺ the advantage GROWS with k (its 13%-more-diffuse policy covers answer-modes dense's tight
  policy never samples). A constant offset independent of k = a generic operating-point
  difference → dense-n16 will catch up. **Pre-registered falsifier:** compressed pass@k advantage
  flat in k ⇒ not compression-specific ⇒ killed.
- **Honest magnitude caveat.** The diffuseness edge is SMALL (rollout_ppl 1.40 vs 1.24 ≈ 13%
  more diffuse, NOT 2×), and raising `n` reduces GRPO advantage variance for BOTH arms (a generic
  gain that could swamp the small compression-specific coverage gain). This is exactly why the
  pass@k coverage curve — not the scalar comparison — is the load-bearing instrument.
- **Expected payoff.** The single most likely path to a *real, compression-specific* greedy-bar
  win, because it is the only bet whose mechanism *requires* the surplus dense lacks.
- **Cost.** ~8 arms (4 compressed + 4 dense control) × 50 steps; raised `n` and `T` raise
  rollout cost — the length cap (1024–2048) keeps it bounded. The dense control doubles the cell
  count but is non-negotiable for the compression-specific claim.
- **BET 1b (follow-up if `n` alone under-converts) — rank/best-of-group-weighted advantage.**
  Reweight the GRPO advantage toward the best-in-group completion (a sharper Route-B credit
  mechanism than raising `n`). Not promoted above `n` (more code; changes the loss surface), but
  the natural next rung if best-of-group rises (§4-ii) yet greedy-val stays flat.

### BET 2 (GATE, run FIRST — cheapest) — Surface calibration on dense: is this surface exploration/credit-limited AT ALL?

- **Mechanism.** Before spending the compound grid, ask whether *any* `T`/`n` change lifts
  **dense** above 0.7536. This is the prior team's T0 control, promoted to a cheap gate.
- **Why first.** It is the cheapest possible read on the operator's whole thesis (no codec, ~4
  dense arms: `{n=8,16}×{T=1.0,1.2}`). If raising `T`/`n` lifts dense, RL *is*
  exploration/credit-limited here → the surpass thesis is live and BET 1 is well-motivated. If
  **nothing lifts dense**, the surface is NOT exploration-limited at the policy level → the
  exploration thesis is on its knees → pivot to the honest parity deliverable (BET 5) without
  burning the full compound grid. Either result is decisive and cheap.
- **Cost.** ~4 dense arms × 50 steps. These arms double as BET 1's dense control — zero waste.

### BET 3 (CONTROL) — Codec-free Gaussian-noise probe

- **Mechanism.** Add `σ·N(0,1)` (σ relative to running grad RMS, swept) to the boundary
  activation on the train forward — exactly zero-mean by construction (mechanist §9.1).
- **Role: the interpreter, not a surpass lever.** Given Finding 1 (all train-side perturbations
  are the same channel class), the Gaussian probe does NOT test a new surpass route; it
  *interprets* any compression-specific edge BET 1 finds: is the edge the train-side noise (then
  ideal Gaussian noise reproduces it) or the compression *structure*/diffusion (then it does
  not)? It is also the cleanest codec-independent FALSIFIER: if even ideal zero-mean train-side
  noise cannot move greedy-val, "compression-as-train-side-exploration" is dead regardless of
  codec.
- **The correct kill mechanism — Adam AMPLIFIES the noise on near-zero coords (it is NOT washed
  away by GRPO cancellation).** A zero-mean ξ preserves the mean update (E[g+ξ]=g), so unlike
  signed_ema it does not systematically corrupt direction. BUT Adam normalizes per-coordinate by
  √v_i: on the near-zero bulk (where g_i²≈0, the §1.2 coin-flip coords), if Var(ξ_i) ≳ g_i² the
  Adam update on that coord ≈ sign(g_i+ξ_i) ≈ sign(ξ_i) — a random-sign step of magnitude ~lr,
  re-drawn each step (β1=0.9 momentum smooths across steps but the walk persists). So Adam's
  normalization makes the noise bite HARDER on the near-zero bulk, not softer — it is a random
  walk on the bulk, not an averaged-away jitter. This matches EXP-16 exactly: high-p mask stalled
  at `pearson(masked,rollout)=0.006` with NEAR-DENSE grad norm (rescale fixed magnitude) — the
  variance destroyed DIRECTION without inflating norm. ⇒ BET 3's operative variable is whether
  there is a σ low enough that Var(ξ) stays BELOW g_i² on the solution-bearing coords; mechanist
  §10.2 (bias 7.6% already at p=0.1) says that window is NARROW and may be empty. **Instrument:**
  log the **noise-dominated-coord fraction** (fraction of coords where |ξ_i| > |g_i|); if it is
  >50% at the smallest σ that perturbs val, BET 3 is mechanically a random walk and dead.
- **Cost.** Tiny code; σ-sweep {0.1, 0.3, 1.0}×‖g‖ × 50 steps. Run alongside/after BET 1.

### BET 4 (CONDITIONAL FOLLOW-UP) — Explore-then-exploit anneal

- **Mechanism.** Schedule diversity HIGH early (explore the diffuse regime, surface better
  completions) → LOW late (sharpen onto the better basin found). **Run as ROLLOUT-TEMPERATURE
  anneal (T high→low), NOT mask-p anneal** — temperature is the only knob that perturbs the
  actual sampling distribution directly (Finding 1: the mask is train-only), it is free
  (rollout-side, out of scope for comm), and it directly widens then narrows the sampled support
  (the C2/C4 channel). Mask-p anneal would only anneal a train-side gradient perturbation we are
  not sure converts (BET 3's weakness) — temperature-anneal avoids that.
- **Why it is MORE than speculative — the data SUPPORT the shape.** The entropy
  anti-correlation (COLLAPSE_ANALYSIS:445-446): "the arms that sustain high entropy LONGER
  (α=0.5, α=0) do WORSE; dense runs at entropy 0.122 with the BEST val." Sustained diversity
  hurts; this argues diversity must be **front-loaded then collapsed**, which is exactly the
  anneal shape. Run only after BET 1 establishes whether *constant* T-diffusion converts — if
  constant converts, anneal may push it further; if constant nulls, anneal tests whether the
  *timing* (not the amount) was the leak.
- **Cost.** Schedule only (no new codec) — cheap; 2 arms (T-anneal vs best-constant-T from BET 1).

### BET 5 (PARITY — off the surpass path) — Error-feedback PowerSGD, the banked comm-savings win

- **Mechanism.** Accumulate the dropped `(I−P)g` and re-inject next step (issue #24), removing
  the periodic clean step. Mechanically PARITY at lower comm (COLLAPSE_ANALYSIS §8.1).
- **Role.** The honest comm-efficiency deliverable (GOAL.md parity + savings), run in parallel
  so the project banks a real result even if every surpass bet nulls. The activation mask lives
  here too, as the comm-eff codec — NOT as a surpass lever (Finding 1).

### BET 6 (CONSIDERED, RED-FLAGGED, ranked LAST) — Entropy / exploration-credit reward shaping

Named for completeness (the team-lead asked us to consider the entropy-bonus / reward-shaping
family) and **explicitly deprioritized with reason**, not silently dropped.

- **Mechanism (two variants).** (i) Raw entropy bonus `entropy_coeff>0` to sustain diversity;
  (ii) a *directed* exploration-credit term that rewards within-group ANSWER diversity (attacks
  cause (d): undirected diversity).
- **Why it is the most DANGEROUS arm (critic-escalated red-flag).** COLLAPSE_ANALYSIS §6.2: under
  the no-KL/no-entropy surface the only shaping signal is reward, and the collapse channel is a
  LENGTH-degeneration reward-hack (the H3 loop: low-entropy-gone → longer non-EOS → length
  runaway). A raw entropy bonus pushes the EXACT axis the KL+length-cap brakes are fighting — it
  is the single arm most likely to detonate the length-hack *even with* the brakes. Predicted
  outcome: either the KL cancels the bonus (does nothing) or it re-collapses. ⇒ ranked LAST,
  run only after BET 1, with the length cap mandatory.
- **The directed variant is more interesting but costlier.** Rewarding within-group answer
  diversity is *directed* credit (attacks (d) head-on), but it is NEW reward-shaping code and
  changes the surface. **Scope it as a follow-up ONLY if BET 1's pass@k coverage curve proves the
  diversity is real-but-undirected** (best-of-group rises, greedy-val flat, pass@k flat-in-k) —
  i.e. only if the diagnosis says the diversity exists but points the wrong way.
- **Cost.** Raw bonus = config (cheap but high detonation risk); directed variant = new loss
  code (higher), follow-up only.

### Ranked summary

| rank | bet | route | beats-greedy-dense path | compression-specific? | cost |
|---|---|---|---|---|---|
| GATE | **BET 2** surface calibration on dense | — | does ANY T/n lift dense? | no (control) | cheapest (~4 arms) |
| 1 | **BET 1** raise n + temperature T (+1b rank-weighted adv) | **B** | diffuse training → mode relocation via clean credit | **YES** (d(val)/dn steeper) | ~8 arms |
| 2 | **BET 3** Gaussian probe | — | interprets BET 1; codec-free falsifier | control | tiny |
| 3 | **BET 4** explore-then-exploit anneal (run as TEMPERATURE-anneal, not mask-p — the only direct sampling-distribution knob) | B | front-load diversity, collapse late | yes | cheap (schedule) |
| LAST | **BET 6** entropy / exploration-credit shaping | — | sustain/direct diversity | risky | config (raw) / new code (directed) |
| — | **BET 5** EF-PowerSGD (+ mask as codec) | — | PARITY + comm-savings (NOT surpass) | n/a | low |

---

## 4. The conversion-metric set (the hard instrumentation gate)

The prior plan's update-cosine gate is **necessary but not sufficient** (critic objection #2): a
zero-mean δ gives `cosine = 1 − O(variance)` whether or not it converts — cosine measures
"variance reached the update," not "diversity converted to reward." The fix is to measure
conversion at the **reward** level. Log per-step on EVERY arm:

1. **`rollout_ppl`** (uncompressed-generator perplexity) — *is the diversity present?* The only
   exploration proxy comparable across all runs (COLLAPSE_ANALYSIS §8.0B-NB; do NOT use
   `rollout_probs_diff_mean` across the anchor boundary).
2. **best-of-group reward** `max_b r_b` per prompt — *does diffuse training surface better
   completions?* The Route-B precondition.
3. **within-group reward variance / advantage magnitude** — *does the better completion get a
   non-washed advantage?* The conversion step itself (COLLAPSE_ANALYSIS:314 is this metric
   running backward).
4. **GREEDY `val@50` mean@1** — **THE BAR.**
5. **sampled `pass@k`/`mean@k` at val** (k=8, T=1) — the Route-A diversity probe, **SECONDARY**
   (legitimate compression-specific result if it lands; NOT the headline).
6. **update-cosine to dense** + entropy — retained as the necessary "perturbation reached the
   update / moved the policy" check.
7. **noise-dominated-coord fraction** (BET 3 arms only) — fraction of coords where |ξ_i| > |g_i|.
   >50% at the smallest σ that perturbs val ⇒ the perturbation is a random walk on the near-zero
   bulk (Adam-amplified, §3 BET 3) ⇒ that arm is mechanically dead, independent of val.

**On the update-cosine gate (why metrics 2–3 replace it as sufficient).** A zero-mean δ gives
`cosine = 1 − O(variance)` whether or not it converts — cosine drops below 1 for ANY nonzero
variance, so a dead Adam-amplified noise and a productive exploration BOTH show cosine < 1.
Cosine measures "variance reached the update," NOT "diversity converted to reward." It is
necessary (perturbation must reach the update) but the reward-level pair (best-of-group rising +
within-group variance sustained + greedy val rising) is what is *sufficient* for a conversion
claim.

**Every outcome is interpretable (the anti-loop deliverable):**

| (1) diversity | (2) best-of-group | (3) variance | (4) greedy val | (5) pass@k | verdict |
|---|---|---|---|---|---|
| ↑ | ↑ | sustained | ↑ > 0.765 | ↑ | **ROUTE B SURPASS** (the win) |
| ↑ | ↑ | sustained | flat | ↑ | **ROUTE A only** — real diversity, misses the greedy bar (honest fallback result) |
| ↑ | ↑ | shrinks | flat | flat | **credit-assignment leak** → raise n / BET 1b |
| ↑ | flat | — | flat | flat | **off-task diversity** (the diversity is on irrelevant tokens) → dead end |
| flat | — | — | flat | — | perturbation didn't reach the policy → check cosine/config |

---

## 5. Honest assessment — which bet is most likely to beat dense, and why

**BET 1 (raise n + temperature) is the most likely to actually beat dense on the greedy bar**,
for three reasons, in order of weight:

1. **It is the only bet that is mechanically compression-SPECIFIC.** Its edge *requires* the
   diffuse surplus dense lacks (§2: steeper d(val)/dn for the high-within-group-variance
   compressed policy). BET 3 (Gaussian) and the demoted mask are train-side perturbations of the
   same class that *already failed to convert* (PowerSGD); BET 4 is a timing variant of BET 1.
   Only BET 1 has a mechanism dense cannot replicate by spending the same budget — and the
   mandatory dense control is built to falsify exactly that.
2. **It attacks the PROVEN bottleneck.** The conversion leak is documented
   (COLLAPSE_ANALYSIS:314 — advantage-washing as within-group variance shrinks). BET 1 is the
   direct inverse of that mechanism; the others attack *generation* (which the spine says is not
   the limit) or are controls.
3. **The greedy-eval keystone (Finding 2) makes it necessary, not optional.** Route A
   (eval-time diversity) is a dead end for the real bar; the ONLY path to greedy mean@1 surpass
   is Route B (training-time diversity relocates the mode), and Route B *is* BET 1's mechanism.

**But the honest prior remains < 20%**, and the most likely single outcome is the **"Route A
only" fallback** (row 2 of §4): compression's diversity shows up as a real `pass@k` edge while
greedy `mean@1` ties dense — because relocating the greedy mode above a confident-correct
policy's mode is genuinely hard (a confident correct policy *is* the best mode; dense wins
greedy-val *because* it is confident, COLLAPSE_ANALYSIS §6.1). That fallback is still a real,
publishable, compression-specific result (compression buys diversity that pays in pass@k for
free) — it just is not the operator's stated bar, and the doc reports it as such rather than
moving the goalposts.

**The single most likely way the whole thesis dies cheaply:** BET 2 (the dense surface
calibration gate) shows that *no* `T`/`n` lifts dense above 0.7536 — i.e. the surface is not
exploration/credit-limited at the policy level. Run it first, for ~4 arms, before committing to
the compound grid.

---

## 6. References

**Internal (this fork).**
- `runs/EXP-25/COLLAPSE_GRADIENT_FLOW_ANALYSIS.md` — compression is benign/parity; residual is
  structured bias not noise; §6 length-hack-not-low-entropy; §7 KL-is-guardrail; §8.0B
  real-but-unconverted diversity; §8.1 parity-vs-surpass ceiling; §10.2 mask Jensen-bias.
- `runs/EXP-25/PATH_TO_SURPASS_DENSE.md` — C1–C4 RL≠SFT channels; bias/fluctuation
  decomposition; the prior mask-p + Gaussian + T0 gating experiment this program supersedes.
- `runs/EXP-25/DEEP_FINDINGS.md`, `ENTROPY_COLLAPSE_FINDINGS.md`, `verdict.md`.
- `runs/FIXED_CONTROL_SURFACE.md` — locked surface; `rollout.n=8` + no-KL/no-entropy are pinned,
  so BET 1/2/4 are labelled new lineages.
- Code: `verl/workers/comm_eff/state.py:76` (TRAIN_TAG, mask train-only),
  `activation_mask.py:301` (eligibility assert), `verl/workers/config/rollout.py:58,177`
  (rollout temperature). Val sampling: `train.log:478-483` (greedy mean@1); train sampling:
  `train.log:368` (do_sample=True).
- W&B (`shamanework-pl/verl_compression_research`): dense `5e2jpho9` (0.7536), PowerSGD
  r77+clean5 `oquyeic3` (0.7415, rollout_ppl 1.40), α=0.5 `1wulaelw`, α=0.3 `r8kc702g`, α=0.0
  `uyrpaftw`.

**External prior art (from PATH_TO_SURPASS_DENSE §2.4, carried forward).**
- Barrett & Dherin, *Implicit Gradient Regularization*, ICLR 2021 (zero-mean noise → flat
  minima).
- *The alignment property of SGD noise…*, arXiv 2207.02628 (noise must be the *right kind*).
- *Gradient Regularization Prevents Reward Hacking in RLHF and RLVR*, arXiv 2602.18037 (KL/reg
  as the enabler that closes the reward-hack channel — supports the KL+length-cap guardrail).
- pass@k vs greedy decode: the standard result that higher-entropy policies win pass@k while
  greedy reads only the mode — the literature basis for the Route-A/Route-B split (§2).
