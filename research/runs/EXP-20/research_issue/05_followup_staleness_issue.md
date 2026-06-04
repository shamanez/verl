# Does compressed PP-GRPO survive WITHOUT a fresh full-gradient refresh? — the clean-step realism confound (amortized cost + staleness), and the test plan

> **Operator-raised concern (2026-06-04), captured + analyzed by the orchestrator.** Direct follow-up to **EXP-20 / issue #21**. Where #21 asked "is the ~0.74 just the 10 clean steps?" and answered (internally) "no — the compressed steps carry 57–95% of the within-run gain," **this issue asks the harder, deployment-facing question that #21's framing under-weighted: does the result depend on a crutch the real target setting cannot provide?** If it does, the comm-efficiency claim is benchmarked in a regime that hides the actual difficulty.

---

## 1. The concern (operator)

Our **actual target** is communication-efficient **pipeline-parallel GRPO on community / decentralized GPUs** — the original-paper setting. We compress *because* we lack fast-interconnect clusters: dense (uncompressed) PP training is bottlenecked by the slow boundary transfer and is too slow on community hardware. Compression is the whole point.

But EXP-20's result leans on a **`clean_cadence=5` refresh: a fresh, full-precision boundary gradient injected every 5th step.** In the real decentralized-PP setting that refresh is doubly impossible:

1. **It is the expensive transfer compression exists to avoid.** A clean step sends the full `H`-dim activation/gradient across the pipeline boundary — the exact object the codec was built to shrink.
2. **It is *fresh* / zero-staleness.** Real async + pipelined training delivers cross-stage tensors **stale** — delayed by pipeline depth and network latency. A perfect, current full gradient every 5 steps is precisely what a slow, heterogeneous, decentralized interconnect *cannot* give you.

So a reviewer can argue: *"the method didn't do the work — the periodic fresh full-gradient did, and that's unavailable where it matters."* We have to answer this directly: **does the method survive when the full-gradient refresh is stale, infrequent, or absent — on a task hard enough to discriminate?**

---

## 2. Verdict (orchestrator analysis): the concern is valid and important

EXP-20 proved a real but **narrow** claim — *the compressed (PowerSGD-projected) boundary gradient is a good descent direction **given** frequent, fresh re-anchoring.* It did **not** prove the method works **without** that re-anchoring. Four substantiating points:

**2.1 The clean step silently collapses the advertised compression ratio.** With `clean_cadence=5` over 50 steps: 40 compressed steps (`r` coords each) + 10 clean steps (`H` coords each). Amortized per step:

```
amortized coords/step = (40·r + 10·H) / 50 = (4r + H)/5
  r=77,  H=1536 →  (308+1536)/5 = 368.8  →  1536/368.8 ≈ 4.2× compression   (advertised: 1536/77 ≈ 19.9×)
  r=102, H=1536 →  (408+1536)/5 = 388.8  →  1536/388.8 ≈ 4.0× compression
```

**The headline ~20× budget ignored the clean step's full-`H` transfer entirely.** The honest amortized number is **~4×**, and that 4× still *includes* an unrealistic zero-staleness full transfer. (Cf. issue #21 §2.5, which flagged the basis-sync `H·r` traffic as uncounted; the clean step is the far larger omission.) The clean cadence is really a **comm-efficiency ↔ convergence-speed knob**, and we have measured PowerSGD at exactly **one** point on it (clean@5).

**2.2 Prior project evidence says the full-gradient refresh is load-bearing — for the mask.** Pure-masked GRPO **stalls** without clean steps (EXP-16: reward 0.13 → 0.15 with no clean; clean@4 unlocks 0.13 → 0.62, ~5×). The masked path barely learns on its own. So for the random-mask codec, the clean step is not a minor "bias flush" (as #21 §5.7 frames it) — it is the engine.

**2.3 …but PowerSGD is theoretically different, and that difference is UNTESTED.** Per #21 §5.2/§5.7, the mask is *unbiased but high-variance* (rescaled dropout, variance ∝ p/(1−p) ≈ 19) while PowerSGD is *biased but low-variance* (energy-preserving projection onto the top-`r` activation subspace; bias = the off-subspace energy `√(1−recon²)` ≈ 0.14, small because the boundary gradient is low-rank). **The prediction is that PowerSGD's low-bias step keeps descending without clean steps where the mask cannot** — but no PowerSGD run with `clean_cadence > 5` (let alone ∞, or with staleness) has ever been launched. This is the crux the whole comm-efficiency claim turns on, and it is a hole in the evidence.

**2.4 The val curves across all runs confirm the dependence — and the missing control.**

| run id | codec | clean cadence | GSM8K `val-core/openai/gsm8k/acc/mean@1` trajectory |
|---|---|---|---|
| `5e2jpho9` **dense** | none (every step is full-rank) | — (n/a) | @0 .085 → **@10 .732** → @20 .738 → @30 .742 → @40 .748 → @50 _(pending; see #21)_ |
| `kqozxfr0` PowerSGD r=102 | powersgd | **5** | @0 .077 → @25 .732 → **@50 .744** |
| `oquyeic3` PowerSGD r=77 (byte-matched) | powersgd | **5** | @0 .086 → @25 .710 → **@50 .742** |
| `3yxzzwn3` mask p=0.95 | prf_mask | **5** | @0 .083 → @25 .720 → **@50 .738** |
| `t03dn4nh` mask p=0.9 | prf_mask | **20** | @0 .085 → @10 .083 → @20 .132 → @30 .488 → @50 .690 → @110 **.722** |

(`lwl9yk4y` = `grpo_dense_bigmath_baseline` is genuine dense but **Big-Math / MATH-lighteval, not GSM8K** — excluded from this GSM8K table; it shows the same early-jump shape on its own eval, val 0.536→0.558→0.584 over steps 0/10/20.)

**The story:** more frequent fresh full gradients ⇒ faster convergence. Dense (full every step) hits 0.732 by **step 10** and plateaus ~0.74. clean@5 reaches ~0.73 by **step 25**. The mask at clean@20 is **stuck below 0.13 until its first clean step at step 20**, then climbs — taking ~110 steps to reach 0.72. **But the only "infrequent-clean" point (`t03dn4nh`) is also a different codec and mask rate (mask p=0.9 vs the EXP-20 controls), so it cannot isolate the cadence effect.** There is **no controlled PowerSGD cadence or staleness sweep**. That is the experiment this issue defines.

**2.5 The benchmark may be too forgiving to discriminate.** GSM8K + Qwen2.5-1.5B + 50 steps reaches ~0.74 under dense, clean@5-mask, *and* clean@5-PowerSGD — a 0.53 pp spread (#21 §3). An easy task can make a crutch look unnecessary *and* make a broken method look fine. The discriminating regime is harder/longer training **and** the realistic (stale / no-clean) refresh condition together.

**Bottom line:** EXP-20 is correct but tested the *easy* regime. The realistic regime — **stale, infrequent, or absent full-gradient refresh, on a discriminating task** — is untested, and it is the regime that decides whether the method solves the real problem. The decisive variable the operator named is **staleness of the full-gradient propagation**.

---

## 3. Hypotheses + experiments (controlled, codec is held vs. swept explicitly)

All on the locked control surface (`research/runs/FIXED_CONTROL_SURFACE.md`): Qwen2.5-1.5B-Instruct, GSM8K, vanilla GRPO no-KL/no-entropy, lr 1e-6, bs 128, n=8, resp 16384, 4×H200, `test_freq=10`. Track **val@every-10**, train-reward per step, and — mandatory — the **amortized comm budget** including clean-step bytes.

### H2 (CENTERPIECE — the operator's test): propagate the full-gradient refresh WITH staleness
**Mechanism.** Replace the *fresh* clean step (full gradient w.r.t. current weights θ_t) with a **stale** full refresh — the full-rank gradient/activation computed against weights θ_{t−K} from `K` optimizer steps ago, applied at step `t` — mimicking real async-PP delay. Reuse/adapt the existing `comm_eff.anchor.delay_K` stale-snapshot machinery (it already forwards from a delayed weight snapshot).
**Design.** PowerSGD r=77, clean cadence 5, sweep **delay_K ∈ {0 (=EXP-20 fresh baseline), 1, 5, 20}**.
**Predicted.** Tolerable at small K (re-anchoring still points roughly right); degrades / destabilizes as K grows and θ has moved away from θ_{t−K} (the stale full gradient points to an old optimum). Find the **staleness tolerance** K\*.
**Falsified if** even K=20 matches K=0 (⇒ staleness is free here — strongly *supports* deployability) **or** if K=1 already collapses (⇒ the method needs *exactly* fresh refresh — a serious problem for decentralized PP).

### H1 (the necessary control): clean-cadence sweep for BOTH codecs
**Design.** `clean_cadence ∈ {5, 10, 25, ∞(off)}` × {PowerSGD r=77, mask p=0.95}, GSM8K, fixed surface.
**Predicted (from #21 §5.7 + EXP-16):** the **mask falls off a cliff** as cadence grows (its clean step is a variance reset it cannot do without); **PowerSGD degrades gently** and `clean=∞` still learns (its compressed step is low-bias). The cross-codec *difference* is the headline.
**Falsified if** PowerSGD falls off as steeply as the mask (⇒ its clean step is also load-bearing, not a small flush — the #21 §5.7 theory is wrong and the method does **not** clear the realism bar without a replacement).
This is also the honest **comm-efficiency vs convergence** Pareto curve: plot amortized-bytes/step (which *rises* as cadence shrinks) against val@50.

### H3 (the principled end-state): error feedback removes the need for a full refresh
**Mechanism.** The real deployment likely forbids full-`H` transfers *entirely* — so the durable answer is no clean step ever, with the dropped off-subspace component `(I−P)g` **accumulated and re-injected** (classic PowerSGD error feedback; currently ABSENT — #21 assumption #2, lever §7.2), making the estimator asymptotically unbiased.
**Design.** {EF, no-EF} × {clean@5, clean=∞}, PowerSGD r=77; track val@50 and off-subspace energy `‖(I−P)g‖/‖g‖` (→0 under EF).
**Predicted.** EF + clean=∞ ≈ the clean@5 baseline (EF replaces the periodic flush with continuous correction) ⇒ a method that needs **zero** full refresh.
**Falsified if** EF + no-clean underperforms ⇒ the clean step does more than flush off-subspace bias.
(Three real subtleties to handle — memory at 16K tokens, applying the residual identically to both paired forwards to preserve ρ≈1, basis-rotation of the stale residual; see #21 §7.2.)

### H4 (make the benchmark discriminate)
GSM8K@50 is too forgiving (§2.5). Re-run the decisive H1/H2/H3 contrasts on a **harder / longer** regime (longer horizon, or harder data — e.g. the Big-Math/MATH-lighteval split the project already has a dense control for, `lwl9yk4y`) so the gap between "works" and "leans on the crutch" actually opens. A method that matches dense on GSM8K@50 *and* on a hard task without fresh clean steps is the real result.

### Cross-cutting requirement
**Every future comm-eff comparison must report the amortized comm budget *including* clean-step bytes** (`(c−1)·r + H)/c` for cadence `c`), not the per-compressed-step `r`. The current "20×" is a ~4× method as run.

---

## 4. Connection to the project goal + prior work

- **Goal alignment.** `GOAL.md` / `CLAUDE.md` define the target as comm-efficient PP GRPO on community GPUs. This issue is the test of whether we are actually *on* that target or benchmarking a proxy that assumes a luxury (fresh full gradients) the target forbids.
- **The launcher already warns about this.** The comm-eff launcher header calls `clean_cadence` "the NAIVE cadence method… NOT sustainable… **Opt-in knob only, do not ship it.**" EXP-20 shipped the result *on* that knob. This issue is how we get *off* it.
- **Prior staleness work.** `anchor.delay_K` (stale weight snapshot) + the spectral correction were the project's earlier attempt at a stale-tolerant correction. Findings: the spectral correction as-implemented was **inert by orthogonality**; clean@K re-anchoring "worked" but only as elicitation/parity and **stalled on hard data** (the moment-of-truth pivot). That is *direct prior signal that stale re-anchoring is the hard part* — and motivation to test PowerSGD (a genuinely different, low-bias codec) under the same staleness stressors that defeated the earlier approach.

---

## 5. The decisive question

> **Does PowerSGD-compressed PP-GRPO reach dense parity (~0.74 on GSM8K, and on a harder task) WITHOUT a fresh full-gradient clean step — i.e. with `clean_cadence=∞`, or with only a *stale* (delay_K) full refresh — while honestly counting the clean-step bytes in the budget?**
>
> - **If yes** → the method solves the real decentralized-PP problem; the compressed gradient (optionally + error feedback) carries training on its own.
> - **If it only works with fresh `clean@5`** → we are benchmarking a regime the real deployment cannot provide, the true compression is ~4× not ~20×, and **error feedback or a stale-tolerant correction is mandatory, not optional**, before any deployability claim.

EXP-20 + #21 settled the *within-run* decomposition. **This issue settles whether the win is real under the constraints we actually face.**

---

*Grounds: EXP-20 arms (`runs/EXP-20/`, issue #21), the dense control `ce_dense_50s_gsm8k` (`5e2jpho9`), prior EXP-16/17/18 clean-cadence + anchor/delay_K findings, and `research/runs/FIXED_CONTROL_SURFACE.md`. Operator concern verbatim: the clean step is unrealistic for decentralized community-GPU PP (full gradients are expensive AND arrive stale); test the method by propagating full gradients with staleness and reviewing val across all runs.*
