# Theory & literature — anchor-gradient extrapolation

**Companion to [`discussion-2026-06-22.md`](discussion-2026-06-22.md)** (Shamane × Sameer).
*Seed for the issue-#39 deep-research program. Read the discussion doc first — this
adds (a) the broader prior-art lineage, (b) a formalization of what the projection
must do, (c) the parity-vs-surpass theory tied to the project's σ(M) ceiling, and
(d) the GPU-free offline-validation protocol.*

> **Status:** seed / pre-plan. Not a verdict. The numbers are from EXP-38
> (`reports/dense-run-behaviour/*_findings.json`); the prior-art claims are cited
> in §8 and should be re-read at full-text depth by the planner/analyst.

---

## 1. The problem, precisely (from EXP-38)

The anchor circuit serves a gradient computed on `delay_K`-stale weights:
`g(θ_{t−K})` instead of the live `g(θ_t)`. EXP-38 measured how fast that gradient
goes stale by cosine between the dense gradient at step *t* and *t+k*:

| lag *k* (ticks) | GSM8K cos | GSM8K sign | GSM8K ‖·‖ ratio | Big-Math cos |
|---:|---:|---:|---:|---:|
| 1  | **0.507** | 0.63 | 0.89 | **0.018** |
| 2  | 0.464 | 0.61 | 0.99 | 0.011 |
| 5  | **0.176** | 0.54 | 0.92 | 0.011 |
| 10 | 0.023 | 0.51 | 1.01 | 0.005 |
| 20 | −0.008 | 0.50 | 0.98 | 0.004 |

Four facts the projection has to live with:

1. **It is (almost) a pure rotation, not a rescale.** Norm ratio ≈ 1.0 at every
   lag ⇒ the stale gradient has the *right magnitude* and has simply **rotated
   away** from the live gradient. The projection's job is to **un-rotate**, not
   re-scale.
2. **Direction decays fast.** At the operating `delay_K=5` only **0.176** cosine
   survives on GSM8K; orthogonal by *k*≈10. That gap is exactly what extrapolation
   must close.
3. **Tiny weight move, large rotation.** `‖θ_t − θ_{t−1}‖ ≈ 0.0009` already drops
   cosine to 0.51 ⇒ **high local curvature**: a small step in weight space rotates
   the gradient a lot. Curvature is therefore the load-bearing signal (see §3).
4. **Strongly task-dependent.** Big-Math is **near-orthogonal at k=1 (0.018)** —
   the rotation is essentially complete in one step. **GSM8K is the feasible case;
   Big-Math may be infeasible.** Report both; never average. (Ties to the project
   memory entry "staleness budget is TASK-DEPENDENT".)
5. **Low-rank target.** Gradient `rank90 ≈ 50` (GSM8K) / `78` (Big-Math), stable-rank
   ≈ 3 ⇒ the rotation lives in a small subspace, so the projection has a **tractable
   low-dimensional operator** to model.

**Restatement.** We need an operator `R_K : g(θ_{t−K}) ↦ g(θ_t)` that is close to a
norm-preserving rotation, acts on a ~50-dim subspace, and is learned/estimated from
quantities the anchor already has.

---

## 2. Prior art — three tiers of staleness handling

| Method | Space | Correction form | Uses curvature? | Uses error-feedback? | Delay assumption |
|---|---|---|---|---|---|
| Staleness-aware Async-SGD (Zhang 2015) | grad | **down-scale by staleness** | no | no | variable |
| PipeDream (Narayanan 2018) | weight | **stash** stale weights (exact, memory-heavy) | no | no | fixed per-stage |
| PipeMare (Yang 2021) | weight | discrepancy correction + LR reschedule (approx older weights) | partial | no | fixed per-stage |
| **DC-ASGD (Zheng 2017)** | grad | **fixed analytic**: 1st-order Taylor `g(θ_t) ≈ ḡ + H·Δθ`, cheap outer-product (≈ diagonal) Hessian approx | **yes (diagonal)** | no | variable |
| DANA (Hakimi 2019) | weight | gradient at an **estimated future** parameter position | implicit | no | variable |
| PC-ASGD | grad | **predict** (Taylor) + **clip** unreliable predictions | yes | no | variable |
| **Nesterov-async (AJ 2025, the seed)** | weight | **fixed linear**: NAG look-ahead `d_t=γ(w_t−w_{t−1})`, `(1−γ)` grad discount | implicit (momentum) | no | **fixed, known** τ |
| **→ proposed (issue #39)** | grad (or weight) | **learned** `R_K` on anchor trajectory | **yes — target is beyond-diagonal** | **yes — refresh = ground truth** | **variable** |

Two observations drive the design:

- **The seed paper is the *minimal* end of a well-populated spectrum.** It is a
  *fixed, linear, hyperparameter-free* weight-space rule tuned to a *known fixed*
  delay. DC-ASGD is the gradient-space analytic cousin (Taylor + Hessian-approx).
  Both are deterministic functions of the recent (weight, gradient) history.
- **Nobody in this list both (a) learns the correction from the trajectory and
  (b) closes the loop with periodic ground truth.** That is the gap the anchor
  setting uniquely affords (a stream of `(θ, g)` pairs + periodic refresh), and it
  is what we'd *need* anyway because our delay is **variable**, not the paper's
  fixed τ.

---

## 3. What the projection must do — the curvature formalization

First-order Taylor of the gradient field around the stale point:

```
g(θ_t) = g(θ_{t−K}) + H(θ_{t−K})·(θ_t − θ_{t−K}) + O(‖Δθ‖²)
                      └──────────── the rotation ───────────┘
```

So **the "rotation" EXP-38 measured *is* the Hessian-vector product `H·Δθ`.** This is
exactly DC-ASGD's correction. Two consequences:

- Because `‖Δθ‖` is tiny (0.0009) yet the rotation is large (cos → 0.51 in one
  step), the local curvature `H` must be **large / ill-conditioned**. High curvature
  is *precisely* the regime where adding `H·Δθ` back matters most — and where a
  crude (diagonal) approximation is least adequate.
- **Why learned beats fixed *here* specifically:**
  1. our delay is **variable** — the paper's fixed-τ schedule does not apply;
  2. DC-ASGD's outer-product Hessian approx is essentially **diagonal** — cheap but
     blind to the off-diagonal curvature that produces a genuine rotation (a diagonal
     `H` can only rescale per-coordinate, it cannot rotate);
  3. we have **supervised signal** the fixed rules cannot use — `(θ, g)` pairs along
     the anchor trajectory + periodic exact `g_true` at refresh.

The projection model is therefore a **learned surrogate of `H·Δθ`** on the ~50-dim
active subspace — and the interesting bit is whether it can be **beyond-diagonal**
(see §4).

---

## 4. Parity vs surpass — the σ(M) ceiling and the diagonal trap

The project has a formal surpass criterion (memory: "σ(M) surpass ceiling"). Let
`M = σ(g(θ_t), g(θ_{t−K}), …)` be the information in the stale + past dense
gradients.

- **σ(M)-measurability ⇒ parity ceiling.** Any *deterministic function* of past
  dense gradients (the paper's fixed look-ahead, a re-weighting, an accumulation) is
  σ(M)-measurable and **cannot inject information dense-Adam lacks** ⇒ caps at
  **parity**. This is why the prior frontier sweeps (#31/#33) topped out at B2.
- **The escape is curvature.** A learned projection that models `H·Δθ` uses
  **second-order** information that dense-**Adam (diagonal `v_t`) does not have**.
  This is route **R5 (beyond-diagonal curvature)** from the surpass-routes memo — one
  of only two categories that inject info Adam lacks.
- **The diagonal trap (the single most important kill-check).** The control is
  dense-Adam, whose `v_t` is already a per-coordinate (diagonal) second moment. **If
  the learned projection collapses to a per-coordinate rescale, it is "just a better
  Adam diagonal" and will NOT clear the bar.** Surpass requires the projection's gain
  to be **genuinely off-diagonal** (a rotation, `H·Δθ` with non-trivial cross-terms),
  not a diagonal reweighting. Every offline and on-GPU result must be probed for
  this: *is the lift coming from off-diagonal structure, or from a diagonal rescale?*

**Bottom line.** Fixed-linear projection → **parity ceiling** (still valuable: better
merger quality + stability at low comm than raw stale-anchor). Learned
**beyond-diagonal** projection → the only theoretically-credible **surpass** route in
this family. Stating which regime a result is in is mandatory.

---

## 5. Error feedback (improvement #2) — well-precedented in-project

Karimireddy et al. (2019) proved **error feedback** makes a *biased* compressor
converge like SGD by carrying the compression residual forward. Map it one level up:

- the "compressor" is the **projection**;
- the residual is `r = g_true − ĝ`, measurable **at each anchor refresh** (the
  refresh delivers fresh weights + a true gradient — the *delayed ground truth* EF
  needs);
- two uses: **(a) train** the projection (supervised regression on `r`), and **(b)
  accumulate** `r` as classic EF so projection **bias does not accumulate**.

This is structurally identical to our **current SOTA merger B2 `delayed_ef`** (EF on
the PowerSGD *codec* residual, `G_corr = G_comp + λ·δ`, λ=1). EF-on-the-projection is
the same trick on the *prediction* residual instead of the *compression* residual —
so it is precedented here, not speculative.

**Caveat to test.** Refreshes are sparse (every `delay_K`), so the supervised/EF
signal is **low-frequency**. Whether that is enough to fit a useful projection is an
empirical question for the offline gate.

---

## 6. Admissibility — the GOAL.md tension (operator decision required)

[`/.claude/GOAL.md`](../../.claude/GOAL.md) line 62 lists **"no delay-compensation /
anchor-lead"** as an async-realism constraint. Extrapolation *is* a form of
delay-compensation, so this must be resolved before GPU spend.

- **The resolution argument (from the discussion doc).** Weight-space extrapolation
  **continues a known recent trajectory** — it is *not* asking the slow anchor to
  *lead* or *predict the swarm*. Continuing your own observed momentum is admissible;
  forecasting the future state of other nodes is the inadmissible "lead."
- **Two hard constraints any admissible projection MUST satisfy:**
  1. **cross-rank-identical** — a projection each rank fits *differently* breaks the
     shared-`Q` / shared-`M` invariant. (Correction must be derived from
     all-reduced sufficient statistics, slow-varying, identical on every rank.)
  2. **tolerate variable staleness** — the real anchor lags by a *variable* amount;
     a correction tuned to a single fixed τ is inadmissible.
- **⇒ Decision for the operator:** does extrapolation-as-trajectory-continuation
  count as admissible under the GOAL.md constraint, or does GOAL.md need an explicit
  amendment carving out "trajectory-continuation, cross-rank-identical, variable-
  staleness-tolerant" as the permitted form? **Flag this as open-question #1 in the
  issue; it gates everything.**

---

## 7. Offline-validation protocol — the cheap kill-gate (GPU-free)

The whole idea can be killed or confirmed **without a single GPU-hour**, using the
EXP-38 captures (`(θ, g)` pairs at multiple lags, both datasets).

```
Inputs : EXP-38 captures — runs/EXP-38/{gsm8k,big-math}/ (1071 fp32 tensors each + manifest)
Metric : stale→live cosine LIFT.  Baselines (raw, no projection):
           GSM8K  cos@k5 = 0.176,  cos@k10 = 0.023
           Big-Math cos@k1 = 0.018  (near-orthogonal — the hard case)
Rungs  : (a) FIXED-LINEAR   — paper's look-ahead / DC-ASGD diagonal Taylor correction
         (b) LEARNED         — fit R_K on the trajectory, evaluate on HELD-OUT lags
         (c) LEARNED + EF    — use held-out refresh points as ground truth + EF accumulate
```

Gates:

- **Kill-test.** If **no** rung lifts cosine materially above baseline — propose
  threshold **cos@k5: 0.176 → ≥ 0.40 on GSM8K** — the idea **dies cheaply**; no GPU.
- **Diagonal-trap probe (§4).** Decompose each rung's gain into diagonal vs
  off-diagonal. If the lift is purely diagonal, label the result **"parity-only"** up
  front (it cannot surpass even if it improves the merger).
- **Task split.** Run GSM8K and Big-Math **separately**; GSM8K is the feasibility
  case, Big-Math the stress case. A method that only works on GSM8K is still a result
  — just a scoped one.
- **Cross-rank / variable-staleness dry-check.** Confirm the learned form *could* be
  made cross-rank-identical and fit across a *range* of lags (not one fixed τ) before
  proposing the GPU merger.

Only if the offline gate is **positive** do we design the on-GPU merger experiment
(new `correction_mode` in `verl/workers/comm_eff/spectral_filter.py` + projection
module + config knobs in `verl/workers/config/comm_eff.py`), judged on **val/score**
vs the **B2 `delayed_ef`** reference, with the diagonal-trap kill-check wired in.

---

## 8. References

1. **Nesterov Method for Asynchronous Pipeline Parallel Optimization** — Ajanthan,
   Ramasinghe, Zuo, Avraham, Long (Pluralis Research), 2025.
   [arXiv:2505.01099](https://arxiv.org/abs/2505.01099). Code:
   `github.com/PluralisResearch/AsyncPP`. *The seed — fixed linear NAG look-ahead as a
   delay corrector; surpasses synchronous GPipe up to 1B params.*
2. **Asynchronous Stochastic Gradient Descent with Delay Compensation (DC-ASGD)** —
   Zheng, Meng, et al., ICML 2017. [arXiv:1609.08326](https://arxiv.org/abs/1609.08326).
   *The gradient-space analytic cousin: 1st-order Taylor + cheap (≈diagonal) Hessian
   approx. The "un-rotate via curvature" formalization in §3 is DC-ASGD.*
3. **Error Feedback Fixes SignSGD and other Gradient Compression Schemes** —
   Karimireddy, Rebjock, Stich, Jaggi, ICML 2019.
   [arXiv:1901.09847](https://arxiv.org/abs/1901.09847). *Grounds improvement #2;
   same family as our B2 `delayed_ef`.*
4. **PipeMare: Asynchronous Pipeline Parallel DNN Training** — Yang, Lipton, et al.,
   MLSys 2021. [arXiv:1910.05124](https://arxiv.org/abs/1910.05124). *Discrepancy
   correction + LR reschedule; approximates older weights instead of stashing.*
5. **PipeDream: Fast and Efficient Pipeline Parallel DNN Training** — Narayanan et al.,
   2018. [arXiv:1806.03377](https://arxiv.org/abs/1806.03377). *Weight stashing — the
   exact-but-memory-heavy baseline the async methods improve on.*
6. **Staleness-aware Async-SGD for Distributed Deep Learning** — Zhang et al., 2015.
   [arXiv:1511.05950](https://arxiv.org/abs/1511.05950). *Tier-0 baseline: down-scale
   by staleness, no correction.*
7. **PowerSGD: Practical Low-Rank Gradient Compression** — Vogels, Karimireddy, Jaggi,
   2019. [arXiv:1905.13727](https://arxiv.org/abs/1905.13727). *Our codec substrate;
   uses EF.*

Project-internal: σ(M) ceiling / R5 beyond-diagonal / diagonal-trap, B2 `delayed_ef`
SOTA, async-realism constraint → `research/runs/SUMMARY.md`, `.claude/GOAL.md`, and
the comm-eff-grpo report.
