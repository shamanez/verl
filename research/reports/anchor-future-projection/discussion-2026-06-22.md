# Next direction — extrapolating the stale anchor gradient ("project it to the future")

**Discussion: Shamane × Sameer — 2026-06-22**
*Summary of what we talked about and what to do next. Successor to issue #38 (dense temporal-drift probe).*

---

## TL;DR

The anchor circuit's gradient is **stale** — computed on weights that are `delay_K`
ticks behind the live (fast-circuit / swarm) weights, so it is a gradient for an
*older* model. The plan: don't consume it as-is — **extrapolate / project it
forward** so it matches the gradient at the *current* weights. This is
**anchor-gradient extrapolation**: we build a **model that projects the stale
gradient (or the weights it was taken at) into the future**.

The seed is Sameer's own group's paper — **"Nesterov Method for Asynchronous
Pipeline Parallel Optimization"** (Ajanthan, Ramasinghe, Zuo, Avraham, Long,
Pluralis Research; [arXiv:2505.01099](https://arxiv.org/abs/2505.01099)). It does
exactly this delay-correction, but with a **fixed, linear, hyperparameter-free**
weight-space extrapolation. Sameer's framing: *don't adopt the whole paper — take
the linear extrapolation kernel and improve it.* The two improvements we discussed:

1. **Linear → learned.** Replace the paper's fixed linear look-ahead with a
   **learned / adaptive projection model** of the anchor's gradient dynamics.
2. **Error feedback.** Because the anchor **periodically receives fresh weights**
   (and a true gradient at them), we get **ground truth** to measure how wrong the
   projection was, and **feed that error back to improve the projection model**.

---

## The paper Sameer pointed to (arXiv:2505.01099) — what it actually does

> "AJ" = **Ajanthan** (first author); **Sameer = Sameera Ramasinghe** (co-author).
> Pluralis Research. Code: `github.com/PluralisResearch/AsyncPP`.

**Problem it solves.** Async pipeline parallelism gives 100% pipeline utilization
but the weights are updated several times between a microbatch's forward and
backward passes ⇒ **stale/delayed gradients**, which hurt convergence.

**Its fix — delay correction in *weight space* (not gradient forecasting).** It
re-purposes the Nesterov Accelerated Gradient (NAG) **look-ahead** step as a delay
corrector. Standard NAG (their Eq. 8):

```
d_t   = γ_t · (w_t − w_{t−1})            # look-ahead = momentum extrapolation
w_{t+1} = w_t + d_t − η ∇f(w_t + d_t)
```

Their modified version for *delayed* gradients (their Eq. 10), where the bar
denotes the stale/delayed quantity (`w̄_t = w_{t−τ}`, `d̄_t = d_{t−τ}`):

```
d_t   = γ_t · (w_t − w_{t−1})
w_{t+1} = w_t + d_t − η·(1 − γ_t)·∇f(w̄_t + d̄_t)
```

- The one change vs. NAG is the **`(1 − γ_t)` discount on the gradient term**. As
  the momentum coefficient `γ_t → 1`, the **look-ahead `d_t` aligns with the delay
  direction** `Δ_t` in weight space (their Prop. 1, `cos(Δ_t, d̄_t) → 1`) — so the
  momentum step *automatically* fills in the staleness gap, while the (stale)
  gradient term is down-weighted toward zero.
- **It is a FIXED LINEAR rule.** The extrapolation `d_t = γ_t·(w_t − w_{t−1})` is a
  **linear combination of the recent weight trajectory** with a **deterministic,
  hyperparameter-free** coefficient (`γ_t = (t−2)/t` in theory; in practice just
  NAdam with `β_1 = 0.99`, plus stage-dependent LR `η_i = η/τ_i^{ρ}` and momentum
  `γ_i = 0.9 + (P−i)/P·0.09`). **This is the "linear weights" Sameer meant.**
- **Assumptions:** constant, *known* per-stage delay `τ_i = ⌊(2(P−i)+1)/2K⌋`, and
  "update directions change slowly." **No learned component. No error feedback.**
- **Result:** beats all async baselines (PipeDream, PipeMare) and **even surpasses
  synchronous GPipe** (WikiText val ppl **27.72 vs 30.63**), up to **1B params**,
  including in the decentralized SWARM framework.

**The opening Sameer sees.** The paper's projection is deliberately minimal — fixed,
linear, no hyperparameters, tuned to a *known fixed* delay. Our anchor setting has
two things the paper doesn't exploit: (a) a stream of `(weights, gradient)` pairs
along the anchor trajectory we could *learn* from, and (b) **periodic
ground-truth** when the anchor refreshes. So we can replace "fixed linear" with
"learned + error-feedback-corrected" — which is also what we'd *need*, because our
real delay is **variable**, not the paper's fixed `τ_i`.

---

## Reframing the direction: anchor-gradient EXTRAPOLATION

Yes — the cleanest statement of the direction is **extrapolation**: we **model /
project** the stale anchor gradient forward in time so it approximates the live
gradient. Two equivalent framings:

- **Weight-space** (the paper's): extrapolate where the weights are going
  (`w̄ + d̄`) and evaluate the gradient there.
- **Gradient-space** (our learned option): directly map *stale gradient → current
  gradient* with a learned projection, trained on the anchor's own trajectory.

The EXP-38 drift data says the projection's job is specifically to **un-rotate**, not
rescale (see below).

## Improvement #2 — error feedback to improve the projection (is this true?)

**Short answer: yes, this is sound — and it's a genuine addition the paper does
not have.** The mechanism:

1. The anchor periodically receives fresh weights and computes a **true** gradient
   `g_true` at the newer point.
2. Just before that refresh, our projection produced a **prediction** `ĝ` of what
   the gradient would be at that point.
3. The **residual** `r = g_true − ĝ` is a supervised error signal. Use it two ways:
   - **Train the projection** (online/self-supervised): the residual is the loss
     gradient for the learned projection model → it keeps improving and adapts to
     *this* model's actual (and variable) staleness.
   - **Accumulate it (error feedback)**: carry `r` forward and add it back into the
     next projected gradient, so projection **bias does not accumulate** — the same
     trick that makes biased gradient compressors behave like unbiased ones.

**Why this is credible here, not just plausible:**
- It fits the structure exactly — EF needs "ground truth arrives later," and the
  **anchor refresh *is* that delayed ground truth**.
- We already have a **proven EF lever**: our current SOTA merger is **B2
  `delayed_ef`** (error-feedback on the PowerSGD codec residual, `G_corr = G_comp +
  λ·δ`), which reaches dense parity. EF-on-the-projection is the same idea applied
  one level up — on the *prediction* residual instead of the *compression* residual.

**Honest caveats:**
- The paper's whole selling point is "no hyperparameters / no learned parts." Going
  learned + EF reintroduces moving parts (a model to train, an accumulator to tune,
  stability to watch). We must show it *beats the free linear baseline*, not just
  that it works.
- It must respect the async constraints: **cross-rank-identical** and **tolerate
  variable staleness** (see below). A learned projection that each rank fits
  differently would violate the first.

---

## Why now — the EXP-38 drift data quantifies the gap (issue #38)

Cosine similarity between the dense gradient at step *t* and *t+k* (how fast it goes
stale) — `reports/dense-run-behaviour/exp38-dense-drift-gsm8k_findings.json`:

| lag *k* (ticks) | grad **cosine** (direction) | grad **sign** | grad **norm** ratio |
|---:|---:|---:|---:|
| 1  | **0.51** | 0.63 | ~1.0 |
| 5  | **0.18** | 0.54 | ~1.0 |
| 10 | 0.02 | 0.51 | ~1.0 |
| 20 | ~0 | 0.50 | ~1.0 |

- **Direction decays fast** — at the operating `delay_K=5` (~2.5 global steps) only
  ~0.18 cosine survives; orthogonal by k≈10. This is the gap extrapolation must close.
- **It's pure rotation** — norm ratio ≈1.0, so the stale gradient has the right
  *magnitude* and has simply **rotated**. The projection's job is to **un-rotate**.
- **Tiny weight moves, big rotation** (`weight_drift ≈ 0.0009` at k=1 already at 0.51
  cosine) ⇒ high local curvature; a projection that knows *how* `g` rotates as `θ`
  moves is exactly the needed signal — and it is **trajectory/curvature** info, which
  is what the paper's slow-changing-direction assumption (and a learned model) bank on.
- **Low-dimensional target** (gradient stable-rank ≈3, rank90 ≈50–65) ⇒ the
  projection has a tractable subspace to model.

---

## Open questions to resolve first

1. **Async-realism / admissibility.** I previously flagged that `.claude/GOAL.md`
   lists "no delay-compensation." The paper largely **resolves** this: weight-space
   extrapolation is *not* asking the slow node to "lead" — it continues a known
   recent trajectory, which is admissible. **But** the paper assumes **fixed known
   delay** and does **stage-dependent** correction; our target is a single slow
   anchor with **variable staleness** serving a swarm, and corrections must be
   **cross-rank-identical**. Closing that gap (variable staleness, identical across
   ranks) is precisely the case for the **learned + EF** upgrade over the fixed
   linear rule.
2. **Parity vs. surpass.** A projection that's a deterministic function of past
   gradients is σ(M)-measurable ⇒ caps at **parity** (still valuable: better
   stability + merger quality at low comm). The **learned, trajectory-trained**
   projection is the interesting case: by modelling how gradients rotate as weights
   move it implicitly captures **curvature** — our ranked **R5 (beyond-diagonal
   curvature)** surpass route. Kill-check (the "diagonal trap"): if the learned
   projection collapses to a per-coordinate rescale, it's just "a better Adam
   diagonal" and won't surpass.

---

## Concrete next steps

1. **Read the paper's method + repo properly** (`PluralisResearch/AsyncPP`,
   arXiv:2505.01099) — lift the exact look-ahead update and how `τ`/`γ` enter, since
   it's the literal starting point. *(Citation now pinned — done.)*
2. **Offline validation on EXP-38 captures first (no GPU).** We already have
   `(weights, gradient)` pairs at multiple lags. Test the core claim directly: does a
   forward-projection raise the stale→live cosine above the raw **0.18 @k=5 / ~0
   @k=10** baseline? Compare three rungs: (a) the paper's **fixed linear**
   look-ahead, (b) a **learned** projection on the trajectory, (c) **learned + EF**
   using held-out refresh points as ground truth. If none lifts the cosine, the idea
   dies cheaply.
3. **If offline-positive, design the merger experiment** on the fixed control
   surface: projected-anchor vs. the current **B2 (`delayed_ef`)** reference, judged
   on **val/score** (not grad_norm), with the diagonal-trap kill-check wired in and
   the variable-staleness / cross-rank-identical constraints enforced.
4. **Open it as the next research issue** (issue-first; natural successor to #38) —
   this doc is the seed.

---

### Provenance / facts
- Paper: **Nesterov Method for Asynchronous Pipeline Parallel Optimization**,
  Ajanthan, Ramasinghe, Zuo, Avraham, Long (Pluralis Research),
  [arXiv:2505.01099](https://arxiv.org/abs/2505.01099). Mechanism quoted above from
  its Eqs. 8/10 + Prop. 1 + Table 1.
- Staleness numbers — `reports/dense-run-behaviour/exp38-dense-drift-gsm8k_findings.json` (issue #38).
- σ(M) ceiling, R3/R5 surpass routes, diagonal-trap, B2 `delayed_ef` — project memory + `runs/SUMMARY.md` + comm-eff-grpo report.
- Async constraint ("anchor always lags, never leads"; cross-rank-identical; variable staleness) — `.claude/GOAL.md`.
