# Next direction — projecting the stale anchor gradient "into the future"

**Discussion: Shamane × Sameer — 2026-06-22**
*Summary of what we talked about and what to do next. Successor to issue #38 (dense temporal-drift probe).*

---

## TL;DR

The anchor circuit's gradient is **stale** — it is computed on weights that are
`delay_K` ticks behind the live (fast-circuit / swarm) weights, so by the time we
fold it into the live update it is a gradient for an *older* model. Sameer's
proposal: instead of consuming the stale gradient as-is, **project it forward**
("to the future") so it better matches the gradient at the *current* weights. The
mechanism he pointed to is **Nesterov-momentum-style look-ahead / extrapolation**
(he referenced a paper — noted in the discussion as "AJ's paper"; exact citation
to confirm). 

Sameer's key addition over plain momentum extrapolation: the projection should
**not be a fixed rule**. Because the anchor circuit *also* receives its own weight
updates over time, the **projection model itself can be learned and improved** as
training proceeds — it adapts to how this model's gradients actually evolve, rather
than applying a static momentum coefficient.

---

## The problem (current state)

- The settled comm-eff base is the **anchor circuit on a PowerSGD codec**: a
  continuously-maintained, `delay_K=5`-stale gradient EMA `M` that owns the
  projection basis `Q`, folded into the fast compressed gradient (see
  `.claude/GOAL.md`, `runs/SUMMARY.md`).
- The anchor gradient is a **valid gradient for the wrong (older) policy** — low
  variance, high bias, and increasingly mis-aligned with the live model the more
  the weights have moved since it was computed.
- In the real async target this is **structural, not incidental**: the anchor is a
  single **slow** node serving a fast **swarm**, so it **always lags and never
  leads**. We cannot make it predict ahead; we can only decide how to *use* a
  lagging reference.

## How bad is the staleness? (EXP-38 / issue #38 data — `reports/dense-run-behaviour/`)

The dense-drift probe measured the cosine similarity between the dense gradient at
step *t* and at step *t+k* (i.e. how fast a gradient goes stale):

| lag *k* (ticks) | grad **cosine** (direction) | grad **sign** agreement | grad **norm** ratio |
|---:|---:|---:|---:|
| 1  | **0.51** | 0.63 | ~1.0 |
| 2  | 0.46 | 0.61 | ~1.0 |
| 5  | **0.18** | 0.54 | ~1.0 |
| 10 | 0.02 | 0.51 | ~1.0 |
| 20 | ~0 (−0.01) | 0.50 | ~1.0 |

What this tells us, and why it *motivates* Sameer's idea:

1. **Direction decays fast.** At the operating `delay_K=5` (≈2.5 global steps) the
   stale gradient keeps only ~0.18–0.5 directional alignment with the live
   gradient; by k≈10 it is essentially **orthogonal**. As a raw optimizer signal it
   is nearly dead — exactly the gap a forward-projection would try to close.
2. **It's pure rotation, not shrinkage.** The norm ratio stays ≈1.0 — the stale
   gradient has the right *magnitude*, it has simply **rotated away**. So the
   projection's job is to *un-rotate* it, not rescale it.
3. **Sign survives longer than direction** (0.63→0.50 only by k≈10-20) — this is
   why our sign-based mergers (`signed_ema`) had partial traction but capped.
4. **Tiny weight moves cause large gradient rotation** (`weight_drift` ≈ 0.0009 at
   k=1 yet cosine already at 0.51) ⇒ the local landscape is **high-curvature**. A
   correction that knows the curvature (how `g` rotates as `θ` moves) is precisely
   what would recover the lost alignment — i.e. a **2nd-order / trajectory** signal.
5. **The target is low-dimensional** (gradient stable-rank ≈3, rank90 ≈50-65) ⇒ a
   learned projection has a *tractable* subspace to model, not all of parameter
   space.

## Sameer's proposal, in two parts

**Part 1 — project the stale gradient forward (Nesterov / look-ahead).**
Treat the staleness as a known lag and extrapolate the gradient (or the
weights it was taken at) toward the current weights before merging — the same idea
as Nesterov momentum's look-ahead step, and the same family as
gradient-**staleness-compensation** methods used in asynchronous SGD (e.g.
Nesterov look-ahead; DC-ASGD-style Taylor/curvature correction). The paper Sameer
cited ("AJ's paper", Nesterov-momentum-based) is the concrete reference — **to be
pinned down**.

**Part 2 — make the projection itself adaptive/learned (Sameer's extension).**
Plain momentum extrapolation uses a *fixed* coefficient. But the anchor circuit
continuously receives its own weight updates, so it naturally produces a stream of
`(weights, gradient)` pairs over its trajectory. We can therefore **learn the
projection** from that stream and keep improving it as training proceeds — a model
of "given a gradient at these weights and this much staleness, where will the
gradient be at the current weights." The projection adapts to *this* model's actual
gradient dynamics instead of assuming a static momentum rule.

This is the genuinely novel part of the discussion: **a learnable, self-improving
forward-projection trained on the anchor's own trajectory**, rather than a
hand-tuned extrapolation constant.

---

## Why this is the right next thing now

Issue #38 ("measure drift **before designing the next comm-eff PP method**") just
delivered the drift numbers above. This discussion is that next method: #38
diagnosed *how* the anchor goes stale; Sameer's proposal is a *mechanism to correct
it*. The hand-off is clean.

## Open tension to resolve first (be honest about this)

`.claude/GOAL.md` currently lists **"no delay-compensation / anchor-lead"** as a
constraint, and our theory note framed any deterministic function of the
stale+current gradient means as **σ(M)-measurable ⇒ capped at parity** (can't
*surpass* dense). Sameer's Part 1 is, on its face, exactly delay-compensation. So
step 0 is to decide whether this is admissible and what bar it targets:

- **Admissibility (async-realism):** the original "no delay-compensation" was about
  not asking the *slow* node to predict ahead of the swarm (impossible). Sameer's
  version is different — a **consumer-side** correction that locally un-rotates a
  lagging reference using its own trajectory. We need to confirm it can stay
  **cross-rank-identical** and **tolerate variable staleness** (the two hard async
  constraints). If yes, it's admissible; if it needs each rank to extrapolate
  differently, it isn't.
- **What bar does it clear — parity or surpass?**
  - A projection that is a deterministic function of the stale gradient + its
    history is σ(M)-measurable → it can **improve stability/parity and merger
    quality**, which is already valuable, but is **capped at dense**.
  - The **learned, trajectory-trained** projection (Part 2) is the interesting case:
    by modelling *how gradients rotate as weights move*, it is implicitly capturing
    **curvature** — which is the one of our two ranked "surpass-dense" routes (R5,
    beyond-diagonal curvature; the other being R3 cross-rank 2nd-moment). To
    actually surpass, the learned projection must inject curvature information that
    dense Adam's diagonal preconditioner does **not** already have (the "diagonal
    trap" kill-check: a learned projection that collapses to a per-coordinate
    rescale is just "a better Adam diagonal" and won't clear the bar).

So the framing for the program: **default expectation = a parity/stability/merger
improvement; the upside bet = the learned projection captures off-diagonal curvature
and becomes a surpass route.** Both are worth it; we should design the experiment so
we can tell which one we got.

---

## Concrete next steps

1. **Pin the citation.** Get the exact "AJ's paper" reference from Sameer (Nesterov
   momentum / staleness projection). Confirm whether it's a look-ahead method, a
   Taylor/curvature staleness-compensation method (DC-ASGD family), or something
   else — it determines the math we port.
2. **Decide admissibility under async-realism** (cross-rank-identical + variable
   staleness). This is a paper/whiteboard check, GPU-free; it gates everything.
3. **Cheap offline validation on EXP-38 captures first (no training run).** We
   already have stored `(weights, gradient)` pairs at multiple lags. Test the core
   claim directly: *does a forward-projection raise the stale→live gradient cosine
   above the raw 0.18 @k=5 / ~0 @k=10 baseline?* Compare (a) plain Nesterov
   extrapolation vs (b) a small learned projection trained on the trajectory. If
   neither lifts the cosine, the idea fails cheaply before we spend a GPU.
4. **If offline-positive, design the merger experiment** on the fixed control
   surface: projected-anchor vs the current B2 (`delayed_ef`) reference, judged on
   **val/score** (not grad_norm), with the diagonal-trap kill-check wired in.
5. **Open it as the next research issue** (issue-first) once steps 1–2 land — this
   doc is the seed.

---

### Provenance / to-confirm
- "AJ's paper" / Nesterov reference — **exact citation pending** from Sameer.
- Staleness numbers — `reports/dense-run-behaviour/exp38-dense-drift-gsm8k_findings.json` (issue #38).
- σ(M) ceiling, R3/R5 surpass routes, diagonal-trap — project memory + comm-eff-grpo report.
- Async constraint ("anchor always lags, never leads") — `.claude/GOAL.md`.
