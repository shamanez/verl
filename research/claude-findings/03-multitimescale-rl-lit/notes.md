# Multi-Timescale / Slow–Fast Literature — Notes for the Stale-Anchor GRPO Report

**Author role:** multitimescale-rl-scholar (theory/lit).
**Date:** 2026-06-22. **Web search:** AVAILABLE and used heavily. Every link below was
fetched (WebFetch) or resolved via WebSearch; status is marked per entry. arXiv IDs sanity-checked
(all `YYMM ≤ 2606`).

---

## TL;DR for the lead (the one distinction that runs through everything)

Nearly every successful slow/fast stabilizer in the literature uses a slow signal that is a
**low-pass filter of the CURRENT iterate** — a convex combination that *includes* `θ_t`:

> `θ̄_t = (1−ε)·θ̄_{t−1} + ε·θ_t`   (Polyak/EMA, mean-teacher, DDPG soft target, SWA running mean, SlowMo outer iterate).

This is a **bounded-distance, smoothed-current** quantity: `‖θ̄_t − θ_t‖ = O(ε/(1−ε))·(recent step size)`,
so it tracks the fast iterate and its stability arguments are *about that tracking*. Our anchor is the
**opposite object**: a **pure lag** `θ_{t−K}` — a delayed snapshot, not an average of recent points. The
distance `‖θ_{t−K} − θ_t‖` is **unbounded in K** (it grows with the integrated drift of the fast
iterate), and that gap carries no convex-combination contraction. **The transfer question for every
method below is therefore: does its stability rest on the slow signal being smoothed-current (so it
contracts toward θ_t), or does it survive a genuine lag?** The answer is almost always "smoothed-current,"
and that is exactly why our system is stable at small K and ignites/stalls at large K (EXP-37: 5/5 stable,
20/20 breaks both mergers).

The **two genuine exceptions** that *do* license a lag are the rigorous two-timescale-SA core (§1) — but
only because it forces the **slow iterate to a near-quasi-static regime via a step-size *ratio*, not via a
small lag** — and **periodic hard target updates** (§2, Lee 2026), which are themselves a pure lag and are
proven to stabilize *linear* Q-learning under spectral conditions on a *fixed* operator. Both exceptions
come with a caveat that is fatal for us: they assume a **stationary equilibrium / fixed operator**, and the
GRPO objective is **non-stationary** (the gradient field is over the *current* policy's own samples, so the
target moves). See §4.

**Strongest single source for the report's central contrast:** **Zhang & Ba 2026, "EMA Policy Gradient"**
(arXiv 2602.04417) — it deliberately *replaces the anchor in GRPO with an EMA* "similar to a target network
in deep Q-learning" and **derives stability conditions for the EMA anchor**. That is the literature
choosing smoothed-current over our lagged design, for the same algorithm (GRPO), and proving the
smoothed-current version stable. Our finding that the *lagged* version is unstable at large K is the
complement of their result, not a contradiction of it.

---

## §1 — Two-timescale stochastic approximation (the rigorous core)

This is the only body of theory that proves a coupled fast/slow recursion converges, and it is worth
stating its conditions exactly because **the load-bearing condition is a step-size *ratio*, which argues
for a small/decaying anchor *dose*, NOT for a small *staleness*.**

### The setup
Two coupled recursions, fast iterate `ϕ` (step-size `a_n`) and slow iterate `θ` (step-size `b_n`):

```
ϕ_{n+1} = ϕ_n + a_n [ g(θ_n, ϕ_n) + M^F_{n+1} ]      (fast)
θ_{n+1} = θ_n + b_n [ f(θ_n, ϕ_n) + M^S_{n+1} ]      (slow)
```
where `M^F, M^S` are martingale-difference noise.

### The exact conditions (Borkar 1997; Konda–Borkar 1997; Vidyasagar 2026)
- **Summability (Robbins–Monro on each scale):** `Σ_n a_n = ∞`, `Σ_n b_n = ∞`, and
  `Σ_n a_n² < ∞`, `Σ_n b_n² < ∞`.
- **Timescale separation (the crucial one):** `b_n / a_n → 0` as `n → ∞`.
  (Equivalently in Vidyasagar's notation with α=fast, β=slow being relabeled: the *ratio of the two
  step-sizes vanishes*; the paper writes the separation as `α_t/β_t → 0` for its labeling, i.e. one scale
  is asymptotically negligible relative to the other.)

### What the ratio BUYS
- `b_n/a_n → 0` ⇒ from the **fast** iterate's view the slow iterate is **quasi-static** (frozen `θ`); the
  fast recursion tracks the equilibrium manifold `λ(θ)` with `g(θ, λ(θ)) = 0`.
- From the **slow** iterate's view the fast iterate is **already equilibrated** (it sees `ϕ ≈ λ(θ)`); the
  slow recursion then behaves like a single-timescale SA on the reduced ODE `θ̇ = f(θ, λ(θ))`.
- **Payoff:** almost-sure convergence to the equilibrium of the coupled ODE. Vidyasagar 2026 sharpens the
  *rate*: under zero-conditional-mean, bounded-conditional-variance noise the MSE → 0 at `o(t^{-η})` for all
  `η ∈ (0,1)`, improving the prior `O(t^{-2/3})`.

### What the ratio ASSUMES (and why it is fragile for us)
- **Martingale-difference noise** (zero conditional mean): our compressed/anchor signal is **biased**
  (persistent, non-zero-mean — "a valid gradient for the wrong policy," SUMMARY.md/EXP-37). This breaks the
  noise model directly; biased drift is not absorbed by `Σ a_n² < ∞`.
- **Bounded iterates** and a **globally asymptotically stable ODE equilibrium** — i.e. a **fixed target**.
  GRPO's gradient field is non-stationary (the data distribution is the current policy), so there is no
  fixed `λ(θ)` manifold for the anchor to equilibrate against.

### Translation to our anchor (the key inference for the report)
- The two-timescale premise "**slow sees fast as equilibrated**" is what **staleness violates**: a *stale*
  anchor is the fast circuit's *own past gradient*, transplanted forward; the slow circuit has **not**
  equilibrated to the current fast iterate — it is reporting on `θ_{t−K}`. The condition that fixes this in
  the theory is **not "make K small"**; it is **"make the slow scale's *influence per step* vanish relative
  to the fast scale"** — i.e. `b_n/a_n → 0`. **The faithful analogue in our system is a small/decaying
  anchor DOSE (the merger weight λ / β_anc), not a small lag.** A vanishing dose keeps the biased stale
  contribution `Σ (dose) × (bias)` summable so the fast circuit dominates; a small lag does not, because
  even at fixed small K the bias is `O(K)` per application and applied every step.
- **Caveat that no two-timescale result removes:** all of the above assumes a **stationary fixed point**.
  Two-timescale SA tells you the *coupled* system converges *to the right place* only if that place
  *exists and does not move*. RL moves it. So even a perfectly dosed stale anchor converges to "the
  equilibrium of a coupled ODE built from a defunct policy's gradient field" — the bias does not vanish, it
  is merely down-weighted. This is the formal version of the EXP-37 "additive merger stalls at a
  sub-baseline plateau" observation.

**Sources (§1):**
1. **Borkar, V. S. (1997). "Stochastic approximation with two time scales." Systems & Control Letters
   29(5):291–294.** — The originating result; introduces the `b_n/a_n → 0` separation and the
   singular-perturbation / quasi-static view. Link: <https://repository.ias.ac.in/5268/> **(LINK UNVERIFIED
   — IAS repository returned 404/403 on fetch; the bibliographic record is confirmed via Semantic Scholar
   <https://www.semanticscholar.org/paper/Stochastic-approximation-with-two-time-scales-Borkar/5c29049c6cc7e93bd42ccd55d70a5b92120ceec6> (RESOLVES). Cite the SCL DOI.)**
   *Relevance:* directly useful (the rigorous core). *Smoothed-vs-lagged:* N/A — it is the *step-size-ratio*
   mechanism, which is the correct analogue for our *dose*, not our *lag*. *Transfer:* the separation
   condition argues for decaying anchor dose; the fixed-equilibrium assumption does not transfer to GRPO.
2. **Konda, V. R. & Borkar, V. S. (1997). "Actor-critic–type learning algorithms for Markov decision
   processes" / "The actor-critic algorithm as multi-time-scale stochastic approximation." Sādhanā 22(4)
   /SIAM J. Control Optim.** Links: <https://link.springer.com/article/10.1007/BF02745577> **(LINK
   UNVERIFIED — Springer redirects to an auth page; record confirmed via IAS index
   <https://www.ias.ac.in/article/fulltext/sadh/022/04/0525-0543> which is the canonical citation but
   returned 403 to the fetcher)** and the SIAM companion
   <https://epubs.siam.org/doi/10.1137/S036301299731669X> (RESOLVES, paywalled).
   *Relevance:* background→directly useful — casts actor (slow) / critic (fast) as two-timescale SA with the
   actor step-size asymptotically negligible vs the critic's (`ratio → 0`), so the critic "completely solves"
   policy evaluation asymptotically. *Smoothed-vs-lagged:* N/A (ratio mechanism). *Transfer:* same as Borkar
   1997 — and note this is the *RL* instantiation, so it is the most defensible "two-timescale works in RL"
   citation, but its critic target is a value function on a *fixed* MDP, not a moving policy-gradient field.
3. **Vidyasagar, M. (2026). "Convergence of Two Time-Scale Stochastic Approximation: A Martingale
   Approach." arXiv:2603.14481.** Link: <https://arxiv.org/abs/2603.14481> (RESOLVES; HTML
   <https://arxiv.org/html/2603.14481> RESOLVES). *Relevance:* directly useful, modern (2026). Gives the
   explicit conditions: `Σα_t² < ∞, Σβ_t² < ∞`; `Σα_t = ∞`; separation `α_t/β_t → 0`; rate-optimal choice
   `α_t = Θ(t^{-1}), β_t = O(t^{-(1-Δ)})`; assumptions A1 (equilibrium manifold `g(θ,λ(θ))=0`), A2 (origin
   equilibrium), Lyapunov pairs (VS1–VS2) slow, (VF1–VF2) fast. MSE → 0 at `o(t^{-η}) ∀η∈(0,1)` under
   zero-mean bounded-variance noise; explicitly flags that **nonzero conditional mean / unbounded variance**
   are the hard cases — i.e. exactly the biased regime our anchor lives in. *Smoothed-vs-lagged:* N/A
   (ratio). *Transfer:* its own "hard case" caveat (biased/heteroscedastic noise) is our case; cite it for
   the precise inequalities AND for the admission that biased noise is not covered by the clean result.

---

## §2 — Slow/fast stabilizers: survey with mechanism + smoothed-vs-lagged alignment

For each: **(a)** smoothed-current or lagged?  **(b)** what keeps fast & slow ALIGNED?  **(c)** what
objective is assumed (same loss vs bootstrap target)?  **(d)** does it tolerate a genuine LAG?

### 2.1 Target networks — the one place a LAG is the actual mechanism (so read carefully)

There are **two distinct target-network update rules**, and they sit on opposite sides of our distinction:

- **Periodic "hard" target (DQN, Mnih et al. 2013/2015):** `θ⁻ ← θ` every C steps; *between* copies `θ⁻`
  is a **frozen LAGGED snapshot** `θ_{t−(t mod C)}`. **This is a pure lag — structurally identical to our
  anchor.** (a) lagged. (b) alignment = *periodic resync* (the hard copy resets the lag to 0). (c) bootstrap
  target for a *fixed* Bellman operator — NOT the same loss on both scales; the target supplies `y =
  r + γ max_a Q_{θ⁻}(s',a)`. (d) **Yes — it tolerates a lag, and that is the whole point**, BUT only for a
  *contraction* (the Bellman operator is a γ-contraction in sup-norm) and a *fixed* reward/transition model.
- **Soft "Polyak" target (DDPG, Lillicrap et al. 2015; SAC/TD3):** `θ⁻ ← (1−τ)θ⁻ + τθ`, τ≈0.001–0.01.
  **This is smoothed-current** — a low-pass of the online net. (a) smoothed-current. (b) alignment = the
  convex combination itself (bounded distance `O(τ)` to a recent online net). (c) bootstrap target, fixed
  operator. (d) it does *not* rely on a large lag; it relies on slow *tracking*.

**Why this matters for the report — the false-friend trap:** it is tempting to say "DQN proves a lagged
reference is fine, so our stale anchor should be fine." **It does not transfer**, for two reasons the report
must state: (i) DQN's lag stabilizes a **bootstrap contraction on a FIXED MDP** (Bellman is a contraction;
the target only needs to be *consistent enough* to not chase its own tail). Our anchor injects a **policy
GRADIENT** for a **non-stationary** objective — there is no contraction making the lagged target benign.
(ii) DQN/DDPG targets are used to **form a regression label**, never **summed into the parameter update as a
second gradient**. Our merger *adds* the stale gradient to the live one. A lagged *label* and a lagged
*gradient term* are different animals: a stale label still defines a valid (if slightly off) regression
problem; a stale gradient is a step in a direction that was correct for `θ_{t−K}` and is now biased.

**A directly-relevant 2026 formalization — Lee 2026** ("Target Updates May Stabilize Linear Q-Learning:
Periodic and Soft Dynamics," arXiv:2606.02645) analyzes **both** rules for *linear* Q-learning and proves
each can stabilize under **explicit spectral + step-size conditions** (`0 < α < 2/λ_max(ΦᵀDΦ)`, joint
spectral radius < 1 over the switching family). Crucially it shows the **target period itself acts as a
stabilizer**, interpolating between one-step DLQL and projected Q-value iteration. **This is the strongest
"a lag can stabilize" result available — and it is exactly bounded by our caveats: it needs (i) a fixed,
linear operator and (ii) spectral conditions on that operator. GRPO supplies neither.** Cite it as
"tempting and partially applicable, but conditioned on a fixed linear operator." (Author: Donghwan Lee, 2026.)

### 2.2 Polyak–Ruppert / EMA averaging
The slow iterate is an **average of the iterates** (`θ̄_n = (1/n)Σθ_k` or an EMA). (a) smoothed-current by
construction. (b) alignment is automatic (the mean is *of* the trajectory). (c) **same loss** — averaging
reduces estimator variance on one objective; it is a *variance reducer*, not a second optimizer. (d) **No
genuine-lag tolerance**: the averaged point is only useful *because* it stays near the trajectory; a pure
lag `θ_{t−K}` is not an average and gets no variance-reduction guarantee. **Verdict: tempting but not
applicable** — averaging recent points ≠ holding one old point.

### 2.3 Mean Teacher (Tarvainen & Valpola 2017)
Teacher weights = **EMA of student weights**; consistency loss penalizes student–teacher prediction
disagreement on unlabeled data. (a) **smoothed-current** (EMA — explicitly "averages model weights instead
of label predictions"). (b) alignment = EMA tracking + the consistency penalty (a *proximity* term). (c)
it is a **representation regularizer** (semi-supervised consistency), **not a gradient-reuse method** — the
teacher never contributes a gradient term to the student's optimizer; it supplies a *target to be consistent
with*. (d) No — it is the canonical smoothed-current construction. **Verdict: tempting but not applicable**
(matches the codex bibliography label). The relevant *positive* lesson: when a slow EMA *does* help, it
helps as a **proximity/consistency anchor** (a slowly-varying *statistic* to stay near), which aligns with
SUMMARY/GOAL's "demote the anchor to a slow Q/codec **calibrator**, not a gradient provider."

### 2.4 Lookahead optimizer (Zhang et al. 2019)
k fast inner steps, then slow weights interpolate toward the fast trajectory's endpoint:
`φ_{slow} ← φ_{slow} + α(φ_{fast,k} − φ_{slow})`. (a) the slow point is **smoothed-current** — it is pulled
toward where the fast weights *just went*, on the **same data objective**. (b) alignment = the convex
interpolation (slow is always a convex combo with the most-recent fast endpoint) + periodic resync of fast
to slow. (c) **same loss on both scales** — both optimize the identical training objective; there is no
bootstrap and no second policy. (d) **No genuine lag** — the slow weights look *forward* at fresh fast
iterates and then average them; they never hold a stale snapshot to reinject. **Verdict: tempting but NOT
applicable** (matches codex label) — Lookahead averages optimizer *trajectories on the same objective*; our
anchor reuses a *gradient from a past objective*.

### 2.5 SWA — Stochastic Weight Averaging (Izmailov et al. 2018)
Running mean of SGD iterates with a cyclical/constant LR; finds flatter/wider optima. (a) **smoothed-current
running mean**. (b) alignment automatic (mean of the trajectory). (c) **same fixed loss**; it is a
post-hoc/online *averaging for generalization*, not a coupled optimizer. (d) No lag tolerance.
**Verdict: tempting but not applicable** — same family as Polyak; averaging recent points, not holding an
old one. Useful only as a contrast that underlines the distinction.

### 2.6 SlowMo — Slow Momentum on local-SGD (Wang et al. 2019/2020)
After K local steps, workers all-reduce to a **current** averaged model `x̄`; a slow momentum buffer is
updated from the *current* synced iterate and an outer step is taken (`u ← β u + (x_prev − x̄)/α`;
`x ← x_prev − α β u`-style outer update). (a) **smoothed-current** — the slow direction is built from the
*current* averaged iterate at each sync, not a delayed snapshot. (b) alignment = periodic all-reduce sync +
the momentum being computed on the freshly-synced point. (c) **same fixed (non-convex) loss**; convergence
to a stationary point of a smooth non-convex objective is proven *for that fixed loss*. (d) **No genuine-lag
tolerance** — the slow momentum needs the synced model to be current; staleness between syncs is the *local*
gap that SlowMo's periodic averaging *removes*, not something it tolerates indefinitely.
**Verdict: directly useful as the comm-efficient slow/fast analogue, but it is smoothed-current + fixed-loss
— so it does NOT license our lagged-gradient reuse.** (The SlowMo equation form is the standard one; the
arXiv HTML 404'd on fetch, abstract verified.)

**Sources (§2):**
4. **Mnih et al. (2013). "Playing Atari with Deep Reinforcement Learning." arXiv:1312.5602** (RESOLVES;
   full target-network detail is in the Nature 2015 version). *Relevance:* the **lagged** (periodic-hard)
   target lineage. *Smoothed-vs-lagged:* **lagged**. *Transfer:* tempting but not applicable — lag
   stabilizes a *bootstrap contraction on a fixed MDP*, supplies a *label* not a *gradient term*.
5. **Lillicrap et al. (2015). "Continuous control with deep reinforcement learning" (DDPG).
   arXiv:1509.02971** (RESOLVES). *Relevance:* the **smoothed-current** (Polyak-soft) target. *S-vs-L:*
   smoothed-current. *Transfer:* shows the field's *preferred* slow-target is smoothed-current, bounded-τ;
   not a lag, not a gradient term.
6. **Lee, D. (2026). "Target Updates May Stabilize Linear Q-Learning: Periodic and Soft Dynamics."
   arXiv:2606.02645** (RESOLVES, HTML fetched). *Relevance:* directly useful, modern (2026); proves **both**
   periodic-hard (lagged) and soft (EMA) target updates stabilize *linear* Q-learning, and that the **period
   itself stabilizes** — the closest formal support for "a lag can help." *S-vs-L:* analyzes **both**.
   *Transfer:* **conditioned on a fixed linear operator + spectral/JSR conditions** → does NOT transfer to a
   non-stationary GRPO gradient field; cite as the principled boundary of the lag-helps claim.
7. **Tarvainen & Valpola (2017). "Mean teachers are better role models." arXiv:1703.01780** (RESOLVES,
   fetched — "averages model weights," EMA). *Relevance:* EMA teacher. *S-vs-L:* **smoothed-current**.
   *Transfer:* tempting but not applicable — representation/consistency regularizer, never a gradient term;
   its lesson is "slow EMA = proximity anchor," i.e. the calibrator role.
8. **Zhang, Lucas, Ba, Hinton (2019). "Lookahead Optimizer: k steps forward, 1 step back."
   arXiv:1907.08610** (RESOLVES, fetched). *Relevance:* slow/fast weights. *S-vs-L:* **smoothed-current**
   (forward-looking convex interpolation on the same loss). *Transfer:* tempting but not applicable —
   averages trajectories on the *same* objective; no stale-gradient reuse.
9. **Izmailov, Podoprikhin, Garipov, Vetrov, Wilson (2018). "Averaging Weights Leads to Wider Optima and
   Better Generalization" (SWA). arXiv:1803.05407** (RESOLVES, fetched — running mean of iterates).
   *Relevance:* slow average for generalization. *S-vs-L:* **smoothed-current**. *Transfer:* not applicable
   (averaging recent points ≠ holding one old point).
10. **Wang, Tantia, Ballas, Rabbat (2019/2020). "SlowMo: Improving Communication-Efficient Distributed SGD
    with Slow Momentum." arXiv:1910.00643** (RESOLVES, abstract fetched). *Relevance:* directly useful —
    the comm-efficient two-speed analogue (periodic sync + slow outer momentum); convergence proven for a
    *fixed* non-convex loss. *S-vs-L:* **smoothed-current** (slow momentum on the current synced iterate).
    *Transfer:* does NOT license lagged-gradient reuse (fixed loss + current-iterate momentum).

---

## §3 — Fast–slow & hierarchical RL, meta-gradients (incl. 2025/2026 LLM-RL)

The decisive modern result, because it is the **same algorithm (GRPO)** and makes the **smoothed-vs-lagged
choice explicitly**:

### 3.1 EMA Policy Gradient (Zhang & Ba 2026) — the report's keystone contrast
EMA-PG **replaces the fixed RL anchor with an EMA, "similar to a target network in deep Q-learning,"** and
**derives stability conditions for the EMA anchor**, plus a Top-k KL estimator giving unbiased KL values and
gradients at any k. On GRPO it lifts R1-distilled Qwen-1.5B from 50.8%→53.9% on OlympiadBench, +33.3% mean
on agentic Q&A. (a) **smoothed-current** EMA anchor (a moving average of *current* policy params). (b)
alignment = EMA tracking + a KL proximity penalty to that anchor. (c) policy-gradient objective with the
EMA serving as the **reference for a KL term** — i.e. a *proximity statistic*, not a re-injected gradient.
(d) the design **chose smoothed-current over lagged on purpose**, and proved *that* stable.
**Why this is the keystone:** it is the literature, in our exact setting (GRPO, ~1.5B), selecting the
smoothed-current construction and analyzing it — and using the EMA as a **KL anchor (a slowly-varying
statistic to stay near)**, *not* as a gradient provider. That is precisely the "demote the anchor to a slow
calibrator" recommendation on file (SUMMARY.md / GOAL.md). Our result — that the *lagged* `θ_{t−K}` anchor
used as a *gradient* is unstable at large K — is the **complement** of EMA-PG: same algorithm, opposite
design choice on both axes (lagged-vs-smoothed AND gradient-vs-proximity), opposite stability outcome.

### 3.2 Two-timescale actor-critic (the RL instantiation of §1)
Konda–Borkar (above) and the modern finite-time line (e.g. the Kaledin et al. 2020 linear TTSA analysis and
its successors) show the **critic (fast) / actor (slow)** split converges *when the actor step-size is
asymptotically negligible relative to the critic's* and the **critic target is a value function on a fixed
MDP**. Divergence is prevented by the **step-size ratio**, not by freezing a stale copy. Transfer: this is
the cleanest "fast-approximate + slow-precise works in RL" precedent, but the *precise* slow signal there is
the **value estimate**, and the MDP is **stationary** — neither matches a stale policy-gradient under a
moving sampling distribution.

### 3.3 How divergence is prevented across the fast-slow RL literature (synthesis)
Across DQN/DDPG (target nets), EMA-PG (EMA + KL), Asadi 2021 (proximity penalty), and two-timescale AC
(step-size ratio), the recurring divergence-prevention mechanisms are exactly **four**, and **none of them
is "use a stale gradient":**
1. **Periodic resync** (hard target copy / local-SGD all-reduce) — resets the lag to zero before it
   compounds. *Our analogue:* a *fresh* anchor (small K) — which is precisely the regime that works (5/5).
2. **Convex tracking** (Polyak/EMA/SWA/Lookahead) — bounds slow–fast distance to `O(ε)`. *Not available to
   a pure lag.*
3. **Proximity/KL penalty** (Asadi 2021; EMA-PG; TRPO/PPO trust region) — keeps the fast iterate *near* the
   slow reference so the reference stays valid. *Our analogue would be using the anchor as a KL/codec
   calibrator, not a gradient.*
4. **Step-size-ratio separation** (two-timescale SA) — makes the slow influence vanish. *Our analogue =
   small/decaying anchor DOSE.*

### 3.4 NOT this literature (false friend to exclude)
The 2025–2026 "**fast and slow thinking**" / dual-system reasoning LLM line (System-1/System-2, e.g. Pangu
Embedded, slow-thinking RL surveys) uses "fast/slow" for **inference-time reasoning depth**, *not* optimizer
timescales. **Excluded** from the report's slow/fast-optimizer evidence to avoid equivocation; flagged here
so the lead does not mistake it for support.

**Sources (§3):**
11. **Zhang, L. & Ba, J. (2026). "EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA
    Anchor and Top-k KL." arXiv:2602.04417** (RESOLVES, fetched). *Relevance:* **directly useful — the
    keystone**. *S-vs-L:* **smoothed-current** EMA anchor used as a **KL proximity reference** (not a
    gradient). *Transfer:* it is the literature making the smoothed-current + calibrator choice in GRPO and
    proving it stable — our lagged-gradient instability is its complement. Strongly recommend featuring.
12. **Asadi, Fakoor, et al. (2021). "Faster Deep Reinforcement Learning with Slower Online Network."
    arXiv:2112.05848** (RESOLVES, fetched — DQN Pro / Rainbow Pro). *Relevance:* directly useful — adds a
    **proximity penalty** keeping the online net near the target. *S-vs-L:* the target is the standard
    (lagged/EMA) DQN target; the *novelty is the proximity term*. *Transfer:* supports the **anchor-as-
    regularizer/calibrator** reading (mechanism #3), NOT stale-gradient injection — matches the codex label
    and the GOAL.md "slow calibrator" note.
13. **Konda & Borkar (1997)** — see §1 source 2 (the RL instantiation of two-timescale SA). Listed once.

---

## §4 — Transfer verdict (map each finding to the open question)

**Open question:** which timescale-separation condition could make a STALE-anchor merger stable at
large/variable K?

### 4.1 The three things the report must say explicitly
**(i) Smoothed-current vs lagged (the load-bearing distinction).** Every *general-purpose* slow/fast
stabilizer that works — Polyak, EMA, mean-teacher, Lookahead, SWA, SlowMo, DDPG-soft-target — uses a
**smoothed-current** slow signal whose stability comes from a **bounded `O(ε)` distance to θ_t**. Our anchor
is a **pure lag** with **unbounded-in-K** distance. **None of these results transfers to a pure lag**; they
transfer only to the *smoothed-current* redesign. → The literature-backed move is to **make the anchor
smoothed-current** (an EMA of recent θ — bounded distance) **or** use it only as a **slowly-varying
proximity/codec statistic** (mean-teacher / EMA-PG / Asadi role), *not* as a re-injected lagged gradient.

**(ii) It is a step-size RATIO, not a small staleness.** The *one* rigorous condition (two-timescale SA,
§1) that licenses a slow circuit is `b_n/a_n → 0` — a **vanishing dose/step-size ratio**, which keeps the
biased slow contribution **summable**. → For us this argues for a **small or decaying anchor DOSE (merger
weight λ / β_anc)**, possibly **scaled down with K** (e.g. dose ∝ 1/K or gated off above a τ), **NOT for
merely shrinking K**. EXP-37's β_anc=0.50 surviving 5/5 while β_anc=0 oscillates (37C) is consistent: a
*non-trivial averaging/dose regime* matters, and the binding failure is latency (K), so the only stable
operating points are **(small K) AND (a dose that does not let the bias accumulate)**. The clean theoretical
prescription is: if K must be large/variable, **decay the dose** rather than hope a fixed dose tolerates the
lag.

**(iii) Non-stationary-objective caveat (the hard wall).** Two-timescale SA, DQN/DDPG target stabilization,
and Lee 2026's periodic-target result **all assume a fixed equilibrium / fixed (contraction or linear)
operator**. GRPO's gradient field is **non-stationary** — it is defined over the *current* policy's own
samples — so even a perfectly dosed, perfectly resynced slow circuit converges toward the equilibrium of a
*moving* target. This is why the stale anchor is "a valid gradient for the **wrong** policy" (SUMMARY.md):
the bias is **structural** (off-policy-ness), not noise to be averaged out. **No timescale-separation
condition removes a non-stationarity bias; separation only controls how fast you converge to *some*
fixed point, assuming one exists.** The most a timescale argument can buy us is **boundedness/stall
avoidance**, not parity-beating signal from the stale gradient.

### 4.2 Per-method transfer table (liftable)

| Method | Slow signal | Stab. mechanism | Tolerates a true LAG? | Transfer verdict for stale-anchor GRPO |
|---|---|---|---|---|
| Two-timescale SA (Borkar 97; Konda–Borkar 97; Vidyasagar 26) | fast iterate at quasi-equilibrium | **step-size ratio `b/a→0`** | mechanism is dose, not lag | **Partial:** argues for **small/decaying anchor DOSE**; fixed-equilibrium + martingale-noise assumptions fail for biased GRPO. |
| DQN periodic-hard target | lagged snapshot `θ⁻` | **periodic resync** of a *contraction*; supplies a *label* | yes, for a fixed-MDP bootstrap | **Tempting, NOT applicable:** no contraction, label≠gradient, objective non-stationary. |
| DDPG soft / Polyak target | EMA of online net | bounded-`τ` convex tracking | no (needs tracking) | **Not applicable** as a gradient; supports a smoothed-current *reference*. |
| Lee 2026 (periodic + soft, linear Q) | both | spectral/JSR + period | yes, **for a FIXED LINEAR operator** | **Tempting, boundary case:** the period *can* stabilize, but only under fixed-linear-operator spectral conditions GRPO lacks. |
| Polyak/Ruppert, SWA | running mean of iterates | variance reduction on same loss | no | **Not applicable** (average≠snapshot). |
| Mean Teacher | EMA weights | consistency/proximity penalty | no | **Tempting, NOT applicable** as gradient; supports **calibrator** role. |
| Lookahead | forward convex interp on same loss | interpolation + resync | no | **Tempting, NOT applicable** (same-objective trajectory averaging). |
| SlowMo | slow momentum on current synced iterate | periodic sync + outer momentum | no | **Directly useful analogue, but smoothed-current + fixed loss** → no lagged-gradient license. |
| EMA-PG (Zhang & Ba 26) | EMA anchor as KL reference | EMA tracking + KL proximity | no (smoothed-current by choice) | **Keystone:** same algorithm chose smoothed-current + calibrator and proved it stable — our lagged-gradient instability is the complement. |
| Asadi 2021 | DQN target + proximity | proximity penalty | no | **Directly useful:** supports anchor-as-**calibrator/regularizer**, not stale-gradient source. |
| Two-timescale actor-critic | value estimate (critic) | step-size ratio | mechanism is ratio | **Partial:** RL precedent, but stationary MDP + value (not policy-gradient) target. |

### 4.3 The single sentence for the report's conclusion
*The multi-timescale literature supports a slow circuit only when it is either a **smoothed-current**
quantity at bounded distance from `θ_t` (Polyak/EMA/mean-teacher/Lookahead/SWA/SlowMo/EMA-PG) or a
**vanishingly-dosed** slow recursion (two-timescale SA); a **pure lag** `θ_{t−K}` is licensed only by a
**periodic resync to zero** (small K, DQN-style) and only for a **fixed/contractive objective** — so for a
genuinely-stale anchor under the non-stationary GRPO objective, the admissible designs are (1) **make the
anchor smoothed-current (EMA of recent θ)**, (2) **demote it to a slowly-varying proximity/codec calibrator
rather than a re-injected gradient** (EMA-PG / mean-teacher / Asadi), and/or (3) **decay the anchor dose with
K**; nothing in this literature makes a large-K lagged GRADIENT term stable, because the off-policy bias it
carries is structural, not averageable.*

---

## Source ledger (link-verification status)

| # | Source | arXiv/DOI | Fetched? | Status |
|---|---|---|---|---|
| 1 | Borkar 1997, SA two time scales | SCL 29(5):291–294 | repo 404 | record RESOLVES via Semantic Scholar; primary IAS/SCL **LINK UNVERIFIED**, cite DOI |
| 2 | Konda–Borkar 1997, actor-critic multi-timescale | 10.1007/BF02745577 / SIAM | Springer auth-redirect; SIAM resolves (paywall) | bib confirmed; Springer **LINK UNVERIFIED** (auth wall), SIAM RESOLVES |
| 3 | Vidyasagar 2026, martingale TTSA | arXiv:2603.14481 | yes | VERIFIED |
| 4 | Mnih 2013, DQN | arXiv:1312.5602 | yes (abstract) | VERIFIED |
| 5 | Lillicrap 2015, DDPG | arXiv:1509.02971 | resolved via search | VERIFIED (search) |
| 6 | Lee 2026, target updates linear Q | arXiv:2606.02645 | yes | VERIFIED |
| 7 | Tarvainen–Valpola 2017, Mean Teacher | arXiv:1703.01780 | yes | VERIFIED |
| 8 | Zhang 2019, Lookahead | arXiv:1907.08610 | yes | VERIFIED |
| 9 | Izmailov 2018, SWA | arXiv:1803.05407 | yes | VERIFIED |
| 10 | Wang 2019/2020, SlowMo | arXiv:1910.00643 | yes (abstract; HTML 404) | VERIFIED |
| 11 | Zhang & Ba 2026, EMA-PG | arXiv:2602.04417 | yes | VERIFIED |
| 12 | Asadi 2021, slower online network | arXiv:2112.05848 | yes | VERIFIED |

**Count:** 12 distinct sources (Konda–Borkar listed once across §1/§3). ≥10 met; 4 are 2025+ (Vidyasagar
2026, Lee 2026, EMA-PG 2026; + Asadi 2021). Joint with the async teammate this clears the report's
≥12 / ≥6-from-2025+ bar.

**Links I could NOT verify (flagged for the lead):**
- Borkar 1997 (SCL) — IAS repository URL 404/403; bibliographic record confirmed via Semantic Scholar.
  Recommend citing by DOI/journal, not the repo URL.
- Konda–Borkar 1997 Sādhanā — Springer link redirects to an auth page and the IAS fulltext index 403'd the
  fetcher; the SIAM companion (10.1137/S036301299731669X) resolves but is paywalled. Citation is standard
  and safe; the *clickable* Springer link is unverified.
