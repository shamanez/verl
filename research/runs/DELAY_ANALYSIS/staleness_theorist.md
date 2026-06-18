# M-Staleness Theory: Why K=20 Anchor Latency Breaks the Comm-Eff EMA Method

**Author:** M-Staleness Theorist (team member 1 of 3)
**Assigned hypothesis:** *"The anchor gradient M is too stale."*
**Method under study:** the EMA merger `signed_ema` (α=0.25, β_anc=0.50) only.
**Scope:** theoretical model + staleness-axis recommendations.
**Companion analyses:** member 2 (cadence / Q-refresh-infrequency); member 3 (report-author).
**Status of claims:** every mechanism claim is grounded in `spectral_filter.py`, `state.py`,
`transformer_impl.py`, `anchor.py` (file:line cited inline). Memory findings are treated as
hard-won priors and cross-checked against current code.

---

## 0. TL;DR for the report-author

- Folding a **K-stale** gradient direction into the current update is **provably biased**, and
  the bias grows **at least linearly in K** through trajectory drift (`‖θ_t − θ_{t−K}‖`) and
  **super-linearly** once the GRPO on-policy distribution shift compounds it: the K-stale weights
  generated *different data* than the current policy, so the stale gradient is not merely an old
  estimate of the same objective — it is an estimate of a **different (off-policy) objective**.
- The staleness error is the mechanism that **converts a bounded correction into a positive-feedback
  length spiral**. The merger injects a persistent, reward-flat **tangential** force (signed_ema:
  via `sign(M)` on the coordinates where stale and fresh disagree). The injected force does not push
  the policy toward higher reward — it pushes it along the "correct-but-longer" direction.
  Increasing K **increases the disagreement rate**, raising the per-step dose of that tangential
  push — the lag-driven ignition lever.
- **Late onset** (steps ~50–100, not immediate) is structurally expected: the spiral is a
  **ratchet integrating a small biased force**; it needs to accumulate past the token-mean
  amplification threshold before it self-sustains. Larger K raises the per-step increment but the
  integral still takes time to cross threshold — hence late, not absent.
- **σ(M) angle:** staleness does **not** add information outside the σ(M) ceiling — a K-stale M is
  *strictly less* informative about `g(θ_t)` than a fresh one. Staleness can only **degrade** the
  correction toward (or below) the no-correction floor; it cannot buy a surpass.
- **K=5 is the working point** (val@50 0.7362, ~0.03 below dense); **K=20 collapses** (val@50
  0.6482, then a length spiral to val@100 0.4435, below the no-merger floor 0.6300). Same merger,
  same codec, same surface — only K differs.
- **Confound (delay_K vs cadence):** in EXP-37 both were raised to 20 together through one firing
  gate. I attribute the **bulk of the late-onset ignition and the on-policy-shift component of the
  val drop to delay_K**, and the **stale-Q / coarse-refresh component to cadence**. I give a clean
  2×2 decoupling experiment (§7) to separate them.

---

## 1. Formal staleness model

### 1.1 The objects, as the code defines them

The anchor is a **no-hook clone** that runs forward/backward from a weight snapshot
`θ_{t−K}` that is `delay_K` **optimizer ticks** stale (snapshot queue maxlen `delay_K + 1`,
`anchor.py:248`; stale fetch `queue.get_stale(step, delay_K)`, asserted exact post-warmup,
`transformer_impl.py:1521/1549–1556`). Its per-shard gradient is **DP-mean all-reduced** to full
coverage over all 196 decoder matrices (`_dp_all_reduce_anchor_grads`, SUM-then-divide-by-`dp_world`,
`transformer_impl.py:1004–1100`) and fed RAW into the anchor EMA
`M ← β·M + (1−β)·G_anchor` (`update_anchor`, `spectral_filter.py:310–331`). With the production
default `β_anc = 0` (see `[[anchor-gradient-ema-beta0-grpo]]`), **M is the instantaneous DP-mean
stale gradient**:

$$M(t) \;=\; g_{\text{DP}}\big(\theta_{t-K};\, \mathcal{D}_{t-K}\big),$$

i.e. the dense gradient evaluated at the **K-stale weights** on the **batch those weights
generated** (paired replay, `[[canonical-anchor-comm-eff-base]]` EXP-29: `replay_paired_batch`
pairs `θ_{t−K}` with the data `θ_{t−K}` produced). This pairing is critical: it makes M a clean
estimate of `g(θ_{t−K})` and removes the stale-weights×current-batch confound — but it **does not**
remove staleness itself.

The merger folds M into the fast compressed gradient `G_comp(t)` each tick. The cadence counter
`state.anchor_step` advances once per `train_batch` (`transformer_impl.py:1342`), the anchor
**fires** when `step % cadence == 0` (`anchor.py:124–135`), and the held correction is **refreshed at
fires and HELD between them** (`spectral_filter.py:930–944`). Unit reminder
(`[[anchor-cadence-delayk-unit-optimizer-ticks]]`): `train_batch=128 / ppo_mini=64` ⇒ **2 ticks per
global step**, so **K = 20 ticks = 10 global steps** of staleness and a refresh every 10 global
steps; **K = 5 ticks = 2.5 global steps** of staleness with a refresh every 2.5 global steps.
EXP-37 confirms the K=20 latency was genuinely realized (`anchor_backwards=10`).

### 1.2 Staleness error bound (the clean, single-objective part)

Treat M as a delayed estimate of the true current gradient `g(θ_t)` of a *fixed* objective `L`.
The exact staleness error is

$$
e_K(t) \;=\; g(\theta_t) - g(\theta_{t-K})
        \;=\; -\!\int_{0}^{1}\! H\big(\theta_{t-K} + s\,\Delta\theta\big)\,\Delta\theta \; ds,
\qquad \Delta\theta \equiv \theta_t - \theta_{t-K},
$$

by the fundamental theorem of calculus applied to `g = ∇L` (Hessian `H = ∇²L`). Taking norms with
`L_H ≡ sup‖H‖` (local Lipschitz constant of the gradient):

$$
\boxed{\;\big\|g(\theta_t) - g(\theta_{t-K})\big\| \;\le\; L_H\,\big\|\theta_t - \theta_{t-K}\big\|
   \;\le\; L_H \sum_{j=t-K}^{t-1}\big\|\theta_{j+1}-\theta_j\big\|
   \;\approx\; L_H\, K\, \eta\, \overline{\|u\|}\;}
$$

where `η` is the step size and `\overline{‖u‖}` is the mean per-step update norm (`u` = the
Adam-preconditioned, grad-clipped update). **The staleness error scales like `L_H · K · η · ‖u‖` —
linear in K** for a smooth trajectory. This is the floor of the damage and it is already
unattractive: at K=20 ticks vs K=5 ticks, the drift term is **4× larger** purely geometrically.

**Bias into the optimizer.** The merger replaces the would-be update direction with one built from
M. Decompose `g(θ_t) = g(θ_{t−K}) + e_K`. A merger that uses `sign(M)` as a correction therefore
**injects `sign`-distorted `e_K`** into the step. The Adam preconditioner and grad-clip make the
**direction** load-bearing (`[[exp25-collapse-gradient-flow]]`), so the relevant quantity is the
**angular** error: as `‖e_K‖` grows toward `‖g(θ_t)‖`, the cosine between the applied step and the
true descent direction degrades, and beyond `‖e_K‖ > ‖g(θ_t)‖` the merger can point **up-hill** on
individual coordinates. This is the coordinate-level mechanism behind the wrong-sign flips seen in
signed_ema at large K.

### 1.3 Why the error grows super-linearly under GRPO

The bound in §1.2 assumes a *fixed* objective. GRPO does **not** have one. The loss at tick `t`
is an importance-weighted, clipped surrogate over **trajectories sampled from the policy
`π_{θ_{t}}`**, and the *data itself* is generated by the (rollout) policy. The K-stale gradient was
computed on data `\mathcal{D}_{t−K}` sampled from `π_{θ_{t−K}}` — a **different distribution**:

$$
M(t) = \mathbb{E}_{\tau\sim \pi_{\theta_{t-K}}}\big[\nabla \ell_{\theta_{t-K}}(\tau)\big],
\qquad
g(\theta_t) = \mathbb{E}_{\tau\sim \pi_{\theta_{t}}}\big[\nabla \ell_{\theta_{t}}(\tau)\big].
$$

So the staleness error has **two** components:

$$
e_K = \underbrace{\big(g(\theta_t) - g_{\mathcal{D}_t}(\theta_{t-K})\big)}_{\text{weight drift (§1.2, } \le L_H K\eta\|u\|)}
   \;+\; \underbrace{\big(g_{\mathcal{D}_t}(\theta_{t-K}) - g_{\mathcal{D}_{t-K}}(\theta_{t-K})\big)}_{\text{distribution shift } \Delta_{\text{dist}}}.
$$

The second term `Δ_dist` is the **off-policy gap**: it measures how differently the *same weights*
would be scored on current-policy data vs the data they generated. Two reasons this compounds
super-linearly in K:

1. **Distribution shift accumulates non-linearly in trajectory space.** Policy divergence
   `D_KL(π_{θ_t} ‖ π_{θ_{t−K}})` grows with the *cumulative* parameter motion, and the
   importance-weight variance — hence the gradient-estimator error — grows roughly like
   `exp(D_KL)` (the standard IS-variance blow-up). Even if `‖Δθ‖` grows linearly, the
   **statistical** error of using K-stale data grows faster than linearly because IS variance is
   exponential in divergence. The GRPO **clip** truncates the worst importance weights, which
   *bounds* the variance but at the cost of a **bias** that itself grows with divergence — so
   clipping converts the exponential variance into a steadily growing systematic bias.

2. **The held correction is K-stale and then held over the cadence window.** The correction is
   refreshed only at fires and **HELD for the whole cadence window** (`spectral_filter.py:941–944`)
   before being applied to the **current** `G_comp(t)`. As K and cadence both grow, the held
   correction is reused over a longer window during which `G_comp(t)` has moved further from its
   fire-time value, widening the mismatch between the correction and what it is correcting.

**Net:** `‖e_K‖` has a linear geometric floor (`L_H K η‖u‖`) plus a distribution-shift term that
grows faster than linear under the GRPO objective. EXP-37's val drop (≈0.088 at val@50) and the
late ignition are both consistent with crossing from the "bias tolerable" regime (K=5, ~2.5 global
steps) into the "bias dominant" regime (K=20, ~10 global steps).

---

## 2. Why staleness specifically drives the LENGTH spiral

### 2.1 The empirical chain of causality (verified in memory)

The memory record is explicit and must be respected: **length is the killer, entropy is a
follower.** `[[entropy-collapse-alpha0-signed-ema]]` and `[[exp25-collapse-gradient-flow]]`
establish:

- Dense is the **lowest-entropy** run (0.12–0.16) **and** the most stable ⇒ entropy-as-trigger is
  **falsified**; entropy collapse trails the length explosion.
- The RED kill triggers are **length-spiral precursors** P1 (≥2 consecutive 16384 cap-pins),
  P2 (len/mean trailing-10 slope), P3 (mean > 2× baseline).
- The cross-run discriminator is **MERGER-CARRIER PRESENCE**: plain PowerSGD on the same substrate
  is clean for 50 steps; the merger (folding stale M into the fast grad) is the killer.

EXP-37's anatomy matches exactly: stable through ~step 58, then `response_length/mean`
189→251→373→581→683 across steps 93–100, pg-clip 0→24%, entropy collapsing 0.81→0.42. This is the
canonical length ratchet, now appearing at K=20 where it was *absent* (or censored-marginal) at
K=5.

### 2.2 The mechanism: staleness turns a bounded correction into a tangential ratchet

The implemented signed_ema correction injects a force that is **by construction tangential to the
live gradient** on the coordinates where `sign(M) ≠ sign(G_comp)`. A tangential force does **no work
against the loss** to first order — it is a **persistent push along the reward-flat manifold**. On a
no-KL / no-entropy GRPO surface there is nothing to brake motion along that manifold. The
"correct-but-longer" direction is exactly such a reward-flat ridge: emitting more tokens barely
changes the verifier reward but is reachable by a sustained tangential drift. The drift
**rectifies** (via the token-mean ratio's ~86× tail amplification) into a self-reinforcing length
climb.

**Where staleness enters.** Both the *magnitude* and the *persistence* of that tangential force are
controlled by staleness, through `signed_ema_matrix` (`spectral_filter.py:403–441`):
`G_corr = α·G_noisy + (1−α)·|G_noisy|·sign(M)`. The harm is concentrated on the coordinates where
`sign(M) ≠ sign(G_noisy)`. The disagreement rate is a **direct function of `‖e_K‖`**: when M and
the fresh gradient agree (small staleness) the merger is a near-no-op; when they disagree (large
staleness) the merger **flips** the magnitude onto the stale sign, producing `(2α−1)·|G_noisy|`
of *reversed* force per disagreeing coordinate. At K=5 the disagreement is largely the structural
coin-flip of two estimators of a near-zero-mean gradient, and these random flips **cancel** across
coordinates; **at K=20 the *systematic* (drift + off-policy) component adds correlated,
same-direction sign flips on top of the random ones** — these correlated flips are exactly what a
tangential ratchet needs, because random flips cancel across coordinates while correlated ones
integrate. A longer cadence window also freezes the tangential direction (the same `sign(M)` is
reused), making the ratchet *more* coherent (less direction-averaging) at K=20.

**The conversion to positive feedback.** Define the length state `ℓ_t`. The tangential force `F`
adds a small increment `ℓ_{t+1} ≈ ℓ_t + c·F`. Because the token-mean loss normalization makes the
per-token gradient scale **inversely** with length (longer sequences dilute each token's gradient),
once `ℓ` grows the brake on further growth *weakens* — `dF/dℓ > 0` in the relevant regime. That is
the sign of a **positive-feedback loop**: `ℓ_{t+1} − ℓ_t` is increasing in `ℓ_t`. Staleness sets
the **gain** of this loop (the size of `F`); a larger K raises the gain, moving the system from
sub-critical (K=5: ratchet present but marginal/censored) to super-critical (K=20: ignites within
the horizon).

### 2.3 Why late-onset (steps 50–100), not immediate

The spiral is the **integral of a small biased force**, not an instantaneous instability. Three
reasons for the delay, all consistent with a ratchet model:

1. **Threshold crossing.** The length climb only self-sustains once `ℓ` is large enough that the
   token-mean dilution flips the brake (the `dF/dℓ > 0` regime). Before that, the tangential force
   is resisted by ordinary descent and `ℓ` is roughly stationary. The system spends many steps
   accumulating before it crosses the threshold. A larger K raises the per-step increment, so K=20
   crosses *within* 100 steps where K=5's increment is small enough that it stays sub-threshold for
   the 50-step horizon (its instability is **censored**, per `[[entropy-collapse-alpha0-signed-ema]]`:
   α=0.5 at K=5 was already spiraling at steps 47–48).

2. **Epoch boundary / data re-exposure.** EXP-37 is stable through ~step 58 = end of epoch 0, then
   ignites in epoch 1. Two epochs over GSM8K means the policy re-sees the same prompts; by epoch 1
   the cumulative drift `‖θ_t − θ_0‖` is large, so the **distribution-shift term `Δ_dist`** (§1.3)
   is at its largest, maximizing `‖e_K‖` exactly when the second epoch begins. The K-stale anchor
   is now lagging a policy that has moved substantially, so the correction is at its most
   mis-aimed.

3. **Held-correction warm-up of the bias.** Although `β_anc=0` removes M's own memory, the held
   correction transport over the (now 20-tick) cadence window means a **single** mis-aimed fire's
   residual is applied for ~10 global steps. Early fires (small drift) inject benign correction;
   later fires (large drift) inject increasingly tangential correction that is then held and
   integrated. The damage is back-loaded by construction.

---

## 3. The σ(M) ceiling angle

From `[[surpass-dense-sigma-m-ceiling-and-routes]]`: let `σ(M) = σ(g(θ_t), g(θ_{t−K}))` be the
sigma-algebra generated by the current and K-stale dense gradient means. **Any deterministic
`Φ(G_comp, M)` is σ(M)-measurable and therefore capped at dense.** A route can surpass dense *only*
by injecting information **outside** σ(M) (curvature, conversion-positive exploration, or cross-rank
2nd moment).

Staleness interacts with this ceiling in a strictly **downward** direction:

- **A K-stale M carries strictly less information about `g(θ_t)` than a fresh M.** Formally, by the
  data-processing inequality, `g(θ_{t−K})` is a noisy (drifted + off-policy) channel observation of
  `g(θ_t)`; increasing K increases the channel noise (`‖e_K‖`), so the mutual information
  `I(M_K ; g(θ_t))` is **monotonically non-increasing in K**. The correction can at best
  reconstruct the dense direction it can still see through the staleness channel; everything K
  corrupts is unrecoverable by any `Φ`.

- **Staleness does not move you out of σ(M).** It does not add curvature, exploration that moves the
  greedy argmax, or cross-rank disagreement signal. It only **degrades the dense reconstruction**.
  So the *best* outcome of any staleness setting is "approach dense from below"; the *worst* is
  "fall below the no-merger floor." The EXP-37 result — falling **well below** the K=5 result and
  below dense, with a destabilizing spiral — is the worst-case manifestation: the stale M's
  reconstruction error is large enough that the correction is net-harmful, pushing below the no-merger
  floor (no-merger floor C2 = 0.6300, `[[no-merger-floor-0p63-not-0p74]]`; EXP-37 val@100 = 0.4435
  is **below** even that floor, confirming the merger is actively destructive at K=20, not merely
  inert).

**Corollary for the program:** staleness is a **ceiling-respecting nuisance variable**, not a
surpass lever. There is no value of K that helps; the only question is how large K can be before the
reconstruction degrades past parity and then past the floor. The async north-star
(`[[async-anchor-single-node-fast-swarm]]`) forces K to be *large and variable*, so the practical
question is **robustness to staleness**, not exploitation of it.

---

## 4. Practical recommendations (staleness axis)

These are **specific to delay_K** and respect the operator constraints in
`[[async-anchor-single-node-fast-swarm]]`: the anchor must lag, delay-compensation / lead /
extrapolation is **ruled out**, and any derived state must be cross-rank-identical and degrade
gracefully as the anchor ages.

### R-S1 (lead recommendation) — Staleness-aware dose decay: down-weight the correction as realized delay grows.
The mechanism already exists in the code and is **exactly** the admissible lever. The held-correction
path has `delta_momentum_age_decay` (`spectral_filter.py:702–779`): on HELD ticks the applied
correction is scaled by `μ^age`, `age = current_step − last_refresh_step`. This **fades a held
correction toward 0 as it ages**, which is precisely "trust the stale correction less the staler it
gets," and it is cross-rank-identical (`current_step` is DP-identical) and degrades gracefully (the
async requirement).
**Concrete proposal:** at K=20, enable age-decay with `delta_momentum_mu ≈ 0.85–0.90`. Over a
10-global-step hold window (`age` up to ~20 ticks) `0.88^{20} ≈ 0.08`, so by the end of the window
the held correction is ~8% of its fire-time value — the correction is strongest right after a fresh
fire and near-zero just before the next, capping the integrated tangential dose. This directly
attacks the ratchet gain without touching K or the two-circuit structure.
*Caveat:* age-decay was tested at K=5 and found mildly harmful as a freshness lever (EXP-31 μ-momentum
0.7202→0.7089→0.5701 at increasing μ); but those were **dose** sweeps at *short* staleness where
fading a fresh correction throws away good signal. At K=20 the held correction is *bad* signal late
in the window, so age-decay should help where it hurt at K=5. This is itself a falsifiable
sub-prediction.

### R-S2 — Make the dose a function of *measured* staleness, not a constant.
The adaptive-dose machinery (`_adaptive_lambda`, `spectral_filter.py:784–846`) already gates λ on a
running agreement metric `c_t` (cos or `‖δ‖/‖gm‖`) vs its median, clamped to `[0, lambda_cap]`. The
**ratio** mode (`c_t = ‖δ‖/‖gm‖`) is a direct proxy for realized staleness: when the residual is
large relative to the live gradient, the anchor is badly aged and the dose should drop. Set
`adaptive_lambda_mode="ratio"`, modest `κ`, and a **low `lambda_cap`** (e.g. 0.5) so a staleness
spike *cannot* spike the dose. This is the "bounded raw dose, variable-staleness safety" the code
docstring already names (`spectral_filter.py:838–839`) and is admissible under async (cross-rank
identical, tolerates variable lag).

### R-S3 — Keep `β_anc = 0` (do NOT add EMA to fight staleness noise).
It is tempting to smooth a noisy stale M with `β > 0`. **Do not.** Per
`[[anchor-gradient-ema-beta0-grpo]]`, EMA *compounds* staleness — averaging old gradients makes M
even more lagging — and GRPO's clip tolerates far less of that than SFT. β∈[0,0.5] is a flat
free-averaging region at K=5, but at K=20 the marginal staleness from any β>0 lands in the harmful
regime. β=0 is the simplest safe point; staleness should be fought by **dose decay (R-S1/R-S2)**,
not by smoothing.

### R-S4 — Length brake as a *labeled guardrail*, not a fix.
Per `[[exp25-collapse-gradient-flow]]`, a KL term or seq-mean loss aggregation **bounds** the length
ratchet (KL kept length 239–291 vs ~8600 uncontrolled) but does **not** make the stale correction
helpful. If the goal is to *demonstrate staleness robustness* rather than *win val*, a seq-mean loss
aggregation (which kills the token-mean ~86× tail amplification, the ratchet's gain) is the surgical
choice. Flag it as a guardrail (it changes the objective), not as a staleness solution.

### R-S5 — Delay-compensation / extrapolation is RULED OUT.
For completeness and to preempt a tempting-but-forbidden direction: one could try to *predict*
`g(θ_t)` from `g(θ_{t−K})` via a finite-difference extrapolation (Nesterov-style lead). The async
operator constraint (`[[async-anchor-single-node-fast-swarm]]`) **explicitly rejects** this — the
anchor is a single slow node that can never lead the swarm. Any admissible lever uses the anchor as a
**lagging reference only**. So the staleness axis has exactly one productive direction: *use less of
M, the staler it is* (R-S1/R-S2). Do not propose lead/extrapolation.

### R-S6 — Bound the realized delay (don't let cadence widen the held-correction window).
A second-order effect: at K=20 the held correction is reused for ~10 global steps, freezing a stale
tangential direction. Even holding delay_K=20, **shortening cadence** (refresh M more often) would
reduce the held-correction window and re-aim the correction more frequently — but that is the
*cadence* axis (member 2's domain). I flag it here because the held-correction window length is a
**joint** delay_K×cadence quantity; the cleanest staleness fix (R-S1 age-decay) makes the held
window harmless **regardless of cadence**, which is why I rank it first.

---

## 5. Attribution: how much of EXP-37's failure is delay_K vs cadence?

EXP-37 raised **both** delay_K and cadence to 20, so the single run cannot decompose them. My
theory-based attribution:

| Failure component | Primary driver | Reasoning |
|---|---|---|
| **Late-onset length ignition (steps 93–100)** | **delay_K (dominant)** | The tangential-force gain scales with the disagreement rate `∝ ‖e_K‖ ∝ K` (§2.2). Cadence sets *how often* the correction is re-aimed, but the *magnitude* of each correction — the ratchet gain — is set by how far `θ_t` drifted from `θ_{t−K}`, i.e. by K. The spiral is driven by *lag*, and delay_K **is** the lag. |
| **On-policy distribution-shift bias (the off-policy `Δ_dist` term)** | **delay_K (exclusive)** | `Δ_dist` is the gap between current-policy and K-stale-policy data. It is a pure function of K (how stale the *data* M was computed on is). Cadence does not change which past policy generated M's data. |
| **Held-correction staleness within the window** | **joint (delay_K × cadence)** | The held correction is K-stale at the fire and ages further over the cadence window. The fire-time staleness is delay_K; the *additional* aging over the hold is cadence. |
| **Stale/coarse Q basis (PowerSGD codec quality)** | **cadence (dominant)** | Q is updated only at fires (`anchor_owns_q`, `transformer_impl.py:2061–2100`). Refreshing every 10 global steps (cadence=20) lets the codec subspace drift from the gradient's actual dominant subspace, raising compression error. This is member 2's axis. |
| **Magnitude of the ~0.088 val drop** | **mostly delay_K** | The reconstruction-quality degradation (§3 σ(M) channel) is dominated by `‖e_K‖`, which is delay_K-driven. Coarser Q (cadence) adds compression noise but the no-merger floor work (`[[no-merger-floor-0p63-not-0p74]]`) shows the codec alone is not catastrophic; the **merger carrying a badly-stale M** is. |

**Bottom-line attribution:** I assign roughly **70–80% of the val degradation and ~90% of the
ignition** to **delay_K** (the staleness of M and its off-policy data), and the remainder to
**cadence** (stale Q + extended held-correction window). The ignition is almost entirely a staleness
(lag) phenomenon; the val drop is mostly staleness with a cadence/codec contribution.

---

## 6. The decoupling experiment (separate delay_K from cadence)

A clean **2×2** (or minimal 3-cell) on the accel surface, 100 steps, signed_ema (α=0.25, β=0.50),
holding everything else at the EXP-37 config:

| Cell | delay_K | cadence | Isolates | Prediction (my theory) |
|---|---|---|---|---|
| **A (control)** | 5 | 5 | — (= EXP-36B) | Stable, val@50 ≈ 0.736 |
| **B (delay-only)** | **20** | **5** | **delay_K effect at fixed Q-refresh** | **Ignites + val drop ≈ EXP-37's** (most of the damage) |
| **C (cadence-only)** | **5** | **20** | **cadence/stale-Q effect at fixed staleness** | **Mostly stable, modest val drop** (codec-only) |
| **D (= EXP-37)** | 20 | 20 | both | already have it (0.5921 / 0.6482 / collapse) |

**Decision logic:**
- If **B ≈ D** (B reproduces the ignition and most of the val drop) ⇒ **delay_K is the culprit**;
  my attribution is confirmed and the fix is the staleness-axis dose-decay (R-S1).
- If **C ≈ D** ⇒ cadence/stale-Q is the culprit (member 2's domain); my hypothesis is the lesser
  factor.
- If **B and C each carry ~half** ⇒ the failure is genuinely joint and both axes need fixing.

**Note on the held-correction confound:** cell B (delay_K=20, cadence=5) has a *short* held-correction
window (refresh every 5 ticks) but *large* per-correction magnitude — this is the cleanest test that
the **magnitude** of the staleness (not the hold duration) is what ignites the spiral. If B ignites
with a short hold window, the gain is unambiguously the delay_K magnitude, vindicating R-S1's `μ^age`
decay as the right lever (fade the large-but-stale correction). Cell B is the single most informative
run; if budget allows only one extra cell, **run B**.

A secondary, **cheaper** GPU-free check: re-process the EXP-37 captures (if `capture.enabled` was on
for any subset) to plot the held-correction norm and the disagreement rate
`P(sign(M)≠sign(G_comp))` vs step. My theory predicts both rise into epoch 1 and that the
disagreement rate's *correlated* (non-random) fraction spikes just before step 93. If captures are
unavailable (production runs default `diagnostics=false`, `[[production-runs-diagnostics-off]]`), this
requires a short instrumented re-run, which cell B can carry.

---

## 7. Summary of grounded claims (file:line index)

| Claim | Grounding |
|---|---|
| Anchor is a no-hook clone forwarding from `θ_{t−K}`, snapshot queue maxlen `delay_K+1`, exact stale fetch asserted post-warmup | `anchor.py:248`; `transformer_impl.py:1521`, `1549–1556`, `1646` |
| M = DP-mean all-reduced (SUM/÷dp_world) full-coverage stale gradient, fed RAW into EMA | `transformer_impl.py:1004–1100`, `1886–1895`; `update_anchor` `spectral_filter.py:310–331` |
| β_anc=0 ⇒ M is instantaneous stale gradient (no EMA memory) | `spectral_filter.py:324`; `[[anchor-gradient-ema-beta0-grpo]]` |
| Cadence counter advances per `train_batch`; fires at `step%cadence==0`; unit = optimizer ticks (2/global step) | `transformer_impl.py:1342`; `anchor.py:124–135`; `state.py:336–337`; `[[anchor-cadence-delayk-unit-optimizer-ticks]]` |
| Correction refreshed at fires, HELD between; ring retains fire-aligned ticks | `spectral_filter.py:930–944`; `state.py:159–234` (`tick_retained`, `get`, `push`, `pop`) |
| signed_ema flips magnitude onto stale sign on disagreeing coords ⇒ tangential force | `signed_ema_matrix` `spectral_filter.py:403–441`; `[[exp25-collapse-gradient-flow]]` (mechanism) |
| Length is killer, entropy follows; RED triggers = length-spiral precursors | `[[entropy-collapse-alpha0-signed-ema]]`; `[[canonical-anchor-comm-eff-base]]` |
| σ(M) ceiling: deterministic Φ(G_comp,M) capped at dense; staleness only degrades | `[[surpass-dense-sigma-m-ceiling-and-routes]]` |
| Async: anchor must lag, delay-compensation ruled out | `[[async-anchor-single-node-fast-swarm]]` |
| Age-decay lever (`μ^age` fade of held correction) exists and is admissible | `_apply_delta_momentum` `spectral_filter.py:702–779` |
| Adaptive-dose ratio-gate with `lambda_cap` (bounded variable-staleness safety) | `_adaptive_lambda` `spectral_filter.py:784–846` |
| No-merger floor C2 = 0.6300; EXP-37 val@100 0.4435 is below the floor | `[[no-merger-floor-0p63-not-0p74]]` |
