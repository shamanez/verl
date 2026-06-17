# The mathematics of communication-efficient pipeline-parallel GRPO

*Author: `theorist` (comm-eff-grpo team). Companion docs: `systems` (measured
numbers), `strategist` (surpass routes). Ground truth: `verl/workers/comm_eff/`,
`research/runs/SUMMARY.md`, `CODE_WALKTHROUGH.md`.*

This note derives, in order: (1) the GRPO objective and its policy gradient;
(2) a bias/variance model of the PowerSGD-compressed pipeline-boundary gradient;
(3) the **error-feedback** merger (`delayed_ef`, "B2", the SOTA) as an additive,
magnitude-preserving residual that de-biases the compressed gradient toward the
*stale dense* gradient; (4) the **`signed_ema`** merger as a multiplicative
sign-replacement, and a formal contrast of the two; and (5) the
**stale-reconstruction ceiling** — a proof-sketch that *de-biasing a stale
estimate of the dense gradient lands at dense, never past it*, which is the
bridge to the strategist's surpass program.

Notation is fixed once: $\theta$ are policy parameters; $g(\theta)$ is the true
(dense) GRPO mini-batch gradient at $\theta$; $C(\cdot)$ is the PowerSGD codec
applied at the pipeline-parallel (PP) stage boundary; $G_{\text{comp}}=C(g)$ is
the compressed gradient the fast circuit actually produces; $M$ is the anchor's
per-matrix gradient estimate; $K=\texttt{delay\_K}=5$ is the anchor staleness in
optimizer ticks. All matrix quantities are the $196$ logical 2D decoder matrices
(28 layers $\times$ 7 matrices).

---

## 1. GRPO: objective, advantage, policy gradient

### 1.1 Group-relative advantage

GRPO removes the value network. For a prompt $q$, sample a **group** of $n=8$
completions $\{o_1,\dots,o_n\}\sim \pi_{\theta_{\text{old}}}(\cdot\mid q)$ and
score each with the verifier reward $r_i = R(q,o_i)$ (for GSM8K, $r_i\in\{0,1\}$
plus format terms). The advantage is the **group-standardized** reward, assigned
identically to every token $t$ of completion $i$:

$$
A_{i,t} \;=\; \hat A_i \;=\; \frac{r_i - \operatorname{mean}_{j\le n}(r_j)}{\operatorname{std}_{j\le n}(r_j) + \varepsilon}.
$$

Two structural facts follow and matter for everything downstream:

1. **The group baseline is the per-group mean.** Subtracting
   $\bar r=\operatorname{mean}_j r_j$ makes $A_i$ **mean-zero within the group**:
   $\sum_{i=1}^n \hat A_i = 0$. The advantage is a *contrast between siblings*,
   not an absolute signal. The $1/\text{std}$ whitening makes the scale of the
   update invariant to how hard the prompt is.
2. **Reward is sequence-level, advantage is broadcast to tokens.** Every token
   in a completion shares one scalar $\hat A_i$. There is no per-token credit
   assignment; the only per-token quantity is the policy ratio below. This is
   why a *uniform* lengthening of correct-but-verbose completions is nearly
   reward-neutral — a fact that becomes load-bearing in §4.3 (the length hack).

### 1.2 Clipped surrogate objective

With per-token importance ratio
$\rho_{i,t}(\theta) = \dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}$,
vanilla GRPO (this project: **no KL, no entropy** bonus) maximizes the PPO-style
clipped surrogate

$$
\mathcal{J}_{\text{GRPO}}(\theta)
= \mathbb{E}_{q}\,\mathbb{E}_{\{o_i\}}
\left[\frac{1}{\sum_i |o_i|}\sum_{i=1}^{n}\sum_{t=1}^{|o_i|}
\min\!\Big(\rho_{i,t}\,\hat A_i,\;
\operatorname{clip}(\rho_{i,t},\,1-\epsilon,\,1+\epsilon)\,\hat A_i\Big)\right].
$$

The token-mean normalization $1/\sum_i|o_i|$ is the standard verl aggregation
(`agg_loss`); it is the same normalization the anchor and ring feeds honor (the
"scale contract" the merger relies on — see §3.4).

### 1.3 Policy gradient and the clip mask

Where the clip is inactive (the unclipped region), the gradient of one token's
contribution is the familiar score-function form

$$
\nabla_\theta \big[\rho_{i,t}\hat A_i\big]
= \hat A_i\,\rho_{i,t}\,\nabla_\theta \log \pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

Let $\mathcal U=\{(i,t):\text{clip inactive}\}$. Then the GRPO gradient is

$$
g(\theta) \;=\; \nabla_\theta \mathcal{J}_{\text{GRPO}}
= \frac{1}{\sum_i|o_i|}\sum_{(i,t)\in\mathcal U}
\hat A_i\,\rho_{i,t}\,\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

The clip is a hard gate: a token whose ratio has already moved past
$1\pm\epsilon$ in the rewarded direction contributes **zero** gradient
($\nabla\min(\cdot)=0$ there). This gate is what *caps* any single noisy update
direction — it bounds how far one mini-batch (or one compression artifact) can
push a token's logit. We return to this in §4.4: the clip is the reason an
*additive* residual correction is safe but a *sign-replacing* one is not.

This $g(\theta)$ — the dense GRPO mini-batch gradient — is the reference signal.
Everything the compression machinery does is an attempt to transmit it across the
PP boundary in $\sim5\%$ of the bytes without losing the part of it that matters.

---

## 2. The compressed boundary gradient: bias and variance

### 2.1 The codec

At a PP stage boundary the gradient flowing backward is a matrix
$g\in\mathbb{R}^{m\times n}$. PowerSGD transmits a rank-$r$ ($r=77$) sketch: it
maintains an orthonormal basis $Q\in\mathbb{R}^{n\times r}$ (the **activation
basis**, $Q^\top Q = I_r$) and sends $P = QQ^\top$, the rank-$r$ orthogonal
projector. The transmitted (compressed) gradient is

$$
G_{\text{comp}} \;=\; C(g) \;=\; P\,g \;=\; QQ^\top g, \qquad P=P^\top=P^2.
$$

The decomposition into kept / dropped parts is exact and orthogonal:

$$
g = \underbrace{Pg}_{G_{\text{comp}}} + \underbrace{(I-P)\,g}_{\text{dropped residual } \rho_{\text{cod}}},
\qquad \langle Pg,\,(I-P)g\rangle = 0.
$$

Byte cost is the sketch, not $g$: bytes ratio $\approx 0.0505$ ($\sim5\%$ of
dense), confirmed across runs. The DP axis is **not** compressed — only the PP
boundary is — and $Q$ is a single shared codebook synchronized across DP ranks
(it must be, or ranks sketch different subspaces and diverge).

### 2.2 Bias and variance vs dense

Treat $C$ as an estimator of $g$. Because $P$ is a deterministic projector (for a
**fixed** $Q$ within a step), the codec is **not** a zero-mean stochastic
perturbation. Its error is a deterministic, structured bias:

$$
\underbrace{\mathbb{E}[G_{\text{comp}}] - g}_{\text{bias}} = (P-I)\,g = -(I-P)g = -\rho_{\text{cod}},
\qquad
\underbrace{\operatorname{Var}(G_{\text{comp}})}_{\text{codec, fixed }Q} = 0.
$$

So relative to the *dense gradient on the same batch*, the codec contributes
**pure bias, near-zero added variance** — the opposite of a dropout-style mask.
The only randomness in $Q$ is its slow block-power-iteration drift, which the
anchor controls (it owns $Q$); empirically $Q$ **converges** to a stable
dominant subspace (reconstruction error flat $\approx 0.024$), so *the same
off-subspace direction is dropped every step*. That is what makes the bias
**persistent** rather than averaging out.

### 2.3 Why the bias is small but the floor is far below dense

The boundary gradient is **low-rank-dominated**: its spectral energy
concentrates in a handful of directions, so a rank-77 projector captures the
overwhelming majority of $\|g\|^2$. Two measured fingerprints (from `systems`,
to be confirmed):

- **Dropped energy is tiny**: $\|(I-P)g\|^2/\|g\|^2 \approx 0.058\%$.
- **High signal-to-bias ratio**: $\|Pg\| / \|(I-P)g\| \approx 42{:}1$ (the
  "$\sim42:1$ SNR" figure; here it is a signal-to-**bias** ratio, since the
  variance term is $\approx0$).

This is why $P g$ is *small-bias / low-variance* and why naive intuition says
"compression is benign." Yet the **realistic no-merger floor is val $\approx
0.6300$**, far below dense's $0.75$–$0.78$. The reconciliation — and it is the
central quantitative fact this whole document must explain — is:

> A $0.06\%$ **per-step** energy drop that is **biased** (same direction every
> step, never averaged out) is **not** a $0.06\%$ effect on the optimization. A
> persistent bias is integrated by the optimizer over all 50 steps. Adam's
> momentum ($\beta_1$) and the GRPO clip both *accumulate* directional
> consistency; a small but **coherent** off-subspace force that the dense
> trajectory would have followed is systematically never taken, and the policy
> ends up in a measurably worse basin (the $0.63$ floor).

Formally: the merger-free update consumes $G_{\text{comp},t}=g_t+b_t$ with the
coherent bias $b_t=-(I-P_t)g_t$ (coherent because $P_t\to P$, so $b_t$ keeps a
stable direction). Compare two trajectories driven by $g_t$ vs $g_t+b_t$. For
**zero-mean** noise $n_t$ the per-step errors partially cancel and the trajectory
gap grows sublinearly ($O(\sqrt T)$ in the random-walk sense). For a **coherent
bias** there is no cancellation: the gap accumulates *every* step in the same
off-subspace direction, so it grows $\Theta(T)$ in the small-step regime. This
holds under Adam too — Adam's per-coordinate rescaling and the first-moment EMA
($\beta_1$) reweight the step but do **not** cancel a direction that is present
*every* step; a persistent off-subspace component survives the normalization and
is integrated into a $\Theta(T)$ trajectory displacement (formally: with bias the
first-moment EMA converges to a biased $\bar m\neq \bar g$, so the normalized
step is persistently misdirected). The merger's entire job is to cancel
$\sum_t b_t$ before Adam integrates it. This also explains why the genuinely
zero-mean alternative — the PRF activation mask, $\mathbb E[\tilde h]=h$ — does
**not** help: it trades the codec's coherent bias for the $O(\sqrt T)$ regime,
but its *gradient* bias is the Jensen/curvature term $\propto
\operatorname{Var}\propto p/(1-p)$ and the variance it injects is so large
($\sim9$–$19\times$ at high $p$) that it stalls. Neither codec is simultaneously
zero-mean, low-variance, and tunable — which is the gap the anchor exists to
close.

---

## 3. Error-feedback merger `delayed_ef` (B2, the SOTA)

### 3.1 The two circuits and what the anchor produces

There are two circuits (`CODE_WALKTHROUGH.md`):

- **Fast swarm**: the normal compressed actor backward; produces $G_{\text{comp}}=C(g)$;
  is a **read-only consumer** of $Q$.
- **Slow anchor**: a single node that, at cadence $K$, replays a **paired**
  $(\text{batch},\theta_{t-K})$ snapshot through an *uncompressed, no-optimizer*
  clone, reads the **raw** full gradient, DP-mean-reduces it, and updates the
  per-matrix EMA

  $$
  M \;\leftarrow\; \beta_{\text{anc}}\, M + (1-\beta_{\text{anc}})\,G_{\text{anc}}.
  $$

  At the project default $\beta_{\text{anc}}=0$ this collapses to
  $M = G_{\text{anc}}^{\text{rep}}$ — *the latest fire's raw DP-mean dense
  gradient on the replayed (batch, $\theta$) pair*. The anchor is the **only**
  updater of $Q$. (Verified: `spectral_filter.py:302-323`; the $\beta$-sweep
  EXP-33 found $\beta\in[0,0.5]$ a flat free-averaging tie, $\beta=1$ a cold-$M$
  collapse — consistent with $M$ being a *stale signal* that EMA-smoothing only
  makes staler.)

The async-realism constraint: the anchor **always lags by $K$, never leads**
(one slow node serving a fast swarm). So $M$ is an estimate of
$g_{\text{dense}}(\theta_{t-K})$ on the $(t-K)$ batch — **a stale dense
gradient**. Hold this; it is the whole story of §5.

### 3.2 The paired-replay residual

`delayed_ef` (`spectral_filter.py:838`) forms, per matrix,

$$
\boxed{\;\delta(t) = M_{\text{rep}} - G_{\text{comp}}^{\text{ring}}(t-K),\qquad
G_{\text{corr}}(t) = G_{\text{comp}}(t) + \lambda\,\delta(t)\;}
\qquad (\lambda=1).
$$

The crucial design point is **what $\delta$ is a residual of**. The ring
($\texttt{FastGradRing}$) stores the *raw compressed* gradient
$G_{\text{comp}}^{\text{ring}}(t-K)=C\big(g(\theta_{t-K})\big)$ produced on the
**same** $(\text{batch},\theta_{t-K})$ pair the anchor just replayed densely. So

$$
\delta = \underbrace{M_{\text{rep}}}_{\;\approx\,g_{\text{dense}}(\theta_{t-K})}
\;-\; \underbrace{C\big(g(\theta_{t-K})\big)}_{\text{compressed, same pair}}
\;=\; g_{\text{dense}}(\theta_{t-K}) - P\,g(\theta_{t-K})
\;=\; (I-P)\,g(\theta_{t-K}) \;=\; \rho_{\text{cod}}(t-K).
$$

**$\delta$ is exactly the dropped codec residual** of the matching pair — the
$(I-P)g$ that §2.1 split off. Because it is computed on the *identical*
$(\text{batch},\theta)$, it is the codec's *weight-gradient error*, not a batch
or staleness artifact. This is the property that distinguishes B2 from
`ef_powersgd` (§4.5), which uses $M-\operatorname{proj}_G(M)$ on the *current*
live gradient and is therefore exactly orthogonal-to-$G$ and not a true
residual.

### 3.3 Why it reaches parity (additive, magnitude-preserving de-biasing)

Substitute $\delta=(I-P)g(\theta_{t-K})$:

$$
G_{\text{corr}}(t)
= \underbrace{P\,g(\theta_t)}_{\text{kept, current}}
+ \underbrace{(I-P)\,g(\theta_{t-K})}_{\text{dropped, stale}}.
$$

Decompose the *target* (dense current) the same way,
$g(\theta_t)=Pg(\theta_t)+(I-P)g(\theta_t)$. Then the **correction error** is

$$
G_{\text{corr}}(t) - g(\theta_t)
= (I-P)\big[g(\theta_{t-K}) - g(\theta_t)\big]
= -(I-P)\,\Delta_K g,
\qquad \Delta_K g \equiv g(\theta_t)-g(\theta_{t-K}).
$$

This is the entire content of the merger, and it is worth reading slowly:

1. **The persistent codec bias is gone.** Without the merger the error is the
   full $-(I-P)g(\theta_t)$ (the $0.63$ floor). With it, the error collapses to
   $-(I-P)\Delta_K g$ — the off-subspace part of *how much the gradient drifted
   over $K$ steps*. The bias is replaced by a **staleness residual**.
2. **It is additive and magnitude-preserving.** $G_{\text{corr}}=G_{\text{comp}}+\lambda\delta$
   adds a vector; it never rescales or reorients $G_{\text{comp}}$. At $\lambda=0$
   it returns $G_{\text{comp}}$ *bit-for-bit* (verified: the exact same tensor
   object, `spectral_filter.py:895`). The kept subspace $Pg(\theta_t)$ is
   transmitted untouched; only the missing complement is refilled.
3. **The residual lives in the dropped subspace.** Since $P\delta = P(I-P)g=0$,
   the correction adds energy *only* where the codec is blind. It is the precise
   complement of what was sent — the two are orthogonal and together telescope to
   (stale) dense.

So $G_{\text{corr}}$ equals the dense gradient up to a second-order-small term:
the gradient changes slowly over $K=5$ ticks ($\Delta_K g$ small for $\eta=10^{-6}$),
and we only miss its **off-subspace** part (which is itself the small $\approx
0.06\%$-energy direction). Quantitatively this predicts the observed jump:

$$
\text{floor error } \|(I-P)g\| \;\xrightarrow[\text{merger}]{}\; \|(I-P)\Delta_K g\|
\;\approx\; \|(I-P)g\|\cdot\frac{\|\Delta_K g\|}{\|g\|}\ll \|(I-P)g\|,
$$

i.e. the residual error shrinks by the *fractional gradient drift over 5 ticks*,
recovering essentially all of the $0.63\to0.75$ gap. **B2 lands at dense
because its fixed point is dense-on-stale-data**, and the staleness penalty
$-(I-P)\Delta_K g$ is small but **strictly $\ge 0$ in expected loss** — which is
exactly why it reaches parity *and not beyond* (§5).

### 3.4 Staleness, the hold, and the scale contract

Three implementation facts keep the derivation honest:

- **Hold between fires.** $\delta$ is refreshed only on fire-aligned ticks (when
  the exact $t-K$ ring entry exists) and **held** on the in-between ticks
  (`delayed_ef_refreshed`/`held` counters). Over a cadence window the per-tick
  injection telescopes: $\sum_t G_{\text{corr}}(t)\approx \sum_t Pg(\theta_t) +
  \sum_{\text{fires}}(I-P)g(\theta_{\cdot})$. The held $\delta$ is the
  cadence-window transport of the residual; $\beta_{\text{anc}}=0$ means $M$
  itself carries **no** EMA memory — the hold, not an EMA, is the carrier.
- **$\delta$ uses the *raw* anchor gradient** (read before any combiner;
  `update_anchor` docstring), so the residual is never itself corrected — no
  feedback loop through the merger.
- **Scale contract.** $M_{\text{rep}}$ is DP-mean reduced and
  $G_{\text{comp}}^{\text{ring}}$ is FSDP-mean under the *same* `agg_loss`
  normalization; the merger applies no rescale, so $\delta$ is well-scaled iff
  both feeds match (pinned by the scale-consistency unit test). If they did not
  match, $\delta$ would be a scaled — hence biased — residual.

---

## 4. The `signed_ema` merger and a formal contrast

### 4.1 The formula

`signed_ema` (`spectral_filter.py:393`) is

$$
\boxed{\;G_{\text{corr}} = \alpha\,G_{\text{noisy}} + (1-\alpha)\,\lvert G_{\text{noisy}}\rvert \odot \operatorname{sign}(M)\;}
$$

(element-wise $\odot$, $|\cdot|$). The **magnitude** comes from the fast
compressed gradient $G_{\text{noisy}}=G_{\text{comp}}$; the **sign** comes from
the stale anchor EMA $M$. $\alpha=1$ returns $G_{\text{noisy}}$; $\alpha=0$ is
pure sign-replacement, $G_{\text{corr}}=|G_{\text{comp}}|\odot\operatorname{sign}(M)$.
(Cold-$M$ guard: if $\|M\|\le\varepsilon$ it returns $G_{\text{comp}}$ unchanged
— never silently zeroes a matrix.) This is a **sign-SGD-like** estimator: it
discards the compressed gradient's own sign and trusts the anchor's.

### 4.2 Why it is biased: structural sign-disagreement

Write the per-coordinate signs $s^c_{\text{comp}}=\operatorname{sign}(G^c_{\text{comp}})$
and $s^c_M=\operatorname{sign}(M^c)$. On a coordinate where they **agree**,
$|G^c_{\text{comp}}|\,s^c_M = G^c_{\text{comp}}$ (untouched at any $\alpha$). On a
coordinate where they **disagree**,

$$
G^c_{\text{corr}} = \alpha\,G^c_{\text{comp}} + (1-\alpha)\lvert G^c_{\text{comp}}\rvert\,s^c_M
= \big(\alpha - (1-\alpha)\big)G^c_{\text{comp}} = (2\alpha-1)\,G^c_{\text{comp}}.
$$

At $\alpha=0$ this is $-G^c_{\text{comp}}$: a **full sign reversal**. The measured
disagreement fraction is $\approx50.4\%$ — *and it is structural*: it is already
$\approx50\%$ at the first warm step (one fresh anchor grad, near-zero EMA depth),
flat for the whole run, and uniform across all matrix types and all 28 layers
(not concentrated at the 7 compressed boundaries). The mechanism is simple and
unfixable by freshening:

> GRPO per-coordinate gradients are **near-zero-mean** (the group baseline
> subtracts the mean, §1.1; the advantage is a sibling contrast). Two independent
> estimators of a near-zero-mean quantity — the compressed live gradient and the
> stale anchor — agree on the sign of each coordinate at the **coin-flip** rate.
> So $\approx50\%$ disagreement is the *expected* behavior of any
> sign-replacement on this signal, not a defect of staleness, compression, or
> EMA depth.

Hence `signed_ema` injects a **bias** even in the limit of a perfect, fresh
anchor: on half the coordinates it overwrites a *correct, informative* sign with
an uncorrelated one. It is not de-biasing the gradient; it is **destroying the
per-coordinate sign-cancellation** that the dense gradient relies on. Measured
consequence: grad-norm inflates from dense $\approx0.387$ to $\approx3.3$ at
$\alpha=0$ (mean $\approx11$), because reversing signs on half the coordinates
removes the destructive interference that keeps $\|g\|$ small.

### 4.3 Why it sharpens, and why $\alpha\to0$ ignites a spiral

The failure mode is **not** low entropy — dense trains at *lower* entropy
($\approx0.122$) than any non-collapsing comm-eff arm and never ignites. The
killer is a **length reward-hack**, and the sign mechanism feeds it:

1. Sign-replacement is a **sign-SGD-like sharpening** operator. sign-SGD takes
   steps of fixed magnitude $|G_{\text{comp}}|$ in the sign-of-$M$ direction;
   combined with Adam's own normalization this *amplifies* small-magnitude
   coordinates and removes the gradient's natural magnitude weighting. On a
   no-KL/no-entropy surface there is **no brake** on the resulting drift.
2. Because reward is sequence-level and broadcast to tokens (§1.1), the
   reward-flat direction "same answer, more tokens" is nearly free. A sharpening
   operator that injects a small persistent force along this flat direction
   ratchets length upward. The token-mean aggregation then amplifies the tail
   ($\sim86\times$ in the EXP-27 mechanism analysis), locking in the runaway.
3. At $\alpha\to0$ the per-coordinate reversal coefficient $(2\alpha-1)\to-1$ is
   maximal, so the sharpening force is strongest and ignition is earliest
   (length explodes by step $\sim30$ at $\alpha=0$, step $\sim33$ at $\alpha=0.3$).
   $\alpha=0.5$ is the neutral point — $(2\alpha-1)=0$, disagreeing coordinates
   are *zeroed* rather than reversed — but even there the run is censored-unstable
   (already spiraling at steps 47–48; $P(\text{ignite by }100)\approx55$–$70\%$).

So $\alpha=0.5$ ($\approx0.7066$ on the invalid-M circuit) is the **best**
signed-EMA setting yet remains **dominated** by B2 ($0.7528$) and is the *only*
one worth keeping as a reference; $\alpha\to0$ is catastrophic
($\approx0.354$).

### 4.4 Formal contrast: additive vs multiplicative

| property | `delayed_ef` (B2) | `signed_ema` |
|---|---|---|
| operation | **additive**: $G_{\text{comp}}+\lambda\delta$ | **multiplicative**: $\lvert G_{\text{comp}}\rvert\odot\operatorname{sign}(M)$ (at $\alpha{=}0$) |
| magnitude | **preserved** ($G_{\text{comp}}$ sent untouched, residual added in the *orthogonal* dropped subspace) | $G_{\text{comp}}$'s **own** magnitude kept, but its **sign discarded** |
| direction | refilled along $(I-P)$, complementary to what was sent | reoriented per-coordinate to $\operatorname{sign}(M)$ |
| target / fixed point | dense-on-stale: error $=-(I-P)\Delta_K g$ (small) | sign-of-stale-dense: bias on $\approx50\%$ of coords (large) |
| bias in the perfect-anchor limit | $\to 0$ as $K\to0$ (unbiased in the limit) | **stays $\approx50\%$** even at $K\to0$ (structurally biased) |
| interaction with GRPO clip | safe: bounded additive nudge in a blind subspace; clip still caps each token | unsafe: reverses informative signs *before* the clip sees them, breaking sign-cancellation |
| measured | $0.7528$ = parity | $0.7066$ (best $\alpha$) → $0.354$ ($\alpha{=}0$) |

The one-sentence version: **B2 transmits the dense gradient and refills the part
the codec dropped; `signed_ema` throws away the gradient's sign and substitutes a
stale one.** The first is a consistent estimator of $g$ (up to staleness); the
second is an inconsistent one (biased on half the coordinates for all time).

### 4.5 Aside — `ef_powersgd` is *not* the same residual as B2

A subtlety worth pinning so the two are never conflated: `ef_powersgd`
(`spectral_filter.py:433`) uses $\text{comp}_t = M - \frac{\langle G,M\rangle}{\|G\|^2}G$,
the part of the *stale anchor* orthogonal to the *current live* $G$. By
construction $\langle \text{comp}_t, G\rangle = 0$ — it is *exactly tangential*,
a persistent force along a reward-flat direction, and the EXP-27 mechanism
analysis showed this rectifies into a length ratchet (it ignites at step $\sim61$;
dose-capping only *delays* it — "lag-not-dose"). B2's $\delta=(I-P)g(\theta_{t-K})$
is a genuine **codec residual** on a *paired* batch, not a projection of $M$
against the live grad. This is why B2 is stable and `ef_powersgd` is not, even
though both are "additive off-subspace" corrections. The distinction is *whose*
gradient and *which* projector defines the off-subspace.

---

## 5. The stale-reconstruction ceiling

This is the formal statement of why **no tested merger surpasses dense**, and the
specification of what a surpass method would have to inject.

### 5.1 Claim

> **Stale-reconstruction ceiling.** Let the anchor maintain $M$, an estimator of
> the dense gradient on a $K$-stale paired snapshot,
> $M \approx g_{\text{dense}}(\theta_{t-K})$, and let the merger be any
> deterministic, cross-rank-identical map
> $G_{\text{corr}} = \Phi(G_{\text{comp}}, M)$ whose only non-compressed input is
> $M$. Then the trajectory's attainable fixed point is the **dense** trajectory
> (up to the staleness residual), and **cannot strictly improve on dense**. To
> surpass dense, $\Phi$ must consume information that is **not a function of a
> stale dense gradient estimate**.

### 5.2 Argument

Both promoted mergers are of the form $\Phi(G_{\text{comp}},M)$ with $M$ a stale
dense estimate. Their *targets* (the quantity $G_{\text{corr}}$ converges toward
when the codec and staleness errors vanish) are:

- **B2**: $G_{\text{corr}} = Pg(\theta_t) + (I-P)g(\theta_{t-K}) \xrightarrow[K\to0]{} g(\theta_t)$.
  The fixed point is the **dense gradient**.
- **signed_ema**: $G_{\text{corr}} = |G_{\text{comp}}|\odot\operatorname{sign}(M)
  \xrightarrow[K\to0,\,\text{perfect}]{} |g|\odot\operatorname{sign}(g)$ on
  agreeing coords. The *best case* it aims at is again the dense **sign** pattern
  — it cannot aim past dense; it can only fail to reach it (the $50\%$ bias).

The argument is a **sufficient-statistic / fixed-point** bound, not an appeal to
intuition. The two non-compressed inputs available to $\Phi$ are
$G_{\text{comp}}=Pg(\theta_t)$ (a lossy view of the *current* gradient) and $M$
(a view of the *stale* gradient). Together they are measurable with respect to
$\mathcal F_t = \sigma\big(g(\theta_t),\,g(\theta_{t-K})\big)$ — the
$\sigma$-algebra generated by the current and stale **dense** gradients. Any
$\Phi(G_{\text{comp}},M)$ is therefore an $\mathcal F_t$-measurable random
variable. The *best* output in this class is the conditional target that
minimizes the update error, and since the dense update direction
$g(\theta_t)$ is itself $\mathcal F_t$-measurable, the error-minimizing
$\Phi^\star$ returns exactly $g(\theta_t)$ (achievable in the limit $K\to0$,
perfect inversion). No $\mathcal F_t$-measurable map can return a *better
descent direction than the dense gradient it can at best reproduce*, because
"better than dense" is by definition a direction that the dense first-order
oracle at this $(\theta_t,\text{LR})$ does **not** take — and that direction is
not a function of $\{g(\theta_t),g(\theta_{t-K})\}$. The very best $\Phi$ can do
is invert the codec and staleness maps and recover $g_{\text{dense}}(\theta_t)$
— i.e. reproduce the dense update, hence the **dense trajectory**, whose terminal
performance is the dense band ($0.75$–$0.78$). Hence:

$$
\text{(any merger using only $M$, a stale dense estimate)} \;\le\; \text{dense}.
$$

Empirically this is exactly what the EXP-31 four-lever tournament found: you
cannot beat dense by **reweighting** ($\lambda_t$ adaptive dose: null),
**accumulating** ($\delta$-momentum: null/regress), **perturbing** (isotropic
$\sigma$: null), or **de-noising** (control-variate: gated out, $\operatorname{cov}\approx0$)
a stale estimate of dense. Each lever is a different $\Phi$ with the same
information bottleneck $M$; the ceiling binds all of them.

### 5.3 What "surpassing dense" formally requires

The ceiling is an *information* statement, so the escape must be an *information*
injection — a signal $\xi$ that is **not** $\sigma(M)$-measurable (not derivable
from the stale dense gradient). Three admissible categories, with the test each
must pass:

1. **Curvature / second-order structure.** Dense-SGD-at-fixed-LR uses only the
   first moment $g$. A correction that uses genuine curvature (a Hessian-vector
   product, a preconditioner estimated from the *spread* of swarm gradients, a
   Fisher/natural-gradient term) injects information the dense first-order
   trajectory does not have. **Test**: does it require a quantity beyond a
   gradient mean? If yes, it can escape; if it is just a reweighted gradient, no.

2. **Genuinely new exploration that *converts*.** The codec sustains a measurably
   more diffuse policy than dense (uncompressed-generator $\text{ppl}$ $1.40$ vs
   dense $1.24$ at step 25) — *real* exploration the dense trajectory never
   sampled. But this is **generation-time** diversity, and val is **greedy
   mean@1**. The ceiling-relevant question is **conversion**: does the diversity
   relocate the **greedy mode** (training-time), or only widen the sampling
   distribution (eval-time)? Per the conversion-spine finding, all of mask /
   Gaussian / PowerSGD diversity is **train-only** at the boundary and repeats
   the PowerSGD null on the greedy bar. **Test**: does $\xi$ change the
   $\arg\max$ of the policy, not just its entropy? A surpass route here must be
   *conversion-positive*, e.g. raising $n$ and rollout temperature (the only
   compression-*specific* exploration knob), **with a dense$\times\{T,n\}$
   control** — otherwise any gain is an eval-diversity artifact, not a dense
   surpass. Prior: likely yields at most a pass@k edge, not a greedy-mean
   surpass.

3. **Multi-rank disagreement as signal, not noise.** The swarm has many
   independent compressed gradients. Their *disagreement* (variance across ranks)
   is a quantity $M$ — a mean — discards. If that disagreement encodes
   task-relevant structure (e.g. a robust/sharpness-aware direction: descend
   where ranks *agree*, damp where they *fight*), it is information beyond the
   dense mean. **Test**: is the signal in the *cross-rank second moment*, not the
   first? Caveat: the async-realism constraint requires it stay
   cross-rank-identical *after* aggregation and tolerate variable staleness — a
   genuine SAM-style term qualifies; isotropic perturbation (EXP-31 L4) does not,
   because it is uncorrelated noise, not disagreement structure.

The disqualifier, stated once: **any $\Phi$ that is a deterministic function of
$(G_{\text{comp}}, M)$ alone — reweighting, accumulating, sign-copying,
perturbing with rank-identical noise, de-noising — is inside the ceiling and
caps at dense.** The escape must read a quantity *outside* the stale-dense
sufficient statistic: curvature, a conversion-positive exploration signal, or a
cross-rank second moment.

---

## 6. Summary of the load-bearing results

1. **GRPO** optimizes a clipped surrogate with a **mean-zero, std-whitened
   group-relative advantage** broadcast to all tokens; reward is sequence-level,
   so uniform lengthening is nearly reward-neutral (the seed of the length hack).
2. The PowerSGD codec is **pure bias, near-zero variance** vs dense:
   $\mathbb E[G_{\text{comp}}]-g = -(I-P)g$. The bias is tiny per step ($\approx
   0.06\%$ energy, $\approx42{:}1$ SNR) but **persistent** (fixed $Q$), so it
   integrates **linearly** over 50 steps and drops the no-merger floor to
   $\approx0.63$.
3. **B2 / `delayed_ef`** is **additive, magnitude-preserving** de-biasing: its
   residual $\delta = M_{\text{rep}} - G_{\text{comp}}^{\text{ring}}(t-K)$ is the
   *exact paired codec residual* $(I-P)g(\theta_{t-K})$, refilling only the
   dropped subspace. Correction error $= -(I-P)\Delta_K g$ (second-order small)
   $\Rightarrow$ recovers the $0.63\to0.75$ gap and lands at **parity**.
4. **`signed_ema`** is **multiplicative sign-replacement**, structurally biased
   on $\approx50\%$ of coordinates (coin-flip sign-agreement of two estimators of
   a near-zero-mean gradient), a sign-SGD-style **sharpening** that ignites the
   length hack as $\alpha\to0$. Best $\alpha=0.5$ ($\approx0.71$) is **dominated**
   by B2.
5. **Stale-reconstruction ceiling**: any merger whose only non-compressed input
   is the stale dense estimate $M$ has its fixed point at **dense** and cannot
   surpass it (data-processing on the stale-dense sufficient statistic). EXP-31's
   four nulls are this ceiling, instantiated. **Surpass requires injecting
   information outside $\sigma(M)$**: curvature/2nd-order, a *conversion-positive*
   exploration signal (must move the greedy mode, with a dense$\times\{T,n\}$
   control), or a *cross-rank second-moment* (disagreement-as-signal) term.

*Cross-examination status (folded below in §7 once `systems`/`strategist` reply).*

---

## 7. Cross-examination log

*(populated as peers respond — see SendMessage thread)*
