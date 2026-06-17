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
estimate of the dense gradient lands at dense, never past it*, why EF reaches
that ceiling while signed_ema sits below it (a consistency argument matching the
measured $0.6300 < 0.7271 < 0.7528 \approx$ dense), and what new information a
method must inject to surpass it — the bridge to the strategist's surpass
program.

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
or staleness artifact. (Strictly, the ring stores the gradient compressed with
the *fire-time* projector $P_{t-K}$, so $\delta=(I-P_{t-K})g(\theta_{t-K})$; since
$Q$ is anchor-owned and converges, $P_{t-K}\approx P_t$ and the distinction is
negligible — I write a single $P$ below.) This is the property that distinguishes
B2 from `ef_powersgd` (§4.5), which uses $M-\operatorname{proj}_G(M)$ on the
*current* live gradient and is therefore exactly orthogonal-to-$G$ and not a true
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

So $\alpha=0.5$ is the **best** signed-EMA setting yet remains **dominated** by
B2 and is the *only* one worth keeping as a reference; $\alpha\to0$ is
catastrophic. The current **valid-M** number (EXP-32, post-#29 paired-replay
circuit) is $\alpha{=}0.5 \approx \mathbf{0.7271}$ — clears the $0.6300$ floor by
$\approx0.10$ but caps $\approx0.026$ **below** B2 ($0.7528$). (The often-quoted
$0.7066$ is the **legacy, invalid-M** EXP-25 number — do not cite it as current;
$\alpha{=}0$ collapsed to $\approx0.354$ on that same legacy circuit.)

### 4.4 Formal contrast: additive vs multiplicative

| property | `delayed_ef` (B2) | `signed_ema` |
|---|---|---|
| operation | **additive**: $G_{\text{comp}}+\lambda\delta$ | **multiplicative**: $\lvert G_{\text{comp}}\rvert\odot\operatorname{sign}(M)$ (at $\alpha{=}0$) |
| magnitude | **preserved** ($G_{\text{comp}}$ sent untouched, residual added in the *orthogonal* dropped subspace) | $G_{\text{comp}}$'s **own** magnitude kept, but its **sign discarded** |
| direction | refilled along $(I-P)$, complementary to what was sent | reoriented per-coordinate to $\operatorname{sign}(M)$ |
| target / fixed point | dense-on-stale: error $=-(I-P)\Delta_K g$ (small) | sign-of-stale-dense: bias on $\approx50\%$ of coords (large) |
| bias in the perfect-anchor limit | $\to 0$ as $K\to0$ (unbiased in the limit) | **stays $\approx50\%$** even at $K\to0$ (structurally biased) |
| interaction with GRPO clip | safe: bounded additive nudge in a blind subspace; clip still caps each token | unsafe: reverses informative signs *before* the clip sees them, breaking sign-cancellation |
| measured | $0.7528$ = parity (valid-M) | $0.7271$ (best $\alpha{=}0.5$, valid-M) → $0.354$ ($\alpha{=}0$, legacy) |

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

### 5.3 Where the two mergers land *under* the ceiling — and why EF is closer

The ceiling is an *upper* bound ($\le$ dense); it says nothing by itself about
*how close* a given $\Phi$ gets. The measured ordering
$0.6300 < \mathbf{0.7271}_{\text{signed\_ema}} < \mathbf{0.7528}_{\text{EF}}
\approx \text{dense } 0.75\text{–}0.78$ is explained by **how each $\Phi$ uses
the same $\mathcal F_t$ information** — i.e. by the *consistency* of the estimator,
which §3.3 and §4.2 already computed:

| | error of $\Phi$ vs dense $g(\theta_t)$ | behavior as $K\to0$ (perfect, fresh anchor) | lands |
|---|---|---|---|
| **no-merger** | $-(I-P)g(\theta_t)$ — the **full** persistent codec bias | unchanged (no anchor used) | $0.6300$ (floor) |
| **signed_ema** | sign flips/zeroing on the $\approx50\%$ disagreeing coords $\Rightarrow$ a **non-vanishing** $O(\|g\|)$ error | error **stays** $\approx50\%$-coord bias — does **not** vanish | $0.7271$ |
| **EF (B2)** | $-(I-P)\Delta_K g$ — only the **off-subspace gradient drift over $K$** | error $\to 0$ (consistent) | $0.7528 \approx$ dense (within noise; band $0.735$–$0.754$ overlaps the bottom of dense $0.75$–$0.78$, so EF $\le$ dense-ceiling, not $\ge$ pointwise) |

The decisive distinction is **consistency** in the $K\to0$ limit:

- **EF is a *consistent* estimator of the dense update.** Its error
  $-(I-P)\Delta_K g$ is $\le \|(I-P)g\|\cdot\frac{\|\Delta_K g\|}{\|g\|}$ and
  $\to 0$ as staleness $\to 0$. So EF approaches the ceiling from below and its
  residual gap to dense is *second-order small* (the off-subspace part of a
  5-tick drift). It lands **at** the ceiling, up to a tiny staleness penalty —
  hence parity, $0.7528 \lesssim$ dense.
- **signed_ema is an *inconsistent* estimator.** Its error does **not** vanish
  even with a perfect fresh anchor ($K\to0$): on $\approx50\%$ of coordinates it
  overwrites the (near-unbiased) compressed sign with a coin-flip sign, a
  *first-order* $O(\|g\|)$ bias that no amount of freshness removes (§4.2). It is
  strictly inside the ceiling by a finite margin — it recovers the part of the
  floor gap where the signs *happen* to agree (clearing $0.6300\to0.7271$,
  $\approx+0.10$) but forfeits the disagreeing half, capping $\approx0.026$
  **below** EF and below dense.

So both are bounded by the ceiling ($\le$ dense), but **EF reaches it and
signed_ema does not**, for the same reason a consistent estimator beats an
inconsistent one: EF's only residual is *staleness* (vanishing), signed_ema's is
*structural sign-bias* (non-vanishing). This is the within-bound ordering the
data shows, derived — not fitted.

The calibrated claim (stated verbatim and identically in `systems.md` §3, to keep
the two documents word-for-word aligned and to guard against over-reading a
one-draw difference):

> Both mergers recover most of the $0.123$ floor$\to$dense gap, but by different
> mechanisms: error-feedback re-injects the exact dropped residual (recovering
> $\approx$all of it, capped only by a second-order staleness term), whereas
> signed_ema carries no residual and recovers the gap only indirectly — by
> suppressing wrong-sign descent on the $\approx50\%$ of coordinates where the
> stale anchor disagrees (at $\alpha=0.5$ those coordinates are zeroed rather than
> corrected). Because signed_ema injects no off-subspace information and discards
> rather than reconstructs the disagreeing half, it caps strictly below
> error-feedback; the measured $\approx0.026$ shortfall is of the order of the
> disagreement fraction and is within single-draw noise, so it should be read as
> "dominated, by a margin consistent with the sign-bias mechanism," not as a
> precisely-predicted constant.

### 5.4 What "surpassing dense" formally requires

The ceiling is an *information* statement, so the escape must be an *information*
injection — a signal $\xi$ that is **not** $\sigma(M)$-measurable (not derivable
from the stale dense gradient). Three admissible categories, with the test each
must pass:

1. **Curvature / second-order structure.** A correction that uses genuine
   curvature (a Hessian-vector product, a preconditioner estimated from the
   *spread* of swarm gradients, a Fisher/natural-gradient term, or a
   trajectory-direction proxy $M_t-M_{t-1}\approx H\,\Delta\theta$ differenced
   from successive stale anchor gradients) injects second-order structure the
   first-order trajectory discards. **Test (sharper than it first looks)**: the
   honest control is **dense-Adam, not dense-SGD** — Adam *already* carries a
   *diagonal* second moment $v_t$. So a **diagonal** curvature proxy merely
   duplicates Adam and collapses back inside the ceiling; an escape must supply
   curvature Adam lacks — **non-diagonal / off-diagonal Hessian** information (e.g.
   the trajectory $H\Delta\theta$ term used as a non-diagonal preconditioner). The
   $M_t-M_{t-1}$ proxy is **async-admissible** (lagging anchor snapshots, DP-mean
   ⇒ cross-rank-identical, slow-varying ⇒ tolerates staleness) and ~free on the
   anchor — but only escapes if it is used *non-diagonally*.

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

3. **Multi-rank disagreement as signal, not noise.** The fast **swarm** (the DP
   ranks — the uncompressed axis, each holding a different data shard) produces
   many independent gradients per step. Their *disagreement* — the cross-rank
   **second moment** $\frac1R\sum_r (g_r-\bar g)(g_r-\bar g)^\top$ or its diagonal
   — is a quantity that both the dense mean $\bar g$ **and** the anchor $M$
   (itself a DP-mean) **discard**. If that disagreement encodes task-relevant
   structure — e.g. a robust / sharpness-aware direction (descend where ranks
   *agree*, damp where they *fight*), which provably optimizes a *different*
   objective than $\mathbb E[g]$ — it is information outside $\sigma(M)$. **Test**:
   is the signal in the *cross-rank second moment*, not the first? Async-realism
   caveats: (a) it must be computed from the **swarm** ranks, not the single slow
   anchor (the anchor is one node and has no within-itself disagreement); (b) the
   *aggregated* correction must remain cross-rank-identical (compute the
   second-moment-derived direction once on an all-reduced sufficient statistic —
   e.g. $\sum_r g_r$ and $\sum_r g_r\odot g_r$ — then every rank applies the
   identical direction, exactly as $Q$/$M$ are already broadcast) and tolerate
   variable staleness (the disagreement *geometry* is slow-varying, so a lagged
   correction is still useful); and **(c) the second moment must be formed from
   concurrent per-rank gradients at the *same* $\theta$** — it measures
   *data-induced* variance (different shards, identical weights). Differencing
   gradients across *different* $\theta$/lags contaminates data-variance with
   trajectory drift and collapses back to the heterogeneous-staleness null (a
   noisier stale-dense estimate, inside $\sigma(M)$); the staleness tolerance is on
   *applying* the derived correction, not on mixing $\theta$'s into the estimate.
   A genuine SAM-style or variance-adaptive term qualifies; isotropic perturbation
   (EXP-31 L4) does **not** — it is rank-identical *uncorrelated noise*, carrying
   no disagreement structure, which is exactly why it was null. **(d) — the
   decisive condition: it must use the variance as an *objective*, not as a
   step-scale.** A *variance-normalized step* $g/\sqrt{\operatorname{Var}_r[g_r]}$
   is just a **diagonal preconditioner**, and although it is not literally Adam's
   $v_t$ (Adam's $v_t=\mathbb E_t[g^2]$ is the *temporal* second moment, conflating
   $\bar g^2$ + temporal variance; the cross-rank $\operatorname{Var}_r$ is the
   *spatial/data-shard* variance with $\bar g^2$ subtracted, so they differ on
   high-mean-low-disagreement coordinates), it is the **same category of object** —
   a per-coordinate rescale of an $\mathbb E[g]$ step — so it only buys "a better
   diagonal than Adam's" and is **dominated by / equivalent to** the curvature
   route (category 1). The clean escape is the **variance-as-objective** form:
   optimize $\min_\theta \big[\,L(\theta)+\lambda\,\mathcal R(\operatorname{Var}_r[g_r(\theta)])\,\big]$
   (or SAM's $\max_{\|\epsilon\|\le\rho}L(\theta+\epsilon)$ ascent-then-descent),
   whose gradient carries a $\nabla_\theta\operatorname{Var}_r$ term and therefore a
   **different fixed point** (a flat/robust minimum) than dense's $\mathbb E[g]$
   optimum — that, not a rescaled step, is what genuinely escapes both $\sigma(M)$
   and dense-Adam. So this route is the **most promising** of the three *only in
   its objective-level form*; as a step-scale it ties the curvature route. It is
   the one signal the compressed swarm has in abundance that a single dense
   trajectory structurally lacks.

The disqualifier, stated once: **any $\Phi$ that is a deterministic function of
$(G_{\text{comp}}, M)$ alone — reweighting, accumulating, sign-copying,
perturbing with rank-identical noise, de-noising — is inside the ceiling and
caps at dense.** The escape must read a quantity *outside* the stale-dense
sufficient statistic: curvature, a conversion-positive exploration signal, or a
cross-rank second moment.

**The diagonal-preconditioner kill-check (shared by the two second-moment
co-leads, categories 1 and 3).** Categories 1 (curvature) and 3 (cross-rank
second moment) are the only two that read a quantity dense-**Adam** structurally
lacks — but they share a single failure mode, and the honest control for both is
dense-Adam (not dense-SGD), because Adam already applies a *diagonal*
preconditioner $D=\operatorname{diag}(v_t)^{-1/2}$. Both routes, in their naive
forms, also reduce to a diagonal rescale $D\odot g$ of the $\mathbb E[g]$ step:
the curvature route as a diagonal Fisher/Hessian proxy, the second-moment route
as a variance-normalized step $g/\sqrt{\operatorname{Var}_r}$. **Any output
reducible to $D\odot g$ for some diagonal $D$ — however cleverly $D$ is estimated
(from $M_t-M_{t-1}$, from cross-shard variance, …) — lives inside dense-Adam's
hypothesis class and collapses** (a different or "better" diagonal is still a
diagonal). The two co-leads are precisely the **two distinct exits** from that
class:

- **Category 1 (R5) exits via off-diagonal *structure*** — a non-separable
  $H\Delta\theta$ / off-diagonal coupling no diagonal can represent (same
  objective, richer preconditioner).
- **Category 3 (R3) exits via a different *objective*** — a variance *penalty*
  $\min_\theta[L+\lambda\mathcal R(\operatorname{Var}_r)]$ (or SAM's min-max) with
  a different fixed point (not a rescale of the same objective's gradient).

So the unified kill-check is: *is the correction reducible to $D\odot g$? If yes,
it collapses to dense-Adam; if no — off-diagonal coupling, or a fixed-point-
changing objective — it is an admissible escape.* This is why R5 and R3 are
genuinely distinct surpass routes yet share one disqualifier.

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
   length hack as $\alpha\to0$. Best $\alpha=0.5$ ($\approx0.7271$, valid-M) is
   **dominated** by B2.
5. **Stale-reconstruction ceiling**: any merger that is a deterministic map
   $\Phi(G_{\text{comp}}, M)$ — i.e. $\mathcal F_t$-measurable w.r.t.
   $\sigma(g(\theta_t), g(\theta_{t-K}))$, the stale+current dense gradients — has
   its error-minimizing fixed point at the **dense** update and cannot surpass it
   ($\le$ dense). **Within** that bound, EF lands *at* it and signed_ema *below*
   it (the measured $0.6300 < 0.7271 < 0.7528 \approx$ dense) for a consistency
   reason: EF's residual is *staleness* $-(I-P)\Delta_K g$ (vanishing as $K\to0$,
   a consistent estimator), signed_ema's residual is *structural sign-bias* on
   $\approx50\%$ of coords (non-vanishing even at $K\to0$, inconsistent) — §5.3.
   EXP-31's four nulls (reweight / accumulate / perturb / de-noise) are the bound,
   instantiated. **Surpass requires injecting information outside $\sigma(M)$**:
   curvature/2nd-order, a *conversion-positive* exploration signal (must move the
   greedy mode, with a dense$\times\{T,n\}$ control), or — most promisingly — a
   *cross-rank second-moment* (swarm disagreement-as-signal) term, the one
   quantity the compressed swarm has and a single dense trajectory lacks (§5.4).

*Cross-examination status: **CONVERGED with `systems`** on all four points (both
quantitative predictions confirmed against W&B; the signed_ema number corrected
to the valid-M $0.7271$; signed_ema instability settled as STRUCTURAL by three
agreed grounds — see §7.2). `strategist` completed independently; the §5.4
scaffold stands as their ceiling constraint.*

---

## 7. Cross-examination

### 7.1 Anticipated objections and standing answers

**O1 — "B2 keeps the fresh kept-subspace $Pg(\theta_t)$, so it's a fresh/stale
*hybrid*, not purely stale. Can the hybrid beat dense?"** No. The hybrid
$Pg(\theta_t)+(I-P)g(\theta_{t-K})$ is still $\mathcal F_t$-measurable with
$\mathcal F_t=\sigma(g(\theta_t),g(\theta_{t-K}))$ — the ceiling's sufficient
statistic was *deliberately* defined to include **both** gradients precisely so
the hybrid falls inside it. The fresh kept-subspace is what makes the staleness
penalty $-(I-P)\Delta_K g$ second-order (hence parity), not a route past dense.

**O2 — "B2's central band (0.735–0.754) may sit a hair below dense-this-box
(0.7506). Does 'parity' overstate?"** The math predicts B2 $=$ dense $-$ a small,
**non-negative** staleness penalty $-(I-P)\Delta_K g$ — i.e. B2 should land *at or
just below* dense, never above. A central tendency a hair under dense, inside
single-draw noise ($\pm0.024$), is *exactly* the prediction, and is the honest
meaning of "parity." If B2 ever measured *above* dense beyond noise, my model
would be **wrong** — that has not happened, which is corroboration, not a gap.

**O3 — "signed_ema $\alpha=0.5$ survived 50 steps in some runs — is it really
unstable?"** That survival is **censored**: the EXP-27 post-mortem found
$\alpha=0.5$ already in the early spiral at its endpoint (consecutive cap-pins at
steps 47–48). The math says $\alpha=0.5$ only *zeroes* the disagreeing
coordinates rather than reversing them — it removes the strongest sharpening
term but still discards half the gradient's sign information, so it is the
least-bad, not stable. Stability is a censored observation; do not read 50-step
survival as safety.

**O4 — "Is the $\sim50\%$ sign-disagreement maybe a staleness/EMA artifact that a
fresher anchor would fix?"** No (this is the structural-vs-tuning question
settled with `systems`). It is $\approx50.4\%$ at the *first warm step* (fresh
$M$, near-zero EMA depth, $K{=}4$) and flat/uniform across all layers — the
coin-flip agreement of two estimators of a **near-zero-mean** GRPO per-coordinate
gradient. No $K$, $\beta$, or $\alpha$ removes it; sign-replacement is the wrong
operator for a near-zero-mean signal.

### 7.2 Live cross-examination log

**`systems` — CONVERGED on all four points** (their `systems.md` §2, §4, §6; we
exchanged predictions and they confirmed against W&B receipts):

- **Prediction 1 (the $0.6300\to0.7528$ jump) — confirmed.** systems confirms
  the PowerSGD drop is a *deterministic structured bias, not zero-mean noise*;
  $\|(I-P)g\|$ carries only $0.058\%$ of gradient energy at **SNR $\approx42{:}1$**
  (a signal-to-**bias** ratio, variance term negligible — my §2.2 model), and $Q$
  converges to a stable subspace (recon error flat $\approx0.024$) so the *same*
  off-subspace direction is dropped every step. $\delta=M_{\text{rep}}-G_{\text{comp}}^{\text{ring}}$
  is exactly that dropped residual on the same $(\text{batch},\theta)$ ⇒ EF
  de-biases toward dense ⇒ parity. Corroboration they added: EXP-20 found
  compressed steps booked **57–95%** of the train-reward gain, the clean step
  flushing only the small ($4.8$–$19.6\%$) accumulated off-subspace bias — i.e.
  the bias is small per step but **integrated**, exactly the §2.3 argument.
- **Prediction 2 (signed_ema grad-norm vs $\alpha$) — confirmed.** grad-norm
  inflates dense $0.387 \to 3.3$ at $\alpha{=}0$ (mean $11$); length-hack ignites
  only under sign-reversal ($\alpha{=}0$ @step 30, $\alpha{=}0.3$ @step 33),
  $\alpha{=}0.5$ censored-unstable (cap-pins steps 47–48).
- **Number correction (material) — folded in.** The current **valid-M** signed_ema
  number is **$0.7271$** (EXP-32, post-#29), **not** $0.7066$ (legacy invalid-M,
  EXP-25). My §4.3 / §4.4 / §6 now cite $0.7271$. Measured ordering
  $0.6300 < 0.7271 < 0.7528 \approx \text{dense } 0.75$–$0.78$ — matches my
  derivation's predicted ordering exactly.
- **SETTLED: signed_ema instability is STRUCTURAL** (both sides, independently).
  systems' three grounds = my three: (1) it spans $\alpha$ (collapse at $0$,
  censored-unstable at $0.5$); (2) $50.4\%$ sign-disagreement at the *first* warm
  step with $\approx$zero EMA depth, flat/uniform across layers ⇒ not a
  staleness/$\beta$/$\alpha$ artifact; (3) the **direction-preserving** mergers
  (B2 `delayed_ef`, `ef_powersgd`, cos$(G_{\text{comp}},G_{\text{corr}})=0.956$
  vs signed_ema's $0.717$) do **not** spiral on the same substrate ⇒ the cause is
  the **sign-replacement operator**, not the substrate. *One agreed answer:
  STRUCTURAL.*

**`strategist` — four candidate surpass routes adjudicated** (R1–R4, relayed via
team-lead). The §5.4 scaffold is the rubric; verdicts in §7.3. Headline: only two
routes are even admissible escapes (R1, R4), and both are *conditional* — R1
because it must demonstrate a compression$\times$exploration *interaction*, R4
because it *redefines the objective* to OOD generalization. R2 and R3 collapse
back inside the ceiling.

### 7.3 Surpass-route verdicts (against the $\sigma(M)$ ceiling)

The ceiling test: a route escapes iff it injects information **not** measurable
w.r.t. $\sigma(g(\theta_t), g(\theta_{t-K}))$ — the stale+current *dense
gradients at the fixed control surface*. The surpass bar is the apples-to-apples
dense **band $0.75$–$0.78$** (dense draw $0.7839$), not $0.75$.

**R1 — raise $n$ + rollout temperature on the compressed circuit (matched
dense$\times\{T,n\}$ control). VERDICT: admissible escape, win conditional on a
measured interaction.** $n$ and $T$ change *which* gradient is computed (the
rollout/data distribution $g$ is taken over), so they move $g(\theta_t)$ itself
— they are **outside** $\sigma(M)$ for the baseline-$(T,n)$ statistic. So R1 is
**not** a $\Phi(G_{\text{comp}},M)$ reconstruction lever; the hard-reject list
does not bind it. *But* against the mandatory dense$\times\{T,n\}$ control you
have defined a **new** ceiling = dense-at-matched-$(T,n)$; R1 beats it **only if
compression and high-$T/n$ interact** — the compressed circuit must *convert* the
extra exploration into a better trajectory than dense does at the same $(T,n)$.
That interaction is the one genuinely untested, compression-specific bet. On the
greedy bar: val is greedy mean@1, so raising $T/n$ almost surely buys a *pass@k*
edge; a greedy-mode surpass needs the exploration to be **conversion-positive**
(§5.4 cat-2). Prior: pass@k-only is most likely — the codec *already* sustains a
more diffuse policy (ppl $1.40$ vs dense $1.24$) and did **not** convert. Top
live bet, honest prior modest, control mandatory.

**R4 — compression as a flat-minima regularizer $\to$ test/OOD edge (conceding
train parity). VERDICT: escapes the ceiling, but changes the objective.** The
ceiling bounds the *train* gradient / train-objective optimum; it says nothing
about *test-distribution* risk. The persistent codec bias $(I-P)g$ is a coherent,
fixed-direction per-step perturbation — *plausibly* a flatness-biasing regularizer
(though, being coherent rather than stochastic, it is closer to a fixed
preconditioner than to SGD noise, so flat-minima is **not** guaranteed). It
escapes because it targets a metric the ceiling doesn't cover — but it is **not**
a GSM8K-greedy-mean surpass; it *redefines* "done." Caveats: unmeasurable today
(no OOD eval wired — GSM8K-only); cheapest test = eval-only OOD pass
(SVAMP/ASDiv/MATH) over existing B2-vs-dense checkpoints; needs a sharpness/Hessian
measurement. Honest prior: coherent bias is ~50/50 to land in a *worse* basin.

**R2 — anchor $M$ as an advantage-side baseline. VERDICT: collapse — REJECT.**
The GRPO advantage is **already** optimally baselined by the group mean
($\hat A_i=(r_i-\text{mean})/\text{std}$, the variance-minimizing constant
baseline, §1.1). An $M$-derived baseline is $\sigma(M)$-measurable ⇒ inside the
ceiling. At best it reduces variance — which does **not** move the fixed point of
an unbiased estimator (same dense optimum). And EXP-31 L1 already gated out:
$\operatorname{cov}(G_{\text{comp}},M)\approx0$ ⇒ nothing to cancel ⇒ an
$M$-baseline *raises* variance.

**R3 — heterogeneous (per-worker) staleness. VERDICT: collapse — REJECT.** A
spread $\{g(\theta_{t-K_j})\}$ is still a set of *dense gradients at various
lags*, all measurable w.r.t. $\sigma(g(\theta_{t-K_1}),\dots)$ — combining them is
exactly "accumulating a stale dense estimate" (the $K$-uniform version is EXP-31
L2 $\delta$-momentum, null). Two extra strikes: the substrate has **no**
per-worker-staleness axis ($\texttt{delay\_K}$ is one uniform scalar), and it
presses on the async-realism invariant (one slow anchor, not workers-at-different
-lags — a different architecture).

**Net:** the only signal genuinely outside $\sigma(M)$ that lives on *this*
substrate is the **cross-rank second moment** (swarm disagreement, SAM-style;
§5.4 cat-3). R1 is the top *runnable* near-term bet but conditional on a measured
interaction; R4 is a clean escape only by redefining the goal to generalization.
No route gives a *free* greedy-mean surpass.
