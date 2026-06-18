# Q-Cadence / Codec Analysis — why comm-eff GRPO degrades at cadence/delay 20/20

**Analyst:** member 2 (Q-cadence / codec).
**Assigned hypothesis:** *"the Q correction / PowerSGD basis refresh is too infrequent."*
**Scope:** the PowerSGD basis `Q` and its refresh cadence (`anchor.cadence`), grounded in the
codec source + the EXP-36B / EXP-37 record. The merger-staleness axis (`delay_K`, M) is my
teammate's; I argue the confound split explicitly in §6.

**Verdict in one line:** the cadence hypothesis is **partially true but NOT the dominant cause.**
A frozen `Q` injects a *bounded, self-healing* reconstruction error that does not by itself explain
the −0.088 val@50 drop or the late spiral. The dominant damage at 20/20 comes from **M-staleness
(delay_K) feeding the merger**, which cadence *amplifies* through two specific, code-visible
mechanisms: (a) a **per-refresh subspace-rotation discontinuity** that shows up as a clipfrac spike
at every Q-refresh (steps 10/20/.../100 in EXP-37), and (b) a **longer hold window** over which the
merger applies a stale correction against a frozen basis. I attribute roughly **20–35% of the
degradation to cadence-specific effects** and the rest to delay_K. Decoupling (cadence=5 + delay_K=20
vs cadence=20 + delay_K=5) is cheap and is the single experiment that settles it.

---

## 0. Mechanism map (file:line ground truth)

How `Q` lives and is refreshed, end to end:

- **`Q` is a frozen, per-boundary, cross-rank-identical orthonormal basis `(H=1536, r=77)`**;
  the codec compresses the boundary activation/gradient as the self-adjoint projection
  `P·g = Q Qᵀ g` (`Q` detached, `M` in-graph ⇒ no STE).
  `powersgd_activation.py:17-43` (docstring), bootstrap `init_basis` `:142-162`,
  `orthonormalize` (fp32 QR, required) `:97-141`.
- **The anchor is the SOLE writer of `Q`** (`anchor.owns_q=true`). The fast path's
  `maybe_update_basis` is **fail-closed** by a hard assert when `anchor_owns_q` is set
  (`powersgd_activation.py:708-716`) and is gated off in the engine
  (`engine_workers.py:916-918`, `fast_owns_q = not anchor_owns_q`).
- **`Q` is refreshed only at anchor fires.** The anchor refresh runs
  `anchor_update_basis()` then `broadcast_basis(src=0)`
  (`transformer_impl.py:2081-2082`), both **inside the same `anchor_should_fire(step, cadence)`
  gate** (`transformer_impl.py:1474`). `anchor_update_basis` is the same block-power-iteration
  `Q ← orth(V)` math as the fast path, DP-synced (`sync_basis=true`) then `orth`
  (`powersgd_activation.py:1110-1207`). `V` is built from the **anchor's stale-weight
  (θ_{t−K}) forward activations** (`set_anchor_sketch_mode`, `transformer_impl.py:1717`).
- **`warm_start=true`** ⇒ `Q` is NOT reset between refreshes; each fire takes one
  block-power-iteration step `orth(V)` from the *current* `Q`
  (`powersgd_activation.py:740-772`, `warm_start` branch is the non-cold path).
- **Cadence and delay_K are gated by the SAME per-tick counter.**
  `state.anchor_step += 1; step = state.anchor_step` (`transformer_impl.py:1342-1343`).
  This counter advances once per `train_batch` = once per optimizer/mini-batch tick (verl does
  one `optimizer_step` per `train_batch`), NOT per global step. With the locked surface
  (`train_batch=128`, `ppo_mini=64`) there are **2 ticks per global step**, so:
  - `cadence=20` ⇒ `Q` refreshed **every 10 global steps**.
  - `cadence=5` ⇒ `Q` refreshed **every 2.5 global steps** (4× more often).
  - **Empirically confirmed in EXP-37:** `actor/comm_eff/anchor_q_updates` increments at exactly
    steps 10, 20, 30, …, 100 (q_upd 0→1 at s10, …, 9→10 at s100). 10 refreshes over 100 steps.
- **`anchor.cadence` default = 20, `delay_K` default = 20** (`transformer_impl.py:1338-1339`).
  Both read from `anchor_cfg`; the accel surface pins them to 5/5 (FIXED_CONTROL_SURFACE.md:61).
  **EXP-37 raised BOTH to 20 together** (resolved_params: `anchor.cadence=20`, `anchor.delay_K=20`).

**Merger coupling (`spectral_filter.py`):**
- **signed_ema** (EXP-36B / EXP-37): `G_corr = α·G_noisy + (1−α)·|G_noisy|·sign(M)`
  (`spectral_filter.py:403-440`). Magnitude from `G_noisy` (= the **Q-projected** fast gradient
  `G_comp`), **sign from M** (the stale anchor EMA). The merger does not index `Q` directly, but
  `G_noisy` *is* the Q-projection, so a stale `Q` enters through the magnitude term.
- **ef_powersgd** (EXP-38): `comp_t = M_anchor − P_{G_comp}(M_anchor)` (the part of M the current
  Q-projected `G_comp` does NOT span), norm-clipped, `G_corr = G_comp + e_t`, no sign
  (`spectral_filter.py:446-507`).
- **delayed_ef** (legacy B2 0.7528): `delta = M_rep − G_comp_ring(t−K)`, refreshed at fires, held
  between (`spectral_filter.py:22-27, 142-145`).

---

## 1. Formal Q-staleness / reconstruction-error model

### 1.1 Why PowerSGD works at all (the low-rank premise)
The boundary gradient `g_t` is empirically **low-rank**: at `r=77` the relative reconstruction
error `‖g − P_Q g‖/‖g‖ ≈ 2%`, flat across `r ∈ [77,102]` (EXP-20, issue #21 theory). So
`P_Q = Q Qᵀ` captures ~98% of the gradient energy when `Q` spans the dominant singular directions.
The codec is therefore a **structured biased estimator: small bias, low variance** — bias is the
off-subspace energy `−(I−P_Q)g`, relative size `√(1−recon²)`.

### 1.2 What "Q frozen longer" does to the error
Let `U_t = span(top-r right singular vectors of the boundary 2nd moment at step t)` be the true
dominant subspace, and `Q_τ` the basis last refreshed at fire `τ ≤ t`. Define the
**subspace drift angle** `θ(t,τ) = ∠(span Q_τ, U_t)` (largest principal angle). The
reconstruction error decomposes cleanly:

```
‖g_t − P_{Q_τ} g_t‖²  =  ‖(I − Q_τ Qᵀ_τ) g_t‖²
                      =  E_off^floor(t)              (the always-present off-r tail, ~2%)
                      +  E_drift(t,τ)                (energy of the in-U_t directions that
                                                       Q_τ has rotated away from)
```

- **`E_off^floor`** is the irreducible ~2% tail that even a fresh `Q` leaves — present at every
  cadence, NOT a cadence effect.
- **`E_drift(t,τ)` grows with the hold age `(t−τ)`** because `U_t` rotates as θ evolves. To first
  order, if the dominant subspace rotates at angular rate `ω` per step, the captured-energy loss
  over a hold of length `Δ = (t−τ)` scales like `Σ_k σ_k² sin²(ω_k Δ)` — **monotone increasing in Δ
  until the next refresh resets it to ~0**. Cadence 20 (10-step holds) gives `E_drift` up to 4× the
  hold length of cadence 5 (2.5-step holds), so the *peak* drift energy per window is larger and the
  *time-average* drift is larger.

### 1.3 The decisive property: warm-start makes drift SELF-HEALING and BOUNDED
Because `warm_start=true` and `Q ← orth(V_t)` with `V_t = Mᵀ(M Q_τ)` is exactly **one block-power
iteration toward the current `U_t`** (`powersgd_activation.py:740-772, 1179-1196`), each refresh does
not start cold — it pulls `Q` one power step toward the *current* dominant subspace. The drift that
accumulated over the hold is **largely erased at the very next fire** (power iteration converges
geometrically in the eigengap). Consequence:

> A frozen `Q` injects a **bounded, oscillating, self-correcting** reconstruction error: it ramps up
> over each hold window and snaps back at each refresh. It does **not** secularly diverge, because the
> refresh re-locks onto `U_t`. Even at cadence 20 the worst case is "a somewhat-staler projector for
> 10 steps," not "a basis that drifts off forever."

This is why the cadence hypothesis is *bounded* in impact. Contrast with **M-staleness**, which has
no such self-healing inner loop on the fast path between fires — the merger holds a fixed stale
correction (`delayed_ef` literally `_delayed_ef_held`, `spectral_filter.py:237-244`; signed_ema's
`sign(M)` is whatever the last fire wrote).

### 1.4 Energy LOST vs BIAS injected — they are different objects
- **Energy lost** = `E_drift`: directions in `U_t` that `Q_τ` no longer spans ⇒ that gradient
  component is **dropped** (not seen by the optimizer that step). This is a *coverage* loss — it
  slows learning (a smaller effective step in the right directions) but is **mean-preserving in the
  spanned subspace** and self-heals at refresh.
- **Bias injected** = the merger's interaction with the stale projection. signed_ema takes
  `|G_noisy|` (magnitude of the **Q_τ-projected** grad) and multiplies by `sign(M)`. If `Q_τ` has
  rotated, `|G_noisy|` is a magnitude measured in the *wrong* coordinates, then stamped with a
  possibly-wrong stale sign. This is a **coherent, fixed-direction bias** for the whole hold window —
  it does NOT average out across steps (the same `Q_τ` and same `sign(M)` are reused). This is the
  damaging channel, and it is *the cadence × delay_K interaction*, not pure cadence.

**Takeaway:** pure-cadence (energy-lost) damage is bounded and self-healing; the harmful part is the
bias the *merger* builds on top of a frozen `Q`, which is the coupling term (§3).

---

## 2. "Q computed stale" vs "Q refreshed rarely" — separating the two sub-effects

These are genuinely distinct knobs that EXP-37 merged:

| Sub-effect | Source | Controlled by | Sign of harm |
|---|---|---|---|
| **(A) Q is computed from θ_{t−K} activations** | `V` built on stale-weight forward (`transformer_impl.py:1717`, anchor sketch mode) | **delay_K** | `Q` aims at the dominant subspace of the activations *as they were K ticks ago* |
| **(B) Q is refreshed rarely** | refresh only at `cadence` fires | **cadence** | `Q` is then *held* frozen, so even that stale aim degrades further over the hold |

- **(A)** is a *delay_K* effect: the *target* `U_{t−K}` the basis aims at is itself stale. At
  delay_K=20 (10 global steps) the activation subspace `Q` is fit to is 10 steps behind the live one.
- **(B)** is the *cadence* effect proper: how long that already-stale aim is reused.

**Which dominates at 20/20?** The activation **subspace `U`** is far more slowly-varying than the
gradient **sign pattern**. Activation second-moment geometry (what `Q` tracks) is dominated by the
token-embedding / residual-stream statistics, which barely move across 10 steps at lr=1e-6 — the
~2% reconstruction floor was *flat* across a wide window in EXP-20. So both (A) and (B) are **small
for `Q` itself**: a 10-step-stale, 10-step-frozen activation basis still captures the dominant
activation directions well (the eigengap is large and slow). **The cadence/delay damage to the
*codec* is second-order.** The damage that matters is to **M** (the gradient *mean/sign*, my
teammate's axis), which moves fast and whose staleness the merger folds in directly.

> **Conclusion for §2:** at 20/20, neither (A) nor (B) meaningfully breaks the *codec*. A 10-step
> stale, 10-step frozen `Q` is still a good projector. The cadence hypothesis fails to explain the
> magnitude of the regression *on its own*.

---

## 3. Coupling with the merger — why infrequent Q makes the merger worse, and why the spiral is LATE

### 3.1 The per-refresh discontinuity (the cleanest cadence signature in the data)
`anchor_update_basis()` and `broadcast_basis()` fire together (`transformer_impl.py:2081-2082`), so
**at each fire `Q` jumps discontinuously** from `Q_τ` to `Q_{τ+cadence}`. The compressed gradient
`G_comp = Q Qᵀ g` therefore changes coordinates discontinuously, and the importance ratio `ρ`
(old-logprob recompute vs current) — which is exactly `1` *within* a frozen-Q window because both
GRPO forwards share the projector (issue #21: `ρ→0.999`) — gets a one-step **mismatch shock** at the
boundary where the projector changed between the batch's generation and its training.

**This is visible in EXP-37 as a clipfrac spike at every single Q-refresh step:**

```
step:  10    20    30    40    50    60    70    80    90    100
clipfrac: .188  .177  .143  .146  .113  .158  .128  .116  .095  .125    <- refresh steps
between:  ~.004 .004  .006  .025  .03   .02   .017  .02   (rising)       <- held-Q steps
q_upd:    1     2     3     4     5     6     7     8     9     10
```

Every 10th step (each new `Q`) has a clipfrac ~3–40× the surrounding floor. At cadence 5 these shocks
would land every 2.5 steps but be **individually smaller** (the projector moves less per refresh,
warm-start having taken more frequent, smaller power steps). **Bigger, rarer projector jumps are a
direct cadence cost** — and a larger per-refresh clip means more clipped (lost) policy-gradient signal
concentrated at those steps. This is the most defensible *purely-cadence* harm in the record.

### 3.2 Why infrequent Q amplifies the merger's destabilization
The memory's locked finding is that **folding stale M into the fast grad is the destabilizer**
(canonical-anchor memory §post-mortem (3): the *merger* is the killer, not the substrate). Cadence
amplifies this through the **hold window**:

- signed_ema applies `sign(M_τ)` and `|Q_τ-projected g|` for the **entire** `cadence`-tick window.
  Both the stale sign *and* the frozen projector are reused for 10 steps at cadence 20 vs 2.5 at
  cadence 5. So a *single bad M/Q estimate persists 4× longer*. If `sign(M_τ)` is wrong on a subset
  of coordinates (the memory's structural ~50% sign-disagreement at warm steps,
  no-merger-floor memory §instability), that wrong sign is hammered into the same coordinates for 10
  consecutive steps — a far stronger push toward a degenerate (length-hack) basin than 2.5 steps of
  it before a correction.
- The merger correction is therefore **piecewise-constant in M/Q with a `cadence`-long plateau**.
  Longer plateaus = lower-frequency correction = the fast policy drifts further between corrections
  and the correction it eventually gets is staler.

### 3.3 Why the spiral is LATE (steps 90–100), not immediate
This is the key qualitative fact and cadence explains its *timing* well:

1. **Entropy ratchets down monotonically, one notch per hold window.** EXP-37 entropy: 2.33 (s11) →
   1.99 (s31) → 1.31 (s41) → 1.04 (s58) → 0.85 (s86) → 0.42 (s100). Each ~10-step plateau (one
   frozen-`Q`/`sign(M)` window) pushes entropy down a step, because the same sharpening direction is
   applied coherently for the whole window. The system **loses exploration in discrete cadence-sized
   decrements**.
2. **The length spiral ignites once entropy crosses a low threshold** (~0.6–0.7 here, between s90 and
   s92): len_mean 188 (s90) → 250 (s92) → 373 (s96) → 683 (s100), grad_norm 2→32, clipfrac climbing.
   This is the known length-hack / sharpening-spiral mechanism (entropy-collapse memory: entropy is a
   *follower* of the length spiral; the carrier is the merger).
3. **Why late:** the spiral is a *cumulative* threshold-crossing, not an instantaneous instability. It
   takes ~9 hold-windows of coherent sharpening to ratchet entropy low enough to ignite. **Fewer,
   longer windows (cadence 20) ratchet harder per window and reach the threshold sooner / harder than
   many short windows (cadence 5)**, where each window's coherent push is shorter and is interrupted by
   a fresh M/Q 4× as often (re-injecting some diversity / re-aligning the correction). At cadence 5
   (EXP-36B) the same merger stayed stable to step 50 with no spiral and val 0.7362.

So: **cadence does not *cause* the spiral (the merger does), but cadence sets the *clock* — longer
holds let the coherent sharpening accumulate undisturbed for longer, which both deepens each ratchet
step and removes the periodic re-alignment, bringing ignition within the 100-step horizon.**

---

## 4. Falsifiable predictions for EXP-38 (ef_powersgd, ef_decay=0.9/ef_clip=1.0, at 20/20)

EXP-38 swaps signed_ema for **ef_powersgd** at the same 20/20 cadence/delay. The mechanistic
difference: ef has **no sign term** and is **direction-preserving** —
`comp_t = M_anchor − P_{G_comp}(M_anchor)`, `e_t = decay·e_t + clip(comp_t)`, `G_corr = G_comp + e_t`
(`spectral_filter.py:446-507`). It folds the **codec residual** (the off-Q_τ part of M) against the
**same frozen `Q`** as signed_ema (the projection `P_{G_comp}` uses the live Q-projected `G_comp`).

**Prediction 4a (does infrequent Q hurt EF more or less than signed_ema?): EF is hurt LESS by the
*cadence/Q* axis but is NOT immune, and is hurt MORE by the *error-feedback accumulation* axis at
long holds.** Reasoning:

- **Less Q-sensitive on direction:** EF preserves `G_comp`'s direction and only *adds* the off-subspace
  residual. A staler `Q_τ` means a larger off-Q_τ residual `comp_t` (more energy outside the frozen
  basis), but EF is *designed* to re-inject exactly that — so a frozen `Q` partly **feeds** EF rather
  than starving it. EF has no `sign(M)` to get *wrong*, so it avoids the coherent wrong-sign hammering
  that is signed_ema's cadence-amplified failure (§3.2). On the *coverage/energy* axis EF should track
  better than signed_ema at 20/20.
- **More EF-accumulation risk at long holds:** `e_t` is a *decayed accumulator* (`ef_decay=0.9`) of
  the residual, refreshed against M only at fires (M is delay_K-stale, held between). With
  `ef_clip=1.0` (the un-damped dose) the residual can accumulate per-step for the **entire 10-step
  hold** before the next M refresh re-bases it. At cadence 5 (EXP-26 record, val 0.7210) EF re-bases
  every 2.5 steps; at cadence 20 the accumulator runs ~4× longer between re-basings against a frozen
  `Q`. The EXP-26/27 lineage already showed `ef_clip=1.0` **ignites a length explosion** (decay-0.9
  full-dose ignited ~step 29–42; even damped 0.5/0.5 ignited ~step 61). The 20/20 hold makes the
  accumulator-against-frozen-basis run longer between resets.

**Concrete EXP-38 predictions (falsifiable):**

- **P1 (most likely):** EXP-38 **ignites a length explosion and STOPs**, *earlier* and/or *harder*
  than EXP-37's step-90 onset — because `ef_clip=1.0` is the known ignition-prone dose AND the 20/20
  hold lets the residual accumulator run undamped for 10-step windows. Expect a cap-pin / len-mean
  spiral by **~steps 40–70** (between the parent EF's ~30–42 full-dose and EXP-37's ~90).
- **P2 (val, if it survives to a val point):** **val@25 ≥ EXP-37's 0.5921** and plausibly higher
  (EF's better coverage / no wrong-sign hammering), but **val degrades after ignition**, so val@50/75
  likely below EXP-37's 0.6482/0.4898 if it ignites earlier. If it does NOT ignite (less likely), EF
  at 20/20 should *beat* signed_ema at 20/20 on val (no sign-hammer), landing somewhere between the
  no-merger floor 0.6300 and the cadence-5 EF record 0.7210, but **below** the 5/5 results.
- **P3 (cadence signature):** the **per-refresh clipfrac spike at steps 10/20/.../100 will still be
  present** (it is a `Q`-discontinuity artifact independent of merger family — both project against the
  same broadcast `Q`). If EXP-38 shows the spike at the same steps, that confirms it is a pure-cadence
  (projector-jump) effect, not a merger effect. **This is the single cleanest cadence-isolation
  readout already available without a new run.**

If P1 is **false** (EF stable to 100 at 20/20 with no spiral), that would *falsify* the "long-hold
amplifies the merger" mechanism and shift weight toward "the wrong-sign term is signed_ema-specific
and EF's direction-preservation is robust to cadence" — still consistent with my thesis that the
*codec/Q* is not the dominant failure.

---

## 5. Practical recommendations (Q / cadence axis specifically)

Ranked by expected payoff per unit cost, with the async constraint front of mind
(**a single slow anchor serves a fast swarm; the anchor ALWAYS lags; corrections must be
cross-rank-identical and staleness-tolerant** — async-anchor memory).

### R-Q1 — DECOUPLE Q-refresh cadence from M-staleness delay_K. **(top recommendation)**
**Knob:** today `anchor.cadence` gates *both* `anchor_update_basis()` and the M/G_anchor refresh in
the *same* `anchor_should_fire` block (`transformer_impl.py:1474, 2081-2082`). Split the Q refresh
onto its own faster cadence: e.g. **`q_cadence = 5` (or even fire `Q` every tick) while `delay_K = 20`
and the M-refresh cadence stays 20.**
**Why it's cheap & feasible:** the Q update is just `orth(V)` from the activation sketch + an `H·r`
broadcast per boundary (`add_amortized_q_broadcast_bytes`, `powersgd_activation.py:343-359`). It does
**not** require a full stale-weight backward — the activation sketch `V` is harvested on the fast
path's own forward (the act basis). So **refreshing `Q` more often is nearly free** relative to the
anchor's stale fwd/bwd that produces M. This directly tests my §2 claim: if Q-refresh frequency barely
matters, R-Q1 will *not* recover much val ⇒ confirms delay_K is the culprit. If it recovers a lot ⇒ I
was wrong and cadence is the culprit. **Either way it is the decisive, low-cost knob.**
**Async note:** `Q` is already cross-rank-identical via `sync_basis` all-reduce + broadcast; refreshing
it more often keeps that property and is staleness-tolerant (a fresher `Q` is strictly better-aimed).
**Caveat:** this is a substrate change — `q_cadence ≠ anchor.cadence` is a NEW knob; flag it to the
operator as off the locked surface, justified as the decoupling experiment.

### R-Q2 — Run the decoupling 2×2 (the experiment that splits the confound). **(must-run)**
Two cells, same surface, EXP-37 merger (signed_ema 0.25/0.50), 100 steps:
- **Cell A: `cadence=5, delay_K=20`** — frequent `Q`, stale `M`. Isolates **delay_K** harm.
- **Cell B: `cadence=20, delay_K=5`** — rare `Q`, fresh `M`. Isolates **cadence** harm.
Plus the two anchors already run: **5/5 (EXP-36B, 0.7362)** and **20/20 (EXP-37, 0.6482)**.
**Decision rule:** if **Cell B ≈ 0.73 and Cell A ≈ 0.65**, cadence is benign and delay_K owns the
regression (my prediction). If **Cell A ≈ 0.73 and Cell B ≈ 0.65**, cadence owns it (hypothesis
confirmed). Mixed ⇒ interaction (most likely a partial split ~25/75 cadence/delay_K).
**Cost:** 2 cells × 100 steps on the accel surface (resp 2048) — cheap; one box back-to-back. *Today
`delay_K`-vs-`cadence` cannot be set independently if they share the gate, so Cell B specifically
needs the R-Q1 decoupling to even be expressible* — note this dependency.
**Feasibility check needed:** confirm whether `delay_K` and the M-refresh can fire on a different
cadence than the Q refresh without breaking the replay ring's fire-aware retention
(`AnchorReplayRing._keep_residue = (−delay_K) % cadence`, `anchor.py:426`) — the ring keys retention
on `cadence`, so a split cadence requires the ring to key on the **M-refresh** cadence, not the Q one.
This is the one real engineering subtlety; flag to systems.

### R-Q3 — Warm-start is already on; make the per-refresh jump smaller, not the cadence longer.
The damaging cadence artifact is the **discrete projector jump** at each fire (§3.1, the clipfrac
spikes). `warm_start=true` already means each `orth(V)` is one power step from the prior `Q`, so the
jump is bounded — but at cadence 20 ten steps of drift accumulate before that one step. **Cheaper than
shortening cadence: do a *partial* basis update** (a small-angle rotation toward `orth(V)` rather than
replacing `Q` outright), or **Grassmann-average** the old and new `Q` (`Q_new = orth((1−γ)Q_old + γ
orth(V))`, γ<1). This shrinks the per-refresh `ρ`-shock (smaller clipfrac spike) without paying for a
more frequent stale fwd/bwd. **Cost:** ~free (one extra `orth` of a blended `H×r`); cross-rank-identical
if γ is fixed. **Risk:** slows `Q`'s tracking of `U_t` — only worth it if the spikes (R-Q3-relevant)
turn out to carry real lost gradient signal (verify via P3 in §4).

### R-Q4 — Raise rank r (weak lever; do NOT lead with it).
A larger `r` lowers `E_off^floor` and *also* `E_drift` (more directions captured ⇒ slower relative
drift), so a frozen `Q` at higher `r` is staler-tolerant. **But** EXP-20 showed the recon error is
already flat across `r ∈ [77,102]` ⇒ the knee is *below* 77 ⇒ raising `r` buys little reconstruction
and costs comm linearly (`r/H` budget). **Recommendation:** do NOT raise `r` to fix cadence — the codec
is not coverage-starved. Only consider if R-Q2 surprisingly shows cadence dominates *and* a per-fire
recon-error log shows `E_drift` is large (instrument `‖g−P_Q g‖` per step — see R-Q6).

### R-Q5 — More frequent basis *sync* is already the default and is NOT the issue.
`sync_basis=true` all-reduces `V` across DP at every refresh (`powersgd_activation.py:766-772`), so the
basis is cross-rank-consistent at each fire. Syncing more often than refreshing is meaningless (nothing
changes between fires). **No action** — flagging only to close off "sync too rare" as a non-cause.

### R-Q6 — Instrument the per-step reconstruction error (zero-cost diagnostic, do this regardless).
Log `recon_t = ‖g_t − Q_τ Qᵀ_τ g_t‖ / ‖g_t‖` per step (boundary-mean). My model predicts a **sawtooth**:
rises within each hold window, snaps down at each refresh (steps 10,20,…). The *amplitude* of the
sawtooth is the direct measurement of `E_drift` and hence of the pure-cadence energy-loss. If the
sawtooth amplitude is small (≪ the ~2% floor), §1.3/§2 are confirmed and cadence is exonerated as a
*codec* problem. This is the cheapest possible test and disambiguates without a new GPU run if the
captures exist. (Note: diagnostics are OFF on production arms by policy — this needs a dedicated
diagnostic cell or a re-run with `spectral.diagnostics=true`.)

---

## 6. The confound — what I attribute to cadence vs delay_K

**The confound is structural and unavoidable in EXP-37:** `anchor.cadence` and `anchor.delay_K` are
gated by the *same* `state.anchor_step` counter through the *same* `anchor_should_fire` call
(`transformer_impl.py:1342-1343, 1474`), and the Q-update + M/G_anchor refresh + broadcast all happen
in the *same* fire block (`:2081-2082`). EXP-37 raised both to 20, so the run cannot, by construction,
separate "Q frozen 4× longer" from "M 4× staler / refreshed 4× less often."

**My attribution (with reasoning, to be settled by R-Q2):**

| Channel | Axis | Est. share of the −0.088 val@50 + spiral | Basis |
|---|---|---|---|
| Activation-subspace coverage loss (energy lost to frozen `Q`) | **cadence** | **~5–10%** | §1.3/§2: `U` is slow-varying, warm-start self-heals, recon flat in EXP-20 |
| Per-refresh projector-jump shock (clipfrac spikes, lost clipped signal) | **cadence** | **~10–20%** | §3.1: visible spikes at every fire; bigger because rarer |
| Stale `sign(M)` / stale correction held over a long window, coherent sharpening | **delay_K (×cadence hold-length)** | **~50–65%** | §3.2/§3.3: matches the locked "merger is the killer" finding; the hold *length* is cadence but the *content* staleness is delay_K |
| Cumulative entropy ratchet → late spiral | **interaction** | (the timing of the above) | §3.3 |

**Net:** I attribute **~20–35% to cadence-specific effects** (coverage + projector-jump) and
**~65–80% to delay_K / M-staleness**, with the **interaction** (cadence sets the hold length over
which the stale M is hammered) being the mechanism that converts a static staleness into a *dynamic*
spiral. **The cadence hypothesis is real but secondary; the codec/Q is robust to 10-step staleness;
the damage is the merger folding stale M, amplified by the longer cadence hold.**

**Design to separate them = R-Q2's 2×2** (requires R-Q1's decoupling to express Cell B). Predicted
result: **Cell A (cadence 5 / delay_K 20) ≈ 0.64–0.68** (near EXP-37 — delay_K carries it) and
**Cell B (cadence 20 / delay_K 5) ≈ 0.71–0.73** (near EXP-36B — cadence alone is mild). If the
opposite, my analysis is falsified and cadence is the lever.

---

## 7. Summary for the report-author

- **Hypothesis status: partially supported, NOT dominant.** Infrequent `Q` is a real cost but
  bounded and self-healing (warm-start + one power-step per fire + slow activation geometry). It does
  not explain the −0.088 / spiral on its own.
- **The one clean cadence fingerprint in the data:** a **clipfrac spike at every Q-refresh step
  (10,20,…,100)** in EXP-37 — the discrete projector-jump shock. Bigger because the jumps are rarer.
- **Mechanism of the late spiral:** cadence sets the **hold length** over which the *merger's* stale,
  coherent sharpening (wrong `sign(M)` for signed_ema) is applied uninterrupted. Longer holds ratchet
  entropy down harder per window (2.33→0.42 over 9 windows) until it crosses the length-hack ignition
  threshold ~step 90. Cadence is the *clock*, the merger is the *engine*.
- **EXP-38 (EF at 20/20) prediction:** likely **ignites earlier/harder** (ef_clip=1.0 is the known
  ignition dose; 10-step holds let the residual accumulator run undamped longer) — but EF is **less**
  Q/cadence-sensitive on the *direction* axis (no wrong-sign term), so if it survives it should beat
  signed_ema at 20/20. The **per-refresh clipfrac spike should reappear at the same steps** (pure
  cadence artifact) — the cleanest already-available confirmation.
- **Top action: R-Q1 (decouple `q_cadence` from `delay_K`) + R-Q2 (2×2 decoupling run).** Refreshing
  `Q` is nearly free (no stale backward), so testing "frequent Q + stale M" is cheap and is the single
  experiment that settles the confound. Predicted: it recovers **little** ⇒ confirms delay_K, not
  cadence, is the lever.
- **Do NOT** chase rank `r` (codec isn't coverage-starved, EXP-20) or more frequent sync (already
  per-fire). **Do** instrument the per-step recon sawtooth (R-Q6) — zero-cost, directly measures the
  pure-cadence energy loss.

*All file:line refs against the working tree at analysis time (vast-ai-workload, comm_eff/ mtime
2026-06-18). EXP-37 timeline extracted from `research/runs/EXP-37/train.log`; configs from
`EXP-37/resolved_params.txt` (cadence 20 / delay_K 20 / signed_ema 0.25,0.50 / r=77 / sync_basis=true)
and `EXP-38/handles/41475643.json` (ef_powersgd 0.9/1.0, 20/20, 100 steps).*
