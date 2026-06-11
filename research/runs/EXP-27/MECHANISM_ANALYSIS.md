# EXP-27 — Mechanism Analysis: why damped ef_powersgd still ignites the length-explosion

**Deliverable for team task #1 (exp27-postmortem / mechanist-math).**
Question: why does a direction-preserving merger (cos(G_comp,G_corr)≈0.956 parent band; EXP-27 dose
capped at rel_change 0.02–0.19 ⇒ per-step rotation ≤ ~11°) still drive the run into the
length-explosion / entropy-collapse attractor at step ~61–66, while every per-step health metric
looked fine — and why did halving the dose (clip 1.0→0.5, decay 0.9→0.5) only *delay* ignition
(~29–42 → ~61) with zero val gain?

**One-line answer:** the implemented "error feedback" is not error feedback — it is a persistent,
*exactly-orthogonal* (tangential) forcing term whose direction is set by a ~50-global-step-memory
EMA of stale gradients; a persistent tangential force transports the policy along reward-**flat**
directions (on GSM8K under this surface: "correct but longer"), where the fresh gradient provides
**zero restoring force**; dose magnitude therefore controls only the *time* to reach the
length-hack basin, never *whether* it is reached. Per-step direction preservation certifies the
descent component and says nothing about the tangential one — i.e. nothing about closed-loop
stability.

Provenance: all EXP-27 numbers re-extracted from
`runs/EXP-27/train_exp27_B_ef_damped.log` (steps 1–68) + `runs/EXP-27/verdict.md`; parent/reference
numbers from `runs/EXP-26/verdict.md`, `runs/EXP-26/stepA_decision.md`,
`runs/EXP-25/{COLLAPSE_GRADIENT_FLOW_ANALYSIS,DEEP_FINDINGS,ENTROPY_COLLAPSE_FINDINGS}.md`. Code
cites are `file:line` on `vast-ai-workload`.

---

## 0. The implemented update, written out exactly

Per targeted 2D matrix, per optimizer tick *t* (`verl/workers/comm_eff/spectral_filter.py:327-410`):

```
comp_t = M_t − (⟨G_t, M_t⟩ / ‖G_t‖²) · G_t          # off-G component of the STALE anchor EMA
e_t    = δ · e_{t−1} + comp_t                        # δ = ef_decay (0.5 here, 0.9 parent)
e_t   ← e_t · min(1, c·‖G_t‖/‖e_t‖)                  # c = ef_clip (0.5 here, 1.0 parent)
G_corr = G_t + e_t
```

where `G_t = G_comp` is the fast PowerSGD-compressed gradient (fresh, every tick) and `M_t` is the
anchor EMA: `M ← β·M + (1−β)·G_anchor(θ_{t−K})` with β=0.95, updated **only at anchor refreshes**
(`spectral_filter.py:198-219`, fed by the anchor circuit at cadence=5 ticks, delay_K=5 ticks;
2 ticks per global step ⇒ refresh every 2.5 global steps, snapshot staleness 2.5 global steps).

Three structural facts that drive everything below:

**(F1) comp_t ⊥ G_t exactly.** `⟨comp_t, G_t⟩ = ⟨M,G⟩ − ⟨G,M⟩ = 0` by construction
(`spectral_filter.py:378-379`). The freshly injected force never aids and never opposes the
current descent direction. (The carried part `δ·e_{t−1}` was orthogonal to `G_{t−1}`, not `G_t`,
so e_t is only ~tangential — but gradients decorrelate across rollout batches, so the carried
along-G component is small and of random sign.) Quantitative cross-check: a purely tangential
residual of relative size λ gives `cos(G, G+e) = 1/√(1+λ²)`. EXP-26 measured cos median 0.9558 ⇔
λ = tan(17.1°) = 0.308 — and the independently logged dose median was **0.3215**. The two
measurements agree to 4%: the residual is, as the math says, almost purely orthogonal.

**(F2) M is a ~50-global-step-memory direction source.** β=0.95 per refresh ⇒ effective memory
1/(1−β) = 20 refreshes × 2.5 global steps = **50 global steps**; mean age of M's directional
content ≈ 2.5·β/(1−β) + 2.5 ≈ **50 global steps** (confirmed cadence bookkeeping:
`anchor_q_updates=14 @ step 37` in EXP-25 ⇒ 74 ticks/5 ≈ 14.8 ✓). At EXP-27's ignition (step 61),
M is a weighted average of clean-PG gradients of policy versions from roughly steps 10–60. The
forcing *direction* is therefore persistent on the timescale of the entire run.

**(F3) the projection is nearly vacuous, so comp_t ≈ M itself.** Step-A (EXP-26) measured
`cos(G_fresh_anchor, G_comp) ≈ +0.01…+0.06` on codec arms (loss/operand mismatch:
clean-PG weight-grad vs PPO-clip compressed grad — `runs/EXP-26/stepA_decision.md`). M is an EMA
of exactly those anchor grads ⇒ ⟨G,M⟩ ≈ 0 ⇒ the subtracted projection removes almost nothing and
**comp_t ≈ M**. The implemented merger is therefore, in effect, *“add a decayed, norm-clipped copy
of the stale anchor EMA”* — the same object as the old `inject` combiner at small γ (the
`inject_matrix` docstring even states this limit: under orthogonality the injection is
scale-matched direct injection of M_anchor, `spectral_filter.py:224-233`). The "error-feedback"
name is nominal.

### Dose bookkeeping (why the observed doses are what they are)

‖M‖ is small relative to ‖G_comp‖ (M is an EMA of dense-scale clean-PG grads ~0.39 with further
cancellation across stale gradients; G_comp runs ~1.6–3 — EXP-25 §2.4), call the ratio
ρ = ‖comp‖/‖G‖ ≈ 0.05–0.10. Steady-state unclipped residual is geometric:
‖e‖ ≈ ρ/(1−δ)·‖G‖.

| arm | δ | c (clip) | predicted ‖e‖/‖G‖ | observed dose |
|---|---|---|---|---|
| parent ef (EXP-26) | 0.9 | 1.0 | ρ·10 ≈ 0.5–1.0 → clip-shaped | 0.30→0.47 climb, median 0.32 |
| EXP-27 damped | 0.5 | 0.5 | ρ·2 ≈ 0.10–0.20 → **clip never binds** | 0.02–0.19, peak 0.189 |

Two consequences: (i) in EXP-27 the clip was **inert** (dose never reached 0.5·‖G‖) — the decay
alone set the dose, so a further clip cut (0.5→0.25, the verdict's flagged optional probe) would
change almost nothing; (ii) at δ=0.5 the residual *saturates within ~2 ticks*
(1+δ+δ² ≈ 1.75 of 2.0), which kills the "reset e_t on refresh" mitigation in advance (§7, P3).

**The decisive log fact:** the per-step dose **decayed** over the run (per-sweep means ~0.19–0.30
around steps 18–30 → 0.04–0.13 by steps 48–68; re-extracted from the `[EXP-7][spectral]`
rel_change lines in the train log — ‖G‖ grew late while ‖M‖ did not. Caveat: Ray dedupes repeated
worker lines, so per-step samples are partial, n≈14–28 of 392; the W&B `spectral/rel_change_mean`
series shows the same shape, peak 0.189@s12 per verdict.md). Through the ignition window
(steps 61–66) the forcing sat in the **lowest band of the entire run** (0.05–0.13). Ignition is
not a dose event. This single observation already falsifies every "the residual kicked too hard"
story and demands a mechanism where smallness of the per-step push is irrelevant.

---

## (a) This is NOT classic error feedback — it is a lag operator on old gradient directions

Classic EF (Seide'14; Stich'18; Karimireddy'19 EF-signSGD) co-evolves the memory with the codec:

```
send_t = C(g_t + e_{t−1});   e_t = (g_t + e_{t−1}) − send_t
```

Summing telescopes: `Σ_{t≤T} send_t = Σ_{t≤T} g_t + e_0 − e_T`. The cumulative *applied* update
equals the cumulative *true* gradient up to one bounded residual. That telescoping identity **is**
the stability mechanism: the memory holds exactly what has not yet been applied, so it can only
*compensate*, never *force*. Under a δ-contractive compressor, ‖e_t‖ stays bounded and the
iterates track full-gradient SGD with bounded lag — a closed-loop, self-cancelling correction.

The implemented update has **no telescoping identity**. e_t is fed by an *exogenous* signal
(M: a different network pass, at different weights θ_{t−K}, under a different loss — clean-PG
ratio≡1 vs the fast path's PPO-clip), filtered through two EMAs. Summing gives

```
Σ G_corr = Σ G_t + Σ e_t ,   Σ_{t≤T} e_t ≈ Σ_t Σ_{j≥0} δ^j comp_{t−j} = (1/(1−δ))·Σ_t comp_t
```

— an **unbounded, O(T) integral of stale off-subspace directions** that is compensated against
nothing. In signal-processing terms the chain

```
θ → [sample every 5 ticks, delay 5 ticks] → [EMA β=0.95/refresh : ~100-tick memory]
  → [orthogonalize against fresh G] → [EMA δ/tick + clip] → +G → Adam → θ
```

is a *positive feedback path of old state into the input with a ~100-tick group delay*. Classic EF
is a feedback loop around the **codec**; ours is a feedback loop around the **policy's own
history**. The first is contractive; the second is a forcing term.

One more disanalogy that matters for the fix (§7): for a *projector* codec with self-adjoint
P = QQᵀ, classic EF applied at the boundary is only nontrivial because Q **rotates** (refresh every
5 ticks) — with a frozen P, `e ∈ range(I−P)` forever and `C(g+e) = C(g)`; with a rotating Q each
refresh flushes the accumulated null-space residual into the new subspace. So a correct EF on this
substrate is mathematically a *compressed-bytes clean-step at refresh cadence* — bounded lag,
no exogenous carrier. That is the principled successor primitive, and it is not what was built.

---

## (b) Why cos(G_comp, G_corr) ≈ 0.956 is not a stability certificate

Let the update be `u_t = −η(G_t + e_t)` with `⟨e_t, G_t⟩ ≈ 0`, `‖e_t‖ = λ_t‖G_t‖`,
λ_t ∈ [0.02, 0.19]. Then for every t:

- `⟨u_t, −G_t⟩ = η‖G_t‖² > 0` — every step is a strict descent step on the instantaneous
  surrogate; **any** per-step angle/descent test passes at all t, by construction;
- `cos(G_t, G_t+e_t) = 1/√(1+λ_t²) ≥ 0.982` — the per-step rotation is tiny, by construction.

Stability, however, is a property of the closed loop `θ_{t+1} = θ_t − η(G(θ_t) + e_t(θ_{t−τ},…))`,
i.e. of how perturbations **integrate**, not of any single-step angle. The integrated displacement
from the forcing after T steps is `η·Σ e_t`. Two regimes:

- **Decorrelated forcing** (fresh zero-mean noise, correlation time ~1 step): ‖Σ e_t‖ ~ √T·λ‖G‖ —
  diffusive, slow, and Adam's averaging suppresses it. This is what ordinary sampling noise does;
  dense GRPO lives with it.
- **Persistent forcing** (our case: direction correlation time ≈ M's memory ≈ 50 global steps ≈
  100 ticks ≫ run length): ‖Σ e_t‖ ~ T·λ‖G‖ — **linear, coherent integration**. At λ ≈ 0.1 the
  forcing integrates to the size of ~6–7 full gradient steps over the 65-step healthy phase, all
  pointed in a slowly-varying direction the fresh gradient does not choose.

The cosine measures λ. It does not measure the correlation time τ. The loop-relevant quantity is
the product **λ·τ** (dose × persistence), and damping halved λ while leaving τ untouched (τ is
M's, set by β=0.95/cadence — unchanged 26→27). Halving λ at fixed τ ⇒ ~2× longer to integrate the
same displacement ⇒ ignition ~29–42 → ~61–66. **Observed: delay ×~1.7–2, prevention: none, val
gain: zero.** Exactly what the λτ-model predicts and exactly what a per-step certificate cannot
see.

The cross-arm data even makes the per-step angle point the **wrong way**: signed_ema α=0.5
rotates its update by ~44° *every step* (cos 0.7165) and never ignites in 50 steps; ef rotates
~17° (EXP-27: ≤11°) and ignites. Per-step rotation angle anti-correlates with ignition across
arms, because what matters is not how far one step turns but whether the turning has a persistent
exogenous component that integrates (§e).

The Lyapunov-flavored version: a certificate needs a function V with V̇ < 0 along trajectories.
Any reward-derived V satisfies V̇ ≈ 0 along reward-*flat* directions — and the forcing is, by F1,
**concentrated** in directions where the reward gradient does not act (it is orthogonal to G; its
reward-coupled tangential components are continually pulled back by the next fresh gradient, its
reward-flat component is not — see (c)/(d)). No certificate over this surface exists without
adding a potential on the flat directions (KL / entropy / length terms — precisely the LABELED
guardrails, §7). Score staying 0.73–0.84 *through* ignition is this statement made empirical:
the policy moved a long way along a reward level set.

---

## (c) Where dense RL's self-correction lives, and where our loop opens

Dense on-policy PG is a **closed loop with unit delay**: θ_t → fresh rollouts from π_{θ_t} →
g_t = ∇J(θ_t) → θ_{t+1}. Every excursion the policy makes is *seen by the very next gradient*,
which supplies a restoring force along every reward-**coupled** direction. The only uncontrolled
subspace is the reward-flat one, and dense's update has no systematic component there (its
flat-direction content is zero-mean sampling noise ⇒ √T diffusion ⇒ slow), and dense closes its
own exposure window quickly: it sharpens fast (entropy 0.12–0.13 by s50) so the long-tail seed
faucet (§d) shuts before anything can catch. Dense at entropy 0.122 with bounded length 193 and
val 0.7536 (5e2jpho9) is the proof that *low entropy per se is benign* (EXP-25 finding 4).

Our substrate splits this into **two loops**:

1. **Fast loop (healthy):** G_comp every tick — fresh, direction-faithful (PowerSGD r77 recon
   ~0.025, psgd-only ties dense 0.7415). This loop self-corrects exactly like dense.
2. **Slow loop (the defect):** θ → [5-tick-stale snapshot] → [β=0.95 refresh-EMA] → comp ⊥ G →
   [δ-EMA + clip] → **added to the input** of the fast loop. This is a second feedback path with
   ~100-tick group delay and ~100-tick memory. The fast loop can and does compensate the slow
   loop's reward-coupled content (that is why per-step metrics look healthy); it **cannot**
   compensate the reward-flat content, because along flat directions it produces no signal at all.
   The slow loop therefore *owns* the flat subspace.

This is the precise sense in which "the feedback loop opens": it is not that correction is absent
— it is that a delayed positive-feedback channel is **added** whose only un-cancelled output lands
in the subspace where the primary loop is blind. A control engineer would say: the plant has an
uncontrollable mode (length, under this reward), and we wired a lagged disturbance generator
straight into it.

---

## (d) The GSM8K length-hack attractor, quantitatively

### d.1 The flat direction

Under the locked surface (no KL, no entropy bonus, lenient answer-match reward, 16384 cap), the
reward is **length-agnostic on correct answers**: "correct and 170 tokens" and "correct and 2000
tokens" score identically. So the reward landscape at score ≈ 0.75–0.84 has an (effectively)
unbounded flat direction: *make correct answers longer*. Motion along it draws no restoring
gradient until the 16K-cap truncation cliff (truncated ⇒ unparseable ⇒ reward 0), which is the
*far end* of the flat valley. EXP-27 was killed mid-valley: mean 575, clip% only 0.021, score
still 0.80 — the reward-preserving phase. (EXP-25 α=0 shows the far end: mean ~5–8.6k, clip 0.46,
score crashed to 0.32.)

### d.2 The token-mean rectifier (why the loss itself amplifies the tail)

`loss_agg_mode=token-mean` (verl default; confirmed in actor.yaml, not overridden in
resolved_params): the batch loss is `Σ_i Σ_{t≤T_i} ℓ_{i,t} / Σ_i T_i`, each token of sample i
carrying the same group-normalized advantage A_i (GRPO), ratio≈1. So **sample i's weight in the
update is ∝ T_i·A_i**. The batch is 128 prompts × n=8 = 1024 responses; at mean length ~175 the
token pool is ~1.8e5. One 16384-token rollout is **9.1% of the entire batch's gradient mass**
(fair share 0.098% ⇒ ~93× amplification); the ~6 capped rollouts at s61 (clip 0.006) carried
~25–30% of the batch's token mass between them.

GRPO's group normalization adds the gate: A_i = (r_i − μ_g)/(σ_g+ε). In an all-correct or
all-wrong group, A ≡ 0 — the group emits no learning signal. As the policy sharpens, within-group
σ collapses for the "easy" groups and the surviving advantage mass concentrates in *mixed* groups
— exactly the groups where a deviant rollout lives. A **long-and-correct** deviant in a mixed
group gets the same positive A as a short-and-correct one but T_i/T̄ ≈ 10–90× the gradient mass:
token-mean turns a length-neutral reward into a **length-favoring update** whenever
positive-advantage mass exists at long length. Pushing up the probability of every continuation
token in a long correct rollout is, position-by-position, pushing **down** EOS; lower EOS ⇒
longer samples next batch ⇒ more token mass at long length ⇒ the ratchet. (Token-mean's length
bias is the documented Dr.GRPO/DAPO critique; here it is the engine of the basin's
self-reinforcement, not the initiator.)

The entropy crash at the catch (0.34 → 0.079 in 5 steps) is the symptom, partly compositional:
long repetitive continuations are near-deterministic per token, and `actor/entropy` is token-mean
too, so thousands of tail tokens drag the average down — while reinforcing themselves through the
same token-mean loss. Entropy collapse here is the *signature of entering the basin*, not the
cause (dense reaches 0.122 without any of this).

### d.3 Three-stage anatomy of the EXP-27 ignition (all from the train log)

1. **Susceptibility (≈ s44–55).** Healthy sharpening brings entropy through ~0.48→0.33; group σ
   shrinks; the policy is now "sticky" (an update on a sharp policy persists) and advantage mass
   concentrates in mixed groups. This stage is driven by *learning itself* — it is
   force-independent and unavoidable on this surface.
2. **Seeding (s45, s53, s59).** Isolated `response_length/max = 16384` rollouts appear
   (clip% ≈ 0.001 — single capped samples), with growing instability markers: length-mean creep
   165 → 185–201 over s52–60, grad_norm spikes 21.6 (s57) / 43.7 (s59), erratic ppo_kl from s53.
   Seeds at high entropy don't catch (s2 had a 16384 fluke at entropy 5.8 — no effect: diffuse
   policy, large group σ, diluted token share).
3. **Catch + ratchet (s61–68).** A seed lands in the susceptible regime; gnorm 58.9 (s61); len
   mean 171→268→295→395→448→558→566→575; lmax pinned at 16384 for 8 consecutive steps; entropy
   0.34→0.25→0.22→0.079; IS gap 0.62→0.40; **score 0.73–0.84 throughout** — reward-preserving,
   mid-valley. Killed s68 (mem 123/143GB).

The merger enters this anatomy **twice**:

- **Transport (the main role):** the persistent tangential force (F1+F2) moves the policy along
  the flat valley *during the healthy phase*, so by the time susceptibility arrives the policy
  already sits near the long-mode region and emits long-tail seeds. Compare matched step ~50:
  EXP-27 lmax hits 16384 (s45/53/59) vs α=0.5's **max length 288 over the whole batch** at s50
  and dense's bounded ~600 — the ef policy's tail is qualitatively different *before* ignition.
- **Window-widening (the secondary role):** merger arms sharpen more slowly (entropy at s50:
  ef ≈ 0.40, α=0.5 0.371 vs dense 0.122) — they sit in the seed-emitting-but-susceptible window
  ~3× longer than dense, multiplying seed exposure.

Catching is a stochastic first-passage event (a long+correct deviant must land in a mixed group
while σ is low). That is the r1-vs-r2 stochasticity (§e.4): same forcing, different seed luck.

---

## (e) Reconciling every observed run with one model

Write every merger as `G_corr = G + F` and classify F by **(i) exogenous-carrier content** (does F
contain a direction *not* derived from the current G?) and **(ii) reward-coupling** (does F have a
systematic anti-descent component?):

| arm | F | carrier? | anti-descent? | predicted | observed |
|---|---|---|---|---|---|
| dense / psgd-only | 0 | no | no | stable | val 0.7536 / 0.7415, no ignition |
| plain (B_plain) | 0 (substrate: stale-Q forward only) | no (grad path) | no | stable, but refresh drag | 0.6437, **no ignition** in 50 |
| signed_ema α=0.5 | −G⊙1[sign(M)≠sign(G)] — a negated *subvector of G* (mask) | **no** (contraction of current G; cos = √½ ✓ measured 0.7165) | no | stable; val cost from discarding ~half the gradient | 0.7066, no ignition in 50, lmax 288 |
| signed_ema α=0.3 | (2α−1)·G on disagree = −0.4·G_disagree + carrier | yes (sign(M)) | **yes (−0.4)** | ignite, slower than α=0, reward crash | ignited s33, val 0.6164 |
| signed_ema α=0 | −1.0·G_disagree + carrier | yes | **yes (−1.0)** | fast ignite + reward crash | ignited s30, entropy→0.06, len→8.6k, val 0.354 |
| ef parent (c=1, δ=0.9) | clip(filter(M_⊥)) ≈ 0.32·‖G‖·M̂ | **yes (M)** | no (⊥) | ignite, reward-preserving, timing stochastic | r1 ignited 29–42; r2 clean to 50 (first-passage) |
| **ef damped (EXP-27)** | same, λ ≈ 0.02–0.19 | **yes (M)** | no (⊥) | ignite **later** (λ halved, τ unchanged), reward-preserving | ignited 61–66, score 0.73–0.84 ✓ |

The empirical law across all eight runs is exact: **ignition ⇔ F contains a persistent exogenous
carrier** (sign(M) or M itself). Dose sets the timing; reward-coupling of F sets whether reward
crashes during ignition (α=0/0.3: anti-descent ⇒ crash) or is preserved (ef: orthogonal ⇒
score 0.73–0.84 — the EXP-27 signature, predicted by F1).

### e.1 Why α=0 exploded fast
F has an O(‖G‖) anti-descent component on ~50% of gradient mass (the √2 fingerprint) — it
destroys the sign-cancellation regularizer (grad-norm 0.39→3.3–11), actively sharpens, *and*
carries the M-direction. It drives its own susceptibility (entropy 5.7→0.47 by s30) and ignites
the moment it gets there. Fast, deterministic, reward-crashing.

### e.2 Why α=0.5 survived 50 steps
At the (2α−1)=0 knee, F = −G⊙1[disagree] is a *masked current gradient*: no exogenous direction
at all (the stale M only selects *which* coordinates pass). A subvector of G has ~zero component
along reward-flat directions (flat ⊥ G by definition of flat), so the transport mechanism is
**absent**, not merely damped. Cost: ~half the gradient discarded ⇒ val 0.7066 < psgd 0.7415.
Residual risks are second-order (anisotropic progress from persistent stale coordinate-selection;
the widened window §d.3). Its s50 state corroborates: lengths 165–170 *shrinking*, batch lmax
288 — the seed faucet is closed.

### e.3 Why EXP-27 ignited at ~61 with the dose at its minimum
λτ-model: λ halved (and the late-run dose actually *decayed* to 0.04–0.13 as ‖G‖ grew), τ (M's
memory) unchanged ⇒ transport rate halved ⇒ time-to-basin ~×2 (29–42 → 61–66 ✓). The
susceptibility stage is force-independent (healthy sharpening), so damping cannot push ignition
past it — only stretch the seeding lag. And because the run was *extended* to 100 steps to chase
parity, the slower transport still had time to complete. Damping the dose is a time-dilation of
the same trajectory, which is why it also bought zero val: the healthy-phase information content
is the same.

### e.4 r1 vs r2 stochasticity
Time-to-catch = susceptibility (deterministic-ish, ~s25–30 at clip 1.0 given the EMA-inflated
early dose) + first-passage of a seed catching (Poisson-ish: a long+correct deviant in a mixed
low-σ group). Near the critical regime the catch is fluctuation-dominated ⇒ realization-dependent:
r1 caught at 29–42; r2's seeds missed within the 50-step window. EXP-27 (single realization,
lower λ) caught at 61. Nothing about r2 was mechanistically safer — it was censored by the
50-step horizon. (Comparator check, §8: whether r2 shows late lmax spikes / length creep at
s40–50 would confirm it sat in the same pre-ignition state.)

---

## (f) B_plain's no-ignition: merger-as-carrier vs substrate-generic-past-60

Two hypotheses for EXP-27's late ignition:

- **H_carrier (mine):** ignition requires the persistent exogenous carrier; plain/α=0.5/dense are
  safe at *any* horizon on this surface (up to the slow √T diffusion floor).
- **H_generic:** any run on this surface (no KL/entropy, 16K cap, token-mean) ignites once past
  ~step 55–65; EXP-27 just ran long enough to see it, and the 50-step controls are all censored.

Evidence already against H_generic: ef r1 ignited at 29–42 — *inside* the window where plain,
α=0.5, psgd-only, and dense were all clean. So the merger demonstrably ignites where the substrate
does not. But EXP-27's own ignition at 61–66 is past every control's horizon, so H_generic cannot
be excluded *for the late regime* by existing data — every no-carrier run is right-censored at 50.
(Weak prior evidence: the EXP-17 core diagnostic ran ~2 epochs ≈ 110+ steps on the mask+clean@20
substrate — same GRPO/no-KL/16K surface, different codec — with no recorded length explosion.)

**Falsifiable discriminating predictions (pre-registered):**

- **P1 — signed_ema α=0.5 @100** (same surface, 1wulaelw config, step_target 100):
  NO ignition by s100. Specifically: batch lmax < ~2k at every step (no pinned-16384 step), no
  length-mean creep ≥ +25% over its s40–50 running min, entropy lands 0.25–0.35, val@100 ≈
  0.70–0.72 (the masked-gradient cost persists; it does not close parity).
  **P(no ignition) ≈ 0.75.** (Residual 0.25: window-widening + anisotropic-selection second-order
  effects + generic unknowns.)
- **P2 — plain @100** (B_plain config): NO ignition by s100; val@100 0.65–0.70 (climbing from
  0.6437 but the refresh-alone drag keeps it at/below the floor band). **P(no ignition) ≈ 0.85.**
- **P3 — ef-with-residual-reset** (damped settings + e_t←0 at every anchor refresh): **still
  ignites**, ~s55–75. At δ=0.5, e saturates in ~2 ticks (1+δ+δ²+δ³ = 1.94/2.0) while resets come
  every 5 ticks — the average dose drops only ~20%, and the carrier persistence lives in **M**,
  which the reset does not touch. **P(still ignites by s80) ≈ 0.7.** If this arm does NOT ignite,
  my model is wrong about where the persistence lives (it would be in e, not M) — that is the
  cleanest single-run falsifier of this document.
- **P4 — outcome table:** if α=0.5@100 and plain@100 both stay clean while any ef@100 rerun
  ignites, H_carrier is confirmed and H_generic dead. If α=0.5 or plain ignite at ~s55–70 with the
  same precursor sequence (lmax spikes → length creep → catch), H_generic wins and the merger is
  only an accelerant — in that case the *surface* (token-mean + no-brake + cap) must be fixed
  before any merger work continues.
- **P5 — precursor universality (retroactively checkable on r1):** every ignition on this surface
  is preceded by ≥2 isolated lmax=16384 events and a length-mean creep ≥15% above its running min
  for ≥5 steps before the T4 (2×) trigger fires. If an ignition ever occurs *without* these, the
  seeding stage of §d.3 is wrong.

---

## (g) Mitigations, ranked, with the math

Locked control surface honored: items that touch the objective are LABELED guardrail arms, never
the core.

1. **Rebuild EF as *true* error feedback on the codec's own dropped residual (core-eligible,
   the principled successor).** At the boundary, the upstream stage computes the full
   activation-grad dL/dÂ locally; transmit C(dL/dÂ + e) and keep e ← (dL/dÂ + e) − C(·) on the
   *sender* (zero extra comm — the memory never crosses the boundary). Telescoping (§a) ⇒
   cumulative applied = cumulative true ± bounded residual ⇒ **no exogenous carrier, no O(T)
   integral** — the entire §b mechanism is structurally excluded. Because P=QQᵀ rotates at refresh
   cadence, EF here = a compressed-bytes clean-step every ≤5 ticks (bounded-lag full information).
   Retires M from the fast path entirely (anchor keeps Q ownership only). Predicted: no ignition;
   val in the 0.72–0.74 band (parity question stays open — EF recovers the ~0.06% off-subspace
   energy, EXP-25 §8.1's parity ceiling applies). P(no ignition) ≈ 0.85; P(val ≥ 0.7414) ≈ 0.3.
   This is also the variant the task names "(G_full_anchor − P_Q·G_full_anchor)" done *right*:
   built from the CURRENT step's own dropped component, not from a stale different-loss EMA.
2. **Retire the M-fed merger class (strategic).** The cross-arm table (§e) is a completed
   dose-response over the class {sign-carrier, M-carrier, contraction, none}: every carrier arm
   ignites, every non-carrier arm is clean, and the best carrier arm (0.7210) < psgd-only's own
   0.7415. There is no interior optimum to tune toward — same shape of conclusion as EXP-25's
   α-monotonicity, one level up. Consistent with the standing conversion-spine memo: the
   productive axis is elsewhere (training/eval diversity), not merger-dose tuning.
3. **LABELED guardrail arms (objective changes — never the core):**
   - **KL to reference (proven):** the EXP-25 KL probe closed the length channel outright
     (len 294→120, clip 0.000 for 50 steps, no crash) — it adds the missing potential on the flat
     direction (V̇ < 0 along length). Cost: sub-parity val (0.6793) at coef 0.001 — a brake, not a
     fix for the carrier bias.
   - **Length-normalized aggregation** (`loss_agg_mode=seq-mean-token-mean` or
     advantage/length normalization): sample weight becomes ∝ A_i instead of T_i·A_i ⇒ the ~93×
     tail amplification (§d.2) collapses to 1× ⇒ the ratchet's loop gain < 1 ⇒ a caught seed
     cannot self-reinforce. The cheapest *single-knob* change that breaks the basin's interior
     dynamics (it leaves the flat direction flat but removes the rectifier). Run only as a
     labeled arm: it changes the estimator the entire program is normalized on.
   - **Entropy floor (entropy_coeff > 0):** delays susceptibility (holds σ_g up). Weakest of the
     three — it stretches stage 1 but leaves transport + rectifier intact.
4. **Early-stop / checkpoint-at-peak science (measurement, not a fix).** The precursors give
   ~8–15 steps of warning (lmax-pinned events, length-mean creep, gnorm spikes, erratic ppo_kl) —
   formalize them as a standing tripwire in ENTROPY_COLLAPSE_WATCH (a "T8 seed watch":
   ≥2 isolated lmax=cap events within 15 steps + len-mean creep ⇒ snapshot + alert). Banks
   nothing for parity (val@50 0.7202 < 0.7414 regardless) but protects long-horizon runs.
5. **Residual reset on refresh (cheap falsifier only).** Run it to test P3 — it is the
   discriminating experiment for *where the persistence lives* — but the math (§0, e-saturation
   in ~2 ticks vs 5-tick resets; carrier in M) says it will not prevent ignition. Do not sell it
   as a fix.
6. **Rank anneal (NO).** comp_t is built from M, not from the codec error; rank enters only
   through G_comp's direction. Raising r late buys recon fidelity the run already has (0.975+
   energy) and does not touch the carrier. Decline.

---

## (h) THE CORE IDEA

The implemented ef_powersgd is not error feedback; it is a **persistent tangential forcing loop**.
By construction (comp_t ⊥ G_comp, with the projection nearly vacuous because M ⊥ G_comp anyway),
every step injects a small force that neither helps nor hurts the current descent direction — it
pushes *sideways*, in a direction set by an EMA whose memory (~50 global steps) makes it
effectively constant over the run. On GSM8K under the locked surface (length-agnostic reward,
no KL/entropy, 16K cap, token-mean GRPO), the policy's reward landscape has exactly one unbounded
flat direction — *make correct answers longer* — and motion along it draws zero restoring force
from the fresh gradient, while the token-mean loss stands ready to amplify any long-and-correct
deviant by ~10–90× its fair gradient share. Persistent sideways forcing on a manifold with one
unopposed flat direction is rectified into net transport along that direction (every other
component gets pulled back; the flat one integrates linearly, ~λ·T). So the run looks healthy by
every per-step metric — score is *blind* to level-set motion, the cosine *certifies only* the
descent component, entropy tracks the healthy profile — right up until the transported policy
enters the susceptible regime (sharp, low group-variance) while still emitting long-tail seeds,
one seed catches, and the token-mean ratchet finishes the job reward-preservingly
(score 0.73–0.84 through ignition). Dose magnitude divides the transport rate and nothing else:
halving it doubled time-to-ignition (29–42 → 61–66) at zero val gain, and the run ignited with
the forcing at its run-minimum. The only real exits are to remove the carrier (true EF on the
codec's own per-step dropped residual — telescoping, no O(T) integral), or to tilt the flat
direction with an explicitly-labeled potential (KL / length-normalized aggregation). Tuning λ
(clip/decay/reset) only buys time inside the same trajectory — falsified twice, now with the
mechanism that says why.

---

## 8. Cross-check ledger (comparator-runs)

Sent to comparator-runs for verification against W&B (their task #2 curves): (A) batch
lmax curves s35–50 for 1wulaelw / tilwe80t / 5e2jpho9 / u1v94opv — prediction: no 16384 spikes in
the non-carrier arms; r2 *may* show them (would confirm censored-not-safe, §e.4); (B) entropy@s50
ordering (ef ≈ α0.5 ≈ 0.37–0.40 ≫ dense 0.12); (C) r1's precursor sequence before its 29–42
ignition (P5 retro-check); (D) any contradiction with the tangential-forcing / dose-sets-lag-only
/ α=0.5-has-no-carrier claims. This section to be updated when they reply; their findings at the
time of writing (3-run comparison, task #2) did not contradict (i)–(iii).
