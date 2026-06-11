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

**Two operator-confirmed findings (2026-06-11) this analysis is built around:**

1. **Entropy is a FOLLOWER of the length spiral, not its trigger.** Dense (5e2jpho9) is the
   *lowest*-entropy run of the cohort (0.12–0.16 from step 36) and the *most* stable; ef r1
   ignited at entropy **0.83** (high) and only then collapsed to 0.13; EXP-27 sharpened to ~0.34
   before its catch. Entropy-causes-explosion causality is falsified three independent ways. In
   the mechanism below, entropy decline is the policy-sharpening component of the *healthy*
   dynamics plus, at ignition, a compositional artifact of the long repetitive tail (§d.2/§d.3).
   The standing watch is re-centered accordingly
   (`research/diagnostics/ENTROPY_COLLAPSE_WATCH.md` §2026-06-11: kill triggers = consecutive
   cap-pins / len-mean trailing slope / mean>2×, plus the E1 early gate; entropy demoted to
   corroborator). Throughout this document, "watch P1/P2/P3/E1" refers to those triggers;
   my pre-registered *predictions* are renamed PRED-1…6 to avoid collision.
2. **The instability is NOT EF-specific — it is M_anchor-carrier-generic across both merger
   families.** signed_ema α=0.5 + anchor, the EXP-25 "survivor", was on the same spiral at its
   censored 50-step endpoint (consecutive 16384 cap-pins s47–48, len-mean slope +5.92/step;
   P(ignite by 100) ≈ 55–70%, comparator). The mechanism predicts this naturally: nothing in
   §a–§d uses any EF-specific property — only that the merged update has a nonzero *expected*
   exogenous component along M's persistent direction, which sign-replacement (α<0.5), the
   α=0.5 veto-rectifier (in expectation), and EF-injection all share in different functional
   forms (§e). What discriminates family-specific vs carrier-generic vs substrate-generic is
   laid out, with predictions, in §f.

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

| arm | δ | c (clip) | predicted ‖e‖/‖G‖ | observed dose (W&B `rel_change_mean`) |
|---|---|---|---|---|
| ef r1 (EXP-26) | 0.9 | 1.0 | ρ·10 ≈ 0.5–1.0 → clip-shaped | median 0.200 over s18–27 |
| ef r2 (EXP-26) | 0.9 | 1.0 | same | median 0.250 over s18–27; 0.30→0.47 climb across captured ticks |
| EXP-27 damped | 0.5 | 0.5 | ρ·2 ≈ 0.10–0.20 → **clip rarely binds** | median **0.092** (s18–27) → **0.021** (s45–67); peak 0.189@s12 |

(Dose source: comparator-runs' W&B pull, `RUN_COMPARISON.md` §7a — authoritative over my first
local-log extraction, which Ray's line-dedup had biased toward the unique high-dose matrices,
n≈14–28 of 392 per sweep, per-matrix max 0.41 vs true mean 0.092.)

Two consequences: (i) in EXP-27 the clip was essentially **inert** (mean dose never approached
0.5·‖G‖) — the decay alone set the dose, so a further clip cut (0.5→0.25, the verdict's flagged
optional probe) would change almost nothing; (ii) at δ=0.5 the residual *saturates within ~2–3
ticks* (1+δ+δ² ≈ 1.75 of 2.0), which kills the "reset e_t on refresh" mitigation in advance
(§g, PRED-3).

**The decisive fact:** the per-step dose **decayed monotonically** over the run — 0.092 (s18–27)
→ 0.021 (s45–67) — so through the ignition window (steps 61–66) the forcing sat at its run
**minimum**, ~10× below the parent's igniting dose. Ignition is not a dose event. This single
observation already falsifies every "the residual kicked too hard" story and demands a mechanism
where smallness of the per-step push is irrelevant.

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

One more disanalogy that matters for the fix (§g): for a *projector* codec with self-adjoint
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
rotates its update by ~44° *every step* (cos 0.7165) and did not lock in within 50 steps (though
its s50 state carries DANGER precursors — §e.2); ef rotates ~17° (EXP-27: ≤11°) and ignites
outright. Per-step rotation angle does not order ignition across arms, because what matters is
not how far one step turns but whether the turning has a persistent exogenous component that
integrates (§e).

The Lyapunov-flavored version: a certificate needs a function V with V̇ < 0 along trajectories.
Any reward-derived V satisfies V̇ ≈ 0 along reward-*flat* directions — and the forcing is, by F1,
**concentrated** in directions where the reward gradient does not act (it is orthogonal to G; its
reward-coupled tangential components are continually pulled back by the next fresh gradient, its
reward-flat component is not — see (c)/(d)). No certificate over this surface exists without
adding a potential on the flat directions (KL / entropy / length terms — precisely the LABELED
guardrails, §g). Score staying 0.73–0.84 *through* ignition is this statement made empirical:
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
update is ∝ T_i·A_i**. The batch is 128 prompts × n=8 = 1024 responses; at mean length ~175 one
16384-token rollout carries **~8.4% of the entire batch's gradient mass**
(16384 / (1023·175 + 16384); fair share 0.098% ⇒ ~86× amplification). At s61 (clip 0.006 ⇒ ~6
capped rollouts, mean 267.9) the capped tail alone carried ~36% of the batch's token mass from
0.6% of its samples.

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

1. **Susceptibility (≈ s44–55 at THIS dose).** Healthy sharpening brings entropy through
   ~0.48→0.33; group σ shrinks; the policy is now "sticky" (an update on a sharp policy
   persists) and advantage mass concentrates in mixed groups. This stage is driven by *learning
   itself* — it is force-independent and unavoidable on this surface. (Note: susceptibility is a
   joint (dose × sharpness) boundary, not an entropy line — at 2.2× the dose, ef r1 caught at
   entropy 0.83; see §e.3. Entropy here indexes the *healthy* sharpening, it does not cause the
   catch — finding #1.)
2. **Seeding (emission from ~s19; cap-pins s45, s53, s59).** Long-tail emission starts well
   inside the healthy phase — lmax 3220@s19, 9764@s23 (the watch's E1 early gate flags EXP-27
   here, ~38 steps before lock-in; it flags α=0.5 at s17), then isolated
   `response_length/max = 16384` rollouts at s45/53/59 (clip% ≈ 0.001 — single capped samples),
   with growing instability markers: length-mean creep 165 → 185–201 over s52–60, grad_norm
   spikes 21.6 (s57) / 43.7 (s59), erratic ppo_kl from s53. Each early spike *recovers*
   (s59 201→171 at s60) — emission is necessary but not sufficient. Seeds at high entropy with
   no transport history don't catch (s2 had a 16384 fluke at entropy 5.8 — no effect: diffuse
   policy, large group σ, diluted token share; dense's s6 and plain's s1 warmup flukes likewise).
3. **Catch + ratchet (s61–68).** A seed *fails to recover* — the watch-P1 signature (2nd
   consecutive cap-pin, s61–62); gnorm 58.9 (s61); len mean 171→268→295→395→448→558→566→575;
   lmax pinned at 16384 for 8 consecutive steps; IS gap 0.62→0.40; **score 0.73–0.84
   throughout** — reward-preserving, mid-valley. Entropy 0.34→0.25→0.22→0.079 *follows* the
   length curve down (the compositional effect of §d.2 plus self-sharpening on the long mode) —
   it is the trailing edge of the spiral, never its leading edge. Killed s68 (mem 123/143GB).

The merger enters this anatomy **twice**:

- **Transport (the main role):** the persistent tangential force (F1+F2) moves the policy along
  the flat valley *during the healthy phase*, so by the time susceptibility arrives the policy
  already sits near the long-mode region and emits long-tail seeds. The W&B cross-check
  (`RUN_COMPARISON.md` §7b/§8) shows the seed emission tracks the carrier exactly: **every**
  M-fed arm emits isolated cap-pins before any lock-in (ef r2: 16384@27, 7817@32, 4061@47
  despite finishing "clean"; α=0.5: 16384@{17,47,48} + 5806@50; EXP-27: s45/53/59; ef r1:
  5782@12, pin@29), while **plain — the same substrate with the merger OFF — emits nothing**
  (lmax ≤ 826 after s30, zero pins after the s1 warmup fluke) and dense has one warmup fluke
  (s6) only. The carrier, not the codec substrate, is the emission source.
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

| arm | F | additive carrier? | anti-descent? | predicted | observed |
|---|---|---|---|---|---|
| dense / psgd-only | 0 | no | no | stable | val 0.7536 / 0.7415, no ignition |
| plain (B_plain) | 0 (substrate: stale-Q forward only) | no (grad path) | no | stable, but refresh drag | 0.6437, **no ignition** in 50 |
| signed_ema α=0.5 | −G⊙1[sign(M)≠sign(G)] — a *veto-rectifier* on G (see §e.2) | **yes, in expectation**: E[F_i] ≈ ½E\|G_i\|·sign(M_i) on coin-flip coords (per-step a contraction, cos = √½ ✓ measured 0.7165) | no (never reverses a determined coord) | delayed ignition (carrier ~½-strength) | no lock-in by 50, but **DANGER state @50**: consecutive 16384 pins s47–48, lmax 5806@50, len slope +5.9/step — censored mid-path |
| signed_ema α=0.3 | (2α−1)·G on disagree = −0.4·G_disagree | **yes** (sign(M)-replacement reverses fresh energy) | **yes (−0.4)** | ignite, slower than α=0, reward crash | ignited s33, val 0.6164 |
| signed_ema α=0 | −1.0·G_disagree | **yes** | **yes (−1.0)** | fast ignite + reward crash | ignited s30, entropy→0.06, len→8.6k, val 0.354 |
| ef parent (c=1, δ=0.9) | clip(filter(M_⊥)) ≈ 0.32·‖G‖·M̂ | **yes (M̂ injected)** | no (⊥) | ignite, reward-preserving, timing stochastic | r1 ignited 29–42; r2 clean to 50 (first-passage) |
| **ef damped (EXP-27)** | same, λ ≈ 0.02–0.19 | **yes (M̂ injected)** | no (⊥) | ignite **later** (λ halved, τ unchanged), reward-preserving | ignited 61–66, score 0.73–0.84 ✓ |

The empirical law across all eight runs (sharpened by the comparator's W&B cross-check, which
removed what had looked like an α=0.5 exception): **long-tail emission and ignition ⇔ E[F | M]
contains a persistent exogenous carrier** — a nonzero expected force along stale-M directions.
That covers sign-replacement (α<0.5: reverses fresh energy onto stale signs), M̂ injection (ef),
*and* the α=0.5 veto-rectifier (whose per-step contraction still has expectation
½E|G|⊙sign(M) on the coin-flip half — and whose run indeed shows the carrier's emission
signature, §e.2). The two arms with E[F]=0 along any persistent direction — plain and
dense/psgd-only — emit nothing and never ignite. Dose sets the timing (the comparator's numbers
make it almost exactly linear: ef r1 dose 0.200 → lock-in s30; EXP-27 dose 0.092→0.021 → lock-in
s61; ratio ≈ 2.2 vs 2.0); reward-coupling of F sets whether reward crashes during ignition
(α=0/0.3: anti-descent ⇒ crash) or is preserved (ef: orthogonal ⇒ score 0.73–0.84 — the EXP-27
signature, predicted by F1).

### e.1 Why α=0 exploded fast
F has an O(‖G‖) anti-descent component on ~50% of gradient mass (the √2 fingerprint) — it
destroys the sign-cancellation regularizer (grad-norm 0.39→3.3–11), actively sharpens, *and*
carries the M-direction. It drives its own susceptibility (entropy 5.7→0.47 by s30) and ignites
the moment it gets there. Fast, deterministic, reward-crashing.

### e.2 α=0.5 — the veto-rectifier is a half-strength carrier, and the data agree
At the (2α−1)=0 knee, G_corr = G⊙1[agree]: per step, the stale sign(M) only **vetoes or passes**
the fresh per-coordinate push; F = −G⊙1[disagree] has no energy of its own (F_i = 0 whenever
G_i = 0) and is never an ascent direction (cos = √(agree-energy/total) = √½ = 0.707, measured
0.7165 ✓). That per-step contraction is why α=0.5 escaped the EXP-25 *reward-crash* fate.

But the veto is **not neutral in time average**. On near-zero-mean (coin-flip) coordinates —
~50% of the gradient mass (the √2/coin-flip finding) — rectifying a zero-mean fresh push through
a persistent sign pattern leaves E[G_corr,i] ≈ ½·E|G_i|·sign(M_i) ≠ 0: a half-magnitude sign-SGD
drift along sign(M), i.e. a **persistent exogenous carrier in expectation**, energy-limited by
the fresh |G_i| but pointed by the same ~50-step-memory M as the ef carrier.

An earlier draft of this analysis claimed α=0.5's empirics were clean (a misread of the EXP-25
table, whose `clip_ratio 0.000` was the s50 snapshot, not the run). The comparator's full W&B
pull (`RUN_COMPARISON.md` §4/§7b) corrected this decisively: α=0.5 hits lmax=16384 at s17, s47,
s48 (**consecutive** 47–48, the lock-in-onset signature), lmax 5806 still elevated at s50, and a
trailing len-mean slope of **+5.9 tokens/step** over s41–50 — scored DANGER at s50, *worse than
EXP-27 looked at its own s50* (EXP-27 locked in 11 steps later). α=0.5 was not stable; it was the
same carrier trajectory at ~half strength, right-censored at 50 — consistent with the
expectation math above, and removing the one apparent exception to the carrier law (§e table).
Cost of the veto on top of that: ~half the gradient discarded ⇒ val 0.7066 < psgd-only 0.7415.

### e.3 Why EXP-27 ignited at ~61 with the dose at its run minimum
λτ-model: λ cut to ~0.46× (W&B dose 0.092 vs r1's 0.200, decaying further to 0.021), τ (M's
memory) unchanged ⇒ transport rate ~halved ⇒ time-to-basin ~×2. Observed: lock-in s30 (r1) →
s61 (EXP-27) — **ratio 2.03 vs dose ratio 2.17, almost exactly linear** (comparator §7a). The
catch threshold is a *joint* (dose × policy-sharpness) boundary, not a fixed entropy line: at
dose 0.20, r1 caught while entropy was still **0.83** — high — and entropy collapsed only
afterwards, 0.58@36 → 0.13@42 (measured, watch doc §2026-06-11 — the third independent
entropy-trails-ignition confirmation); at dose 0.02–0.09, EXP-27's catch needed the policy much
sharper (entropy 0.25–0.34 at s61). Damping therefore cannot push ignition past the horizon —
it slides the catch along the boundary toward later/sharper, and because the run was *extended*
to 100 steps to chase parity, the slower transport still had time to complete. Damping the dose
is a time-dilation of the same trajectory, which is why it also bought zero val: the
healthy-phase information content is the same.

### e.4 r1 vs r2 stochasticity
Time-to-catch = drift to the (dose × sharpness) boundary + first-passage of a seed catching
(Poisson-ish: a long+correct deviant in a mixed low-σ group). Near the boundary the catch is
fluctuation-dominated ⇒ realization-dependent: r1 caught at 29–30; r2's seeds missed within the
50-step window. EXP-27 (single realization, lower λ) caught at 61. Nothing about r2 was
mechanistically safer — **confirmed by the comparator**: r2 emits the same isolated long-tail
spikes (16384@27, 7817@32, 2648@38, 4061@47) that plain never produces; its pins simply never
landed consecutively before the 50-step censor (`RUN_COMPARISON.md` §7b: "first-passage-lucky").

---

## (f) Attribution: merger-family-specific vs M_anchor-carrier-generic vs substrate-generic

Three candidate scopes for the instability (the operator's discrimination ask), narrowest first:

- **H_family — EF-mechanism-specific** (the additive e_t loop is the defect; signed_ema at the
  α=0.5 knee is safe): **ALREADY FALSIFIED on existing data.** α=0.5 — the other merger family,
  no additive residual, no e_t state — shows the same spiral signature at its censored endpoint
  (consecutive cap-pins s47–48, len slope +5.92/step, DANGER scorecard; P(ignite by 100) ≈
  55–70%), and its anatomy is the same emission→cluster sequence as ef r1/EXP-27. Both families
  share exactly one object: the stale anchor EMA **M** folded into the fast gradient. The
  mechanism never used an EF-specific property (§a–§d need only a persistent E[F|M] ≠ 0; §e
  derives that term for every family member, including α=0.5's rectified ½E|G|·sign(M)).
- **H_carrier — M_anchor-carrier-generic (mine):** any merger with a persistent expected
  exogenous component along M ignites, with dose-proportional lag; plain/dense (no carrier) are
  safe at *any* horizon on this surface (up to the slow √T diffusion floor).
- **H_substrate — substrate/surface-generic:** any run on this surface (no KL/entropy, 16K cap,
  token-mean) ignites once past ~step 55–65; EXP-27 just ran long enough to see it, and the
  50-step no-carrier controls are all censored.

Evidence for H_carrier over H_substrate, now strong: (i) ef r1 locked in at s29–30 — *inside*
the window where plain, psgd-only, and dense were all clean — so the merger demonstrably ignites
where the substrate does not; (ii) the comparator's single-knob isolate (`RUN_COMPARISON.md`
§8): plain differs from ef r2 in EXACTLY `spectral.enabled`, and plain shows **zero long-tail
emission** (lmax ≤ 826 after s30, len slope −1.46, zero pins after warmup) while ef r2 — same
substrate + merger — emits repeatedly (16384@27, 7817@32, 4061@47); (iii) plain spends the
*longest* time of any arm in the high-entropy seed window (entropy never below 0.478) and still
emits nothing, so "exposure time" doesn't produce seeds — the carrier does. Remaining honest
limit: every no-carrier run is right-censored at 50, so H_substrate past s50 is unfalsified by
direct observation — plain@100 (PRED-2) is the discriminator the comparator's addendum also
calls for. (Weak prior evidence: the EXP-17 core diagnostic ran ~2 epochs ≈ 110+ steps on the
mask+clean@20 substrate — same GRPO/no-KL/16K surface, different codec — with no recorded length
explosion.)

**Falsifiable discriminating predictions (pre-registered).** PRED-1 was REVISED after the
comparator's cross-check overturned my reading of α=0.5's endpoint (see §e.2); the original
"no ignition, P≈0.7" is superseded — recorded here for honesty since the whole point of
pre-registration is not to quietly rewrite it:

- **PRED-1 (revised) — signed_ema α=0.5 @100** (same surface, 1wulaelw config, step_target 100):
  **DOES ignite by s100**, lock-in window ~s55–85, preceded by the PRED-5 precursor sequence
  (its s47–48 consecutive pins were the onset signature). **P(ignite by s100) ≈ 0.6** (comparator
  independently: 0.55–0.70). If it ignites: extends the carrier law to rectified carriers (the
  ½E|G|·sign(M) expectation term is sufficient). If it survives 100 clean (P≈0.4): the additive
  vs rectified distinction is load-bearing, and the carrier law narrows to *additive* carriers
  only — still excluding plain/dense either way. Whatever the outcome, val@100 stays 0.69–0.72
  (the veto cost does not close parity).
- **PRED-2 — plain @100** (B_plain config): NO ignition by s100 AND no long-tail emission (zero
  lmax ≥ 4k events after warmup); val@100 0.65–0.70 (climbing from 0.6437 but the refresh-alone
  drag keeps it at/below the floor band). **P(no ignition) ≈ 0.85.** This is now the single
  cleanest H_carrier/H_substrate discriminator: plain is the substrate-only control with zero
  carrier and zero emission at 50.
- **PRED-3 — ef-with-residual-reset** (damped settings + e_t←0 at every anchor refresh): **still
  ignites**, ~s55–75. At δ=0.5 the residual saturates geometrically — within a 5-tick reset cycle
  e reaches (1−δ⁵)/(1−δ) = 1.94 of its 2.0 steady state, and the cycle-average dose is
  (1/5)·Σ_{j=0..4}(1−δ^{j+1})/(1−δ) = 1.61 vs 2.0, i.e. resets shave only **~19%** of the average
  dose — while the carrier persistence lives in **M**, which the reset does not touch.
  **P(still ignites by s80) ≈ 0.7.** If this arm does NOT ignite, my model is wrong about where
  the persistence lives (it would be in e, not M) — that is the cleanest single-run falsifier of
  this document.
- **PRED-4 — outcome table (the decisive pair is plain@100 vs any carrier arm @100):** if
  plain@100 stays emission-free while α=0.5@100 and/or an ef@100 rerun ignite, H_carrier is
  confirmed and H_substrate dead (α=0.5 igniting is *consistent* with H_carrier under the
  rectified-carrier reading, §e.2 — it is plain that separates the hypotheses; H_family is
  already dead either way). If **plain** ignites at ~s55–70 with the precursor sequence,
  H_substrate wins and the merger is only an accelerant — in that case the *surface*
  (token-mean + no-brake + cap) must be fixed before any merger work continues.
- **PRED-5 — precursor universality:** every ignition on this surface is preceded by ≥2 lmax-pin
  events and a length-mean creep before the watch-P3 (2×) trigger fires. **Already
  retro-confirmed on ef r1** (comparator §7a: isolated 5782@s12 → pin@29 fails to recover →
  sustained pins s30–42 → len creep 143→328 → entropy follows down — the identical anatomy at
  ~half the lag). If a future ignition ever occurs *without* these, the seeding stage of §d.3
  is wrong.
- **PRED-6 — carrier-content control (discriminates "any persistent direction" vs "stale
  GRADIENT direction"):** run the ef formula with M replaced by a **frozen norm-matched random
  matrix** per target (a merger-axis variant, allowed — the merger is the variable axis; no
  locked invariant touched). H_carrier-as-written predicts ignition with the same anatomy
  (the manifold rectifies *any* persistent exogenous direction, §b/§d) — **P(ignite by s100) ≈
  0.55**, weaker-than-M because a random direction has less overlap with the policy-relevant
  subspace than a real stale gradient. If it does NOT ignite, the stale-*gradient* content of M
  is load-bearing (transport needs policy-shaped directions), sharpening the theory and raising
  the value of carrier-content (not just carrier-magnitude) fixes. Either outcome is
  informative; cheap (50–100 steps, one arm).

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
2. **Retire the M-fed merger class — but mind the program bind.** The cross-arm table (§e) is a
   completed dose-response over the class {sign-carrier, M-carrier, rectified carrier, none}:
   every carrier arm either locked in (α=0/0.3, ef r1, EXP-27) or shows the emission/DANGER
   signature censored at 50 (ef r2, α=0.5); both no-carrier arms are emission-free; and the best
   carrier arm (0.7210) < psgd-only's own 0.7415. There is no interior optimum to tune toward —
   same shape of conclusion as EXP-25's α-monotonicity, one level up. **The bind:** the merger
   cannot simply be dropped on the locked substrate, because plain (no merger) sits BELOW the
   no-refresh floor (0.6437 < 0.6914) — the merger was buying +7.7 pts of val *while* carrying
   the spiral (EXP-26 finding 2: it partly compensates the substrate's stale-refresh drag). The
   successor must therefore deliver the rescue WITHOUT the carrier — which is exactly what
   true-EF (#1) is shaped to test: it re-supplies dropped signal content with zero persistent
   exogenous direction. If the +7.7 was M's *signal content* (off-Q gradient information),
   true-EF keeps it; if it was M's *persistence* (an accidental momentum), true-EF loses it and
   the substrate-drag question (§i) becomes the front. Either result decides the next move.
3. **LABELED guardrail arms (objective changes — never the core):**
   - **KL to reference (proven):** the EXP-25 KL probe closed the length channel outright
     (len 294→120, clip 0.000 for 50 steps, no crash) — it adds the missing potential on the flat
     direction (V̇ < 0 along length). Cost: sub-parity val (0.6793) at coef 0.001 — a brake, not a
     fix for the carrier bias.
   - **Length-normalized aggregation** (`loss_agg_mode=seq-mean-token-mean` or
     advantage/length normalization): sample weight becomes ∝ A_i instead of T_i·A_i ⇒ the ~86×
     tail amplification (§d.2) collapses to 1× ⇒ the ratchet's loop gain < 1 ⇒ a caught seed
     cannot self-reinforce. The cheapest *single-knob* change that breaks the basin's interior
     dynamics (it leaves the flat direction flat but removes the rectifier). Run only as a
     labeled arm: it changes the estimator the entire program is normalized on.
   - **Entropy floor (entropy_coeff > 0):** delays susceptibility (holds σ_g up). Weakest of the
     three — it stretches stage 1 but leaves transport + rectifier intact.
4. **Early-stop / checkpoint-at-peak science (measurement, not a fix) — NOW STANDING.** The
   precursor sequence gives two layers of warning, and the watch doc has been re-centered on
   exactly them (`ENTROPY_COLLAPSE_WATCH.md` §2026-06-11, retro-validated on all 6 runs):
   **E1** (any lmax > 4000 in steps 10–30 ⇒ UNSTABLE-LIKELY, suspicion-only — flags EXP-27 via
   its s19–23 emission ~38 steps pre-ignition, and α=0.5 at s17) arms **watch-P1** (2nd
   consecutive cap-pin ⇒ kill — retro-dicts EXP-27 at s62 vs the actual ~s66 kill, and α=0.5 at
   s48), with **watch-P2** (len-mean slope > +2/step) and **watch-P3** (mean > 2×) as
   co-triggers; entropy demoted to corroborator (finding #1). Note what *failed* retro-testing
   as early signals (do not gate on them): entropy decline rate (identical −0.06…−0.08/step
   across all merger arms regardless of outcome), len-mean slope at ≤30, p90(lmax), grad-norm
   spikes. Banks nothing for parity (val@50 0.7202 < 0.7414 regardless) but protects every
   long-horizon run from now on.
5. **Residual reset on refresh (cheap falsifier only).** Run it to test PRED-3 — it is the
   discriminating experiment for *where the persistence lives* — but the math (§0, e-saturation
   in ~2 ticks vs 5-tick resets; carrier in M) says it will not prevent ignition. Do not sell it
   as a fix.
6. **Rank anneal (NO).** comp_t is built from M, not from the codec error; rank enters only
   through G_comp's direction. Raising r late buys recon fidelity the run already has (0.975+
   energy) and does not touch the carrier. Decline.

---

## (i) Gradient quality — the upstream axis (operator directive 2026-06-11)

### i.1 The facts, placed honestly in the causal chain

The grad-norm ladder: dense ≈ **0.35** flat and clean; psgd-only ≈ 1.6; plain 3.4 median /
10.5 max; merger arms 3–7 healthy; **20–60 in the ignition window**. Two different phenomena
live in that ladder and must not be conflated:

- **The substrate inflation (4–20× dense, shared by ALL comm-eff arms).** This is NOT the
  ignition discriminator — the watch-doc correction is explicit and the controls prove it:
  plain carries the same noisy gradient class as the merger arms (median 3.4/max 10.5 vs ef
  r2's 4.9/13.5) yet emits zero spikes, and psgd-only (1.6) ties dense at 0.7415. Within
  comm-eff arms, grad-norm character does not separate exploders from survivors; carrier
  presence does (§f). Dense's clean 0.35 is a *consequence* of being merger-free and
  codec-free, not a protective cause.
- **The ignition-window spikes (20–60).** These are the *catch signature*, not a cause: one
  capped rollout is ~36% of the batch's token mass (§d.2), so the batch gradient it produces is
  huge. They are downstream of the spiral. (Note `grad_clip` then caps the applied update's
  norm, so what survives into Adam is the spike's *direction* — norm-centric telemetry
  systematically understates what these events do.)

So where IS gradient quality causally upstream? Three places, none of them "noise causes
ignition":

1. **The val gap and the substrate drag.** Comm-eff arms run 4–20× the gradient norm at (by
   construction) no more signal — i.e. a much lower SNR estimator — and every comm-eff arm
   lands 3.3–4.7 pts below dense; plain lands below the no-refresh floor outright (0.6437 <
   0.6914). The B_plain-below-floor mystery (what exactly does the stale-Q refresh do to the
   gradient?) is an unanswered gradient-quality question, and it is the program's binding
   constraint (§g.2).
2. **The susceptibility window.** Low-SNR gradients sharpen the policy more slowly (entropy
   crosses 0.4 at s45–51 for comm-eff arms vs s1 for dense) — the arms sit in the
   seed-emitting-while-sticky window ~45–50 steps instead of ~1. Gradient quality sets the
   *width of the window* in which a carrier can catch; the carrier still has to do the catching
   (plain: widest window, zero emission).
3. **The ratchet gain.** Token-mean makes the gradient estimator heavy-tailed *by construction*
   (per-sample weight ∝ T_i) — the ~86× tail amplification is a property of the estimator, not
   of the policy. This is the gradient-quality defect that converts one caught seed into a
   runaway.

That yields the organizing decomposition for the research axis — gradient quality has three
nearly-orthogonal components here:

| component | what it is | what it causes | who has it |
|---|---|---|---|
| **bias / persistence** | E[F\|M] ≠ 0 along a slow direction | ignition (transport) | merger arms only |
| **variance / SNR** | 4–20× norm at fixed signal | val gap; wide susceptibility window; plain<floor | all comm-eff arms |
| **heavy tails** | per-sample weight ∝ T_i (token-mean) | ratchet gain ≫1 once seeded | every arm incl. dense (latent) |

### i.2 Standing telemetry to instrument (ranked by information/cost)

1. **Token-mass concentration (ratchet-gain telemetry).** Per step, from already-logged lengths:
   participation ratio `PR = (Σ T_i)² / (N·Σ T_i²)` and top-1%-of-samples token share. Directly
   measures the §d.2 amplification as it happens (s61: ~36% from 0.6% of samples). Zero cost, no
   new hooks — the highest-value missing number. Complements watch-E1/P1 (which see the rollout
   side; this sees the *gradient-weight* side).
2. **Micro-batch SNR + sign-cancellation ratio (the implicit-regularizer gauge).** Per target,
   across the gradient-accumulation micro-batches within a tick: `SNR = ‖ḡ‖² / Var_mb(g)` and
   the cancellation ratio `‖Σ_mb g‖ / Σ_mb ‖g_mb‖`. EXP-25 §3b identified destroyed
   sign-cancellation as the step-size regularizer; this makes it a logged per-step quantity and
   gives the dense-vs-comm-eff SNR gap a number (run one dense probe for the reference band).
   Cheap hooks at accumulation boundaries.
3. **Direction-persistence pair — the loop gain λ·τ as telemetry.** Per target:
   `cos(G_t, G_{t−1})` (fresh-gradient correlation time) and, on merger arms,
   `cos(F_t, F_{t−τ})` for τ ∈ {1, 5, 25} ticks plus the running mean **‖mean₁₀(F)‖/mean₁₀(‖G‖)**
   (the E[F] carrier estimator). The carrier law (§e) becomes directly monitorable: this metric
   would have shown α=0.5's rectified drift and ef's M̂ injection on day one, and λ·τ — the
   quantity that actually predicts time-to-ignition — becomes a dashboard number instead of a
   post-mortem reconstruction.
4. **cos to a dense-reference probe under the SAME loss (the never-logged EXP-20 criterion).**
   Cadence-gated (every K steps) measurement-only fresh full backward with
   `fresh_anchor_loss=ppo_clip` (the capture machinery already supports it; the stepA lesson:
   a clean-PG reference makes the cosine meaningless). Settles per-step how much signal the
   codec+merger path loses, and is the validity anchor for metrics 2–3. Cost: one extra full
   backward per K steps, measurement-only (no realism violation — probes never feed the
   optimizer).
5. **Heavy-tail indicators on the gradient itself.** Per target: kurtosis of coordinate values,
   top-0.1%-coordinate share of squared mass; across targets: max/median grad-norm ratio.
   Catches concentration events that norm alone hides (and is robust to grad_clip's
   norm-flattening).

### i.3 Mitigations that target gradient quality DIRECTLY (vs symptom guardrails), ranked

Symptom guardrails (KL, entropy floor, length caps) leave the estimator broken and tilt the
landscape instead; these act on the estimator itself:

1. **True-EF (bias axis — core-eligible).** §g.1. Removes the only nonzero E[F] persistence and
   re-supplies the codec's dropped signal with bounded lag — the bias-axis fix and the carrier
   killer in one move. Also the decisive experiment for the §g.2 bind (signal-content vs
   accidental-momentum reading of the +7.7).
2. **Diagnose-then-fix the substrate inflation (variance axis — diagnostic first,
   core-eligible).** Instrument i.2 metrics 2+4 on plain vs psgd-only vs dense for ~20 steps:
   is the 4–20× inflation (and B_plain's below-floor val) PPO-clip ratio drift post-warm,
   stale-Q forward error, or destroyed micro-batch cancellation? Whichever it is points at its
   own fix (e.g. Q-refresh cadence/warm-start if stale-Q; recompute-path consistency if ratio
   drift). This is the cheapest experiment that addresses the program's binding constraint
   (plain < floor) and it is pure measurement on existing arms.
3. **Length-normalized aggregation (tails axis — LABELED estimator arm).** seq-mean-token-mean
   sets per-sample weight ∝ A_i, collapsing the ~86× tail amplification to 1× ⇒ ratchet gain
   < 1 even with a carrier present (§g.3). It is an *estimator* change, not a reward change —
   but it re-weights the objective the whole program is normalized on, so it stays a labeled
   arm, never silent. (Worth one labeled run paired with a carrier arm: if carrier+seq-mean
   does NOT ignite, the ratchet — not the transport — is confirmed as the irreversibility
   step.)
4. **Optimizer-side tail robustness (Adam β2/eps, per-coordinate update clipping) — LABELED,
   low priority.** Changes the locked optimizer surface for at most a second-order gain; only
   if 1–3 leave residual instability.

---

## (h) THE CORE IDEA

The implemented ef_powersgd is not error feedback; it is a **persistent tangential forcing loop**
— and the forcing object is not EF-specific, it is the **anchor EMA M itself**, which every
merger family folds into the fast gradient in a different functional form (sign-replacement,
veto-rectification, additive injection) with the same expected effect: a nonzero persistent
exogenous component along M's ~50-global-step-memory direction. By construction (comp_t ⊥
G_comp, with the projection nearly vacuous because M ⊥ G_comp anyway), the ef variant injects
this force exactly *sideways* — it neither helps nor hurts the current descent direction. On
GSM8K under the locked surface (length-agnostic reward, no KL/entropy, 16K cap, token-mean
GRPO), the policy's reward landscape has exactly one unbounded flat direction — *make correct
answers longer* — and motion along it draws zero restoring force from the fresh gradient, while
the token-mean loss stands ready to amplify any long-and-correct deviant by ~10–90× its fair
gradient share. Persistent sideways forcing on a manifold with one unopposed flat direction is
rectified into net transport along that direction (every other component gets pulled back; the
flat one integrates linearly, ~λ·T). So the run looks healthy by every per-step metric — score
is *blind* to level-set motion, the cosine *certifies only* the descent component, and entropy
is a **follower**, not a trigger (dense trains lower-entropy than every comm-eff arm and never
ignites; ef r1 ignited at entropy 0.83 and only then collapsed) — right up until the
transported policy is sharp enough *at its dose* for a long-tail seed to catch, and the
token-mean ratchet finishes the job reward-preservingly (score 0.73–0.84 through ignition).
Dose magnitude divides the transport rate and nothing else: cutting it ~2.2× stretched
time-to-lock-in almost exactly linearly (s30 → s61) at zero val gain, and the run ignited with
the forcing at its run-minimum. The exits, in order: remove the carrier while keeping its
signal content (true EF on the codec's own per-step dropped residual — telescoping, no O(T)
integral; simultaneously the test of whether the merger's +7.7-over-plain was signal content or
accidental momentum, §g.2), fix the estimator's heavy tail (length-normalized aggregation,
labeled), or tilt the flat direction with an explicitly-labeled potential (KL). Tuning λ
(clip/decay/reset) only buys time inside the same trajectory — falsified twice, now with the
mechanism that says why.

---

## 8. Cross-check ledger (comparator-runs) — RESOLVED

Four asks were sent to comparator-runs; their answers (`RUN_COMPARISON.md` §7–8,
`comparison_metrics/*.csv`) and the resulting revisions to this document:

- **(A) lmax curves — prediction PARTLY WRONG, document revised.** I predicted α=0.5's lmax
  "stays < ~1k": FALSE — α=0.5 pins 16384 at s17/47/48 (consecutive 47–48) + 5806@s50, DANGER
  scorecard at s50. My other half held: ef r2 *does* emit late isolated spikes (16384@27,
  7817@32, 4061@47) despite finishing clean — first-passage-lucky, censored-not-safe (§e.4).
  Revisions: §e.2 rewritten (α=0.5 = half-strength rectified carrier, expectation math now
  primary); §e table + carrier law restated over E[F|M]; PRED-1 flipped to "ignites by 100, P≈0.6."
- **(B) entropy@s50 ordering — CONFIRMED.** dense crosses 0.4 at s1 and sits 0.12–0.16 from s36;
  carrier arms cross 0.4 only at s45–51; plain never (min 0.478). Merger arms sit in the
  seed-emitting window ~45–50 steps vs dense's ~1. Comparator's caveat accepted: window time is
  correlational — plain has the *longest* window and zero emission, so the window multiplies
  carrier-driven seeds, it does not create them.
- **(C) r1 precursor sequence — CONFIRMED (PRED-5 retro-check passes).** Identical anatomy at ~half
  the lag: isolated spike (5782@12) → pin fails to recover (s29–30, first clip>0) → sustained
  pins s30–42 → len creep 143→328 → entropy follows down (0.58@36 → 0.13@42) → score 0.73–0.78
  no-warning throughout. Dose-lag linearity: dose 0.200 vs 0.092 (2.17×), lock-in s30 vs s61
  (2.03×).
- **(D) contradictions —** (i) tangential-forcing: not contradicted; strengthened by the §8
  single-knob isolate (plain = ef r2 minus `spectral.enabled`, zero emission). (ii)
  dose-sets-lag-only: confirmed, now near-exactly linear. (iii) α=0.5-has-no-carrier:
  **contradicted and withdrawn** — replaced by the rectified-carrier expectation math (§e.2),
  which the comparator's data independently demanded. Their counter-candidate in their §5
  ("noisy large gradients = the susceptibility factor") is contradicted by their own §8: plain's
  grad_norm (mean 7.4/max 10.5) is in the same class as ef r2's (9.3/13.5) with zero emission,
  and EXP-25's psgd-only (grad_norm ~1.6, no merger) was clean at 0.7415 — gradient
  size/noisiness does not separate exploders from survivors; carrier presence does.

Also corrected via (their) W&B pull: my early-dose extraction from the local log was biased high
by Ray line-dedup (kept the unique high-dose matrices, n≈14–28 of 392); §0's dose table now uses
their `rel_change_mean` numbers. Net effect of the cross-check: two of my claims were wrong
(α=0.5 endpoint state; early-dose magnitude), both revised; the core mechanism (carrier →
tangential transport → seed → token-mean ratchet; dose = lag only) survived every test and came
out sharper.

**2026-06-11 operator-directive rebuild (second revision).** Per the operator's confirmation of
the comparator's challenge, this document was restructured around the two load-bearing findings
(intro): entropy-as-follower (finding #1 — woven through §d.3, §e.3, §h; watch-doc re-centering
adopted in §g.4, and my prediction tags renamed PRED-1…6 to avoid colliding with the watch's
P1/P2/P3/E1 triggers) and carrier-generality across both merger families (finding #2 — §f
restructured into the three-way H_family / H_carrier / H_substrate discrimination, with H_family
retro-falsified and PRED-6 added as the carrier-content control). The open question from my
first cross-check message — ef r1's entropy at lock-in — is answered by the comparator's pull:
**0.83** (high), now cited in §e.3 as the third entropy-trails-ignition confirmation. The
comparator's task-#6 retro-test of the E1 early gate (lmax > 4000 in steps 10–30; sensitivity
1.00, zero false negatives on n=6; flags EXP-27 at s19–23 and α=0.5 at s17; entropy decline rate
FAILED as an early signal — identical −0.06…−0.08/step across all merger arms regardless of
outcome) is folded into §d.3 stage 2 and §g.4. New §i develops gradient quality as the upstream
research axis (bias/variance/tails decomposition, ranked standing telemetry, ranked
direct-vs-symptom mitigations) — placed carefully so it does not contradict the watch-doc
correction that grad-noisiness is NOT the ignition discriminator (plain falsifies it; §i.1).
