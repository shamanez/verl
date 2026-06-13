# The Best Communication-Efficient GRPO Solution To Date — Final Verdict

> **Author:** synthesizer (team `commeff-grpo-verdict`, task #1) · **Date:** 2026-06-13
> **Scope:** the single most optimal, best-performing comm-efficient GRPO configuration this
> program has produced, with its full anatomy (moving parts + what must be stored).
> **Sources:** `.claude/GOAL.md`, `runs/EXP-30/{verdict.md, PATH_FORWARD.md, stepA_gate.md,
> resolved_params_B2.txt, PROGRESS.md}`, `runs/EXP-30/pathforward/{mechanist,critic,strategist}.md`,
> `LOG.md`, `runs/SUMMARY.md`, `CODE_WALKTHROUGH.md`, `runs/FIXED_CONTROL_SURFACE.md`.
> Every number is quoted from those artifacts; W&B run IDs are cited inline. Numbers from the
> in-flight controls (C1 dense-rerun, C3 frozen-Q) are explicitly marked **PENDING**.

---

## 0. The verdict in one line

The best solution is **B2 — a K-delayed *exact codec-residual* correction
(`correction_mode=delayed_ef`, λ=1, β_anc=0) layered on the EXP-29 generator-consistent
paired-replay substrate (PowerSGD r=77 act-basis codec, anchor-owned `Q`, cadence/delay_K=5
ticks, `clean_cadence=0`).** It reached **val@50 = 0.7528** (W&B `u9okvgzz`/`exp30_B2_delayed_ef`)
— **past the 0.7210 realistic floor (+0.0318), past the 0.7414 parity bar, 0.0008 under the
dense ceiling 0.7536** — **emission-free**, and a 100-step extension (W&B `b59ncque`,
`exp30_B2_ext100`) stayed emission-free through the EXP-27 ignition band (vals
0.7278@25 / 0.7536@50 / 0.7475@75 / 0.7400@100). It is the **first** correction-carrying cell in
the program's history to convert without igniting.

It is the best for three reasons, in order of weight: (1) it is the only configuration that
**both** matches the dense learning curve at the measurement point **and** survives the stability
horizon that killed every prior merger; (2) its mechanism is *understood and de-risked*, not
lucky — it cancels a measured, dominant, near-orthogonal codec artifact rather than fighting it;
(3) it adds **zero new inter-stage traffic** over the substrate the program had already settled.

**Two honest caveats carried throughout, neither of which unseats B2:** the val win is **one seed
at one validation point** — statistically *consistent with* dense but not *proven equal* to it
(B2−dense z ≈ −0.05 at binomial SE ≈ 0.0119; critic T2) — and "stable" means **emission-free,
censored at 100 steps for seed 0**, not "stable forever" (the carrier law gives no finite-horizon
guarantee; m6 ≈ 0.62; critic T1).

---

## 1. THE solution, stated precisely

B2 is a **substrate** + a **single correction primitive**. The substrate is locked program-wide
(`FIXED_CONTROL_SURFACE.md` ☆); the correction is the one knob B2 changes. Ground truth for every
value below is `runs/EXP-30/resolved_params_B2.txt` (last-write-wins Hydra extract), not prose.

### 1a. The codec (inter-stage compression — the GOAL-3 mechanism)

- **PowerSGD-style activation compression** on the pipeline-parallel boundary
  (`compression_type=powersgd`). A shared low-rank orthonormal basis `Q` (per boundary) projects
  each boundary activation: send only `Y = h·Q` (rank **r = 77**), reconstruct `ĥ = Y·Qᵀ`.
  `Q` is **frozen within a step**, fp32 QR (`qr_dtype=fp32`), deterministic seed.
- **r = 77** is byte-matched to the legacy mask `p=0.95` at Qwen2.5-1.5B's **H = 1536**
  (`0.05·1536 ≈ 77`). This is the **only** codec compatible with anchor-owned `Q`.
- DP-axis is **not** compressed; only the PP boundary is. `sync_basis=true` (the sketch `V` is
  all-reduced over the DP group so `Q` is one shared codebook — `sync_basis=false` diverges).

### 1b. `Q` management (the anchor circuit — MANDATORY substrate)

- **The anchor is mandatory** (`anchor.enabled=true`). It is a continuously-maintained,
  **`delay_K=5`-stale**, **full-coverage** (all **196** weight-matrix gradients = 28 layers × 7),
  **DP-reduced** anchor gradient, refreshed every **`cadence=5` optimizer ticks** from a *no-hook
  isolated clone* loaded with a `delay_K`-stale weight snapshot. (Cadence/delay_K count **optimizer
  ticks**, not global steps; at batch128/mini64 = 2 ticks/step, so cadence-5 = a refresh every
  ~2.5 global steps.)
- **The anchor is the ONLY thing that updates `Q`** (`anchor.owns_q=true`): `Q ← orth(V)` is
  computed by block power iteration on the anchor's **stale-weight forward activations** and
  broadcast DP-wide each refresh. The fast (compressed) circuit is a **fail-closed, read-only**
  consumer of `Q` — its own basis-update path raises if ever entered.
- **`clean_cadence = 0`** — the periodic full-rank "clean dense step" crutch is **dead**; the
  anchor circuit *is* the realistic, comm-efficient replacement for it (a low-frequency, stale,
  full-gradient reference instead of an unrealizable full-H clean transfer).
- **`q_basis = act`** — the activation basis is the only production basis. The update-energy /
  hybrid-`Q` corner is **falsified** (EXP-26 Step C: it anti-converts, because rollouts are
  uncompressed and a gradient-tuned `Q` degrades the forward reconstruction). `hybrid_*_cols = −1`.

### 1c. The generator-consistent paired-replay feed (EXP-29 substrate addition)

- **`replay_paired_batch = true`** (the EXP-29 contribution, PR #16): the anchor's stale weights
  are paired with the **trajectories those weights actually generated**, so the anchor gradient
  `G_anc_rep` is a *valid* on-policy policy gradient for the stale snapshot — not a stale-weights ×
  current-batch chimera. `snapshot_device = cpu` (snapshots held off-GPU). Fire-aware ring
  retention keeps the matching `(weights, batch)` pair alive across the delay window.
- This is the substrate property that makes B2's residual *exact*: it lets the anchor produce a
  gradient on the **identical (batch, θ)** that the ring's compressed gradient was computed on, so
  their difference is pure codec error (§3).

### 1d. The correction primitive — the one knob B2 changes (`delayed_ef`, λ=1, β_anc=0)

```
G_corr(t) = G_comp(t)  +  λ · δ(t),     λ = 1
δ(t)      = G_anc_rep(t)  −  G_comp_ring(t − K)        # ring pair, identical (batch, θ)
```

- **`correction_mode = delayed_ef`** — a **K-delayed exact codec-residual** correction. `δ` is the
  difference, at the *identical (batch, θ)*, between the **valid full-rank anchor gradient**
  `G_anc_rep(t)` and the **stored compressed gradient** `G_comp_ring(t−K)` the codec actually
  emitted K ticks ago. With **λ = 1** the two compressed terms telescope and `G_corr(t)` collapses
  toward the **true gradient** `G_anc_rep(t)` (the cancellation is exact when the fast circuit is
  K-step-stationary; §3).
- **`β_anc = 0`** — there is **no EMA**. The anchor signal is *not* accumulated across fires; it is
  refreshed-and-held. This is the deliberate choice that distinguishes B2 from the EXP-27 family
  that ignited: β_anc=0 prevents *compounding* the carrier (it does not make it memoryless — see
  the m6 caveat in §3/§4).
- **`ef_decay = 0`, `ef_clip = 0`** — no error-feedback decay, no clipping. The δ lifecycle is:
  cold-fallback (= plain PowerSGD) until the first valid replay pair exists (tick 10 = step 5),
  then refresh-at-fire / hold-between. `merger_coldM_fallbacks = 0` post-warmup.
- **Hygiene (all verified inert in `resolved_params_B2.txt`):** the inherited
  `signed_ema_alpha=0.5`, `blend_eta=0.3`, `inject_gamma=1.0`, `max_targets=−1` are
  present-but-dead defaults — `delayed_ef` is the *only* active correction; `signed_ema` is not
  live; `β_anc=0`. The A→B2 controlled-variable diff is **exactly**
  `{correction_mode none→delayed_ef, total_training_steps, experiment_name}`; substrate
  byte-identical.

### 1e. The fixed control surface (held constant, not part of "the solution" — quoted for completeness)

Qwen2.5-1.5B-Instruct · GSM8K · **vanilla GRPO, no-KL / no-entropy** (`use_kl_loss=False`,
`use_kl_in_reward=False`, `entropy_coeff=0` — the `kl_loss_coef=0.001` token in the launcher echo
is a dead knob behind the `False` gate; **resolved_params is the only authority**, false alarm
resolved by team-lead + critic) · lr 1e-6 AdamW · batch 128 / mini 64 / micro 1 · `rollout.n=8` ·
max_response 16384 · 4×H200. Standing OOM guards: `expandable_segments:True`, `ema_device=cpu`,
actor token budget 18432 while the anchor is on.

---

## 2. The empirical case — the full ladder

All numbers are GSM8K greedy `mean@1` on the 1319-problem test set, **val@50** unless noted.
"Emission" = a post-warmup step in [10, 50] with `response_length/max > 4000` (the length-explosion
reward-hack signature that killed prior mergers).

| rank | configuration | val | Δ vs dense | emission | W&B / source |
|---|---|---|---|---|---|
| **— ceiling** | **dense control** (comm-eff OFF, byte-identical to verl) | **0.7536** | — | n/a | `5e2jpho9` |
| 1 | **B2 — K-delayed exact residual** (λ=1, β_anc=0) ← **THE SOLUTION** | **0.7528** | **−0.0008** | **ZERO** | `u9okvgzz` |
| — bar | **parity bar** | 0.7414 | −0.0122 | — | derived |
| 2 | A0 — fresh-clean@5 (the old *unrealistic* clean-step PASS, EXP-20) | 0.7415 | −0.0121 | — | `oquyeic3` |
| 3 | **B1 — blend merger** on valid M (η=0.3, magnitude-matched) | **0.7422** | −0.0114 | (operator run) | PROGRESS 14:47 |
| 4 | ef_powersgd r2 floor (EXP-26 best-realistic; **the pre-registered floor**) | 0.7210 | −0.0326 | — | `tilwe80t` |
| 5 | signed_ema α=0.5 (EXP-25, falsified merger) | 0.7066 | −0.0470 | — | `1wulaelw` |
| 6 | no-refresh floor (EXP-23 A1) | 0.6914 | −0.0622 | — | — |
| 7 | plain PowerSGD on substrate, pre-replay (EXP-26) | 0.6437 | −0.1099 | — | `u1v94opv` |
| 8 | **C2 — plain PowerSGD + Q-updated, NO merge** (on replay substrate) | **0.6300** | −0.1236 | — | `k6nmcuyd` |

### 2a. The B2 trajectory and its de-censoring extension

- **B2 (50 steps, `u9okvgzz`):** val 0.0864@0 / 0.7036@25 / **0.7528@50**. Response length **declines**
  274 → ~204; entropy settles flat ~2.0–2.2; grad_norm ~2–5; `clip_ratio ≡ 0`. The only 16384 pin in
  the whole run is a single *pre-injection* step-2 rollout (1/1024, `clip_ratio=1/1024`, non-consecutive).
  `delta_ratio` bounded and **declining** 1.37 → 1.03 over the run (no monotone climb). `bytes_ratio`
  0.05037–0.05056 every step (in-band). max_mem 28.66 GB (< 30.77 ceiling).
- **ext100 (100 steps, same settings, `b59ncque`):** **de-censored for seed 0 — no ignition through
  step 100.** It traversed the EXP-27 ignition band (51–66) cleanly. Vals 0.7278@25 / **0.7536@50
  (= the dense ceiling value)** / 0.7475@75 / 0.7400@100 — mild late decay (@75 still above parity,
  @100 above the floor). Emission reported honestly: steps 10–93 fully clean; two **isolated
  single-rollout cap-pins at steps 94 and 99** (each `clip_ratio = 1/1024`, len/mean flat 190–227,
  entropy 1.3–1.8 healthy, max reverts immediately) — the benign stochastic-outlier base rate
  (3 single-rollout pins in ~150 observed steps), **not** the ignition signature (EXP-27's signature
  was len/mean climbing 171→575 with entropy → 0.08). max_mem 30.75 GB (**100-step headroom ≈ zero** —
  any future ≥100-step cell must re-derive the ceiling). `delta_ratio` settled ≈ 1.001.

### 2b. The two decisive deltas (what the controls price)

**The controls were run specifically to attribute B2's gain.** Their reading (PROGRESS 14:47) is
**decisive and partly overturns the earlier pathforward framing — stated honestly below.**

- **Merge-value (the headline delta): B-cells − C2 ≈ +0.09 … +0.12.**
  C2 (`k6nmcuyd`, plain PowerSGD + Q-updated, **NO** gradient merging, on the full EXP-29 replay
  substrate) bottoms out at **0.6300** — essentially the old plain floor (0.6437, `u1v94opv`). Both
  merging cells clear it by a wide margin: **residual B2 0.7528 (+0.1228 over C2)**, **blend B1
  0.7422 (+0.1122 over C2)**. **The substrate does not drive the gain; the merger does.** This is the
  cleanest single-knob attribution the program has — C2 is byte-identical substrate to B2 with only
  the correction removed, so it **resolves the T3 confound favorably**: the old "+0.1091 over
  pre-replay plain" two-delta caveat is now a one-delta read against the replay-substrate plain.
- **Q-update-value: C2 − C3 = PENDING.** C3 (frozen-Q, `owns_q` effectively off) has **not** returned.
  The C2−C3 delta will price what the anchor's power-iteration `Q`-refresh buys on its own (i.e. how
  much of the substrate is the *adaptive basis* vs the *rank-77 projection*). Until C3 lands this is
  **unquantified**; do not assert a value.
- **Dense re-run on current code (C1, `RUNNING`, last cell): val@25 = 0.7566 (PENDING @50).**
  C1 re-establishes the dense ceiling *on the exact current code* (the canonical `5e2jpho9` = 0.7536
  predates this branch). val@25 0.7566 is already in the dense-ceiling neighborhood; **val@50 is
  pending** and is the apples-to-apples ceiling B2's 0.7528 should ultimately be read against.

### 2c. The honest statistical read (do not overclaim — critic T2, PATH_FORWARD §1)

At N = 1319 greedy trials, binomial SE ≈ **0.0119** (95% CI ≈ ±0.023). Therefore:

- **B2 vs dense:** Δ = −0.0008 ≈ **one flipped problem**, two-sample z ≈ **−0.05**. B2@50 is
  **statistically indistinguishable from dense** — "parity point-estimate reached," **not** "parity
  proven." This cuts both ways: it is equally *not* evidence B2 underperforms dense.
- **B2 vs the 0.7210 floor (the actual pre-registered success bar):** Δ = +0.0318 ≈ 42 problems,
  z ≈ **+1.86**. Clears the pre-registered point-estimate rule cleanly; as a difference-of-proportions
  it is ~1.9σ (marginal at p<0.05, but the rule asked only for the point estimate).
- **Single val point per run.** "best val@50" is the max of {0.7036@25, 0.7528@50}; there is no
  within-run neighbour to show 0.7528 is not a lucky validation draw.
- **The binding statistical fix** (critic, PATH_FORWARD R2): **one additional B2 seed @50 (~5 GPU-hr)**
  — higher decision-value than any single-seed extension. Not yet authorized.

**Bottom line of the ladder:** B2 is the unique point that is (a) at the top of the realistic ladder,
(b) statistically at parity with dense, (c) emission-free past the horizon that killed every rival,
and (d) attributable — the merger, not the substrate, delivers the +0.12 (C2 control). The blend
(B1, 0.7422) is a *close second that also converts* — see §3 for why the residual still wins.

---

## 3. The mechanism — why it works

The mechanism is now a *weight-space measurement*, not a hypothesis. It rests on the Step-A geometry
probe (20 steps, 7 post-warmup fires, all 196 targets, per-target sidecar) recomputed by the
mechanist and stress-tested by the critic.

### 3a. F1 — the codec error is ~92% of the compressed gradient's energy, near-orthogonal to the truth

At **identical (batch, θ)**, the probe measures `δ = G_anc_rep − G_comp_ring`, `m5_ratio = ‖δ‖/‖C‖`,
`m5_cos = cos(δ, C)` (where `C = G_comp_ring`). The settled values are **m5_cos ≈ −0.92 … −0.98**
and **m5_ratio ≈ 1.03 … 1.05**. From the exact algebra (`A = C + δ`):

```
‖A‖/‖C‖   = √(1 + r² + 2cr)            cos(A,C) = (1 + c·r) / √(1 + r² + 2cr)
```

with `c ≈ −0.95, r ≈ 1.05` this gives **‖G_anc_rep‖/‖G_comp‖ ≈ 0.29** (settled; 0.33 is the looser
early-fire figure) and **cos(G_anc_rep, G_comp) ≈ +0.007 pooled** (statistically ⊥; only **6.9%** of
the 196 matrices have |cos| > 0.2; min −1.000). This is **per-target uniform, not a median artifact**
(critic T5: frac(cos > −0.5) ≈ 0 at converged fires).

**Reading:** the compressed fast gradient is **dominated by orthogonal codec error** — the true
gradient is **~3.4× smaller in norm and statistically orthogonal**, carrying only `‖A‖²/‖C‖² ≈ 0.084`
≈ **8% of the compressed gradient's energy** (equivalently, the codec error is ~92%). This is the
**weight-space confirmation of EXP-26's activation-proxy 0.318** (the act-basis captures ~⅓ of update
energy; on the *valid* gradient in weight space the recoverable-true fraction is even smaller because
the off-principal residual EXP-26 left out lands here as pure error).

*Interpretive caveat (critic T5, flagged for honesty):* cos ≈ −1 with ratio ≈ 1 is *also* the
algebraic signature of `G_anc_rep` being small. The geometry "codec error dominates `C`" is
decision-grade and robust; the stronger claim "δ injects the *true dense* direction" rests on the
loss-mismatch relevance probe (≤ 0.0103 nats ≈ EXP-29's relevance band — so δ is codec error, not
objective mismatch), not on a direct cos(`G_anc_rep`, fresh-dense) measurement. The mechanism that B2
*at minimum* does is **cancel the dominant biased compressed gradient and step on the small valid
residual** — which is consistent with the modest, slow-converging val curve.

### 3b. m4/m1 — what survives the K-delay (the residual-over-blend *selector*)

- **m1** = cos(G_comp(t), G_anc_rep(t)), same tick, **different batches**: pooled median **+0.012**
  (≈ m2's old-M null 0.004). Cross-batch true-gradient cosine is **dead** — making the estimator
  *valid* (generator-matched) does **not** raise it. The limiter is *batch decorrelation*, not
  validity.
- **m4** = cos(G_comp(t), G_comp(t−j)), **same circuit, lagged**: medians j1 0.086, j2 0.200,
  j3 0.115, **j4 0.295, j5 0.169** — decidedly **nonzero** out to j=5.

These are different objects: m1 compares two independently-sampled gradients (dead); m4 compares the
compressed circuit to its own recent past. Because `C` is dominated by the **codec error** (F1) and
the codec error is **autocorrelated** (the act-basis `Q` drifts slowly), the *codec-error structure*
is what survives the K-delay — **not** any shared true-gradient signal. **m4-survives + m1-dead is the
exact signature that selects the residual over the blend.**

### 3c. Why the residual converts (and why the blend, though it also converts, wins less)

With λ=1 and the ring pair, the residual **telescopes exactly**: when the fast circuit is K-step
near-stationary (which m4's nonzero lag-autocorrelation makes approximately true),
`G_corr(t) ≈ G_anc_rep(t)` — the true gradient. **The residual is exact codec-error subtraction**: it
removes the ~92%-energy orthogonal artifact and leaves the true direction. This is K-delayed
telescoping error-feedback: the compressed step plus the dropped residual, re-injected one period
late.

A **blend** *adds* a scaled, near-orthogonal, ~0.29-norm partner (`G_anc_rep`) to `G_comp`; at no η
does adding an orthogonal partner *subtract* the dominant codec error. **The honest update (PROGRESS
14:47): the blend is NOT inert** — B1 reached **0.7422**, overturning the Step-A GATE-B1 prediction
once the blend was **magnitude-matched** (the gate's `cos ≥ 0.10` rule mis-predicted closure;
magnitude-matching the orthogonal partner still injects enough valid signal to convert most of the
way). So the mechanism statement is sharper than "blend dead, residual works": **both convert; the
residual converts *more completely* (0.7528 vs 0.7422) because subtraction removes the artifact
whereas addition only dilutes it.** The residual is the better operator; the blend is a viable
second.

### 3d. F3/m7 — the gradient is rank-~2; the defect is basis mismatch, not capacity

On the **valid** PG gradient: **stable rank ‖G‖²_F/‖G‖²₂ ≈ 1.8–2.05** (ambient 1536) and **top-1%
coordinate mass ≈ 0.58–0.61**. The replay gradient is *extremely* low-rank and concentrated. A
rank-77 codec has **abundant** capacity for a stable-rank-2 object — **the failure is basis MISMATCH**
(the act-basis `Q`, built from activation second moments, does not contain the ~2 gradient
directions), **not capacity**. This reframes GOAL-3: the live lever is compressing the (low-rank)
*residual* to ≪77 columns, **without** re-entering the falsified hybrid-Q corner (EXP-26 Step C).

*Why plain PowerSGD still trains decently despite F1* (mechanist, open but ranked): primary — **Adam's
per-coordinate normalization** on the concentrated top-1% true coordinates (a globally-orthogonal `C`
can still produce a per-coordinate step aligned with `A` where it matters); secondary — running-moment
cancellation of the rotating codec error. The cheap closing probe is top-1%-coordinate cos(A, C).

### 3e. The stability / carrier-law caveat (m6 ≈ 0.62 — REQUIRED statement)

Cross-fire autocorrelation of the valid anchor signal **m6 = cos(M_rep(t), M_rep(t−5)) ≈ 0.62** on
real cross-pair fires (0.59–0.75, drifting up). **β_anc=0 does NOT make the carrier memoryless** — the
valid policy gradient on a slowly-drifting policy is intrinsically persistent; β_anc=0 only stops
*compounding* it. As an AR(1): per-tick ρ ≈ 0.909 → autocorrelation time **τ ≈ 10.5 ticks ≈ 2× cadence**
— *marginal*, below EXP-27's compounding τ ≈ 19.5 but **not** memoryless.

Therefore: **all B2 stability claims are emission-judged and CENSORED** — at 50 steps in the verdict,
**de-censored to 100 steps for seed 0** by ext100 (the EXP-27 ~step-61 ignition band is cleared for
seed 0). The carrier law gives **no finite-horizon guarantee**: ext100 cannot prove stability past 100,
cannot speak to other seeds, and does **not** clear any small-β_anc EMA successor (which would inherit
the 0.62 base persistence and compound it — **BLOCKED** by PATH_FORWARD §4). The working discriminator
(F4, mechanist §5e), now with its first positive evidence from a clean ext100: **ignition needs a
persistent *exogenous* direction; the δ-residual is *endogenous*** (it cancels the circuit's own
artifact and re-injects the *current* objective's true gradient), which is why it does not pump length
the way EXP-27's stale-compounded EMA carrier did. This endogenous-vs-exogenous carrier law is the most
reusable mechanism finding the program has produced.

---

## 4. Moving parts + what must be stored (decentralized deployment)

This enumerates every component B2 requires on a real decentralized pipeline-parallel link, with byte
costs. Symbols: H = 1536 (hidden), r = 77 (rank), N = 196 (covered weight matrices = 28 layers × 7),
delay_K = 5, P = number of weight params ≈ 1.5e9, dtype bf16 = 2 B (fp32 buffers = 4 B).
Wiring: `CODE_WALKTHROUGH.md` §1–§4; code in `verl/workers/comm_eff/{powersgd_activation,anchor,
spectral_filter,state}.py`.

### 4a. Moving parts (the live circuit)

| # | part | role | where |
|---|---|---|---|
| 1 | **PowerSGD codec** (`Q` per boundary, `Y=hQ` / `ĥ=YQᵀ` hooks) | the only thing on the wire each step — projects boundary activations to rank r | `powersgd_activation.py` |
| 2 | **Anchor circuit** (clone-no-hook, staleness queue, full-coverage DP-reduced `G_anc`) | maintains the stale full-gradient reference + **owns `Q`** (`Q←orth(V)`, power iteration on stale-weight activations, broadcast DP-wide) | `anchor.py` + `transformer_impl.py` |
| 3 | **Paired-replay feed** (EXP-29: `replay_paired_batch`, fire-aware ring) | pairs stale weights with the trajectories they generated ⇒ `G_anc_rep` is a *valid* PG ⇒ δ is exact codec error | `anchor.py` |
| 4 | **The δ-merger** (`delayed_ef`, λ=1, β_anc=0) | computes `δ = G_anc_rep − G_comp_ring(t−K)` and writes `G_corr = G_comp + δ` to `p.grad` pre-AdamW | `spectral_filter.py` |
| 5 | **Fast-grad ring** (stores `G_comp_ring(t−K)`) | holds the compressed gradient from K ticks ago for the telescoping subtraction | `state.py` / anchor ring |

Ordering invariant: **anchor refresh → compressed fwd/bwd → FSDP all-reduce → merger → AdamW.**

### 4b. What must be stored (per replica / rank), with byte costs

| store | what | size (formula) | bytes (bf16 unless noted) | notes |
|---|---|---|---|---|
| **Q basis** | one H×r orthonormal matrix **per boundary** | H·r per boundary × #boundaries | H·r = 1536·77 ≈ 1.18e5 elts ≈ **0.24 MB/boundary** (fp32) | the codebook; broadcast DP-wide each refresh |
| **Anchor weight snapshots** | a **delay_K-stale queue** of full model-weight snapshots | (delay_K+1) × P | one ~3 GB bf16 clone is cached; a deep `delay_K` queue **multiplies** this → **~(K+1)·3 GB** if held naively | **CPU-resident** (`snapshot_device=cpu`); the dominant memory term, kept off-GPU |
| **Fast-grad ring** | stored compressed gradient(s) for the K-delay subtraction | ≤ 2 × (compressed grad ≈ N·H·r footprint) | small relative to a full gradient (rank-r, not full-H) | the ring depth is the telescoping window |
| **Paired-batch snapshots** | the trajectories/batches matched to each stale snapshot (replay) | ring of (batch tokens + logits) per retained fire | host-side, **not inter-stage**; bounded by ring retention | EXP-29 machinery; CPU |
| **Anchor gradient `M`** | full-coverage DP-reduced anchor gradient (here held, not EMA'd: β_anc=0) | P (full) | ~6 GB fp32 if materialized | **CPU-resident** (`ema_device=cpu`) — kept off-GPU by the standing OOM guard |
| **(merger `M`, if a blend variant)** | for B1/blend only — the M used as the added partner | P | — | **B2 stores no separate merger state** beyond `M`/the ring; `signed_ema`/EF buffers are inert (decay/clip = 0) |
| **Codec scratch** | `V` sketch, QR workspace | O(H·r) per boundary | negligible | transient per refresh |

**Per-step inter-stage wire cost (what crosses the PP boundary every tick):** only the rank-r
projection `Y = h·Q`. Measured: **`comm/bytes_compressed` ≈ 19.6–22.9 MB/step** against
**`comm/bytes_dense_equiv` ≈ 387–405 MB/step** ⇒ **`comm/bytes_ratio` ≈ 0.0505** every step (B2:
0.05037–0.05056), i.e. **~19.8× fast-path savings**. **B2 adds NO new traffic over the substrate** —
δ is built entirely from quantities the anchor already computes and transfers (the replay fire's
gradient + the ring's stored compressed gradient); the injection is local arithmetic at the consumer,
not a new collective.

### 4c. The honest-bytes caveat (REQUIRED — critic T6, PATH_FORWARD R1)

**`bytes_ratio = 0.0505` counts ONLY the fast compressed boundary traffic.** It does **not** count the
mandatory **anchor-circuit traffic**:

- the **DP all-reduce of the full-coverage `M`** (all 196 matrices, full-H, every cadence=5 ticks),
- the **broadcast of `Q`** to every DP rank each refresh.

These are full-H transfers, low-frequency (cadence-amortized) but real inter-stage cost that GOAL.md
counts. The program's standing estimate is **amortized comm ~4×, not ~20×** once the anchor is
included. So the **honest GOAL-3 savings number is NOT 19.8×** — it is the fast-path 0.0505 *plus* an
amortized anchor term, and the program owes a single anchor-inclusive figure (PATH_FORWARD R1 — a
**zero-GPU accounting pass**, the byte counters `comm/bytes_*` + `add_amortized_q_broadcast_bytes` are
already logged). **The 0.7528-at-19.8× headline is provisional on this accounting; the win survives
(savings are still material), but the bare 19.8× must be retired from comparison tables.** The
paired-batch and CPU snapshot stores are **host-side, not inter-stage**, so they do not enter the wire
budget — they are a *memory* cost (the (K+1)·3 GB snapshot queue is the dominant one, deliberately on
CPU).

---

## 5. Disposition

**B2 is the best communication-efficient GRPO solution this program has produced**, on the realistic
(anchor-circuit, no-clean-step) substrate, by every available measure: top of the realistic ladder
(0.7528), statistically at parity with dense (0.7536), emission-free de-censored to 100 steps (seed 0),
mechanism understood (cancels a measured 92%-energy near-orthogonal codec artifact), attributable
(C2 control: the merger delivers +0.12, the substrate does not), and zero added inter-stage traffic.

It is the canonical-launcher promotion candidate **conditional on** three honest, named follow-ups
that do not unseat the verdict but bound its claims:

1. **C1 dense-rerun @50 (PENDING; val@25 0.7566)** — the apples-to-apples ceiling on current code.
2. **C3 frozen-Q (PENDING)** — prices the Q-update (C2−C3) and isolates the adaptive basis's contribution.
3. **A second B2 seed @50 (~5 GPU-hr, not yet authorized)** — the binding statistical fix to turn
   "parity point-estimate reached" into "parity established."

Plus the **zero-GPU honest-bytes accounting** (R1) to replace the provisional 19.8× with the
anchor-inclusive GOAL-3 number. The blend (B1, 0.7422) is a genuine close-second converter — kept on
the record, not discarded — but the residual is the better operator.

*— synthesizer, team `commeff-grpo-verdict`, 2026-06-13*
