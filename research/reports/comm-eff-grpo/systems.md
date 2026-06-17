# Communication-Efficient PP/GRPO — Setup & What Works (Systems / Evidence)

**Author:** `systems` (comm-eff-grpo team). **Scope:** what the setup *is* and what
it has *empirically shown*, with receipts, covering **both** mergers (error-feedback
"B2" / `delayed_ef` **and** `signed_ema`).

**Ground-truth sources used** (all paths relative to `/Users/shamane/Documents/verl`):
`research/runs/SUMMARY.md`, `research/runs/FIXED_CONTROL_SURFACE.md`, `research/LOG.md`,
`research/.claude/GOAL.md`, `CODE_WALKTHROUGH.md`, the verl source under
`verl/workers/comm_eff/` and `verl/workers/config/comm_eff.py`, git log, and the
operator memory dir. Every numeric claim below is tied to a source row, code line,
or W&B run id. Places where the evidence is **thin / censored / legacy** are flagged
inline with **[THIN]**, **[CENSORED]**, **[LEGACY]**.

---

## 1. Architecture — precisely, each component tied to code

### 1.1 The mental model (two circuits)

The training path is split into two circuits (`CODE_WALKTHROUGH.md` §"Mental Model";
`GOAL.md` §"Where we are"):

1. **Fast circuit** — the normal actor forward/backward, with the PP-boundary
   gradient **compressed** by PowerSGD low-rank projection. This is the "swarm": many
   workers, each a **read-only consumer** of the shared projection basis `Q`.
2. **Anchor circuit** — a single, uncompressed, **no-optimizer** clone pass run at a
   fixed cadence. It produces a stale full-coverage gradient EMA `M`, **owns** the
   PowerSGD basis `Q`, and is the only thing that updates `Q`. This is the single
   "slow node".

The merger, if enabled, **rewrites selected fast gradients after backward and before
`optimizer_step()`** (`CODE_WALKTHROUGH.md` §"Mental Model"; the dispatch is
`spectral_filter.py:1159-1167`).

> **Async-realism constraint** (`GOAL.md` §"Async-realism constraint";
> memory `async-anchor-single-node-fast-swarm`): the anchor is ONE slow node serving a
> fast swarm over a network ⇒ it **always lags, never leads**. The fixed `delay_K=5`
> lock-step in the current code is a *simulation* of that lag. Admissible methods use
> the anchor as a **lagging** reference, tolerate **variable staleness**, and stay
> **cross-rank-identical**. This rules out delay-compensation / anchor-lead levers and
> makes the two-circuit structure **mandatory** (the practical-future-use point).

### 1.2 Component → file:line map

| Component | What it does | Source (file:line) |
|---|---|---|
| Hydra dataclass + validation | Schema for mask / anchor / spectral merger / PowerSGD / capture / probe | `verl/workers/config/comm_eff.py` |
| Per-worker state | counters, path tags, replay/ring buffers, compressor/filter construction, merger selection | `verl/workers/comm_eff/state.py` (merger wired at `state.py:567-639`) |
| PowerSGD boundary projection | boundary projection hooks, basis bootstrap/update, anchor-owned `Q`, byte counters | `verl/workers/comm_eff/powersgd_activation.py` |
| Spectral merger (BOTH mergers live here) | per-target anchor EMA + modes `delayed_ef` / `ef_powersgd` / `signed_ema` / `inject` / `blend` / `none` | `verl/workers/comm_eff/spectral_filter.py` |
| Anchor circuit | staleness queue, paired-replay ring, snapshot/canary, gradient extraction, DP-reduce, geometry probe | `verl/workers/comm_eff/anchor.py` (DP-mean reduce `_dp_all_reduce_anchor_grads`, anchor.py:934) |
| Backend-neutral hook points | `train_batch()` hook points | `verl/workers/engine/base.py` |
| FSDP integration | anchor refresh, G_dense capture, geometry probe, grad-correction writeback | `verl/workers/engine/fsdp/transformer_impl.py` (mode dispatch guard `transformer_impl.py:2859`) |
| Actor update wrapper | path tags, PowerSGD lifecycle, metrics, end-of-step basis update | `verl/workers/engine_workers.py` |
| Optional fp32 capture | tensor dumps keyed by `(global_step, optimizer_tick, role, target)` (diagnostic-only) | `verl/workers/comm_eff/capture.py` |

### 1.3 Train-step flow (`CODE_WALKTHROUGH.md` §"Train Step Flow")

1. `BaseEngine.train_batch()` zeroes grads.
2. `_maybe_comm_eff_anchor_refresh()` may run first: load a stale/paired-replay
   snapshot into an **isolated clone**, run the clean anchor backward, read **raw**
   anchor grads, **DP-reduce (MEAN)**, update `M`, optionally update + broadcast `Q`.
3. `forward_backward_batch()` runs the fast path; PowerSGD hooks are registered
   **only** for the train path and unregistered in `finally`.
4. *(diagnostic-only)* optional `G_dense` capture on a no-hook clone.
5. `_maybe_comm_eff_grad_correction()` may apply the merger to selected full logical
   2D gradients.
6. *(diagnostic-only)* geometry probe stages fast gradients, emits telemetry.
7. `optimizer_step()` consumes the final gradients.

### 1.4 The codec — PowerSGD, r=77, activation basis

- **What is compressed:** the PP-boundary **activation gradient** is projected to a
  rank-`r` subspace defined by `Q` (`CODE_WALKTHROUGH.md` §"Main Files";
  `powersgd_activation.py`). Per boundary, per forward, the swarm sends `N·r`
  coordinates (`Y = M@Q`) instead of `N·H` (`powersgd_activation.py:315-319`).
- **r = 77** is byte-matched to the legacy PRF mask at `p=0.95` for **H=1536**
  (Qwen2.5-1.5B): `0.05·1536 ≈ 77` (`FIXED_CONTROL_SURFACE.md` ☆ table;
  memory `qwen25-1p5b-hidden-size-1536` — note `powersgd_activation.py:25` still
  carries an `r=102≡p=0.95` comment that assumed H=2048; that comment is **stale**, the
  config value is 77).
- **`Q` is shared & DP-synced.** `Q` is one codebook per boundary, identical across DP
  ranks (memory `powersgd-basis-must-sync-across-dp`). With `anchor.owns_q=true` the
  fast path's basis update is **fail-closed**; `Q` is harvested from uncompressed anchor
  activations (`Q ← orth(V)`) and broadcast to DP ranks
  (`CODE_WALKTHROUGH.md` §"Q Ownership").

### 1.5 The anchor `M` and paired replay (the validity machinery)

- `M` is a per-matrix EMA of the anchor model's gradient over **all 196 targeted 2D
  decoder projection matrices** (28 layers × 7), DP-mean-reduced
  (memory `anchor-gradient-ema-beta0-grpo`; `CODE_WALKTHROUGH.md` §"Invariants";
  `max_targets=-1` = full coverage, `CODE_WALKTHROUGH.md` invariant). At the default
  `β_anc=0`, `M` = the instantaneous (most recent fire's) anchor gradient — no averaging.
- **Paired replay (the validity boundary).** `replay_paired_batch=true` +
  `snapshot_device=cpu` pair the anchor's `delay_K`-stale weights with the **batch those
  same weights generated**, so `M_rep` aligns with the fast gradient at the **same
  `(batch, θ)`** (`CODE_WALKTHROUGH.md` §"Invariants"; `SUMMARY.md` §"Evidence
  Boundary"; merged in **PR #16 / `d26176b44`**, memory `canonical-anchor-comm-eff-base`
  EXP-29 entry). This **removes** the standing *stale-weights × current-batch* confound.

> **EVIDENCE BOUNDARY (load-bearing):** only **valid-M** measurements count — `M_rep`
> paired with the retained fast gradient at the same `(batch, θ)`. **Pre-#29 and
> clean-step numbers are history, not current evidence** (`SUMMARY.md` §"Evidence
> Boundary"). This is the single most important caveat in this report; it is what makes
> the signed_ema number question (§3) non-trivial.

### 1.6 The two mergers (both implemented; one promoted)

Both live in `spectral_filter.py` and are selected by `spectral.correction_mode`
(dispatch `spectral_filter.py:1159-1167`):

**(A) `delayed_ef` = "B2" — the CONFIRMED SOTA** (`spectral_filter.py:838` /
`delayed_ef_matrix`):

```text
δ(t)      = M_rep(t) − G_comp_ring(t − K)     # codec error on IDENTICAL (batch, θ)
G_corr(t) = G_comp(t) + λ·δ                    # δ refreshed at fires, HELD between
```

- This is **error-feedback on the PowerSGD codec residual**: δ is the *exact* gradient
  the codec dropped at tick `t−K`, recovered from the retained fast gradient ring and
  the replayed anchor gradient (`spectral_filter.py:843-844, 920-929`).
- **λ=1, β_anc=0, cadence=delay_K=5, clean_cadence=0** (`SUMMARY.md` "Canonical B2
  Settings"; `FIXED_CONTROL_SURFACE.md` ☆ substrate table).
- **λ=0 is a bitwise identity** → plain PowerSGD (`spectral_filter.py:894-896`), which
  is why "no-merger" and B2 differ in *exactly one knob*.

**(B) `signed_ema` — candidate/legacy, NOT promoted** (`spectral_filter.py:393` /
`signed_ema_matrix`):

```text
G_corr = α·G_noisy + (1−α)·|G_noisy|·sign(M)
```

- Keep the **magnitude** of the compressed gradient `G_noisy`; take the **sign** from
  the stale anchor EMA `M` (`spectral_filter.py:394-398, 430`). `α=0` = pure sign;
  `α=1` returns `G_noisy`. `α=0.5` is the only setting worth tracking
  (`SUMMARY.md` "Parameters Tested" row; memory `no-merger-floor-0p63-not-0p74`).
- Has a **cold-M fallback**: when `M` is unwarmed, returns `G_noisy` unchanged rather
  than silently zeroing the gradient (`spectral_filter.py:400-422`).
- **As of commit `421567ec6` signed_ema is purged from the forward/launch surface**:
  the dataclass + `actor.yaml` default merger is `none`, the canonical launcher defaults
  to `delayed_ef`. signed_ema "remains a supported-but-unused mode in the merger code +
  its tests" (commit `421567ec6` body). I verified the dispatch is still wired
  (`spectral_filter.py:1164-1165`, `correction_mode=="signed_ema"` → `signed_ema_matrix`),
  so `correction_mode=signed_ema` is still runnable — it is simply not the default and
  not on any launcher.

---

## 2. Authoritative results table (val@50)

All numbers are **GSM8K greedy mean@1, deterministic (seed 0), val@50** unless noted.
"Same-config" = PowerSGD r=77 activation basis, anchor owns `Q`, cadence=delay_K=5,
β_anc=0, the locked control surface (`FIXED_CONTROL_SURFACE.md`). W&B entity/project for
all run ids below = **`shamanework-pl/verl_compression_research`**
(memory `exp31-anchor-usage-tournament-stop:41`).

| Method | val@50 | Comm (bytes ratio) | Source / receipt | Validity |
|---|---|---|---|---|
| **dense** (method OFF) | **0.75–0.78** band | 1.0 (uncompressed) | `SUMMARY.md` row; `FIXED_CONTROL_SURFACE.md` ☆ note: current-code apples-to-apples rerun **`73ntu76u` = 0.7839**, old-code **`5e2jpho9` = 0.7536** | reference band; run-variance-dominated (±0.024/draw) |
| **B2 `delayed_ef`** (λ=1, β=0) | **0.7528** (first valid-M proof); reproduced **0.735–0.754** | **~0.0505** (band 0.0504–0.0506) | `SUMMARY.md` "Current Best Method" row 1 + "Canonical B2 Settings"; `LOG.md` §Current State; W&B `fy920fty` (EXP-31 B2_live ref) | **valid-M** ✓ confirmed SOTA |
| `signed_ema` α=0.5 | **0.7271** (valid-M, EXP-32) — DOMINATED | ~0.0505 | memory `anchor-gradient-ema-beta0-grpo:14` + `MEMORY.md:6`; commit `b29191c72` ("post-EXP-32 / #29 findings") | **valid-M** ✓ but **[THIN]** single draw, see §3 |
| `signed_ema` α=0.5 | 0.7066 | ~0.0505 | memory `no-merger-floor-0p63-not-0p74:14` | **[LEGACY]** invalid-M (EXP-25 circuit) — do NOT cite as current |
| **no-merger** (PowerSGD+Q, `correction_mode=none`) | **0.6300** | ~0.0505 | `SUMMARY.md` "control" row; memory `no-merger-floor-0p63-not-0p74:12`; W&B **`k6nmcuyd`** | **valid-M** ✓ realistic floor |

**Interpretation** (`SUMMARY.md` §"Interpretation"; `GOAL.md`): B2 reaches **dense
parity within single-draw eval noise at ~5% fast-path gradient communication**. The
merger's measured value is **+0.123** (0.7528 − 0.6300 = the gap the codec opened, which
EF closes). **No tested anchor-usage or β lever gives a credible dense surpass** (§ closed
frontier, see strategist's section).

**Ordering (the headline):** `no-merger 0.6300  <  signed_ema 0.7271  <  B2 0.7528 ≈
dense 0.75–0.78`. signed_ema clears the floor by ~+0.10 but caps ~0.026 **below** B2.

### Caveat on the dense band

The dense "ceiling" is **run-variance-dominated** — report it as a band **≈ 0.75–0.78**,
not a point (rollout nondeterminism ≈ ±0.024/draw even at seed 0;
`FIXED_CONTROL_SURFACE.md` ☆ note, corrected 2026-06-13). The current-code rerun
(`73ntu76u` = 0.7839) confirms the valid-M merges did **not** regress dense. Because
B2's reproduction band (0.735–0.754) overlaps the bottom of the dense band, "parity" is
a **noise-bounded** statement, not B2 ≥ dense pointwise. Compare any comm-eff cell only
to a dense run sharing its **code + hyperparameters**.

---

## 3. CRITICAL EVIDENCE GAP — signed_ema's real post-#29 VALID-M number

**Question (from the task):** is signed_ema's ~0.70 a valid-M number, or
legacy / EXP-32 config-only / pre-#29?

**Answer (honest, with receipts): there are TWO distinct signed_ema numbers, and the
valid-M one is 0.7271, not 0.7066.**

- **0.7066 = [LEGACY] invalid-M.** Memory `no-merger-floor-0p63-not-0p74:14` states it
  verbatim: *"signed_ema α=0.5 = 0.7066 (EXP-25, **invalid-M** circuit)"*. Line 19 of
  the same file: *"signed_ema was **never run on the #29 corrected (valid-M) circuit**"*
  — as of when that memory was written (it then opened issue #32 to fix exactly this).
  **Do not cite 0.7066 as a current number.**
- **0.7271 = the post-#29 VALID-M number.** Issue #32 / EXP-32 **did subsequently run**.
  Two independent receipts:
  - memory `anchor-gradient-ema-beta0-grpo:14`: *"EXP-32 signed_ema α0.5=**0.7271**<B2
    0.7528 (dominated)"*, and line 18 defines *"valid-M (#29/EXP-32 circuit) = M with
    corrected full coverage + scale + cross-rank-identical"* — i.e. EXP-32 explicitly ran
    on the valid-M circuit.
  - `MEMORY.md:6` repeats it: *"EXP-32 M-merger 0.7271<B2"*.
  - git commit **`b29191c72`** ("reports: refresh HTML for **post-EXP-32 / #29**
    findings (**sign method works**; error feedback still default; β_anc=0)") confirms
    EXP-32 happened and its verdict: the sign method *works* (clears the floor) but EF
    stays the default (B2 dominates).
  - commit **`421567ec6`** then **purged signed_ema from the forward path** — consistent
    with "ran it on the valid circuit, confirmed dominated, retired it from the launcher."

**So the valid-M signed_ema number IS 0.7271**, and it is **DOMINATED** by B2 (0.7528).

**LOUD FLAG — SUMMARY.md undersells this.** `SUMMARY.md:19` says signed_ema has *"no
durable post-#29 verdict that promoted it"*. That is **technically true but misleading by
omission**: a post-#29 **valid-M** number exists (0.7271); it simply wasn't *promoted*
because it lost to B2. SUMMARY.md never prints 0.7271 anywhere. A reader of SUMMARY.md
alone would wrongly conclude signed_ema was never validly measured post-#29. **The
report should state 0.7271 explicitly and label it valid-M-but-dominated.** *(I have not
been able to recover an EXP-32 W&B run id — the run dir was de-bloated and SUMMARY.md
omits it; 0.7271 rests on the two memory entries + the two commits above. I rate the
number **medium-high confidence, single-draw [THIN]**, and flag the missing W&B id as an
open provenance gap.)*

---

## 4. signed_ema instability vs B2 stability

**B2 (`delayed_ef`) is stable on the evidence we have; signed_ema is unstable across α —
including at its best α — and that instability is treated as STRUCTURAL, not a tuning
artifact.**

### signed_ema instability — the evidence

- **α=0 (pure sign) is a documented policy-collapse spiral** (`EXP-25`,
  memory `entropy-collapse-alpha0-signed-ema`): entropy decayed monotonically
  5.69→0.06 over 48 steps, `response_length/mean` exploded ~step 30 (300→~8600 tok, near
  the 16384 cap), `critic/score/mean` peaked 0.787@28 then **degraded to 0.318@45**.
  Mechanism: at α=0 the merger is **magnitude-preserving sign-SGD with persistent
  stale-anchor signs** — no cross-batch sign cancellation ⇒ full-magnitude steps in a
  fixed stale direction ⇒ sharpening spiral, with no-KL/no-entropy to arrest it.
- **~50% sign-disagreement with dense is STRUCTURAL, not staleness** (memory
  `exp25-collapse-gradient-flow`; `no-merger-floor-0p63-not-0p74:19`: the sign from `M`
  is a measured ~coin-flip on valid `M`, cos≈0.012). The sign term carries an
  **irreducible** disagreement that doesn't shrink with fresher `M`.
- **α=0.5's 50-step "survival" was CENSORED** (memory
  `entropy-collapse-alpha0-signed-ema` 2026-06-11 correction;
  `canonical-anchor-comm-eff-base` EXP-27 post-mortem): it had **consecutive 16384
  cap-pins at steps 47–48** and a `len/mean` slope of **+5.92/step** (≈60× the
  pre-ignition slope of a sibling run). The comparator estimates **P(ignites by step
  100) ≈ 55–70%**. **[CENSORED]** — so even at the best α, the 0.7271 number is a
  50-step observation of a run that was *already spiraling*; its longer-horizon stability
  is **unproven**.

### B2 stability — the evidence and its limit

- B2 ran clean through the runs that produced 0.7528 and the 0.735–0.754 reproductions;
  EXP-31 reports "no ignition/OOM/divergence" across the B2 baseline and all four levers
  (memory `exp31-anchor-usage-tournament-stop`, process criteria PASS).
- **Mechanism contrast:** `delayed_ef` is **direction-preserving** — it *adds* the
  dropped off-subspace residual `λ·δ` and never replaces signs, so it does not introduce
  the sign-SGD sharpening dynamic. `ef_powersgd` (a direction-preserving sibling of B2,
  also no sign term) confirmed the same: cos(G_comp, G_corr)=0.956 vs signed_ema's 0.717
  (memory `canonical-anchor-comm-eff-base` EXP-26). The cross-run discriminator for the
  spiral is **merger-carrier presence**, and within that, **sign-replacement** is the
  sharp mover.
- **[CENSORED] caveat for B2 too:** the EXP-27 post-mortem notes *all* 50-step "stable"
  claims are censored observations; dense@100 itself is unproven (trained exactly 50
  steps, still climbing). GOAL.md criterion 1 ("stable") is met at the 50-step horizon
  the project actually runs; a 100-step stability proof for B2 is **not yet on record**.
  The asymmetry that matters: signed_ema shows *active* pre-ignition precursors at 50
  steps, B2 does not.

### Structural vs tuning — the agreed framing

I treat signed_ema instability as **structural** (a property of sign-replacement under
no-KL/no-entropy GRPO with a stale anchor), not a tuning artifact, on three grounds:
(1) it spans α (collapse at α=0, censored-unstable at α=0.5 — the whole tested range);
(2) the sign-disagreement is a measured ~coin-flip on *valid* `M`, so fresher `M` does
not fix it; (3) the direction-preserving mergers (B2, ef_powersgd) on the *same
substrate* do not spiral, isolating the cause to the sign term. **This is the question I
am settling with `theorist`; their reply is folded in at §6.** *(If theorist dissents,
this paragraph is the one to revise.)*

---

## 5. Savings, concretely

- **Bytes ratio ≈ 0.0505** (measured, band **0.0504–0.0506**), **identical across all
  compressed arms** (no-merger, B2, signed_ema) because the **codec is identical** and
  only the merger differs (`SUMMARY.md` "Canonical B2 Settings"; `LOG.md`;
  memory `exp31-anchor-usage-tournament-stop` "bytes_ratio==B2 ∈[0.0504,0.0506]").
- **Where the number comes from:** per PP boundary, the swarm sends `N·r` element-coords
  (`Y = M@Q`) instead of `N·H` (`powersgd_activation.py:315-319`), so the ratio ≈
  `r/H = 77/1536 ≈ 0.0501`, plus the **amortized `Q`-broadcast** bytes
  (`add_amortized_q_broadcast_bytes`, `powersgd_activation.py:343`) which lift the
  measured ratio to ~0.0505. "bytes" counts elements / fp-count; the **ratio** is what is
  reported (`powersgd_activation.py:319`). ≈ **19.8× reduction** on the compressed traffic
  (memory `canonical-anchor-comm-eff-base`).
- **What IS compressed:** the **PP-boundary activation gradient** of the fast/training
  path (PowerSGD r=77, activation basis).
- **What is NOT compressed:**
  - the **DP axis** — only the PP boundary is projected; the DP gradient all-reduce is
    full-precision (memory `powersgd-basis-must-sync-across-dp`: "the DP axis is NOT
    compressed, only the PP boundary is").
  - the **rollouts / generation** — produced by ordinary non-PP verl + vLLM,
    **out of scope** for compression (`GOAL.md` §"The goal").
  - the **anchor circuit's own pass** is uncompressed by construction (it is the
    full-coverage reference), but it is **low-frequency** (1 slow node, every 5 ticks),
    which is what keeps it amortizable — the amortized-comm cost of the realistic anchor
    is ~4× the per-step compressed traffic, not the ~20× a periodic *clean step* would
    cost (memory `clean-step-realism-confound`).
- **Caveat — the clean-step confound is retired.** Earlier "savings" numbers that
  counted a periodic full-rank `clean_cadence` step are **not** the current regime; the
  anchor replaced the clean step (`clean_cadence=0`), and the ~5% ratio above is the
  realistic, anchor-substrate number (`SUMMARY.md`; memory
  `no-merger-floor-0p63-not-0p74`, `clean-step-realism-confound`).

**Goals scorecard** (`GOAL.md` §"Done means"): **(1) Stable** — met at 50 steps for B2
(signed_ema NOT, see §4); **(2) Parity** — met (B2 ≈ dense within noise at ~5% comm);
**(3) Savings** — met (~0.0505 bytes ratio, measured); **(4) Reproducible** — one
canonical launcher exists (`vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh`) but GOAL.md
holds Goal 4 "pending a surpass". B2 is the reference; no surpass found.

---

## 6. Cross-examination (theorist, strategist) — folding in

**Status at first write:** cross-exam messages sent to `theorist` (does the bias/variance
math predict floor < signed_ema < B2 ≈ dense, and is signed_ema instability structural?)
and `strategist` (closed null-lever list + savings). Replies pending; this section will
be updated. If a teammate is unresponsive after my own work is done, the open question is
noted here and I proceed (per task instruction).

- **[PENDING — theorist]** Does the bias/variance decomposition predict BOTH the EF
  parity jump (0.6300→0.7528) AND signed_ema's shortfall (0.7271)? Proposed shared
  mechanism: EF returns the dropped codec energy **asymptotically unbiasedly** (it adds
  the exact residual δ), while signed_ema injects a **non-vanishing sign-bias** (keeps
  compressed magnitude, borrows a ~coin-flip stale sign). *To be confirmed/corrected.*
- **[PENDING — theorist]** STRUCTURAL vs tuning for signed_ema instability — §4 asserts
  structural; needs theorist's sign-off for ONE agreed answer.
- **[PENDING — strategist]** Acknowledgement of the closed null-lever list so it is not
  resurrected; their surpass thesis.

---

## 7. Honest-evidence flags (collected)

- **[LEGACY]** signed_ema 0.7066 is the *invalid-M* EXP-25 number; the valid-M number is
  **0.7271** (§3). SUMMARY.md prints neither and says "no durable post-#29 verdict" —
  **undersells** the existence of a valid-M measurement.
- **[THIN]** 0.7271 is single-draw and its **W&B run id was not recoverable** (run dir
  de-bloated; SUMMARY.md omits it). Rests on two memory entries + commits `b29191c72`,
  `421567ec6`. Medium-high confidence.
- **[CENSORED]** signed_ema α=0.5's 50-step stability — consecutive cap-pins at 47–48,
  P(ignite by 100) ≈ 55–70%. Even the best-α number is from a run already spiraling.
- **[CENSORED]** All 50-step "stable" claims (incl. B2 and dense) are censored at the
  50-step horizon; no 100-step stability proof for B2 is on record. The asymmetry: B2
  shows no pre-ignition precursors, signed_ema does.
- **Stale pointer:** `GOAL.md:52` cites `runs/EXP-31/B2_baseline/resolved_params_B2.txt`;
  that path no longer exists (runs dir de-bloated). Authoritative config source is now
  `SUMMARY.md` + the b2_sota launcher. *(Flagging as a doc-drift nit; not load-bearing
  for any number.)*
- **Stale comment:** `powersgd_activation.py:25` says `r=102≡p=0.95` (assumes H=2048);
  the real config is r=77 for H=1536. Comment only; the value in use is correct.
- **Dense parity is noise-bounded:** B2's band (0.735–0.754) overlaps the bottom of the
  dense band (0.75–0.78); "parity" means within ±0.024/draw, not B2 ≥ dense pointwise.
