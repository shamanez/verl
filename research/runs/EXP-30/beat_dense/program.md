# Beat-Dense Program — a ranked, executable path from B2 to SURPASS

> **What this is.** A ranked, pre-registered, executable program whose objective is to **surpass the
> dense control** starting from the current best comm-efficient GRPO solution (**B2**). It is written
> to become the *plan* section of a hand-off GitHub issue: the operator picks it up in a new session,
> the orchestrator files the issue. **This file does not create an issue and authorizes nothing** — no
> threshold below softens a pre-registered rule; no cell launches without the operator approving it.
>
> **Author:** bd-strategist (team `beat-dense-grpo`, task #2) · 2026-06-13
> **Ordering doctrine (the one thing that makes this different from the path-forward docs):**
> *Parity must be **established** before "surpass" is a meaningful word.* The current B2−dense gap is
> **−0.031 ≈ 1.3 nondeterminism-σ** at a single seed — neither parity nor a deficit is established.
> So **STEP 0 = seed-replicate BOTH B2 and dense** (apples-to-apples, operator-sanctioned). Everything
> labeled "beat dense" is gated behind STEP 0 returning two pinned, separated (or overlapping) bands.
>
> **Sources (quoted, not re-derived):** `runs/EXP-30/verdict.md` (canonical record incl. TL;DR, findings
> F1–F5, ext100 + Bars-correction addendum — the intermediate team syntheses PATH_FORWARD/pathforward/
> final_synthesis were consolidated into verdict.md + this file and deleted 2026-06-13, see git history),
> `runs/EXP-30/resolved_params_B2.txt`, `runs/EXP-30/train_dense_rerun.log` (dense@50 = 0.7839 confirmed
> from the log), `runs/EXP-30/metrics/{stepA_fires.jsonl, B2_delta_per_fire.jsonl,
> ext100_delta_per_fire.jsonl}`, `.claude/GOAL.md`. The deep theoretical feasibility of each surpass
> mechanism is the companion file `runs/EXP-30/beat_dense/feasibility.md` (bd-theorist, task #1) —
> cited inline as **[feas]**, not duplicated here.

---

## Current situation (≤25 lines — the issue may quote this verbatim)

- **Best solution = B2:** K-delayed **exact codec-residual** correction. `G_corr(t)=G_comp(t)+λ·δ`,
  `δ=G_anc_rep(t)−G_comp_ring(t−K)`, **λ=1, β_anc=0**, on the EXP-29 paired-replay substrate
  (PowerSGD **r=77** act-basis `Q`, **anchor owns Q** updated by power iteration, **cadence=delay_K=5
  optimizer ticks**, **clean_cadence=0**, `replay_paired_batch=true`, `snapshot_device=cpu`). Ground
  truth: `resolved_params_B2.txt`.
- **Dense baseline — CORRECTED 2026-06-13.** Same-code, same-config dense rerun (`exp30_dense_rerun`,
  `73ntu76u`) = **val@50 0.7839** (val@25 0.7567). The old `5e2jpho9` 0.7536 was **old code**. Dense is
  best read as a **band ≈ 0.75–0.78** (rollout nondeterminism ≈ **±0.024/draw**, measured).
- **B2 vs same-config dense: 0.7528 is −0.031 (≈96% of dense).** That gap ≈ **1.3 nondeterminism-σ** —
  **near-parity, NOT established** (also: dense led at the 25-step checkpoint, 0.7567 vs B2 0.7036/0.7278).
  At binomial SE 0.0119 (N=1319 greedy mean@1) B2 is indistinguishable from dense **and** only ~1.9σ
  above the 0.7210 floor. Parity is unresolved **at n=1 seed** — this is the binding fact.
- **Controls (one-knob, same config):** **C2** plain-PowerSGD+Q-updated **no merge** = **0.6300** (the
  clean single-knob floor → **merge-value +0.123**); **B1** magnitude-matched blend = **0.7422** (blend
  is *dominated, not inert*); **C3** frozen-Q **PENDING** (prices the Q-update = C2−C3).
- **Stability:** emission-free **through 100 steps, seed 0** (ext100); val 0.7536@50 → 0.7475@75 →
  0.7400@100 (mild decay). **All stability is CENSORED** (EXP-27 ignited at ~61; this cleared that band
  for seed 0 only). Carrier: **m6≈0.62** ⇒ τ≈10.5 ticks ≈ 2× cadence — marginal, β_anc=0 stops
  compounding not persistence.
- **Mechanism:** codec error ≈ **92%** of the compressed-gradient energy, **⊥** the true gradient (F1,
  cos≈+0.007); the residual *subtracts* it (λ=1 telescopes). Gradient **stable-rank ≈ 2** (m7) ⇒ r=77 is
  basis-**mismatched**, not capacity-limited.
- **What's stored (decentralized):** per-boundary `Q` (H·r, fp32), a `delay_K`-stale **CPU** weight-snapshot
  queue (~(K+1)·3 GB, the dominant *memory* term), full-coverage anchor `M` (**CPU**), a fast-grad ring
  for the telescoping subtraction, and the CPU paired-batch replay ring. Branch `exp/30-valid-m-geometry`
  / run dir `runs/EXP-30/` (B2 launcher `launch_B2.sh`, bundle `exp.bundle`).
- **Honest-bytes caveat:** the headline `bytes_ratio=0.0505` (~19.8×) is **fast-path only** — it omits the
  mandatory anchor `M` DP-all-reduce + `Q`-broadcast. The honest amortized inter-stage number is the
  program's standing **~4×**, not 20×. B2 adds **zero new traffic** over the substrate. The savings claim
  is provisional on a (zero-GPU) accounting pass.

---

## 1. The two unmet halves of "done", and why this program is ordered the way it is

GOAL.md §"Done" = {1 Stable, 2 Parity, 3 Savings, 4 Reproducible}. B2's status: Stable (censored@100/seed0),
Parity (**point-estimate near, statistically unestablished**), Savings (fast-path number only), Reproducible
(one seed, no `examples/` promotion). The operator's objective for *this* team is sharper than "close done":
**surpass dense.** That reorders the work, because:

1. **You cannot claim "surpass" against a number you have not established.** The path-forward docs
   (the prior forward-check syntheses, now folded into verdict.md) correctly rank the *can't-be-wasted* measurements
   (honest-bytes, Q-telemetry) at the top **for the "close done" objective**. For the **surpass**
   objective those are necessary but not sufficient: a surpass claim is a *difference of two bands*, and
   right now **neither band is pinned** (dense is two single draws 0.7567/0.7839 spanning a 0.024-σ band;
   B2 is one draw 0.7528). **STEP 0 pins both bands.** It is cheap, it is operator-sanctioned (an
   apples-to-apples dense control *alongside a sweep* is explicitly sanctioned even though dense-for-
   production is forbidden), and **nothing downstream is interpretable without it.**

2. **The surpass surface is narrow, and the theorist [feas] prices it sub-even — so the ranking leads
   with the most-likely *real* win, not the most-aspirational.** [feas §7] puts **P(greedy surpass,
   seed-replicated, beyond noise) ≈ 10%**, with the expected outcome being **seed-replicated parity plus
   a likely secondary pass@k edge** — itself a complete win against GOAL (criterion 2 asks for *≥ dense
   within noise*, which parity satisfies). The structural reason ([feas §3c]): B2's mechanism is
   *subtractive reconstruction of the dense gradient* — `G_corr ≈ G_anc_rep` — and **a faithful copy of
   dense cannot beat dense**; surpass requires injecting something dense *lacks* on the *greedy* bar.
   That sorts the candidates:
   - **Additive off-principal sub-basis** — the **only route with double-digit greedy-surpass odds**
     ([feas §5]: P(surpass | run) ≈ 12–18%). It attacks the *root cause* (m7 basis-mismatch: the rank-~2
     true direction lies *outside* the act-basis), and B2 proves those directions are recoverable+helpful.
     Even so, its own ceiling is "clean parity unless a compression-specific conversion effect materializes
     on the greedy bar" — so it is a **gated, sub-even bet, not the headline.**
   - **Eval-diversity / pass@k / temperature-n** — the **most likely actual "win" (~30–40%, [feas §6])**,
     but a *secondary, measurement-frame* win on a **more-lenient bar**, explicitly **not** the greedy GOAL
     bar. The measured diversity edge is real (uncompressed-generator `rollout_ppl` 1.40 vs 1.24); pass@k
     is where it would show. Must carry a **dense×{T,n} control** or it proves nothing.
   - **Residual-compression (r_δ≪77)** — a **GOAL-3 *savings* surpass** (parity val at strictly-better
     honest savings), **NOT a val-surpass** ([feas §4c]: truncation error is biased, expected val effect
     ≤ 0). Filing it under "beat dense (val)" would be a category error; it *is* a legitimate "better than
     dense" on the savings axis (dense trains at 1×).
   - **Shorter delay_K** — the cheapest **parity-defense**: it reduces the K-tick staleness drift that
     [feas §3c] identifies as putting B2 *below* dense at horizon (the 0.7536@50 → 0.7400@100 slide). It
     **dominates partial-λ** ([feas §4a]) and is the most likely single knob to close the −0.031 gap toward
     *true* parity — a parity promise, not a surpass one.
   - **Partial residual λ<1** — **demoted from a surpass bet to (at most) one damping-robustness point.**
     [feas §4a] is decisive: λ<1 re-admits the **biased, near-orthogonal codec artifact** (not zero-mean
     exploration noise — the "compression-noise-as-regularizer" class §3 lists as BLOCKED), so it "loses
     less slowly" and is *dominated by shorter-K*. λ>1 is **forbidden** (over-subtraction → carrier → ignition).

So the ranking is: **STEP 0 (parity, gating) → STEP 1 the cheap on-mechanism parity-defense → STEP 2/3
the savings surpass and the most-likely (secondary) pass@k win → STEP 4 the one gated ~10–18%
greedy-surpass bet → STEP 5 the optional damping point** — i.e. the highest *real* decision-value per
GPU-hr first, the aspirational greedy-surpass bet placed where its odds and cost put it. The two zero-GPU
can't-be-wasted items (honest-bytes, Q-telemetry) ride **alongside** STEP 0 (they need no GPU and gate
the expensive surpass cells), exactly as the path-forward docs rank them — I keep them, I just stop
calling them the headline, because they cannot themselves produce a surpass.

---

## STEP 0 — ESTABLISH PARITY (gating; cheap; nothing labeled "surpass" is meaningful until this returns)

**The whole program is gated here.** Two parallel tracks; both are operator-sanctioned and neither can
be wasted.

### 0A — Seed replicates of BOTH B2 and dense, @50 (the binding statistical fix)

- **Hypothesis (falsifiable, numeric).** Across **≥3 seeds each**, the B2 band mean and the same-config
  dense band mean differ by **≤ the pooled SE** (parity established) OR the dense band sits **> 2 pooled-SE
  above** B2 (a real deficit — surpass is then a long shot and the program re-scopes). Pre-register the
  read: with 3 seeds each and per-draw σ≈0.024, the SE of each mean ≈ 0.024/√3 ≈ **0.0139**, pooled
  two-sample SE ≈ **0.020**. **Decision bands:** |mean_B2 − mean_dense| ≤ 0.020 ⇒ **PARITY ESTABLISHED**;
  mean_dense − mean_B2 > 0.040 ⇒ **DEFICIT** (re-scope, surpass unlikely); in between ⇒ **inconclusive,
  add seeds 4–6**.
- **Why BOTH, not just B2.** The −0.031 point gap is dominated by *dense* moving from 0.7536 (old) to
  0.7839 (new code) — i.e. the **dense band is the larger unknown**, not B2. Replicating only B2 (the
  critic's T2 recommendation, correct for the "is B2 reproducible" question) leaves the dense band as two
  draws spanning 0.027. You cannot subtract a fuzzy number from a fuzzy number and call the sign
  meaningful. **The apples-to-apples dense control alongside this sweep is operator-sanctioned** (dense
  is forbidden *for production*, not as a measured control in a comparison).
- **Exact knobs.** B2: `launch_B2.sh` as-is, varying **only** the global seed across {0 (done, ext100),
  1, 2}. Seeds enter at `actor_rollout_ref.actor.comm_eff.powersgd.seed`, `...mask.seed`, and the data/
  rollout seed — **vary the top-level training seed, hold the codec seeds fixed** so the codec is identical
  and only the rollout/data draw varies (matches how dense nondeterminism was measured). Dense: the dense
  rerun config (`done_dense_rerun.flag` / `train_dense_rerun.log`, all comm_eff counters 0), seeds {already
  have 0.7839 + the 0.7567@25 draw; add 2 more full seeds @50}. **Identical** batch128/mini64/lr1e-6/n=8/
  resp16384/2-epoch/test_freq25 across all six runs (the locked surface).
- **Cost.** ~5 GPU-hr/run × (2 new B2 + 2 new dense) = **~20 GPU-hr**. (Seed-0 B2 and one dense draw
  already exist, so the *incremental* spend is 4 runs.) Can pack 2 runs/box (4×H200) ⇒ ~2 box-sessions.
- **Decision rule.** Compute the two bands; apply the pre-registered decision bands above. **This is the
  gate:** if parity is established or near, the program proceeds against *real* pinned bands; if a
  **deficit**, **STEP 1 (shorter delay_K)** runs first (it is the on-mechanism lever most likely to recover
  the staleness-drift gap toward parity), and the expensive **STEP 4 (sub-basis)** is deferred until STEP 1
  shows whether the deficit is closeable at all (no point spending ~20 GPU-hr chasing a *greedy surpass*
  while still below the dense band).
- **Exclusion / risk check.** Dense-as-control is sanctioned (not production). No new primitive. The only
  risk is seed-dependent **ignition** in a B2 replicate (seed 0 was clean to 100 but that is censored and
  seed-specific — critic T1): **run each B2 seed with the P1/P2/P3 + m6<0.85 trip-wires live**, and treat
  an ignition in any seed as a first-class result (it would mean B2's stability is seed-fragile, which is
  more important than the val band). Re-derive the mem ceiling for any ≥100-step seed (ext100 hit 30.75/30.77).

### 0B — Two zero-GPU measurements that gate the expensive surpass cells (ride alongside 0A)

- **0B-i — Honest inter-stage byte accounting (GOAL-3, the savings *denominator* for every cell below).**
  *H:* anchor `M` DP-all-reduce (196 matrices, full-H, every cadence=5) + `Q`-broadcast each refresh,
  amortized and added to the fast-path 0.0505, still yields a **materially-<1** inter-stage ratio (the
  program's standing ~4×). *Knob:* **accounting pass, no training** — byte counters `comm/bytes_*` and
  `add_amortized_q_broadcast_bytes` are already logged; sizes known (196, r=77, H=1536, cadence 5). *Cost:*
  **~0** (≤2 GPU-hr to validate with on-collective counters if the operator wants belt-and-suspenders).
  *Decision value:* this is the **denominator** any residual-compression (STEP 2) win must beat, and it
  retires the bare 19.8× from all tables. *Exclusion:* none — pure accounting, cannot ignite.
- **0B-ii — Q-rotation telemetry (gates STEP 4 and the #28 current-step-EF route).** *H:* the act-basis `Q`
  is effectively **frozen** in the gradient-relevant directions after warm-up (`‖Q_new−Q_old‖_F` / top
  principal angle ≈ 0 per refresh), consistent with EXP-25 recon-flat ~0.024 and m7 rank-2 mismatch. *Knob:*
  **zero-GPU re-read** of the existing B2/ext100 logs (`anchor_q_updates`, `powersgd_q_cond`,
  `powersgd_reconstruction_rel_error` per layer) **+ wait for C3** and compute **C2−C3** (prices the
  Q-update). *Cost:* **~0** (C3 already a pending control). *Decision value:* **decides whether STEP 4
  (additive sub-basis) is even necessary** — if Q rotates materially, simpler forms may reach the missing
  directions and STEP 4's ~20 GPU-hr is avoidable; if Q is frozen, STEP 4 is the *only* route to the basis
  and is justified. *Exclusion:* none — telemetry re-read + a control that changes Q *update frequency*, not
  Q *construction* (so it does not re-enter the falsified EXP-26 Step-C hybrid-Q corner).

**Gate-out budget.** STEP 0 in full = **~20 GPU-hr** (the 4 incremental seeded runs) **+ ~0** (the two
accounting/telemetry reads). At the end of STEP 0 the program has: a pinned B2 band, a pinned dense band,
the honest savings number, and the Q-frozen verdict — i.e. **a real bar to beat and a priced denominator,**
before one surpass-specific cell is built.

---

## 2. The program — ranked by P(real win)·decision-value per GPU-hr (parity-defense → savings → secondary → the gated greedy-surpass bet)

> Each cell: **hypothesis** (falsifiable, numeric) · **exact knob/config** · **expected cost (GPU-hr)** ·
> **decision rule** · **exclusion/risk check**. All are gated behind STEP 0 returning pinned bands (you
> cannot measure "above B2" or "above dense" without them). The theoretical case for each is in **[feas]**;
> I state the operational version. **Ordering reflects the theorist's priced surpass surface ([feas §7]):**
> the expected win is **seed-replicated parity + a likely secondary pass@k edge (~30–40%)**; a *greedy*
> surpass is **~10%, via one gated route only**. So the ranking leads with the cheapest parity-defense and
> the most-likely *real* win, and places the genuine greedy-surpass bet where its odds and cost put it.

### STEP 1 — Shorter delay_K (K ∈ {3, 2}) — **the cheapest parity-defense; closes the gap toward TRUE parity**

- **Hypothesis (falsifiable, numeric).** Reducing K from 5 → 3 (then 2) shrinks the **fast-circuit drift
  term** `[G_comp(t) − G_comp_ring(t−K)]` that makes B2's telescoping only *approximate* — moving `G_corr`
  closer to the pure true gradient `G_anc_rep`. Predicted effect: **val@50 moves up toward the dense band
  and/or the late-run decay (0.7536@50 → 0.7400@100) lessens**, with no increase in emission. **Falsified**
  if val@50 is unchanged within pooled-SE AND decay is unchanged (⇒ drift was already negligible at K=5).
- **Why it is the parity-defense, not a surpass promise.** [feas §3c, §4a]: B2's ceiling is the dense
  gradient itself (it *reconstructs* dense-on-stale-data), and the K-tick **staleness drift is exactly what
  puts B2 *below* dense at horizon** — the ext100 0.7400@100 slide is that drift expressing. Shrinking K
  reduces the drift *without re-admitting the biased artifact*, which is why it **dominates partial-λ**
  ([feas §4a] — λ<1 re-admits bias, shorter-K removes drift). Its theory ceiling is **clean parity**, and
  parity is the program's expected, GOAL-satisfying win — so this is rank 1: cheapest, on-mechanism, the
  most likely single knob to convert "near-parity" into "parity established."
- **Exact knob/config.** `actor_rollout_ref.actor.comm_eff.anchor.delay_K` 5 → **3**, then **2**. Hold
  cadence=5 fixed (decouple from a cadence change). **Config-only, no code.** One 50-step cell per K; bundle
  as a 2-point mini-sweep on the *converting* primitive (admissible — not the EXP-23 inert exclusion). Run
  the most promising K on the seeded protocol so any parity claim is band-vs-band.
- **Cost.** ~5 GPU-hr/K × {3, 2} = **~10 GPU-hr**.
- **Decision rule.** Per K: best val@50 + late-decay slope. **PARITY-IMPROVED** iff val@50 band rises toward
  the dense band OR the @75/@100 decay flattens (run the winning K to ≥100 to read decay; re-derive mem
  ceiling). **NULL** iff flat within noise (⇒ drift negligible, K=5 stands). A *surpass* is not expected
  here; if a K band lands above dense beyond noise, treat it as a bonus to be seed-confirmed, not the goal.
- **Exclusion / risk check.** Admissible (delayed_ef converts). **Carrier check mandatory:** shorter K
  refreshes the residual against a *fresher* ring entry — verify **m6 does not rise** (fresher pairs could
  be *more* autocorrelated, pushing τ/cadence up); P1/P2/P3 + m6<0.85 trip-wires live. K cannot reach 0
  (that is the #28 current-step codec-internal EF — a *separate*, lower-byte route that needs no anchor
  backward, gated on the Q-rotation read 0B-ii — not the K→0 limit of this cell). Not dense/signed_ema/
  blend/clean_cadence/hybrid-Q.

### STEP 2 — Residual-compression r_δ≪77 — **a GOAL-3 *savings* surpass (NOT a val-surpass)**

- **Hypothesis (falsifiable, numeric).** The residual δ inherits the valid gradient's concentration (m7:
  stable-rank≈2, top-1% mass≈0.60), so δ transmitted in **r_δ ∈ {16, 8}** columns (or a top-k≈1–2%
  coordinate-sparse form) preserves the B2 val band (within pooled-SE) **at a strictly lower honest
  inter-stage byte cost than B2** (it must beat the **0B-i** denominator, not the bare 0.0505). This is a
  surpass on the **savings axis** — "dense-parity val at strictly-better-than-B2 honest savings" is a
  *better solution than dense* (which trains at 1×). **Falsified** if val drops below the 0.7210 floor at
  r_δ=16, OR if the byte saving is illusory once the anchor `M`/`Q` traffic (0B-i) is counted.
- **Why it is a savings win, explicitly NOT a val-surpass (category discipline).** [feas §4c] is decisive:
  the truncation error δ introduces is itself a **biased, structured** residual (the dropped low-energy
  tail), so its **expected effect on val is ≤ 0** — it can only degrade B2's reconstruction of the dense
  gradient. Filing it under "beat dense (val)" is a **category error.** It *is* a legitimate GOAL-3 surpass
  (better savings at held val), which m7 strongly licenses (the residual is the rank-~2 object). **[feas]**
  carries the info-theoretic floor on how far r_δ can drop before the rank-2 signal is lost.
- **Exact knob/config.** A residual-codec rank `r_δ` ∈ {16, 8} **or** a top-k% mask on δ (k≈1–2%).
  **Requires a small code change** on the correction path (sketch δ before injection) on an `exp/<N>`
  branch — *not* config-only. **Leave the forward/recon act-basis `Q` untouched** (compress only the
  *correction* δ, never the forward pass) — keeps it clear of EXP-26 Step C by construction. Hold all else
  at B2. **Zero-GPU pre-gate:** recompute the per-target SVD energy of the *stored* δ (Step-A sidecar /
  `B2_delta_per_fire.jsonl`) to confirm a rank-16 truncation retains ≥~95% of δ's energy *before* a cell —
  if it doesn't, the cell is killed for free.
- **Cost.** ~1–2 GPU-hr code + a scale-consistency unit test (mirror #25's mean-vs-sum invariant) +
  ~5 GPU-hr per r_δ ⇒ **~12 GPU-hr** for {16, 8}, **minus** whatever the zero-GPU SVD pre-gate rules out.
- **Decision rule.** Per r_δ: best val@50 **and** the honest inter-stage ratio from 0B-i recomputed with the
  compressed δ. **GOAL-3 SURPASS** iff val band ≥ B2 band (within pooled-SE) AND honest ratio strictly below
  B2's. **NULL** iff val drops below floor or savings don't beat the denominator.
- **Exclusion / risk check.** Avoids EXP-26 Step C **by construction** (forward Q unchanged; only δ
  compressed). δ stays endogenous + K-delayed ⇒ no new exogenous carrier, **but re-verify m6** (a coarser δ
  could change its autocorrelation — run the trip-wire). **Heavy-tailed-early-δ caveat** (critic T4: ~43% of
  matrices have ‖δ‖/‖G‖>1.5 at the first fire) ⇒ aggressive truncation is roughest in the first few fires —
  **keep the warmup**. Not dense/signed_ema/clean_cadence/hybrid-Q.

### STEP 3 — Eval-diversity: pass@k / temperature-n — **the MOST LIKELY actual "win" (~30–40%), but secondary**

- **Hypothesis (falsifiable, numeric).** Greedy **mean@1** is **blind to a diversity edge unless the mode
  moves** ([feas §3b]); a comm-eff trainer makes a measurably **more-diffuse policy** (uncompressed-generator
  `rollout_ppl` 1.40 vs dense 1.24 at step 25 — a real ~13% diversity edge, [feas §3a]). Under **pass@k**
  (k∈{4,8}) or **temperature-T sampling (mean@n)**, the B2 band **exceeds the dense band under the identical
  eval** by > pooled-SE — a *real* win that the greedy bar hides. **Falsified** if B2 and dense move together
  under pass@k (edge is generic, dense catches up).
- **Why it is the most-likely win AND why it is secondary.** [feas §6]: P(a real pass@k / diversity edge) ≈
  **30–40%** — the single most probable "beat dense" result this program lands. **But** it is on a
  **more-lenient bar**, not the GOAL's greedy mean@1, and conflating the two would overclaim. The
  discriminating signature: the **pass@k coverage curve vs k** — if the (compressed − dense) advantage
  **grows with k** the edge is compression-specific (real); if **flat in k** it is generic. **[feas]** carries
  whether the more-diffuse policy is the same mode-quality (the pass@k-positive case) or just a blurred mode.
- **Exact knob/config.** **Eval-only — no training.** Re-evaluate the **existing** B2 (ext100) and dense
  checkpoints with rollout sampling on: `actor_rollout_ref.rollout.val_kwargs` `do_sample=true`, `n∈{4,8}`,
  `temperature∈{0.7,1.0}`; compute pass@k and mean@n. **Mandatory dense×{T,n} control under the identical
  eval** (operator-sanctioned; [feas §6]: raising dense's temperature/samples is the cheap kill — if dense
  matches the curve, no comm-eff edge). If checkpoints weren't saved at the needed step, this needs a short
  re-emit run.
- **Cost.** **~1–3 GPU-hr** (inference-only over 1319×k for both checkpoints), **+~5** if a checkpoint must
  be regenerated.
- **Decision rule.** **SECONDARY WIN** iff B2 pass@k band > dense pass@k band by > pooled-SE under identical
  sampling **AND the advantage grows with k** — report explicitly as *"surpasses under pass@k (coverage
  edge), parity under greedy"* (never as a greedy surpass). **NULL** iff flat in k / tracks dense.
- **Exclusion / risk check.** Eval-diversity is **secondary by construction** (val is the greedy GOAL bar; a
  pass@k edge is a *different, more-lenient* claim). No training, no carrier, cannot ignite. **Must** carry
  the dense×{T,n} control or it proves nothing. Does not touch any excluded primitive.

### STEP 4 — Additive off-principal sub-basis — **the ONLY genuine greedy-surpass route (~12–18% | run); gated, run-later**

- **Hypothesis (falsifiable, numeric).** A small gradient sub-basis (rank≈2–4) sketched from the EXP-29
  replay gradients and routed **additively into the K-delayed, fire-refreshed, β_anc=0 correction term
  only** — leaving the act-basis forward/recon `Q` **unchanged** — injects the rank-~2 true direction the
  act-basis *misses* (F1 cos≈0 / m7 mismatch) **more cleanly than B2's drift-limited telescoping**, lifting
  the seed-replicated B2 band **above the dense band by > pooled-SE on the greedy bar**. **Falsified** if val
  does not exceed the dense band beyond noise, OR if the static sub-basis **ignites**.
- **Why it is the only double-digit greedy-surpass bet — and why its prior is still sub-even.** [feas §5]:
  it is the **only** route that attacks the *root cause* (m7 basis-mismatch) rather than reconstructing
  dense, and B2 *proves* the missing directions are recoverable+helpful — so P(surpass | run) ≈ **12–18%**,
  the only route with double-digit odds. **The honest deflation [feas §5]:** a sub-basis sketched from the
  *valid PG* gradient injects the **same true gradient dense already follows** — injecting dense's own
  directions more cleanly gets you *to* dense, not *past* it; its realistic ceiling is **clean parity**
  unless an unproven compression-specific conversion effect ([feas §3a], rated <20%) materializes on the
  greedy bar. So: highest greedy-surpass ceiling, but **more likely to *tie* or *ignite* than to beat**
  (P(parity)≈45%, P(parity-but-decay)≈25%, P(ignite)≈15–20%). Hence gated and run-later, not the headline.
- **Exact knob/config.** New code: a `q_basis_passive`-style additive correction sub-basis sketched from
  replay gradients, rank 2–4, injected only into the delayed_ef correction (the `powersgd.q_basis_passive=[]`
  hook in `resolved_params_B2.txt` suggests the plumbing partly exists — **verify before scoping**). Hold
  forward `Q`=act, r=77. `exp/<N>` branch, substantial code + CPU test. Run ≥100 steps with the seeded
  protocol (a surpass claim is band-vs-band).
- **Cost.** ~2–4 GPU-hr code+test + ~15–20 GPU-hr to run with a proper ≥100-step horizon ⇒ **~20 GPU-hr**.
- **Decision rule.** **GREEDY SURPASS** iff seed-replicated val band > dense band by > pooled-SE,
  emission-free through ≥100 steps (re-derive mem ceiling). **PARITY** iff it lands in the dense band (still
  a win — root-cause-clean parity). **STOP** iff below-band or ignition (record ignition as
  "endogenous-carrier-law extends to an injected sub-basis" — a high-information negative).
- **Exclusion / risk check.** **Admissible ONLY under three conditions** (strategist §3 / [feas §5], the
  gate): (a) it **augments** rather than **replaces** the forward act-`Q` (replacement re-enters the
  falsified EXP-26 Step-C hybrid-Q corner); (b) delivered **K-delayed / fire-refreshed / β_anc=0** (a static
  sub-basis injected every step is the persistent exogenous direction the carrier law convicts — it carries
  B2's m6≈0.62 base persistence *plus* a fresh ignition surface); (c) **gated behind STEP 0 (a real band to
  beat) AND 0B-ii (run only if Q is frozen — if Q rotates, the simpler forms reach the directions and this
  is unnecessary complexity)**. Same 100-step censoring caveat as B2.

### STEP 5 — Partial residual λ=0.8 — **OPTIONAL single damping-robustness point (demoted; not a surpass cell)**

- **Hypothesis (falsifiable, numeric).** A single **λ=0.8** point *reduces the late-run decay*
  (0.7536@50 → 0.7400@100) relative to λ=1, as a gentler injection — a stability read, **not** a val
  surpass. **Falsified** if decay is unchanged within noise.
- **Why it is demoted to one point (or skipped).** [feas §4a]: λ<1 **re-admits the biased, near-orthogonal
  codec artifact** (the §3 BLOCKED "compression-noise-as-regularizer" class) — it is *not* zero-mean
  exploration noise, so it "loses less slowly" at best and is **dominated by STEP 1 (shorter-K)**, which
  reduces the same staleness drift *without* re-admitting bias. So λ<1 is **not a surpass lever** (retiring
  my own earlier framing) and is worth **at most one robustness point**, and **only if STEP 1 does not
  already flatten the decay**. **Skip-rule:** if STEP 1's shorter-K flattens the @75/@100 decay, **do not
  run this** (STEP 1 dominates it). λ>1 is **forbidden** (over-subtraction → raises the carrier → ignition).
- **Exact knob/config.** `spectral.delayed_ef_lambda` 1.0 → **0.8**, single point, all else at B2.
  Config-only. **Never a grid; never λ>1.**
- **Cost.** **~5 GPU-hr** (one point), or **0** if STEP 1 already settles the decay.
- **Decision rule.** **DAMPING** iff @75/@100 decay flattens vs λ=1 without losing val@50 band → a minor
  stability knob. **NULL** iff unchanged → λ=1 stands, route retired.
- **Exclusion / risk check.** Admissible as a single point (delayed_ef converts; not the inert-primitive
  exclusion), but **dose-chasing in spirit** — capped at one λ<1 point. Carrier risk: the uncancelled
  `(1−λ)` artifact is the carrier B2 cancels at λ=1, so re-admitting it raises ignition risk → **m6<0.85 +
  P1/P2/P3 trip-wires mandatory**; the ≤50-step read is censored. Never λ>1, signed_ema, blend, hybrid-Q.

---

## 3. What is BLOCKED / out-of-scope for surpass (standing exclusions, re-checked)

| route | status | reason (cite) |
|---|---|---|
| **λ > 1** (over-subtraction) | **FORBIDDEN** | injects `(λ−1)·δ ≈ −(λ−1)·G_comp`, a persistent larger-magnitude push that **raises the carrier** — the exact dose-escalation EXP-27 convicts (dose-raising ignites *sooner*). STEP 5's λ-point is **λ<1 only**. |
| **small-β_anc EMA smoothing** | **BLOCKED** | m6≈0.62 was the pre-registered safety number and came back **unsafe** (τ≈2× cadence; β_anc>0 *compounds* it). One narrow contingent re-opening: only if a **seed-replicated** B2 (STEP 0A) holds emission-free well past 100, on the δ-residual (endogenous) object only, with a fresh carrier budget. Not in this program. |
| **signed_ema (sign-replacement)** | **DEAD** | EXP-25/26 structurally convicted (≈50% sign-disagreement at delay 0, M-validity-independent). `signed_ema_alpha=0.5` in B2's params is an inert leftover default. |
| **update-energy / hybrid Q** (build/blend Q from gradient energy) | **DEAD** | EXP-26 Step C anti-converts (rollouts uncompressed ⇒ a gradient-energy Q breaks forward recon). STEP 2 avoids it (compresses δ, not Q); STEP 4 avoids it (augments, never replaces, Q); 0B-ii avoids it (changes Q *update freq*, not *construction*). |
| **clean_cadence > 0** | **EXCLUDED** | clean_cadence=0 is the locked realistic substrate (the anchor replaces the unrealizable periodic-dense step). Do not reintroduce. |
| **dense re-run FOR PRODUCTION** | **FORBIDDEN** | operator directive 2026-06-10. **But** an apples-to-apples dense **control** alongside a sweep (STEP 0A, STEP 3) is **operator-sanctioned** — that is a measured comparison, not a production claim. |
| **λ grid (fine) / dose-chasing on any inert primitive** | **DISCOURAGED** | STEP 5 is a *single* λ=0.8 damping point on a *converting* primitive (admissible), capped at one point — **not** a grid, and **dominated by STEP 1 (shorter-K)** for the decay question. No sweeps on signed_ema/blend/inert codecs. |
| **blend as a route to surpass** | **OUT (dominated, not inert)** | B1=0.7422 converts but is *geometrically dominated* by the residual (F1: 92%-energy ⊥ contamination can be subtracted, only weakly offset). Not a surpass candidate; kept on the record as the second-best converter. |
| **plain mask-p / Gaussian as "generate more diversity"** | **OUT (conversion-limited)** | TRAIN-ONLY primitives that repeat the psgd null under a greedy bar (memory `surpass-dense-conversion-spine`). [feas §4a/§4b] further convicts *any* deliberate train-side perturbation (biased λ<1, or an exogenous carrier that ignites); eval-diversity (STEP 3, pass@k) is the surviving diversity route. |

---

## 4. Next session's run order (decision-value-per-GPU-hr shortlist)

> Framed as **"what the operator runs next, in order."** Tiered: the **parity gate** (seed replicates +
> the two zero-GPU can't-be-wasted reads) first; then the **cheap parity-defense + the most-likely real
> win**; then the **savings surpass**; then the **gated greedy-surpass bet** last. Total fits inside ~2
> budget cycles even in the run-everything case.

| order | cell | objective (P, [feas]) | what it could prove | cost (GPU-hr) | gate / decision rule |
|---|---|---|---|---|---|
| **0** | **STEP 0A — seed-replicate B2 ×3 + dense ×3 @50** | **establish parity** (gating) | pins both bands ⇒ a *real* bar; or a deficit to recover | **~20** (4 incremental runs) | \|Δmean\| ≤ 0.020 ⇒ parity; > 0.040 ⇒ deficit; else +seeds. **Gates everything.** Trip-wires live per B2 seed. |
| **0′** | **0B-i honest-bytes + 0B-ii Q-telemetry/C2−C3** | savings denominator + basis verdict | the honest GOAL-3 number; whether STEP 4 is needed | **~0** (≤2 to validate) | ride alongside 0A; cannot be wasted, cannot ignite |
| **1** | **STEP 1 — shorter delay_K {3, 2}** | **parity-defense** (closes staleness-drift gap) | val toward dense band / flatter @100 decay = *true* parity | **~10** | val band ↑ toward dense OR decay flattens. Config-only; m6 trip-wire (fresher pairs). *Surpass not expected.* |
| **2** | **STEP 2 — residual-compression r_δ∈{16,8}** | **SAVINGS surpass** (P moderate) | dense-parity val at strictly-better-than-B2 honest savings | **~12** (−SVD pre-gate) | val ≥ B2 band AND honest ratio < B2's. Zero-GPU SVD pre-gate first; forward Q untouched. *Not a val-surpass.* |
| **3** | **STEP 3 — pass@k / temperature-n eval** | **MOST LIKELY win** (~30–40%), *secondary* | a coverage edge greedy mean@1 hides — *with* dense×{T,n} control | **~1–3** (+5 if regen) | B2 pass@k band > dense by >pooled-SE AND grows with k; report "pass@k coverage edge, greedy parity" |
| **4** | **STEP 4 — additive off-principal sub-basis** | **GREEDY surpass** (~12–18% \| run) | inject the rank-~2 direction the act-basis misses → above dense | **~20** | **gated** behind STEP 0 (a band to beat) + 0B-ii (run only if Q frozen) + horizon. Augment-not-replace, K-delayed, β_anc=0. *More likely to tie/ignite than beat.* |
| **5** | **STEP 5 — λ=0.8 single point** *(optional)* | damping-robustness (not surpass) | gentler injection → flatter decay | **~5** (or 0) | run **only if STEP 1 didn't flatten the decay** (STEP 1 dominates). Never grid; never λ>1. |

**Budget shape.** The **parity gate** (STEP 0A + the two zero-GPU reads) = **~20 GPU-hr** and produces a
pinned B2 band, a pinned dense band, the honest savings number, and the Q-frozen verdict — *before any
surpass-specific cell*. The cheap tier (STEP 1 ~10 parity-defense + STEP 3 ~1–3 pass@k, the most-likely
real win) runs next. STEP 2 (~12) is the GOAL-3 savings surpass. STEP 4 (~20) is the gated greedy-surpass
bet, deliberately last (and skippable if 0B-ii shows Q rotates). STEP 5 is ~0–5 and often skipped.
**Two budget cycles** cover the full program; the gate-it-out path (parity establishes, Q rotates ⇒ STEP 4
unnecessary) reaches a defensible *"parity established + measured savings + seed-replicated + pass@k coverage
edge checked"* — a **complete GOAL-2/3 win** — in **~35 GPU-hr** without ever spending STEP 4.

**The one sentence.** **First spend ~20 GPU-hr establishing whether B2 even reaches parity** (3 seeds of
B2 and 3 of dense + the two free honest-bytes + Q-telemetry reads) — **nothing labeled "surpass" is
meaningful until both bands are pinned** — **then run shorter delay_K as the cheapest parity-defense and
the pass@k eval as the most-likely real (but secondary) win**, keep **residual-compression** as the GOAL-3
*savings* surpass, and hold the **additive off-principal sub-basis** as the **one gated ~10–18% greedy-
surpass bet** — demoting partial-λ to an optional damping point and never running λ>1, small-β_anc EMA,
signed_ema, hybrid-Q, clean_cadence, or dense-for-production.

---

## Honest prior on the headline question (does B2 → surpass dense?)

Calibrated from [feas §7], so the issue carries an honest expectation, not hype. **The single number:
P(greedy surpass, seed-replicated, beyond ±0.024 nondeterminism) ≈ 10%** (8–12%), and that mass comes
**almost entirely from one route** (STEP 4 sub-basis: P(surpass | run) ≈ 12–18% × P(run+gates+baseline)
≈ 0.6–0.7); every other route contributes ≈0 to a *greedy* surpass. The **expected** outcome is
**seed-replicated parity + a likely secondary pass@k edge** — itself a complete GOAL win (criterion 2 asks
for *≥ dense within noise*, which parity satisfies). The structural ceiling ([feas §3c]): B2 *reconstructs*
the dense gradient, and a faithful copy of dense cannot beat dense.

- **P(parity established by STEP 0):** **moderate-high.** The −0.031 gap is ~1.3σ and dense moved *up*
  from old code; with 3 seeds each the bands likely *overlap* (parity), with a real chance dense's band
  sits modestly above (a small deficit B2 must recover before surpass is plausible).
- **P(a train-side parity-defense, STEP 1 shorter delay_K):** **moderate** — shrinking K removes the
  staleness drift that puts B2 *below* dense at horizon ([feas §3c]) *without* re-admitting bias, so it is
  the single knob most likely to convert "near-parity" into "parity established." Its ceiling is parity,
  not surpass.
- **P(a train-side *surpass*, STEP 5 partial-λ — demoted to an optional damping point):** **low** — the
  exact-telescoping read says λ=1 is special and λ<1 re-admits the *biased* artifact (not zero-mean
  exploration noise — [feas §4a]); a surpass would need that re-admitted bias to act as *useful*
  exploration, and the base rate of "biased perturbation beats the unbiased estimator" is low. It is
  dominated by STEP 1 (which removes drift without re-admitting bias) and kept only as a ≤1-point decay
  read. (Defer to **[feas]** for the number.)
- **P(a GOAL-3 savings surpass, STEP 2):** **moderate** — m7 strongly licenses compressing the rank-~2
  residual; "dense-parity at better-than-B2 honest savings" is the *most likely* form of "better than
  dense this program lands," and it is a legitimate surpass on GOAL-3 even if the *val* never exceeds dense.
- **P(a measurement-frame surpass, STEP 3 pass@k):** **low-moderate** — the conversion-spine analysis
  rated a Route-A pass@k edge as the likely (often only) surpass outcome; cheap enough to settle.
- **P(a root-cause surpass, STEP 4):** **low but highest-ceiling** — the only route that could put val
  *above* dense by design; gated precisely because it is expensive and carrier-exposed.

**Net:** the most probable "beat dense" the operator banks from this program is **STEP 2's parity-val-at-
strictly-better-savings** (a GOAL-3 surpass), with **STEP 3 (pass@k, secondary/measurement-frame) and
STEP 4 (root-cause, greedy)** as the lower-probability shots at a *val* surpass — while **STEP 1
(shorter-K)** is the cheapest *parity-defense* (its ceiling is parity, not surpass) and **STEP 5
(partial-λ)** is a demoted optional damping point. The first dollar, regardless, goes to **STEP 0** —
because the entire vocabulary of "surpass" is undefined until the two bands are pinned.
