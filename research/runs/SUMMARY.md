# Research runs — summary

The **durable record** of what has run on this harness and what it means. Full
per-experiment artifacts are pruned (folded here by the `de-bloat` skill); the
lasting record is **here + git history + W&B + the merged code**.

> This file is the **single source of truth for results + why + what's next.**
> Other docs (`GOAL.md`, `CODE_WALKTHROUGH.md`, `FIXED_CONTROL_SURFACE.md`, the
> launchers) describe the *goal*, the *wiring*, and the *config* and link back
> here for outcomes — they do **not** restate these numbers. Keep it that way:
> duplicated results drift into contradiction.

## ⭐ CURRENT STATE OF THE ART — start here

**SOTA comm-eff method = `B2` — `delayed_ef`** (K-delayed exact codec-residual = error-feedback),
`G_corr = G_comp + λ·δ`, **λ=1, β_anc=0**.

- **Result:** greedy GSM8K **val@50 ≈ 0.74–0.75 = PARITY with dense** (band 0.75–0.78) at ~5%
  fast-path gradient-comm cost (~4× honest amortized). W&B B2 `u9okvgzz` 0.7528; reproduce/ext draws
  0.7400–0.7536.
- **The exact SOTA settings are the ground truth in** [`runs/EXP-31/B2_baseline/resolved_params_B2.txt`](EXP-31/B2_baseline/resolved_params_B2.txt)
  **+** [`runs/EXP-31/B2_baseline/launch_B2.sh`](EXP-31/B2_baseline/launch_B2.sh) — the substrate **every new arm holds fixed**
  (migrated from EXP-30, now deleted; see `runs/EXP-31/B2_baseline/README.md`).
- **Locked substrate:** PowerSGD **r=77** act codec · anchor **owns Q** · **cadence=delay_K=5** ·
  **clean_cadence=0** · **replay_paired_batch=true** · snapshot_device=cpu · batch128/mini64/lr1e-6/n8/
  resp16384/seed0 · **disable_custom_all_reduce=true** (required for the box to init; greedy-val-neutral).
- **Current frontier status (2026-06-17):** The 4-lever anchor-usage tournament (EXP-31, issue #31)
  is **CLOSED — VERDICT STOP**. All four admissible levers (L4 perturbation, L2 δ-momentum, L3 adaptive
  dose, L1 control-variate) are NULL. The β_anc EMA sweep (EXP-33, issue #33) is **CLOSED — VERDICT PASS
  (measurement)**: the β axis is a flat free-averaging region (all β∈[0,0.75] tie within ±0.024; β=1
  collapses). β=0 holds as the default. No nameable non-blocked knob with a credible path to ≥0.78 remains
  on either the anchor-usage or β_anc axes. See §EXP-31 tournament and §EXP-33 below.

Everything below B2 in the table (EXP-20/23/25/26/27/29) is **superseded history**, folded into this file
(detailed run dirs de-bloated 2026-06-15; full detail in git history + W&B + the merged code).

## The settled communication-efficient base (as of issue #25, 2026-06-09)

The comm-eff training base is the **anchor circuit on a PowerSGD codec**. Two
properties are now **mandatory** for every comm-eff run and define the substrate:

1. **The anchor network is a must.** A continuously-maintained, `delay_K=5`-stale,
   full-coverage (all 196 weight-matrix gradients, DP-reduced) anchor gradient EMA
   `M`, refreshed every `cadence=5` ticks from a no-hook isolated clone. This
   **replaces** the old `clean_cadence` periodic-dense-step crutch (unrealizable
   in a real decentralized-PP setting: full-H transfer, and itself stale on a slow
   link).
2. **The anchor is the ONLY thing that updates `Q`.** The PowerSGD basis `Q` is
   computed by the anchor from its stale-weight forward activations (`Q←orth(V)`)
   and broadcast to every DP rank each refresh. The fast (compressed) circuit is a
   pure read-only consumer (`Y=hQ`, `ĥ=YQᵀ`) and is fail-closed from ever writing
   `Q`.

The exact knob values are **not** restated in prose anywhere — they live in the
launcher `${VAR:-default}` defaults
(`examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`) and in
the ground-truth `resolved_params.txt` of each run. The locked training surface
(model/data/objective/batch/lr/steps) is `FIXED_CONTROL_SURFACE.md`. How the
circuit is wired in the verl source is `CODE_WALKTHROUGH.md`.

## Runs

| id | milestone | what | result (val@50 GSM8K) | record |
|---|---|---|---|---|
| **baseline** | M1 | Dense GRPO, Qwen2.5-1.5B-Instruct, GSM8K — verl unmodified (the control = comm-eff OFF) | **0.7536** (the bar) | W&B `5e2jpho9` |
| EXP-20 | M6 | PowerSGD r=77 (byte-matched) + fresh clean@5 — the prior comm-eff PASS | 0.7415 | W&B `oquyeic3` |
| EXP-23 | M6 | PowerSGD r=77, **no** re-anchor (the floor); inject/blend stale-anchor mergers (STOP) | 0.6914 (floor) | W&B; erratum `a46fd0191` |
| **EXP-25** | M6 | **Anchor circuit default** — full-coverage DP-reduced stale `M` (R1) + anchor-owns-`Q` (R2) + sign-replacement merger (R3), α swept | **0.7066** (best, α=0.5) — **STOP** | code on `vast-ai-workload`; issue #25 |
| EXP-26 | M6 | Real-gradient geometry audit + ef_powersgd merger; Step-C (gradient-tuned forward Q) **falsified** by recon collapse | 0.7210 (ef best) — REVISE→closed | folded here; W&B pruned; LOG.md |
| EXP-27 | M6 | Damped ef_powersgd (clip/decay 0.5) to 100 steps — ignition test | 0.7202 (**ignited @~66**, length-explosion) — STOP | folded here; W&B `qa6sll3h`; LOG.md |
| EXP-29 | M6 | On-policy anchor **replay** (`replay_paired_batch` + `snapshot_device` knobs) — the valid-M infra B2 uses | infra PASS (no val bar) | merged PR #16 `d26176b44`; now part of the B2 substrate |
| **EXP-30** | M6 | K-delayed codec residual (B2, delayed_ef λ=1, β_anc=0) + valid anchor M via on-policy replay (geometry-gated) | **0.7528** (B2 @50) — **PASS = SOTA** | PR #17 merged `ca5f4b002`; verdict + B2 ground truth migrated to `runs/EXP-31/B2_baseline/` |
| **EXP-31 (sub-basis)** | M6 | Stale-anchor rank-2 sub-basis merger: additive off-principal correction into δ, forward Q untouched; dense reframe (dense-here=0.7506) | **0.7400** (B2/Cell A) — **PARITY (operator-accepted)** | branch `exp/31-subbasis-merger` (unmerged); verdict `runs/EXP-31/verdict.md` |
| **EXP-31 (tournament)** | M6 | 4-lever anchor-usage tournament (L4 perturbation / L2 δ-momentum / L3 adaptive dose / L1 control-variate) on B2 substrate | **NULL across all levers** — **STOP** | box i_41048644 4×H200; W&B fy920fty/ybemd5ux/knlzxh2x/kzohyuod/wmpmmdj1; verdict `runs/EXP-31/verdict.md` |
| **EXP-33** | M6 | β_anc EMA sweep {0, 0.25, 0.50, 0.75, 1.00} on B2 delayed_ef substrate — first direct β-curve on the valid-M circuit | **FLAT free-averaging region** (all β∈[0,0.75] within ±0.024; C2 β=0.5 nominal peak 0.75284 NOT a SOTA promotion; C4 β=1 cold-M collapse) — **PASS (measurement)** | box i_41194490 4×H200 (torn down); WandB `verl_compression_research_beta_sweep`; verdict `runs/EXP-33/verdict.md` |

## EXP-25 — what we learned (issue #25, VERDICT = STOP)

**The substrate is proven; the merger is not.** R1 (full-coverage, DP-reduced,
correct-scale `M`) and R2 (anchor-owns-`Q`, fast net never writes `Q`) both passed
their on-box probe gates — the anchor circuit is mechanically correct and is the
**realistic** setting (continuously-maintained stale anchor, no clean step). But
R3, the **sign-replacement** merger `G = α·G_noisy + (1−α)·|G_noisy|·sign(M)`, **does not
beat — or even match — plain PowerSGD**:

- Dose-response is **monotonic and net-harmful**: α=0.5 → 0.7066, α=0.3 → 0.6164,
  α=0.0 → 0.3541. More signed correction ⇒ worse. The best arm sits at the
  least-correction edge, below the PowerSGD-only reference (0.7415).
- **Mechanism:** the stale-anchor sign disagrees with the live gradient on ~50% of
  magnitude-weighted coords (structural, ≈√2 `rel_change`, not staleness), so
  replacing the live update *direction* with `sign(M)` destroys the per-coordinate
  sign-cancellation that regularizes the GRPO step. The proximate val-killer is a
  **response-length reward-hack** that ignites only under sign-reversal (α<0.5)
  on the no-KL/no-entropy surface — not low entropy per se.

So we **cannot surpass dense** with this merger, and it is **falsified**. The
honest reading: this is a **good research base to start from** — the realistic
circuit is wired and proven — and the open question narrows to the **merger /
gradient-correction primitive**.

Scientific detail: the EXP-25 deep writeups (`COLLAPSE_GRADIENT_FLOW_ANALYSIS.md`,
`DEEP_FINDINGS.md`, `ENTROPY_COLLAPSE_FINDINGS.md`) were de-bloated 2026-06-15 — their essence is
the paragraph above + the memory entries; the full text is in git history.


## EXP-31 — the parity result (issue #31, VERDICT = PARITY, operator-accepted 2026-06-14)

**The key finding: comm-efficient GRPO already matches dense at ~5% gradient-comm cost.**

The most important result is a **dense-reference reframe**: the dense bar on THIS config
(same box, same code, same `disable_custom_all_reduce`, seed 0) is **0.7506**, not 0.7839.
The 0.7839 was a high draw on a different box. Re-running dense apples-to-apples gives:

- dense-here: 0.7506 (val@50, single draw, ±0.024 noise)
- B2 / best comm-eff (delayed_ef λ=1, r=77 act, cadence=delay_K=5): **0.7400**
- Gap: 0.011 — inside the ±0.024 eval noise = **statistical PARITY at ~5% comm cost**

The "0.044 gap to dense" EXP-31 was designed to close was a wrong-reference artifact.

**What the rank-2 sub-basis proved (and did not):**
- Mechanically correct: captures 88-90% of the off-principal energy; CPU suite 213 tests green; rank-0/weight-0 = bitwise-B2.
- Accelerates early learning: r2 arm +0.036 vs B2 at step 25 (0.7293 vs 0.6937).
- Does NOT convert to a greedy surpass: constant weight over-amplifies near convergence (r2 regresses 0.7293→0.6983@50); γ-decay fixes regression (0.7210@50) but tempers early gain; every variant clustered at ~0.70–0.74, none cleared the dense band.
- Mechanistic conclusion: the sub-basis speeds the path to the optimum but does not find a better optimum. Surpass requires a different mechanism.

**Caveats:** single-draw vals (±0.024); seed bands deferred (box 40806688 stopped Vast-side,
hold25-decay25 val@50 on disk but unsynced). Full detail: `runs/EXP-31/verdict.md`.

**Code:** branch `exp/31-subbasis-merger` pushed, unmerged. No PR / no launcher promotion
(`promote_launcher_as=none`; PARITY is not a PASS in the mandate terms).

## EXP-31 — the tournament result (issue #31, VERDICT = STOP, 2026-06-16)

**The anchor-usage axis is exhausted on the locked B2 substrate.** All four admissible levers — the
complete set of ways to *use* the stale anchor gradient `M` without changing the codec, Q, batch, or
generation — are NULL.

**Reference:** B2_live (Cell A reproduced, box i_41048644 4×H200, seed 0): val@25=0.7202 / val@50=0.7354.
Dense-this-box=0.7506 (band 0.75–0.78). Eval noise ±0.024.

**Lever results (all val@25, greedy GSM8K):**

| Lever | Config | val@25 | verdict |
|---|---|---|---|
| L4 perturbation | σ=0.01 isotropic noise on δ | 0.7157 | NULL (parity) |
| L2 δ-momentum | μ=0.9 EMA accumulation | 0.5701 | REGRESS (−0.15, over-smoothed) |
| L2 δ-momentum | μ=0.5 EMA accumulation | 0.7089 | NULL (parity) |
| L3 adaptive dose | ratio κ=1.0 cap | 0.7119 | NULL (parity) |
| L3 adaptive dose | cos κ=1.0 cap | 0.7134 | NULL (parity) |
| L1 control-variate | — | SKIPPED | gate F1 fails: cov(G_comp,M)≈0 |

**4 process criteria PASS:** off-path parity bitwise, bytes_ratio∈[0.0504,0.0506]=B2, B2 reproduced the band,
no ignition/OOM/divergence. Code verified GO (adversarial 8-agent workflow — every null is a trustworthy null).

**Mechanistic close:** B2 caps at parity because δ reconstructs the *dense gradient on stale data*. You
cannot exceed dense by reweighting (L3), accumulating (L2), perturbing (L4), or de-noising (L1) a stale
estimate of dense. To surpass, a signal dense genuinely lacks is required; no admissible lever on the
anchor-usage axis provides it. Full detail: `runs/EXP-31/verdict.md`.

## EXP-33 — β_anc sweep (issue #33, VERDICT = PASS, 2026-06-17)

**The β_anc axis is a flat free-averaging region: β=0 (freshness) is the right default.**

This was the first direct β→accuracy curve on the corrected valid-M circuit. All five cells ran clean
on the locked B2 substrate (PowerSGD r=77, delayed_ef λ=1, anchor cadence=delay_K=5,
replay_paired_batch=true) with only `spectral.beta_anc` varied.

**β→accuracy curve** (val@50 = `val-core/openai/gsm8k/acc/mean@1` at global_step 50, same box/seed):

| cell | β_anc | val@50 | gap vs C0 (0.73844) | interpretation |
|---|---|---|---|---|
| C0 b0p00 | 0.00 | **0.73844** | — (control) | reproduces B2 band [0.716,0.774] |
| C1 b0p25 | 0.25 | **0.73995** | +0.00151 | TIE (within ±0.024) |
| C2 b0p50 | 0.50 | **0.75284** | +0.01440 | TIE / nominal peak (NOT a SOTA promotion) |
| C3 b0p75 | 0.75 | **0.72176** | −0.01668 | TIE / mild down |
| C4 b1p00 | 1.00 | — (30-step bracket) | n/a | cold-M collapse: merger_coldM_fallbacks=196/196 → plain PowerSGD (degenerate) |

**Key findings:**
- **Freshness-best hypothesis SUPPORTED.** max(val@50[C1,C2,C3]) − val@50[C0] = +0.0144 ≤ +0.024 bar. No β>0 cell strictly beats β=0 beyond noise.
- **Flat free-averaging region.** β∈[0,0.75] all lie within ±0.024 of C0 — mild EMA averaging is neither harmful nor helpful. Consistent with the "freshness ≥ variance-reduction" design invariant.
- **β=1 degenerate cliff confirmed.** Cold-M freeze (M stays at zeros) → delayed_ef strict no-op → plain PowerSGD. Mechanism predicted and confirmed.
- **β is comm-neutral.** bytes_ratio identical all 5 cells (0.0504–0.0506, gate [0.0500,0.0510]).
- **promote_launcher_as: none.** B2 (= C0, β=0) stays the reference. No new launcher promoted.
- **blend_eta divergence noted (non-confound).** All 5 cells used blend_eta=0.5 vs B2 snapshot 0.3; dead under delayed_ef, identical across cells — no confound.

Full detail: `runs/EXP-33/verdict.md`. WandB project: `verl_compression_research_beta_sweep`.

## Milestone M6

M6 goal: validated, canonical communication-efficient GRPO launcher achieving ≥ dense parity (0.7536 ± noise) at materially lower gradient-comm cost.

**PASS experiments contributing to M6:**

- **EXP-30** (2026-06-13): K-delayed codec residual B2 (delayed_ef λ=1, β_anc=0) on valid-M anchor circuit with on-policy replay. val@50=0.7528 — first correction-carrying cell to convert without ignition, ZERO post-warmup emission across 50 steps. Success criteria checked: GATE-B2 OPEN (‖δ‖/‖G_comp_ring‖=1.05 ∈ [0.1,1.5]), bytes_ratio=0.0505, recon act-band, 196 targets, bitwise-identical with comm-eff OFF. Key metrics: val@50=0.7528, bytes_ratio=0.05052, max_mem=28.66 GB. PR #17 merged (ca5f4b002). Ground truth: `runs/EXP-31/B2_baseline/`.
- **EXP-33** (2026-06-17): β_anc EMA sweep {0, 0.25, 0.50, 0.75, 1.00} on B2 delayed_ef substrate — first direct β-curve on the valid-M circuit. All success criteria checked: off-axis parity (only beta_anc varies), identical bytes_ratio (0.0504–0.0506 all cells), act-band recon, all cells complete+stable, C0 reproduces B2 band, C4 cold-M collapse confirmed. Key metrics: β→accuracy flat (max gap C2 +0.0144 < ±0.024 bar), β=0 confirmed as the right default. Verdict: `runs/EXP-33/verdict.md`.

## "Done" means

Stable, ≥ dense parity (0.7536) within noise, with measured + materially-lower
inter-stage communication, reproduced by one canonical launcher. Full definition:
`.claude/GOAL.md`.
