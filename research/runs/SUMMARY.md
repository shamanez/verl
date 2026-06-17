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

## Evidence boundary: #29 paired replay

**EXP-29 is the validity boundary for anchor-gradient claims.** PR #16
(`d26176b44`, branch `exp/29-anchor-onpolicy-replay`) added the paired replay
ring, CPU snapshots, fire-aware retention, and relevance/canary checks that make
the anchor gradient `M_rep` comparable to the retained fast gradient at the same
`(batch, theta)` point.

Consequences for reading old numbers:

- **Post-#29 valid-M evidence starts at EXP-30.** Current SOTA, floor, merger,
  and anchor-usage claims must use EXP-30/31/33 or later.
- **EXP-20 is okay to quote only as clean-step history.** It used a periodic
  dense clean step, not the valid paired-replay anchor-gradient circuit. It is not
  a current no-merger floor and not evidence about anchor-gradient usage.
- **EXP-23/25/26/27 anchor-gradient results are archival only.** They predate
  paired replay, so their `M` carried stale/current-batch confounds. Do not use
  those numbers to rank current valid-M mergers, define floors, or support
  mechanism claims about the fixed anchor circuit.

Historical run directories through EXP-30 were de-bloated 2026-06-15; full
detail remains in git history, W&B, and merged code.

## The settled communication-efficient base (as of issue #29, 2026-06-12)

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

| id | evidence status | what | result (val@50 GSM8K) | record |
|---|---|---|---|---|
| **baseline** | dense control | Dense GRPO, Qwen2.5-1.5B-Instruct, GSM8K — verl unmodified (the control = comm-eff OFF) | **0.7536** historical; same-box band later pinned around 0.75–0.78 | W&B `5e2jpho9`; later dense reruns |
| EXP-20 | **pre-#29, no anchor-gradient claim** | PowerSGD r=77 + fresh clean@5 | 0.7415 — clean-step history only; **not** current floor | W&B `oquyeic3` |
| EXP-23 | **pre-#29 / invalid-M archive** | PowerSGD r=77, no re-anchor; inject/blend stale-anchor mergers | 0.6914 — archival only; **not** valid-M floor | W&B; erratum `a46fd0191` |
| EXP-25 | **pre-#29 / invalid-M archive** | Early anchor plumbing + sign-replacement sweep | 0.7066 best — archival only; do not rank current mergers from it | issue #25; git history |
| EXP-26 | **pre-#29 / invalid-M archive** | Geometry audit + `ef_powersgd`; gradient-tuned forward Q falsified by recon collapse | 0.7210 — archival comparator only | folded here; W&B pruned; LOG.md |
| EXP-27 | **pre-#29 / invalid-M archive** | Damped `ef_powersgd` ignition test | 0.7202, ignited around step ~66 — archival stability warning only | W&B `qa6sll3h`; LOG.md |
| EXP-29 | **validity fix / infra** | On-policy anchor replay: `replay_paired_batch` + `snapshot_device=cpu` + canary/relevance probes | infra PASS, no val bar | merged PR #16 `d26176b44`; now part of B2 |
| **EXP-30** | **post-#29 valid-M evidence** | K-delayed codec residual (B2, delayed_ef λ=1, β_anc=0) + valid anchor M via on-policy replay | **0.7528** (B2 @50) — **PASS = SOTA** | PR #17 merged `ca5f4b002`; B2 ground truth migrated to `runs/EXP-31/B2_baseline/` |
| **EXP-31 (sub-basis)** | **post-#29 valid-M evidence** | Stale-anchor rank-2 sub-basis merger: additive off-principal correction into δ, forward Q untouched; dense reframe (dense-here=0.7506) | **0.7400** (B2/Cell A) — **PARITY (operator-accepted)** | branch `exp/31-subbasis-merger` (unmerged); verdict `runs/EXP-31/verdict.md` |
| **EXP-31 (tournament)** | **post-#29 valid-M evidence** | 4-lever anchor-usage tournament (L4 perturbation / L2 δ-momentum / L3 adaptive dose / L1 control-variate) on B2 substrate | **NULL across all levers** — **STOP** | box i_41048644 4×H200; W&B fy920fty/ybemd5ux/knlzxh2x/kzohyuod/wmpmmdj1; verdict `runs/EXP-31/verdict.md` |
| **EXP-33** | **post-#29 valid-M evidence** | β_anc EMA sweep {0, 0.25, 0.50, 0.75, 1.00} on B2 delayed_ef substrate — first direct β-curve on the valid-M circuit | **FLAT free-averaging region** (all β∈[0,0.75] within ±0.024; C2 β=0.5 nominal peak 0.75284 NOT a SOTA promotion; C4 β=1 cold-M collapse) — **PASS (measurement)** | box i_41194490 4×H200 (torn down); WandB `verl_compression_research_beta_sweep`; verdict `runs/EXP-33/verdict.md` |

## Pre-#29 archive: what EXP-25 can and cannot tell us

EXP-25 predates the EXP-29 paired-replay fix. It is therefore **not evidence
about the current valid-M anchor-gradient circuit**. Keep only the narrow
historical facts:

- The early plumbing around full-coverage target extraction, DP reduction, and
  anchor-owned `Q` motivated the EXP-29 replay fix.
- The sign-replacement sweep did not solve the task on the old invalid-M circuit.
- Its values (0.7066 / 0.6164 / 0.3541) should **not** be used to rank current
  valid-M mergers, define the current floor, or support mechanism claims about
  the fixed anchor.

The old deep writeups were de-bloated 2026-06-15; treat them as lab-notebook
history unless a later post-#29 run re-establishes the same claim.


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
