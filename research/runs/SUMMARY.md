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
- **The exact SOTA settings are the ground truth in** [`runs/EXP-30/resolved_params_B2.txt`](EXP-30/resolved_params_B2.txt)
  **+** [`runs/EXP-30/launch_B2.sh`](EXP-30/launch_B2.sh) — the substrate **every new arm holds fixed**.
- **Locked substrate:** PowerSGD **r=77** act codec · anchor **owns Q** · **cadence=delay_K=5** ·
  **clean_cadence=0** · **replay_paired_batch=true** · snapshot_device=cpu · batch128/mini64/lr1e-6/n8/
  resp16384/seed0 · **disable_custom_all_reduce=true** (required for the box to init; greedy-val-neutral).
- **Active frontier (issue #31):** push B2 → **0.80** by changing ONLY *how the stale anchor gradient is
  USED* — the **4-lever tournament** (perturbation / δ-momentum / adaptive-dose / control-variate) under
  the async single-slow-anchor constraint. Authoritative plan: [`.claude/plans/31.md`](../.claude/plans/31.md).

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
| **EXP-25** | M6 | **Anchor circuit default** — full-coverage DP-reduced stale `M` (R1) + anchor-owns-`Q` (R2) + signed_ema merger (R3), α swept | **0.7066** (best, α=0.5) — **STOP** | code on `vast-ai-workload`; issue #25 |
| EXP-26 | M6 | Real-gradient geometry audit + ef_powersgd merger; Step-C (gradient-tuned forward Q) **falsified** by recon collapse | 0.7210 (ef best) — REVISE→closed | folded here; W&B pruned; LOG.md |
| EXP-27 | M6 | Damped ef_powersgd (clip/decay 0.5) to 100 steps — ignition test | 0.7202 (**ignited @~66**, length-explosion) — STOP | folded here; W&B `qa6sll3h`; LOG.md |
| EXP-29 | M6 | On-policy anchor **replay** (`replay_paired_batch` + `snapshot_device` knobs) — the valid-M infra B2 uses | infra PASS (no val bar) | merged PR #16 `d26176b44`; now part of the B2 substrate |
| **EXP-30** | M6 | K-delayed codec residual (B2, delayed_ef λ=1, β_anc=0) + valid anchor M via on-policy replay (geometry-gated) | **0.7528** (B2 @50) — **PASS = SOTA** | PR #17 merged `ca5f4b002`; verdict `runs/EXP-30/verdict.md` |
| **EXP-31** | M6 | Stale-anchor rank-2 sub-basis merger: additive off-principal correction into δ, forward Q untouched; dense reframe (dense-here=0.7506) | **0.7400** (B2/Cell A) — **PARITY (operator-accepted)** | branch `exp/31-subbasis-merger` (unmerged); verdict `runs/EXP-31/verdict.md` |

## EXP-25 — what we learned (issue #25, VERDICT = STOP)

**The substrate is proven; the merger is not.** R1 (full-coverage, DP-reduced,
correct-scale `M`) and R2 (anchor-owns-`Q`, fast net never writes `Q`) both passed
their on-box probe gates — the anchor circuit is mechanically correct and is the
**realistic** setting (continuously-maintained stale anchor, no clean step). But
R3, the `signed_ema` merger `G = α·G_noisy + (1−α)·|G_noisy|·sign(M)`, **does not
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

## Frontier — how the stale anchor gradient is USED (issue #31)

The substrate (anchor on + owns `Q` + PowerSGD r=77) is **held fixed**; the single research axis is
**how the stale anchor `M` is folded into the fast compressed update**. The lineage settled it:
`signed_ema` falsified (sign-replacement is the defect, EXP-25) → **error-feedback on the PowerSGD
residual WON** and is the SOTA (delayed_ef = B2, EXP-30, parity with dense) — issue #24's premise,
realized. The open question is no longer *whether* a residual correction helps but **how to use the
anchor to BEAT dense, not just match it.**

The current program (issue #31) is a **4-lever tournament**, all changing only anchor-gradient *usage*
(codec/Q/batch/generation locked): **L4 perturbation** (zero-mean regularization), **L2 δ-momentum**
(accumulate the persistently-missed correction), **L3 adaptive dose** (trust the anchor more on
disagreement), **L1 control-variate** (cancel sampling variance, SVRG/SARAH). **Async-realism filter:**
the anchor is a single **SLOW** node serving a fast **SWARM** ⇒ always lagging, never leads — admissible
levers use it as a *lagging* reference, tolerate variable staleness, and stay cross-rank-identical.
Authoritative plan + thresholds: [`.claude/plans/31.md`](../.claude/plans/31.md). A standing watch on
entropy / response-length / IS-gap applies to every run (`research/diagnostics/ENTROPY_COLLAPSE_WATCH.md`).

## "Done" means

Stable, ≥ dense parity (0.7536) within noise, with measured + materially-lower
inter-stage communication, reproduced by one canonical launcher. Full definition:
`.claude/GOAL.md`.
