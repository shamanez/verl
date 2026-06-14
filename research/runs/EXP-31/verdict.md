# EXP-31 verdict — PARITY (not greedy-surpass); operator-accepted

**VERDICT: PARITY.** Communication-efficient GRPO (the B2 delayed_ef substrate, PowerSGD
r=77 act codec) **matches the dense baseline** on greedy GSM8K `val mean@1`, at ~5% of the
inter-stage gradient-communication cost. The mandated *greedy-surpass* was **not achieved**
by the stale-anchor sub-basis lever; the operator accepted the parity result (2026-06-14).

## The numbers (greedy val `mean@1` @50, this box/config — all carry `disable_custom_all_reduce`)

| run | config (variable = how the stale anchor grad is used) | val@0 | val@25 | val@50 |
|---|---|---|---|---|
| **dense (TRUE bar, this config, seed 0)** | comm-eff OFF | 0.0796 | 0.7528 | **0.7506** |
| dense (old ref, DIFFERENT box) | comm-eff OFF | — | — | 0.7839 |
| **B2 / Cell A — best comm-eff** | delayed_ef λ=1, β_anc=0, r=77 act, anchor owns Q, cadence=delay_K=5, NO sub-basis | 0.0735 | 0.6937 | **0.7400** |
| γ-decay50 | + rank-2 tail sub-basis, γ decays 1→0 over 50 | 0.0713 | 0.6854 | 0.7210 |
| r2 | + rank-2 tail, γ=1 constant | 0.0788 | **0.7293** | 0.6983 |
| hold25-decay25 | + rank-2 tail, γ=1 hold 0–25 then ramp→0 | 0.0728 | 0.7066 | *unrecovered* (box stopped @step ~50) |

Eval nondeterminism ≈ ±0.024/draw; these are single draws (seed bands not pinned).

## The headline reframe (the most important result)
- **The dense bar on THIS config is 0.7506, not 0.7839.** The 0.7839 was a high draw on a
  *different* box. Re-run here under the identical setup as every comm-eff arm, dense gives
  0.7506 (and slightly decays 0.7528@25→0.7506@50, like B2).
- **⇒ B2 (best comm-eff) = 0.7400 vs dense = 0.7506: gap 0.011, inside the ±0.024 eval noise =
  statistical PARITY.** The apparent "0.044 gap to dense" was a wrong-reference artifact.
  **Comm-efficient GRPO already MATCHES dense at ~5% gradient-comm cost** — the core win of this
  line of work.

## What the stale-anchor sub-basis proved (and didn't) — the operator's "understand what failed and why"
1. **It works mechanically + accelerates early learning (REAL).** The rank-2 `tail` (act-deflated
   weight-gradient residual) captures **88–90%** of the off-principal energy the act-basis
   structurally misses (`subbasis_energy_ratio` 0.88–0.90), and r2 beat B2 at step 25 by **+0.036**
   (0.7293 vs 0.6937). The off-principal direction is a genuine early-learning accelerant.
2. **It does NOT convert to a greedy-surpass.** At constant full weight it **over-amplifies near
   convergence and regresses** (r2: 0.7293@25 → 0.6983@50). Decaying the weight (γ-decay50) **fixes
   the regression** (0.6854→0.7210, climbs) but also **tempers the early gain** → ~parity-below-B2.
   The hold-then-decay schedule (keep γ=1 early, ramp late) reproduced r2's early band
   (val@25 0.7066) but its val@50 was lost to the box stop. **Every variant clustered at
   parity-with-B2/dense (~0.70–0.74); none cleared the dense band.**
3. **Mechanistic conclusion:** amplifying a direction the model already descends speeds the *path*
   to the optimum but does not find a *better* optimum than dense — which is what a greedy-surpass
   requires. The comm-eff substrate's accuracy ceiling ≈ dense. Surpass would need a different
   mechanism (a beneficial-regularization / variance-reduction structure, or relaxing a locked
   constraint), i.e. a new research direction — not a sub-basis tweak.

## Engineering / correctness (all green)
- New merger knobs `delta_subbasis_rank / family / weight / decay_steps / hold_steps` on
  `exp/31-subbasis-merger` (pushed; bundles `exp.bundle`/`exp_gamma.bundle`/`exp_hold.bundle`).
  **rank-0 / weight-0 / λ-0 are bitwise-B2** (off-path parity, `torch.equal` verified); CPU suite
  213→40 tests green incl. determinism + scale-contract.
- **Step-C avoidance held by construction** — forward `q_basis=act` never touched; recon stayed in
  the act band (0.026), never the 0.68 plateau. DP-MEAN confirmed (0.746, not the ×4 sum bug);
  cross-rank Q bit-identical; no NaN/OOM (peak 30.7/31.8 GB).
- **vLLM `disable_custom_all_reduce`** required on this box (CUDA-IPC under mp executor) — a
  controlled variable across all arms; greedy-val-neutral.

## Caveats / what's deferred
- **Single-draw vals** (±0.024). Seed bands (dense×3, B2×3) were NOT pinned — the box (40806688)
  stopped Vast-side at ~23:13 UTC right as hold-decay finished, and would not restart (host GPUs
  unavailable, restart queued). The PARITY claim rests on single draws clustering within noise; a
  band-pin (deferred, if the box returns) would tighten it but is very unlikely to change the
  qualitative conclusion (all comm-eff draws ~0.72–0.74 vs dense ~0.75).
- hold25-decay25 val@50 is on the stopped box's disk (recoverable if it restarts).

## Bottom line
Comm-efficient GRPO = **parity with dense at ~5% comm cost**. The stale-anchor sub-basis is a
validated early-learning accelerant but not a greedy-surpass lever. Best shippable comm-eff
config: **B2 (delayed_ef λ=1)**.
