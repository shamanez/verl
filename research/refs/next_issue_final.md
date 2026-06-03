## Working mode — strictly ITERATIVE, theory-driven research (HARD, non-droppable)

This is a pure-research cycle run as a recursive sequence of real training experiments (same contract as #18, **harder bar**). The executing agent MUST:
- reason **mathematically** from the gradient/optimization geometry (not knob-sweep), propose a concrete mechanism, run a real ≤50-step experiment (then a longer confirmation run), read the result, theorize WHY (tie to the off-principal / entropy / staleness theory below), and **propose the next mechanism FROM that evidence** — observe → theorize → propose → test;
- **NOT stop** until the goal is achieved OR the candidate space + iteration budget is genuinely exhausted with a real, loggable negative result. Every iteration backed by hardware evidence. Think like a top scientist cracking a hard problem.

## THE STRICT GOAL — SURPASS dense (not just match)

**Beat the dense GRPO baseline while training in a communication-efficient pipeline-parallel (activation-masking) setting.** Matching dense (the M4/#18 result) is now the *floor*, not the target.

- **Primary win condition:** masked+method **strictly exceeds** dense on a generalization metric beyond seed noise — most plausibly **pass@K (K∈{8,16}) and/or a held-out / harder slice**, and ideally **GSM8K val pass@1**.
- *Honesty (from the paper read, §6):* a strict **val-mean** surpass is ambitious — the paper's own off-principal restriction only *edges* dense on 1–2 metrics (Math500 +1.0, AIME24 +2.0) and is ≈parity on average. So the **highest-confidence surpass axis is pass@K / generalization** (RL mostly *elicits* latent pretrained skill → preserving breadth should beat dense's over-sharpened mode). Target a surpass on pass@K/held-out first; a val-mean surpass is the stretch goal.

## Why this is plausible — the scientific bet

1. **RLVR mostly elicits, doesn't install.** Base-model pass@K ≈/≥ RL pass@1 at large K; RL raises pass@1 by reshaping the distribution, often *losing diversity*. The enemy is **entropy collapse** (no-KL no-entropy GRPO on easy GSM8K over-exploits the elicited mode).
2. **Masking is a perturbation of activation/gradient *geometry and entropy*, not just a lossy dense gradient.** So a masked update is a *different*, potentially *better-conditioned* RL step.
3. **The off-principals theory (below) says dense is not the optimum of RL's geometry.** RL *wants* to learn off the principal singular directions (spectrum-/entropy-preserving); dense (the full gradient) spends some budget on principal directions it doesn't need. A masked + geometry-aware correction can be off-principal *by construction* and entropy-preserving *by construction* — a strictly better-conditioned RL step that could surpass dense, **because** of the masking.

## STRICT constraints (the comm-efficient envelope — non-negotiable)

- **Pipeline-parallel activation masking stays ON** (per-(token,channel) at the 7 PP boundaries, p=0.9, rescale ON). This is the regime we must *win in* — no solution that quietly reverts to dense full-comm. **Report net inter-stage comm vs dense** for any PASS.
- Fixed control variables (CLAUDE.md / project.yaml): **Qwen2.5-1.5B-Instruct; GSM8K; vanilla GRPO (no DAPO/GSPO); MAX_RESPONSE 16384; 4–8 GPU H100/H200** on the locked Vast template; single-GPU forbidden.
- If a stale full-gradient anchor is used, it stays **STALE** (`delay_K≥1`, realistic pipeline latency; never a fresh full gradient applied as the update). Carry #18's HARD CONSTRAINTS 1–3 (`NEXT_RESEARCH.md`): no periodic clean step; staleness mandatory; the correction must SUPPLY the missing component, not reweight `G_mask`.

## What we already know — the #18 / M4 starting point (the most stable known base)

- **Proven base (build from here):** masked (p=0.9, rescale ON) + a continuous **convex blend toward a CLEAN stale (delay_K=5) policy-gradient anchor** (cadence=5, NO clean step) RECOVERS dense-level learning: reward 0.13 floor → **~0.81–0.84**. Knobs: `correction_mode=blend`, `blend_eta≈0.7–0.9`, `beta_anc=0`, `ema_device=cpu`; code on `exp/18-anchorcleangrad-c5d5` (PRs #11/#12).
- **The anchor must emit the TRUE gradient** — `anchor_pg_loss` = plain policy gradient (ratio≡1) on real stale weights (two bugs had hidden this: random-weight FSDP clone; importance-ratio corruption). See `runs/EXP-18/candidates.md` §4.
- **ADD fails, BLEND works** (replace at stable magnitude ≤‖G_mask‖; orthogonality is real, `cos(G_mask,M_anchor)≈0`).
- **The residual gap to dense = the M5 entry point:** the blend does NOT surpass — it lags ~0.15 below dense in steps 1–15 (cadence-5 anchor warmup) and sits a **persistent ~0.04 BELOW dense in the plateau** (steps 20–50). M5 must close AND reverse this.
- **#18 measured only the training signal (val OFF). M5 must measure val / pass@1 / pass@K / entropy** — surpassing dense is a *generalization* claim.

## Theoretical levers — "RLVR Provably Learns Off the Principals" (arXiv 2511.08567)

Full team-member read: **`research/refs/2511.08567_notes.md`** (rigorous, 254 lines, grounded in our code + #18 curves). **Calibration win:** the paper's RL recipe is *nearly identical to ours* — DAPO **β=0 (= our no-KL GRPO)**, verifiable ±1 reward, AdamW lr=1e-6, verl+vLLM+FSDPv2 bf16/fp32-optimizer, spectral figures on **DS-Qwen-1.5B (our scale)** — so the theory transfers credibly.

- **Three-Gate Theory.** Gate I (KL leash): even at β=0, clip imposes an O(ε²) per-step KL bound ⇒ a small, **curvature-weighted** weight move (Fisher-norm budget). Gate II (model geometry): the pretrained spectrum *steers* that bounded step **off the principal singular directions** into low-curvature, spectrum-preserving subspaces (Wedin sin-Θ rotation bound; Weyl/Ky-Fan spectral stability). Gate III (precision): bf16 hides sub-ULP off-principal micro-updates → looks like "sparsity" (an artifact, not a cause).
- **What RLVR changes vs SFT (parameter-space):** RLVR — updates land **off-principal / on low-magnitude weights**, spectrum **preserved** (NSS≈1e-5), small principal rotation. SFT — targets **principal** weights, **distorts** the spectrum (NSS≈1e-1), large rotation, and on RL's own objective even **lags** RLVR. ⇒ RL is a *distinct optimization regime*; SFT-era PEFT (principal-targeted LoRA/PiSSA, sparse-FT) is the wrong tool (destabilizes by forcing principal moves).
- **Off-principal ⇔ low-curvature ⇔ spectrum-preserving ⇔ entropy-preserving** are the *same condition viewed four ways* (Gate I×II): a low-KL/low-Fisher step can't collapse entropy in one step; the principal directions carry the dominant input→output gain (the entropy leverage), so leaving them alone preserves entropy. **The paper does NOT measure entropy — that is OUR edge.**
- **Existence proof for surpass (thin but real):** the paper's off-principal-restricted RL mask (`M_low ∪ M_princ^c`) tracks dense's KL curve and **edges dense** on Math500/AIME24 — *freezing principal weights doesn't hurt, sometimes helps*. We must *show our activation masking realizes an off-principal parameter update* (not assumed — measured; see H1).
- **Two surpass mechanisms (note §4.2):**
  - **A — masking as a free off-principal projector + correction as the de-biaser:** blend toward the **off-principal projection** of the stale true gradient (withhold the principal-direction correction) → off-principal + entropy-preserving learning, purer than dense → better generalization/pass@K.
  - **B — masking as an entropy-preserving regularizer:** p=0.9 masking injects unbiased (rescaled) forward variance ≈ an implicit entropy bonus (which we lack) → resists collapse → higher pass@K than dense. (Caveat §6: at β=0 the KL-leash entropy argument is *softer*, so H3 may have to *carry* entropy preservation — which is exactly why it could matter more for us.)

## Candidate directions to seed the planner (from note §5 — agent must derive/extend its own)

All respect the PP comm-efficient constraints; all build on the working C4/C5 clean-PG-anchor blend.

- **H1 — Measure-first (prerequisite diagnostic, gates everything).** Instrument a masked+C4 run + a cached dense run; compute NSS, top-k principal rotation, off-principal update fraction `ρ_princ`, and `b̂` principal fraction on the 7 boundary matrices. Decides whether masking is already an off-principal projector (→ pursue H2/H3) or principal-distorting (→ H4).
- **H2 — Off-principal correction (★ primary off-principal surpass lever).** `G_corr=(1−η)G_mask + η·scale·(M_anchor − P_princ(M_anchor))`, `P_princ` = top-k (k≈32) singular subspace of the boundary matrix's **base** weights (computed once; cheap). Supply only the off-principal ascent direction; withhold principal moves → smaller NSS/rotation, higher entropy than dense → better generalization. Confirm: plateau reward ≥ dense **and** `ρ_princ` < dense **and** val/pass@8 ≥ dense.
- **H3 — Entropy-preserving masked update (★ parallel entropy surpass lever).** Keep masking as the forward entropy bonus; tune `η`/`p`(-schedule) so masked policy **token entropy stays a margin above dense** while reward tracks. Confirm: entropy(masked) > entropy(dense) at steps 25/50 **and** pass@8/16 + val above dense. (Refuted if higher entropy comes with lower reward.)
- **H4 — Explicit principal-`b` removal (conditional on H1 flagging distortion).** C3-style stale `b̂=G_anchor^mask−G_anchor`, subtract only its principal projection `G_corr=G_mask−λ·P_princ(b̂)` (+ C4 off-principal blend). Kills the SFT-like principal-distorting drift.
- **H5 — Staleness-rotation fix (close the plateau gap → open room to surpass).** Replace the EMA anchor with first-order **gradient extrapolation** `M̂=G_anchor(t)+[G_anchor(t)−G_anchor(t−cadence)]`; and/or sweep `delay_K∈{1,2,5}`, `beta_anc∈{0,0.5}`. The ~0.04 plateau gap is most likely 5-step-stale principal-rotation lag — cancel it and the only thing below dense in the plateau goes away.
- **H6 — Warmup fix (strict-match cleanup; cadence is plan-pinned so flag as follow-on).** anchor `cadence=1` for a short warmup so the correction engages from step 1 and matches dense's steep 0→10 climb (the sole reason #18 missed whole-trajectory mean|Δ|≤0.05).

**Suggested order:** H1 (gate) → H5 (cheap, close plateau gap on the working C4) → H2 ‖ H3 (the two surpass bets, run with full instrumentation vs a cached dense reference) → H4 (only if H1 flags principal distortion) → H6 (strict-match cleanup).

## Metrics to LOG (note §4.4 — cheap, on the 7 boundary matrices in fp32, every ~5 steps or at {1,10,25,50})

Geometry: (1) spectrum drift `NSS=‖σ(Wₜ)−σ(W₀)‖/‖σ(W₀)‖`; (2) top-k principal-subspace rotation (sin-Θ); (3) off-principal update fraction `ρ_princ=‖P_U0k ΔW P_V0k‖_F/‖ΔW‖_F`; (4) live `cos(G_mask, M_anchor)` / `cos(G_corr, g_true)`; (5) `b̂` principal fraction. Entropy (our edge): (6) per-token entropy of the policy; (7) per-sequence NLL + dispersion across the n=8 rollouts (collapse monitor); (8) `pg_clipfrac` / IS-ratio histogram. Capability (the prize): (9) GSM8K val pass@1 every K steps; (10) **pass@K (K∈{1,8,16})** on a fixed 200-prompt slice, masked vs dense vs **base (zero-RL)**.
*Minimal set if budget-bound:* {NSS, ρ_princ, token-entropy, val, pass@8} on masked+correction vs a cached dense run at steps {1,10,25,50}.

## Success criteria (machine-checkable — planner tightens)

- [ ] **(surpass)** masked+method **strictly exceeds dense beyond seed noise** on pass@K (K∈{8,16}) and/or a held-out/harder slice (run ≥2 seeds or a longer confirmation run for significance); GSM8K val pass@1 ≥ dense (a val-mean surpass is the stretch).
- [ ] **(entropy/geometry consistency)** masked policy token-entropy ≥ dense at matched steps AND/OR `ρ_princ` (off-principal-ness) and NSS more favorable (≤) than dense — evidence the surpass comes from the off-principal/entropy mechanism, not noise.
- [ ] **(envelope)** net inter-stage comm ≤ dense.
- [ ] grad_norm finite, no NaN/Inf; constraints verified (mask actor-train-only; stale anchor if used; `clean_cadence=0`).
- [ ] On a surpass: draft PR (head `exp/<N>-<slug>`, base `vast-ai-workload`) promoting the geometry-aware correction; report the comm + the geometry/entropy evidence.

## Compute budget (inherit #18 defaults; planner may adjust)

gpu_filter_chain 4×H200 → 8×H100; max_dph 24; max_gpu_hr 96 (whole search); iterations ≥ 3 (recursive); box reuse across candidates; a longer confirmation run for the surpass claim.

## Dependencies
depends_on: [18] (PASS) — M5 builds on the proven clean-stale-anchor blend.

---
*Provenance: paper downloaded to `research/refs/2511.08567.pdf`, read end-to-end by a dedicated team member → `research/refs/2511.08567_notes.md`. #18 lessons from `runs/EXP-18/candidates.md` §4 + `runs/SUMMARY.md` + `findings/NEXT_RESEARCH.md`.*
