# DRAFT — successor issue to #18 (M5): SURPASS dense in PP comm-efficient GRPO
# (paper-independent scaffold; the off-principals theory section is filled after rlvr-theorist returns)

TITLE: M5 — SURPASS the dense GRPO baseline from a communication-efficient (pipeline-parallel masked) setting on GSM8K — a strictly iterative, theory-driven recursive search (NOT just match dense)

LABELS: research:claim, kind:experiment, milestone:M5

## Working mode (HARD, non-droppable — same spirit as #18, harder bar)
Strictly **iterative research**: the executing agent must reason **mathematically** from theory, propose a mechanism, run a real ≤50-step (then longer) experiment, read the result, theorize WHY, and propose the next mechanism FROM that evidence — an observe→theorize→propose→test loop. **Do NOT stop until the goal is achieved OR the candidate space + iteration budget is genuinely exhausted with a loggable negative result.** Every iteration backed by hardware evidence. Think like a top scientist: a real mechanism derived from the geometry of RLVR, not a knob sweep.

## THE STRICT GOAL (this is the whole point — and it is hard)
**SURPASS the dense GRPO baseline** — higher final GSM8K val (pass@1) AND/OR a better reward trajectory — while training in a **communication-efficient pipeline-parallel (activation-masking) setting**. Matching dense (M4/#18) is the floor now, not the target. Concretely: final GSM8K val(masked+method) > val(dense) by a margin beyond noise (target ≥ +1–2 pts pass@1, and/or higher pass@K), under the same control variables.

## Why this is plausible (not crazy) — the scientific bet
- RLVR (incl. GRPO) largely **elicits latent pretrained capability** rather than installing new skill: base-model **pass@K is on par with or exceeds** the RL model at large K (prior work) — RL sharpens/raises pass@1 by reshaping the distribution, often at the cost of diversity.
- The dominant failure mode of RL fine-tuning is **entropy collapse** — the policy sharpens too fast, loses exploration/diversity, caps gains and hurts pass@K.
- **Masking is fundamentally a perturbation of the activation/gradient geometry and entropy.** So a masked update is not merely a lossy dense update — it is a *different* update with potentially *better* regularization properties. The bet: a **geometry-aware, entropy-preserving** masked update can beat dense by exploring the right (off-principal) subspace and resisting entropy collapse — i.e. masking as a feature, not a bug.

## STRICT constraints (non-negotiable — the comm-efficient envelope)
- **Pipeline-parallel communication-efficient setting is mandatory**: per-(token,channel) activation masking at the pipeline-stage boundaries stays ON (this is the regime we must win in). The method must reduce / not exceed dense's inter-stage comm. No solution that quietly reverts to dense full-comm.
- **Fixed control variables (from CLAUDE.md / project.yaml):** Qwen2.5-1.5B-Instruct; GSM8K; vanilla GRPO (no DAPO/GSPO); MAX_RESPONSE 16384; 4–8 GPU H100/H200 on the locked Vast template. Single-GPU forbidden.
- Report **net inter-stage comm vs dense** for any PASS.
- If the method needs a stale full-gradient anchor, it stays STALE (realistic pipeline latency) — carry over #18's delay_K discipline; no fresh full gradient applied as the update.

## What we already know — the #18 / M4 lessons (the starting point)
1. **The proven base:** masked (p=0.9, rescale ON) + a continuous **blend toward a CLEAN stale (delay_K=5) policy-gradient anchor** (cadence=5, NO clean step) RECOVERS dense-level learning (reward 0.13 floor → ~0.81–0.84). `correction_mode=blend`, blend_eta≈0.7–0.9, beta_anc=0, ema_device=cpu; code on `exp/18-anchorcleangrad-c5d5` (PRs #11/#12). This is the most stable known starting point — build from here.
2. **The anchor must emit the TRUE gradient.** Two bugs had hidden this (random-weight FSDP clone; importance-ratio corruption from masked old-logprobs vs unmasked forward). Fix = `anchor_pg_loss` = plain policy gradient (ratio≡1). See [[anchor-clone-fsdp-naming-bug]].
3. **ADD fails, BLEND works.** Adding an orthogonal force at full magnitude destroys the policy; a convex blend (replace, stable magnitude ≤‖G_mask‖) recovers it. Orthogonality is real: cos(G_mask, M_anchor)≈0.
4. **The residual gap to dense (the M5 entry point):** the blend does NOT surpass dense — it (a) lags ~0.15 below dense in steps 1–15 (cadence-5 anchor warmup) and (b) sits a **persistent ~0.04 BELOW dense in the plateau** (steps 20–50). Closing AND reversing this gap is the M5 problem.
5. Validation was OFF in #18 (training-signal match only). **M5 must measure val/pass@1 + pass@K + entropy** — surpassing dense is a generalization claim, not a training-curve claim.

## Theoretical levers — from "RLVR Provably Learns Off the Principals" (2511.08567)
<<FILL FROM rlvr-theorist NOTE: refs/2511.08567_notes.md — the Three-Gate Theory (KL anchor / model geometry / precision), off-principal vs principal weight updates, spectral drift / principal-subspace rotation / off-principal alignment, entropy connection, and the concrete geometry-aware + entropy-preserving levers + metrics for surpassing dense in the masked setting>>

## Candidate directions to seed the planner (OPEN — the agent must derive its own from the geometry)
<<FILL FROM rlvr-theorist NOTE: 3–6 concrete, testable hypotheses, each with mechanism / why-it-could-surpass-dense / comm-efficiency / confirming metric>>

## Success criteria (machine-checkable — the planner tightens)
- [ ] Final GSM8K val (pass@1) of masked+method **strictly exceeds** dense by a margin beyond seed noise (target ≥ +1–2 pts; run ≥2 seeds or a longer run to establish significance).
- [ ] (Generalization) pass@K and/or token-entropy of masked+method ≥ dense — evidence it resists entropy collapse / preserves diversity.
- [ ] Geometry metrics logged (spectral drift, principal-subspace rotation, off-principal alignment) and consistent with the proposed mechanism.
- [ ] Net inter-stage comm ≤ dense (the comm-efficient envelope holds).
- [ ] grad_norm finite, no NaN/Inf; constraints (mask actor-train-only, stale anchor if used) verified.

## Compute budget (inherit #18 defaults; planner may adjust)
gpu_filter_chain 4×H200 → 8×H100; max_dph 24; max_gpu_hr 96 for the whole search; iterations ≥ 3 (recursive); box reuse across candidates.

## Dependencies
depends_on: [18] (PASS) — M5 builds on the proven clean-stale-anchor blend.
