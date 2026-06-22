# Research brief — stable use of stale anchor gradients in communication-efficient GRPO

> Working brief for an agent-team research effort. ALL artifacts (this plan, every
> teammate's notes, and the final report) live under `research/claude-findings/`.
> The final deliverable `research/claude-findings/report.html` is publication-grade —
> it will be shared with world-class research scientists.

## ROLE
You are the TEAM LEAD for a theory-research effort whose output will be read by world-class research scientists. Run it as a Claude Code agent team: spawn the teammates named below, give each its own dedicated output directory under `research/claude-findings/`, let them work in parallel in their own context windows, and have them challenge each other's claims before you synthesize. (Agent teams are experimental — if the session reports they are disabled, tell the operator to set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings.json and restart; otherwise fall back to running each member as a sequential subagent into the same directories.) This brief is SELF-CONTAINED: teammates do not inherit this conversation, so pass each one the context it needs in its spawn prompt, and point it at the prior-art files below.

## THE SYSTEM (what is being built)
A communication-efficient, pipeline-parallel GRPO trainer (Qwen2.5-1.5B-Instruct, GSM8K/Big-Math, vanilla GRPO, no-KL no-entropy). Two circuits:
- **FAST circuit** — the online policy. Compresses pipeline-boundary activations with a PowerSGD-style rank-r frozen-basis projector (M̂=(M@Q)@Qᵀ, r≈77 of H=1536). Its gradient is BIASED but functional.
- **SLOW "ANCHOR" circuit** — at cadence C it runs ONE unmasked full-gradient forward/backward from a `delay_K`-tick-STALE weight snapshot on the paired stale rollouts those weights generated (ratio≡1 clean PG loss), producing a clean reference gradient M. The anchor also OWNS the projection basis Q. A "merger" folds M into the fast gradient before `optimizer.step()`. With the method off, training is byte-identical to dense GRPO.

Read `CODE_WALKTHROUGH.md` and `research/.claude/GOAL.md` for the exact wiring and the project's definition of "done" (stable / parity / measured savings / one launcher).

## THE PROBLEM (the empirical wall — already measured, do NOT re-run GPU)
At low latency this works; at high latency it collapses.
- **K=5 (low):** STABLE, ≈ dense parity. delayed_ef merger → val@50 ≈ 0.7528 (dense band ≈ 0.76–0.78). signed_ema → ≈ 0.735.
- **K=20 (high):** BOTH mergers fail — ONE cause, TWO symptom geometries:
  - signed_ema (sign-replacing): catastrophic IGNITION — response length explodes to ~683 tokens, entropy 0.81→0.42, grad_norm 2→32, val crashes 0.648→0.44 (BELOW the no-merger floor 0.63 ⇒ the merger is actively destructive).
  - delayed_ef (additive, direction-preserving): non-convergent DRAG — no crash, but a sub-baseline ≈0.61 plateau (val@75 0.581), grad_norm creeps 2→55 as the held stale δ tracks policy drift.

"The symptom is set by merger geometry (sign-replace → ignite; additive → drag); the cause is shared: K>τ off-policy staleness."

## THE THEORY SO FAR (your starting point — verify, formalize, extend; cite the source artifact for each)
1. **TWO-GAP DECOMPOSITION** (`reports/comm-eff-grpo/why-grpo-fails-sft-works.html`). The transplanted anchor gradient carries two independent mismatches; SFT has only the first:
   - (a) parameter-point gap: ‖g(θ_{t−K}) − g(θ_t)‖ = O(K·‖Δθ‖·L)  [curvature only]
   - (b) distribution gap: trajectories + group-relative advantages are from a policy the current one may never generate.
   Under SFT gap (b) ≡ 0 (frozen dataset, no policy) → the stale gradient is a lagged estimate of the SAME descent direction (latency-tolerant, telescoping). Under GRPO both gaps stack.
2. **OFF-POLICY AS BIAS, NOT VARIANCE.** The anchor is a valid, self-consistent, LOW-VARIANCE on-policy gradient of the OLD policy π_{θ_{t−K}}, transplanted into θ_t WITHOUT importance-sampling reweighting. Textbook off-policy converts staleness into variance (noisy, unbiased, averages out); this converts it into BIAS (a precise estimate of the WRONG gradient). "Low variance plus persistence is more dangerous than noise."
3. **THE RL IGNITION CHANNEL.** A persistent low-variance off-policy carrier only ignites if there is a reward-hackable direction. GRPO (verifier-only reward, group-normalized advantage, no-KL no-entropy) leaves RESPONSE LENGTH nearly free/reward-flat, and token-mean loss aggregation gives ~86× tail amplification → the carrier rectifies along "correct-but-longer" until a ratchet locks. The killer is LENGTH-HACKING, not low entropy (dense is the lowest-entropy AND most-stable run; entropy is a follower).
4. **THE σ(M) CEILING** (`research/runs/SUMMARY.md`, `.claude/GOAL.md`). Any deterministic Φ(G_comp, M) is σ(M)-measurable ⇒ capped at dense PARITY. You cannot beat dense by reweighting / accumulating / de-noising a stale estimate of the dense gradient (EXP-31 tournament was all-null). Surpass requires injecting information OUTSIDE the span of the stale+current dense means: (i) curvature/2nd-order, (ii) conversion-positive exploration that moves the greedy argmax, (iii) cross-rank 2nd moment (disagreement-as-objective). The DIAGONAL TRAP: the control is dense-Adam (diagonal v_t), so any correction that collapses to a per-coordinate rescale is "a better Adam diagonal" and stays at parity — escape needs beyond-diagonal structure.
5. **TASK-DEPENDENT STALENESS BUDGET** (`runs/EXP-38/verdict.md`, `reports/dense-run-behaviour/*findings.json`). cos(g_t, g_{t−k}):
   - GSM8K (easy): 0.507(k1) → 0.176(k5) → 0.023(k10, ~dead) → −0.008(k20). τ ≈ 3–4 ticks; half-life ≈ 5 ticks.
   - Big-Math (hard): 0.0175(k1, already at noise floor) → 0.011(k5). ZERO usable window.
   Weight half-drift ≈ 7.9 global steps (~16 ticks; 2 optimizer ticks/global step), but pg_clipfrac / ppo_kl / response_length reach half-drift in ~1 global step ⇒ the DISTRIBUTION gap (b) dominates, not curvature×‖Δθ‖.
6. **FORWARD/BACKWARD ASYMMETRY** (same source). Forward activation h is rank-1 (top-1 ≈99% energy), its subspace overlap FLAT across k=1–40 (~0.77) ⇒ Q is intrinsically staleness-tolerant. Backward grad_h rank-for-90% = 105 (GSM8K) / 180 (Big-Math) ⇒ task-dependent, exceeds r=77. Recommendation on file: compress in ACTIVATION space, split forward/backward codec budgets, recast the anchor as a slow cross-rank-identical Q/CODEC CALIBRATOR rather than a gradient provider.
7. **THE NEWEST DIRECTION — anchor-gradient extrapolation** (`reports/anchor-future-projection/theory-and-literature-2026-06-22.md` + `discussion-2026-06-22.md`). Learn R_K: g(θ_{t−K}) ↦ g(θ_t) that "un-rotates" the stale gradient forward. The stale→live rotation IS the Hessian-vector product H·Δθ (first-order Taylor; measured norm ratio ≈1.0 ⇒ pure ROTATION not rescale; ‖Δθ‖≈0.0009/step yet cosine drops to 0.51 in one step ⇒ H large/ill-conditioned). Use the periodic anchor ground-truth refresh as supervision + error feedback. Must clear the diagonal trap to be more than parity.
8. **ASYNC-REALISM CONSTRAINT** (`.claude/GOAL.md`). The real target is a single SLOW anchor node serving a fast SWARM ⇒ the anchor ALWAYS lags, never leads. Admissible levers use it as a LAGGING reference, tolerate VARIABLE staleness, and stay CROSS-RANK-IDENTICAL. There is an OPEN tension: does "trajectory-continuation" extrapolation count as admissible, or is it forbidden anchor-lead? Resolve this explicitly.

**PRIOR-ART FILES the team must read before searching (avoid rediscovery):**
- `reports/comm-eff-grpo/{why-grpo-fails-sft-works.html, rollout-corr-off-policyness.html}`
- `reports/anchor-future-projection/{theory-and-literature-2026-06-22.md, discussion-2026-06-22.md}`
- `reports/dense-run-behaviour/{exp38-dense-drift-gsm8k_findings.json, exp38-dense-drift-big-math_findings.json, _joint_narrative.html}`
- `runs/DELAY_ANALYSIS/{staleness_theorist.md, cadence_analyst.md}`
- `runs/EXP-38/verdict.md` ; `runs/Q_BASIS_ANALYSIS/Q_BASIS_REPORT.md` ; `runs/SUMMARY.md` ; `.claude/GOAL.md` ; `CODE_WALKTHROUGH.md`

**SEED LITERATURE (already found — extend, don't stop here; find ≥6 more, 2025-or-later):**
- Ajanthan et al., "Nesterov Method for Asynchronous Pipeline-Parallel Optimization," Pluralis 2025, arXiv:2505.01099 (fixed linear NAG look-ahead delay corrector — the seed).
- Zheng et al., DC-ASGD, ICML 2017, arXiv:1609.08326 (delay compensation via 1st-order Taylor + diagonal Hessian).
- Karimireddy et al., "Error Feedback Fixes SignSGD…," ICML 2019, arXiv:1901.09847 (EF for biased compressors — grounds delayed_ef).
- Yang et al., PipeMare, MLSys 2021, arXiv:1910.05124 ; Narayanan et al., PipeDream, 2018, arXiv:1806.03377 ; Zhang et al., Staleness-aware Async-SGD, 2015, arXiv:1511.05950 ; Vogels et al., PowerSGD, 2019, arXiv:1905.13727.

## THE OPEN QUESTION you must answer
Design a robust, theoretically-grounded way to use the anchor's full gradient at LARGE / VARIABLE K without ignition or drag — i.e. raise the usable staleness budget — while respecting the σ(M) ceiling (be explicit about whether each idea targets PARITY-at-larger-K or SURPASS), the diagonal trap, and the async constraint.

## PHASE 0 — write the plan FIRST
Before spawning anyone, read the prior-art files above and write `research/claude-findings/00-plan.md`: the decomposition of the question, each teammate's charter + dedicated directory, the web-search angles, the cross-checks (diagonal-trap probe, async-admissibility test, parity-vs-surpass label), and the acceptance gates from the goal. Then spawn the team.

## THE TEAM (4 teammates, each its OWN context window + dedicated dir under research/claude-findings/; tell each to write notes.md there; have them message each other to challenge claims, scientific-debate style)
1. **off-policy-theorist** → `research/claude-findings/01-off-policy-theory/notes.md`
   Charter: formalize the math. Derive the two-gap error bound and the bias-vs-variance characterization rigorously. State the K>τ stability condition formally and connect τ to the measured cosine-decay / policy-drift numbers. Survey off-policy policy-gradient theory: importance sampling & its variance, per-decision/truncated/clipped IS, V-trace / Retrace(λ), trust-region (TRPO/PPO) bounds as staleness controllers, control variates & variance reduction. Decide which (if any) give a principled reweight/correction for a STALE on-policy gradient with no fresh samples from π_{θ_t}.
2. **async-sgd-scholar** → `research/claude-findings/02-async-delayed-lit/notes.md`   [USE WEB SEARCH HEAVILY]
   Charter: the distributed/optimization literature on delayed & stale gradients — asynchronous SGD, staleness-aware scaling, delay compensation (DC-ASGD and successors), pipeline-parallel async (PipeDream/PipeMare/Nesterov-async), gradient compression + error feedback, local/periodic-averaging (Local-SGD/SlowMo/Lookahead). Prioritize 2025–2026 work. For each: mechanism, what it assumes about delay (fixed vs variable, known vs unknown), and whether it transfers to a NON-STATIONARY RL objective.
3. **multitimescale-rl-scholar** → `research/claude-findings/03-multitimescale-rl-lit/notes.md`   [USE WEB SEARCH HEAVILY]
   Charter: two-timescale/multi-timescale stochastic approximation (Borkar) and its convergence conditions; meta-gradients; hierarchical / fast-slow RL; slow-fast weight schemes (EMA teachers, Polyak/target networks, Lookahead, SWA) used as STABILIZERS; any RL work that splits an approximate fast learner from a precise slow learner. Prioritize 2025–2026. Extract the exact step-size-ratio / timescale-separation conditions that buy stability, and how alignment between the two updates is maintained.
4. **algorithm-architect** → `research/claude-findings/04-algorithm-design/notes.md`
   Charter: synthesize the other three into ≥3 concrete candidate algorithms for the open question. Candidates to evaluate (not exhaustive): (i) staleness-scaled / age-decayed dose on the merger (down-weight M with realized lag); (ii) trust-region cap on the correction's contribution; (iii) IS-reweighted anchor (and why it is hard with no fresh π_{θ_t} samples); (iv) learned extrapolation R_K ≈ H·Δθ that un-rotates the stale gradient (fixed-linear vs learned vs learned+EF), with the beyond-diagonal requirement; (v) multi-timescale optimizer with explicit timescale separation; (vi) the activation-space recast (anchor as Q-calibrator). For EACH: parity-or-surpass label, σ(M)/diagonal-trap verdict, async-admissibility verdict, and a cheap GPU-FREE offline kill-test (e.g. fit/evaluate on the existing EXP-38 drift tensors; threshold like "cosine lift at k=5 GSM8K 0.176→≥0.40"). Recommend ONE primary approach.

Run them in parallel; require at least one explicit cross-challenge round (theorist vs architect on the diagonal trap; async-scholar vs multitimescale-scholar on whether fixed-vs-variable-delay results survive). Then synthesize.

## DELIVERABLE (audience: world-class research scientists — publication/presentation grade)
`research/claude-findings/report.html` — a single self-contained HTML file (inline CSS/JS, no external assets/CDNs/fonts/network calls; math rendered legibly via inlined MathJax or clean Unicode/MathML; wide tables/figures wrapped in `overflow-x:auto`; professional typography; a concise `<title>`). Write to the standard of a paper appendix or a research memo you would hand to a senior scientist: precise definitions, explicit assumptions, honest limitations, no hand-waving, no marketing tone. Sections: Executive summary → Problem & system → Mathematical analysis (the two-gap derivation, off-policy-as-bias, K>τ + task-dependent budget, σ(M) ceiling & diagonal trap) → Literature review (≥12 cited, ≥6 from 2025+, with relevance notes & working links) → Candidate algorithms with the three kill-checks each → Recommended approach + its falsifiable offline kill-test + the parity-vs-surpass call + the async-admissibility resolution → Provenance table (finding → source artifact) → Limitations & open questions. Keep the four teammates' notes.md files in place under `research/claude-findings/` as the audit trail.

## GUARDRAILS
- Do NOT relitigate the settled substrate (anchor-on-PowerSGD, paired replay, anchor-owns-Q). Judge ideas on theoretical soundness, not on re-deriving what EXP-29/30 proved.
- Cite a source for every empirical number (file path or paper). Flag anything you assert without one as "unverified" — never invent a citation or a result.
- Stay inside `research/claude-findings/` for ALL writes; this is a theory/analysis task — no GPU runs, no edits to `verl/` source.
- If web search is unavailable for a teammate, it must say so in its notes rather than inventing citations.
