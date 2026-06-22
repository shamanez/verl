# 00 — Team-lead plan: stable use of stale anchor gradients in comm-eff GRPO

**Author:** team lead (synthesis owner). **Date:** 2026-06-22.
**Status:** written in Phase 0, BEFORE any teammate is spawned (per `BRIEF.md` §Phase 0).
**Deliverable:** `research/claude-findings/report.html` — publication-grade, self-contained.

> This file is the audit trail for how the question was decomposed, who was tasked
> with what, the cross-checks every claim must pass, and the acceptance gates. It is
> frozen as the v1 plan; teammate notes and the cross-challenge are separate files.

---

## 1. The open question (verbatim target)

> Design a robust, theoretically-grounded way to use the anchor's full gradient at
> **LARGE / VARIABLE K** without ignition or drag — i.e. **raise the usable staleness
> budget** — while respecting the **σ(M) ceiling** (explicit PARITY-at-larger-K vs
> SURPASS label per idea), the **diagonal trap**, and the **async constraint**.

The empirical wall (already measured, GPU-frozen):

| latency | signed_ema (sign-replace) | delayed_ef (additive) | source |
|---|---|---|---|
| **5/5** (K≈2.5 global steps) | STABLE, val@100 0.735 | STABLE, val@50 **0.7528** (near-dense) | EXP-37B / EXP-30 |
| **20/20** (K≈10 global steps) | IGNITION: len→683, ent 0.81→0.42, val→0.44 (below 0.63 floor) | DRAG: flat plateau ~0.61, grad_norm 2→55 | EXP-37 / EXP-37C / EXP-37E |
| dense control | — | — | val@100 **0.7832**, monotone (EXP-37D) |

One cause (`K>τ` off-policy staleness), two symptoms set by merger geometry.

## 2. Decomposition of the question

The question splits into four sub-questions, mapped to the four teammates:

- **Q1 (theory).** *What exactly goes wrong, formally?* Derive the two-gap staleness
  error; characterize off-policy-as-bias vs variance; state the `K>τ` stability
  condition and tie τ to measured cosine-decay/policy-drift. Decide whether any
  off-policy PG correction (IS, V-trace/Retrace, TRPO/PPO trust region, control
  variates) can reweight a stale *on-policy* gradient with **no fresh π_{θ_t}
  samples**. → `01-off-policy-theory/`
- **Q2 (async/optimization lit).** *What does the distributed-systems literature
  already know about delayed/stale gradients?* ASGD, staleness-aware scaling, DC-ASGD
  & successors, pipeline-async (PipeDream/PipeMare/Nesterov-async/AsyncMesh),
  compression+EF, local/periodic-averaging. Per method: delay assumption (fixed vs
  variable, known vs unknown) and whether it survives a **non-stationary RL**
  objective. → `02-async-delayed-lit/`
- **Q3 (multi-timescale/RL stabilizer lit).** *What does RL/optimization know about
  splitting a fast approximate learner from a slow precise one and keeping it stable?*
  Two-timescale stochastic approximation (Borkar), target/EMA networks, Lookahead/SWA/
  SlowMo, meta-gradients, fast-slow RL. Extract the **step-size-ratio / timescale-
  separation** conditions that buy stability and how the two updates are kept aligned.
  → `03-multitimescale-rl-lit/`
- **Q4 (algorithm synthesis).** *Given Q1–Q3, what concrete algorithms raise the
  staleness budget, and which one do we recommend?* ≥3 candidates, each with the three
  kill-checks + a GPU-free offline kill-test; recommend ONE primary. → `04-algorithm-design/`

## 3. Teammate charters + directories

Each teammate gets its OWN context window and dedicated dir; writes `notes.md` there.
All are told: this is theory/analysis only — **no GPU, no edits to `verl/`**, stay
inside `research/claude-findings/`; cite a source (file path or paper) for every
empirical number; flag anything unverified; if web search is unavailable, say so
rather than invent citations.

| # | Teammate | Dir | Web? | Core charter |
|---|---|---|---|---|
| 1 | **off-policy-theorist** | `01-off-policy-theory/notes.md` | light | Formalize two-gap bound, bias-vs-variance, `K>τ` condition; survey off-policy PG corrections; verdict on whether any reweights a stale on-policy grad w/o fresh samples. |
| 2 | **async-sgd-scholar** | `02-async-delayed-lit/notes.md` | **HEAVY** | Distributed/optimization delayed-gradient lit; per-method delay assumption + RL-transfer verdict. Verify + extend the seed bibliography. |
| 3 | **multitimescale-rl-scholar** | `03-multitimescale-rl-lit/notes.md` | **HEAVY** | Two-timescale SA + slow/fast stabilizers; extract exact timescale-separation conditions + alignment mechanisms. |
| 4 | **algorithm-architect** | `04-algorithm-design/notes.md` | light | Synthesize ≥3 candidates; each gets parity/surpass + σ(M)/diagonal-trap + async verdict + offline kill-test; recommend one. |

**Seed bibliography handed to scholars (avoid rediscovery):**
`research/codex-findings/paper-bibliography.md` (a parallel effort's web-searched set,
~35 sources w/ links + relevance labels, dated 2026-06-22) **and** `BRIEF.md` §SEED
LITERATURE. Scholars must (a) **verify each link they cite actually resolves** via
WebFetch, (b) **find ≥3 genuinely new** 2025+ sources beyond the seed, (c) mark any
unverifiable link.

## 4. The three mandatory kill-checks (every candidate algorithm)

1. **σ(M) ceiling → PARITY-or-SURPASS label.** `σ(M) = σ(g(θ_t), g(θ_{t−K}), …)`.
   Any deterministic `Φ(G_comp, M)` is σ(M)-measurable ⇒ **capped at dense parity**.
   Surpass needs info OUTSIDE σ(M): (i) curvature/2nd-order, (ii) conversion-positive
   exploration that moves the greedy argmax, (iii) cross-rank 2nd moment. Each idea
   labelled **PARITY-at-larger-K** (raises the budget, stays ≈dense) or **SURPASS**
   (claims info Adam lacks) — and *which escape category* if surpass.
2. **Diagonal trap.** Control is dense-**Adam** (diagonal `v_t`). Any correction that
   collapses to a per-coordinate rescale is "a better Adam diagonal" ⇒ stays at parity.
   Surpass requires **beyond-diagonal** structure (a genuine rotation / off-diagonal
   `H·Δθ`, or a variance-AS-OBJECTIVE term with a different fixed point). Probe every
   idea: *is the lift diagonal or off-diagonal?*
3. **Async-admissibility.** Real target = ONE slow anchor lagging a fast swarm ⇒ anchor
   **always lags, never leads**; corrections must be **cross-rank-identical** (derived
   from all-reduced sufficient statistics) and **tolerate variable staleness**. OPEN
   tension to resolve: does "trajectory-continuation extrapolation" count as admissible,
   or is it forbidden anchor-lead? (GOAL.md line 62 lists "no delay-compensation /
   anchor-lead"; the 2026-06-22 discussion doc argues continuation ≠ lead.)

## 5. Offline kill-test protocol (GPU-free; from the captures)

EXP-38 left `(θ, g)` pairs at multiple lags for both datasets
(`research/reports/dense-run-behaviour/*_findings.json`; raw tensors per the verdict).
Baselines to beat (raw, no correction): **GSM8K cos@k5 = 0.176, cos@k10 = 0.023;
Big-Math cos@k1 = 0.018** (near-orthogonal — the hard/stress case).

- **Kill threshold (proposed):** a correction must lift **GSM8K cos@k5 0.176 → ≥ 0.40**
  on held-out lags, or the idea dies cheaply (no GPU).
- **Diagonal-trap probe:** decompose each method's lift into diagonal vs off-diagonal;
  purely-diagonal lift ⇒ label "parity-only" up front.
- **Task split:** run GSM8K (feasibility) and Big-Math (stress) separately; never
  average. A GSM8K-only win is still a scoped result.
- **Cross-rank / variable-staleness dry-check:** confirm the form *could* be made
  cross-rank-identical and fit across a *range* of lags, not one fixed τ.

The architect designs the concrete kill-test for the recommended approach (what to
fit, on which tensors, the pass/fail threshold, the diagonal-trap decomposition).

## 6. Cross-challenge round (required by the brief)

After v1 notes exist, run an explicit adversarial round (`05-cross-challenge/`):

- **Debate A — diagonal trap (theorist vs architect).** Does the recommended
  curvature/extrapolation route genuinely clear the diagonal trap, or does it collapse
  to a better Adam diagonal? Reach a shared verdict.
- **Debate B — fixed-vs-variable delay survival (async-scholar vs multitimescale-
  scholar).** Do the fixed-known-delay results (Nesterov-async, DC-ASGD, PipeMare) and
  the timescale-separation conditions survive when delay is **variable & unknown** and
  the objective is **non-stationary RL**? What is the weakest assumption each relies on?

Each debate writes a short adversarial memo with a resolution (or a flagged open
disagreement). These feed the report's "Limitations & open questions."

## 7. Acceptance gates (from the goal / brief DELIVERABLE)

`report.html` is DONE only if it is self-contained (inline CSS/JS, no external assets/
CDNs/fonts/network), math legible, wide tables wrapped, and contains:

1. ✅ two-gap staleness-error derivation + off-policy-as-bias argument.
2. ✅ `K>τ` stability condition tied to EXP-38's task-dependent staleness budget.
3. ✅ literature review ≥12 cited, **≥6 from 2025+**, working links + relevance notes.
4. ✅ ≥3 candidate algorithms, each checked vs σ(M) ceiling + diagonal trap + async;
   ONE recommended primary + a GPU-free offline kill-test.
5. ✅ provenance table (finding → source artifact).
6. Every empirical number cited or flagged "unverified"; no invented citations.
7. Sections: Exec summary → Problem & system → Mathematical analysis → Literature
   review → Candidate algorithms (3 kill-checks each) → Recommended approach + kill-test
   + parity/surpass call + async resolution → Provenance table → Limitations.

## 8. Prior-art map (what's settled — do NOT relitigate)

- Substrate (anchor-on-PowerSGD r=77, paired replay, anchor-owns-Q) is settled
  (EXP-29/30). Judge ideas on theory, not on re-deriving this.
- Frontier sweeps #31/#33 (perturbation, δ-momentum, adaptive-λ, control-variate,
  sub-basis, β averaging) are **all null for surpass**. Don't re-propose them as surpass.
- B2 `delayed_ef` (λ=1, β_anc=0) = parity SOTA at 5/5 (0.7528). `signed_ema`
  (α=0.25, β_anc=0.50) = active research merger. Both fail at 20/20.
- EXP-38 next-method recommendation on file: compress in **activation** space, split
  fwd/bwd codec budgets, demote anchor to a slow **Q/codec calibrator**, set K & backward
  rank **per task**, and for surpass use cross-rank 2nd-moment or curvature — never a
  stale-gradient-reuse term.

**Key source artifacts** (full list → report provenance table):
`reports/comm-eff-grpo/why-grpo-fails-sft-works.html`,
`reports/anchor-future-projection/{theory-and-literature,discussion}-2026-06-22.md`,
`runs/EXP-38/verdict.md`, `runs/DELAY_ANALYSIS/{staleness_theorist,cadence_analyst}.md`,
`runs/SUMMARY.md`, `runs/Q_BASIS_ANALYSIS/Q_BASIS_REPORT.md`, `.claude/GOAL.md`,
`CODE_WALKTHROUGH.md`, `codex-findings/paper-bibliography.md`.

## 9. Execution order

1. [Phase 0] Read prior art; write this plan. ✅
2. [Phase 1] Spawn 4 teammates in parallel → v1 `notes.md` each.
3. [Phase 2] Cross-challenge round → `05-cross-challenge/{debateA,debateB}.md`.
4. [Phase 3] Team-lead link-verification pass on headline 2025+ citations.
5. [Phase 4] Synthesize `report.html`; check all 7 acceptance gates.
