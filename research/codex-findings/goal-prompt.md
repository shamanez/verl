/goal Produce the strongest evidence-backed HTML research report possible at `/Users/shamane/Documents/verl/research/codex-findings/index.html`, using only artifacts written under `/Users/shamane/Documents/verl/research/codex-findings/` for new outputs. The report must answer how to stably use, correct, down-weight, or reject stale anchor gradients in comm-efficient GRPO, grounded in the local verl experiments and current literature. Continue until the HTML exists, is self-contained, has no TODO placeholders, cites local evidence and web papers, passes a final self-review, and gives a concrete next algorithm plus kill-gates. If blocked, stop only with a precise blocker, attempted paths, and the missing input needed.

This is a long research task. Use Codex Goals and subagents intentionally:

- Goals documentation: https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex
- Subagents documentation: https://developers.openai.com/codex/subagents

Before doing the research, read those two docs enough to respect the workflow: this is an evidence-checked `/goal` task with a final artifact, and subagents should be spawned explicitly for parallel, context-saving work. Do not overload the main context with full logs or huge HTML blobs. Use subagents to read, search, and summarize into compact briefs. The main agent should synthesize, verify, and write the final HTML.

All new files must go under:

`/Users/shamane/Documents/verl/research/codex-findings/`

Required output files:

- `index.html` - final polished report for expert RL/optimization researchers.
- `subagent-briefs/` - concise markdown briefs from each subagent.
- `evidence-ledger.md` - claim-to-source ledger mapping every important claim to local files or web citations.
- `paper-bibliography.md` - web-searched paper list with links and one-line relevance notes.

Do not open a PR. Do not change training code unless a tiny local analysis helper is absolutely necessary, and if so put it under `codex-findings/`.

## Required local reading

Read these before forming conclusions:

- `/Users/shamane/Documents/verl/CLAUDE.md`
- `/Users/shamane/Documents/verl/CODE_WALKTHROUGH.md`
- `/Users/shamane/Documents/verl/research/runs/EXP-37E`
- `/Users/shamane/Documents/verl/research/runs/EXP-38`
- `/Users/shamane/Documents/verl/research/runs/Q_BASIS_ANALYSIS`
- `/Users/shamane/Documents/verl/research/runs/DELAY_ANALYSIS`
- every report under `/Users/shamane/Documents/verl/research/reports`

Use targeted extraction for large HTML/log files. Do not paste huge raw outputs into the main context. Keep GSM8K and Big-Math separate; never merge their tensors, curves, or statistics.

## Background story that must be preserved

We are training GRPO with a fast compressed network and a slow anchor network. The fast path uses a PowerSGD-style activation codec. The anchor periodically loads an old snapshot of the fast network and computes a clean full gradient on the paired stale rollout batch. Paired replay is implementation-correct: the anchor should not regenerate rollouts, because the stored rollout is the batch produced by the stale policy snapshot. Re-rolling would add a new Monte Carlo sample and would not guarantee the same trajectory. However, this only makes the anchor gradient a clean estimate of the old policy's gradient; it does not make it a current on-policy gradient.

The current working thesis is:

- The stale anchor gradient is the main problem when used as an optimizer signal in on-policy GRPO.
- The Q/activation codec is comparatively stale-tolerant and should probably be the anchor's natural role.
- In GRPO, stale anchor gradients contain two gaps: a parameter-point gap, `theta_t` versus `theta_{t-K}`, and a rollout-distribution gap, `D(pi_t)` versus `D(pi_{t-K})`.
- In SFT, only the parameter-point gap exists because the data distribution is fixed. That is why SFT-style stale-gradient telescoping/error-feedback intuition does not transfer cleanly.
- Paired replay makes `M = g(theta_{t-K}; D(pi_{t-K}))` internally valid, but it is still the gradient of a defunct policy when applied to `theta_t`.
- Any deterministic reuse of stale `M` risks the sigma(M) ceiling: reweighting, accumulating, or error-feedback over stale gradients can approach parity at best and can become harmful when the stale term is biased.

Key local evidence to incorporate:

- EXP-37/DELAY_ANALYSIS: signed_ema at K=5 was near stable; signed_ema at K=20 collapsed. K=20 val@50 was about 0.648, val@100 about 0.444, below the no-merger floor around 0.630. The mechanism was a stale coherent correction driving a response-length ratchet.
- EXP-37E: delayed_ef at 20/20 did not ignite the same length spiral but degraded into drag. Val@25/50/75/100 was about 0.649 / 0.676 / 0.581 / 0.608. Response length stayed roughly bounded, Q stayed conditioned near 1.0, reconstruction error stayed around 0.03-0.04, but grad_norm rose sharply.
- Q_BASIS_ANALYSIS: before first anchor Q update, Q reconstruction error is about 0.975; after the first update, it drops to about 0.04 in both K=5 and K=20. Fast PowerSGD basis updates stay 0 when anchor owns Q. Final Q condition is near 1.0. Q is not the main collapse.
- EXP-38: dense temporal drift shows gradient staleness is task-dependent and often tiny. GSM8K cosine: k1 0.507, k5 0.176, k10 0.023, k20 -0.008. Big-Math cosine: k1 about 0.018, k5 about 0.011, k20 about 0.004. Big-Math has almost zero stale-gradient budget.
- EXP-38 H2: rollout/logprob/response signals drift comparably or faster than weights, so the GRPO distribution gap matters.
- EXP-38 H3: forward boundary activation `h` is effectively rank-1 on both GSM8K and Big-Math, with lag-flat subspace overlap. Q can be slow or frozen on the forward link. Backward `grad_h` rank is much higher: about 105 on GSM8K and 180 on Big-Math, so forward and backward codec ranks must be split.

Also audit the implementation details:

- Anchor should not generate rollouts again in paired replay.
- The stale snapshot and stale rollout should match the same old generator policy as strictly as possible.
- Check stale snapshot queue, replay ring, canary/assertions, anchor counters, Q ownership, DP reduction, and delayed_ef ring matching.
- Specifically examine whether `delayed_ef` is truly pairing the same batch and same weights under multiple PPO mini-batch optimizer ticks, or whether later mini-batch ticks create a subtle mismatch between rollout-generation weights and `G_comp_ring(t-K)`.

## Mandatory subagent design

Spawn these subagents early. Each subagent must write a concise markdown brief under `codex-findings/subagent-briefs/`, and each brief must include exact local paths and line references when possible. Each brief should be compact, not a raw dump.

1. `local-evidence-archivist`
   - Role: Read the required local reports and run folders.
   - Output: `subagent-briefs/local-evidence-archivist.md`
   - Must produce: experiment timeline, key numbers, known verdicts, caveats, and "do not mix GSM8K/Big-Math" reminders.

2. `implementation-auditor`
   - Role: Inspect comm-efficient GRPO implementation files related to anchor snapshots, paired replay, anchor gradients, stale weights, stale rollouts, delayed_ef, signed_ema, Q ownership, and PowerSGD activation hooks.
   - Output: `subagent-briefs/implementation-auditor.md`
   - Must answer: is anchor no-rerollout correct, are stale weights matched to the right stale rollouts, and are there implementation glitches or mini-batch tick caveats?

3. `theory-mathematician`
   - Role: Formalize the stale-anchor GRPO problem.
   - Output: `subagent-briefs/theory-mathematician.md`
   - Must include: decomposition of stale error into parameter drift and distribution drift; why stale on-policy gradients become off-policy when transplanted; trust-region/importance-sampling implications; sigma(M) ceiling; why SFT differs.

4. `literature-scout`
   - Role: Perform fresh web search for papers.
   - Output: `subagent-briefs/literature-scout.md` and entries in `paper-bibliography.md`
   - Must search: delayed/stale gradients, async SGD, delay compensation, asynchronous pipeline training, off-policy policy gradient, PPO/GRPO stale trajectories, trust-region correction, importance sampling, gradient compression, PowerSGD, error feedback, multi-timescale learning, slow/fast networks, and post-2025 papers.
   - Must include: classic foundations and recent post-2025 work. Include the 2025 Nesterov async pipeline paper and compare it to DC-ASGD, PipeDream, PipeMare, stale-synchronous/async SGD, and error-feedback compression literature.

5. `algorithm-designer`
   - Role: Propose practical algorithms that could survive large and variable anchor delay.
   - Output: `subagent-briefs/algorithm-designer.md`
   - Must compare: demote anchor to Q/codec calibrator; age-decayed stale correction; adaptive lambda by measured staleness; trust-region or IS-corrected stale data; learned projection/extrapolation; cross-rank second moment; curvature/off-diagonal routes; forward/backward codec split.
   - Must give: pseudocode-level proposals and offline kill-gates using EXP-38 captures before any GPU run.

6. `red-team-reviewer`
   - Role: Attack the final thesis before the HTML is finalized.
   - Output: `subagent-briefs/red-team-reviewer.md`
   - Must check: overclaiming from n=1 runs, small-lag capture confounds, unsupported causal claims, missing citations, impossible implementation assumptions, and whether the final recommendation violates async constraints.

Subagent context policy:

- Subagents should read raw material and return distilled findings.
- The main agent should not import giant logs into its context if a subagent can summarize them.
- If subagents disagree, the main agent must resolve by checking the original source.
- The main agent owns the final `evidence-ledger.md` and final `index.html`.

## Literature search requirements

Use web search. Cite sources with links. Prefer primary sources: papers, arXiv, conference pages, official docs, and repository pages. Include at least:

- Asynchronous SGD with stale or delayed gradients.
- Delay-compensated ASGD / DC-ASGD.
- PipeDream, PipeMare, weight stashing, async pipeline optimization.
- Nesterov-style asynchronous pipeline optimization, especially Ajanthan/Ramasinghe/Zuo/Avraham/Long 2025.
- Off-policy policy gradient, PPO/TRPO trust-region theory, importance sampling variance/bias, stale actor-learner systems.
- Gradient compression with error feedback, including PowerSGD and error-feedback theory.
- Multi-timescale optimization and slow/fast networks.
- Any post-2025 papers that directly bear on delayed updates, stale gradients, async RL, or hybrid slow/fast optimization.

The bibliography must label each paper as one of:

- directly useful
- background
- tempting but not applicable
- contradicts or challenges the thesis

## Questions the final HTML must answer

1. Is it correct that the anchor does not recompute rollouts? Give the strict answer, then explain the statistical reason.
2. What exactly is the stale anchor gradient estimating?
3. Why does paired replay make the old gradient cleaner but not current?
4. Why does stale-gradient reuse break or drag in on-policy GRPO but work better in SFT-like fixed-data settings?
5. What does EXP-38 imply about the usable stale-gradient window on GSM8K and Big-Math?
6. Why is Q/activation codec staleness not the dominant failure?
7. What implementation glitches or caveats remain, especially around delayed_ef and mini-batch tick pairing?
8. Which methods are ruled out by evidence or theory?
9. Which methods remain credible, and which should be tested first?
10. What is the concrete next algorithm, with pseudocode and offline kill-gates?

## Required final recommendation shape

The final report must not merely say "stale gradients are bad." It must decide among these options:

- Reject stale anchor gradients as optimizer signals and demote anchor to Q/codec calibrator.
- Use stale anchor gradients only with strict age decay/adaptive dose and safety gates.
- Use learned projection/extrapolation only if offline cosine-lift tests pass and the method avoids the diagonal trap.
- Use trust-region or IS correction only if the variance/bias trade is defensible for GRPO rollouts.
- Use cross-rank second moment or curvature routes if the aim is surpassing dense, because those may inject information outside sigma(M).

The final report should likely recommend:

- Do not use raw stale anchor full gradients as the optimizer signal at large K.
- Prefer activation-space compression with a slow/frozen forward Q.
- Split forward and backward codec budgets; do not use symmetric rank for `h` and `grad_h`.
- Treat anchor primarily as a slow cross-rank-identical codec/Q calibrator.
- If any stale-gradient correction is tested, start with dose decay/adaptive lambda and an offline EXP-38 kill-gate.
- If projection/extrapolation is explored, first prove stale-to-live cosine lift offline and prove the lift is not just diagonal rescaling.

If the evidence changes during the research, update the recommendation, but the report must clearly separate confirmed evidence, theory-supported inference, and speculation.

## HTML quality bar

Write `index.html` as a polished, standalone expert report:

- Clear title and abstract.
- Executive verdict.
- Evidence table with exact local source paths.
- Mathematical section with readable equations.
- Implementation audit section.
- Literature review with linked citations.
- Algorithm proposals with pseudocode.
- Kill-gate plan.
- Red-team caveats.
- Final decision and next experiment.

The report should be readable by top RL, optimization, and systems researchers. It should be precise, restrained, and evidence-backed. Avoid hype. Avoid generic filler. Do not hide uncertainty.

Before marking the Goal complete:

- Verify `index.html` exists under `/Users/shamane/Documents/verl/research/codex-findings/`.
- Verify every major claim appears in `evidence-ledger.md`.
- Verify all subagent briefs exist.
- Verify bibliography exists and contains web links.
- Verify no TODO/TBD placeholders remain.
- Verify local paths in the report are correct.
- Verify the final recommendation is explicit and not hedged into uselessness.
