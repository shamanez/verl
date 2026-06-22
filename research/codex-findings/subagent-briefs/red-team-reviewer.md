# Red-Team Reviewer Brief

Status: pre-final. `/Users/shamane/Documents/verl/research/codex-findings/index.html` was not present when inspected; only `goal-prompt.md` and `subagent-briefs/algorithm-designer.md` existed. Treat this as a checklist the main report must satisfy before finalizing.

## Verdict

The thesis is directionally defensible if stated conservatively: raw large/variable-lag anchor gradients are unsafe optimizer signals in this GRPO setting; the anchor is better justified as a slow codec/Q calibrator; stale-gradient reuse needs offline cosine-lift gates before any GPU run.

It is not defensible as a clean causal proof that stale gradients alone caused the failures, nor as a universal rejection of every stale-gradient correction. The local evidence is mostly n=1, diagnostic, and has coupled knobs (`delay_K`, cadence, merger geometry, task, early-vs-late capture schedule).

## Claims To Downgrade Or Qualify

1. **Do not say "EXP-38 proves gap 1 alone is fatal."**
   - `exp38_drift_analysis.py` computes `cos(g_t, g_{t-k})` from dense gradients at different ticks, not two gradients recomputed on the same rollout batch (`scripts/exp38_drift_analysis.py:183-207`). This is temporal gradient drift along training, with rollout/batch stochasticity still present.
   - The generated report text currently labels this as "parameter-point gap (gap 1)" and says "gap 1 alone already de-correlates" (`scripts/exp38_report.py:953-965`). Final report should rephrase: "dense temporal gradient drift is already severe; a same-batch parameter-only decomposition was not measured."

2. **Do not make the 5/5 vs 20/20 boundary sound causally isolated.**
   - EXP-37 raised cadence and `delay_K` together; local theorist explicitly flags the confound (`runs/DELAY_ANALYSIS/staleness_theorist.md:38-41`).
   - Cadence analyst says cadence is partially true, estimates 20-35% of degradation, and calls for a 2x2 decoupling (`runs/DELAY_ANALYSIS/cadence_analyst.md:10-18`, `:247-257`).
   - Safe wording: "the 20/20 latency bundle failed; evidence is consistent with stale/off-policy anchor gradients being the main mechanism."

3. **Avoid universal language like "stale gradients are dead/unusable."**
   - GSM8K 5/5 remains stable/near-dense in the run summary (`runs/SUMMARY.md:50-64`), despite EXP-38 showing low k5 cosine.
   - Stronger supported claim: "raw stale full gradients at large K, especially k20 and Big-Math k>=1, have no measured directional budget; any use must be dose-gated or killed by offline cosine tests."

4. **Keep n=1 and noise visible.**
   - Current baseline comparisons are explicitly n=1 and noisy by about +/-0.024 per draw (`runs/SUMMARY.md:21-29`).
   - EXP-38 is n=1 per task, 75 global steps (`runs/EXP-38/verdict.md:5-7`, `:72-73`).
   - EXP-37E is one 100-step draw; "drag" is a plausible diagnosis, not a replicated law (`reports/comm-eff-grpo/why-grpo-fails-sft-works.html:269`).

5. **Small-lag capture is early-phase only.**
   - EXP-38 k<=5 gradient pairs are almost entirely early training (`scripts/exp38_drift_analysis.py:366-371`).
   - Findings show GSM8K k5 uses only 2 pairs/matrix over global steps 1-5; k10+ spans steps 5-75 (`reports/dense-run-behaviour/exp38-dense-drift-gsm8k_findings.json`).
   - Final report must not infer late-training k5 behavior from early-only k5 captures.

6. **"Q is not the dominant failure" is supported, but not "Q is irrelevant."**
   - Q analysis lacks full Q tensors and cannot measure principal angles directly (`runs/Q_BASIS_ANALYSIS/Q_BASIS_REPORT.md:8-11`).
   - Scalar Q error drops after first update and stays around 0.03-0.04, but K=20 Q error rises late with the spiral (`runs/Q_BASIS_ANALYSIS/Q_BASIS_REPORT.md:50-61`, `:67-70`).
   - EXP-38 strongly supports forward `h` staleness tolerance and forward/backward asymmetry (`runs/EXP-38/verdict.md:34-37`), but the backward codec/rank recommendation still needs implementation and quality validation.

## Implementation/Admissibility Traps

1. **Paired replay/no-rerollout is safe to state.**
   - Anchor explicitly does not generate rollouts or recompute rewards (`verl/workers/comm_eff/anchor.py:21-39`; `verl/workers/engine/fsdp/transformer_impl.py:1270-1302`).
   - Replay stores generator snapshots and paired batches, then asserts exact delayed batch and never-fresher-than-K snapshot post-warmup (`verl/workers/engine/fsdp/transformer_impl.py:1422-1452`, `:1480-1518`).

2. **Projection error-feedback assumes a current-gradient target that normal training does not provide.**
   - The future-projection discussion says anchor refresh gives "ground truth" for `r = g_true - g_hat` (`reports/anchor-future-projection/discussion-2026-06-22.md:98-121`).
   - But `capture_fresh_anchor` is measurement-only, dump-only, disabled by default, and forbidden as a training config (`verl/workers/config/comm_eff.py:498-524`; `verl/workers/engine/fsdp/transformer_impl.py:1974-1983`).
   - Final report must say what "ground truth" means operationally. Is it an expensive fresh probe, a later stale anchor gradient, or a held-out offline EXP-38 label? These are not equivalent.

3. **Learned projection may violate async realism unless tightly constrained.**
   - Project goal says the real anchor is always lagging, must tolerate variable staleness, remain cross-rank-identical, and implies no delay-compensation/anchor-lead (`research/.claude/GOAL.md:58-62`).
   - Local theory flags extrapolation as a GOAL.md tension requiring operator decision (`reports/anchor-future-projection/theory-and-literature-2026-06-22.md:164-184`).
   - Any final recommendation involving projection must be conditional: offline-only until cross-rank identical sufficient statistics, variable-age conditioning, and no private per-rank fitting are specified.

4. **Delayed-EF pairing is implemented, but held-residual semantics matter.**
   - FastGradRing is exact-tick/no-fallback for `G_comp(t-K)` (`verl/workers/comm_eff/state.py:159-221`).
   - `delayed_ef` refreshes delta when the exact ring grad exists and holds it between fires (`verl/workers/comm_eff/spectral_filter.py:848-887`, `:930-944`).
   - Final report should ask whether "same batch, same weights" remains exact across multiple PPO mini-batch ticks and global-step snapshot timing; source asserts weights are never fresher than K, but realized delay can be K or K+1 on the 2-tick substrate (`transformer_impl.py:1510-1518`).

5. **Trust-region/IS correction is not ready from aggregate evidence.**
   - Algorithm-designer correctly gates this on per-sample logprobs/ESS (`codex-findings/subagent-briefs/algorithm-designer.md:87-105`).
   - Final report should not recommend IS/TRPO-style correction unless it names required per-token/per-sample data and the kill condition for high variance/low ESS.

## What Local Evidence Can Support

- **Reject raw large-K stale anchor gradients as primary optimizer signals:** supported for this setting, especially k20 and Big-Math k>=1, with caveats (`runs/EXP-38/verdict.md:25-28`, `:49-59`, `:72-73`).
- **Down-weight stale gradients by age/staleness:** plausible safety policy, not yet validated. It should be framed as a conservative no-harm gate, not a performance method.
- **Correct/project stale gradients:** unsupported until offline EXP-38 held-out cosine lift is shown, with diagonal-vs-off-diagonal attribution and task-separated results (`reports/anchor-future-projection/theory-and-literature-2026-06-22.md:188-220`; `codex-findings/subagent-briefs/algorithm-designer.md:106-160`).
- **Demote anchor to Q/codec calibrator:** strongly plausible for forward link; still needs an implementation plan and parity test because current SOTA uses anchor-derived delayed-EF in the optimizer (`research/.claude/GOAL.md:49-53`; `runs/EXP-38/verdict.md:34-37`).

## Questions The Main Agent Must Answer Before Finalizing

1. Does the final report clearly distinguish **temporal gradient drift** from a same-batch **parameter-point-only** gap?
2. Does it avoid saying `delay_K` alone caused EXP-37 without acknowledging the cadence/delay coupling and the requested 2x2 decoupling?
3. Are all n=1/noise/capture-schedule caveats visible near the headline conclusions, not buried?
4. Does it separate confirmed evidence, theory-supported inference, and speculation for SFT transfer, length-ratchet causality, sigma(M) ceiling, and projection/extrapolation?
5. If recommending "reject stale gradients," is the scope limited to **raw large/variable-lag optimizer signals**, while preserving low-lag/dose-gated diagnostic possibilities?
6. If recommending learned projection or EF-on-projection, what exact current-gradient label exists without violating the async constraint or enabling a forbidden delay-zero training path?
7. If recommending forward/backward codec split, does the implementation currently support asymmetric ranks/Q bases, or is that a new code path with its own validation?
8. If citing literature, are PPO/TRPO/off-policy IS, async SGD/DC-ASGD/PipeDream/PipeMare/Nesterov-async, PowerSGD, and error-feedback claims backed by primary links rather than project memory alone?
9. Are GSM8K and Big-Math kept separate everywhere, with no averaged staleness budget?
10. Are kill-gates written as falsifiable thresholds, especially: held-out stale-to-live cosine lift, diagonal-trap attribution, ESS/KL gates for IS, and no material dose when sign agreement is chance?

## Source Paths Inspected

- `/Users/shamane/Documents/verl/research/runs/SUMMARY.md`
- `/Users/shamane/Documents/verl/research/runs/EXP-38/verdict.md`
- `/Users/shamane/Documents/verl/research/runs/Q_BASIS_ANALYSIS/Q_BASIS_REPORT.md`
- `/Users/shamane/Documents/verl/research/runs/DELAY_ANALYSIS/staleness_theorist.md`
- `/Users/shamane/Documents/verl/research/runs/DELAY_ANALYSIS/cadence_analyst.md`
- `/Users/shamane/Documents/verl/research/reports/anchor-future-projection/discussion-2026-06-22.md`
- `/Users/shamane/Documents/verl/research/reports/anchor-future-projection/theory-and-literature-2026-06-22.md`
- `/Users/shamane/Documents/verl/research/reports/comm-eff-grpo/why-grpo-fails-sft-works.html`
- `/Users/shamane/Documents/verl/research/reports/dense-run-behaviour/*_findings.json`
- `/Users/shamane/Documents/verl/research/scripts/exp38_drift_analysis.py`
- `/Users/shamane/Documents/verl/research/scripts/exp38_report.py`
- `/Users/shamane/Documents/verl/research/.claude/GOAL.md`
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/anchor.py`
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py`
- `/Users/shamane/Documents/verl/verl/workers/comm_eff/spectral_filter.py`
- `/Users/shamane/Documents/verl/verl/workers/config/comm_eff.py`
- `/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py`
- `/Users/shamane/Documents/verl/research/codex-findings/subagent-briefs/algorithm-designer.md`
