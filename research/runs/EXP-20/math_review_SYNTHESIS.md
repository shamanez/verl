# EXP-20 PowerSGD — Consolidated Math/Theory Validity Synthesis

**Lead synthesis** over the 5-lens `powersgd-mathcheck` agent-team review.
**Reviewed commit:** `f748dbc1` (origin/exp/20-powersgd-activation)
**Date:** 2026-06-04
**Panel (all Opus, independent, adversarial):** mathematical-checker (core math), autograd-checker, distributed-correctness, rl-grpo-checker, numerics-stability.
**Lead on-box verification:** the live `exp-20-sync` run was directly checked — `sync_basis=true` on all 4 workers; log line `cross-rank Q agreement: max_rel_dev=0.000e+00 sync_basis=True (hard-invariant #4)` printed by all 4 ranks; q_cond≈1.0, logical_pp_bytes=102, mask counters 0 (powersgd-only), GPUs 92–100%.

## Combined verdict
**The PowerSGD implementation at `f748dbc1` is a faithful, mathematically-correct realization of issue #20.** All five independent lenses concur the code is CORRECT AS WRITTEN. **No INVALID claims.** The remaining items are: one latent bug that does NOT affect the sanctioned `sync_basis=true` sweep, two reporting/metric-honesty caveats, four analyst verdict-gate refinements, and one decisive EMPIRICAL question (r=102 sufficiency) that the sweep is designed to answer.

## Per-lens results
| Lens | Member | Verdict | Headline |
|---|---|---|---|
| Core math | mathematical-checker | VALID (10/10) | projector/no-STE, power-iter→Eckart–Young, seed, frozen-Q, r=H, fp32-QR, byte-budget (caveat), clean-cadence, consensus — all VALID |
| Autograd | autograd-checker | VALID (4/4) | claim-1 confirmed at torch-op level (0.0 diff); grad-ckpt dedup is `_fwd_generation`, NOT `grad_enabled` (comment fix); `use_reentrant=False` load-bearing |
| Distributed | distributed-correctness | CORRECT-as-written; 2 HIGH | consensus math confirmed; HIGH-1 on-disk-unverified (RESOLVED by lead); HIGH-2 verifier mis-gate (latent) |
| RL/GRPO | rl-grpo-checker | VALID + 2 caveats | frozen-Q ρ≈1; train-vs-rollout gap (consistent, fair vs PRF); ρ≈1 necessary-not-sufficient |
| Numerics | numerics-stability | VALID, no stability bug | RMSNorm absorbs `‖M_hat‖≤‖M‖` shrink → no rescale knob needed; grad_norm 268→70 benign warm-start; q_cond over-promises |

## Cross-lens conflicts — both resolved
1. **`sync_basis` dataclass default** (math-checker said True, rl-checker said False): RESOLVED → committed dataclass default IS `True` (`config/comm_eff.py:355`, CI-pinned by `test_sync_basis_defaults_true`). The "false" sightings are red herrings — `state.py:345` getattr-fallback (never reached) + the stale FIRST probe's CLI override. So safe-by-default AND launcher-forced. rl-checker's HIGH "footgun" → downgraded to LOW (stale issue/plan prose only).
2. **HIGH-1 consensus-unverified-on-disk** (distributed-checker): RESOLVED by lead on-box check (above). The consensus path ran multi-GPU; invariant #4 substantiated (`max_rel_dev=0.000e+00` ×4 ranks). distributed-checker had read the stale earlier-run artifact.

## Actionable findings (ranked)
- **HIGH-1 — RESOLVED (no action).** sync_basis=true consensus verified on-disk on the live run.
- **HIGH-2 — latent verifier mis-gate (does NOT block the sweep; one-line follow-up).** `verify_basis_agreement_across_ranks` asserts cross-rank Q identity with no `sync_basis` guard; under `sync_basis=false` (supported diagnostic mode) it would RAISE on legitimate per-rank divergence → hard crash. The sanctioned sweep uses `sync_basis=true` (verifier correctly asserts identity, which holds), so the sweep is unaffected. FIX (recommended, not blocking): gate the verifier on `self.sync_basis`.
- **MEDIUM — q_cond over-promises (numerics CAVEAT-1).** q_cond is measured on the orthonormal QR *output* → ≈1 always; it only catches a NON-FINITE Q (true collapse), never a poorly-fit basis. ANALYST: use `reconstruction_rel_error` as the basis-health metric, not q_cond. (The plan's "q_cond finite ⇒ no collapse" box over-promises.)
- **MEDIUM — byte-accounting honesty (math-checker claim 7).** Under `sync_basis=true` the `H·r` consensus all-reduce isn't counted/logged; the issue-spec'd `logical_pp_bytes_powersgd_with_basis_sync` metric + `account_basis_sync_bytes` knob are missing. ANALYST: report the head-to-head as forward-payload-matched (Y vs masked-h) and footnote the PowerSGD-only `H·r/cadence` consensus traffic (a DP-axis maintenance cost, not PP-boundary traffic) — or add the metric.
- **EMPIRICAL (decisive, NOT a code defect) — r=102 sufficiency (math-checker + numerics, converged).** reconstruction_rel_error 0.72–0.97 with DEPTH STRUCTURE: layer_3→0.025 (strong gap, 1-iter convergence) but deep layers 21/24 stay 0.89–0.92 (weak/absent gap) → r=102 is below the deep-layer effective rank. This BOUNDS the train-vs-rollout representation gap (rl INF-19) = the live risk. If it persists in the 50-step sweep → REVISE to r=205 (plan-sanctioned), NOT a code fix.
- **LOW — code refinements (follow-up, non-blocking):** (a) autograd code-comment attributes the grad-ckpt dedup to `grad_enabled` when it's actually `_fwd_generation`; `use_reentrant=False` is load-bearing; `powersgd_applications` cosmetically double-counts the recompute fire. (b) distributed DP-group bind wrapped in a broad `except`→silent WORLD fallback (harmless at SP=1, wrong under future SP>1/TP/PP); reconcile the plan/constructor `sync_basis=false` text with the operative default `true`.

## Verdict-gate refinements for the analyst
1. Treat ρ≈1 (frozen-Q probe gate) and reward/cosine/reconstruction (representation-gap health) as INDEPENDENT gates — a clean ρ≈1 with degrading reward is the expected INF-19/20 signature, not a contradiction.
2. Basis health = `reconstruction_rel_error`, NOT `q_cond` (≈1 always).
3. Judge the head-to-head by reward trajectory + dense-vs-compressed update cosine + reconstruction error.
4. grad_norm: expect a one-time warm-start drop (random Q → aligned), flat thereafter; flag only if it CLIMBS over the 50 steps.

## Bottom line
Mathematically and structurally the codec is a correct, faithful implementation of issue #20 — including the operator-mandated cross-DP consensus basis, now verified bit-identical on-disk. **Nothing blocks the 50-step sweep.** The one genuinely-open question is empirical (does r=102 capture enough of the activation subspace?), which is exactly what the sweep measures; the panel's convergent prediction is that the deep-layer weak spectral gap may force a REVISE to r=205.

## Source reports
- `math_validity_review.md` (core math) · `review_autograd.md` (autograd) · `review_distributed_correctness.md` (distributed; also `review_distributed.md` = math-checker interim) · `review_rl.md` (RL/GRPO) · `review_numerics.md` (numerics)
