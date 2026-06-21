# EXP-38 — Verdict: **PASS** (diagnostic)

- **id:** EXP-38 — Uncompressed GRPO temporal-drift probe (gradient / boundary-activation / GRPO-state drift)
- **date:** 2026-06-20
- **kind:** experiment + investigation (diagnostic). PASS = a usable, interpretable drift dataset + strong standalone HTML reports answering all deliverable questions, with located staleness knees and a next-method recommendation — **not** a performance threshold. Which way H1/H2/H3 resolved does not change PASS.
- **arms:** ARM A = **GSM8K** (easy, base `\boxed` ≈0.72) · ARM B = **Big-Math** (hard, base ≈0.48). Both **uncompressed** (comm_eff OFF), Qwen2.5-1.5B-Instruct, accel surface, 75 global steps = 150 optimizer ticks, n=1 each. **The two datasets' tensors/curves are never merged.**
- **scope of this analysis:** laptop-only, free, no Vast/GPU. Both capture phases were already complete + verified; both boxes are destroyed.

## Deliverables (all exist, self-contained, base64 plots, 0 external refs, 0 broken embeds)

| file | size | plots |
|---|---|---|
| `research/reports/dense-run-behaviour/exp38-dense-drift-gsm8k.html` (+ `_findings.json`) | 1.04 MB | 16 |
| `research/reports/dense-run-behaviour/exp38-dense-drift-big-math.html` (+ `_findings.json`) | 1.03 MB | 16 |
| `research/reports/dense-run-behaviour/exp38-dense-drift-joint.html` (GSM8K↔Big-Math comparative) | 0.36 MB | 6 |

Engine: `research/scripts/exp38_drift_analysis.py` + `exp38_report.py` (per-arm); **NEW** `research/scripts/exp38_compare.py` (the joint comparative renderer, built for this task) + `reports/dense-run-behaviour/_joint_narrative.html` (theorist fragment, embedded into the joint report).

## Independent verification (MANDATORY — satisfied)

Every headline number was reproduced **from the raw `.pt` tensors by a second, independent teammate per arm**, using from-scratch metric code that never reads the analysis engine. The engine findings match the independent ground-truth **exactly** on all 14 headline metrics × 2 arms (cos@k1/k5/k20, sign@k5/k20, cos-ratio, weight-drift@k5/k20, grad rank-90, boundary-h rank/top1/stable-rank, top-77 overlap@k20, grad_h rank-90). The verifiers also independently flagged the GSM8K epoch-2 "rise" as a warmup-binning artifact (see below), which was then corrected in the engine. Verifier scripts: `scratchpad/verify_exp38.py` (GSM8K), `scratchpad/bigmath_verify.py` (Big-Math).

## Headline findings (verified numbers; lag k in optimizer ticks, 2/global-step; k≈5 = stable 5/5 anchor, k≈20 = broken 20/20 anchor)

### H1 — gradient-anchor staleness budget — **SUPPORTED on both, but task-dependent**
- **GSM8K (easy): SUPPORTED.** cos(g_t,g_{t−k}) = **0.507 (k1) → 0.464 (k2) → 0.176 (k5) → 0.023 (k10) → −0.008 (k20) → 0.005 (k40)**; sign-agreement 0.543 (k5) → 0.498 (k20, = chance). The direction is nearly memoryless past ~5 ticks; even the *stable* 5/5 anchor sees only cos≈0.18. **Knee: between k≈5 and k≈10** (cos hits chance by k≈10), i.e. *tighter* than H1 hypothesized (k5=0.18 is already below the 0.35 "high" threshold).
- **Big-Math (hard): SUPPORTED (budget ≈ 0).** cos = **0.018 (k1) → 0.011 (k5) → 0.004 (k20)**; sign-agreement ≈ 0.50 (chance) at every lag. The gradient is decorrelated **even at lag 1**. **Knee: at or below k=1** — a stale uncompressed gradient is unusable at *any* latency on the hard task.
- *(k1/k2/k5 are sampled only in early training — a clean within-(early)-phase lag decay; k10/20/40 span mid/late. n=1.)*

### H2 — drift is GRPO-coupled (distribution gap), not a pure parameter-point gap — **SUPPORTED on both**
- Method: normalize each signal's lag-k drift `D(k)=median|x_t−x_{t−k}|` to its max-lag value; compare its half-drift lag to the weight half-drift lag (≈**7.9 global steps** both arms). A behaviour signal that reaches half its drift sooner than the smooth/cumulative weight drift drifts faster.
- **GSM8K: 7/10** · **Big-Math: 9/10** rollout/logprob/response signals drift comparably-or-faster than the weights. Fastest in both: `pg_clipfrac`, `ppo_kl`, `response_length/mean` — all reach half their drift within **~1 global step**. ⇒ the dangerous term is the **distribution gap** (gap 2), not curvature×‖Δθ‖.

### H3 — activation-codec staleness budget — **SUPPORTED (staleness-insensitive) on both; codec primitive is task-invariant**
- **Forward `h` is rank ≈1 on both** (top-1 singular direction holds **99.1% / 98.6%** of energy; stable rank ≈1.01; rank-for-90% = 1 ≪ r=77 ≪ H=1536). Rank-r activation compression is the right primitive and **r=77 is hugely over-provisioned** for the forward link — **task-independently**.
- **Subspace overlap is flat across lag** (not decaying): top-77 o(t,t−k) ≈ 0.77 (GSM8K) / 0.71 (Big-Math) constant for k=1…40; **top-1 overlap = 1.0 at every lag** (the energetic direction is perfectly stable); smallest principal angle 0°, largest ~88°. ⇒ **codec staleness is NOT the limiter** — Q (especially a rank-1/few Q) can be frozen far beyond the current cadence. Periodicity: **smooth/aperiodic** (no dominant period; FFT power-frac 0.35–0.44).
- **Forward/backward asymmetry (codec-decisive):** backward `grad_h` rank-for-90% = **105 (GSM8K) / 180 (Big-Math)** — *above* r=77 and *growing with task hardness*. The backward link is **not** as compressible as the forward; symmetric forward/backward codec budgets would starve the backward path, worst on the hard task.

### Nature of learning (gradient rank over training)
- Uncompressed-gradient rank-for-90% median = **50 (GSM8K) / 78 (Big-Math, ≈ r=77)** — the hard task's gradient is higher-rank. Stable rank ≈3.1 both; participation ratio 6.9 / 8.8.
- **No clean epoch-2 jump on GSM8K.** Per-step rank ramps from ~16 (warmup, steps 1–5) to a stationary ~62–67 by step 10; post-warmup pre/post-step-58 = **65/61 (flat)**. The naive ≤58-vs->58 split (48→61) is a warmup-binning artifact (early low-rank warmup steps in the pre bin), **not** an epoch effect. Big-Math crosses no epoch boundary in 75 steps.

### Joint / task-dependence (the headline science)
**Is the next method's staleness/codec budget task-dependent? YES — except the forward-activation primitive.**
- Gradient staleness budget: short window on GSM8K (cos 0.51→0.18 over k1→k5) vs **~zero** on Big-Math (cos≈0 at k1). **Task-dependent.**
- Gradient rank 50→78; backward `grad_h` rank 105→180 (easy→hard). **Task-dependent.**
- Forward `h` rank ≈1 (massive activation) on both. **Invariant** — the only quantity safe to budget globally.

## Located unsafe-staleness knees (vs the EXP-37 5/5-stable / 20/20-broken boundary)
- **Gradient-anchor knee:** GSM8K between k≈5 and k≈10 (cos→chance by k≈10); Big-Math at/below k=1. Consistent with: 5/5 survived because the small frequent dose rode a still-weakly-aligned (cos≈0.18) gradient; 20/20 broke because it injected a near-orthogonal/biased (cos≈0) force. On the hard task even 5/5 is marginal.
- **Activation-codec (Q) knee:** none within k≤40 — overlap is flat, so the limiter is **not** Q staleness (Q is stale-tolerant); the limiter is forward/backward **rank asymmetry**, not lag.

## Next-method recommendation (from the numbers)
1. **Compress in activation space, not gradient space.** A stale uncompressed gradient is dead as an optimizer signal (H1; doubly dead on the hard task). Do **not** reweight/accumulate/error-feedback a stale gradient (σ(M) ceiling: capped at parity at best, harmful when biased).
2. **Forward link: low-rank codec (rank-1/few) with a frozen / slowly-refreshed Q.** h is rank-1, top-1 subspace overlap = 1.0 across lag ⇒ Q is intrinsically staleness-tolerant; r=77 is wasteful.
3. **Backward link: a SEPARATE, higher-rank codec** — ≥105 (GSM8K-class) / ≥180 (Big-Math-class). Never set symmetric forward/backward budgets.
4. **Demote the anchor to a slow Q/codec calibrator, not a gradient provider** (the slowly-varying, cross-rank-identical role the flat overlap supports) — answers issue Q5: **yes**.
5. **Set staleness K and backward rank PER TASK** (the easy↔hard divergence is too large for a constant); only the forward-codec rank is safe to fix globally.
6. **If surpassing the uncompressed baseline is sought,** layer a cross-rank 2nd-moment (disagreement-as-objective) or curvature term — information outside σ(M) — never a stale-gradient-reuse term.

## Success-criteria checklist (plan §Success criteria, machine-checkable)
- [x] off-parity gate passed (both arms, capture phase — see CHECKPOINT_STATUS / plan ARM-B result)
- [x] 75-step uncompressed-run capture clean, no NaN, all 5 roles, 1071 dumps/arm, bounded
- [x] artifacts local + boxes torn down (capture phase)
- [x] **strong standalone HTML report exists** — 3 reports, self-contained (0 external refs), base64 plots, 0 broken embeds
- [x] **answers all deliverable questions** with numbers (weight drift, grad cos/sign/norm-ratio, grad effective/stable rank + participation + full SVD spectrum evolution, boundary h low-rank + multi-r subspace overlap + principal angles + periodicity, boundary grad_h rank, GRPO-signal + advantage-dispersion correlations)
- [x] **H1/H2/H3 resolved either way as numbers** and explicitly marked (SUPPORTED ×3 both arms; H1 Big-Math = "budget ≈ 0")
- [x] **knees located** vs k≈5 / k≈20 (+ GSM8K epoch-2 corrected to "no jump")
- [x] **next-method recommendation** present (per-arm §9 + joint §6 theory)
- [x] **headline numbers independently reproduced** from raw tensors by a second teammate (certified exact match, both arms)

## Caveats
n=1 per task, 75 steps. Small-lag (k≤5) gradient cosines are sampled only in early training (capture-schedule confound — disclosed in every report's §1 per-lag-count table). Knees framed as "consistent / inconsistent with" the EXP-37 5/5-stable, 20/20-broken boundary, not proof. This uncompressed normal run measures the parameter-point gap (gap 1) directly; gap 2's contribution is inferred from H2's behaviour-drift signals, not observed under a live stale anchor.

## Close-out / optional follow-ups (NOT required by this analysis; need operator authorization)
- log-writer → `LOG.md` / `runs/SUMMARY.md` / `STATUS.md`
- draft PR `exp/38-dense-drift-probe` → `vast-ai-workload` (gated instrumentation, all default-OFF). No canonical launcher promoted (throwaway diagnostic probe).
