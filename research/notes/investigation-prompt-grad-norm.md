# Investigation prompt — paste into a new session to draft the GitHub issue

> Paste the block below (everything between the long horizontal rules) into
> a fresh Claude Code session. It is self-contained: the new session does
> not need access to any prior conversation. Its output will be a single
> `gh issue create` command to print to stdout for review (it will NOT post
> the issue automatically).

---

You are picking up a research investigation on `shamanez/verl`, branch
`vast-ai-workload`. The research repo (issue queue) is
`shamanez/verl-compression-research` and is set as the local gh-default;
the code repo (PR target) is `shamanez/verl` with base `vast-ai-workload`
(NEVER `main` — `main` tracks upstream).

Your only job is to draft a GitHub issue. Do NOT write a plan file, do
NOT modify code, do NOT provision compute. Output a single
`gh issue create` invocation (title + body in markdown) — do not actually
post it; print to stdout for human review.

### Operator constraints (load-bearing — read these first)

1. **The communication-efficient method's design is no-KL no-entropy.**
   `actor.use_kl_loss=False`, `algorithm.use_kl_in_reward=False`,
   `actor.entropy_coeff=0`. Every experiment that tests the COMM-EFF
   METHOD itself runs no-KL.

   The one exception is the gate test below (Test 1, Cell A), which
   reproduces the dense baseline WITH KL purely as a reference / sanity
   point — it is not part of the method evaluation.

2. **Every post-gate experiment (Tests 2, 3, 4) must have
   `comm_eff.enabled=true`.** The method is the thing under
   investigation. The only experiments that disable the method are the
   gate's Cell B (regression check) and Test 4's no-comm-eff comparison
   cell (diagnostic baseline for the FSDP audit).

3. **Run the tests sequentially.** Test 1 is the gate; if it fails,
   stop and fix the scaffolding before running anything else. Test 2
   only runs if Test 1 passes. Test 3 only runs if Test 2 is
   inconclusive (cells indistinguishable) or its signal is partial.
   Test 4 is a diagnostic re-run; do it last and only if Tests 2 and 3
   haven't already identified the dominant cause.

---

## Observation

A dry-run scale-up of the communication-efficient baseline to paper-scale
rollouts (`TRAIN_BATCH=128, ROLLOUT_N=8, MAX_PROMPT=1024,
MAX_RESPONSE=16384`) produced symptoms that the verified-PASS smoke
configuration (see `runs/communication-baseline/` and
`findings/communication-baseline.md`) did not. The dense baseline
(`runs/baseline/`, verl unmodified, WITH KL loss enabled) also did not
show these symptoms.

Per-step trajectory from the comm-eff dry-run training log:

| step | grad_norm | ppo_kl | entropy | pg_loss | pg_clipfrac |
|---:|---:|---:|---:|---:|---:|
| 1 | 1134 | 0.04 | 6.42 | 0.28 | 0.32 |
| 10 | 1283 | 0.07 | 6.48 | 0.26 | 0.31 |
| 20 | 1342 | 0.17 | 6.24 | 0.28 | 0.31 |
| 30 | 1477 | 0.19 | 5.39 | 0.25 | 0.28 |
| 40 | 1465 | 0.41 | 4.36 | 0.22 | 0.25 |
| 50 | 1712 | 0.77 | 1.85 | 0.25 | 0.31 |
| 56 | 1884 | 1.38 | 0.057 | 0.29 | 0.35 |
| 58 | 1662 | 1.04 | 0.023 | 0.24 | 0.29 |

Three signals:
- **Entropy collapse** 6.4 → 0.023 over 58 steps (policy near-deterministic).
- **`ppo_kl` explosion** 0.04 → 1.4 (PPO trust region assumes < 0.1).
- **Step-1 grad_norm = 1134** — high even BEFORE any policy drift could
  matter. The variance amplification is happening from the very first
  gradient.

`response_length/max` repeatedly hits the truncation cap (multiple steps),
consistent with policy collapse generating repetitive output until
truncation.

## Reference: dense baseline numbers (WITH KL loss)

Running `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
verbatim on `vast-ai-workload` (no comm-eff, no scaffolding edits, **KL
loss enabled at `kl_loss_coef=0.001`**) gives, at step 1:

| metric | step-1 value |
|---|---:|
| `actor/grad_norm` | **0.36** |
| `actor/entropy` | 0.37 |
| `critic/score/mean` | 0.12 |

The dense baseline goes on to improve markedly over 100 steps. The
comm-eff dry-run's step-1 grad_norm of **1134** is **~3000× the dense
reference** — that is the specific number this investigation has to
explain.

## Full comm-eff baseline configuration (the method under investigation)

The configuration that PASSED at smoke scale (recorded in
`runs/communication-baseline/`):

| component | knob | value | notes |
|---|---|---|---|
| master | `comm_eff.enabled` | `true` | the method |
| mask | `comm_eff.mask.enabled` | `true` | PRF Bernoulli at pipeline-boundary decoder blocks |
| mask | `comm_eff.mask.p` | `0.9` | fraction zeroed; `h_tilde = h * mask` (no 1/(1-p) rescale) |
| mask | `comm_eff.mask.mask_recompute` | `true` | mask fires on BOTH gradient-feeding forwards |
| anchor | `comm_eff.anchor.enabled` | `true` | hookless cloned-module backward |
| anchor | `comm_eff.anchor.cadence` | `5` | every 5 PPO substeps |
| anchor | `comm_eff.anchor.delay_K` | `5` | 5-substep stale weight snapshot |
| spectral | `comm_eff.spectral.enabled` | `true` | EMA → SVD → Tikhonov → α-blend |
| spectral | `comm_eff.spectral.alpha` | `0.5` | `G_proj = α·G_mask + (1−α)·G_filt` |
| spectral | `comm_eff.spectral.tau` | `0.01` | Tikhonov damping |
| spectral | `comm_eff.spectral.beta_anc` | `0.9` | EMA decay |
| spectral | `comm_eff.spectral.seed_anchor_cache` | `false` | live anchor populates M_anchor from zero |
| spectral | `comm_eff.spectral.ema_device` | `gpu` | M_anchor in HBM |
| spectral | `comm_eff.spectral.svd_mode` | `full` | full thin SVD |
| spectral | `comm_eff.spectral.basis_cache` | `cache` | reuse U/S/V across PPO mini-batches |
| spectral | `comm_eff.spectral.max_targets` | `4` | smoke cap |
| objective | `actor.use_kl_loss` | `False` | the method's design |
| objective | `algorithm.use_kl_in_reward` | `False` | the method's design |
| objective | `actor.entropy_coeff` | `0` | the method's design |
| FSDP | `actor.fsdp_config.use_orig_params` | `true` | spectral hook needs full 2D Tensor post-reduce |

That's the "full method enabled" configuration. The launcher
`examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
encodes this as its defaults.

## Candidate root causes — enumerate ALL of these in the issue body

### A. KL anchor removal (known driver of LATE-step collapse, NOT under test)
`actor.use_kl_loss=False` removes the soft anchor that holds π close to
π_ref. At step 1 this is irrelevant (`KL ≈ 0`); cumulatively it explains
the late-step entropy collapse + `ppo_kl` explosion. Documented as
background. Operator constraint: KL stays off in the method evaluation
(the gate's Cell A is the only with-KL run, and it's a reference-only
sanity reproduction).

### B. Importance-sampling variance under independent PRF masks (step-1 cause candidate)
`mask_recompute=true` masks BOTH the actor-train forward AND the
`compute_log_prob` (old_logprob) forward. But the PRF key (substep ×
layer × seed) **differs between the two paths by design** —
`compute_log_prob` fires once per train step, the PPO inner loop fires
`N×E` times. So `log_p_current` and `log_p_old` see different mask
realizations.

The PPO importance ratio
```
r = exp(log_p_current − log_p_old)
```
is then the ratio of two stochastic estimates under different masks,
even when the underlying actor weights are identical. **At step 1,
`r ≠ 1` even though `π_new == π_old`** — purely from the mask-realization
difference.

Expected variance of `log(r)` at step 1 scales like `2 · Var(masked
logit)` per token. Most likely explanation for the step-1 grad_norm
being ~3000× the dense reference.

**Diagnostic**: at step 1, log per-token `(log_p_current − log_p_old)`
histogram; expect non-zero spread even though `π_new == π_old`.

### C. Spectral filter on empty `M_anchor` at startup
`seed_anchor_cache=false` → `M_anchor = 0` for substeps 1–4 (first
anchor refresh fires at substep 5). The SVD basis of a zero matrix is
degenerate; the filter either silently identity-passes or produces
NaN-quietly-replaced-by-zero, so the gradient applied to the optimizer
in those first substeps is not what the method's theory prescribes.

**Diagnostic**: at substep 1, log `||M_anchor||_fro` and
`||G_proj − G_mask||` per target.

### D. FSDP integration of the spectral correction hook (implementation-bug class)
The spectral correction runs at
`after_actor_backward__before_optimizer_step` with `use_orig_params=True`
so `p.grad` surfaces as a 2D Tensor / DTensor post-reduce. Two
non-obvious risk surfaces:

- **(a)** The hook reads `p.grad` and writes it back; if it runs BEFORE
  FSDP1's gradient all-reduce completes, it operates on a shard not the
  reduced tensor → per-rank inconsistent projection.
- **(b)** `use_orig_params=True` surfaces some params as DTensor and
  others as ordinary tensors depending on FSDP wrap boundaries; mixed
  matching produces shape-inconsistent `G_proj` across ranks.

**Diagnostic**: at step 1, log each target's grad type, rank-local vs
full-tensor shape, AND `p.grad` value immediately before vs after the
FSDP reduce.

### E. Anchor clone gradient harvest correctness (silent-no-op class)
The cached anchor clone's `named_parameters()` may return DTensor-wrapped
tensors while `target_substr` matching uses live-module names. If
matching misses targets silently, `M_anchor` stays empty for those
targets → spectral filter is broken for them → mask noise passes through
unfiltered.

**Diagnostic**: on the first anchor refresh, log
`{targets_in_substr_set, targets_matched_in_clone, targets_with_nonzero_grad}`;
expect all three counts equal.

### F. Mask × spectral interaction at α=0.5 (conditioning class)
At α=0.5, `p.grad = 0.5 · G_mask + 0.5 · G_filt`. If `M_anchor` is
poorly conditioned (narrow basis, rank deficiency), `G_filt` could
amplify rather than damp `G_mask` in some directions, so the blend has
both the raw mask variance AND the amplified projection.

**Diagnostic**: log per-target singular values of `M_anchor` across the
first 10 anchor refreshes; flag any target whose condition number > 1e6.

### G. Memory-mitigation knobs as confounds (host-side class)
The comm-eff dry-run halved `ppo_mini_batch_size` (64→32) and
`PPO_MAX_TOKEN_LEN_PER_GPU` (36864→18432) and dropped vLLM mem-util
(0.4→0.3) to fit the anchor clone's ~3 GB park-cost on 4×H200. These
each add ~2× per-substep gradient variance, stacked ~4×. They DIFFER
from baseline only because of comm-eff overhead.

**Diagnostic**: any post-gate experiment that wants to attribute
grad_norm to the method must restore baseline batch knobs and provision
8×H100/H200 for headroom — otherwise the smaller-batch variance
confounds the comm-eff signal.

---

## Sequential test plan — Tests 1 → 4

Run them **in order**. Each later test assumes the prior one's verdict
landed; if Test 1 fails the gate, stop and fix the scaffolding before
running Test 2.

Cost discipline (per `.claude/plans/TEMPLATE.md` §"Vast.ai utilization
discipline"): each test's cells run **back-to-back on a single
provisioned Vast.ai instance**, not one provisioned box per cell. Tear
down only after the test's last cell rsyncs its metrics.

### Test 1 — `scaffold-noop-at-baseline-knobs` (the GATE)

**Question**: does turning comm-eff OFF reproduce the dense baseline's
step-1 shape at baseline batch knobs, AND does the dense baseline itself
reproduce on this branch?

**Two cells**, both on 4×H200, baseline batch knobs (mini=64,
wedge=36864, util=0.4, total_epochs=2, val_before_train=True,
test_freq=25), `actor.fsdp_config.use_orig_params=true`,
**25 trainer steps each**.

| Cell | `comm_eff.enabled` | `use_kl_loss` | `kl_loss_coef` | What it verifies |
|---|---|---|---|---|
| **A — dense-reference-reproduction** | `false` | `True` | `0.001` | We can reproduce the dense baseline's step-1 shape (`grad_norm=0.36, entropy=0.37, score=0.12`) on this branch. Pure reproducibility check. |
| **B — scaffold-noop** | `false` | `False` | `0` | The comm-eff scaffolding (config schema, disabled hooks, `use_orig_params=true`) does NOT silently regress the no-comm-eff path. KL is off, but `KL ≈ 0` at step 1, so Cell B's step-1 grad_norm should match Cell A's. |

**Expected** (at step 1):

| metric | Cell A target | Cell B target | tolerance |
|---|---:|---:|---|
| `actor/grad_norm` | 0.36 | ≈ Cell A | ±0.10 |
| `actor/entropy` | 0.37 | ≈ Cell A | ±0.10 |
| `critic/score/mean` | 0.12 | ≈ Cell A | ±0.05 |

**Verdict**:
- Cell A in tolerance AND Cell B within tolerance of Cell A → **GATE
  PASS**. Scaffolding is clean; the comm-eff dry-run's explosion is in
  the method or in the comm-eff-related batch knobs. Proceed to Test 2.
- Cell A in tolerance, Cell B step-1 grad_norm > 1.0 → **scaffold
  regressed** by one of {`use_orig_params=true`, schema additions,
  hook structure} even when comm-eff is disabled. Investigation pauses
  until parity restored; the fix likely targets the FSDP integration
  (cause D). Do not proceed to Tests 2-4 — they would conflate the
  scaffold bug with the method.
- Cell A outside tolerance → the branch itself doesn't reproduce dense
  baseline. Something more fundamental regressed. Diff this branch
  against the recorded baseline commit before going further.

Document the step-2 through step-25 trajectory for context, but DO NOT
draw method-level conclusions from Test 1 — it's a regression check,
not a method evaluation.

### Test 2 — `mask-recompute-isolates-step-1-IS-variance`

**Runs only if Test 1 passes the gate.**

**Question**: does the IS-variance-under-independent-PRF-masks (cause B)
explain the step-1 grad_norm in the comm-eff dry-run?

**Configuration**: two cells, both at baseline batch knobs (mini=64,
wedge=36864, util=0.4), no KL, no entropy, full comm-eff method
otherwise (p=0.9, α=0.5, τ=0.01, cadence=5, β_anc=0.9). **Provision
8×H100 or 8×H200** so the anchor clone has headroom.

| Cell | `comm_eff.mask.mask_recompute` | What it tests |
|---|---|---|
| **A — recompute=true (current)** | `true` | mask fires on both the actor-train forward and the `compute_log_prob` recompute |
| **B — recompute=false** | `false` | mask fires only on the actor-train forward; `compute_log_prob` runs unmasked so `log_p_old` is a clean estimate |

**10 trainer steps each** (the step-1 metric is the headline; ~10
steps of trajectory for context is enough).

**Verdict**:
- Cell A step-1 grad_norm **>>** Cell B step-1 grad_norm → cause B is
  real. Fix: either share PRF keys between the two paths or disable
  `mask_recompute`.
- Cell A ≈ Cell B → cause B is NOT the dominant driver; move to Test 3.

### Test 3 — `spectral-conditioning-at-startup`

**Runs if Test 2 is inconclusive or partial.**

**Question**: is the spectral filter's behavior on an empty / poorly
conditioned `M_anchor` (causes C and F) contributing to the
early-substep grad_norm?

**Configuration**: three cells, all at baseline batch knobs, no KL, no
entropy, full comm-eff method except for the variable being probed.
8×H100/H200.

| Cell | `comm_eff.spectral.alpha` | `comm_eff.spectral.seed_anchor_cache` | What it tests |
|---|---|---|---|
| **A — alpha=1.0** | `1.0` | `false` | spectral correction is an exact no-op (`G_proj = G_mask`) — isolates whether the spectral hook itself contributes |
| **B — alpha=0.5 (current)** | `0.5` | `false` | the current default — control |
| **C — seeded basis** | `0.5` | `true` | `M_anchor` pre-seeded with a deterministic PSD basis instead of starting empty — isolates the early-substep empty-basis pathology |

**10 trainer steps each** (the early-substep window is where the
signal lives; ~10 steps captures 2 anchor refreshes).

**Verdict**:
- Cell A **<<** Cell B at step 1 → spectral correction is amplifying
  rather than damping noise. Cause F is real; consider lower α or
  higher τ.
- Cell C **<<** Cell B at steps 1–5 → the empty-`M_anchor` pathology
  (cause C) is real. Fix: seed the cache or delay spectral correction
  until after the first anchor refresh.
- Cells A, B, C all roughly equal → causes C and F are not the
  dominant drivers; move to Test 4.

### Test 4 — `fsdp-spectral-integration-audit` (diagnostic, minimal training)

**Runs if Tests 2 and 3 didn't identify the dominant cause.**

**Question**: are causes D and E (FSDP integration + anchor harvest
correctness) producing silent shape inconsistencies or silent no-ops?

**Two cells**, both at baseline batch knobs (mini=64, wedge=36864,
util=0.4), no KL, no entropy.

| Cell | `comm_eff.enabled` | What it produces |
|---|---|---|
| **A — disabled** | `false` | baseline grad shape and FSDP behavior for comparison; same configuration as Test 1 Cell B |
| **B — enabled, instrumented** | `true` (full method) | the same comm-eff method as Test 2/3, but with diagnostic logging added per causes D and E (grad-type, shapes, before/after reduce values, target-match counts, M_anchor condition numbers across the first 10 anchor refreshes) |

**5 trainer steps each** (this is a diagnostic — the logs at step 1
and the first anchor refresh are the deliverable; 5 steps gives a
small buffer).

**Verdict** — structural rather than numeric. Compare the diagnostic
logs:
- Any target silently mis-typed (DTensor vs Tensor mismatch) → cause D(b)
  is real; the fix specifies the matching path.
- Anchor harvest's three target counts differ → cause E is real; fix
  the target-match logic in the clone.
- Spectral hook sees `p.grad` before the FSDP all-reduce completes →
  cause D(a) is real; the fix moves the hook insertion point.
- Any target's M_anchor condition number > 1e6 → cause F is real even
  if Test 3 didn't catch it.

The issue body should call out the specific module path(s) where the
anomaly was logged so the fix can target them precisely.

---

## Reference artifacts (cite these in the issue body)

- `research/runs/baseline/config.yaml` — dense baseline fixed config
  (the source of the step-1 grad_norm=0.36 reference)
- `research/runs/baseline/REPRODUCIBILITY.md` — baseline launcher SHA pin
- `research/runs/communication-baseline/verdict.md` — comm-eff baseline PASS
- `research/runs/communication-baseline/train.log` — comm-eff baseline log
- `research/runs/communication-baseline/REPRODUCIBILITY.md` — comm-eff
  baseline reproducibility manifest
- `research/notes/anchor-memory-cost.md` — anchor 3 GB clone explanation
- `research/notes/fast-circuit-vs-anchor-pass.md` — which forwards get
  masked
- `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` — dense
  baseline launcher (with KL loss enabled)
- `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
  — comm-eff baseline launcher (KL off by design)
- `verl/workers/config/comm_eff.py` — config schema (knob meanings)
- `verl/workers/comm_eff/state.py` — runtime state + `MASK_ELIGIBLE_TAGS`
- `verl/workers/comm_eff/activation_mask.py` — the PRF mask hook
- `verl/workers/comm_eff/anchor.py` — cloned-no-hook anchor module
- `verl/workers/comm_eff/spectral_filter.py` — EMA + SVD + Tikhonov + α-blend
- `verl/workers/engine_workers.py` — `compute_log_prob` `mask_active` stamp
- `verl/workers/engine/fsdp/transformer_impl.py` — `_comm_eff_mask_active`
  gating

## Issue format

**Title**: "Investigate: paper-scale grad_norm explosion in comm-eff —
sequential test plan from gate to integration audit (KL stays off in
the method evaluation)"

**Labels**: `kind:investigation`, `milestone:M2`

**Body sections (markdown)**:
1. Observation (with the numeric step-by-step grad_norm/ppo_kl/entropy
   table)
2. Reference dense-baseline step-1 numbers (`grad_norm=0.36 /
   entropy=0.37 / score=0.12` with KL=0.001)
3. Full comm-eff baseline configuration (the verified-PASS knob table)
4. Candidate root causes A-G (each with diagnostic suggested).
   **Section A is documented as a known late-step driver but is NOT
   proposed for toggle-back-on in the method evaluation.**
5. Sequential test plan — Tests 1, 2, 3, 4 — with Cell A/B(/C)
   configurations, step counts, and verdicts as spelled out above. The
   tests run in order; Test 2 only runs if Test 1's gate passes; Test
   3 only if Test 2 is inconclusive; Test 4 only if Tests 2 and 3
   don't identify the dominant cause. Cost discipline: one provisioned
   instance per test, cells back-to-back.
6. Reference artifacts (file paths)
7. Out-of-scope for this issue: the fix itself. This issue's
   deliverable is a verdict on which root cause is dominant and a
   target module/knob for the follow-up fix.

**Do NOT** specify a fix. **Do NOT** request `status:approved`. The
issue is investigation-only; the human operator decides whether to
dispatch experiments based on the test plan.

Print the full `gh issue create --title ... --body ...` command to
stdout for review.
