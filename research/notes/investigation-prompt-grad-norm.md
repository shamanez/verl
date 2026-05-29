# Investigation prompt — paste into a new session to draft the GitHub issue

> Paste the block below (everything between the long horizontal rules) into a
> fresh Claude Code session. It is self-contained: the new session does not
> need access to any prior conversation. Its output will be a single
> `gh issue create` command to print to stdout for review (it will NOT post
> the issue automatically).

---

You are picking up a research investigation on `shamanez/verl`, branch
`vast-ai-workload`. The research repo (issue queue) is
`shamanez/verl-compression-research` and is set as the local gh-default; the
code repo (PR target) is `shamanez/verl` with base `vast-ai-workload` (NEVER
`main` — `main` tracks upstream).

Your only job is to draft a GitHub issue. Do NOT write a plan file, do NOT
modify code, do NOT provision compute. Output a single `gh issue create`
invocation (title + body in markdown) — do not actually post it; print to
stdout for human review.

### Operator constraint (load-bearing)

**The KL anchor (`actor.use_kl_loss`) stays OFF for all tests in this
investigation.** That is a deliberate design choice for the
communication-efficient method (it isolates the compression effect from
KL regularization). Do not propose toggling KL back on as a fix. The
hypothesis about KL removal acting as a late-step driver of policy
collapse is still WORTH DOCUMENTING in the issue body as background, but
the test plan must remain at `use_kl_loss=False / kl_loss_coef=0`
everywhere.

---

## Observation

A dry-run scale-up of the communication-efficient baseline to paper-scale
rollouts (`TRAIN_BATCH=128, ROLLOUT_N=8, MAX_PROMPT=1024, MAX_RESPONSE=16384`)
produced symptoms that the verified-PASS smoke configuration (see
`runs/communication-baseline/` and `findings/communication-baseline.md`)
did not. The dense baseline (`runs/baseline/`, verl unmodified, WITH KL
loss enabled) also did not show these symptoms.

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
- **Entropy collapse** 6.4 → 0.023 over 58 steps (policy becomes
  near-deterministic).
- **`ppo_kl` explosion** 0.04 → 1.4 (PPO trust region assumes < 0.1).
- **Step-1 grad_norm = 1134** — high even BEFORE any policy drift could
  matter. The variance amplification is happening from the very first
  gradient.

`response_length/max` repeatedly hits the truncation cap (multiple steps),
consistent with policy collapse generating repetitive output until
truncation.

## Reference: dense baseline numbers (WITH KL loss)

Running `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
verbatim on `main` (no comm-eff, no scaffolding edits, **KL loss enabled
at `kl_loss_coef=0.001`**) gives, at step 1:

| metric | step-1 value |
|---|---:|
| `actor/grad_norm` | **0.36** |
| `actor/entropy` | 0.37 |
| `critic/score/mean` | 0.12 |

That's the "healthy step-1" shape. The dense baseline run goes on to
improve markedly over 100 steps. The comm-eff dry-run's step-1
grad_norm of **1134** is ~3000× the dense reference — that is the
specific number this investigation has to explain.

## Candidate root causes — enumerate ALL of these in the issue body

### A. KL anchor removal (known driver of LATE-step collapse, but NOT under test)
`actor.use_kl_loss=False` removes the `0.001·KL(actor || ref)` term. At
step 1 this is irrelevant (`KL ≈ 0`), but cumulatively over 58 steps it
removes the soft anchor that holds π close to π_ref. The entropy
collapse + `ppo_kl` explosion are textbook symptoms. **By operator
decision the KL toggle stays off** — document this as the late-step
driver but DO NOT propose toggling it on. The investigation must explain
whether the comm-eff method itself (independent of KL) is well-conditioned
at paper scale.

### B. Importance-sampling variance under independent PRF masks (the step-1 cause candidate)
`mask_recompute=true` means BOTH the actor-train forward AND the
`compute_log_prob` (old_logprob) forward route through the masked path.
But the PRF key (substep counter × layer index × base seed) **differs
between the two paths by design** — `compute_log_prob` fires once per
train step; the PPO inner loop fires `N×E` times per train step.

So mask realizations DIFFER between `log_p_current` and `log_p_old`. The
PPO importance ratio
```
r = exp(log_p_current − log_p_old)
```
is the ratio of two stochastic estimates under different masks, even
when the underlying actor weights are identical. **At step 1, `r ≠ 1`
even though `π_new == π_old`** — purely from the mask-realization
difference.

Expected variance of `log(r)` at step 1 scales like `2 · Var(masked
logit)` per token. This is the most likely explanation for the step-1
grad_norm of 1134 being ~3000× the dense reference.

**Diagnostic**: at step 1, log per-token `(log_p_current − log_p_old)`
histogram; expect non-zero spread even though `π_new == π_old`.

### C. Spectral filter on empty `M_anchor` at startup
`seed_anchor_cache=false` → `M_anchor` starts at zero. First anchor
refresh fires at PPO substep 5 (cadence=5). For substeps 1–4 the
spectral filter operates on `M_anchor = 0`; its SVD basis is degenerate
(`U, S, V` all zero, or NaN-quietly-replaced-by-identity).

The blended gradient `G_proj = α·G_mask + (1−α)·G_filt` at `α=0.5` in
this regime is either:
- (a) silently identity-passing `G_mask` through (no correction applied), or
- (b) producing NaN-quietly-replaced-by-zero outputs that bias the actor.

Either way, the filter is non-functional for the first ~5 substeps and
the gradient that's applied to the optimizer is not what the method's
theory prescribes.

**Diagnostic**: at substep 1, log `||M_anchor||_fro`, log
`||G_proj − G_mask||` per target.

### D. FSDP integration of the spectral correction hook (implementation-bug class)
The spectral correction runs at
`after_actor_backward__before_optimizer_step` with
`use_orig_params=True` so `p.grad` surfaces as a 2D Tensor / DTensor
post-reduce. The anchor backward runs on a hookless clone so FSDP1's
`_post_backward_hook` doesn't fire on the anchor pass.

Two non-obvious risk surfaces:
- **(a)** On the LIVE module, the spectral hook reads `p.grad` and writes
  `p.grad` back. If the hook runs BEFORE FSDP1's gradient all-reduce
  completes, it operates on a shard not the reduced tensor; the
  projection is then per-rank inconsistent.
- **(b)** `use_orig_params=True` surfaces some params as DTensor and
  others as ordinary tensors depending on FSDP wrap boundaries; if
  `target_substr` matching catches a mix, the per-target `G_proj` math
  is shape-inconsistent across ranks.

**Diagnostic**: at step 1, log each target's grad type, rank-local
shape vs full-tensor shape, AND the value of `p.grad` seen by the
spectral hook immediately before vs immediately after the FSDP reduce.

### E. Anchor clone gradient harvest correctness (silent-no-op class)
The anchor clone is cached on `self._anchor_module_cache`. Each refresh:
load K-stale weights → forward → `loss.backward()` → harvest `p.grad`
into the EMA. If `clone.named_parameters()` returns DTensor-wrapped
tensors but `target_substr` matching uses live-module names, the
matching could miss targets (silent no-op: `M_anchor` stays empty →
spectral filter is broken for those targets → mask noise passes through
unfiltered).

**Diagnostic**: on the first anchor refresh, log
`{targets_in_substr_set, targets_matched_in_clone, targets_with_nonzero_grad}`;
expect all three counts equal.

### F. Mask × spectral interaction at α=0.5 (conditioning class)
At α=0.5, `p.grad = 0.5 · G_mask + 0.5 · G_filt`. If `M_anchor` is
poorly conditioned (early steps, narrow basis, or rank deficiency
from the lowrank SVD path), `G_filt` could amplify rather than damp
`G_mask` in some directions.

**Diagnostic**: log per-target singular values of `M_anchor` across
the first 10 anchor refreshes; flag any target whose condition number
is > 1e6.

### G. Memory-mitigation knobs as confounds (host-side class)
The comm-eff dry-run used `ppo_mini_batch_size=32`,
`PPO_MAX_TOKEN_LEN_PER_GPU=18432`, `gpu_memory_utilization=0.3` to fit
the anchor clone's ~3 GB park-cost on 4×H200. These knobs DIFFER from
the dense baseline (mini=64, wedge=36864, util=0.4) only because of
comm-eff overhead. They each add ~2× per-substep gradient variance.

**Diagnostic**: revert to baseline batch knobs and run on 8×H100/H200
where the anchor clone fits without compromise.

---

## Discriminating test plan — include this section in the issue

**Naming convention**: every test name spells out what it isolates so
the analyst doesn't have to guess. Test 1 is the gate — every later
test assumes the gate passes.

KL stays OFF in all tests (`actor.use_kl_loss=False`,
`actor.kl_loss_coef=0`, `algorithm.use_kl_in_reward=False`,
`actor.entropy_coeff=0`). Each test is ≤ 25 steps, ≤ $5 of compute.

### Test 1 — `scaffold-noop-at-baseline-knobs` (the GATE)

**Question**: does turning comm-eff OFF reproduce the dense baseline's
step-1 grad_norm shape, holding everything else at baseline values
except the KL toggle?

**Configuration**:
- `comm_eff.enabled=false` (the comm-eff scaffolding is present in the
  code but the master switch is off — this is the no-op path that
  earlier work claimed is bit-identical to dense)
- `actor.use_kl_loss=False` (the only deliberate deviation from the
  dense baseline — required by the method's design)
- `actor.fsdp_config.use_orig_params=true` (current code's default for
  this branch — the spectral hook requirement)
- **Baseline batch knobs** (NOT the comm-eff dry-run's OOM-mitigated
  values):
  - `ppo_mini_batch_size = 64`
  - `PPO_MAX_TOKEN_LEN_PER_GPU = 36864`
  - `LOG_PROB_MAX_TOKEN_LEN_PER_GPU = 36864`
  - `rollout.gpu_memory_utilization = 0.4`
  - `total_epochs = 2`
- `val_before_train = True`, `test_freq = 25`
- 4×H200, 25 trainer steps.

**Expected, by comparison to the dense baseline reference**:

| metric @ step 1 | dense baseline (WITH KL) | this test (NO KL) | tolerance |
|---|---:|---:|---|
| `actor/grad_norm` | 0.36 | should be **≈ 0.36** | ±0.10 |
| `actor/entropy` | 0.37 | should be **≈ 0.37** | ±0.10 |
| `critic/score/mean` | 0.12 | should be **≈ 0.12** | ±0.05 |

At step 1 the KL term is identically zero (actor == ref), so removing
KL should NOT shift the step-1 metrics in any noticeable way. Any
difference at step 1 means the comm-eff scaffolding (the
`use_orig_params=True` flag, the disabled `comm_eff.*` hooks, the
config schema additions, etc.) silently regressed the no-comm-eff path.

**Verdict criteria**:
- step-1 grad_norm in `[0.26, 0.46]` (within ±0.10 of 0.36) → **GATE PASS**.
  The scaffolding is clean; the explosion is specifically in the
  comm-eff method (run Test 2 next).
- step-1 grad_norm > 1.0 → **GATE FAIL**. The scaffolding regressed the
  dense path. The investigation must first restore the no-op parity
  before any comm-eff-enabled test is informative. Likely culprits:
  `use_orig_params=true` interaction with the FSDP backward path, or a
  hook that fires even when `comm_eff.enabled=false`. Add the
  diagnostic logging from cause D and look at p.grad type and
  shape on the disabled path.

Late-step trajectory in this test is secondary; the step-1 metrics are
the gate. Document the step-2 through step-25 trajectory for context
but don't try to draw conclusions about the comm-eff method from this
test — it's a regression check.

### Test 2 — `mask-recompute-isolates-step-1-IS-variance`

**Runs only if Test 1 passes the gate.**

**Question**: does the IS-variance-under-independent-PRF-masks (cause B)
explain the step-1 grad_norm in the comm-eff dry-run?

**Configuration**: two cells, both at baseline batch knobs (mini=64,
wedge=36864, util=0.4), no KL, no entropy, full comm-eff method otherwise
(p=0.9, α=0.5, τ=0.01, cadence=5, β_anc=0.9). Provision 8×H100 or
8×H200 so the anchor clone has headroom.

- **Cell A**: `comm_eff.mask.mask_recompute = true` (the current
  default; mask fires on both the actor-train forward and the
  `compute_log_prob` recompute).
- **Cell B**: `comm_eff.mask.mask_recompute = false` (mask fires only
  on the actor-train forward; `compute_log_prob` runs unmasked so
  `log_p_old` is a clean estimate).

25 steps each.

**Verdict**:
- Cell A step-1 grad_norm >> Cell B step-1 grad_norm → cause B is real.
  The IS-variance-under-independent-masks fix is to either share PRF
  keys between the two paths or to disable `mask_recompute`.
- Cell A ≈ Cell B → cause B is NOT the dominant driver; move to
  Test 3.

### Test 3 — `spectral-conditioning-at-startup`

**Question**: is the spectral filter's behavior on an empty / poorly
conditioned `M_anchor` (causes C and F) contributing to the early-substep
grad_norm?

**Configuration**: three cells, all at baseline batch knobs, no KL, no
entropy, full comm-eff method except for the variable being probed.
8×H100/H200.

- **Cell A**: `comm_eff.spectral.alpha = 1.0` (spectral correction is
  an exact no-op — `G_proj = G_mask`).
- **Cell B**: `comm_eff.spectral.alpha = 0.5` (current default).
- **Cell C**: `comm_eff.spectral.alpha = 0.5` AND
  `comm_eff.spectral.seed_anchor_cache = true` (`M_anchor` is
  pre-seeded with a deterministic PSD basis instead of starting empty).

25 steps each.

**Verdict**:
- Cell A << Cell B at step 1 → the spectral correction is amplifying
  rather than damping noise. Cause F is real; consider lower α or
  a higher τ.
- Cell C << Cell B at steps 1-5 → the empty-`M_anchor` pathology
  (cause C) is real; seed the cache or delay spectral correction until
  after the first anchor refresh.

### Test 4 — `fsdp-spectral-integration-audit` (diagnostic, no new training)

**Question**: are causes D and E (FSDP integration + anchor harvest
correctness) producing silent shape inconsistencies or silent no-ops?

**Configuration**: same as Test 1 (no comm-eff active) AND a re-run of
Test 1 with `comm_eff.enabled=true` and the diagnostic logging
described in causes D and E added to the spectral hook and the anchor
harvest.

Compare the diagnostic logs. The verdict is structural rather than
numeric: if any target is silently mis-typed (DTensor vs Tensor
mismatch), silently un-matched (anchor harvest miss), or silently
sharded (hook runs before reduce), the issue body should call out the
specific module path so the fix can target it.

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
gate test against dense baseline + scaffolding-vs-method isolation (KL
stays off)"

**Labels**: `kind:investigation`, `milestone:M2`

**Body sections (markdown)**:
1. Observation (with the numeric step-by-step grad_norm/ppo_kl/entropy
   table)
2. Reference dense-baseline step-1 numbers
   (`grad_norm=0.36 / entropy=0.37 / score=0.12` with KL=0.001)
3. Candidate root causes (the 7 hypotheses A–G above, each with
   diagnostic suggested). **Section A is documented as a known
   late-step driver but is NOT proposed for toggle-back-on.**
4. Discriminating test plan (Tests 1–4, named semantically). Test 1 is
   the gate — every later test assumes the gate passes. All tests have
   `use_kl_loss=False`.
5. Reference artifacts (file paths)
6. Out-of-scope for this issue: the fix itself. This issue's
   deliverable is a verdict on which root cause is dominant and a
   target module/knob for the follow-up fix.

**Do NOT** specify a fix. **Do NOT** request `status:approved`. The
issue is investigation-only; the human operator decides whether to
dispatch experiments based on the test plan.

Print the full `gh issue create --title ... --body ...` command to
stdout for review.
