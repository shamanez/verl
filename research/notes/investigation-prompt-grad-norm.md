# Investigation prompt — paste into a new session to draft the GitHub issue

> Paste the block below (everything between the long horizontal rules) into
> a fresh Claude Code session. It is self-contained: the new session does
> not need access to any prior conversation. Its output will be a single
> `gh issue create` command printed to stdout for human review (it will NOT
> post the issue automatically).

---

You are picking up a research investigation on `shamanez/verl`, branch
`vast-ai-workload`. The research repo (issue queue) is
`shamanez/verl-compression-research` and is set as the local gh-default;
the code repo (PR target) is `shamanez/verl` with base `vast-ai-workload`
(NEVER `main` — `main` tracks upstream).

Your only job is to draft ONE GitHub issue. Output a single
`gh issue create` invocation (title + body in markdown). Do NOT actually
post it; print the full command to stdout for human review. Do NOT write a
plan file, do NOT provision compute.

### What kind of issue this is (read before drafting)

This is a **hybrid "peel-and-fix" issue**, not a pure diagnostic. It does
two things in one sequence:

1. **Diagnose** *where* the paper-scale `grad_norm` explosion lives by
   **peeling the comm-eff method apart one circuit at a time** — full
   method → spectral blend off → mask-only (no anchor, no spectral). Each
   peel removes machinery so the next verdict localises the cause.
2. **Validate a minimal candidate stabiliser** — a periodic **clean
   (unmasked) optimizer step** that refreshes AdamW's moments with the
   *true* gradient every `N` steps. This is the headline, **mandatory**
   test (Test 4). It is potentially the smallest possible fix and, because
   it touches none of the FSDP-fragile anchor/spectral code, also the
   safest.

Consequence for labelling: **most cells are `code_change:false`** (they
are reachable with existing config knobs). The **periodic-clean-step cell
is `code_change:true`** — it needs a tiny new knob + train-loop hook and
must ride an `exp/<N>-<slug>` branch (base `vast-ai-workload`). Mark
`code_change` per-cell in the body; the issue overall carries
`code_change:true` because it contains a mandatory code-change experiment.

### Operator constraints (load-bearing — read these first)

1. **The communication-efficient method's design is no-KL no-entropy.**
   `actor.use_kl_loss=False`, `algorithm.use_kl_in_reward=False`,
   `actor.entropy_coeff=0`. Every experiment that tests the COMM-EFF
   METHOD itself runs no-KL. The one exception is the gate test (Test 1,
   Cell A), which reproduces the dense baseline WITH KL purely as a
   reference / sanity point — it is not part of the method evaluation.

2. **The peel is the plan.** Tests run as a sequence that removes one
   circuit at a time. Test 1 is the gate. Tests 2–3 peel the method down.
   **Test 4 (the periodic clean step) is MANDATORY — it runs regardless
   of what Tests 2–3 conclude**, because it is simultaneously the sharpest
   diagnostic (is the instability an optimizer-state problem?) and the
   leading candidate fix. Test 5 (the FSDP/anchor integration audit) is
   the only *conditional* test — run it only if the peel implicates the
   anchor/spectral machinery.

3. **`comm_eff.enabled=true` for every method cell.** The only cells that
   disable the method are the gate's Cell B (regression check) and Test 5
   Cell A (no-comm-eff baseline for the FSDP audit).

4. **FSDP-no-errors is a HARD acceptance criterion for every cell.** Any
   FSDP/DTensor/Ray-unhandled/NaN/OOM error is an automatic STOP for that
   cell with the traceback captured — not a metric to be averaged over.
   The periodic-clean-step cell in particular must be shown to run with
   *zero* FSDP errors (see its explicit checklist in §6); that is the
   whole reason it is built on the leanest path.

---

## 0. Why this matters — relation to the project north-star

The project north-star (`research/.claude/GOAL.md`) is **stable
communication-efficient GRPO at paper scale** — reaching reward/accuracy
parity with the dense GRPO baseline while *measurably* cutting
communication. The step-1 `grad_norm` explosion documented below is the
single symptom that currently blocks "Done":

- it prevents **stable** training (Done criterion 1), and
- everything downstream — the **parity** curve (criterion 2), the
  **savings** number (criterion 3), the **promoted launcher**
  (criterion 4) — is gated on a run that does not diverge.

The peel localises the cause; the periodic clean step is a candidate that,
if it works, reaches the north-star with **far less code and FSDP risk**
than the full anchor+spectral apparatus while keeping ~90% of the
communication savings (9 of every 10 steps remain masked). That is why it
is mandatory, not optional.

## 1. Observation

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
| 1  | 1134 | 0.04 | 6.42  | 0.28 | 0.32 |
| 10 | 1283 | 0.07 | 6.48  | 0.26 | 0.31 |
| 20 | 1342 | 0.17 | 6.24  | 0.28 | 0.31 |
| 30 | 1477 | 0.19 | 5.39  | 0.25 | 0.28 |
| 40 | 1465 | 0.41 | 4.36  | 0.22 | 0.25 |
| 50 | 1712 | 0.77 | 1.85  | 0.25 | 0.31 |
| 56 | 1884 | 1.38 | 0.057 | 0.29 | 0.35 |
| 58 | 1662 | 1.04 | 0.023 | 0.24 | 0.29 |

Three signals, and they live on **two different timescales** — keep them
separate, because the fixes are different:

- **Step-1 magnitude.** `grad_norm = 1134` at step 1 — high *before* any
  policy drift could matter. This is an *initialisation-time* variance/bias
  problem (causes B, C, F, H below), present in the very first gradient.
- **Accumulation over ~50 steps.** Entropy collapses 6.4 → 0.023 and
  `ppo_kl` grows 0.04 → 1.4 (PPO trust region assumes < 0.1). This is a
  *drift* problem — biased/high-variance gradients fed to AdamW
  step-after-step (causes A, G, H).

`response_length/max` repeatedly hits the truncation cap, consistent with
policy collapse generating repetitive output until truncation.

The mandatory periodic-clean-step test (Test 4) targets the **accumulation**
timescale; the peel (Tests 2–3) localises the **step-1** contribution.

## 2. Reference: dense baseline step-1 numbers (WITH KL loss)

Running `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
verbatim on `vast-ai-workload` (no comm-eff, no scaffolding edits, **KL
loss enabled at `kl_loss_coef=0.001`**) gives, at step 1:

| metric | step-1 value |
|---|---:|
| `actor/grad_norm` | **0.36** |
| `actor/entropy` | 0.37 |
| `critic/score/mean` | 0.12 |

The dense baseline improves markedly over 100 steps (`val/test_score`
0.087 → 0.789, recorded in `runs/baseline/`). The comm-eff dry-run's
step-1 `grad_norm` of **1134** is **~3000× the dense reference** — that is
the specific number this investigation has to explain, and the dense
100-step curve is the trajectory the clean-step variant must approach.

## 3. Full comm-eff baseline configuration (the method under investigation)

The configuration that PASSED at smoke scale (`runs/communication-baseline/`):

| component | knob | value | notes |
|---|---|---|---|
| master   | `comm_eff.enabled` | `true` | the method |
| mask     | `comm_eff.mask.enabled` | `true` | PRF Bernoulli at pipeline-boundary decoder blocks |
| mask     | `comm_eff.mask.p` | `0.9` | fraction zeroed; `h_tilde = h * mask` (**no `1/(1-p)` rescale → biased estimator**) |
| mask     | `comm_eff.mask.mask_recompute` | `true` | mask fires on BOTH gradient-feeding forwards |
| anchor   | `comm_eff.anchor.enabled` | `true` | hookless cloned-module backward |
| anchor   | `comm_eff.anchor.cadence` | `5` | every 5 PPO substeps |
| anchor   | `comm_eff.anchor.delay_K` | `5` | 5-substep stale weight snapshot |
| spectral | `comm_eff.spectral.enabled` | `true` | EMA → SVD → Tikhonov → α-blend |
| spectral | `comm_eff.spectral.alpha` | `0.5` | `G_proj = α·G_mask + (1−α)·G_filt` |
| spectral | `comm_eff.spectral.tau` | `0.01` | Tikhonov damping |
| spectral | `comm_eff.spectral.beta_anc` | `0.9` | EMA decay |
| spectral | `comm_eff.spectral.seed_anchor_cache` | `false` | live anchor populates `M_anchor` from zero |
| spectral | `comm_eff.spectral.ema_device` | `gpu` | `M_anchor` in HBM |
| spectral | `comm_eff.spectral.svd_mode` | `full` | full thin SVD |
| spectral | `comm_eff.spectral.basis_cache` | `cache` | reuse U/S/V across PPO mini-batches |
| spectral | `comm_eff.spectral.max_targets` | `4` | smoke cap |
| objective | `actor.use_kl_loss` | `False` | the method's design |
| objective | `algorithm.use_kl_in_reward` | `False` | the method's design |
| objective | `actor.entropy_coeff` | `0` | the method's design |
| FSDP     | `actor.fsdp_config.use_orig_params` | `true` | spectral hook needs full 2D Tensor post-reduce |

Launcher encoding this as defaults:
`examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
(KL off by design). All method cells below keep `mask.p=0.9` for
consistency with this verified-PASS config (the schema default is 0.95;
do not change it without a separate justification).

## 4. Candidate root causes — enumerate ALL of these in the issue body

### A. KL anchor removal — known LATE-step driver, **NOT under test**
`actor.use_kl_loss=False` removes the soft anchor holding π close to π_ref.
At step 1 this is irrelevant (`KL ≈ 0`); cumulatively it explains the
late-step entropy collapse + `ppo_kl` explosion. Documented as background
only. **Per operator constraint, KL stays off in the method evaluation** —
this cause is *not* proposed for toggle-back-on. (Test 1 / Cell A is the
only with-KL run, and it is a reference-only sanity reproduction.)

### B. IS variance **and no-rescale bias** under independent PRF masks — step-1 cause candidate
Two distinct effects from the same mask. **(Variance):**
`mask_recompute=true` masks BOTH the actor-train forward AND the
`compute_log_prob` (old-logprob) forward, but the PRF key
(substep × layer × seed) **differs between the two paths by design**
(`compute_log_prob` fires once per trainer step; the PPO inner loop fires
`N×E` times). So the importance ratio `r = exp(log_p_current − log_p_old)`
is a ratio of two stochastic estimates under *different* mask realisations.
**At step 1, `r ≠ 1` even though `π_new == π_old`** — purely from the
mask-realisation mismatch; `Var(log r)` scales like `2·Var(masked logit)`
per token. **(Bias):** because `h_tilde = h · mask` carries **no
`1/(1-p)` rescale**, the masked forward is a *biased* estimator
(`E[h_tilde] = (1-p)·h`; at `p=0.9` boundary activations are scaled to
~10% magnitude). The gradient computed through a biased forward is itself
biased — independent of the variance effect. Either or both can drive the
step-1 `grad_norm`.
> **Diagnostic:** at step 1, log the per-token `(log_p_current −
> log_p_old)` histogram (expect non-zero spread even though
> `π_new == π_old`), AND the boundary-layer `mean(h_tilde)/mean(h)` ratio
> (expect ≈ `1-p`, confirming the un-rescaled bias).

### C. Spectral filter on empty `M_anchor` at startup
`seed_anchor_cache=false` → `M_anchor = 0` for the substeps before the
first anchor refresh fires. The SVD basis of a zero matrix is degenerate;
the filter may silently identity-pass or produce NaN-quietly-replaced-by-zero,
so the gradient applied in those first substeps is not what the theory
prescribes.
> **Diagnostic:** at substep 1, log `||M_anchor||_fro` and
> `||G_proj − G_mask||` per target.

### D. FSDP integration of the spectral correction hook — implementation-bug class
The spectral correction runs at `after_actor_backward__before_optimizer_step`
with `use_orig_params=True` so `p.grad` surfaces as a 2D Tensor / DTensor
post-reduce. Two non-obvious risk surfaces:
- **(a)** The hook reads `p.grad` and writes it back; if it runs BEFORE
  FSDP1's gradient all-reduce completes, it operates on a shard not the
  reduced tensor → per-rank inconsistent projection.
- **(b)** `use_orig_params=True` surfaces some params as DTensor and others
  as ordinary tensors depending on FSDP wrap boundaries; mixed matching
  produces shape-inconsistent `G_proj` across ranks.
> **Diagnostic:** at step 1, log each target's grad type, rank-local vs
> full-tensor shape, AND `p.grad` value immediately before vs after the
> FSDP reduce.

### E. Anchor clone gradient harvest correctness — silent-no-op class
The cached anchor clone's `named_parameters()` may return DTensor-wrapped
tensors while `target_substr` matching uses live-module names. If matching
misses targets silently, `M_anchor` stays empty for those targets →
spectral filter is broken for them → mask noise passes through unfiltered.
> **Diagnostic:** on the first anchor refresh, log
> `{targets_in_substr_set, targets_matched_in_clone, targets_with_nonzero_grad}`;
> expect all three counts equal.

### F. Mask × spectral interaction at α=0.5 — conditioning class
At α=0.5, `p.grad = 0.5·G_mask + 0.5·G_filt`. If `M_anchor` is poorly
conditioned (narrow basis, rank deficiency), `G_filt` could amplify rather
than damp `G_mask` in some directions, so the blend carries both the raw
mask variance AND the amplified projection.
> **Diagnostic:** log per-target singular values of `M_anchor` across the
> first 10 anchor refreshes; flag any target with condition number > 1e6.

### G. Memory-mitigation knobs as confounds — host-side class
The comm-eff dry-run halved `ppo_mini_batch_size` (64→32) and
`PPO_MAX_TOKEN_LEN_PER_GPU` (36864→18432) and dropped vLLM mem-util
(0.4→0.3) to fit the anchor clone's ~3 GB park-cost on 4×H200 (see
`research/notes/anchor-memory-cost.md`). Each adds per-substep gradient
variance, and they differ from baseline *only* because of comm-eff
overhead.
> **Diagnostic:** any cell that attributes `grad_norm` to the method must
> restore baseline batch knobs and provision enough headroom that the
> smaller-batch variance does not confound the comm-eff signal.

### H. AdamW optimizer-state poisoning by biased + high-variance masked gradients — **the accumulation-timescale cause; the one Test 4 treats**
Because `G_mask` is both **biased** (no `1/(1-p)` rescale, cause B) and
**high-variance** (PRF mask realisations, cause B), feeding it to AdamW on
*every* step poisons the optimizer state itself:
- the **second moment** `v` accumulates inflated variance → the
  per-parameter effective learning rate is systematically distorted;
- the **first moment** `m` accumulates a biased direction → systematic
  drift of the policy.
Over ~50 steps this is exactly the entropy-collapse + `grad_norm`-growth
trajectory in §1. Note AdamW's memory horizons: `β1≈0.9` ⇒ `m` remembers
~10 steps, `β2≈0.999` ⇒ `v` remembers ~1000 steps. A clean (unmasked)
optimizer step every 10 steps therefore strongly re-aligns `m` but only
weakly refreshes `v` — Test 4 measures whether that is *sufficient*.

The anchor+spectral apparatus attacks this same bias in **gradient space**
(project `G_mask` onto an anchor-derived basis). The periodic clean step
attacks it directly in **optimizer-state space** (let AdamW step on the
true gradient periodically), with none of the FSDP-fragile machinery
(causes C/D/E/F all vanish). Test 4 asks whether the cheap optimizer-state
fix makes the expensive gradient-space fix unnecessary.
> **Diagnostic:** log per-target AdamW `||m||` and `||v||` every step for
> the mask-only cell vs the mask-only+clean cell; expect `v` to inflate and
> `m` to drift in the no-clean cell and stay bounded in the clean cell.

---

## 5. Test plan — peel the method down, then test the stabiliser (Tests 1 → 5)

Run them **in order**. The conditional structure is the fail-fast learning
loop: each verdict prunes the search before more GPU-hr is spent. The
runner encodes the headline predicate per test (e.g. step-1 `grad_norm`
ratio vs the dense reference, or "does entropy hold past step 40") and
evaluates it inline. **Test 4 is mandatory and is NOT gated on Tests 2–3's
verdicts** — only on Test 1 passing the gate.

### Execution & GPU-utilization discipline (per `.claude/plans/TEMPLATE.md` §"Vast.ai utilization discipline", HARD RULE)

- **One box for the whole Test 1 → 5 sequence.** Provision a single
  instance on the standard default tier (**4×H200 preferred, else 8×H100** —
  per `project.yaml.default_compute.gpu_filter_chain`) up front and chain
  all tests back-to-back (shared docker / verl checkout / dataset cache).
  Do NOT tear down and re-provision between tests.
- **Restore baseline batch knobs + provision headroom for method cells**
  (cause G): mini=64, wedge=36864, util=0.4 — the mask-only and clean-step
  cells drop the ~3 GB anchor clone, so they fit comfortably; do not
  re-introduce the smaller-batch confound.
- **Keep every GPU busy.** Saturate whatever was provisioned; declare any
  legitimately idle window (e.g. a long eval between cells) in the plan's
  `## Notes for runner` so the stall-watchdog thresholds get loosened.
- **Fail-fast.** If Test 1 fails the gate, short-circuit — do not spend
  GPU-hr on Tests 2–5. Tear down the instant the sequence resolves and
  metrics are rsynced to the laptop.

### Test 1 — `scaffold-noop-at-baseline-knobs` (the GATE) — `code_change:false`

**Question:** does turning comm-eff OFF reproduce the dense baseline's
step-1 shape at baseline batch knobs, AND does the dense baseline itself
reproduce on this branch?

**Two cells**, baseline batch knobs (mini=64, wedge=36864, util=0.4,
total_epochs=2, val_before_train=True, test_freq=25),
`actor.fsdp_config.use_orig_params=true`, **25 trainer steps each**.

| Cell | `comm_eff.enabled` | `use_kl_loss` | `kl_loss_coef` | What it verifies |
|---|---|---|---|---|
| **A — dense-reference-reproduction** | `false` | `True` | `0.001` | We reproduce the dense step-1 shape (`grad_norm=0.36, entropy=0.37, score=0.12`) on this branch. Pure reproducibility check. |
| **B — scaffold-noop** | `false` | `False` | `0` | The comm-eff scaffolding (config schema, disabled hooks, `use_orig_params=true`) does NOT silently regress the no-comm-eff path. KL off, but `KL ≈ 0` at step 1, so Cell B step-1 grad_norm should match Cell A. |

**Expected (step 1):**

| metric | Cell A target | Cell B target | tolerance |
|---|---:|---:|---|
| `actor/grad_norm` | 0.36 | ≈ Cell A | ±0.10 |
| `actor/entropy`   | 0.37 | ≈ Cell A | ±0.10 |
| `critic/score/mean` | 0.12 | ≈ Cell A | ±0.05 |

**Verdict:**
- Cell A in tolerance AND Cell B within tolerance of Cell A → **GATE PASS.**
  Scaffolding is clean; the explosion is in the method itself. Proceed to
  the peel (Test 2).
- Cell A in tolerance, Cell B step-1 grad_norm > 1.0 → **scaffold regressed**
  by one of {`use_orig_params=true`, schema additions, hook structure} even
  with comm-eff disabled (fix likely targets FSDP integration, cause D).
  **Do not proceed** — it would conflate the scaffold bug with the method.
- Cell A outside tolerance → the branch itself doesn't reproduce dense
  baseline. Diff this branch against the recorded baseline commit
  (`runs/baseline/REPRODUCIBILITY.md`) before going further.

### Test 2 — `peel-1-spectral-coefficient-off` (α=1.0) — `code_change:false`

**Runs only if Test 1 passes the gate.**

**Question:** does the spectral *blend* contribute to the explosion? At
α=1.0 the correction is an exact no-op (`G_proj == G_mask`) while the
anchor circuit still fires (harvest → EMA → SVD all run, but do not touch
the grads). So this peels off only the spectral *effect*, keeping every
other circuit live.

**Config:** two cells, baseline batch knobs, no KL, no entropy, full
comm-eff otherwise (p=0.9, anchor cadence=5/delay=5, τ=0.01, β_anc=0.9).

| Cell | `spectral.alpha` | What it tests |
|---|---|---|
| **A — α=0.5 (current default)** | `0.5` | reproduce the explosion on this box (control) |
| **B — α=1.0 (spectral off)** | `1.0` | spectral blend is an exact no-op; isolates whether the spectral projection is amplifying |

**10 trainer steps each** (step-1 + early-trajectory headline).

**Verdict:**
- Cell A explodes, Cell B **<<** Cell A at step 1 → the spectral projection
  is amplifying (causes C/F). The anchor/spectral path is implicated →
  schedule Test 5.
- Cell A ≈ Cell B (both explode) → spectral is NOT the driver; the
  explosion is upstream in the mask itself. Continue peeling (Test 3).

### Test 3 — `peel-2-mask-only` (no anchor, no spectral) — `code_change:false`

**Question:** does **pure masked GRPO** — mask straight to AdamW, with the
entire anchor+spectral apparatus removed — explode on its own? This peel
deletes causes C/D/E/F *and* the ~3 GB anchor clone (so cause G's memory
pressure is gone too). Whatever remains is the mask's own bias/variance
(cause B) and its effect on the optimizer (cause H).

**Config:** baseline batch knobs, no KL, no entropy,
`comm_eff.enabled=true`, `mask.enabled=true`, `mask.p=0.9`,
**`anchor.enabled=false`, `spectral.enabled=false`**.
`use_orig_params=true` kept for config parity (this path does not depend on
it — a further FSDP-risk reduction).

| Cell | `mask.mask_recompute` | What it tests |
|---|---|---|
| **A — recompute=true** | `true`  | mask fires on both gradient-feeding forwards (the dry-run setting), now with no anchor/spectral confound |
| **B — recompute=false** | `false` | mask fires only on the actor-train forward; `compute_log_prob` runs unmasked so `log_p_old` is a clean estimate — isolates the IS-mask-mismatch half of cause B |

**10 trainer steps each.**

**Verdict:**
- Cell A explodes → pure masking alone is unstable; the anchor/spectral
  apparatus is not the cause (it was masking the symptom at smoke scale).
  This is the strongest motivation for Test 4's optimizer-state fix.
- Cell A **>>** Cell B → the IS-mask-mismatch (cause B, variance half) is a
  real contributor; note it for the fix (share PRF keys between the two
  paths, or set `mask_recompute=false`).
- Cell A ≈ Cell B but both explode → the mismatch is not the driver; the
  remaining suspects are the no-rescale bias (cause B, bias half) and
  optimizer-state poisoning (cause H) — exactly what Test 4 addresses.
- Cell A is *stable* (grad_norm bounded, entropy holds) → the explosion
  lives in the anchor/spectral machinery, not the mask → Test 5 is now
  required to find which integration surface (D/E/F) is responsible.

### Test 4 — `mask-only-plus-periodic-clean-step` (MANDATORY; the candidate fix) — `code_change:true`

**Runs whenever Test 1 passed the gate — independent of Tests 2–3.** This
is the headline test and the leading candidate stabiliser.

**Idea.** Run pure masked GRPO (exactly Test 3's config — no anchor, no
spectral), but **every `clean_cadence = 10` trainer steps, run that whole
step unmasked on the live module and take the normal `optimizer.step()`**,
so AdamW's moments are periodically refreshed with the *true* dense
gradient. The other 9 of every 10 steps stay masked (so ~90% of the
communication savings is retained). This is fundamentally different from
the anchor circuit, which runs unmasked on a **stale clone** and **never**
steps the optimizer — here the **live** optimizer state is the thing being
corrected.

**Config:** baseline batch knobs, no KL, no entropy,
`comm_eff.enabled=true`, `mask.enabled=true`, `mask.p=0.9`,
`anchor.enabled=false`, `spectral.enabled=false`, plus the new
`comm_eff.clean_cadence` knob.

| Cell | `comm_eff.clean_cadence` | What it tests |
|---|---|---|
| **A — no clean step (control)** | `0` | pure masked GRPO with no refresh — the same config as Test 3 Cell A (reuse that run if already on the box; do not re-run needlessly). Expected to drift toward collapse. |
| **B — clean every 10** | `10` | masked GRPO with a clean unmasked optimizer step at steps 10, 20, 30, … Expected to bound `grad_norm` and hold entropy. |

**Run length: ≥ 60 trainer steps, target 100.** Unlike the step-1
diagnostic cells, this test is about the **accumulation** timescale — the
dry-run's collapse only became unmistakable around step 50, and the first
clean step is at step 10. Sixty steps is the minimum to see whether the
clean step prevents the collapse; 100 lets the curve be overlaid directly
on the dense baseline's 100-step `val/test_score` trajectory.

**Verdict (trajectory-based, not step-1):**
- Cell B holds entropy and keeps `grad_norm` bounded past step 40 while
  Cell A drifts/collapses → **the periodic clean step is a real
  stabiliser.** This is the minimal-fix result: it sidesteps the entire
  anchor/spectral path and stays FSDP-clean. Flag for the parity follow-up
  (does it reach dense-baseline reward?) and report the realised
  communication savings (fraction of masked steps × per-step mask saving).
- Cell B is no better than Cell A → the optimizer-state refresh at
  cadence 10 is insufficient (consistent with `v`'s ~1000-step memory; see
  cause H). Report the `||m||`/`||v||` diagnostic so the follow-up can
  decide between a shorter cadence and a gradient-space fix.
- **Either way, the FSDP-no-errors checklist in §6 must pass for Cell B**;
  an FSDP error there is a STOP and is itself a finding.

### Test 5 — `fsdp-anchor-spectral-integration-audit` (CONDITIONAL, diagnostic) — `code_change:false`

**Runs only if the peel implicated the anchor/spectral machinery** — i.e.
Test 2 Cell B (α=1.0) was much calmer than Cell A, or Test 3 Cell A
(mask-only) was *stable* while the full method explodes.

**Question:** are causes D and E (FSDP integration + anchor harvest
correctness) producing silent shape inconsistencies or silent no-ops?

| Cell | `comm_eff.enabled` | What it produces |
|---|---|---|
| **A — disabled** | `false` | baseline grad shape + FSDP behaviour for comparison (same as Test 1 Cell B) |
| **B — full method, instrumented** | `true` | the full method with diagnostic logging per causes C/D/E/F (grad-type, rank-local vs full shapes, before/after-reduce values, target-match counts, `M_anchor` condition numbers across the first 10 anchor refreshes) |

**5 trainer steps each** (the logs at step 1 and the first anchor refresh
are the deliverable).

**Verdict — structural, not numeric.** Call out the **specific module
path(s)** where each anomaly was logged so the follow-up fix can target
them precisely:
- target silently mis-typed (DTensor vs Tensor) → cause D(b);
- anchor harvest's three target counts differ → cause E;
- spectral hook sees `p.grad` before the FSDP all-reduce → cause D(a);
- any `M_anchor` condition number > 1e6 → cause F.

---

## 6. The periodic clean step — exact spec, FSDP-safety checklist, minimal patch

This section is for Test 4 (the only `code_change:true` cell). Put it in
the issue body so the implementer who opens the `exp/<N>-<slug>` branch has
an exact, FSDP-safe target.

### Semantics
- On trainer step `s`: if `clean_cadence > 0 and (s % clean_cadence) == 0`,
  the **entire step runs unmasked** — both gradient-feeding forwards (the
  `compute_log_prob` / old-logprob recompute AND the actor-train forward)
  must have masking OFF — and `optimizer.step()` runs as normal on the true
  dense gradient. Otherwise the step is masked exactly as today.
- The clean step **replaces** the masked step at that index (one
  `optimizer.step()` per trainer step either way) — it does not add a
  second optimizer step.
- There is **one** optimizer / param-group shared across masked and clean
  steps, so AdamW's `m`/`v` are genuinely refreshed (not a separate state).

### FSDP-no-errors acceptance checklist (HARD — Test 4 Cell B must satisfy all)
1. **Reuses the existing dense path.** The clean step takes the standard
   `train_batch` flow with the mask hook inert — no clone, no
   `summon_full_params` correction, no DTensor surgery. It is, by
   construction, the same FSDP path as a `comm_eff.enabled=false` step, so
   it inherits that path's FSDP correctness.
2. **Per-step mask toggling does not corrupt FSDP state.** The mask is a
   forward-hook activation multiply; disabling it for a step changes only
   activations, never sharding or reduction. Verify: no shape/stride
   errors, no all-reduce desync, finite `grad_norm` on both masked and
   clean steps.
3. **Both forwards unmasked on a clean step.** Confirm the IS ratio
   `r ≈ 1` on clean steps (log per-token `log_p_current − log_p_old` ≈ 0
   spread) so the gradient really is the true dense GRPO gradient — not a
   half-masked one.
4. **PRF substep counter stays deterministic** across mixed masked/clean
   steps: skipping the mask on a clean step must not desync the seed
   schedule, so masked steps remain run-to-run reproducible. Verify:
   `mask_ratio` on masked steps stays ≈ `p`; a re-run reproduces.
5. **Grad clip, bf16, micro-batching, checkpointing all work unchanged**
   across a clean-step boundary (save/load over step 10 succeeds; no
   NaN/inf).
6. **Anchor stays allocation-free** (`anchor.enabled=false`): no ~3 GB
   clone, anchor counters stay 0, no spectral correction hook fires. This
   is what makes the clean-step config the FSDP-safest comm-eff variant.

### Minimal patch (for the `exp/<N>-<slug>` branch; do NOT implement here, just specify)
- Add `comm_eff.clean_cadence: int = 0` to `CommEffConfig`
  (`verl/workers/config/comm_eff.py`), validated `>= 0`; `0` = off so the
  disabled path stays a strict no-op.
- Thread the trainer `global_step` to the comm-eff state at the two points
  that stamp `mask_active` / `path_tag` (`verl/workers/engine_workers.py`:
  the `old_logprob` and `train` path-tag context managers). When
  `clean_cadence > 0 and (global_step % clean_cadence) == 0`, force
  `mask_active=False` on BOTH paths for that step.
- Leave `train_batch` (`verl/workers/engine/base.py`) otherwise untouched —
  `optimizer_step()` already runs every step. No anchor/spectral code is
  touched; no new optimizer; no clone.
- Add a numeric counter `comm_eff/clean_steps` so the log proves the clean
  steps fired at the right cadence.

## 7. Reference artifacts (cite these in the issue body)

- `research/runs/baseline/config.yaml` — dense baseline fixed config (source of the step-1 `grad_norm=0.36` reference)
- `research/runs/baseline/REPRODUCIBILITY.md` — baseline launcher SHA pin
- `research/runs/communication-baseline/verdict.md` — comm-eff baseline PASS
- `research/runs/communication-baseline/train.log` — comm-eff baseline log
- `research/runs/communication-baseline/REPRODUCIBILITY.md` — comm-eff baseline reproducibility manifest
- `research/notes/anchor-memory-cost.md` — anchor 3 GB clone explanation
- `research/notes/fast-circuit-vs-anchor-pass.md` — which forwards get masked; the anchor is unmasked, on a clone, and never steps the optimizer
- `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` — dense baseline launcher (KL loss enabled)
- `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh` — comm-eff baseline launcher (KL off by design)
- `verl/workers/config/comm_eff.py` — config schema (knob meanings; where `clean_cadence` would be added)
- `verl/workers/comm_eff/state.py` — runtime state + `MASK_ELIGIBLE_TAGS`
- `verl/workers/comm_eff/activation_mask.py` — the PRF mask hook (`h_tilde = h * mask`, no rescale)
- `verl/workers/comm_eff/anchor.py` — cloned-no-hook anchor module (no optimizer step)
- `verl/workers/comm_eff/spectral_filter.py` — EMA + SVD + Tikhonov + α-blend (α=1.0 ⇒ no-op)
- `verl/workers/engine/base.py` — `train_batch`: `zero_grad → anchor_refresh → forward_backward(masked) → spectral_correction → optimizer_step` (the clean-step lives here, as an unmasked whole-step)
- `verl/workers/engine_workers.py` — `compute_log_prob` / `update_actor` `mask_active` + `path_tag` stamps (the clean-step gate location)
- `verl/workers/engine/fsdp/transformer_impl.py` — `_comm_eff_mask_active` gating + the FSDP overrides for anchor/spectral

## 8. Issue format

**Title** (single line, no issue cross-references):
"Comm-eff paper-scale grad_norm explosion: peeling-ablation diagnosis + mandatory periodic clean-step optimizer-refresh test (KL stays off in the method evaluation)"

**Labels:** `kind:experiment`, `milestone:M2`. (The body declares
`code_change` per cell; the issue overall is `code_change:true` because
Test 4 is a mandatory code-change experiment.)

**Body sections (markdown):**
1. **Why this matters** — north-star linkage (§0): the explosion blocks
   stable→parity→savings→launcher; the clean step is the minimal,
   FSDP-safest candidate that keeps ~90% of savings.
2. **Observation** — the numeric step table, with the two-timescale split
   (step-1 magnitude vs ~50-step accumulation).
3. **Reference dense-baseline step-1 numbers** (`grad_norm=0.36 /
   entropy=0.37 / score=0.12` with KL=0.001) + the 0.087→0.789 100-step
   curve as the parity target.
4. **Full comm-eff baseline configuration** — the verified-PASS knob table.
5. **Candidate root causes A–H**, each with its diagnostic. Cause A is
   documented as a known late-step driver but is NOT proposed for
   toggle-back-on. Cause H is the accumulation-timescale cause the clean
   step treats.
6. **Test plan — Tests 1→5** (the peel). Mark `code_change` per test. State
   that **Test 4 is mandatory and runs independent of Tests 2–3** (gated
   only on Test 1). Include the per-cell tables, step counts, and verdicts.
   Carry the GPU-utilization discipline block.
7. **The periodic clean step** — the exact spec, the FSDP-no-errors
   acceptance checklist, and the minimal patch (§6). Be explicit that
   FSDP-no-errors is a hard pass/fail for Test 4 Cell B.
8. **Reference artifacts** (§7 file paths).
9. **Out of scope:** the parity run and the savings-measurement run
   (separate follow-ups) and any code beyond the minimal `clean_cadence`
   patch. This issue's deliverables are (a) a verdict on where the
   explosion lives, from the peel, and (b) a pass/fail on whether the
   periodic clean step stabilises pure masked GRPO with zero FSDP errors.

**Do NOT** reference any prior issue by number. **Do NOT** request
`status:approved` — the human operator decides whether to dispatch.

Print the full `gh issue create --title ... --body ... --label ...`
command to stdout for review.
