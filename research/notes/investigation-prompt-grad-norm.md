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

This is a **hybrid "peel-and-fix" issue**, not a pure diagnostic, and it is
sequenced around one governing principle:

> **Exhaust the lean, FSDP-clean, no-anchor/no-spectral path FIRST. Descend
> into the anchor/spectral machinery ONLY if that lean path fails to localise
> or stabilise the explosion.**

That splits the work into two phases:

**Phase A — the lean path (no anchor, no spectral; Tests 1–3).** Everything
here runs with `anchor.enabled=false, spectral.enabled=false` — a pure-config
path verified to be a strict no-op (§3.6), so none of the FSDP-fragile
clone / SVD / DTensor surfaces are even allocated. It does two things:
1. **Observe** whether the explosion survives peeling the method down to
   *pure masked GRPO* — mask straight to AdamW, no anchor, no spectral
   (Test 2). This is diagnosis-only: we are watching whether masked
   gradients alone diverge, **fixing nothing yet**.
2. **Validate the one minimal candidate fix** that lives entirely on this
   lean path: the **periodic clean (unmasked) optimizer step** (Test 3, the
   headline **mandatory** test) — every `N` steps let AdamW step on the
   *true* dense gradient, fixing the accumulated bias **in optimizer-state
   space**. It touches none of the anchor/spectral code, so it is the safest
   possible stabiliser and the cheapest to ship.

**Phase B — the anchor/spectral audit (Test 4; CONDITIONAL, last resort).**
Only if Phase A implicates the anchor/spectral machinery (mask-only is
*stable* yet the full method explodes, or the clean step cannot stabilise
masked GRPO) do we enable the anchor+spectral circuits and instrument them —
including an α=1.0 cell that makes the spectral blend an exact no-op while
the anchor still fires. Fully validating the lean path before this is a HARD
discipline: it keeps GPU-hr and FSDP risk off the table until the cheap
explanations are exhausted.

A second deviation — the biased no-rescale mask (§3.5 D-1) — has an obvious
candidate fix (an fp32 `1/(1-p)` rescale). It is **documented and made
verifiable here but NOT tested in this issue** (§10 out of scope); Phase A is
deliberately observation plus the optimizer-state fix only.

Consequence for labelling: **all of Phase A's observational cells and all of
Phase B are `code_change:false` by expectation** — they are reachable with
existing config knobs, and the per-circuit `.enabled` flags are *verified*
real toggles (§3.6). **Exactly one cell is unconditionally `code_change:true`**
— the periodic-clean-step (Test 3) — which needs a tiny `clean_cadence` knob
and must ride an `exp/<N>-<slug>` branch (base `vast-ai-workload`). **One
in-scope exception** can flip a lean-path cell (Test 1/Test 2) to
`code_change:true`: a *corrective* patch, if the test reveals that the
comm-eff scaffolding inadvertently regressed the FSDP backend on the
disabled/lean path. Masking is an activation multiply and should be
backend-transparent (§3.6), so any backend breakage on the lean path is our
own scaffolding's bug, and fixing it is in scope — a correction, not new
method functionality. Mark `code_change` per-cell in the body; the issue
overall carries `code_change:true` because it contains at least that one
mandatory code-change experiment.

### Operator constraints (load-bearing — read these first)

1. **The communication-efficient method's design is no-KL no-entropy.**
   `actor.use_kl_loss=False`, `algorithm.use_kl_in_reward=False`,
   `actor.entropy_coeff=0`. Every experiment that tests the COMM-EFF
   METHOD itself runs no-KL. The one exception is the gate test (Test 1,
   Cell A), which reproduces the dense baseline WITH KL purely as a
   reference / sanity point — it is not part of the method evaluation.

2. **Lean path first, anchor/spectral last.** The tests run as a sequence
   that exhausts the no-anchor/no-spectral path before touching the
   anchor/spectral machinery. Test 1 is the gate. Test 2 peels the method
   down to pure masked GRPO and *observes* whether it diverges (no fix).
   **Test 3 (the periodic clean step) is MANDATORY — it runs regardless of
   what Test 2 concludes** (gated only on Test 1 passing), because it is
   simultaneously the sharpest diagnostic (is the instability an
   optimizer-state problem?) and the leading candidate fix, and it stays on
   the lean, FSDP-clean path. Test 4 (the anchor/spectral integration audit,
   which now also carries the α=1.0 spectral-no-op peel) is the only
   *conditional* test — run it only if the lean path implicates the
   anchor/spectral machinery. Fully validating the lean path before Test 4 is
   a HARD discipline, not a preference: the FSDP backend should not need
   touching at all until then (§3.6).

3. **`comm_eff.enabled=true` for every method cell.** The only cells that
   disable the method are the gate's Cell B (regression check) and Test 4
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
than the full anchor+spectral apparatus while still cutting communication
by **~80% (≈5× vs dense at p=0.9** — 9 of every 10 steps stay masked at
~10% transmission, 1 is a full step). That is why it is mandatory, not
optional.

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

The mandatory periodic-clean-step test (Test 3) targets the **accumulation**
timescale; the mask-only peel (Test 2) localises the **step-1** contribution.

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

## 3.5 How the method is *designed* to work vs how this fork implements it — read before the causes

The method's design intent — distilled here so this prompt is fully
self-contained (no external source needed) — reframes every cause below.
Three facts matter.

**(1) The anchor is a periodic refresh-and-recompute circuit — NOT a
separately-trained model.** The intended design is explicit: **only the
fast circuit's weights `w^fast` are updated by AdamW.** The anchor circuit
periodically **pulls a slightly stale weight snapshot from the fast
circuit** (it has no independent optimizer state and does not train on its
own), runs a **real full unmasked fwd/bwd on the data**, and **ships the
clean gradient `G^anc` back** to the fast circuit, which folds it into the
EMA `M^anc` that defines the spectral basis. The anchor supplies **delayed
correction *geometry* only** — never an exact update. Its gradients are
stale by Δ ≈ 20–25 fast-circuit steps, so the basis refreshes roughly every
**K ∈ {10, 20}** steps when a new prior arrives (**K=50 collapses** — the
masked trajectory drifts beyond its correction horizon). The design
allocates a small `Z×Y` (Z=1) mesh slice to the anchor **purely so it runs
in parallel and the fast circuit never waits** — that slice is a *slaved,
periodically-refreshed copy* of the model, not an independently-training
one.

This fork implements the **same gradient semantics**: every
`anchor.cadence` steps it loads a `delay_K`-stale snapshot of the live
weights into a hookless **clone**, runs an unmasked fwd/bwd on the **same
training batch**, harvests the raw `G^anc` into `M^anchor`, takes **no
optimizer step**, and discards the clone (`anchor.py`,
`transformer_impl.py::_maybe_comm_eff_anchor_refresh`). The one real
difference from the design is that the fork runs this **synchronously
inline** (the fast path waits for the clone) instead of on a parallel mesh
slice — a *throughput* difference, not a correctness one. So the anchor
mechanism here is a **faithful port**; the verified-PASS smoke + the dry-run
use `cadence=delay_K=5` (below the design's K-range — conservative/fresher,
not a bug). The likelier culprit for the explosion is therefore the masking
bias (D-1), not the anchor.

**(2) The method is *designed* with approximately-unbiased masking; this
fork's masking is *biased* — a deliberate, acknowledged deviation.** The
design deliberately uses **random PRF masking** precisely because it
produces an approximately-unbiased gradient estimator that the spectral
filter can denoise, whereas top-K masking introduces a structured bias the
filter cannot remove. The whole spectral-correction guarantee rests on
this: it contracts zero-mean isotropic noise **but cannot remove a
structured bias**.

This fork's mask is `h_tilde = h * mask` with **no `1/(1-p)` rescale** —
and `activation_mask.py` says so explicitly: *"There is no `1/(1-p)`
forward rescale … we do not claim unbiasedness for the no-rescale port"*
(the rescale was dropped because `1/(1-0.95)=20×` amplification
destabilises bf16). So `E[h_tilde] = (1-p)·h`: at `p=0.9` the masked
residual stream is scaled to ~10% of its magnitude, and the **downstream
RMSNorm then re-normalises that shrunken residual, amplifying the surviving
~10% of entries by ~3×**, compounded across all 7 boundary layers. **This
injects exactly the structured bias that the spectral filter — by its own
stated property — cannot remove**, making it a prime suspect for the
paper-scale grad_norm explosion, and a *deviation from the intended
design*.

**(3) The method is *designed and validated* for masked *SFT / continual
pretraining* — NOT for RL. This project applies it *during* GRPO RL, a
regime it was never validated in.** All of the method's reported results are
masked **SFT**, and its convergence analysis assumes *fine-tuning*
gradients — "low effective rank" and a "slowly-drifting subspace" — and
explicitly predicts no benefit where the gradient subspace is
high-dimensional or shifts fast. Its only RL evidence is a *downstream
amenability check*: RL is run on the masked-SFT checkpoint **with masking
turned OFF**. So the established evidence is: *mask the SFT, then do RL
unmasked.* This project masks the RL itself. RL adds the PPO/GRPO importance
ratio `r = exp(log_p_current − log_p_old)` — which does not exist in SFT and
which **amplifies** the mask-induced variance/bias (cause B). A plausible
reason the explosion shows up at paper-scale RL but never in the masked-SFT
setting. This does not mean masked RL can't work — it is exactly the
project's research bet — but the diagnosis must treat the explosion as
**partly a regime-mismatch**, and it is another reason the regime-agnostic
clean-step (Test 3) is attractive: clean gradients re-anchor AdamW
regardless of SFT-vs-RL.

### Anchor / spectral implementation review — candidate deviations to verify (and possibly fix)

The spectral math (`spectral_filter.py`: EMA → full SVD → Tikhonov
`d_i = s_i/(s_i+τ)` → two-sided projection → α-blend, with `α=1` an exact
no-op) is a **faithful** implementation of the method's equations; the
staleness queue returns an exactly-`delay_K`-stale snapshot after warm-up;
the clone is correctly isolated from the optimizer/FSDP. The candidate
**deviations** to list in the issue as things to verify — and, if
confirmed, fix on the `exp/` branch — are:

- **D-1 (no-rescale bias — primary).** The mask is biased (above).
  *Verify (in this issue):* boundary-layer `mean(h_tilde)/mean(h) ≈ (1-p)`
  and the post-RMSNorm magnitude inflation. *Candidate fix (documented here,
  NOT implemented or tested in this issue — §10):* restore the `1/(1-p)`
  rescale **computed in fp32** (upcast surviving activations, rescale, cast
  back) so the estimator is unbiased without bf16 overflow — directly
  addressing the "spectral filter cannot remove a structured bias" property.
  Phase A deliberately *observes* this bias rather than fixing it; the
  optimizer-state fix (Test 3 clean step) is the only fix this issue
  validates.
- **D-2 (spectral filter on empty `M_anchor` halves the gradient).** With
  `seed_anchor_cache=false`, `M_anchor=0` until the first refresh ⇒
  SVD-of-zeros ⇒ Tikhonov `d=0` ⇒ `G_filt=0` ⇒ `G_proj = α·G_mask =
  0.5·G_mask`. The filter **silently halves** the gradient before any
  anchor prior exists, instead of passing it through. *Possible fix:* force
  `α=1` (masked-only) until the first anchor refresh has populated
  `M_anchor` — which is what the design intends (no correction before a
  prior arrives).
- **D-3 (staleness-queue memory at the design's K).** A **full-model**
  snapshot is pushed **every** step and the queue retains `delay_K+1` of
  them on GPU; at the design's `delay_K=20` that is ~21 × (full bf16 params)
  of snapshots in HBM, plus a `summon_full_params` all-gather every step.
  This is a large part of why the dry-run had to shrink batch knobs (cause
  G). *Possible fix:* snapshot only the target matrices, offload the queue
  to CPU, or only snapshot on steps that will be read.
- **D-4 (cadence below the design's analysed range).**
  `cadence=delay_K=5` is outside the design's `K∈{10,20}` envelope. Not a
  bug, but the dry-run sits in an un-analysed (fresher) regime; paper-scale
  runs of the *full* method should also test `K∈{10,20}`.

**Why Test 3 is the clean cut through all of this.** The mandatory
periodic-clean-step test (below) is the design's own explicitly-named
*"naive synchronous fix"* — periodically disable masking and run a full
unmasked forward–backward pass through the main pipeline. The design
rejects it **only for bandwidth** (it stalls the pipeline and pays full
activation transfer on the clean step), **not for optimization** — the
async anchor exists precisely to *approximate this same denoising benefit*
without the stall. So if periodic clean steps stabilise training while the
full async-anchor method explodes, the explosion is **not** an inherent
property of masked GRPO — it is in the **masking bias (D-1)** and/or the
**anchor/spectral implementation (D-2/D-3)**, and the simplest correct
method is the clean-step variant itself (no anchor, no spectral, none of
D-1…D-4's risk surfaces, ~5× PP savings at cadence=10 / p=0.9).

## 3.6 The peel is pure-config and backend-transparent — with one corrective caveat

Phase A needs `anchor.enabled=false, spectral.enabled=false` while
`mask.enabled=true`. This is reachable with **existing config knobs only — no
code change** — and the disable is a *verified* strict no-op, not a hopeful
one:

- **`anchor.enabled=false`** → the FSDP anchor-refresh override early-returns
  before building the clone
  (`transformer_impl.py::_maybe_comm_eff_anchor_refresh` returns when
  `anchor.enabled` is false), so **no ~3 GB clone is allocated, no staleness
  queue, no `summon_full_params` all-gather.**
- **`spectral.enabled=false`** → `CommEffState.build`
  (`verl/workers/comm_eff/state.py`) never constructs the `SpectralFilter`,
  so `state.spectral is None` and the grad-correction hook
  (`transformer_impl.py::_maybe_comm_eff_grad_correction`) is a strict no-op
  — no SVD/EMA buffers, no extra collective.
- **`mask.enabled=true` + `mask.p=0.9`** → only the `ActivationMasker` is
  built; the mask hook fires on the train forward (and the `old_logprob`
  recompute iff `mask_recompute=true`).

**Why the FSDP backend should not need touching until Phase B.** The mask is
an *in-graph activation multiply* (`h_tilde = h * mask`) at the logical
pipeline-boundary decoder blocks — it simulates the pipeline-parallel
activation reduction by zeroing a fraction of the residual stream. The FSDP
backend is **agnostic** to it: a masked activation produces a perfectly
ordinary gradient as far as parameter sharding, the gradient all-reduce, and
DTensor are concerned — nothing about the parameter layout, the reduction, or
the collective schedule changes. The **first** comm-eff component that
genuinely reaches into the backend is the *spectral correction*, which reads
and rewrites `p.grad` post-reduce and is the sole reason
`use_orig_params=true` is required; the *anchor* is next (it clones the FSDP
module and runs an isolated backward). So **no test up to — but not including
— the spectral correction and the anchor circuit should need to touch the
FSDP backend at all.** That is the expectation, and it is precisely *why* the
lean path is tested first: the fragile backend-interacting machinery is
provably absent, so any explosion there is unambiguously the mask's own
doing.

**The one corrective caveat.** That expectation can be violated only by our
*own* scaffolding. Bringing comm-eff in added `use_orig_params=true` (set
even on the disabled/lean path), the EXP-6 path-tag stamps, the hook
registration/removal points, and the `_maybe_comm_eff_*` insertions at the
top of and inside `train_batch`. Any of those could have *inadvertently*
perturbed the backend even while every circuit is disabled — "our development
might have nuked something." **Test 1 (the gate) is the designed check for
exactly this**, and Test 2 is the first cell where the mask hook actually
fires. **If Test 1 or Test 2 reveals such a regression** (an FSDP/DTensor
error, or a step-1 `grad_norm` that disagrees with the dense reference even
with comm-eff disabled), **a corrective patch to restore backend-cleanliness
is IN SCOPE** — and that, and only that, flips the affected lean-path cell to
`code_change:true` (on an `exp/<N>-<slug>` branch). It is a *correction*, not
new method functionality: the goal is to make the lean path as
backend-transparent as the math says it already is, *before* any conclusion
is drawn about the mask.

---

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
~10% magnitude, then re-amplified ~3× by the downstream RMSNorm). The
gradient computed through a biased forward is itself biased — independent
of the variance effect. Either or both can drive the step-1 `grad_norm`.
**This bias is a deviation from the intended design** (which requires
approximately-unbiased masking — §3.5 D-1), and the spectral filter
provably *cannot* remove a structured bias. Candidate fix: an fp32
`1/(1-p)` rescale (documented in §3.5 D-1; **NOT tested in this issue** —
§10 out of scope).
> **Diagnostic:** at step 1, log the per-token `(log_p_current −
> log_p_old)` histogram (expect non-zero spread even though
> `π_new == π_old`), AND the boundary-layer `mean(h_tilde)/mean(h)` ratio
> (expect ≈ `1-p`, confirming the un-rescaled bias).

### C. Spectral filter on empty `M_anchor` at startup — silently HALVES the gradient
`seed_anchor_cache=false` → `M_anchor = 0` for the substeps before the
first anchor refresh fires. The SVD of a zero matrix gives zero singular
values ⇒ Tikhonov weights `d_i = 0` ⇒ `G_filt = 0` ⇒ the blend collapses to
`G_proj = α·G_mask + (1−α)·0 = α·G_mask` — at the default α=0.5 the gradient
is **silently halved** for every substep before the first prior arrives,
instead of passing through unfiltered (§3.5 D-2). This *shrinks* rather than
explodes the early gradient, so it is a **fidelity bug, not the explosion
driver** — but it is not what the method prescribes (no correction should
apply before a prior exists). Fix: force α=1 until the first refresh.
> **Diagnostic:** at substep 1, log `||M_anchor||_fro` (expect 0) and
> `||G_proj − G_mask|| / ||G_mask||` per target (expect ≈ `1−α` = 0.5).

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

### H. AdamW optimizer-state poisoning by biased + high-variance masked gradients — **the accumulation-timescale cause; the one Test 3 treats**
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
weakly refreshes `v` — Test 3 measures whether that is *sufficient*.

The anchor+spectral apparatus attacks this same bias in **gradient space**
(project `G_mask` onto an anchor-derived basis). The periodic clean step
attacks it directly in **optimizer-state space** (let AdamW step on the
true gradient periodically), with none of the FSDP-fragile machinery
(causes C/D/E/F all vanish). Test 3 asks whether the cheap optimizer-state
fix makes the expensive gradient-space fix unnecessary.
> **Diagnostic:** log per-target AdamW `||m||` and `||v||` every step for
> the mask-only cell vs the mask-only+clean cell; expect `v` to inflate and
> `m` to drift in the no-clean cell and stay bounded in the clean cell.

---

## 5. Test plan — exhaust the lean path, then (conditionally) audit anchor/spectral (Tests 1 → 4)

Run them **in order**. The structure is two phases: **Phase A (Tests 1–3)**
runs the lean, no-anchor/no-spectral path — backend-transparent, FSDP-clean
(§3.6); **Phase B (Test 4)** enables and instruments the anchor/spectral
machinery and is reached **only** if Phase A implicates it. The conditional
structure is the fail-fast learning loop: each verdict prunes the search
before more GPU-hr is spent. The runner encodes the headline predicate per
test (e.g. step-1 `grad_norm` ratio vs the dense reference, or "does entropy
hold past step 40") and evaluates it inline. **Test 3 (the periodic clean
step) is mandatory and is NOT gated on Test 2's verdict** — only on Test 1
passing the gate.

### Execution & GPU-utilization discipline (per `.claude/plans/TEMPLATE.md` §"Vast.ai utilization discipline", HARD RULE)

- **One box for the whole Test 1 → 4 sequence.** Provision a single
  instance on the standard default tier (**4×H200 preferred, else 8×H100** —
  per `project.yaml.default_compute.gpu_filter_chain`) up front and chain
  all tests back-to-back (shared docker / verl checkout / dataset cache).
  Do NOT tear down and re-provision between tests.
- **Restore baseline batch knobs + provision headroom for method cells**
  (cause G): mini=64, wedge=36864, util=0.4 — the lean-path cells (Tests
  1–3) drop the ~3 GB anchor clone entirely, so they fit comfortably; do not
  re-introduce the smaller-batch confound. Only Test 4's full-method cells
  re-introduce the clone.
- **Keep every GPU busy.** Saturate whatever was provisioned; declare any
  legitimately idle window (e.g. a long eval between cells) in the plan's
  `## Notes for runner` so the stall-watchdog thresholds get loosened.
- **Fail-fast.** If Test 1 fails the gate, short-circuit — do not spend
  GPU-hr on Tests 2–4. If Phase A (Tests 2–3) fully explains and fixes the
  explosion on the lean path, Test 4 may be skipped (it is conditional). Tear
  down the instant the sequence resolves and metrics are rsynced to the
  laptop.

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
  Scaffolding is clean AND backend-transparent; the explosion is in the
  method itself. Proceed to the peel (Test 2).
- Cell A in tolerance, Cell B step-1 grad_norm > 1.0 → **scaffold regressed
  the backend** via one of {`use_orig_params=true`, schema additions, the
  path-tag stamps, the `_maybe_comm_eff_*` insertions in `train_batch`} even
  with comm-eff disabled (likely cause D — FSDP integration). Per §3.6 this is
  a **corrective-code-change trigger** (`code_change:true`, `exp/` branch),
  NOT a dead end: fix the backend regression the scaffolding introduced,
  re-run the gate until it is clean, *then* proceed. Do **not** run the method
  tests on a dirty gate — it would conflate the scaffold bug with the method.
- Cell A outside tolerance → the branch itself doesn't reproduce dense
  baseline. Diff this branch against the recorded baseline commit
  (`runs/baseline/REPRODUCIBILITY.md`) before going further.

### Test 2 — `peel-mask-only` (no anchor, no spectral) — `code_change:false` (corrective exception, §3.6)

**Runs only if Test 1 passes the gate.**

**Question — observation only, no fix:** does **pure masked GRPO** — mask
straight to AdamW, with the entire anchor+spectral apparatus *not even
allocated* — explode on its own? This peel removes causes C/D/E/F *and* the
~3 GB anchor clone (so cause G's memory pressure is gone too). Whatever
remains is the mask's own bias/variance (cause B) and its effect on the
optimizer (cause H). **This test fixes nothing** — it exists purely to *see*
whether masked gradients alone diverge; the candidate fix is Test 3.

**Config:** baseline batch knobs, no KL, no entropy,
`comm_eff.enabled=true`, `mask.enabled=true`, `mask.p=0.9`,
**`anchor.enabled=false`, `spectral.enabled=false`** — a pure-config path,
verified strict no-op (§3.6), **no code change**. `use_orig_params=true`
kept for config parity (this path does not depend on it — a further
FSDP-risk reduction; the mask is backend-transparent, §3.6).

| Cell | mask config | `code_change` | What it observes |
|---|---|---|---|
| **A — recompute=true (biased, as shipped)** | `mask_recompute=true`, no rescale | false | mask fires on both gradient-feeding forwards (the dry-run setting), now with no anchor/spectral confound — the biased estimator exactly as shipped. The headline observation. |
| **B — recompute=false** | `mask_recompute=false`, no rescale | false | mask fires only on the actor-train forward; `compute_log_prob` runs unmasked so `log_p_old` is clean — isolates the IS-mask-mismatch (the variance half of cause B) from the pure bias. |

**`code_change` for this test is `false` by expectation** (pure-config,
§3.6). **The one in-scope exception is corrective:** if Cell A surfaces an
FSDP/DTensor error or a scaffold-induced backend regression rather than a
clean `grad_norm` explosion, fixing that backend-cleanliness bug is in scope
and flips the affected cell to `code_change:true` (§3.6) — the mask should be
backend-transparent, so any backend breakage here is our own scaffolding's,
not the method's.

**10 trainer steps each** (step-1 magnitude + early trajectory).

**Verdict (diagnostic only — no fix is applied or expected here):**
- Cell A explodes → pure masking alone is unstable at paper scale; the
  anchor/spectral apparatus is **not** what drives the explosion (consistent
  with §3.5 — it is a faithful port). This is the expected outcome; it
  motivates the Test 3 fix and means Phase B (Test 4) is likely unnecessary.
- Cell A **>>** Cell B → the IS-mask-mismatch (cause B, variance half) is a
  meaningful contributor; note it for a follow-up (share PRF keys, or run
  with `mask_recompute=false`).
- Cell A ≈ Cell B (both explode) → the bias half dominates the variance half;
  the documented fp32-rescale candidate (§3.5 D-1) is the natural follow-up
  fix to try, alongside Test 3.
- **Cell A is *stable* (grad_norm bounded, entropy holds) → the explosion
  does NOT live in the mask; it must live in the anchor/spectral machinery.
  This is the one outcome that makes Phase B (Test 4) MANDATORY** — it is the
  trigger condition for the otherwise-conditional audit.

### Test 3 — `mask-only-plus-periodic-clean-step` (MANDATORY; the candidate fix) — `code_change:true`

**Runs whenever Test 1 passed the gate — independent of Test 2's verdict.**
This is the headline test and the leading candidate stabiliser, and it stays
entirely on the lean, FSDP-clean path (anchor/spectral still not allocated;
the only backend touch-point is the existing optimizer step, §3.6).

**Idea.** Run pure masked GRPO (exactly Test 2's config — no anchor, no
spectral), but **every `clean_cadence = 10` trainer steps, run that whole
step unmasked on the live module (mask off / `p=0`) and take the normal
`optimizer.step()`**, so AdamW's moments are periodically refreshed with
the *true* dense gradient. The other 9 of every 10 steps stay masked (so
~90% of the communication savings is retained). This is fundamentally
different from the anchor circuit, which runs unmasked on a **stale clone**
and **never** steps the optimizer — here the **live** optimizer state is
the thing being corrected.

This is, exactly, the method's own explicitly-named **"naive synchronous
fix"** (periodically disable masking and run a full unmasked fwd/bwd
through the main pipeline; §3.5). The design rejects it **only for
bandwidth**, never for optimization — the async anchor exists to
*approximate this very benefit* without the pipeline stall. That makes this
test simultaneously (a) the sharpest diagnostic — if it stabilises
training, the explosion is the masking bias (D-1) and/or the anchor/spectral
implementation (D-2/D-3), not masked GRPO itself — and (b) a candidate
*method* in its own right: dramatically simpler, FSDP-clean, ~5× PP
savings at cadence=10 / `p=0.9`.

**Config:** baseline batch knobs, no KL, no entropy,
`comm_eff.enabled=true`, `mask.enabled=true`, `mask.p=0.9`,
`anchor.enabled=false`, `spectral.enabled=false`, plus the new
`comm_eff.clean_cadence` knob.

| Cell | `comm_eff.clean_cadence` | What it tests |
|---|---|---|
| **A — no clean step (control)** | `0` | pure masked GRPO with no refresh — the same config as Test 2 Cell A (reuse that run if already on the box; do not re-run needlessly). Expected to drift toward collapse. |
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
  anchor/spectral path and stays FSDP-clean, and it **likely makes Phase B
  (Test 4) unnecessary**. Flag for the parity follow-up (does it reach
  dense-baseline reward?) and report the realised communication savings
  (fraction of masked steps × per-step mask saving).
- Cell B is no better than Cell A → the optimizer-state refresh at
  cadence 10 is insufficient (consistent with `v`'s ~1000-step memory; see
  cause H). Report the `||m||`/`||v||` diagnostic so the follow-up can
  decide between a shorter cadence and a gradient-space fix — and this, paired
  with a *stable* mask-only Test 2, is a second trigger for Phase B (Test 4).
- **Either way, the FSDP-no-errors checklist in §6 must pass for Cell B**;
  an FSDP error there is a STOP and is itself a finding.

### Test 4 — `fsdp-anchor-spectral-integration-audit` (Phase B; CONDITIONAL, diagnostic) — `code_change:false`

**Phase B. Runs only if Phase A implicated the anchor/spectral machinery** —
i.e. Test 2 (mask-only) was *stable* while the full method explodes, or
Test 3's clean step failed to stabilise masked GRPO. This is the LAST resort
by design: it is the only phase that enables the FSDP-fragile anchor+spectral
circuits and therefore the only one that can legitimately interact with the
backend (§3.6). If Phase A already explained and fixed the explosion, **skip
this test.**

**Question:** (i) is the **spectral blend itself** amplifying — α=0.5 vs the
α=1.0 exact no-op, anchor still firing (the folded-in α=1.0 peel)? and
(ii) are causes D and E (FSDP integration + anchor harvest correctness)
producing silent shape inconsistencies or silent no-ops?

**Config:** baseline batch knobs, no KL, no entropy, **full comm-eff**
(`comm_eff.enabled=true`, `mask.enabled=true`, `mask.p=0.9`,
`anchor.enabled=true` cadence=5/delay=5, `spectral.enabled=true`, τ=0.01,
β_anc=0.9), `use_orig_params=true`, with per-cause C/D/E/F diagnostic logging
on the method cells.

| Cell | config | What it produces |
|---|---|---|
| **A — disabled** | `comm_eff.enabled=false` | baseline grad shape + FSDP behaviour for comparison (same as Test 1 Cell B) |
| **B — full method, α=0.5, instrumented** | full comm-eff, `spectral.alpha=0.5` | the full method as shipped — reproduce the explosion on this box (control) with diagnostic logging per causes C/D/E/F (grad-type, rank-local vs full shapes, before/after-reduce values, target-match counts, `M_anchor` condition numbers across the first 10 anchor refreshes) |
| **C — full method, α=1.0 (spectral no-op), instrumented** | full comm-eff, `spectral.alpha=1.0` | the spectral blend is an exact no-op (`G_proj == G_mask`) while the anchor circuit STILL fires (harvest → EMA → SVD all run, but do not touch the grads). Isolates whether the spectral *projection* is what amplifies — the folded-in α=1.0 peel. |

**10 trainer steps each** (the step-1 logs and the first anchor refresh are
the structural deliverable; the early trajectory gives the α=0.5-vs-α=1.0
comparison).

**Verdict — numeric (B vs C) AND structural (per-cause logs):**
- **B explodes, C `<<` B at step 1** → the spectral projection is amplifying
  (causes C/F): the blend, not the mask, drives it. Point the follow-up fix
  at the spectral path (e.g. force α=1 until `M_anchor` is populated — D-2).
- **B ≈ C (both explode)** → the spectral blend is NOT the driver; the
  explosion is in the anchor harvest / FSDP integration (causes D/E) or
  upstream — read the structural logs to localise.
- Call out the **specific module path(s)** where each anomaly was logged so
  the follow-up fix can target them precisely:
  - target silently mis-typed (DTensor vs Tensor) → cause D(b);
  - anchor harvest's three target counts differ → cause E;
  - spectral hook sees `p.grad` before the FSDP all-reduce → cause D(a);
  - any `M_anchor` condition number > 1e6 → cause F.

---

## 6. The periodic clean step — exact spec, FSDP-safety checklist, minimal patch

This section is for Test 3 (the only mandatory `code_change:true` cell). Put
it in the issue body so the implementer who opens the `exp/<N>-<slug>` branch
has an exact, FSDP-safe target.

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

### FSDP-no-errors acceptance checklist (HARD — Test 3 Cell B must satisfy all)
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
`code_change` per cell; the issue overall is `code_change:true` because it
contains one mandatory code-change cell — the periodic clean step (Test 3).
Every other cell, including the entire Phase-B audit, is `code_change:false`
by expectation — the per-circuit `.enabled` flags are verified real toggles
(§3.6); the only exception is a *corrective* lean-path patch if Test 1/Test 2
reveals the scaffolding regressed the FSDP backend.)

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
5. **Design vs this fork** (§3.5) — three points: (1) the anchor is a
   *periodic refresh-and-recompute* circuit (by design: pulls stale
   weights from the fast circuit, runs unmasked fwd/bwd, returns `G^anc`;
   only `w^fast` is AdamW-updated), faithfully ported here (synchronously
   inline rather than on a parallel slice) — NOT a separately-trained
   model; (2) the mask is approximately-unbiased by design but **biased**
   here (no `1/(1-p)` rescale); (3) **the method is validated for masked
   SFT, never for RL** (its only RL evidence runs with masking OFF) — this
   project masks GRPO RL, a regime it was not validated in, and the PPO
   importance ratio amplifies the mask bias/variance. Include the
   **anchor/spectral implementation review D-1…D-4** (no-rescale bias;
   empty-`M_anchor` halving; staleness-queue memory; cadence below the
   design's K-range) as concrete, verifiable items with their candidate fixes
   — the things the operator wants on record as "possibly wrong, confirm
   and fix." Also carry **§3.6** — masking is an in-graph activation multiply,
   so the FSDP backend should be transparent to it all the way up to (not
   including) the spectral correction and anchor circuits; the lean-path
   disable is a verified pure-config no-op, and the only thing that could
   break backend-cleanliness there is our own scaffolding (the
   corrective-`code_change` caveat).
6. **Candidate root causes A–H**, each with its diagnostic. Cause A is
   documented as a known late-step driver but is NOT proposed for
   toggle-back-on. Cause H is the accumulation-timescale cause the clean
   step treats.
7. **Test plan — Tests 1→4** (Phase A: lean no-anchor/no-spectral path,
   Tests 1–3; Phase B: conditional anchor/spectral audit, Test 4). Mark
   `code_change` per test. State that **Test 3 (the clean step) is mandatory
   and runs independent of Test 2** (gated only on Test 1), and that **Test 4
   is the only conditional test** (reached only if the lean path implicates
   anchor/spectral). Note (§3.6) that Phase A is a verified pure-config,
   backend-transparent no-op path needing no code change — bar the corrective
   exception. Include the per-cell tables, step counts, and verdicts. Carry
   the GPU-utilization discipline block.
8. **The periodic clean step** — the exact spec, the FSDP-no-errors
   acceptance checklist, and the minimal patch (§6). Be explicit that
   FSDP-no-errors is a hard pass/fail for Test 3 Cell B.
9. **Reference artifacts** (§7 file paths).
10. **Out of scope:** the parity run and the savings-measurement run
   (separate follow-ups); the **fp32 `1/(1-p)` mask-rescale fix** (§3.5 D-1 —
   documented and made verifiable here, but not implemented or tested in this
   issue); and any code beyond the minimal `clean_cadence` patch (plus any
   corrective lean-path backend fix per §3.6, should Test 1/Test 2 require
   one). This issue's deliverables are (a) a verdict on where the explosion
   lives, from the lean peel (and, only if reached, the Phase-B audit), and
   (b) a pass/fail on whether the periodic clean step stabilises pure masked
   GRPO with zero FSDP errors.

**Do NOT** reference any prior issue by number. **Do NOT** request
`status:approved` — the human operator decides whether to dispatch.

Print the full `gh issue create --title ... --body ... --label ...`
command to stdout for review.
