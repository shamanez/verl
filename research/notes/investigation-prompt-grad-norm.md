# Investigation prompt — paste into a new session to draft the GitHub issue

> Paste the block below (everything between the long horizontal rules) into a fresh
> Claude Code session. It is self-contained: the new session does not need access
> to this conversation's history. Its output will be a single `gh issue create`
> command to print to stdout for review (it will NOT post the issue automatically).

---

You are picking up a research investigation on `shamanez/verl`, branch
`vast-ai-workload`. The research repo (issue queue) is
`shamanez/verl-compression-research` and is set as the local gh-default; the code
repo (PR target) is `shamanez/verl` with base `vast-ai-workload` (NEVER `main` —
`main` tracks upstream).

Your only job is to draft a GitHub issue. Do NOT write a plan file, do NOT modify
code, do NOT provision compute. Output a single `gh issue create` invocation
(title + body in markdown) — do not actually post it; print to stdout for human
review.

### Operator constraint (load-bearing)

**The KL anchor (`actor.use_kl_loss`) stays OFF for all tests in this investigation.**
That is a deliberate design choice for the communication-efficient method (it
isolates the compression effect from KL regularization). Do not propose toggling
KL back on as a fix. The hypothesis about KL removal causing late-step policy
collapse is still WORTH DOCUMENTING in the issue body as a known driver of the
late-step entropy collapse, but the test plan must remain at
`use_kl_loss=False / kl_loss_coef=0` everywhere.

---

## Observation

EXP-13 iter2 (58-step paper-scale comm-eff GRPO on Qwen2.5-1.5B-Instruct + GSM8K,
launcher at `research/runs/EXP-13/launch_iter2.sh`, log at
`research/runs/EXP-13/train_iter2.log`) shows grad-norm and policy-collapse
symptoms that the dense baseline (`research/runs/communication-baseline/` —
formerly EXP-9 iter2, the comm-eff smoke proof) at smoke scale did not.

The dense-pre-comm-eff baseline (`research/runs/baseline/` — the unmodified-code
EXP-3 reference) also did not show these symptoms.

Per-step trajectory from `train_iter2.log`:

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
- **Entropy collapse** 6.4 → 0.023 (policy becomes near-deterministic).
- **`ppo_kl` explosion** 0.04 → 1.4 (PPO trust region assumes < 0.1).
- **Step-1 grad_norm = 1134** — high even BEFORE any policy drift could
  matter. The variance amplification is happening from the first gradient.

`response_length/max` repeatedly hits the 16384 truncation cap (steps 16, 30,
33-35, 41, 43-44, 49, 51, 54) — consistent with policy collapse generating
repetitive output until truncation.

## Apples-to-apples knob diff (dense baseline → EXP-13 iter2)

These are the ONLY launcher differences. The investigation must discriminate
which are causal vs incidental at step 1 vs late steps.

| Knob | dense baseline (EXP-3) | EXP-13 iter2 | Why changed |
|---|---|---|---|
| `actor.use_kl_loss` | True | **False** | comm-eff design (operator decision, NOT under test) |
| `actor.kl_loss_coef` | 0.001 | n/a | follows `use_kl_loss=False` |
| `ppo_mini_batch_size` | 64 | **32** | inherited from comm-eff smoke (communication-baseline used this) |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | 36864 | **18432** | OOM fix in iter2 (anchor clone ~3 GB park-cost on 4×H200) |
| `LOG_PROB_MAX_TOKEN_LEN_PER_GPU` | 36864 | **18432** | OOM fix |
| `rollout.gpu_memory_utilization` | 0.4 | **0.3** | OOM fix |
| `PYTORCH_CUDA_ALLOC_CONF` | (default) | **expandable_segments:True** | OOM fix |
| `total_epochs` | 2 | **1** | inherited from comm-eff smoke |
| `val_before_train` | False (default) | **True** | EXP-13 added |
| `actor.fsdp_config.use_orig_params` | False (default) | **True** | EXP-7 finding (DTensor full_tensor) |
| `comm_eff.enabled` | False | **True** | the method |
| `comm_eff.mask.{p, mask_recompute}` | n/a | **0.9, true** | the method |
| `comm_eff.anchor.{cadence, delay_K}` | n/a | **5, 5** | the method |
| `comm_eff.spectral.{alpha, tau, β_anc}` | n/a | **0.5, 0.01, 0.9** | the method |

**Critical confound**: the OOM-mitigation knobs (halved wedge, halved mini-batch,
dropped vLLM mem-util) were only needed because of the comm-eff anchor clone's
~3 GB park-cost on 4×H200. The dense baseline never needed these compromises. So
these knobs are themselves a confound for the grad-norm comparison. A clean
apples-to-apples test should restore them to baseline values, which requires
more GPU headroom (8×H100/H200) rather than fewer-knob compromises.

## Candidate root causes — enumerate all of these in the issue body

### 1. KL anchor removal (known driver of LATE-step collapse, but NOT under test)
`actor.use_kl_loss=False` removes the `0.001·KL(actor || ref)` term. At step 1
this is irrelevant (`KL ≈ 0`), but cumulatively over 58 steps it removes the
soft anchor that holds π close to π_ref. The entropy collapse + `ppo_kl`
explosion are textbook symptoms. **By operator decision the KL toggle stays
off** — document this as the late-step driver but DO NOT propose toggling it
on. The investigation must explain whether the comm-eff method itself
(independent of KL) is well-conditioned at paper-scale.

### 2. Importance-sampling variance under independent PRF masks (the step-1 cause candidate)
EXP-9 introduced `mask_recompute=true`: both the actor-train forward AND the
`compute_log_prob` (old_logprob) forward route through the masked path. But the
PRF key (substep counter × layer index × base seed) **differs between the two
paths by design** — `compute_log_prob` fires once per train step;
PPO-inner-loop fires `N×E` times per train step.

So mask realizations DIFFER between `log_p_current` and `log_p_old`. The PPO
importance ratio
```
r = exp(log_p_current − log_p_old)
```
is therefore the ratio of two stochastic estimates under different masks, even
when the underlying actor weights are identical. **At step 1, `r ≠ 1` even
though `π_new == π_old`** — purely from the mask-realization difference.

Expected variance of `log(r)` at step 1 scales like `2 · Var(masked logit)` per
token. This is the most likely explanation for the step-1 grad_norm of 1134
being far above any pure-dense reference.

**Diagnostic**: at step 1, log per-token `(log_p_current − log_p_old)` histogram;
expect non-zero spread even though `π_new == π_old`.

### 3. Mini-batch and wedge variance amplification (the smaller-batch class)
`ppo_mini_batch_size: 64 → 32` and `PPO_MAX_TOKEN_LEN_PER_GPU: 36864 → 18432`
each ~2× the per-substep gradient variance. Stacked ~4×. Largely independent of
comm-eff, but inherited by EXP-13 from the smoke launcher and locked in by the
OOM mitigation.

**Diagnostic**: run T2 below at baseline batch values on more GPUs.

### 4. Spectral filter on empty `M_anchor` at startup
`seed_anchor_cache=false` → `M_anchor` starts at zero. First anchor refresh
fires at PPO substep 5 (cadence=5). For substeps 1-4 the spectral filter
operates on `M_anchor = 0`; its SVD basis is degenerate (`U, S, V` all zero or
NaN-suppressed to identity).

The blended gradient `G_proj = α·G_mask + (1−α)·G_filt` at `α=0.5` in this
regime is either:
- (a) silently identity-passing `G_mask` through (no correction applied), or
- (b) producing NaN-quietly-replaced-by-zero outputs that bias the actor.

Either way, the "filter" is non-functional for the first ~5 substeps and the
gradient that's applied to the optimizer is not what the method's theory
prescribes.

**Diagnostic**: at substep 1, log `||M_anchor||_fro`, log `||G_proj − G_mask||`
per target; both should be exactly 0 if the filter is identity-passing, or
non-zero if `M_anchor` is somehow being used.

### 5. FSDP integration of the spectral correction hook (the implementation-bug class)
EXP-7 placed the spectral correction at
`after_actor_backward__before_optimizer_step` with `use_orig_params=True` so
`p.grad` surfaces as a 2-D DTensor `full_tensor`. EXP-12 isolated the anchor
backward onto a hookless clone so FSDP1's `_post_backward_hook` doesn't fire on
the anchor pass.

Two non-obvious risk surfaces remain:
- **(a)** On the LIVE module, the spectral hook reads `p.grad` and writes
  `p.grad` back. If the hook runs BEFORE FSDP1's gradient all-reduce completes,
  it operates on a shard not the reduced tensor; the projection is then
  per-rank inconsistent and the optimizer step sees a non-deterministic
  gradient.
- **(b)** `use_orig_params=True` surfaces some params as DTensor and others as
  ordinary tensors depending on FSDP wrap boundaries; if `target_substr`
  matching catches a mix, the per-target `G_proj` math is shape-inconsistent
  across ranks.

**Diagnostic**: at step 1, log each target's grad type (`DTensor` vs `Tensor`),
its rank-local shape vs full-tensor shape, AND the value of `p.grad` seen by
the spectral hook immediately before vs immediately after the FSDP reduce.

### 6. Anchor clone gradient harvest correctness (the silent-no-op class)
The anchor clone is cached on `self._anchor_module_cache`. Each refresh: load
K-stale weights → forward → `loss.backward()` → harvest `p.grad` into the EMA.

Risks:
- If `clone.named_parameters()` returns DTensor-wrapped tensors but the
  `target_substr` matching uses live-module names, the matching could miss
  targets (silent no-op: `M_anchor` stays empty → spectral filter is broken
  for those targets → mask noise passes through unfiltered).
- If the clone's grad is harvested while the live model's grad is still being
  computed (parallel autograd), the EMA could mix gradients from different
  steps.

**Diagnostic**: on the first anchor refresh, log
`{targets_in_substr_set, targets_matched_in_clone, targets_with_nonzero_grad}`;
expect all three counts equal.

### 7. Mask × spectral interaction at α=0.5 (the conditioning class)
At α=0.5, `p.grad = 0.5 · G_mask + 0.5 · G_filt`. `G_mask` is the **masked**
gradient (high variance). `G_filt` is the projection of `G_mask` onto
`M_anchor`'s principal subspace.

If `M_anchor` is poorly conditioned (early steps, narrow basis, or rank
deficiency from the lowrank SVD path), `G_filt` could **amplify rather than
damp** `G_mask` in some directions. The blend then has both the raw mask
variance AND the amplified projection — worse than either alone.

**Diagnostic**: log per-target singular values of `M_anchor` across the first
10 anchor refreshes; flag any target whose condition number is > 1e6.

### 8. Total-epochs / dataloader-exhaustion interaction
EXP-13 used `total_epochs=1`; dataloader exhausted at step 58
(7473/128 = 58.4 batches per epoch). With 1 epoch the model sees each prompt
EXACTLY once during training; less data diversity → faster policy-mode-collapse
that no-KL allows.

**Diagnostic**: `total_epochs=2` (matching the baseline). Observe whether
entropy collapse pace slows.

### 9. Memory-mitigation knobs as confounds (the host-side class)
The OOM fix (halved wedge + halved mini-batch + vLLM mem-util 0.3) was forced
by the anchor clone's ~3 GB park-cost on 4×H200. The ONLY reason these knobs
differ from baseline is the comm-eff overhead. If we had run on 8×H100/H200
(more sharding headroom), these knobs would not have changed and EXP-13 would
have run at baseline batch sizes.

Any test on 4×H200 conflates "comm-eff method effect" with "smaller-batch
variance effect"; a true apples-to-apples needs more GPUs.

## Minimal discriminating test plan — include this section in the issue

KL stays OFF in all four tests (`actor.use_kl_loss=False`, `kl_loss_coef=0`,
`algorithm.use_kl_in_reward=False`, `entropy_coeff=0`). Each test is ≤30 steps,
≤$5 of compute.

### T1: Sanity-zero — comm-eff disabled at EXP-13 batch knobs
- `comm_eff.enabled=false` on the current `vast-ai-workload` code
- Everything else at EXP-13 iter2 values (mini=32, wedge=18432, no KL, no
  entropy, val_before_train=True, test_freq=25, total_epochs=1)
- 4×H200 (or 4×H100)

**Purpose**: regression check that the comm-eff scaffolding hasn't broken the
no-comm-eff code path. Also gives a *no-comm-eff + no-KL + smaller-batch*
reference grad_norm to compare against EXP-13's *with-comm-eff* grad_norm.

**Expected**: grad_norm trajectory matches the original
`comm_eff.enabled=false` parity claim from EXP-4. If it does NOT match, the
comm-eff code scaffolding silently regressed the no-comm-eff path. **This is the
gate test.**

### T2: comm-eff at BASELINE batch knobs — isolates the smaller-batch confound
- Full M90+AP knobs (`p=0.9`, `α=0.5`, `τ=0.01`, `cadence=5`, `mask_recompute=true`)
- `use_kl_loss=False` (stays off)
- `ppo_mini_batch_size=64`, `PPO_MAX_TOKEN_LEN_PER_GPU=36864`,
  `gpu_memory_utilization=0.4`, `total_epochs=2`
- **Provision 8×H100/H200** (more headroom for the anchor clone)

**Purpose**: removes the smaller-batch variance confound. Tests whether the
step-1 grad_norm of 1134 is reproducible at full baseline batch knobs — if yes,
the source is the comm-eff mask itself (hypothesis 2); if no, the smaller-batch
knobs were a meaningful contributor.

### T3: comm-eff ablation — mask_recompute on/off
- Two cells, both at EXP-13 iter2 knobs (4×H200, no KL, smaller batch)
- Cell A: `mask_recompute=true` (current)
- Cell B: `mask_recompute=false` (only the actor train forward is masked;
  `compute_log_prob` runs unmasked)

**Purpose**: hypothesis 2 directly. If cell B's step-1 grad_norm is much lower
than cell A's, the IS-variance-under-independent-masks effect is real and
significant.

### T4: comm-eff ablation — α and seed_anchor_cache
- Three cells, all at EXP-13 iter2 knobs except:
- Cell A: `α=1.0` (spectral correction is exact no-op)
- Cell B: `α=0.5` (current EXP-13)
- Cell C: `seed_anchor_cache=true` (M_anchor pre-seeded, hypothesis 4 mitigated)

**Purpose**: hypothesis 4 and 7 directly. If cell A's grad_norm is much lower
than cell B's, the spectral correction is contributing significantly (and
possibly amplifying noise). If cell C is much smoother than cell B at step 1-5,
the empty-M_anchor pathology is real.

## Reference artifacts (cite these in the issue body)

- `research/runs/baseline/config.yaml` — dense baseline (EXP-3) fixed config
- `research/runs/baseline/REPRODUCIBILITY.md` — baseline launcher pin (SHA)
- `research/runs/baseline/launch.sh` — baseline in-container launcher
- `research/runs/communication-baseline/launch_iter2.sh` — comm-eff smoke
  PASS launcher (formerly EXP-9 iter2, now the reference for the
  communication-efficient method)
- `research/runs/communication-baseline/verdict-iter2.md` — verdict for
  comm-baseline (PASS at 20-step smoke scale)
- `research/runs/EXP-13/launch_iter2.sh` — current paper-scale comm-eff launcher
- `research/runs/EXP-13/train_iter2.log` — 58-step grad_norm + actor stats
- `research/runs/EXP-13/verdict.md` — current verdict (PASS, at-risk given
  these symptoms)
- `research/notes/anchor-memory-cost.md` — anchor 3 GB clone explanation
- `research/notes/fast-circuit-vs-anchor-pass.md` — which forwards get masked
- `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh` — committed
  launcher (the contract)
- `verl/workers/config/comm_eff.py` — config schema (knob meanings)
- `verl/workers/comm_eff/state.py` — runtime state + `MASK_ELIGIBLE_TAGS`
- `verl/workers/comm_eff/activation_mask.py` — the PRF mask hook
- `verl/workers/comm_eff/anchor.py` — cloned-no-hook anchor module
- `verl/workers/comm_eff/spectral_filter.py` — EMA + SVD + Tikhonov + α-blend
- `verl/workers/engine_workers.py` — `compute_log_prob` `mask_active` stamp
- `verl/workers/engine/fsdp/transformer_impl.py` — `_comm_eff_mask_active` gating

## Issue format

**Title**: "Investigate: EXP-13 grad_norm explosion vs baseline — IS variance,
spectral conditioning, FSDP/anchor integration audit (KL stays off)"

**Labels**: `kind:investigation`, `milestone:M2`

**Body sections (markdown)**:
1. Observation (with the numeric step-by-step grad_norm/ppo_kl/entropy table)
2. Apples-to-apples knob diff (the table above)
3. Candidate root causes (the 9 hypotheses above, each with diagnostic
   suggested). **Section 1 is documented as a known late-step driver but is
   NOT proposed for toggle-back-on.**
4. Discriminating test plan (T1-T4). All tests have `use_kl_loss=False`.
5. Reference artifacts (file paths)
6. Out-of-scope-for-this-issue: the fix itself; this issue's deliverable is a
   verdict on which hypothesis is true and a plan for the fix to follow

**Do NOT** specify a fix. **Do NOT** request `status:approved`. The issue is
investigation-only; the human operator decides whether to dispatch experiments
based on the test plan.

Print the full `gh issue create --title ... --body ...` command to stdout for
review.
