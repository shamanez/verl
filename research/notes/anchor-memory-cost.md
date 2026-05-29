# Why the anchor circuit needs a separate ~3 GB of GPU memory

> Author note: this is a conceptual / mental-model doc, not an experimental
> finding. The numbers come from EXP-12 (which fixed the FSDP-collision bug by
> introducing the cloned-no-hook anchor module) and EXP-13 (which surfaced the
> memory pressure at paper-scale settings).

## The short answer

The anchor circuit needs **its own materialized copy of the model parameters**
— not its own *optimizer state*, not its own *training rollouts*, just a
second instance of the weights — because the anchor's job is to produce a
gradient `G_anchor` with a backward pass that is **autograd-isolated from the
live training graph**.

For Qwen2.5-1.5B at bf16:

```
1.54 billion params × 2 bytes/param ≈ 3.08 GB
```

That's the floor. It's not "anchor overhead" in the abstract — it's literally
the cost of holding the model twice.

## Why "in-place" doesn't work — the three things we have to escape

The live training model is wrapped in three layers of automatic behavior, all
of which fire on `loss.backward()`. The anchor pass needs *none* of them:

### 1. The activation-mask hook fires on the live module

The PRF Bernoulli mask is registered as a **forward hook** on pipeline-boundary
decoder blocks. Any forward through the live module — `loss.backward()`'s
forward replay during gradient computation under gradient_checkpointing
included — re-fires the mask hook. The whole point of the anchor is to be the
**unmasked, gold-signal** gradient. If the anchor's forward routes through
mask-hooked blocks, GUARD 5 (`comm_eff/anchor_mask_applications == 0`) fails
by definition.

You can try to disable the hook with a flag (`state.mask_active = False` inside
the anchor's forward call). EXP-8 actually tried that path. It failed for a
*different* reason — see #2.

### 2. FSDP1's `_post_backward_hook` is single-shot per param per backward

FSDP1 wraps each parameter in a flat-param shard and registers a
`_post_backward_hook` that:

- Triggers gradient reduction across ranks
- Zeros / consumes `flat_param._saved_grad_shard`
- Marks the shard as "this backward done"

When EXP-8 ran a second backward through the same FSDP-wrapped module (the
anchor pass, immediately after the training backward), the hook re-fired with
`flat_param._saved_grad_shard == None` (because the training backward had
consumed it). Result: `AttributeError: 'NoneType' object has no attribute
'shape'` in `_reduce_grad`.

This is not a bug in our code — it's FSDP1 (correctly) refusing to do two
backwards through one wrap.

### 3. The spectral-correction hook would corrupt `G_anchor`

The grad-correction hook fires at
`after_actor_backward__before_optimizer_step`. Its formula:

```
G_proj = α·G_mask + (1−α)·G_filt   (where G_filt uses M_anchor to define the filter)
```

If the same hook fires on the anchor's own gradient, the anchor — which is
supposed to *define* the filter via the EMA `M_anchor ← β·M_anchor + (1−β)·G_anchor`
— would be feeding itself projected gradients. That's a feedback loop where
the projection redefines the basis that the projection used. GUARD 6
(`anchor_grad_corrected == 0`) catches this.

## The fix: a hookless, FSDP-free, full-precision clone

EXP-12 introduced `build_anchor_module(live_module)`, which returns a
**fresh `nn.Module`** that:

- Holds **its own bf16 parameters** (the K-stale snapshot of the live weights)
- Has **no FSDP wrap** (parameters are full, not sharded)
- Has **no registered hooks** (no mask, no spectral correction, no FSDP
  post-backward)
- Is **excluded from the optimizer's param groups** — `optimizer.step()` will
  not touch it

A backward through this clone is mathematically equivalent to "what would
the gradient be if there were no compression and no FSDP." Exactly what the
anchor is meant to produce.

## The memory cost in detail

### Static cost (parked between refreshes)

After EXP-12 iter04, the clone is **cached on `self._anchor_module_cache`** —
we don't rebuild it every refresh, which would have allocated/freed 3 GB per
anchor step (and tripped vLLM v1's `sleep_replicas` `freed_bytes` assertion).
So between refreshes:

```
clone parameters (bf16):  ~3.08 GB  ← parked
clone gradient buffers:    0        ← zeroed after each refresh
clone activations:         0        ← freed after each backward
```

### Dynamic cost (during refresh)

When the anchor fires (every `cadence` PPO substeps), peak memory adds:

| Component | Size |
|---|---|
| `clone.param.grad` (populated by backward) | ~3 GB (param-shape gradient tensor) |
| Clone forward activations (with grad-checkpointing) | ~5–10 GB at 16K context |
| **Anchor refresh peak** | ~11–16 GB on top of the live model + vLLM |

After the refresh:
- `clone.zero_grad()` releases the gradient memory
- Forward activations are freed by autograd
- `torch.cuda.empty_cache()` returns the freed pages to the allocator
- Only the 3 GB parked params remain until the next refresh

### Why this bit EXP-13 but not the smoke (communication-baseline)

| | communication-baseline (smoke) | EXP-13 paper-scale |
|---|---|---|
| Response length | 256 | 16384 |
| Train batch | 8 | 128 |
| Rollouts per prompt | 2 | 8 |
| Sequences per step | 16 | 1024 |
| Live actor activations | ~5 GB | ~120 GB |
| Anchor clone parked | 3 GB | 3 GB |
| Headroom on 140 GB H200 | comfortable | razor-thin |

The 3 GB clone cost is invariant in absolute terms — but it's a 30% chunk of
the smoke's footprint and a 2% chunk of the paper-scale footprint. The
*relative* cost is small at paper-scale, but the *absolute* spare-headroom
budget at paper-scale is so tight that 3 GB is the difference between fit and
OOM.

EXP-13's fix wasn't to shrink the anchor — it was to lower
`PPO_MAX_TOKEN_LEN_PER_GPU` (the dynamic-batch wedge that decides how many
sequences go through actor forward in one micro-batch) from 36 864 to
18 432 tokens, dropping vLLM's `gpu_memory_utilization` from 0.4 to 0.3,
and enabling `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to combat
fragmentation. The anchor itself was untouched.

## Could the clone be smaller? Options and tradeoffs

| Option | Memory savings | Why we don't (yet) |
|---|---|---|
| FSDP-wrap the clone too | ~75% (sharded across 4 ranks → ~750 MB each) | Brings back the `_post_backward_hook` collision that EXP-12 fixed. Would need a separate FSDP wrap instance with no cross-talk to the live wrap — possible but non-trivial to get correct |
| `ema_device=cpu` + clone-on-CPU between refreshes | Park-cost goes to host memory | Anchor refresh now has to H2D copy 3 GB before forward and D2H after — slow (`cadence=5` means this fires often) |
| Clone only the targeted layers (q/k/v/o_proj of `model.layers.0`) | ~99% (4 matrices × 9 MB ≈ 36 MB) | Can't run a full-model GRPO loss on a 4-matrix sub-model — the loss needs the lm_head, which needs every layer's forward |
| Quantize the clone to int8 / 4-bit | ~75–87% | Quantization noise contaminates the gold-signal gradient — defeats the anchor's purpose. The anchor is supposed to be MORE precise than the masked training path, not less |
| Stash the clone params on CPU, demand-page to GPU during forward | Park cost goes to host | Demand-page latency would dominate the refresh — same problem as `ema_device=cpu` but worse because we'd be paging in inner-loop, not just at the EMA-update step |

The current design picks **fidelity over memory**. The anchor's whole reason
to exist is to be the unbiased reference; the moment we let it shrink in a
way that distorts its gradient, we lose the calibration signal that the
spectral filter is built around.

## What the runtime telemetry confirms

From EXP-13 iter2 at step 11 (after 8 anchor refreshes):

```
comm_eff/anchor_backwards:           8       ✓ (cadence=5 × 2 substeps/step × 11 / 5 ≈ 4 expected; we see 8 because the refresh fires per-PPO-inner-batch in this regime)
comm_eff/anchor_mask_applications:   0       ✓ GUARD 5 — clone has no mask hook
comm_eff/anchor_grad_corrected:      0       ✓ GUARD 6 — clone's grads aren't routed through the spectral hook
comm_eff/anchor_rollouts_generated:  0       ✓ no extra rollouts
comm_eff/anchor_rewards_recomputed:  0       ✓ no extra reward calls
comm_eff/anchor_optimizer_steps:     0       ✓ no optimizer.step on the clone
anchor_backward_isolation_mode:      clone   ✓ EXP-12 isolation mode active
||dM_anchor||_mean (step 40 refresh): 0.144  ✓ EMA evolving non-trivially
```

All six guards are mathematically tight (== 0, not just close to zero), which
is only possible because the clone is a *completely separate computational
graph* — there's no hook to "almost fire" but be skipped by a runtime flag.
The isolation is structural, not flag-gated.

## Mental model summary

The anchor takes ~3 GB *because it has to be a second model*, not just a
second backward. The reason it has to be a second model is that the live
model is entangled with three different state machines (mask hooks, FSDP
shard-state, spectral grad hooks) that would all fire incorrectly on a
"second backward" through it. Isolating the autograd graph is what costs
memory; everything else (no extra rollouts, no reward recompute, no optimizer
step) is structurally free.
