# Fast circuit vs anchor pass — what's masked, what isn't, and what gets recomputed

> Goal: confirm your mental model that **both** gradient-feeding forwards in
> the fast (masked) circuit are masked, while the anchor pass **recomputes
> everything from scratch on an unmasked clone**.

## TL;DR — your understanding is correct, with one clarification

You said:

> "All forward passes in fast circuit get masked, then nothing in anchor
> gets masked and we re-compute everything."

This is almost exactly right. The clarification is just about *which* fast
forwards we count — there are several forward passes per trainer step in
GRPO, but not all of them are part of the "fast circuit" we're compressing.
Only **the two gradient-feeding actor forwards** belong to it, and only those
two get masked. The rollout generation, the validation forward, and the
(absent in this setup) reference-policy forward are deliberately left
unmasked, because they don't feed the training gradient.

The anchor pass, by contrast, is a full fresh forward + backward on a
**hookless clone of the model**, end-to-end unmasked.

## The five forward passes in a GRPO trainer step

To make this concrete, here are all the forwards that happen during one
`global_step` in this code path:

| # | Forward pass | Where it runs | What it produces | Masked? | Why / Why not |
|---|---|---|---|---|---|
| 1 | **Rollout generation** (`vllm_rollout`) | vLLM engine | `response` tokens for each prompt | **NO** | Rollouts are the *data*, not part of the gradient. Masking the rollout would mean the agent is sampling from a different distribution than it's being trained on. `mask_applications/rollout == 0` |
| 2 | **Old-logprob recompute** (`compute_log_prob`) | FSDP train engine | `old_log_prob` for the importance ratio `r = exp(log_prob_current − old_log_prob)` | **YES** (when `mask_recompute=true`) | This forward runs on the actor — same pipeline-boundary bandwidth pressure as #3. Both feed the PPO importance ratio that defines the gradient. The method stamps `mask_active=True` around `compute_log_prob` so the mask hook accepts `path_tag="old_logprob"` |
| 3 | **Actor train forward** (PPO inner loop) | FSDP train engine | `log_prob_current` for the policy gradient; autograd records the graph that `loss.backward()` walks | **YES** | The headline compression target. The mask fires on every pipeline-boundary block. `mask_applications/train > 0` |
| 4 | **Reference-policy logprob** (`ref_log_prob`) | Ref worker (if spawned) | `ref_log_prob` for the KL-vs-ref term | n/a — ref worker not spawned | Disabled in this run: `use_kl_loss=False` AND `use_kl_in_reward=False` → `need_reference_policy()` returns False → ref worker never instantiated. `mask_applications/ref_logprob == 0` |
| 5 | **Validation forward** | FSDP train engine in eval mode | `val/test_score/mean@1` for the curve | **NO** | Validation reads the model's *current state* for evaluation. Masking it would mean validation is reporting performance of a different model than the one being trained. `mask_applications/val == 0` |

### Why only #2 and #3 belong to the "fast circuit"

The fast circuit is defined by **the gradient-feeding bandwidth pressure
between pipeline shards**. In a pipeline-parallel deployment, the
boundary-block activations are what travel across the pipeline. We mask
exactly those activations, *exactly* on the forward passes whose autograd
record will be replayed by `loss.backward()`:

- **#3 (actor train)** — its autograd graph is the one `loss.backward()`
  walks, producing the policy gradient.
- **#2 (old-logprob recompute)** — its output `old_log_prob` is a *constant*
  with respect to `loss.backward()` (it's detached), BUT it enters the
  importance ratio `r = exp(log_prob_current − old_log_prob)`. Bias in
  `old_log_prob` shifts the ratio, and the ratio is what the gradient is
  computed *with respect to*. So while #2 is not differentiated, the
  *distribution of its values* directly affects the gradient `log_prob_current`
  is multiplied by. If we mask #3 but not #2, the two values are computed
  with structurally different forwards — that's a train-inference mismatch
  inside the actor itself.

`mask_recompute=true` extends masking to #2 as well, so **both
gradient-feeding forwards see the same pipeline-boundary bandwidth pattern**.

#1, #4, #5 are deliberately untouched because they measure or sample, they
don't propagate gradient.

## The PRF Bernoulli mask, in code-truth form

```python
# verl/workers/comm_eff/activation_mask.py
def forward_hook(module, args, output):
    state = _get_active_comm_eff_state()
    if not state.mask_active:
        return output                              # GUARD 5: anchor path skipped here
    if state.path_tag not in mask_eligible_tags(state):
        return output                              # GUARD: confinement (rollout/ref/val/etc)
    # per-(token, dim) mask keyed on each token's STABLE identity (sample_id,
    # position_id) + layer + global_step — packing-invariant, no substep/positional key.
    mask = prf_token_mask(masker.sample_ids, masker.position_ids,
                          layer_idx=layer_idx, global_step=masker.global_step,
                          base_seed=masker.base_seed, hidden_size=output.shape[-1],
                          p=masker.p)
    state.mask_applications += 1
    state.mask_applications_per_path[state.path_tag] += 1
    return output * mask                           # h_tilde = h * mask (rescale knob → *1/(1-p), default off)
```

Three gates have to be true for the mask to fire:

1. **Master switch**: `comm_eff.enabled=true` AND `comm_eff.mask.enabled=true`
2. **`mask_active=True`**: set in `engine_workers.py`'s `_comm_eff_path("train")`
   context manager (always true on actor backward) and, when
   `mask_recompute=true`, also inside `_comm_eff_path("old_logprob")` for
   `compute_log_prob`. The anchor's `_maybe_comm_eff_anchor_refresh` runs
   with `mask_active=False`.
3. **`path_tag in MASK_ELIGIBLE_TAGS`**:
   - `MASK_ELIGIBLE_TAGS = frozenset({"train"})` by default
   - widens to `frozenset({"train", "old_logprob"})` iff `mask_recompute=true`
   - **`None` (the anchor's path tag) is never in the set** — GUARD 5 by
     construction, not by runtime flag

The `frozenset` is the structural guarantee: anchor's `path_tag=None` is not
a member of any set we ever build, so the hook returns `output` unchanged on
the anchor's forward.

## The anchor pass — full unmasked recompute

The anchor circuit, from `verl/workers/comm_eff/anchor.py`, every `cadence`
PPO substeps:

```python
def _maybe_comm_eff_anchor_refresh(self, batch, step):
    if step % self.cfg.anchor.cadence != 0:
        return

    snapshot = self._snapshot_live_weights(delay_K=self.cfg.anchor.delay_K)
    clone = self._anchor_module_cache or build_anchor_module(self.live_module)
    clone.load_state_dict(snapshot)                # K-stale params

    with _no_mask_context():                        # mask_active=False, path_tag=None
        logits = clone(batch.tokens)                # FULL unmasked forward
        loss   = grpo_actor_loss(logits, batch)
        loss.backward()                             # FULL unmasked backward

    # harvest gradients
    for name, p in clone.named_parameters():
        if name in self.spectral_targets:
            G_anchor = p.grad                       # gold-signal gradient
            self.M_anchor[name] = self.beta_anc * self.M_anchor[name] + (1 - self.beta_anc) * G_anchor

    clone.zero_grad()
    self._anchor_module_cache = clone
    torch.cuda.empty_cache()
    # NO optimizer.step(). NO rollout. NO reward recompute.
```

What "recompute everything" means concretely:

| Step | What's recomputed | Why |
|---|---|---|
| Forward pass | YES — the clone walks every layer fresh, no checkpoints reused | The clone has its own activation memory; the live module's checkpointed activations are not visible to it |
| Backward pass | YES — full bf16 gradient on every clone parameter | This is the point. `loss.backward()` populates `clone.param.grad` |
| Loss | YES — recomputed from `clone(batch.tokens)` | Same GRPO actor loss as #3, computed on the K-stale weights |
| Rollouts | **NO** — the same batch from the live training step is reused | We're not generating new data, just re-evaluating the loss on existing rollouts with stale weights |
| Reward | **NO** — the reward signal already lives in `batch` | `anchor_rewards_recomputed == 0` (GUARD 8) |
| Optimizer | **NO** — no `optimizer.step()` on the clone | `anchor_optimizer_steps == 0` (GUARD 8) |

So "recompute everything" is more precisely: **recompute the forward, the
loss, and the backward, on the same training batch, with the same reward,
using K-stale weights, on a hookless clone, with no optimizer update.** The
data (rollouts + rewards) is shared with the masked train step; the
*computation* is independent.

## Why the anchor recompute matters

The anchor's job is to give us a gradient that we know is unbiased — no
mask, no spectral correction, no FSDP-state contamination — so we can build
a Tikhonov-projected basis (`M_anchor → SVD → U,S,V`) that the *masked*
gradient is then routed through:

```
G_proj = α · G_mask + (1 − α) · G_filt   where G_filt = U diag(d) U^T G_mask V diag(d) V^T
```

If the anchor were ALSO masked, the basis `M_anchor` would be defined by the
same compression noise we're trying to correct. The projection would be
self-referential — it would project `G_mask` into a sub-space defined by a
masked gradient, which is exactly the directions we *don't* want to trust.

If the anchor were spectrally corrected, the basis would be defined by an
already-projected gradient — again self-referential, this time with the
spectral filter folded back on itself.

If the anchor rolled out / recomputed rewards / took an optimizer step, it
would be doing additional training work, which would (a) double the compute
cost (defeating the "communication-efficient" claim) and (b) make the
"K-stale" snapshot meaningless because the clone's weights would diverge
from the live model on each refresh.

So the design is **deliberately frugal everywhere except the unmasked
forward/backward** — that's the one thing the anchor exists to do.

## Putting your understanding back in your words

| Your phrasing | Code-truth refinement |
|---|---|
| "All forward passes in fast circuit get masked" | **The two gradient-feeding forwards** in the fast circuit (actor train and old-logprob recompute, when `mask_recompute=true`) get masked. Rollouts, validation, and (when present) the reference-policy forward are NOT in the fast circuit and are NOT masked. |
| "Nothing in anchor gets masked" | Correct, structurally: `path_tag=None` is never in `MASK_ELIGIBLE_TAGS`, so the mask hook is a no-op on the anchor's forward. Plus `mask_active=False` during the anchor pass as a belt-and-braces guard. |
| "We recompute everything" | Forward + loss + backward — yes, fully recomputed on the clone. Rollouts and reward — NOT recomputed (reused from the live training batch). Optimizer — NOT applied. So "we recompute the gradient pipeline, but reuse the data and skip the update." |

Your model is essentially right; the precision improvement is just being
specific about *which* forwards count as "fast circuit" and what "everything"
means in the anchor context.

## Runtime confirmation (the paper-scale dry run iter2, step 11)

```
mask_applications/train:        497   ← #3 firing on every PPO inner micro-batch
mask_applications/old_logprob:  455   ← #2 firing because mask_recompute=true
mask_applications/rollout:        0   ← #1 untouched
mask_applications/ref_logprob:    0   ← #4 never (no ref worker)
mask_applications/val:            0   ← #5 untouched
mask_applications/infer:          0   ← any other inference path: clean
mask_applications/ckpt:           0   ← checkpoint save forward (if any): clean
mask_ratio:                   0.8999  ← target p=0.9, observed 0.8999 ≈ p

anchor_backwards:                 8   ← anchor pass fired 8 times so far
anchor_mask_applications:         0   ← unmasked, every time
anchor_grad_corrected:            0   ← uncorrected, every time
anchor_rollouts_generated:        0   ← no new rollouts
anchor_rewards_recomputed:        0   ← no reward recompute
anchor_optimizer_steps:           0   ← no optimizer update on the clone
anchor_backward_isolation_mode:  clone  ← the anchor isolation work hookless clone in use
```

Every measurement matches the design intent. The fast circuit is masked on
exactly two paths; everything else, including the anchor, is structurally
unmasked.
