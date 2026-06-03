# EXP-20 PowerSGD — Autograd & Gradient-Checkpointing Correctness Review

**Reviewer:** mathematical-checker (Task #2; mechanism-level drill-down on claim 1 of `math_validity_review.md`)
**Date:** 2026-06-04
**Reviewed commit:** `f748dbc1c63ef9824a3115b091ed025fe210cf9b` (`origin/exp/20-powersgd-activation`)
**Scope:** the autograd graph mechanics — Q-detached / M-in-graph / self-adjoint backward / no-STE; gradient-checkpoint recompute identity; off-graph no_grad sketch isolation; the `grad_enabled`-at-hook-entry subtlety; FSDP `use_orig_params` / flat-param / `summon_full_params` interaction with an output-replacing forward hook.

## Primary artifacts
- `verl/workers/comm_eff/powersgd_activation.py:302-394` — the forward hook `_hook`.
- `verl/workers/engine/fsdp/transformer_impl.py:298` — `gradient_checkpointing_enable(... use_reentrant=False)`.
- `verl/workers/engine/fsdp/transformer_impl.py:820-843, 864` — register/unregister lifecycle + `ctx = torch.no_grad() if forward_only`.
- `verl/utils/fsdp_utils.py:79-163` — `transformer_auto_wrap_policy` keyed on `_no_split_modules` ⇒ each decoder block is an FSDP unit.
- `research/runs/EXP-20/ce_powersgd_probe_2s_gsm8k.log` — on-box 2-step probe with `use_orig_params=true`, `enable_gradient_checkpointing=true`, the hook live; `q_cond≈1.0000002`, no NaN/OOM.

---

## V1 — Q detached, M in-graph, no STE: **VALID**

Hook body (`powersgd_activation.py:333-341`):
```python
M = h.reshape(-1, hidden_size)
q_fp32 = compressor._ensure_basis(layer_idx, device=M.device, dtype=M.dtype)
q_act  = q_fp32.to(dtype=M.dtype)
Y      = M @ q_act
M_hat  = Y @ q_act.t()
```
- **Q is a non-leaf-of-loss constant.** `self._basis[layer_idx]` is built by `init_basis`→`orthonormalize`, which start from `mat.detach().to(torch.float32)` / a freshly-`torch.randn`'d tensor and run `torch.linalg.qr`. It is never wrapped in `nn.Parameter`, never `.requires_grad_(True)`, and is stored in a plain dict, not registered as a module parameter/buffer that the optimizer or FSDP would touch. `q_act = q_fp32.to(dtype)` is a dtype cast of a non-requires-grad tensor ⇒ still `requires_grad=False`. So `Q` contributes no edge to the autograd graph; `dM_hat/dQ` is never formed.
- **M stays in-graph.** `M` is a `reshape` view of the live module output `h`; both matmuls are ordinary differentiable ops. The returned `M_hat.reshape(orig_shape)` is what the next block consumes, so the loss's graph runs … → `h` → `M` → `Y` → `M_hat` → next block → loss.
- **Backward is the self-adjoint projector.** With `Q` constant, `M_hat = M (Q Qᵀ)`, so `dL/dM = (dL/dM_hat) (Q Qᵀ)ᵀ = (dL/dM_hat) Q Qᵀ`. Since `QᵀQ=I` (QR), `P=QQᵀ` is symmetric idempotent, so the *same* operator hits both the activation (forward) and its gradient (backward) — INF-9. This is genuine autograd of `(M@Q)@Qᵀ`, NOT a straight-through `M_hat = M + (proj(M) − M).detach()` trick. There is no `torch.autograd.Function`, no `.detach()` on `M`, anywhere.
- **Empirical:** `test_autograd_no_ste` builds `M.requires_grad_()`, computes `(Mhat*g).sum().backward()`, and asserts `‖M.grad − (g@Q)@Qᵀ‖/‖·‖ < 1e-5`. I also re-derived/re-ran this numerically — exact to fp tolerance.

## V2 — Off-graph `no_grad` sketch + diagnostics do not perturb gradients: **VALID**

All of q_cond, reconstruction_rel_error, and the `V` sketch are computed inside `with torch.no_grad():` (`powersgd_activation.py:343-383`) AFTER `M_hat` is formed. Inside it:
- `M32 = M.detach().float()`, `Mhat32 = M_hat.detach().float()` — explicit `.detach()`, so even the norms touch no graph.
- `contrib = M32.t() @ Y.detach().to(torch.float32)` — `Y.detach()` severs the sketch from the graph; `V` is a side buffer.
- `cur.add_(contrib)` is an **in-place add on `self._sketch`** (a plain dict tensor), NOT on `M`, `Y`, or `M_hat`. There is no in-place mutation of any graph tensor that autograd would later need, so no "a leaf Variable that requires grad is being used in an in-place operation" / "modified by an inplace operation" error is possible from the sketch. ✓
The diagnostics/sketch therefore cannot change `dL/dM`. The only graph-affecting output of the hook is `M_hat`. ✓

## V3 — The `grad_enabled`-captured-at-hook-entry subtlety: **VALID (and necessary)**

`grad_enabled = torch.is_grad_enabled()` is read at hook entry (`line 317`), *before* the `with torch.no_grad():` block opens. `_should_accumulate_sketch(layer_idx, grad_enabled=grad_enabled)` (called at line 372, inside the no_grad block) uses that captured value rather than re-reading `torch.is_grad_enabled()`.

This is correct and load-bearing:
- The old-logprob recompute runs the *whole* forward under `torch.no_grad()` (`transformer_impl.py:864` `ctx = torch.no_grad() if forward_only else nullcontext()`, with the recompute being `forward_only=True`). At hook entry there, `is_grad_enabled()==False` ⇒ `grad_enabled=False` ⇒ no sketch. ✓ (the recompute must NOT fold into V — Part V.3)
- The actor-train forward runs under `nullcontext()` (grad enabled). At hook entry, `is_grad_enabled()==True` ⇒ `grad_enabled=True`. ✓
- **Why capturing matters:** if the gate re-read `torch.is_grad_enabled()` *inside* the `with torch.no_grad():` block, it would ALWAYS be `False` (the block disables grad), so the train forward would be misclassified and **V would never accumulate** — the basis would never update. The code's comment names exactly this bug; the capture-before-no_grad pattern fixes it. ✓
- `test_old_logprob_recompute_projects_but_no_sketch` and `test_sketch_accumulates_on_train_forward` jointly pin this: the no_grad recompute projects (counter increments) but adds no sketch; the grad-enabled train forward both projects and sketches.

## V4 — Gradient-checkpoint recompute sees identical frozen Q ⇒ identical M_hat ⇒ consistent gradient: **VALID**

Two independent guarantees:
1. **Q is immutable during the forward+backward of a step.** The basis is only ever written in `maybe_update_basis` (`powersgd_activation.py:507-514`), which is called from `engine_workers.py:941-943` in the `finally:` of `train_mini_batch` — i.e. AFTER the backward. The hook only reads `self._basis`. So during the original forward AND its checkpoint recompute (which happens in backward, before `maybe_update_basis`), `Q_t` is the identical tensor. The recomputed `M_hat = (M' @ Q_t) @ Q_tᵀ` uses the same `Q_t` and the same recomputed activation `M'` ⇒ the recomputed `M_hat` matches the value autograd expects, so the gradient is consistent (no "recompute produced a different value than the saved forward" hazard). ✓
2. **Non-reentrant checkpointing.** `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` (`transformer_impl.py:298`). Non-reentrant `torch.utils.checkpoint` re-runs the wrapped region under saved-tensor hooks within the normal autograd engine, so the recomputed forward re-invokes the registered `register_forward_hook` and re-applies the projection deterministically. The frozen `Q_t` (point 1) makes that recompute bit-identical. ✓
3. **Sketch is not double-counted by the recompute.** Even though the recompute re-fires the hook with grad enabled, `_should_accumulate_sketch` gates on `self._sketched_this_gen[layer_idx] == self._fwd_generation` (line 416). `set_context` bumps `_fwd_generation` once per micro-batch (line 297), and the checkpoint recompute reuses the same micro-batch context (no new `set_context`), so the layer's `_sketched_this_gen` already equals the current generation ⇒ the recompute's sketch attempt is skipped. So `V` reflects each micro-batch's activation exactly once, from the *original* forward. ✓ (`test_grad_ckpt_recompute_not_double_counted` re-runs the forward in the same generation and asserts `_sketch` is unchanged.)

Subtlety worth stating: the dedupe keys on `_fwd_generation`, which is a per-`set_context` counter, NOT on autograd internals. This is robust to non-reentrant checkpointing precisely because the recompute does NOT call `set_context` again (context is set once in `prepare_model_inputs` per micro-batch, `transformer_impl.py:1751`). If a future refactor moved `set_context` to fire on every forward invocation (including recompute), the dedupe would break. Flagged as a forward-fragility note, not a current defect.

## V5 — FSDP `use_orig_params` / flat-param / `summon_full_params` interaction with the output-replacing hook: **VALID**

- **Each decoder block is its own FSDP unit.** The wrap policy is `transformer_auto_wrap_policy(transformer_layer_cls = model._no_split_modules)` (`fsdp_utils.py:128-156`); for Qwen2.5-1.5B that wraps `Qwen2DecoderLayer`. The hook is registered on `layers[idx]` (a decoder block) via `register_forward_hook` (`powersgd_activation.py:663`).
- **A forward (post-)hook fires AFTER the FSDP unit's forward completes.** `register_forward_hook` (not pre-hook) runs once the module returns. By then FSDP has all-gathered the block's sharded params, run the block's compute, produced the hidden-state output, and (in FSDP1) is about to reshard. The hook receives the *materialized, full* activation `output[0]` — it does not see or touch sharded flat-params. Replacing `output[0]` with `M_hat` is a pure tensor substitution on the block's *output*, downstream of all flat-param/all-gather/reshard machinery. So it cannot break `summon_full_params`, the flat-param contract, or the all-gather/reshard schedule. ✓
- **`use_orig_params=True`** (actor config, confirmed in the probe log) means the optimizer/grad views are the original per-parameter tensors; this is orthogonal to a hook that operates on *activations* (it never reads `.grad` or params). The PowerSGD hook touches neither parameters nor gradients directly — only the forward activation and an off-graph side buffer — so `use_orig_params` true/false is immaterial to its correctness. ✓
- **`no_sync` (grad accumulation across micro-batches)** is likewise irrelevant to the hook: the hook does not call `.backward()`, does not reduce gradients, and does not depend on whether FSDP defers gradient reduction. The `V` sketch sums per-micro-batch contributions in a plain Python dict, independent of FSDP's gradient-reduction state. ✓
- **Same registration mechanism as the validated PRF masker.** `PowerSGDActivationCompressor.register` mirrors `ActivationMasker.register` (both `find_decoder_layers` → `decoder_boundary_indices` → `register_forward_hook`), and the PRF path has run cleanly under this exact FSDP config through EXP-5…EXP-18. The output-tuple handling (`(M_hat,) + tuple(output[1:])`) matches the masker's `(h_tilde,) + tuple(output[1:])`. ✓
- **On-box corroboration.** The 2-step probe ran with `use_orig_params=true` + grad-ckpt + the hook live across 7 boundaries (layers 3,7,11,15,18,21,24 for Qwen2.5 L=28), completed both steps, took an optimizer step, saved a checkpoint, with `q_cond≈1.0000002` and no NaN/OOM/`summon_full_params`/`flat_param` error in the log. This is direct evidence the hook coexists with FSDP flat-param mechanics. ✓

Guard worth noting: `_comm_eff_register_powersgd_hooks` refuses Ulysses SP>1 (`transformer_impl.py:773-781`) and `_comm_eff_maybe_set_powersgd_context` refuses non-rmpad (padded) inputs (`transformer_impl.py:812-817`) — both are correct guards (SP slices the token axis the projector compresses; padded forwards would fold PAD tokens into M and V). The launcher runs rmpad + SP=1, so these never fire in the sanctioned config.

---

## Bottom line (Task #2)

**The autograd and gradient-checkpointing mechanics are correct.** Q is a detached constant, M stays in-graph, the backward is the genuine self-adjoint projector `dL/dM = (dL/dM_hat) Q Qᵀ` with NO straight-through estimator, and the off-graph `no_grad` diagnostics/sketch (with explicit `.detach()` on M and Y, in-place add only on the side buffer) cannot perturb gradients. The `grad_enabled` value is correctly captured at hook entry before the `no_grad` block (re-reading it inside would always be False and silently disable basis learning — the code fixes exactly that bug). Under non-reentrant gradient checkpointing the recompute re-fires the hook with the *same frozen `Q_t`* (the basis only advances post-backward in `maybe_update_basis`), so the recomputed `M_hat` is bit-identical and the gradient is consistent, while the per-generation dedupe prevents the recompute from double-counting the `V` sketch. The output-replacing `register_forward_hook` fires after each FSDP-wrapped decoder block's forward completes — downstream of all flat-param/all-gather/reshard and `summon_full_params` machinery — and never reads params or gradients, so `use_orig_params`/flat-param/`no_sync` are immaterial to it; this is the same mechanism the validated PRF masker uses, and the on-box probe confirms coexistence (no FSDP errors, `q_cond≈1`, no NaN/OOM).

**No defects.** Two forward-fragility notes (not current bugs): (1) the grad-ckpt sketch dedupe relies on `set_context` firing once per micro-batch and NOT being re-called on recompute — robust today; (2) the SP=1 / rmpad guards are the correctness boundary for the hook and are asserted at registration.
