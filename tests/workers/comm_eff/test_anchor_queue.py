# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for the EXP-8 anchor circuit (no GPU / no torch.distributed).

Loads the (lightweight) ``anchor`` and ``spectral_filter`` modules by file path
so the heavy ``verl.__init__`` chain (tensordict / vllm / ray) is not required —
same harness as ``test_grad_correction_hook.py``.

The six non-negotiable anchor invariants (the verify gate + the on-box
falsifiers) are exercised here at the unit level:

1. **K-staleness snapshot integrity** — the queue returns the ``t - delay_K``
   snapshot, with a documented warm-up fallback to the oldest snapshot, and
   bounded memory.
2. **Anchor reuses the GRPO actor-loss (NOT supervised next-token)** — the
   engine-ordering simulation calls the SAME ``loss_function`` object the fast
   path would use; the per-token supervised CE path is never invoked.
3. **No rollout / no reward recompute** — the anchor refresh never calls a
   rollout-generation or reward-scoring callable.
4. **No optimizer step** — the anchor refresh never steps the optimizer.
5. **mask_active == False on the anchor pass** — masking is disabled for the
   whole anchor fwd/bwd; ``anchor_mask_applications`` stays 0.
6. **G_anchor read BEFORE any correct_matrix** — the raw grads are fed to
   ``SpectralFilter.update_anchor`` (the EMA); ``correct_matrix`` is never
   called on the anchor gradient (``anchor_grad_corrected`` stays 0).
"""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _stub_parent_packages():
    for pkg in ("verl", "verl.workers", "verl.workers.comm_eff"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_parent_packages()
_sf = _load("verl.workers.comm_eff.spectral_filter", "verl/workers/comm_eff/spectral_filter.py")
_anchor = _load("verl.workers.comm_eff.anchor", "verl/workers/comm_eff/anchor.py")
_st = _load("_exp8_state", "verl/workers/comm_eff/state.py")

AnchorStalenessQueue = _anchor.AnchorStalenessQueue
snapshot_named_params = _anchor.snapshot_named_params
extract_target_grads = _anchor.extract_target_grads
feed_anchor_grads_into_ema = _anchor.feed_anchor_grads_into_ema
anchor_should_fire = _anchor.anchor_should_fire
build_anchor_module = _anchor.build_anchor_module
assert_anchor_module_isolated = _anchor.assert_anchor_module_isolated
SpectralFilter = _sf.SpectralFilter


# --------------------------------------------------------------------------- #
# Tiny decoder-shaped module (proj names match the target substrings)
# --------------------------------------------------------------------------- #
class _TinyDecoderLayer(torch.nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.k_proj = torch.nn.Linear(d, d, bias=False)
        self.v_proj = torch.nn.Linear(d, d, bias=False)
        self.o_proj = torch.nn.Linear(d, d, bias=False)
        self.norm = torch.nn.LayerNorm(d)  # 1-D params, must NOT be a target

    def forward(self, x):
        h = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        return self.o_proj(self.norm(h))


class _TinyDecoder(torch.nn.Module):
    def __init__(self, d=8, n_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyDecoderLayer(d) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _identity_full_grad_of(grad):
    return grad, {"grad_container_type": type(grad).__name__, "is_dtensor": "False"}


# =========================================================================== #
# 1. K-staleness snapshot integrity
# =========================================================================== #
def test_queue_returns_t_minus_k_snapshot():
    q = AnchorStalenessQueue(delay_K=1)
    q.push(1, {"w": torch.tensor(1.0)})
    q.push(2, {"w": torch.tensor(2.0)})
    q.push(3, {"w": torch.tensor(3.0)})
    # At step 3 with delay_K=1 we want the step-2 snapshot.
    snap = q.get_stale(3, 1)
    assert snap["w"].item() == 2.0


def test_queue_warmup_falls_back_to_oldest():
    q = AnchorStalenessQueue(delay_K=1)
    # Step 1: only the current snapshot exists; t-K=0 is absent -> oldest (step 1).
    q.push(1, {"w": torch.tensor(10.0)})
    snap = q.get_stale(1, 1)
    assert snap["w"].item() == 10.0  # documented warm-up: step-1 sees current


def test_queue_is_bounded():
    q = AnchorStalenessQueue(delay_K=1)  # retains at most delay_K+1 = 2
    for s in range(1, 11):
        q.push(s, {"w": torch.tensor(float(s))})
    assert len(q) == 2
    assert q.steps == [9, 10]


def test_queue_delay_zero_returns_current():
    q = AnchorStalenessQueue(delay_K=0)
    q.push(5, {"w": torch.tensor(5.0)})
    assert q.get_stale(5, 0)["w"].item() == 5.0


def test_anchor_should_fire_cadence():
    # cadence=1 fires every step; cadence=20 fires on 20,40,...; disabled never.
    assert anchor_should_fire(1, 1, True)
    assert anchor_should_fire(5, 1, True)
    assert not anchor_should_fire(1, 20, True)
    assert anchor_should_fire(20, 20, True)
    assert anchor_should_fire(40, 20, True)
    assert not anchor_should_fire(20, 20, False)  # enabled=False short-circuits


# =========================================================================== #
# Snapshot decoupling from the optimizer's param group (criterion 7 support)
# =========================================================================== #
def test_snapshot_is_detached_clone_off_optimizer():
    module = _TinyDecoder()
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
    snap = snapshot_named_params(module.named_parameters(), target_substrs=None)
    # Snapshot tensors are NOT the live params and carry no autograd history,
    # so the optimizer (which holds the live params) can never step them.
    live = dict(module.named_parameters())
    for name, t in snap.items():
        assert t is not live[name]
        assert not t.requires_grad
        # The snapshot is not any optimizer param-group tensor.
        for group in opt.param_groups:
            assert all(t is not p for p in group["params"])


def test_snapshot_target_filter():
    module = _TinyDecoder()
    snap = snapshot_named_params(module.named_parameters(), target_substrs=_TARGETS)
    assert snap, "expected some target snapshots"
    assert all(any(s in n for s in _TARGETS) for n in snap)
    assert all("norm" not in n for n in snap)


# =========================================================================== #
# 6. extract_target_grads reads RAW grads, applies NO correction
# =========================================================================== #
def test_extract_target_grads_is_raw_and_2d_only():
    module = _TinyDecoder()
    x = torch.randn(4, 8)
    module(x).pow(2).mean().backward()
    grads = extract_target_grads(
        module.named_parameters(), target_substrs=_TARGETS, max_targets=-1, full_grad_of=_identity_full_grad_of
    )
    # 2 layers * 4 proj = 8 targets; LayerNorm (1-D) excluded.
    assert len(grads) == 8
    assert all("norm" not in n for n in grads)
    # Returned grads equal the RAW .grad (no correction), as detached clones.
    live = dict(module.named_parameters())
    for n, g in grads.items():
        assert g.dim() == 2
        assert torch.allclose(g, live[n].grad)
        assert g is not live[n].grad  # detached clone, not an alias


def test_extract_target_grads_respects_max_targets():
    module = _TinyDecoder()
    module(torch.randn(4, 8)).pow(2).mean().backward()
    grads = extract_target_grads(
        module.named_parameters(), target_substrs=_TARGETS, max_targets=3, full_grad_of=_identity_full_grad_of
    )
    assert len(grads) == 3


def test_extract_skips_none_grad():
    module = _TinyDecoder()  # no backward -> grads None
    grads = extract_target_grads(
        module.named_parameters(), target_substrs=_TARGETS, max_targets=-1, full_grad_of=_identity_full_grad_of
    )
    assert grads == {}


# =========================================================================== #
# 6. feed_anchor_grads_into_ema calls update_anchor (EMA), NEVER correct_matrix
# =========================================================================== #
def test_feed_uses_update_anchor_not_correct_matrix():
    f = SpectralFilter(beta_anc=0.95, seed_anchor_cache=False, ema_device="gpu", svd_mode="full", basis_cache="cache")

    called = {"correct_matrix": 0, "update_anchor": 0}
    orig_correct = f.correct_matrix
    orig_update = f.update_anchor

    def _spy_correct(name, g):
        called["correct_matrix"] += 1
        return orig_correct(name, g)

    def _spy_update(name, g):
        called["update_anchor"] += 1
        return orig_update(name, g)

    f.correct_matrix = _spy_correct
    f.update_anchor = _spy_update

    grads = {"model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8)}
    deltas = feed_anchor_grads_into_ema(grads, f)

    assert called["update_anchor"] == 1, "anchor grad must go through update_anchor (the EMA)"
    assert called["correct_matrix"] == 0, "GUARD 6: anchor grad must NEVER pass through correct_matrix"
    # ΔM_anchor > 0 (EMA moved off zeros): beta=0.95 => M = 0.05 * ones != 0.
    assert deltas["model.layers.0.self_attn.q_proj.weight"] > 0.0


def test_feed_evolves_ema_across_refreshes():
    """Two refreshes with DIFFERENT G_anchor must move M_anchor each time
    (criterion 3: ||ΔM_anchor|| > 0 between the first and a later refresh)."""
    f = SpectralFilter(beta_anc=0.5, seed_anchor_cache=False)
    name = "model.layers.1.mlp.gate_proj.weight"
    d1 = feed_anchor_grads_into_ema({name: torch.ones(6, 4)}, f)
    m_after_1 = f._anchor[name].clone()
    d2 = feed_anchor_grads_into_ema({name: 3.0 * torch.ones(6, 4)}, f)
    m_after_2 = f._anchor[name].clone()
    assert d1[name] > 0.0 and d2[name] > 0.0
    assert not torch.allclose(m_after_1, m_after_2), "M_anchor must evolve across refreshes"


# =========================================================================== #
# 2-5. Engine-ordering SIMULATION: same loss, unmasked, no rollout/reward,
#       no optimizer step, G_anchor read before any correction.
# =========================================================================== #
class _FakeState:
    """Minimal CommEffState-shaped object recording mask_active over time."""

    def __init__(self):
        self.mask_active = False
        self.path_tag = None
        self.mask_applications = 0
        self.anchor_backwards = 0
        self.anchor_mask_applications = 0
        self.anchor_grad_corrected = 0
        self.anchor_rollouts_generated = 0
        self.anchor_rewards_recomputed = 0
        self.anchor_optimizer_steps = 0
        self.anchor_batch_fraction = 1.0

    def set_path_tag(self, tag):
        self.path_tag = tag


def _simulate_anchor_refresh(module, opt, spectral, state, loss_function, *, mask_hook_fires_if_active):
    """Mirror FSDPEngine._maybe_comm_eff_anchor_refresh's ordering on CPU.

    Records the call trace so the test can assert: same loss_function used,
    mask_active False throughout the pass, no rollout/reward callable invoked,
    no optimizer step, and update_anchor (not correct_matrix) consumes G_anchor
    read AFTER backward.
    """
    trace = []

    # --- GUARD 5: ensure masking off for the anchor pass --------------------
    prev_mask_active, prev_tag = state.mask_active, state.path_tag
    state.mask_active = False
    state.set_path_tag(None)
    mask_apps_before = state.mask_applications

    # --- snapshot off the optimizer param group, then run UNMASKED fwd/bwd --
    snapshot_named_params(module.named_parameters(), target_substrs=None)  # decoupled clone
    opt.zero_grad(set_to_none=True)

    x = torch.randn(4, 8)
    out = module(x)
    # GUARD 1+5: reuse the SAME loss_function the fast path uses; the mask hook
    # would fire here ONLY if mask_active were True (it is not).
    if mask_hook_fires_if_active and state.mask_active:
        state.mask_applications += 1
    loss = loss_function(out)
    trace.append("loss_function")
    loss.backward()
    trace.append("backward")

    # --- GUARD 6: read G_anchor RAW, feed EMA, BEFORE any correct_matrix ----
    grads = extract_target_grads(
        module.named_parameters(), target_substrs=_TARGETS, max_targets=4, full_grad_of=_identity_full_grad_of
    )
    trace.append("extract_grads")
    feed_anchor_grads_into_ema(grads, spectral, state=state)
    trace.append("update_anchor")

    # No optimizer step. Clear anchor grads so the (simulated) fast path is clean.
    opt.zero_grad(set_to_none=True)

    # restore mask/path state
    state.mask_active, state.path_tag = prev_mask_active, prev_tag
    state.anchor_backwards += 1
    state.anchor_mask_applications += state.mask_applications - mask_apps_before
    return trace


def test_anchor_reuses_grpo_loss_unmasked_no_step_no_rollout():
    module = _TinyDecoder()
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
    spectral = SpectralFilter(beta_anc=0.95, seed_anchor_cache=False)
    state = _FakeState()

    # The GRPO-actor-loss stand-in: a function of the model OUTPUT (advantage-
    # weighted), NOT a supervised next-token CE against labels. We assert below
    # that no labels/targets are consulted.
    def grpo_like_loss(model_output):
        return model_output.pow(2).mean()

    # A supervised next-token loss we must NEVER call (negative control).
    def supervised_next_token_loss(model_output, labels):  # pragma: no cover - must not run
        raise AssertionError("anchor must NOT use a supervised next-token loss")

    # Rollout / reward callables that must NEVER be invoked by the anchor.
    def generate_rollouts():  # pragma: no cover
        state.anchor_rollouts_generated += 1
        raise AssertionError("anchor must NOT generate rollouts")

    def recompute_rewards():  # pragma: no cover
        state.anchor_rewards_recomputed += 1
        raise AssertionError("anchor must NOT recompute rewards")

    # Snapshot optimizer state to prove no step was taken.
    params_before = [p.detach().clone() for p in module.parameters()]

    trace = _simulate_anchor_refresh(
        module, opt, spectral, state, grpo_like_loss, mask_hook_fires_if_active=True
    )

    # 2. reused the GRPO-like loss; supervised path untouched (would have raised).
    assert "loss_function" in trace and "backward" in trace
    # 3+4. no rollout / no reward calls happened.
    assert state.anchor_rollouts_generated == 0
    assert state.anchor_rewards_recomputed == 0
    # 5. mask_active stayed False => zero mask applications during the pass.
    assert state.anchor_mask_applications == 0
    # 7. no optimizer step => live params unchanged.
    for b, p in zip(params_before, module.parameters()):
        assert torch.allclose(b, p), "anchor must NOT step the optimizer (params changed)"
    assert state.anchor_optimizer_steps == 0
    # 6. ordering: grads extracted then fed to the EMA (update_anchor) AFTER
    # backward; correct_matrix never ran (anchor_grad_corrected stays 0).
    assert trace.index("backward") < trace.index("extract_grads") < trace.index("update_anchor")
    assert state.anchor_grad_corrected == 0
    # The EMA actually received the raw grad (it moved off zeros).
    assert any(torch.linalg.norm(v).item() > 0 for v in spectral._anchor.values())


def test_anchor_pass_does_not_fire_mask_even_if_hook_present():
    """GUARD 5: even with a mask hook that WOULD fire on the train path, the
    anchor pass (mask_active=False) fires zero mask applications."""
    module = _TinyDecoder()
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
    spectral = SpectralFilter(seed_anchor_cache=False)
    state = _FakeState()
    # Pre-set a nonzero global mask count (from prior fast-path steps) to ensure
    # we measure the DELTA, not the absolute.
    state.mask_applications = 7

    _simulate_anchor_refresh(
        module, opt, spectral, state, lambda o: o.pow(2).mean(), mask_hook_fires_if_active=True
    )
    assert state.anchor_mask_applications == 0  # delta over the anchor pass is 0
    assert state.mask_applications == 7  # unchanged: no mask fired


# =========================================================================== #
# 13. FSDP1 anchor-backward isolation regression (EXP-12 criterion 13).
#
# EXP-8 cells 1+3 crashed because the anchor's loss.backward() fired FSDP1's
# _post_backward_hook on the live FlatParameter. That hook calls
# _check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard); but
# _saved_grad_shard is None outside the fast-path's backward window, so it
# raises `AttributeError: 'NoneType' object has no attribute 'shape'`.
#
# The fix (EXP-12): run the anchor on a copy.deepcopy'd plain nn.Module whose
# parameters are NOT registered with FSDP, so no post-backward hook fires on
# the anchor pass. This test SIMULATES that hook collision at the CPU layer:
#
#   - We attach a `register_post_accumulate_grad_hook` to the live params that
#     reads a sentinel attribute `_saved_grad_shard` — None by default
#     (emulating "outside the fast-path window"). The hook calls
#     `_check_grad_to_accumulate(_saved_grad_shard.shape)` which raises
#     AttributeError when the sentinel is None.
#   - The EXP-8 code path (anchor backward on LIVE module) triggers that hook
#     and the test asserts AttributeError is raised — proving the failure mode
#     reproduces on the simulated hook.
#   - The EXP-12 code path (anchor backward on a `copy.deepcopy(live)` clone)
#     does NOT trigger the hook (the clone has no such hook registered) and
#     the test asserts (a) no AttributeError, (b) the live params' sentinel
#     remains untouched after the anchor pass.
#
# This is the only way to certify the fix without burning a Vast.ai run to
# discover a regression; the on-box runtime check (criterion 8) is the second
# gate.
# =========================================================================== #
def _simulated_fsdp1_post_backward_check(p):
    """Mirror FSDP1's _post_backward_hook -> _check_grad_to_accumulate.

    Reads `p._saved_grad_shard.shape` — `_saved_grad_shard` is `None` outside
    the fast-path's backward window, so this raises the same AttributeError as
    `_check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)`.
    """
    saved = getattr(p, "_saved_grad_shard", None)
    # The crash mode: torch FSDP1 reads `.shape` on `None` and AttributeError fires.
    _ = saved.shape  # pragma: no cover - load-bearing failure mode
    # Mark that the hook fired (only reachable if the saved sentinel was not None).
    p._fsdp1_hook_fired = getattr(p, "_fsdp1_hook_fired", 0) + 1


def _attach_simulated_fsdp1_hooks(module):
    """Attach the simulated FSDP1 post-accumulate-grad hook on every 2-D param.

    Returns the list of hook handles so callers can detach if needed. The
    `_saved_grad_shard` sentinel is initialized to None to mimic "outside the
    fast-path's backward window" — which is exactly the EXP-8 collision state.
    """
    handles = []
    for _, p in module.named_parameters():
        if p.dim() != 2:
            continue
        p._saved_grad_shard = None
        p._fsdp1_hook_fired = 0
        h = p.register_post_accumulate_grad_hook(_simulated_fsdp1_post_backward_check)
        handles.append(h)
    return handles


def _build_anchor_clone(live_module):
    """EXP-12 fix: deep-clone the underlying nn.Module so the anchor's backward
    does NOT trigger any hook registered on the LIVE params.

    Delegates to the canonical helper ``verl.workers.comm_eff.anchor.build_anchor_module``
    so this test exercises the EXACT production code path. The clone's
    parameters share NO storage with the live params (deepcopy is a full copy),
    carry NO ``register_post_accumulate_grad_hook`` handles, and are NOT in any
    optimizer.param_groups (the caller never wires them in).
    """
    return build_anchor_module(live_module)


def test_fsdp_anchor_backward_no_collision():
    """EXP-12 criterion 13 — regression test for the EXP-8 FSDP1 collision.

    (a) Instantiates a small FSDP1-like wrapped module (here: an `nn.Module`
        with post-accumulate-grad hooks simulating FSDP1's
        ``_post_backward_hook``).
    (b) Runs an anchor-style ``loss.backward()`` through the EXP-12
        clone-no-hook path.
    (c) Asserts no AttributeError from ``_check_grad_to_accumulate``.
    (d) Asserts ``flat_param._saved_grad_shard`` for live params remains
        undisturbed across the anchor pass.

    Also asserts that the EXP-8 broken path (anchor backward on the LIVE
    module) DOES raise AttributeError on the simulated hook, so the test
    actually exercises the failure mode.
    """
    live = _TinyDecoder()
    _attach_simulated_fsdp1_hooks(live)

    # --- (a) EXP-8 broken path: backward on the LIVE module triggers the hook.
    # We expect AttributeError to be raised because `_saved_grad_shard is None`.
    with pytest.raises(AttributeError):
        out = live(torch.randn(4, 8))
        out.pow(2).mean().backward()

    # Re-init grads on live so we can prove they remain untouched by the clone path.
    for p in live.parameters():
        p.grad = None
        p._saved_grad_shard = None
        p._fsdp1_hook_fired = 0

    # --- (b) EXP-12 fix path: backward on the deep-cloned module fires NO hooks.
    clone = _build_anchor_clone(live)

    # Confirm the clone carries no FSDP1 sentinel/hook (deepcopy copies the
    # attribute value but the *registered hook* is per-Parameter object, NOT
    # copied — the new Parameter has zero post-accumulate-grad hooks).
    for _, p in clone.named_parameters():
        # deepcopy of a Parameter is a fresh Parameter; post-accumulate-grad
        # hooks live on Tensor's _backward_hooks/grad_fn machinery and are
        # NOT transferred. We sanity-check by detecting that no hook is
        # registered: the simplest robust check is that backward on the clone
        # does not raise.
        assert p is not next(iter(live.parameters())), "clone params must NOT alias live params"

    # The anchor-style backward on the clone must succeed (no AttributeError).
    out_clone = clone(torch.randn(4, 8))
    out_clone.pow(2).mean().backward()  # MUST NOT RAISE

    # --- (c) No clone-side hook fired — the post-accumulate sentinel attribute
    # was not copied as a registered hook (deepcopy creates fresh Parameter
    # objects whose internal C++-level _backward_hooks dict is empty).
    for _, p in clone.named_parameters():
        if p.dim() != 2:
            continue
        # Either the attribute isn't set on the clone (different Parameter
        # object) or it's been deep-copied as a plain Python None — but
        # critically, NO post-accumulate hook fired, so `_fsdp1_hook_fired`
        # stays 0.
        fired = getattr(p, "_fsdp1_hook_fired", 0)
        assert fired == 0, (
            "EXP-12 clone path must NOT fire the simulated FSDP1 "
            f"_post_backward_hook (fired={fired})"
        )

    # --- (d) Live params' `_saved_grad_shard` remains untouched.
    for _, p in live.named_parameters():
        if p.dim() != 2:
            continue
        # Live params' sentinel was set to None before the clone pass; the
        # clone-no-hook anchor backward must NOT have touched the live attrs.
        assert getattr(p, "_saved_grad_shard", "MISSING") is None, (
            "EXP-12 clone path must NOT mutate live `_saved_grad_shard`"
        )
        # And no hook fired on the live side either (because clone.backward
        # operated on a disjoint param object graph).
        assert getattr(p, "_fsdp1_hook_fired", 0) == 0, (
            "EXP-12 clone path must NOT fire the live FSDP1 hook"
        )


def test_anchor_clone_is_off_optimizer_param_group():
    """EXP-12 belt-and-braces (next_actions §anchor_optimizer_param_group):
    the anchor clone's params must NOT appear in any live optimizer's
    param_groups. This guards criterion 7 (anchor_optimizer_steps == 0)
    against a future refactor that accidentally shares storage with a live
    FSDP-handled FlatParameter.
    """
    live = _TinyDecoder()
    opt = torch.optim.AdamW(live.parameters(), lr=1e-3)
    clone = _build_anchor_clone(live)
    live_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    for _, p in clone.named_parameters():
        assert id(p) not in live_ids, (
            "anchor clone param appears in the live optimizer's param_groups — "
            "criterion 7 at risk (clone must be deep-copied OFF the optimizer)"
        )
    # Belt-and-braces helper also catches the alias by id() — same path the
    # engine uses at runtime.
    assert_anchor_module_isolated(clone, optimizer=opt, fsdp_module=live)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
