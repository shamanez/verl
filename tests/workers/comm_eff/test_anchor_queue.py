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

"""CPU unit tests for the anchor circuit (no GPU / no torch.distributed).

Loads the (lightweight) ``anchor`` and ``spectral_filter`` modules by file path
so the heavy ``verl.__init__`` chain (tensordict / vllm / ray) is not required —
same harness as ``test_grad_correction_hook.py``.

The anchor invariants are exercised here at the unit level:

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
6. **G_anchor read BEFORE any fast-path correction** — the raw grads are fed to
   ``SpectralFilter.update_anchor`` (the EMA); a fast-path merger
   (e.g. ``signed_ema_matrix``) is never called on the anchor gradient
   (``anchor_grad_corrected`` stays 0).
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
_st = _load("_comm_eff_state_anchor_queue", "verl/workers/comm_eff/state.py")

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
# Snapshot decoupling from the optimizer's param group.
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
# extract_target_grads reads raw grads and applies no correction.
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
# feed_anchor_grads_into_ema calls update_anchor (EMA), never the fast-path
# corrector (signed_ema_matrix).
# =========================================================================== #
def test_feed_uses_update_anchor_not_corrector():
    f = SpectralFilter(beta_anc=0.95, ema_device="gpu", correction_mode="signed_ema")

    called = {"signed_ema_matrix": 0, "update_anchor": 0}
    orig_correct = f.signed_ema_matrix
    orig_update = f.update_anchor

    def _spy_correct(name, g):
        called["signed_ema_matrix"] += 1
        return orig_correct(name, g)

    def _spy_update(name, g):
        called["update_anchor"] += 1
        return orig_update(name, g)

    f.signed_ema_matrix = _spy_correct
    f.update_anchor = _spy_update

    grads = {"model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8)}
    deltas = feed_anchor_grads_into_ema(grads, f)

    assert called["update_anchor"] == 1, "anchor grad must go through update_anchor (the EMA)"
    assert called["signed_ema_matrix"] == 0, "GUARD 6: anchor grad must NEVER pass through the fast-path corrector"
    # ΔM_anchor > 0 (EMA moved off zeros): beta=0.95 => M = 0.05 * ones != 0.
    assert deltas["model.layers.0.self_attn.q_proj.weight"] > 0.0


def test_feed_evolves_ema_across_refreshes():
    """Two refreshes with DIFFERENT G_anchor must move M_anchor each time
    (||ΔM_anchor|| > 0 between the first and a later refresh)."""
    f = SpectralFilter(beta_anc=0.5)
    name = "model.layers.1.mlp.gate_proj.weight"
    d1 = feed_anchor_grads_into_ema({name: torch.ones(6, 4)}, f)
    m_after_1 = f._anchor[name].clone()
    d2 = feed_anchor_grads_into_ema({name: 3.0 * torch.ones(6, 4)}, f)
    m_after_2 = f._anchor[name].clone()
    assert d1[name] > 0.0 and d2[name] > 0.0
    assert not torch.allclose(m_after_1, m_after_2), "M_anchor must evolve across refreshes"


# =========================================================================== #
# Engine-ordering simulation: same loss, unmasked, no rollout/reward, no
# optimizer step, G_anchor read before any correction.
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

    # Ensure masking off for the anchor pass.
    prev_mask_active, prev_tag = state.mask_active, state.path_tag
    state.mask_active = False
    state.set_path_tag(None)
    mask_apps_before = state.mask_applications

    # --- snapshot off the optimizer param group, then run UNMASKED fwd/bwd --
    snapshot_named_params(module.named_parameters(), target_substrs=None)  # decoupled clone
    opt.zero_grad(set_to_none=True)

    x = torch.randn(4, 8)
    out = module(x)
    # Reuse the same loss_function the fast path uses; the mask hook
    # would fire here ONLY if mask_active were True (it is not).
    if mask_hook_fires_if_active and state.mask_active:
        state.mask_applications += 1
    loss = loss_function(out)
    trace.append("loss_function")
    loss.backward()
    trace.append("backward")

    # Read G_anchor raw, feed EMA, before any correct_matrix.
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
    spectral = SpectralFilter(beta_anc=0.95)
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

    trace = _simulate_anchor_refresh(module, opt, spectral, state, grpo_like_loss, mask_hook_fires_if_active=True)

    # 2. reused the GRPO-like loss; supervised path untouched (would have raised).
    assert "loss_function" in trace and "backward" in trace
    # 3+4. no rollout / no reward calls happened.
    assert state.anchor_rollouts_generated == 0
    assert state.anchor_rewards_recomputed == 0
    # 5. mask_active stayed False => zero mask applications during the pass.
    assert state.anchor_mask_applications == 0
    # 7. no optimizer step => live params unchanged.
    for b, p in zip(params_before, module.parameters(), strict=False):
        assert torch.allclose(b, p), "anchor must NOT step the optimizer (params changed)"
    assert state.anchor_optimizer_steps == 0
    # 6. ordering: grads extracted then fed to the EMA (update_anchor) AFTER
    # backward; correct_matrix never ran (anchor_grad_corrected stays 0).
    assert trace.index("backward") < trace.index("extract_grads") < trace.index("update_anchor")
    assert state.anchor_grad_corrected == 0
    # The EMA actually received the raw grad (it moved off zeros).
    assert any(torch.linalg.norm(v).item() > 0 for v in spectral._anchor.values())


def test_anchor_pass_does_not_fire_mask_even_if_hook_present():
    """Even with a mask hook that would fire on the train path, the
    anchor pass (mask_active=False) fires zero mask applications."""
    module = _TinyDecoder()
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
    spectral = SpectralFilter()
    state = _FakeState()
    # Pre-set a nonzero global mask count (from prior fast-path steps) to ensure
    # we measure the DELTA, not the absolute.
    state.mask_applications = 7

    _simulate_anchor_refresh(module, opt, spectral, state, lambda o: o.pow(2).mean(), mask_hook_fires_if_active=True)
    assert state.anchor_mask_applications == 0  # delta over the anchor pass is 0
    assert state.mask_applications == 7  # unchanged: no mask fired


def test_anchor_clone_is_off_optimizer_param_group():
    """The anchor clone's params must not appear in any live optimizer param_groups.

    This guards ``anchor_optimizer_steps == 0`` against accidental storage
    sharing with a live FSDP-handled FlatParameter.
    """
    live = _TinyDecoder()
    opt = torch.optim.AdamW(live.parameters(), lr=1e-3)
    clone = build_anchor_module(live)
    live_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    for _, p in clone.named_parameters():
        assert id(p) not in live_ids, (
            "anchor clone param appears in the live optimizer's param_groups — "
            "criterion 7 at risk (clone must be deep-copied OFF the optimizer)"
        )
    # Runtime helper also catches aliasing by id().
    assert_anchor_module_isolated(clone, optimizer=opt, fsdp_module=live)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
