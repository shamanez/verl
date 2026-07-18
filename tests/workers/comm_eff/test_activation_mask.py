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

"""Unit tests for the per-element, stable-keyed activation masker (CPU-only).

Covers: boundary index derivation; per-(token, dim) independence; mask ratio
tracking p; PRF determinism; the load-bearing cross-packing consistency (a token
keyed by (sample_id, position_id) gets the same mask under any token ordering at
the same step); step sensitivity; in-graph form; hook lifecycle; and the
train-only contamination guard.
"""

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import (
    ActivationMasker,
    _derive_seed,
    decoder_boundary_indices,
    find_decoder_layers,
    prf_token_mask,
)

CPU = torch.device("cpu")


def _ids_for(b: int, s: int):
    """Row-major (sample_id, position_id) for a (b, s, H) activation."""
    sid = torch.arange(b).repeat_interleave(s)
    pos = torch.arange(s).repeat(b)
    return sid, pos


def _set_ctx(masker, b, s, step=0):
    sid, pos = _ids_for(b, s)
    masker.set_context(global_step=step, sample_ids=sid, position_ids=pos)


def _scalar_keep(sid, pos, ch, *, layer, step, seed, p):
    """Scalar reference for one (token, channel) keep bit."""
    h = _derive_seed((seed, layer, step, sid, pos, ch))
    return 1.0 if (h >> 11) >= int(p * (1 << 53)) else 0.0


# --------------------------------------------------------------------------- #
# boundary partition
# --------------------------------------------------------------------------- #
def test_boundary_indices_L16_pp8():
    assert decoder_boundary_indices(16, 8) == [1, 3, 5, 7, 9, 11, 13]


def test_boundary_indices_excludes_final_shard():
    idx = decoder_boundary_indices(16, 8)
    assert 15 not in idx
    assert len(idx) == 7


def test_boundary_indices_uneven_partition():
    # L=10, pp_size=4 -> shard lens [3,3,2,2] -> last idx [2,5,7,9] -> drop 9
    assert decoder_boundary_indices(10, 4) == [2, 5, 7]


def test_boundary_indices_pp_size_one_is_empty():
    assert decoder_boundary_indices(16, 1) == []


def test_boundary_indices_pp_capped_at_num_layers():
    assert decoder_boundary_indices(4, 8) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# PRF determinism + scalar-reference equivalence
# --------------------------------------------------------------------------- #
def test_prf_same_key_same_mask():
    sid, pos = _ids_for(2, 8)
    kw = dict(layer_idx=3, global_step=1, base_seed=7, hidden_size=32, p=0.9, device=CPU, dtype=torch.float32)
    m1 = prf_token_mask(sid, pos, **kw)
    m2 = prf_token_mask(sid, pos, **kw)
    assert torch.equal(m1, m2)


def test_prf_matches_scalar_reference():
    """The vectorized PRF is bit-identical to the documented scalar key."""
    sid = torch.tensor([5, 9, 0])
    pos = torch.tensor([0, 3, 7])
    m = prf_token_mask(
        sid, pos, layer_idx=2, global_step=4, base_seed=1, hidden_size=6, p=0.6, device=CPU, dtype=torch.float32
    )
    for t in range(3):
        for ch in range(6):
            expect = _scalar_keep(int(sid[t]), int(pos[t]), ch, layer=2, step=4, seed=1, p=0.6)
            assert float(m[t, ch]) == expect


def test_prf_mask_is_binary():
    sid, pos = _ids_for(4, 16)
    m = prf_token_mask(
        sid, pos, layer_idx=1, global_step=0, base_seed=0, hidden_size=64, p=0.9, device=CPU, dtype=torch.float32
    )
    assert set(m.unique().tolist()).issubset({0.0, 1.0})


def test_different_step_different_mask():
    sid, pos = _ids_for(4, 16)
    kw = dict(layer_idx=3, base_seed=0, hidden_size=64, p=0.9, device=CPU, dtype=torch.float32)
    a = prf_token_mask(sid, pos, global_step=1, **kw)
    b = prf_token_mask(sid, pos, global_step=2, **kw)
    assert not torch.equal(a, b)


def test_per_element_independence():
    """Different tokens get different masks (not one shared row for all tokens)."""
    sid, pos = _ids_for(8, 8)  # 64 distinct (sid,pos) tokens
    m = prf_token_mask(
        sid, pos, layer_idx=0, global_step=0, base_seed=0, hidden_size=128, p=0.5, device=CPU, dtype=torch.float32
    )
    rows = {tuple(r.tolist()) for r in m}
    # With 64 independent 128-bit Bernoulli rows, all should be distinct.
    assert len(rows) == m.shape[0]


# --------------------------------------------------------------------------- #
# mask ratio tracks p
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [0.90, 0.95])
def test_mask_ratio_tracks_p(p):
    sid, pos = _ids_for(8, 64)
    m = prf_token_mask(
        sid, pos, layer_idx=5, global_step=0, base_seed=1, hidden_size=256, p=p, device=CPU, dtype=torch.float32
    )
    measured_zero_fraction = float(1.0 - m.mean().item())
    assert abs(measured_zero_fraction - p) <= 0.02


# --------------------------------------------------------------------------- #
# cross-packing consistency (the load-bearing GRPO test)
# --------------------------------------------------------------------------- #
def test_cross_packing_consistency():
    """A token keyed by (sid, pos) gets the SAME mask under any token ordering.

    Simulates the old_logprob vs train repacking: present the same set of tokens
    in two different orders at the same global_step and require each token's mask
    row to be identical.
    """
    sid_a = torch.tensor([0, 0, 1, 1, 2])
    pos_a = torch.tensor([0, 1, 0, 1, 0])
    perm = torch.tensor([3, 0, 4, 1, 2])  # an arbitrary repacking
    sid_b, pos_b = sid_a[perm], pos_a[perm]

    kw = dict(layer_idx=7, global_step=11, base_seed=3, hidden_size=48, p=0.9, device=CPU, dtype=torch.float32)
    ma = prf_token_mask(sid_a, pos_a, **kw)
    mb = prf_token_mask(sid_b, pos_b, **kw)
    # Row i of ma corresponds to row perm.index(i) of mb.
    assert torch.equal(ma[perm], mb)


def test_cross_packing_consistency_through_hook():
    """Same property end-to-end through the masker hook on two packings."""
    masker = ActivationMasker(p=0.9, base_seed=3, pp_size=8)
    hook = masker._make_hook(7)
    h = torch.ones(1, 5, 48)  # (1, N, H) rmpad-shaped activation

    sid_a = torch.tensor([0, 0, 1, 1, 2])
    pos_a = torch.tensor([0, 1, 0, 1, 0])
    masker.set_context(global_step=11, sample_ids=sid_a, position_ids=pos_a)
    out_a = hook(nn.Identity(), (), h)

    perm = torch.tensor([3, 0, 4, 1, 2])
    masker.set_context(global_step=11, sample_ids=sid_a[perm], position_ids=pos_a[perm])
    out_b = hook(nn.Identity(), (), h)

    assert torch.equal(out_a[0, perm, :], out_b[0])


# --------------------------------------------------------------------------- #
# value-independence + in-graph form
# --------------------------------------------------------------------------- #
def test_mask_independent_of_activation_values():
    masker = ActivationMasker(p=0.9, base_seed=7, pp_size=8)
    hook = masker._make_hook(3)
    shape = (2, 8, 32)

    _set_ctx(masker, 2, 8)
    out_zeros = hook(nn.Identity(), (), torch.zeros(shape))
    _set_ctx(masker, 2, 8)
    out_rand = hook(nn.Identity(), (), (h_rand := torch.randn(shape)))

    sid, pos = _ids_for(2, 8)
    mask = prf_token_mask(
        sid, pos, layer_idx=3, global_step=0, base_seed=7, hidden_size=32, p=0.9, device=CPU, dtype=torch.float32
    ).view(shape)
    assert torch.equal(out_rand, h_rand * mask)
    assert torch.equal(out_zeros, torch.zeros(shape) * mask)


def test_no_forward_rescale():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    _set_ctx(masker, 2, 4)
    out = masker._make_hook(3)(nn.Identity(), (), torch.full((2, 4, 16), 2.0))
    nonzero = out[out != 0]
    assert torch.allclose(nonzero, torch.full_like(nonzero, 2.0))


def test_mask_is_in_graph():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    _set_ctx(masker, 2, 4)
    h = torch.randn(2, 4, 16, requires_grad=True)
    masker._make_hook(3)(nn.Identity(), (), h).sum().backward()
    assert h.grad is not None


def test_tuple_output_first_element_masked():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    _set_ctx(masker, 2, 4)
    extra = torch.randn(2, 4, 16)
    out = masker._make_hook(3)(nn.Identity(), (), (torch.randn(2, 4, 16), extra))
    assert isinstance(out, tuple)
    assert torch.equal(out[1], extra)


def test_missing_context_raises():
    """The per-element mask has no positional fallback; firing without ids fails loud."""
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    with pytest.raises(RuntimeError):
        masker._make_hook(3)(nn.Identity(), (), torch.randn(2, 4, 16))


def test_token_axis_mismatch_raises():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    _set_ctx(masker, 2, 4)  # 8 tokens
    with pytest.raises(RuntimeError):
        masker._make_hook(3)(nn.Identity(), (), torch.randn(3, 4, 16))  # 12 tokens


# --------------------------------------------------------------------------- #
# decoder-layer discovery + hook lifecycle on a toy model
# --------------------------------------------------------------------------- #
class _ToyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


class _ToyDecoder(nn.Module):
    def __init__(self, num_layers=16, d=32):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(d) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_find_decoder_layers():
    layers = find_decoder_layers(_ToyDecoder(num_layers=16, d=32))
    assert layers is not None and len(layers) == 16


def test_register_installs_hooks_on_boundaries_only():
    model = _ToyDecoder(num_layers=16, d=32)
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.register(model)
    assert masker.boundary_indices == [1, 3, 5, 7, 9, 11, 13]
    assert masker.is_registered
    for i, layer in enumerate(model.layers):
        assert (len(layer._forward_hooks) > 0) == (i in masker.boundary_indices)
    masker.unregister()
    assert not masker.is_registered
    for layer in model.layers:
        assert len(layer._forward_hooks) == 0


def test_unregister_leaves_forward_clean():
    torch.manual_seed(0)
    model = _ToyDecoder(num_layers=16, d=32)
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    _set_ctx(masker, 2, 4)

    x = torch.randn(2, 4, 32)
    masker.register(model)
    _set_ctx(masker, 2, 4)
    out_masked = model(x)
    masker.unregister()
    out_clean = model(x)
    assert not torch.allclose(out_masked, out_clean)
    assert torch.allclose(out_clean, model(x))


def test_register_is_idempotent():
    model = _ToyDecoder(num_layers=16, d=32)
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.register(model)
    n_handles = len(masker._handles)
    masker.register(model)
    assert len(masker._handles) == n_handles
    masker.unregister()


def test_mask_applications_counter_increments():
    class _FakeState:
        def __init__(self):
            self.mask_applications = 0

    state = _FakeState()
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8, state=state)
    hook = masker._make_hook(3)
    _set_ctx(masker, 2, 4)
    hook(nn.Identity(), (), torch.randn(2, 4, 16))
    hook(nn.Identity(), (), torch.randn(2, 4, 16))
    assert state.mask_applications == 2


# =========================================================================== #
# path-isolation / contamination guard
# =========================================================================== #
from types import SimpleNamespace  # noqa: E402

from verl.workers.comm_eff.state import (  # noqa: E402
    MASK_ELIGIBLE_TAGS,
    OLD_LOGPROB_TAG,
    PATH_TAGS,
    REF_LOGPROB_TAG,
    TRAIN_TAG,
    CommEffState,
    comm_eff_metrics,
    mask_eligible_tags,
    maybe_build_comm_eff_state,
)


def _make_enabled_state(p=0.95, pp_size=8, seed=0, mask_recompute=False, mask_reference=False):
    cfg = SimpleNamespace(
        enabled=True,
        mask=SimpleNamespace(
            enabled=True,
            p=p,
            seed=seed,
            pp_size=pp_size,
            mask_recompute=mask_recompute,
            mask_reference=mask_reference,
        ),
    )
    state = maybe_build_comm_eff_state(cfg)
    assert isinstance(state, CommEffState)
    model = _ToyDecoder(num_layers=16, d=32)
    state.build(model)
    assert state.masker is not None
    return state, model


def test_state_defaults_path_tag_none_and_all_paths_zero():
    state, _ = _make_enabled_state()
    assert state.path_tag is None
    for tag in PATH_TAGS:
        assert state.mask_applications_by_path[tag] == 0


def test_set_path_tag_rejects_unknown_tag():
    state, _ = _make_enabled_state()
    with pytest.raises(ValueError):
        state.set_path_tag("not_a_real_path")
    state.set_path_tag(None)
    assert state.path_tag is None


def test_mask_fires_only_on_train_tag():
    state, model = _make_enabled_state()
    state.mask_active = True
    state.set_path_tag(TRAIN_TAG)
    _set_ctx(state.masker, 2, 4)
    state.masker.register(model)
    _set_ctx(state.masker, 2, 4)
    model(torch.randn(2, 4, 32))
    state.masker.unregister()

    assert state.mask_applications > 0
    assert state.mask_applications_by_path[TRAIN_TAG] == state.mask_applications
    for tag in PATH_TAGS:
        if tag != TRAIN_TAG:
            assert state.mask_applications_by_path[tag] == 0


@pytest.mark.parametrize("bad_tag", [t for t in PATH_TAGS if t != TRAIN_TAG] + [None])
def test_mask_hook_asserts_on_non_train_path(bad_tag):
    state, _ = _make_enabled_state()
    state.set_path_tag(bad_tag)
    _set_ctx(state.masker, 2, 4)
    hook = state.masker._make_hook(3)
    with pytest.raises(AssertionError):
        hook(nn.Identity(), (), torch.randn(2, 4, 32))
    for tag in PATH_TAGS:
        assert state.mask_applications_by_path[tag] == 0


def test_per_path_counters_surface_in_metrics_by_key_prefix():
    state, model = _make_enabled_state()
    state.mask_active = True
    state.set_path_tag(TRAIN_TAG)
    state.masker.register(model)
    _set_ctx(state.masker, 2, 4)
    model(torch.randn(2, 4, 32))
    state.masker.unregister()

    metrics = comm_eff_metrics(state)
    for tag in PATH_TAGS:
        assert f"comm_eff/mask_applications/{tag}" in metrics
    assert metrics[f"comm_eff/mask_applications/{TRAIN_TAG}"] > 0
    nonzero = [k for k, v in metrics.items() if k.startswith("comm_eff/mask_applications/") and v > 0]
    assert nonzero == [f"comm_eff/mask_applications/{TRAIN_TAG}"]


def test_clean_forward_after_train_on_inactive_tag_is_unmasked():
    torch.manual_seed(0)
    state, model = _make_enabled_state()
    x = torch.randn(2, 4, 32)
    ref = model(x)

    state.mask_active = True
    state.set_path_tag(TRAIN_TAG)
    state.masker.register(model)
    _set_ctx(state.masker, 2, 4)
    model(x)
    state.masker.unregister()
    state.mask_active = False

    state.set_path_tag("old_logprob")
    assert torch.allclose(model(x), ref)


def test_mask_eligible_tags_default_is_singleton_train():
    assert MASK_ELIGIBLE_TAGS == frozenset({TRAIN_TAG})
    assert OLD_LOGPROB_TAG == "old_logprob"


def test_mask_eligible_tags_widens_only_when_recompute_true():
    s_default, _ = _make_enabled_state(mask_recompute=False)
    s_recomp, _ = _make_enabled_state(mask_recompute=True)
    assert mask_eligible_tags(s_default) == frozenset({TRAIN_TAG})
    assert mask_eligible_tags(s_recomp) == frozenset({TRAIN_TAG, OLD_LOGPROB_TAG})
    assert mask_eligible_tags(None) == frozenset({TRAIN_TAG})


def test_mask_eligible_tags_widens_for_reference_only_when_mask_reference_true():
    """ref_logprob is eligible iff mask_reference; independent of mask_recompute."""
    s_noref, _ = _make_enabled_state(mask_reference=False)
    s_ref, _ = _make_enabled_state(mask_reference=True)
    assert REF_LOGPROB_TAG not in mask_eligible_tags(s_noref)
    assert mask_eligible_tags(s_ref) == frozenset({TRAIN_TAG, REF_LOGPROB_TAG})

    # The two widenings compose independently.
    s_both, _ = _make_enabled_state(mask_recompute=True, mask_reference=True)
    assert mask_eligible_tags(s_both) == frozenset({TRAIN_TAG, OLD_LOGPROB_TAG, REF_LOGPROB_TAG})


def test_mask_hook_fires_on_ref_logprob_only_when_mask_reference():
    """The hook accepts ref_logprob iff mask_reference is set; otherwise it asserts."""
    # mask_reference=True: ref_logprob is accepted and counted.
    state, _ = _make_enabled_state(mask_reference=True)
    hook = state.masker._make_hook(3)
    _set_ctx(state.masker, 2, 4)
    state.set_path_tag(REF_LOGPROB_TAG)
    assert torch.is_tensor(hook(nn.Identity(), (), torch.randn(2, 4, 32)))
    assert state.mask_applications_by_path[REF_LOGPROB_TAG] == 1

    # mask_reference=False: ref_logprob is rejected (confinement guard).
    state2, _ = _make_enabled_state(mask_reference=False)
    hook2 = state2.masker._make_hook(3)
    _set_ctx(state2.masker, 2, 4)
    state2.set_path_tag(REF_LOGPROB_TAG)
    with pytest.raises(AssertionError):
        hook2(nn.Identity(), (), torch.randn(2, 4, 32))
    assert state2.mask_applications_by_path[REF_LOGPROB_TAG] == 0


def test_mask_recompute_path_tag_eligibility():
    """The hook fires on eligible tags only; every other tag (and None) raises."""
    state, _ = _make_enabled_state(mask_recompute=True)
    hook = state.masker._make_hook(3)
    _set_ctx(state.masker, 2, 4)

    state.set_path_tag(TRAIN_TAG)
    assert torch.is_tensor(hook(nn.Identity(), (), torch.randn(2, 4, 32)))
    state.set_path_tag(OLD_LOGPROB_TAG)
    assert torch.is_tensor(hook(nn.Identity(), (), torch.randn(2, 4, 32)))
    assert state.mask_applications_by_path[TRAIN_TAG] == 1
    assert state.mask_applications_by_path[OLD_LOGPROB_TAG] == 1

    for tag in PATH_TAGS:
        if tag in (TRAIN_TAG, OLD_LOGPROB_TAG):
            continue
        state.set_path_tag(tag)
        with pytest.raises(AssertionError):
            hook(nn.Identity(), (), torch.randn(2, 4, 32))
    state.set_path_tag(None)
    with pytest.raises(AssertionError):
        hook(nn.Identity(), (), torch.randn(2, 4, 32))

    # mask_recompute=False: old_logprob now rejected too.
    state2, _ = _make_enabled_state(mask_recompute=False)
    hook2 = state2.masker._make_hook(3)
    _set_ctx(state2.masker, 2, 4)
    state2.set_path_tag(TRAIN_TAG)
    hook2(nn.Identity(), (), torch.randn(2, 4, 32))
    assert state2.mask_applications_by_path[TRAIN_TAG] == 1
    state2.set_path_tag(OLD_LOGPROB_TAG)
    with pytest.raises(AssertionError):
        hook2(nn.Identity(), (), torch.randn(2, 4, 32))
    assert state2.mask_applications_by_path[OLD_LOGPROB_TAG] == 0


# --------------------------------------------------------------------------- #
# checkpoint contamination guard
# --------------------------------------------------------------------------- #
class _ToyLM(nn.Module):
    def __init__(self, num_layers=16, d=32, vocab=11):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(d) for _ in range(num_layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.log_softmax(self.head(x), dim=-1)


def test_logprob_equal_mask_on_vs_off_when_tag_inactive():
    torch.manual_seed(1)
    model = _ToyLM(num_layers=16, d=32)
    x = torch.randn(3, 5, 32)
    lp_off = model(x)

    state = maybe_build_comm_eff_state(
        SimpleNamespace(enabled=True, mask=SimpleNamespace(enabled=True, p=0.95, seed=0, pp_size=8))
    )
    state.build(model)
    state.set_path_tag("old_logprob")  # inactive path, no hooks registered
    lp_on = model(x)

    assert torch.allclose(lp_on, lp_off, rtol=1e-6, atol=1e-6)
    assert state.mask_applications_by_path["old_logprob"] == 0
    assert state.mask_applications == 0


def test_checkpoint_guard_passes_on_clean_state_dict():
    from verl.utils.checkpoint.fsdp_checkpoint_manager import _assert_no_comm_eff_state

    _assert_no_comm_eff_state(_ToyLM().state_dict())


def test_checkpoint_guard_rejects_leaked_comm_eff_state():
    from verl.utils.checkpoint.fsdp_checkpoint_manager import _assert_no_comm_eff_state

    leaked = _ToyLM().state_dict()
    leaked["layers.3.comm_eff_mask_buffer"] = torch.zeros(4)
    with pytest.raises(AssertionError):
        _assert_no_comm_eff_state(leaked)


def test_masker_is_hooks_only_no_params_or_buffers():
    model = _ToyLM(num_layers=16, d=32)
    keys_before = set(model.state_dict().keys())
    masker = ActivationMasker(p=0.95, base_seed=0, pp_size=8)
    masker.register(model)
    keys_after = set(model.state_dict().keys())
    masker.unregister()
    assert keys_before == keys_after
