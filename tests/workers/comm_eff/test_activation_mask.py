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


def _make_enabled_state(
    p=0.95,
    pp_size=8,
    seed=0,
    mask_recompute=False,
    mask_reference=False,
    exact_k=False,
    antithetic=False,
    p_by_boundary=None,
    frlr=False,
    frlr_rank=32,
    frlr_k=44,
    frlr_unbiased=False,
    frlr_q_cadence=1,
):
    cfg = SimpleNamespace(
        enabled=True,
        mask=SimpleNamespace(
            enabled=True,
            p=p,
            seed=seed,
            pp_size=pp_size,
            mask_recompute=mask_recompute,
            mask_reference=mask_reference,
            exact_k=exact_k,
            antithetic=antithetic,
            p_by_boundary=p_by_boundary if p_by_boundary is not None else [],
            frlr=frlr,
            frlr_rank=frlr_rank,
            frlr_k=frlr_k,
            frlr_unbiased=frlr_unbiased,
            frlr_q_cadence=frlr_q_cadence,
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


# =========================================================================== #
# issue #89 codec levers: off-path parity, exact-k, antithetic, per-boundary p
# =========================================================================== #
def test_offpath_parity_prf_token_mask_byte_identical():
    """All new lever flags OFF => prf_token_mask is byte-identical to baseline."""
    sid, pos = _ids_for(6, 12)
    kw = dict(layer_idx=3, global_step=7, base_seed=5, hidden_size=64, p=0.95, device=CPU, dtype=torch.float32)
    baseline = prf_token_mask(sid, pos, **kw)
    flags_off = prf_token_mask(sid, pos, exact_k=False, antithetic=False, **kw)
    assert torch.equal(baseline, flags_off)
    # and the default-off path still matches the documented scalar reference.
    sid2, pos2 = torch.tensor([5, 9, 0]), torch.tensor([0, 3, 7])
    m = prf_token_mask(
        sid2,
        pos2,
        layer_idx=2,
        global_step=4,
        base_seed=1,
        hidden_size=6,
        p=0.6,
        device=CPU,
        dtype=torch.float32,
        exact_k=False,
        antithetic=False,
    )
    for t in range(3):
        for ch in range(6):
            assert float(m[t, ch]) == _scalar_keep(int(sid2[t]), int(pos2[t]), ch, layer=2, step=4, seed=1, p=0.6)


def test_offpath_parity_hook_htilde_constant_rescale():
    """All levers OFF, rescale_mode=constant => h_tilde is exactly h*mask/(1-p)."""
    p = 0.95
    masker = ActivationMasker(p=p, base_seed=0, pp_size=8, rescale_mode="constant")
    _set_ctx(masker, 2, 8)
    h = torch.randn(2, 8, 64)
    out = masker._make_hook(3)(nn.Identity(), (), h)
    sid, pos = _ids_for(2, 8)
    mask = prf_token_mask(
        sid, pos, layer_idx=3, global_step=0, base_seed=0, hidden_size=64, p=p, device=CPU, dtype=torch.float32
    ).view(h.shape)
    assert torch.equal(out, h * mask * (1.0 / (1.0 - p)))


def test_exact_k_conserves_rate_exactly():
    """exact_k keeps EXACTLY round((1-p)*H) per token; mask_ratio == 1 - k/H."""
    p, H = 0.95, 256
    sid, pos = _ids_for(8, 16)  # 128 tokens
    m = prf_token_mask(
        sid,
        pos,
        layer_idx=1,
        global_step=3,
        base_seed=2,
        hidden_size=H,
        p=p,
        device=CPU,
        dtype=torch.float32,
        exact_k=True,
    )
    keep = round((1.0 - p) * H)  # 13
    assert torch.all(m.sum(dim=-1) == keep)
    assert set(m.unique().tolist()).issubset({0.0, 1.0})
    measured_zero_fraction = float(1.0 - m.mean().item())
    assert measured_zero_fraction == pytest.approx(1.0 - keep / H, abs=1e-6)
    assert abs(measured_zero_fraction - p) <= 0.5 / H + 1e-9  # exact up to k rounding


def test_exact_k_is_random_not_value_topk():
    """exact_k selection is by PRF hash (value-independent), never a value top-k."""
    masker = ActivationMasker(p=0.9, base_seed=7, pp_size=8, exact_k=True)  # rescale_mode auto->none
    hook = masker._make_hook(3)
    shape = (2, 8, 32)
    _set_ctx(masker, 2, 8)
    out_ones = hook(nn.Identity(), (), torch.ones(shape))
    _set_ctx(masker, 2, 8)
    out_rand = hook(nn.Identity(), (), torch.randn(shape))
    # the kept positions (nonzero) are identical regardless of activation values
    assert torch.equal(out_ones != 0, out_rand != 0)
    # exactly round((1-p)*H)=3 kept per token
    assert torch.all((out_ones != 0).sum(dim=-1) == round(0.1 * 32))


def test_antithetic_within_step_identity_and_cross_step_complement():
    """antithetic: identical within a step; exact bitwise complement across the pair at p=0.5."""
    sid, pos = _ids_for(8, 16)
    kw = dict(layer_idx=2, base_seed=3, hidden_size=64, p=0.5, device=CPU, dtype=torch.float32, antithetic=True)
    m0a = prf_token_mask(sid, pos, global_step=0, **kw)
    m0b = prf_token_mask(sid, pos, global_step=0, **kw)
    assert torch.equal(m0a, m0b)  # within-step identity (key has no forward tag)
    m1 = prf_token_mask(sid, pos, global_step=1, **kw)
    assert torch.equal(m1, 1.0 - m0a)  # antithetic complement across the pair
    m2 = prf_token_mask(sid, pos, global_step=2, **kw)
    assert not torch.equal(m2, m0a)  # next pair draws fresh


def test_antithetic_preserves_mask_ratio_and_disjoint_tails_at_p095():
    """antithetic keeps ~5% each step at p=0.95 (NOT a set complement flipping to 95%)."""
    sid, pos = _ids_for(8, 64)
    kw = dict(layer_idx=5, base_seed=1, hidden_size=256, p=0.95, device=CPU, dtype=torch.float32, antithetic=True)
    for step in (0, 1, 2, 3):
        m = prf_token_mask(sid, pos, global_step=step, **kw)
        assert abs(float(1.0 - m.mean().item()) - 0.95) <= 0.02
    m0 = prf_token_mask(sid, pos, global_step=0, **kw)
    m1 = prf_token_mask(sid, pos, global_step=1, **kw)
    assert float((m0 * m1).sum().item()) == 0.0  # antithetic tails are disjoint


def test_antithetic_within_step_identity_across_path_tags():
    """The antithetic mask is identical across the old / train / reference forwards in one step."""
    state, _ = _make_enabled_state(p=0.5, mask_recompute=True, mask_reference=True, antithetic=True)
    hook = state.masker._make_hook(3)
    h = torch.ones(2, 8, 32)
    outs = {}
    for tag in (TRAIN_TAG, OLD_LOGPROB_TAG, REF_LOGPROB_TAG):
        _set_ctx(state.masker, 2, 8, step=6)
        state.set_path_tag(tag)
        outs[tag] = hook(nn.Identity(), (), h.clone())
    assert torch.equal(outs[TRAIN_TAG], outs[OLD_LOGPROB_TAG])
    assert torch.equal(outs[TRAIN_TAG], outs[REF_LOGPROB_TAG])


def test_p_by_boundary_average_mask_ratio_in_band():
    """A 7-vector averaging 0.95 => aggregate comm_eff/mask_ratio in [0.94, 0.96]."""
    pbb = [0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]  # mean exactly 0.95
    assert sum(pbb) / len(pbb) == pytest.approx(0.95)
    model = _ToyDecoder(num_layers=16, d=512)
    masker = ActivationMasker(p=0.95, base_seed=0, pp_size=8, p_by_boundary=pbb)
    masker.register(model)
    assert masker.boundary_indices == [1, 3, 5, 7, 9, 11, 13]
    b, s = 8, 64  # 512 tokens x 512 channels per boundary
    _set_ctx(masker, b, s)
    model(torch.randn(b, s, 512))
    masker.unregister()
    ratios = masker.last_mask_ratio
    assert len(ratios) == 7
    aggregate = sum(ratios.values()) / len(ratios)
    assert 0.94 <= aggregate <= 0.96
    for i, idx in enumerate(masker.boundary_indices):
        assert abs(ratios[idx] - pbb[i]) <= 0.02  # each boundary tracks its own p


def test_p_by_boundary_length_mismatch_raises():
    """A p_by_boundary vector whose length != boundary count fails loudly at register()."""
    model = _ToyDecoder(num_layers=16, d=32)  # 7 boundaries
    masker = ActivationMasker(p=0.95, base_seed=0, pp_size=8, p_by_boundary=[0.95, 0.95])
    with pytest.raises(ValueError):
        masker.register(model)


def test_p_by_boundary_out_of_range_raises_at_construction():
    with pytest.raises(ValueError):
        ActivationMasker(p=0.95, base_seed=0, pp_size=8, p_by_boundary=[0.95, 1.5])


# =========================================================================== #
# issue #89 FRLR codec: fresh-residual low-rank ("32+44+1")
# =========================================================================== #
from verl.workers.comm_eff.powersgd_activation import init_basis, orthonormalize  # noqa: E402


def _frlr_masker(r=8, k=12, unbiased=False, seed=0, state=None, q_cadence=1, anchor_owns_q=False):
    return ActivationMasker(
        p=0.95,
        base_seed=seed,
        pp_size=8,
        frlr=True,
        frlr_rank=r,
        frlr_k=k,
        frlr_unbiased=unbiased,
        frlr_q_cadence=q_cadence,
        anchor_owns_q=anchor_owns_q,
        state=state,
    )


def _frlr_j_mask(b, s, *, layer, step, seed, H, k):
    """The FRLR residual subset J as the masker draws it (PRF-fresh exact-k)."""
    sid, pos = _ids_for(b, s)
    return prf_token_mask(
        sid,
        pos,
        layer_idx=layer,
        global_step=step,
        base_seed=seed,
        hidden_size=H,
        p=0.95,
        device=CPU,
        dtype=torch.float32,
        exact_k=True,
        exact_keep=k,
    )


def test_frlr_offpath_parity_byte_identical():
    """frlr=false (new defaults present) leaves the baseline PRF byte-identical."""
    torch.manual_seed(0)
    h = torch.randn(2, 8, 64)
    base = ActivationMasker(p=0.95, base_seed=3, pp_size=8)
    _set_ctx(base, 2, 8, step=5)
    out_base = base._make_hook(3)(nn.Identity(), (), h)
    off = ActivationMasker(p=0.95, base_seed=3, pp_size=8, frlr=False, frlr_rank=32, frlr_k=44, frlr_unbiased=False)
    _set_ctx(off, 2, 8, step=5)
    out_off = off._make_hook(3)(nn.Identity(), (), h)
    assert torch.equal(out_base, out_off)
    # exact_keep=None keeps prf_token_mask byte-identical too.
    sid, pos = _ids_for(2, 8)
    kw = dict(layer_idx=3, global_step=5, base_seed=3, hidden_size=64, p=0.95, device=CPU, dtype=torch.float32)
    assert torch.equal(prf_token_mask(sid, pos, **kw), prf_token_mask(sid, pos, exact_keep=None, **kw))


def test_frlr_payload_exactness_77_of_1536():
    """J has exactly k=44 kept channels per token; total kept accounting is 77."""
    torch.manual_seed(0)
    H, r, k = 1536, 32, 44
    masker = _frlr_masker(r=r, k=k)
    hook = masker._make_hook(3)
    _set_ctx(masker, 2, 4, step=1)
    hook(nn.Identity(), (), torch.randn(2, 4, H))
    mj = _frlr_j_mask(2, 4, layer=3, step=1, seed=0, H=H, k=k)
    assert torch.all(mj.sum(dim=-1) == k)
    assert set(mj.unique().tolist()).issubset({0.0, 1.0})
    # 32 (core y) + 44 (res_J) + 1 (norm scalar) = 77 values/token.
    assert masker.frlr_payload_per_token == float(r + k + 1)
    assert masker.last_mask_ratio[3] == pytest.approx(1.0 - (r + k + 1) / H, abs=1e-9)


def test_frlr_within_step_identity_across_path_tags():
    """h_hat is identical across the old/train/reference forwards at a fixed step."""
    torch.manual_seed(0)
    state, _ = _make_enabled_state(mask_recompute=True, mask_reference=True, frlr=True, frlr_rank=8, frlr_k=12)
    hook = state.masker._make_hook(3)
    h = torch.randn(2, 8, 32)
    outs = {}
    for tag in (TRAIN_TAG, OLD_LOGPROB_TAG, REF_LOGPROB_TAG):
        _set_ctx(state.masker, 2, 8, step=6)
        state.set_path_tag(tag)
        outs[tag] = hook(nn.Identity(), (), h.clone())
    # Same step => same frozen Q, same J, same gamma: bitwise identical.
    assert torch.equal(outs[TRAIN_TAG], outs[OLD_LOGPROB_TAG])
    assert torch.equal(outs[TRAIN_TAG], outs[REF_LOGPROB_TAG])


def test_frlr_recompute_replay_deterministic_and_sketch_deduped():
    """Same context (grad-ckpt recompute) => bitwise identical h_hat, one sketch fold."""
    torch.manual_seed(0)
    H = 32
    masker = _frlr_masker(r=4, k=8)
    hook = masker._make_hook(3)
    h = torch.randn(2, 4, H)
    _set_ctx(masker, 2, 4, step=0)
    out_a = hook(nn.Identity(), (), h)  # original forward
    sketch_after_first = masker._frlr_sketch[3].clone()
    out_b = hook(nn.Identity(), (), h)  # recompute: same context, no set_context
    assert torch.equal(out_a, out_b)
    assert torch.equal(masker._frlr_sketch[3], sketch_after_first)


def test_frlr_cross_step_fresh_j_and_q_refresh():
    """J is fresh across steps; Q refreshes at the step boundary from the sketch."""
    torch.manual_seed(0)
    H, r, k = 32, 4, 8
    masker = _frlr_masker(r=r, k=k)
    hook = masker._make_hook(3)
    h = torch.randn(2, 8, H)
    _set_ctx(masker, 2, 8, step=0)
    out0 = hook(nn.Identity(), (), h)
    q0 = masker._frlr_basis[3].clone()
    assert masker.frlr_q_refreshes == 0
    _set_ctx(masker, 2, 8, step=1)
    out1 = hook(nn.Identity(), (), h)
    j0 = _frlr_j_mask(2, 8, layer=3, step=0, seed=0, H=H, k=k)
    j1 = _frlr_j_mask(2, 8, layer=3, step=1, seed=0, H=H, k=k)
    assert not torch.equal(j0, j1)
    assert not torch.equal(out0, out1)
    # Q refreshed once, from step 0's activation sketch (warm start).
    assert masker.frlr_q_refreshes == 1
    assert not torch.allclose(masker._frlr_basis[3], q0)


def test_frlr_gamma_norm_match_capped_and_detached():
    """gamma == ||res||/max(||res_J||,eps) clamped to H/k, and carries NO grad."""
    torch.manual_seed(0)
    H, r, k = 32, 4, 8
    masker = _frlr_masker(r=r, k=k)
    hook = masker._make_hook(3)
    h = torch.randn(2, 4, H, requires_grad=True)
    _set_ctx(masker, 2, 4, step=2)
    out = hook(nn.Identity(), (), h)
    # Reference reconstruction with gamma as an explicit constant.
    q = masker._frlr_basis[3]
    m = h.detach().reshape(-1, H)
    low = (m @ q) @ q.t()
    res = m - low
    mj = _frlr_j_mask(2, 4, layer=3, step=2, seed=0, H=H, k=k)
    res_j = res * mj
    cap = float(H) / float(k)
    gamma = (res.norm(dim=-1, keepdim=True) / res_j.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=cap)
    assert torch.all(gamma <= cap + 1e-6)
    expected = (low + gamma * res_j).reshape(2, 4, H)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6)
    # Detachment: backward equals the adjoint of the gamma-CONSTANT linear map.
    out.sum().backward()
    h2 = h.detach().clone().requires_grad_(True)
    m2 = h2.reshape(-1, H)
    low2 = (m2 @ q) @ q.t()
    res2 = m2 - low2
    (low2 + gamma * (res2 * mj)).sum().backward()
    assert torch.allclose(h.grad, h2.grad.reshape(2, 4, H), rtol=1e-5, atol=1e-6)


def test_frlr_gamma_cap_engages_on_adversarial_token():
    """A token whose residual avoids J would blow up unbounded; the H/k cap holds it."""
    torch.manual_seed(1)
    H, r, k = 32, 4, 8
    masker = _frlr_masker(r=r, k=k)
    # Q0 (deterministic seeded bootstrap) and J (pure PRF) are computable up front.
    q = init_basis(hidden_size=H, rank=r, base_seed=0, layer_idx=3)
    mj = _frlr_j_mask(1, 1, layer=3, step=0, seed=0, H=H, k=k)  # token (sid=0, pos=0)
    nonj = mj[0] == 0
    E = torch.eye(H)[:, nonj]  # off-J coordinate subspace (H, H-k)
    # v: supported off J AND orthogonal to span(Q) => res == v with res_J == 0.
    _, _, Vh = torch.linalg.svd(q.t() @ E)
    v = E @ Vh[-1]
    j_idx = int(torch.nonzero(mj[0]).reshape(-1)[0])
    h = (v + 1e-3 * torch.eye(H)[j_idx]).reshape(1, 1, H)
    hook = masker._make_hook(3)
    _set_ctx(masker, 1, 1, step=0)
    out = hook(nn.Identity(), (), h)
    m = h.reshape(-1, H)
    low = (m @ q) @ q.t()
    res = m - low
    res_j = res * mj
    cap = float(H) / float(k)
    gamma_uncapped = float(res.norm() / res_j.norm().clamp_min(1e-8))
    assert gamma_uncapped > cap  # the crafted token genuinely exceeds the cap
    expected = (low + cap * res_j).reshape(1, 1, H)
    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-6)
    assert torch.isfinite(out).all()


def test_frlr_full_rank_recovers_h():
    """frlr_rank=H => res ~ 0 and h_hat == h within float tolerance."""
    torch.manual_seed(2)
    H = 64
    masker = _frlr_masker(r=H, k=8)
    hook = masker._make_hook(3)
    h = torch.randn(2, 4, H)
    _set_ctx(masker, 2, 4, step=0)
    out = hook(nn.Identity(), (), h)
    assert torch.allclose(out, h, rtol=1e-4, atol=1e-4)


def test_frlr_unbiased_mean_reconstruction_approaches_h():
    """Unbiased mode: E over PRF key draws of h_hat approaches h (fixed h, fixed Q)."""
    torch.manual_seed(3)
    H, r, k = 32, 4, 8
    masker = _frlr_masker(r=r, k=k, unbiased=True)
    hook = masker._make_hook(3)
    h = torch.randn(1, 6, H)
    total = torch.zeros_like(h)
    n_steps = 2000
    with torch.no_grad():  # no sketch under no_grad => Q stays frozen at Q0
        for step in range(n_steps):
            _set_ctx(masker, 1, 6, step=step)
            total += hook(nn.Identity(), (), h)
    assert masker.frlr_q_refreshes == 0
    mean = total / n_steps
    rel_err = float((mean - h).norm() / h.norm())
    assert rel_err < 0.10


def test_frlr_mask_ratio_reports_0949_at_real_geometry():
    """comm_eff/mask_ratio ~ 0.9499 for the 32+44+1 payload over H=1536."""
    torch.manual_seed(0)
    state, _ = _make_enabled_state(frlr=True, frlr_rank=32, frlr_k=44)
    hook = state.masker._make_hook(3)
    state.set_path_tag(TRAIN_TAG)
    _set_ctx(state.masker, 2, 4, step=1)
    hook(nn.Identity(), (), torch.randn(2, 4, 1536))
    metrics = comm_eff_metrics(state)
    ratio = metrics["comm_eff/mask_ratio"]
    assert ratio == pytest.approx(1.0 - 77.0 / 1536.0, abs=1e-6)  # ~0.9499
    assert 0.94 <= ratio <= 0.96
    assert metrics["comm_eff/logical_pp_bytes_prf"] == 77.0
    assert "comm_eff/frlr_q_refreshes" in metrics


def test_frlr_mutually_exclusive_and_bounds_raise():
    with pytest.raises(ValueError):
        ActivationMasker(p=0.95, base_seed=0, pp_size=8, frlr=True, exact_k=True)
    with pytest.raises(ValueError):
        ActivationMasker(p=0.95, base_seed=0, pp_size=8, frlr=True, antithetic=True)
    with pytest.raises(ValueError):
        ActivationMasker(p=0.95, base_seed=0, pp_size=8, frlr=True, p_by_boundary=[0.95] * 7)
    with pytest.raises(ValueError):
        ActivationMasker(p=0.95, base_seed=0, pp_size=8, frlr=True, rescale_mode="constant")
    with pytest.raises(ValueError):
        ActivationMasker(p=0.95, base_seed=0, pp_size=8, frlr=True, frlr_rank=0)
    with pytest.raises(ValueError):
        ActivationMasker(p=0.95, base_seed=0, pp_size=8, frlr=True, frlr_k=0)
    # exact_keep misuse fails loud.
    sid, pos = _ids_for(2, 4)
    with pytest.raises(ValueError):
        prf_token_mask(
            sid,
            pos,
            layer_idx=0,
            global_step=0,
            base_seed=0,
            hidden_size=16,
            p=0.5,
            device=CPU,
            dtype=torch.float32,
            exact_keep=4,
        )


def test_frlr_config_validation():
    from verl.workers.config.comm_eff import CommEffConfig, CommEffMaskConfig

    CommEffConfig(mask=CommEffMaskConfig(frlr=True))  # defaults compose
    with pytest.raises(ValueError):
        CommEffConfig(mask=CommEffMaskConfig(frlr=True, exact_k=True))
    with pytest.raises(ValueError):
        CommEffConfig(mask=CommEffMaskConfig(frlr=True, antithetic=True))
    with pytest.raises(ValueError):
        CommEffConfig(mask=CommEffMaskConfig(frlr=True, rescale=True))
    with pytest.raises(ValueError):
        CommEffConfig(mask=CommEffMaskConfig(frlr=True, rescale_mode="rms_match"))
    with pytest.raises(ValueError):
        CommEffConfig(mask=CommEffMaskConfig(frlr_rank=0))
    with pytest.raises(ValueError):
        CommEffConfig(mask=CommEffMaskConfig(frlr_k=0))


# =========================================================================== #
# issue #89 FRLR slow-Q lever: frlr_q_cadence (frozen Q between refreshes)
# =========================================================================== #
def test_frlr_q_cadence_1_bit_identical_to_every_step_refresh():
    """frlr_q_cadence=1 (the default) reproduces the original every-step refresh bitwise."""
    torch.manual_seed(0)
    H = 32
    default = _frlr_masker(r=4, k=8)  # no q_cadence kwarg: pre-lever construction
    explicit = _frlr_masker(r=4, k=8, q_cadence=1)  # the new lever at its default
    hs = [torch.randn(2, 4, H) for _ in range(4)]
    for step, h in enumerate(hs):
        _set_ctx(default, 2, 4, step=step)
        out_d = default._make_hook(3)(nn.Identity(), (), h)
        _set_ctx(explicit, 2, 4, step=step)
        out_e = explicit._make_hook(3)(nn.Identity(), (), h)
        assert torch.equal(out_d, out_e)
        assert torch.equal(default._frlr_basis[3], explicit._frlr_basis[3])
    # Every step boundary refreshed Q, exactly as before the lever existed.
    assert default.frlr_q_refreshes == 3
    assert explicit.frlr_q_refreshes == 3
    # A cadence below 1 fails loud, in the masker and in the config dataclass.
    with pytest.raises(ValueError):
        _frlr_masker(r=4, k=8, q_cadence=0)
    from verl.workers.config.comm_eff import CommEffConfig, CommEffMaskConfig

    with pytest.raises(ValueError):
        CommEffConfig(mask=CommEffMaskConfig(frlr_q_cadence=0))


def test_frlr_q_cadence_frozen_window_then_refresh():
    """cadence=5: Q is bitwise identical across steps t..t+4 and refreshes at t+5."""
    torch.manual_seed(0)
    H = 32
    masker = _frlr_masker(r=4, k=8, q_cadence=5)
    hook = masker._make_hook(3)
    _set_ctx(masker, 2, 4, step=0)
    hook(nn.Identity(), (), torch.randn(2, 4, H))
    q0 = masker._frlr_basis[3].clone()
    for step in range(1, 5):
        _set_ctx(masker, 2, 4, step=step)
        hook(nn.Identity(), (), torch.randn(2, 4, H))
        assert torch.equal(masker._frlr_basis[3], q0)  # bitwise frozen window
        assert masker.frlr_q_refreshes == 0
    _set_ctx(masker, 2, 4, step=5)
    hook(nn.Identity(), (), torch.randn(2, 4, H))
    assert masker.frlr_q_refreshes == 1
    assert not torch.equal(masker._frlr_basis[3], q0)
    # The next window is frozen again until step 10.
    q5 = masker._frlr_basis[3].clone()
    for step in range(6, 10):
        _set_ctx(masker, 2, 4, step=step)
        hook(nn.Identity(), (), torch.randn(2, 4, H))
        assert torch.equal(masker._frlr_basis[3], q5)
        assert masker.frlr_q_refreshes == 1
    _set_ctx(masker, 2, 4, step=10)
    hook(nn.Identity(), (), torch.randn(2, 4, H))
    assert masker.frlr_q_refreshes == 2


def test_frlr_q_cadence_within_step_identity_across_path_tags():
    """cadence>1 keeps h_hat identical across the old/train/reference forwards.

    Holds on the bootstrap step, inside the frozen window, AND on the refresh
    step itself (the refresh happens once, at the first fire of that step).
    """
    torch.manual_seed(0)
    state, _ = _make_enabled_state(
        mask_recompute=True, mask_reference=True, frlr=True, frlr_rank=8, frlr_k=12, frlr_q_cadence=3
    )
    hook = state.masker._make_hook(3)
    h = torch.randn(2, 8, 32)
    for step in (0, 1, 2, 3):  # 0 bootstrap, 1-2 frozen window, 3 refresh step
        outs = {}
        for tag in (TRAIN_TAG, OLD_LOGPROB_TAG, REF_LOGPROB_TAG):
            _set_ctx(state.masker, 2, 8, step=step)
            state.set_path_tag(tag)
            outs[tag] = hook(nn.Identity(), (), h.clone())
        assert torch.equal(outs[TRAIN_TAG], outs[OLD_LOGPROB_TAG])
        assert torch.equal(outs[TRAIN_TAG], outs[REF_LOGPROB_TAG])
    assert state.masker.frlr_q_refreshes == 1  # exactly the step-3 refresh


def test_frlr_q_cadence_sketch_accumulates_over_window():
    """The refresh consumes the FULL frozen window's sketch, not just one step's.

    With Q frozen at Q0 for steps 0..4 the sketch must equal
    sum_s h_s^T (h_s Q0), and the step-5 refresh must orthonormalize THAT
    accumulation (differing from what a refresh from step 0's contribution
    alone, i.e. an every-step refresh at t+1, would have produced).
    """
    torch.manual_seed(0)
    H, r, k = 32, 4, 8
    masker = _frlr_masker(r=r, k=k, q_cadence=5)
    hook = masker._make_hook(3)
    q0 = init_basis(hidden_size=H, rank=r, base_seed=0, layer_idx=3)
    hs = [torch.randn(2, 4, H) for _ in range(5)]
    expected = torch.zeros(H, r)
    for step, h in enumerate(hs):
        _set_ctx(masker, 2, 4, step=step)
        hook(nn.Identity(), (), h)
        m32 = h.reshape(-1, H)
        expected += m32.t() @ (m32 @ q0)  # Q frozen at Q0 over the whole window
    assert torch.allclose(masker._frlr_sketch[3], expected, rtol=1e-5, atol=1e-5)
    _set_ctx(masker, 2, 4, step=5)
    hook(nn.Identity(), (), torch.randn(2, 4, H))
    q_window = orthonormalize(expected)
    m0 = hs[0].reshape(-1, H)
    q_one_step = orthonormalize(m0.t() @ (m0 @ q0))
    assert torch.allclose(masker._frlr_basis[3], q_window, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(masker._frlr_basis[3], q_one_step, rtol=1e-4, atol=1e-5)


def test_frlr_q_refreshes_metric_counts_only_actual_refreshes():
    """comm_eff/frlr_q_refreshes reflects the slow cadence, not step boundaries."""
    torch.manual_seed(0)
    state, _ = _make_enabled_state(frlr=True, frlr_rank=4, frlr_k=8, frlr_q_cadence=3)
    hook = state.masker._make_hook(3)
    state.set_path_tag(TRAIN_TAG)
    for step in range(8):  # bootstrap at 0; refreshes fire at steps 3 and 6 only
        _set_ctx(state.masker, 2, 4, step=step)
        hook(nn.Identity(), (), torch.randn(2, 4, 32))
    metrics = comm_eff_metrics(state)
    assert metrics["comm_eff/frlr_q_refreshes"] == 2
    assert state.masker.frlr_q_refreshes == 2  # cadence=1 would have logged 7


# =========================================================================== #
# issue #93: anchor-owned FRLR basis Q
#
# Operator instruction 2026-07-26: "Q update only in the anchor and only when it
# fires, like in normal powerSGD Q". The fast path must stop being a Q writer
# entirely: no sketch folding, no refresh. Both move onto the anchor's clean
# stale-weight forward, which is also where PowerSGD has always done it.
# =========================================================================== #
def test_frlr_anchor_owns_q_requires_frlr():
    """The plain PRF mask has no basis to own, so the combination is refused."""
    with pytest.raises(ValueError, match="requires frlr=true"):
        ActivationMasker(p=0.95, base_seed=0, pp_size=8, anchor_owns_q=True)


def test_frlr_anchor_owns_q_fast_path_never_folds_or_refreshes():
    """Under anchor ownership the fast path writes NOTHING to the basis state."""
    torch.manual_seed(0)
    H, r, k = 32, 4, 8
    masker = _frlr_masker(r=r, k=k, anchor_owns_q=True)
    hook = masker._make_hook(3)
    _set_ctx(masker, 2, 8, step=0)
    hook(nn.Identity(), (), torch.randn(2, 8, H))
    q0 = masker._frlr_basis[3].clone()
    assert masker._frlr_sketch == {}, "fast path folded into the sketch under anchor ownership"
    # Ten more fast steps: Q must be BITWISE frozen the whole way.
    for step in range(1, 11):
        _set_ctx(masker, 2, 8, step=step)
        hook(nn.Identity(), (), torch.randn(2, 8, H))
    assert masker._frlr_sketch == {}
    assert masker.frlr_q_refreshes == 0
    assert torch.equal(masker._frlr_basis[3], q0)


def test_frlr_anchor_sketch_mode_returns_raw_and_harvests():
    """In harvest mode the hook is a pass-through that folds the RAW activation.

    The anchor forward must stay dense: its gradient is G_anchor. So the hook
    returns h byte-identically while V += h^T(hQ) lands.
    """
    torch.manual_seed(0)
    H, r, k = 32, 4, 8
    masker = _frlr_masker(r=r, k=k, anchor_owns_q=True)
    hook = masker._make_hook(3)
    h = torch.randn(2, 8, H)
    _set_ctx(masker, 2, 8, step=0)
    masker.set_anchor_sketch_mode(True)
    out = hook(nn.Identity(), (), h)
    assert out is h, "anchor harvest must return the activation object untouched"
    q0 = masker._frlr_basis[3]
    m = h.reshape(-1, H).to(torch.float32)
    assert torch.allclose(masker._frlr_sketch[3], m.t() @ (m @ q0), rtol=1e-5, atol=1e-6)


def test_frlr_anchor_harvest_needs_no_prf_key_and_ignores_path_tag():
    """Harvest runs with path_tag=None and no per-token ids, like the anchor pass.

    The live-path hook asserts on both. Placing the harvest branch ahead of them
    is what lets the anchor's clone forward fire at all.
    """
    torch.manual_seed(0)
    H = 32
    state, _ = _make_enabled_state(frlr=True, frlr_rank=4, frlr_k=8)
    masker = state.masker
    masker.anchor_owns_q = True
    hook = masker._make_hook(3)
    state.set_path_tag(None)
    masker._sample_ids = None
    masker._position_ids = None
    masker.set_anchor_sketch_mode(True)
    out = hook(nn.Identity(), (), torch.randn(2, 8, H))
    assert out.shape == (2, 8, H)
    assert 3 in masker._frlr_sketch


def test_frlr_anchor_harvest_dedupes_per_forward_generation():
    """Grad-checkpoint recompute reuses the generation, so it folds once."""
    torch.manual_seed(0)
    H = 32
    masker = _frlr_masker(r=4, k=8, anchor_owns_q=True)
    hook = masker._make_hook(3)
    h = torch.randn(2, 8, H)
    _set_ctx(masker, 2, 8, step=0)
    masker.set_anchor_sketch_mode(True)
    hook(nn.Identity(), (), h)
    first = masker._frlr_sketch[3].clone()
    hook(nn.Identity(), (), h)  # recompute: no set_context between
    assert torch.equal(masker._frlr_sketch[3], first)
    # A new micro-batch bumps the generation, so it DOES fold again.
    _set_ctx(masker, 2, 8, step=0)
    hook(nn.Identity(), (), h)
    assert not torch.equal(masker._frlr_sketch[3], first)


def test_frlr_anchor_update_basis_refreshes_from_harvest_and_clears():
    """Q <- orth(V) at the anchor fire, exactly the warm-started power step."""
    torch.manual_seed(0)
    H, r, k = 32, 4, 8
    masker = _frlr_masker(r=r, k=k, anchor_owns_q=True)
    hook = masker._make_hook(3)
    masker.register(_ToyDecoder(num_layers=16, d=H))
    assert 3 in masker.boundary_indices
    _set_ctx(masker, 2, 8, step=0)
    masker.set_anchor_sketch_mode(True)
    h = torch.randn(2, 8, H)
    hook(nn.Identity(), (), h)
    q0 = masker._frlr_basis[3].clone()
    v = masker._frlr_sketch[3].clone()
    masker.set_anchor_sketch_mode(False)
    assert masker.anchor_update_basis() is True
    assert masker.frlr_q_refreshes == 1
    assert torch.allclose(masker._frlr_basis[3], orthonormalize(v), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(masker._frlr_basis[3], q0)
    # The sketch is consumed, so the next window starts clean.
    assert masker._frlr_sketch == {}
    assert masker.anchor_update_basis() is False


def test_frlr_anchor_update_basis_refuses_when_fast_path_owns_q():
    """Two Q writers is the failure mode this guard exists to prevent."""
    masker = _frlr_masker(r=4, k=8, anchor_owns_q=False)
    masker.register(_ToyDecoder(num_layers=16, d=32))
    with pytest.raises(RuntimeError, match="anchor_owns_q=false"):
        masker.anchor_update_basis()


def test_frlr_anchor_ownership_leaves_fast_path_codec_bit_identical():
    """a7/a8 regression guard: the reconstruction itself must not change.

    Anchor ownership changes WHEN Q moves, never the transform. With the same
    frozen basis the two configurations must agree bitwise.
    """
    torch.manual_seed(0)
    H, r, k = 32, 4, 8
    fast = _frlr_masker(r=r, k=k, seed=5, anchor_owns_q=False)
    owned = _frlr_masker(r=r, k=k, seed=5, anchor_owns_q=True)
    h = torch.randn(2, 8, H)
    for m in (fast, owned):
        _set_ctx(m, 2, 8, step=0)
    out_fast = fast._make_hook(3)(nn.Identity(), (), h)
    out_owned = owned._make_hook(3)(nn.Identity(), (), h)
    assert torch.equal(out_fast, out_owned)


def test_config_allows_anchor_owned_q_only_for_frlr():
    """comm_eff.py's prf_mask/owns_q veto is relaxed for FRLR ONLY (issue #93).

    The old check rejected the combination outright on the premise that the PRF
    mask "has no PowerSGD basis Q for the anchor to own". That premise is false
    when frlr=true: FRLR carries a per-boundary basis.
    """
    from verl.workers.config.comm_eff import CommEffAnchorConfig, CommEffConfig, CommEffMaskConfig

    # Plain PRF exact-k: still refused, and the message says why.
    with pytest.raises(ValueError, match="unless.*mask.frlr=true"):
        CommEffConfig(
            enabled=True,
            compression_type="prf_mask",
            mask=CommEffMaskConfig(enabled=True, p=0.95, exact_k=True, rescale_mode="constant"),
            anchor=CommEffAnchorConfig(owns_q=True),
        )
    # FRLR + anchor-owned Q: accepted.
    CommEffConfig(
        enabled=True,
        compression_type="prf_mask",
        mask=CommEffMaskConfig(enabled=True, frlr=True, frlr_rank=48, frlr_k=28),
        anchor=CommEffAnchorConfig(owns_q=True, enabled=True),
    )
    # ... but only with an anchor to do the updating.
    with pytest.raises(ValueError, match="requires anchor.enabled=true"):
        CommEffConfig(
            enabled=True,
            compression_type="prf_mask",
            mask=CommEffMaskConfig(enabled=True, frlr=True, frlr_rank=48, frlr_k=28),
            anchor=CommEffAnchorConfig(owns_q=True, enabled=False),
        )
    # a7/a8 regression: FRLR with the fast path owning Q is untouched.
    CommEffConfig(
        enabled=True,
        compression_type="prf_mask",
        mask=CommEffMaskConfig(enabled=True, frlr=True, frlr_rank=48, frlr_k=28),
        anchor=CommEffAnchorConfig(owns_q=False),
    )


def test_state_plumbs_anchor_owns_q_into_the_masker():
    """state.build() must carry anchor.owns_q onto the FRLR codec, and only there."""
    from verl.workers.comm_eff.state import maybe_build_comm_eff_state

    def _build(frlr, owns_q):
        cfg = SimpleNamespace(
            enabled=True,
            anchor=SimpleNamespace(owns_q=owns_q),
            mask=SimpleNamespace(
                enabled=True,
                p=0.95,
                seed=0,
                pp_size=8,
                mask_recompute=False,
                mask_reference=False,
                exact_k=False,
                antithetic=False,
                p_by_boundary=[],
                frlr=frlr,
                frlr_rank=48,
                frlr_k=28,
                frlr_unbiased=False,
                frlr_q_cadence=1,
            ),
        )
        st = maybe_build_comm_eff_state(cfg)
        st.build(_ToyDecoder(num_layers=16, d=32))
        return st.masker

    assert _build(frlr=True, owns_q=True).anchor_owns_q is True
    assert _build(frlr=True, owns_q=False).anchor_owns_q is False
    # A plain-mask config never resolves to anchor ownership, even if the anchor
    # sub-config says owns_q=true (which is its dataclass DEFAULT).
    assert _build(frlr=False, owns_q=True).anchor_owns_q is False
