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

"""Unit tests for the comm_eff pipeline-boundary activation masker (EXP-5).

These tests cover the masking-correctness properties the EXP-5 plan requires
codex-verify to gate on, none of which need a GPU:

* boundary indices == [1,3,5,7,9,11,13] for L=16 / pp_size=8, derived (not hardcoded);
* PRF determinism: same key -> same mask, across calls;
* value-independence: the mask depends only on the PRF key + shape, never on
  the activation values;
* measured mask ratio (zeroed fraction) tracks the configured p within tolerance;
* in-graph form h_tilde = h * mask with NO 1/(1-p) rescale;
* hook lifecycle: register installs hooks on boundaries only; unregister removes
  them so a later forward is clean.
"""

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import (
    ActivationMasker,
    decoder_boundary_indices,
    find_decoder_layers,
    prf_mask,
)


# --------------------------------------------------------------------------- #
# boundary partition
# --------------------------------------------------------------------------- #
def test_boundary_indices_L16_pp8():
    """The spec's canonical example: L=16 / pp_size=8 -> [1,3,5,7,9,11,13]."""
    assert decoder_boundary_indices(16, 8) == [1, 3, 5, 7, 9, 11, 13]


def test_boundary_indices_excludes_final_shard():
    """The final shard's last block (the model's last decoder block) is never masked."""
    idx = decoder_boundary_indices(16, 8)
    assert 15 not in idx  # final block excluded
    assert len(idx) == 7  # pp_size - 1 boundaries


def test_boundary_indices_uneven_partition():
    """Uneven L/pp_size: shards are near-even, larger shards come first."""
    # L=10, pp_size=4 -> shard lens [3,3,2,2] -> last idx [2,5,7,9] -> drop 9 -> [2,5,7]
    assert decoder_boundary_indices(10, 4) == [2, 5, 7]


def test_boundary_indices_pp_size_one_is_empty():
    assert decoder_boundary_indices(16, 1) == []


def test_boundary_indices_pp_capped_at_num_layers():
    # pp_size > L collapses to one block per shard; last shard dropped.
    assert decoder_boundary_indices(4, 8) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# PRF determinism + value-independence
# --------------------------------------------------------------------------- #
def test_prf_same_key_same_mask():
    shape = (2, 8, 32)
    key = (3, 1, 0, 0, 32, 7)
    m1 = prf_mask(shape, key, 0.9, device=torch.device("cpu"), dtype=torch.float32)
    m2 = prf_mask(shape, key, 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(m1, m2)


def test_prf_different_key_different_mask():
    shape = (2, 8, 32)
    a = prf_mask(shape, (3, 1, 0, 0, 32, 7), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    # different substep component -> different mask
    b = prf_mask(shape, (3, 2, 0, 0, 32, 7), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert not torch.equal(a, b)


def test_prf_mask_is_binary():
    m = prf_mask((4, 16, 64), (1, 0, 0, 0, 64, 0), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    uniq = set(m.unique().tolist())
    assert uniq.issubset({0.0, 1.0})


def test_mask_independent_of_activation_values():
    """The mask must depend only on the PRF key + shape, never on h's values."""
    masker = ActivationMasker(p=0.9, base_seed=7, pp_size=8)
    layer_idx = 3
    hook = masker._make_hook(layer_idx)

    shape = (2, 8, 32)
    h_zeros = torch.zeros(shape)
    h_rand = torch.randn(shape)

    masker.set_context(global_step=0, substep=0, seq_shard=0)
    out_zeros = hook(nn.Identity(), (), h_zeros)
    masker.set_context(global_step=0, substep=0, seq_shard=0)  # same key again
    out_rand = hook(nn.Identity(), (), h_rand)

    # Re-derive the mask directly from the key and confirm both inputs were
    # multiplied by the SAME mask (value-independence).
    key = (layer_idx, 0, 0, 0, 32, 7)
    mask = prf_mask(shape, key, 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(out_rand, h_rand * mask)
    assert torch.equal(out_zeros, h_zeros * mask)


# --------------------------------------------------------------------------- #
# measured mask ratio tracks p
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [0.90, 0.95])
def test_mask_ratio_tracks_p(p):
    # large tensor so the empirical zeroed fraction concentrates near p
    shape = (8, 64, 256)
    key = (5, 0, 0, 0, 256, 1)
    m = prf_mask(shape, key, p, device=torch.device("cpu"), dtype=torch.float32)
    measured_zero_fraction = float(1.0 - m.mean().item())
    assert abs(measured_zero_fraction - p) <= 0.02


# --------------------------------------------------------------------------- #
# in-graph form: h_tilde = h * mask, no 1/(1-p) rescale, autograd-tracked
# --------------------------------------------------------------------------- #
def test_no_forward_rescale():
    """Kept elements must equal h exactly (no 1/(1-p) scale-up)."""
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    h = torch.full((2, 4, 16), 2.0)
    out = hook(nn.Identity(), (), h)
    # every nonzero output element equals exactly the input (2.0), not 2.0/(1-p)
    nonzero = out[out != 0]
    assert torch.allclose(nonzero, torch.full_like(nonzero, 2.0))


def test_mask_is_in_graph():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    h = torch.randn(2, 4, 16, requires_grad=True)
    out = hook(nn.Identity(), (), h)
    out.sum().backward()
    assert h.grad is not None  # gradient flows through the masked multiply


def test_tuple_output_first_element_masked():
    """HF decoder blocks return tuples; only the hidden state (elem 0) is masked."""
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    h = torch.randn(2, 4, 16)
    extra = torch.randn(2, 4, 16)
    out = hook(nn.Identity(), (), (h, extra))
    assert isinstance(out, tuple)
    assert torch.equal(out[1], extra)  # second element untouched


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
    model = _ToyDecoder(num_layers=16, d=32)
    layers = find_decoder_layers(model)
    assert layers is not None
    assert len(layers) == 16


def test_register_installs_hooks_on_boundaries_only():
    model = _ToyDecoder(num_layers=16, d=32)
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.register(model)
    assert masker.boundary_indices == [1, 3, 5, 7, 9, 11, 13]
    assert masker.is_registered
    # exactly the boundary blocks carry a forward hook
    for i, layer in enumerate(model.layers):
        has_hook = len(layer._forward_hooks) > 0
        assert has_hook == (i in masker.boundary_indices)
    masker.unregister()
    assert not masker.is_registered
    for layer in model.layers:
        assert len(layer._forward_hooks) == 0


def test_unregister_leaves_forward_clean():
    """After unregister, a forward sees no masking (every element preserved)."""
    torch.manual_seed(0)
    model = _ToyDecoder(num_layers=16, d=32)
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.set_context(global_step=0, substep=0, seq_shard=0)

    x = torch.randn(2, 4, 32)
    masker.register(model)
    out_masked = model(x)
    masker.unregister()
    out_clean = model(x)
    # the masked forward should differ from the clean forward (mask fired)
    assert not torch.allclose(out_masked, out_clean)
    # a second clean forward must reproduce the first clean forward exactly
    out_clean2 = model(x)
    assert torch.allclose(out_clean, out_clean2)


def test_register_is_idempotent():
    model = _ToyDecoder(num_layers=16, d=32)
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    masker.register(model)
    n_handles = len(masker._handles)
    masker.register(model)  # second call must not double-register
    assert len(masker._handles) == n_handles
    masker.unregister()


def test_mask_applications_counter_increments():
    """When a CommEffState is attached, each hook fire bumps mask_applications."""

    class _FakeState:
        def __init__(self):
            self.mask_applications = 0

    state = _FakeState()
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8, state=state)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    hook(nn.Identity(), (), torch.randn(2, 4, 16))
    hook(nn.Identity(), (), torch.randn(2, 4, 16))
    assert state.mask_applications == 2
