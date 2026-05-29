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

"""EXP-14 per-channel mask — the on-policy cross-pass-consistency default.

The substep fix (consistent_across_forwards) was refuted (cellC): the per-element
mask is drawn POSITIONALLY over each micro-batch's activation tensor, and
dynamic-bsz packs/length-sorts/shuffles tokens differently between the
old_logprob recompute (~21 micro-batches) and the actor-train forward (~28), so a
given token lands at a different flat index in the two phases and gets a
different mask — corrupting log pi_new/pi_old.

The PER-CHANNEL mask (the new default) drops the SAME hidden channels for EVERY
token at a boundary, keyed on (layer, global_step, seed) with NO token/packing
component. Being constant along the token axis, it is IDENTICAL across every
forward pass of one global update regardless of how the tokens are packed —
exact cross-pass consistency by construction. These CPU tests prove that and the
seed/step keying; the load-bearing test is the cross-packing invariance one
(simulating the train-vs-old_logprob packing difference).
"""

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import ActivationMasker, prf_channel_mask


# --------------------------------------------------------------------------- #
# prf_channel_mask: shape, binary, ratio, broadcast
# --------------------------------------------------------------------------- #
def test_channel_mask_shape_is_broadcastable():
    """The per-channel mask is (1, 1, hidden) so it broadcasts over the token
    and batch dims of h."""
    m = prf_channel_mask(64, (3, 1, 7), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert m.shape == (1, 1, 64)
    # broadcasts against both packed (1, nnz, H) and padded (B, S, H) activations
    assert (torch.randn(1, 17, 64) * m).shape == (1, 17, 64)
    assert (torch.randn(4, 8, 64) * m).shape == (4, 8, 64)


def test_channel_mask_is_binary():
    m = prf_channel_mask(128, (1, 0, 0), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert set(m.unique().tolist()).issubset({0.0, 1.0})


@pytest.mark.parametrize("p", [0.90, 0.95])
def test_channel_mask_ratio_tracks_p(p):
    # large hidden so the zeroed-channel fraction concentrates near p
    m = prf_channel_mask(4096, (5, 0, 1), p, device=torch.device("cpu"), dtype=torch.float32)
    zeroed_fraction = float(1.0 - m.mean().item())
    assert abs(zeroed_fraction - p) <= 0.02


# --------------------------------------------------------------------------- #
# (i) THE LOAD-BEARING TEST: cross-packing invariance at the same global_step
# --------------------------------------------------------------------------- #
def test_channel_mask_identical_across_different_packings_same_step():
    """Same token => same mask in two DIFFERENTLY shaped/ordered micro-batches at
    the same global_step. Simulates the train (28 micro) vs old_logprob (21 micro)
    packing difference: the per-channel mask must be byte-identical because it is
    keyed only on (layer, global_step, seed), constant along the token axis.

    We drive the masker hook on two activation tensors with DIFFERENT token
    counts AND a shuffled token order, and assert the per-channel keep/zero
    pattern (recoverable from any single token row) is identical.
    """
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8, granularity="channel")
    layer_idx = 3
    hook = masker._make_hook(layer_idx)
    hidden = 32

    # Phase A ("old_logprob"): 21 tokens, one ordering.
    masker.set_context(global_step=4, substep=0, seq_shard=0)
    hA = torch.ones(1, 21, hidden)
    outA = hook(nn.Identity(), (), hA)

    # Phase B ("train"): DIFFERENT token count (28) and a different substep/order
    # — the substep is bumped (as the engine would across forwards) to PROVE the
    # channel key ignores it. The per-channel mask must be unchanged.
    masker.set_context(global_step=4, substep=5, seq_shard=0)
    hB = torch.ones(1, 28, hidden)
    outB = hook(nn.Identity(), (), hB)

    # The kept-channel pattern is identical for every token; recover it from row 0
    # of each phase (h is all-ones so the masked output equals the mask).
    patA = (outA[0, 0] != 0)
    patB = (outB[0, 0] != 0)
    assert torch.equal(patA, patB), "per-channel mask differed across packings at the same step"

    # And EVERY token within a phase shares the same channel pattern (constant
    # along the token axis).
    assert torch.equal((outA[0] != 0), patA.unsqueeze(0).expand(21, hidden))
    assert torch.equal((outB[0] != 0), patB.unsqueeze(0).expand(28, hidden))


def test_channel_mask_invariant_to_substep_and_seq_shard():
    """Directly: changing substep / seq_shard does NOT change the channel mask
    (the token-axis key components are excluded under granularity=channel)."""
    masker = ActivationMasker(p=0.9, base_seed=7, pp_size=8, granularity="channel")
    masker.set_context(global_step=4, substep=0, seq_shard=0)
    out0 = masker._make_hook(3)(nn.Identity(), (), torch.ones(1, 5, 64))
    masker.set_context(global_step=4, substep=99, seq_shard=3)
    out1 = masker._make_hook(3)(nn.Identity(), (), torch.ones(1, 9, 64))
    assert torch.equal((out0[0, 0] != 0), (out1[0, 0] != 0))
    # equals the direct prf_channel_mask draw for the masker's channel key,
    # which is (layer_idx, global_step, hidden_size, base_seed).
    a = prf_channel_mask(64, (3, 4, 64, 7), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal((out0[0, 0] != 0), (a.view(-1) != 0))


# --------------------------------------------------------------------------- #
# (ii) different global_step => different channel set
# --------------------------------------------------------------------------- #
def test_channel_mask_changes_across_global_step():
    a = prf_channel_mask(256, (3, 1, 0), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    b = prf_channel_mask(256, (3, 2, 0), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert not torch.equal(a, b)


# --------------------------------------------------------------------------- #
# (iii) changing mask.seed => different channel set
# --------------------------------------------------------------------------- #
def test_channel_mask_changes_with_seed():
    a = prf_channel_mask(256, (3, 1, 0), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    b = prf_channel_mask(256, (3, 1, 42), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert not torch.equal(a, b)


def test_channel_mask_seed_is_first_class_through_masker():
    """mask.seed (base_seed) folds into the channel key: two maskers differing
    only in base_seed produce different channel patterns at the same step."""
    m0 = ActivationMasker(p=0.9, base_seed=0, pp_size=8, granularity="channel")
    m1 = ActivationMasker(p=0.9, base_seed=123, pp_size=8, granularity="channel")
    m0.set_context(global_step=3, substep=0, seq_shard=0)
    m1.set_context(global_step=3, substep=0, seq_shard=0)
    o0 = m0._make_hook(5)(nn.Identity(), (), torch.ones(1, 4, 64))
    o1 = m1._make_hook(5)(nn.Identity(), (), torch.ones(1, 4, 64))
    assert not torch.equal((o0[0, 0] != 0), (o1[0, 0] != 0))


# --------------------------------------------------------------------------- #
# per-boundary variation: different layers get different channel sets
# --------------------------------------------------------------------------- #
def test_channel_mask_varies_per_boundary_layer():
    a = prf_channel_mask(256, (1, 1, 0), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    b = prf_channel_mask(256, (3, 1, 0), 0.9, device=torch.device("cpu"), dtype=torch.float32)
    assert not torch.equal(a, b)


# --------------------------------------------------------------------------- #
# rescale composes with per-channel (the proven grad_norm fix is unchanged)
# --------------------------------------------------------------------------- #
def test_channel_mask_with_rescale_preserves_expected_magnitude():
    p = 0.9
    masker = ActivationMasker(p=p, base_seed=1, pp_size=8, granularity="channel", rescale=True)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    out = masker._make_hook(3)(nn.Identity(), (), torch.ones(1, 256, 4096))
    # E[h_tilde] = h: kept channels scaled by 1/(1-p), ~ (1-p) fraction kept => mean ~ 1.0
    assert out.mean().item() == pytest.approx(1.0, abs=0.06)
    # kept elements equal exactly 1/(1-p)
    nz = out[out != 0]
    assert torch.allclose(nz, torch.full_like(nz, 1.0 / (1.0 - p)))


def test_channel_mask_ratio_metric_tracks_p():
    """last_mask_ratio (1 - mask.mean()) reads p correctly for a (1,1,hidden) mask."""
    p = 0.9
    masker = ActivationMasker(p=p, base_seed=0, pp_size=8, granularity="channel")
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    masker._make_hook(3)(nn.Identity(), (), torch.randn(1, 100, 4096))
    assert abs(masker.last_mask_ratio[3] - p) <= 0.02


def test_channel_mask_is_in_graph():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8, granularity="channel")
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    h = torch.randn(1, 4, 16, requires_grad=True)
    out = masker._make_hook(3)(nn.Identity(), (), h)
    out.sum().backward()
    assert h.grad is not None


# --------------------------------------------------------------------------- #
# granularity is the default + contrast with element
# --------------------------------------------------------------------------- #
def test_default_granularity_is_channel():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    assert masker.granularity == "channel"


def test_element_granularity_is_row_positional():
    """Contrast: under granularity=element the mask is POSITIONAL along the token
    axis — different rows get different masks within ONE forward. So a token
    moved to a different row by dynamic-bsz reordering/shuffling between phases
    gets a different mask. (Under channel, every row is identical — proven
    above.) This is the per-element packing-dependence that per-channel fixes."""
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8, granularity="element")
    masker.set_context(global_step=4, substep=0, seq_shard=0)
    out = masker._make_hook(3)(nn.Identity(), (), torch.ones(1, 8, 128))
    # Row 0 and row 1 carry DIFFERENT masks (independent per-element draws) — the
    # mask depends on a token's ROW POSITION, not its identity.
    assert not torch.equal((out[0, 0] != 0), (out[0, 1] != 0))

    # And under channel granularity the same activation has IDENTICAL rows.
    masker_ch = ActivationMasker(p=0.9, base_seed=0, pp_size=8, granularity="channel")
    masker_ch.set_context(global_step=4, substep=0, seq_shard=0)
    out_ch = masker_ch._make_hook(3)(nn.Identity(), (), torch.ones(1, 8, 128))
    assert torch.equal((out_ch[0, 0] != 0), (out_ch[0, 1] != 0))


def test_bad_granularity_raises():
    with pytest.raises(ValueError):
        ActivationMasker(p=0.9, base_seed=0, pp_size=8, granularity="token")


# --------------------------------------------------------------------------- #
# config plumbing: granularity threads from config through build() into masker
# --------------------------------------------------------------------------- #
from types import SimpleNamespace  # noqa: E402

from verl.workers.comm_eff.state import maybe_build_comm_eff_state  # noqa: E402


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


@pytest.mark.parametrize("granularity", ["channel", "element"])
def test_build_threads_granularity_into_masker(granularity):
    cfg = SimpleNamespace(
        enabled=True,
        mask=SimpleNamespace(enabled=True, p=0.9, seed=0, pp_size=8, granularity=granularity),
    )
    state = maybe_build_comm_eff_state(cfg)
    state.build(_ToyDecoder(num_layers=16, d=32))
    assert state.masker is not None
    assert state.masker.granularity == granularity


def test_build_defaults_granularity_channel_when_absent():
    """A mask sub-config without the granularity attr defaults to channel (new
    default) via build()'s getattr fallback."""
    cfg = SimpleNamespace(
        enabled=True,
        mask=SimpleNamespace(enabled=True, p=0.9, seed=0, pp_size=8),  # no granularity
    )
    state = maybe_build_comm_eff_state(cfg)
    state.build(_ToyDecoder(num_layers=16, d=32))
    assert state.masker.granularity == "channel"
