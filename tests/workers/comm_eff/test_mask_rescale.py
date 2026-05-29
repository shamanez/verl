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

"""EXP-14 magnitude-preservation (inverted-dropout rescale) knob.

THE SECOND SUSPECT (after the substep-RNG fix was refuted): the explosion may be
the mask's BIAS, not importance-sampling inconsistency. ``h_tilde = h * mask``
with NO ``1/(1-p)`` rescale drops ~90% of boundary-block activations at p=0.9, so
the residual-stream RMS at those positions collapses to ``sqrt(1-p) ≈ 0.316×`` —
a large distribution shift from the weights' training regime, which produces
large gradients. clean_cadence works precisely because its clean step runs the
UNMASKED (full-magnitude) network.

``comm_eff.mask.rescale`` (default False, a flagged DESIGN-CHANGE candidate)
applies inverted-dropout ``h_tilde = h * mask / (1 - p)`` so kept activations are
scaled up to hold ``E[h_tilde] = h``. These CPU tests assert the form and the
magnitude-preservation property; they do NOT assert anything about training
stability (that is what test2_cellD measures on the box).
"""

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import ActivationMasker


# --------------------------------------------------------------------------- #
# default (rescale off): pure h * mask, kept elements unchanged
# --------------------------------------------------------------------------- #
def test_default_no_rescale_keeps_kept_elements_unchanged():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    assert masker.rescale is False
    assert masker._rescale_gain == 1.0
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    h = torch.full((2, 4, 16), 2.0)
    out = hook(nn.Identity(), (), h)
    nonzero = out[out != 0]
    # kept elements equal h exactly (2.0), no scale-up.
    assert torch.allclose(nonzero, torch.full_like(nonzero, 2.0))


# --------------------------------------------------------------------------- #
# rescale on: kept elements scaled by 1/(1-p)
# --------------------------------------------------------------------------- #
def test_rescale_scales_kept_elements_by_inverse_keep_prob():
    p = 0.9
    masker = ActivationMasker(p=p, base_seed=0, pp_size=8, rescale=True)
    assert masker.rescale is True
    assert masker._rescale_gain == pytest.approx(1.0 / (1.0 - p))
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    h = torch.full((2, 4, 16), 2.0)
    out = hook(nn.Identity(), (), h)
    nonzero = out[out != 0]
    # kept elements equal h * 1/(1-p) = 2.0 * 10 = 20.0
    assert torch.allclose(nonzero, torch.full_like(nonzero, 2.0 / (1.0 - p)))


def test_rescale_preserves_expected_magnitude():
    """E[h_tilde] = h: over a large tensor the masked+rescaled mean ~= the
    unmasked mean (inverted-dropout is unbiased in expectation), whereas the
    no-rescale variant collapses the mean by (1-p)."""
    p = 0.9
    shape = (16, 64, 256)
    h = torch.ones(shape)

    m_off = ActivationMasker(p=p, base_seed=1, pp_size=8, rescale=False)
    m_off.set_context(global_step=0, substep=0, seq_shard=0)
    out_off = m_off._make_hook(3)(nn.Identity(), (), h)

    m_on = ActivationMasker(p=p, base_seed=1, pp_size=8, rescale=True)
    m_on.set_context(global_step=0, substep=0, seq_shard=0)
    out_on = m_on._make_hook(3)(nn.Identity(), (), h)

    # no-rescale mean collapses to ~ (1-p) = 0.1
    assert out_off.mean().item() == pytest.approx(1.0 - p, abs=0.02)
    # rescaled mean is preserved at ~ 1.0
    assert out_on.mean().item() == pytest.approx(1.0, abs=0.05)


def test_rescale_same_zero_pattern_as_no_rescale():
    """Rescale changes only the MAGNITUDE of kept elements — the zero/keep
    pattern (which the IS-ratio / boundary structure depends on) is identical to
    the no-rescale mask for the same key."""
    p = 0.9
    shape = (4, 8, 32)
    h = torch.randn(shape)

    m_off = ActivationMasker(p=p, base_seed=7, pp_size=8, rescale=False)
    m_off.set_context(global_step=3, substep=0, seq_shard=0)
    out_off = m_off._make_hook(5)(nn.Identity(), (), h)

    m_on = ActivationMasker(p=p, base_seed=7, pp_size=8, rescale=True)
    m_on.set_context(global_step=3, substep=0, seq_shard=0)
    out_on = m_on._make_hook(5)(nn.Identity(), (), h)

    # Same positions zeroed.
    assert torch.equal((out_off == 0), (out_on == 0))
    # And the rescaled kept values equal the no-rescale kept values * gain.
    keep = out_off != 0
    assert torch.allclose(out_on[keep], out_off[keep] / (1.0 - p))


def test_rescale_mask_ratio_metric_unaffected():
    """last_mask_ratio reads the binary mask, not h_tilde, so the measured zeroed
    fraction still tracks p regardless of the rescale gain."""
    p = 0.9
    masker = ActivationMasker(p=p, base_seed=0, pp_size=8, rescale=True)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    hook(nn.Identity(), (), torch.randn(8, 64, 256))
    assert abs(masker.last_mask_ratio[3] - p) <= 0.02


def test_rescale_is_in_graph():
    """The rescaled multiply stays autograd-tracked."""
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8, rescale=True)
    masker.set_context(global_step=0, substep=0, seq_shard=0)
    hook = masker._make_hook(3)
    h = torch.randn(2, 4, 16, requires_grad=True)
    out = hook(nn.Identity(), (), h)
    out.sum().backward()
    assert h.grad is not None


# --------------------------------------------------------------------------- #
# config plumbing: rescale threads from config through build() into the masker
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


@pytest.mark.parametrize("rescale", [True, False])
def test_build_threads_rescale_into_masker(rescale):
    cfg = SimpleNamespace(
        enabled=True,
        mask=SimpleNamespace(enabled=True, p=0.9, seed=0, pp_size=8, rescale=rescale),
    )
    state = maybe_build_comm_eff_state(cfg)
    state.build(_ToyDecoder(num_layers=16, d=32))
    assert state.masker is not None
    assert state.masker.rescale is rescale
    expected_gain = (1.0 / (1.0 - 0.9)) if rescale else 1.0
    assert state.masker._rescale_gain == pytest.approx(expected_gain)
