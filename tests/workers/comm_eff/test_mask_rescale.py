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

"""Magnitude-preservation (inverted-dropout rescale) knob.

``rescale=False`` (default) writes the raw product ``h * mask``; ``rescale=True``
applies ``h * mask / (1 - p)`` so kept elements are scaled up to hold
``E[h_tilde] = h``. These CPU tests assert the form and the magnitude property;
they say nothing about training stability.
"""

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import ActivationMasker


def _set_ctx(masker, b, s, step=0):
    sid = torch.arange(b).repeat_interleave(s)
    pos = torch.arange(s).repeat(b)
    masker.set_context(global_step=step, sample_ids=sid, position_ids=pos)


def test_default_no_rescale_keeps_kept_elements_unchanged():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8)
    assert masker.rescale is False
    assert masker._rescale_gain == 1.0
    _set_ctx(masker, 2, 4)
    out = masker._make_hook(3)(nn.Identity(), (), torch.full((2, 4, 16), 2.0))
    nonzero = out[out != 0]
    assert torch.allclose(nonzero, torch.full_like(nonzero, 2.0))


def test_rescale_scales_kept_elements_by_inverse_keep_prob():
    p = 0.9
    masker = ActivationMasker(p=p, base_seed=0, pp_size=8, rescale=True)
    assert masker.rescale is True
    assert masker._rescale_gain == pytest.approx(1.0 / (1.0 - p))
    _set_ctx(masker, 2, 4)
    out = masker._make_hook(3)(nn.Identity(), (), torch.full((2, 4, 16), 2.0))
    nonzero = out[out != 0]
    assert torch.allclose(nonzero, torch.full_like(nonzero, 2.0 / (1.0 - p)))


def test_rescale_preserves_expected_magnitude():
    p = 0.9
    h = torch.ones(16, 64, 256)

    m_off = ActivationMasker(p=p, base_seed=1, pp_size=8, rescale=False)
    _set_ctx(m_off, 16, 64)
    out_off = m_off._make_hook(3)(nn.Identity(), (), h)

    m_on = ActivationMasker(p=p, base_seed=1, pp_size=8, rescale=True)
    _set_ctx(m_on, 16, 64)
    out_on = m_on._make_hook(3)(nn.Identity(), (), h)

    assert out_off.mean().item() == pytest.approx(1.0 - p, abs=0.02)
    assert out_on.mean().item() == pytest.approx(1.0, abs=0.05)


def test_rescale_same_zero_pattern_as_no_rescale():
    p = 0.9
    h = torch.randn(4, 8, 32)

    m_off = ActivationMasker(p=p, base_seed=7, pp_size=8, rescale=False)
    _set_ctx(m_off, 4, 8, step=3)
    out_off = m_off._make_hook(5)(nn.Identity(), (), h)

    m_on = ActivationMasker(p=p, base_seed=7, pp_size=8, rescale=True)
    _set_ctx(m_on, 4, 8, step=3)
    out_on = m_on._make_hook(5)(nn.Identity(), (), h)

    assert torch.equal((out_off == 0), (out_on == 0))
    keep = out_off != 0
    assert torch.allclose(out_on[keep], out_off[keep] / (1.0 - p))


def test_rescale_mask_ratio_metric_unaffected():
    p = 0.9
    masker = ActivationMasker(p=p, base_seed=0, pp_size=8, rescale=True)
    _set_ctx(masker, 8, 64)
    masker._make_hook(3)(nn.Identity(), (), torch.randn(8, 64, 256))
    assert abs(masker.last_mask_ratio[3] - p) <= 0.02


def test_rescale_is_in_graph():
    masker = ActivationMasker(p=0.9, base_seed=0, pp_size=8, rescale=True)
    _set_ctx(masker, 2, 4)
    h = torch.randn(2, 4, 16, requires_grad=True)
    masker._make_hook(3)(nn.Identity(), (), h).sum().backward()
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
