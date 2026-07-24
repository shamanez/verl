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

"""Dense-view probe confinement (issue #93 I3): tag None silences every codec.

The trainer's probe passes stamp path tag ``None`` (the anchor's dense
precedent) on the worker state. These tests pin the hook-side contract that
``None`` can never activate a codec: it is in no eligibility set, both engine
gates reject it regardless of ``compression_active``, and a codec hook that
somehow fires under it asserts.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.state import (
    OLD_LOGPROB_TAG,
    PATH_TAGS,
    REF_LOGPROB_TAG,
    TRAIN_TAG,
    CommEffState,
    mask_eligible_tags,
    maybe_build_comm_eff_state,
)
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _ToyBlock(nn.Module):
    def forward(self, h):
        return h


class _ToyDecoder(nn.Module):
    def __init__(self, num_layers=16, d=32):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock() for _ in range(num_layers)])

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


def _mask_state(mask_recompute=False, mask_reference=False):
    cfg = SimpleNamespace(
        enabled=True,
        mask=SimpleNamespace(
            enabled=True,
            p=0.95,
            seed=0,
            pp_size=8,
            mask_recompute=mask_recompute,
            mask_reference=mask_reference,
        ),
    )
    state = maybe_build_comm_eff_state(cfg)
    assert isinstance(state, CommEffState)
    state.build(_ToyDecoder())
    assert state.masker is not None
    return state


def _quant_state(mask_recompute=False, mask_reference=False):
    cfg = SimpleNamespace(
        enabled=True,
        compression_type="sr_quant",
        mask=SimpleNamespace(
            enabled=False,
            p=0.95,
            seed=0,
            pp_size=8,
            mask_recompute=mask_recompute,
            mask_reference=mask_reference,
        ),
        quant=SimpleNamespace(bits=1, block_size=0, rounding="sr"),
    )
    state = maybe_build_comm_eff_state(cfg)
    assert isinstance(state, CommEffState)
    state.build(_ToyDecoder())
    assert state.quantizer is not None
    return state


def test_probe_tag_none_is_legal_and_in_no_eligibility_set():
    state = _mask_state(mask_recompute=True, mask_reference=True)
    state.set_path_tag(None)  # must not raise: None is the dense measurement view
    assert state.path_tag is None
    # Even the widest eligibility set never contains None.
    assert None not in mask_eligible_tags(state)
    assert mask_eligible_tags(state) == frozenset({TRAIN_TAG, OLD_LOGPROB_TAG, REF_LOGPROB_TAG})
    assert None not in PATH_TAGS


@pytest.mark.parametrize("codec", ["prf_mask", "sr_quant"])
@pytest.mark.parametrize("forward_only", [True, False])
def test_engine_mask_gate_rejects_tag_none(codec, forward_only):
    """The engine's boundary-codec gate must reject the probe view even in the
    worst case: codec built, widest eligibility, compression_active stuck True."""
    make = _mask_state if codec == "prf_mask" else _quant_state
    state = make(mask_recompute=True, mask_reference=True)
    state.compression_active = True
    state.set_path_tag(None)
    engine = SimpleNamespace(_comm_eff_state=state)
    assert FSDPEngine._comm_eff_mask_active(engine, forward_only) is False
    # Sanity: the same state DOES activate on its eligible tags.
    state.set_path_tag(OLD_LOGPROB_TAG)
    assert FSDPEngine._comm_eff_mask_active(engine, True) is True


@pytest.mark.parametrize("forward_only", [True, False])
def test_engine_powersgd_gate_rejects_tag_none(forward_only):
    ps_cfg = SimpleNamespace(compress_recompute=True, compress_reference=True)
    state = SimpleNamespace(
        enabled=True,
        powersgd=SimpleNamespace(
            compress_recompute=True,
            reference_basis_ready=lambda: True,
            fast_q_bootstrap_needed=lambda: False,
        ),
        compression_active=True,
        path_tag=None,
        config=SimpleNamespace(powersgd=ps_cfg),
    )
    engine = SimpleNamespace(_comm_eff_state=state)
    assert FSDPEngine._comm_eff_powersgd_active(engine, forward_only) is False
    # Sanity: an eligible tag with the same state does activate.
    state.path_tag = OLD_LOGPROB_TAG
    assert FSDPEngine._comm_eff_powersgd_active(engine, True) is True


@pytest.mark.parametrize("codec", ["prf_mask", "sr_quant"])
def test_codec_hook_asserts_if_fired_under_probe_tag(codec):
    """Defense in depth: were a codec hook still registered during a probe
    pass, its confinement guard must refuse to mask/quantize (tag None)."""
    make = _mask_state if codec == "prf_mask" else _quant_state
    state = make(mask_recompute=True, mask_reference=True)
    codec_obj = state.masker if codec == "prf_mask" else state.quantizer
    b, s = 2, 4
    sample_ids = torch.arange(b).repeat_interleave(s)
    position_ids = torch.arange(s).repeat(b)
    codec_obj.set_context(global_step=0, sample_ids=sample_ids, position_ids=position_ids)
    state.set_path_tag(None)
    hook = codec_obj._make_hook(3)
    with pytest.raises(AssertionError):
        hook(nn.Identity(), (), torch.randn(b, s, 32))
    for tag in PATH_TAGS:
        assert state.mask_applications_by_path[tag] == 0
