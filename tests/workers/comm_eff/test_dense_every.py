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

"""Issue #93: the periodic full-fidelity step (``mask.dense_every``).

The claim under test is narrow and load-bearing: on a step where
``global_step % dense_every == 0`` the codec must be bypassed so that BOTH the
forward activation and the boundary gradient are the true uncompressed values,
on every path, and the ordinary masked behaviour must be bit-identical to the
dense_every=0 baseline on every other step.
"""

import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import ActivationMasker

H = 64
NTOK = 8


class _Block(nn.Module):
    """Minimal stand-in for a decoder block: returns a tuple like the real one."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(H, H, bias=False)

    def forward(self, x):
        return (self.lin(x),)


class _Stack(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n)])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)[0]
        return x


class _State:
    """Just enough CommEffState surface for the masker's confinement guard."""

    path_tag = "train"
    mask_applications = 0

    def note_mask_application(self, *a, **k):
        self.__class__.mask_applications += 1


def _mk(dense_every, state=None):
    return ActivationMasker(
        p=0.95,
        base_seed=0,
        pp_size=2,
        rescale=True,
        rescale_mode="constant",
        exact_k=True,
        dense_every=dense_every,
        state=state,
    )


def _run(masker, model, step):
    x = torch.randn(NTOK, H, requires_grad=True)
    masker.set_context(
        global_step=step,
        sample_ids=torch.arange(NTOK),
        position_ids=torch.arange(NTOK),
    )
    out = model(x)
    out.sum().backward()
    return out.detach().clone(), x.grad.detach().clone()


def test_is_dense_step_arithmetic():
    m = _mk(50)
    assert not m.is_dense_step(0), "step 0 must never count as dense"
    assert not m.is_dense_step(49)
    assert m.is_dense_step(50)
    assert not m.is_dense_step(51)
    assert m.is_dense_step(100)
    off = _mk(0)
    for s in (0, 1, 50, 100, 1000):
        assert not off.is_dense_step(s), "dense_every=0 must disable the gate entirely"


def test_dense_step_matches_an_uncompressed_model_forward_and_backward():
    """On a dense step the masked model must equal a codec-free model exactly."""
    torch.manual_seed(0)
    model = _Stack()
    masker = _mk(50, state=_State())
    masker.register(model)

    torch.manual_seed(1)
    out_dense, grad_dense = _run(masker, model, step=50)

    masker.unregister()
    torch.manual_seed(1)
    out_nocodec, grad_nocodec = _run(_mk(0), model, step=50)

    assert torch.equal(out_dense, out_nocodec), "dense-step forward is not the raw forward"
    assert torch.equal(grad_dense, grad_nocodec), (
        "dense-step BACKWARD is not the raw backward: the hook still wrote the codec "
        "into the autograd graph"
    )


def test_non_dense_step_still_compresses():
    torch.manual_seed(0)
    model = _Stack()
    masker = _mk(50, state=_State())
    masker.register(model)

    torch.manual_seed(1)
    out_masked, grad_masked = _run(masker, model, step=49)
    masker.unregister()

    torch.manual_seed(1)
    out_raw, grad_raw = _run(_mk(0), model, step=49)

    assert not torch.equal(out_masked, out_raw), "step 49 should still be masked"
    assert not torch.equal(grad_masked, grad_raw), "step 49 gradient should still be compressed"


def test_dense_every_zero_is_bit_identical_to_the_baseline():
    """The knob must be inert when off, so the incumbent stays reproducible."""
    torch.manual_seed(0)
    model_a = _Stack()
    torch.manual_seed(0)
    model_b = _Stack()

    ma = _mk(0, state=_State())
    ma.register(model_a)
    torch.manual_seed(1)
    out_a, grad_a = _run(ma, model_a, step=50)
    ma.unregister()

    # dense_every unset at all (the pre-#93 default path)
    mb = ActivationMasker(
        p=0.95, base_seed=0, pp_size=2, rescale=True,
        rescale_mode="constant", exact_k=True, state=_State(),
    )
    mb.register(model_b)
    torch.manual_seed(1)
    out_b, grad_b = _run(mb, model_b, step=50)
    mb.unregister()

    assert torch.equal(out_a, out_b)
    assert torch.equal(grad_a, grad_b)


def test_counters_record_the_bypass_for_verification():
    torch.manual_seed(0)
    model = _Stack()
    state = _State()
    masker = _mk(50, state=state)
    masker.register(model)
    n_boundaries = len(masker.boundary_indices)
    assert n_boundaries > 0

    before = _State.mask_applications
    _run(masker, model, step=50)
    assert masker.dense_bypasses == n_boundaries, "every boundary must record a bypass"
    assert masker.dense_steps_fired == {50}
    assert _State.mask_applications == before, (
        "comm_eff/mask_applications must stay FLAT across a dense step; that flatness "
        "is the WandB-side proof the step really was uncompressed"
    )

    _run(masker, model, step=51)
    assert masker.dense_bypasses == n_boundaries, "step 51 must not bypass"
    assert _State.mask_applications > before, "step 51 must still fire the mask"
    masker.unregister()


def test_bypass_applies_on_every_path_not_just_train():
    """'No compression this step' has to mean every path, including reference."""
    torch.manual_seed(0)
    model = _Stack()
    state = _State()
    masker = _mk(50, state=state)
    masker.register(model)
    for tag in ("train", "old_logprob", "reference", None):
        state.path_tag = tag
        masker.dense_bypasses = 0
        _run(masker, model, step=50)
        assert masker.dense_bypasses == len(masker.boundary_indices), (
            f"dense-step bypass did not fire on path_tag={tag!r}"
        )
    masker.unregister()
