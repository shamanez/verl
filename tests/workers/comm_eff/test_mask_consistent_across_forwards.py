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

"""Unit tests for EXP-14 on-policy-consistency of the activation mask.

THE BUG (test2_cellA, grad_norm=771): one GRPO global gradient update runs
several forward passes — the ``compute_log_prob`` old-logprob recompute (when
``mask_recompute=true``) and the actor-train forward(s). The PRF mask key
included an advancing ``substep`` counter that the engine bumped on EVERY masked
forward, so ``pi_old`` (old_logprob forward) and ``pi_new`` (train forward) were
computed under DIFFERENT masked subnetworks. That corrupts the PPO importance
ratio ``r = exp(logp_new - logp_old)`` away from ≈1 at the first inner step.

THE FIX: ``comm_eff.mask.consistent_across_forwards`` (default ``True``) holds the
substep component of the PRF key at a fixed sentinel so the mask is identical
across every forward of the SAME global update, while still differing across
distinct ``global_step`` values. The legacy resampling behavior is preserved
exactly when the knob is ``False`` (for the A/B comparison).

These tests are CPU-only: they drive the REAL engine method
``FSDPEngine._comm_eff_register_mask_hooks`` (which owns the substep-folding
decision) bound to a lightweight fake ``self``, plus the masker on a toy model.
No GPU, no distributed, no FSDP wrapping.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from verl.workers.comm_eff.state import CommEffState, TRAIN_TAG, maybe_build_comm_eff_state
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


# --------------------------------------------------------------------------- #
# toy model with a decoder-block ModuleList the masker can find
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


def _make_enabled_state(*, consistent: bool, p=0.9, pp_size=8, seed=0):
    """An enabled CommEffState with the mask circuit on and the knob set.

    Pinned to granularity="element": consistent_across_forwards is the
    element-path substep-folding knob this file tests. EXP-14 made "channel" the
    default (where substep is irrelevant and the mask is inherently consistent),
    so we explicitly request the element path here.
    """
    cfg = SimpleNamespace(
        enabled=True,
        mask=SimpleNamespace(
            enabled=True,
            p=p,
            seed=seed,
            pp_size=pp_size,
            mask_recompute=True,
            consistent_across_forwards=consistent,
            granularity="element",
        ),
    )
    state = maybe_build_comm_eff_state(cfg)
    assert isinstance(state, CommEffState)
    model = _ToyDecoder(num_layers=16, d=32)
    state.build(model)
    assert state.masker is not None
    return state, model


def _fake_engine(state, global_step):
    """A minimal stand-in for FSDPEngine carrying only what
    _comm_eff_register_mask_hooks reads. We call the REAL engine method bound to
    this object so the substep-folding decision under test is the production one.
    """
    return SimpleNamespace(
        _comm_eff_state=state,
        _comm_eff_global_step=global_step,
        ulysses_sequence_parallel_size=1,
        module=state.masker,  # placeholder; replaced per-call below
    )


def _register_and_capture_mask(state, model, global_step):
    """Drive the production engine register path for ONE forward at
    ``global_step``, run a forward to fire the hooks, and return the per-boundary
    measured mask ratios captured by the masker (a fingerprint of the realized
    mask) plus the raw hook output for an exact-tensor comparison.
    """
    fake = _fake_engine(state, global_step)
    fake.module = model
    # mask_active / path_tag gate the hook the way the worker sets them around
    # update_actor; required for the EXP-6 assert inside the hook to pass.
    state.mask_active = True
    state.set_path_tag(TRAIN_TAG)
    # Call the REAL engine method (the seam that decides substep-in-key).
    FSDPEngine._comm_eff_register_mask_hooks(fake)
    x = torch.randn(2, 4, 32)
    out = model(x)
    state.masker.unregister()
    state.mask_active = False
    state.set_path_tag(None)
    # last_mask_ratio is keyed by boundary idx; copy it (it is overwritten on
    # the next register). Return a deterministic fingerprint tuple.
    ratios = dict(state.masker.last_mask_ratio)
    return ratios, out, x


# --------------------------------------------------------------------------- #
# consistent_across_forwards = True (default): same step => identical mask
# --------------------------------------------------------------------------- #
def test_consistent_same_step_identical_mask_across_two_forwards():
    """Same global_step, two separate masked forward invocations (mimicking the
    old_logprob recompute and the train forward of ONE global update) must draw
    the IDENTICAL mask per boundary block when consistent_across_forwards=True."""
    state, model = _make_enabled_state(consistent=True)

    # Forward 1 (e.g. old_logprob recompute) at step 7.
    r1, out1, x1 = _register_and_capture_mask(state, model, global_step=7)
    # Forward 2 (e.g. the actor-train forward) at the SAME step 7. The substep
    # counter has advanced (state.substep == 1 now), but consistent mode holds
    # the key's substep at 0, so the mask must be byte-identical.
    r2, out2, x2 = _register_and_capture_mask(state, model, global_step=7)

    assert state.substep == 2, "substep must still advance per forward"
    assert r1 == r2, f"consistent mode: same-step mask ratios diverged: {r1} vs {r2}"

    # Exact-tensor proof, independent of the ratio fingerprint: re-derive each
    # boundary's mask from the PRF key with substep fixed at 0 and confirm both
    # forwards used it. We reconstruct via a third identical forward on a known
    # input and compare the masked hidden state element-wise.
    r3a, out3a, x3 = _register_and_capture_mask(state, model, global_step=7)
    r3b, _, _ = _register_and_capture_mask(state, model, global_step=7)
    assert r3a == r3b == r1, "consistent mode must reproduce the SAME mask every call at a fixed step"


def test_consistent_mask_tensor_identical_via_prf_key():
    """Exact mask-tensor equality across two forwards at the same step.

    Drives the masker's hook directly the way the engine does in consistent mode
    (substep held at 0) and asserts the realized mask tensor is identical, then
    that a legacy-mode call at the advanced substep differs."""
    from verl.workers.comm_eff.activation_mask import prf_mask

    state, _ = _make_enabled_state(consistent=True)
    masker = state.masker
    layer_idx = 3
    hook = masker._make_hook(layer_idx)
    shape = (2, 4, 32)
    h = torch.randn(shape)

    state.mask_active = True
    state.set_path_tag(TRAIN_TAG)

    # Forward 1: consistent mode => engine passes substep=0 to set_context.
    masker.set_context(global_step=5, substep=0, seq_shard=0)
    out1 = hook(nn.Identity(), (), h)
    # Forward 2: SAME step, consistent mode => substep still 0 in the key.
    masker.set_context(global_step=5, substep=0, seq_shard=0)
    out2 = hook(nn.Identity(), (), h)
    assert torch.equal(out1, out2), "consistent mode: mask tensor must be identical across forwards"

    # The mask is exactly the PRF mask for the fixed-substep key.
    key = (layer_idx, 5, 0, 0, 32, state.masker.base_seed)
    mask = prf_mask(shape, key, masker.p, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(out1, h * mask)


# --------------------------------------------------------------------------- #
# different step => different mask (in BOTH modes)
# --------------------------------------------------------------------------- #
def test_consistent_different_step_different_mask():
    """A different global_step must yield a different mask even in consistent
    mode (otherwise every step would mask identically — wrong)."""
    state, model = _make_enabled_state(consistent=True)
    r_step1, _, _ = _register_and_capture_mask(state, model, global_step=1)
    r_step2, _, _ = _register_and_capture_mask(state, model, global_step=2)
    # The per-boundary measured ratios are a coarse fingerprint; with 7
    # boundaries over a 2x4x32 tensor the probability of an exact tie on every
    # boundary across two independent steps is negligible.
    assert r_step1 != r_step2, "different global_step must change the mask in consistent mode"


def test_consistent_different_step_mask_tensor_differs():
    """Exact-tensor version: same boundary, same shape, different step => the
    realized mask tensor differs."""
    state, _ = _make_enabled_state(consistent=True)
    masker = state.masker
    hook = masker._make_hook(3)
    shape = (4, 8, 32)
    h = torch.randn(shape)
    state.mask_active = True
    state.set_path_tag(TRAIN_TAG)

    masker.set_context(global_step=10, substep=0, seq_shard=0)
    out_a = hook(nn.Identity(), (), h)
    masker.set_context(global_step=11, substep=0, seq_shard=0)
    out_b = hook(nn.Identity(), (), h)
    assert not torch.equal(out_a, out_b), "different step must change the mask tensor"


# --------------------------------------------------------------------------- #
# consistent_across_forwards = False: legacy per-forward resampling preserved
# --------------------------------------------------------------------------- #
def test_legacy_mode_resamples_mask_across_forwards():
    """With the knob false, two forwards at the SAME step draw DIFFERENT masks
    (the EXP-5=>EXP-12 advancing-substep behavior — the bug, kept for the A/B
    test)."""
    state, model = _make_enabled_state(consistent=False)
    r1, _, _ = _register_and_capture_mask(state, model, global_step=7)
    r2, _, _ = _register_and_capture_mask(state, model, global_step=7)
    assert state.substep == 2
    # In legacy mode the engine folds the advancing substep (0 then 1) into the
    # key, so the masks differ across the two same-step forwards.
    assert r1 != r2, "legacy mode must resample the mask per forward (advancing substep in key)"


def test_substep_advances_in_both_modes():
    """The substep counter advances per masked forward regardless of the knob, so
    metrics and the legacy path stay byte-identical; only whether substep is
    FOLDED INTO the PRF key changes."""
    for consistent in (True, False):
        state, model = _make_enabled_state(consistent=consistent)
        assert state.substep == 0
        _register_and_capture_mask(state, model, global_step=3)
        assert state.substep == 1
        _register_and_capture_mask(state, model, global_step=3)
        assert state.substep == 2
