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

"""Unit tests for the sr_quant boundary codec (CPU-only).

Covers: PRF-uniform scalar-reference equivalence; unbiasedness of the
stochastic-rounding forward (whole-token and blockwise scales, bits 1 and 2);
determinism / step freshness; level membership; the load-bearing cross-pass
identity (old-logprob / train / reference forwards of one global_step are
bit-identical); the quantized backward wire (own scale, fresh direction
subkey, unbiased, deterministic); round-to-nearest ablation mode; hook
lifecycle; the path-confinement guard; config validation; and the logical PP
bit-budget metric.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import _derive_seed
from verl.workers.comm_eff.activation_quant import (
    BACKWARD_DIRECTION,
    FORWARD_DIRECTION,
    ActivationQuantizer,
    BoundarySRQuant,
    _block_scales,
    prf_token_uniform,
    sr_quantize,
)
from verl.workers.comm_eff.state import (
    OLD_LOGPROB_TAG,
    PATH_TAGS,
    REF_LOGPROB_TAG,
    TRAIN_TAG,
    CommEffState,
    comm_eff_metrics,
    mask_eligible_tags,
    maybe_build_comm_eff_state,
)

CPU = torch.device("cpu")


def _ids_for(b: int, s: int):
    """Row-major (sample_id, position_id) for a (b, s, H) activation."""
    sid = torch.arange(b).repeat_interleave(s)
    pos = torch.arange(s).repeat(b)
    return sid, pos


def _set_ctx(quantizer, b, s, step=0):
    sid, pos = _ids_for(b, s)
    quantizer.set_context(global_step=step, sample_ids=sid, position_ids=pos)


def _grid_check(q2d: torch.Tensor, s2d: torch.Tensor, bits: int):
    """Assert every entry of q lies on the L-level grid of its scale, |q| <= s."""
    n_levels = 2**bits
    spacing = 2.0 * s2d / (n_levels - 1)
    idx = (q2d.float() + s2d) / spacing
    assert torch.all((idx - idx.round()).abs() < 1e-3), "q off the level grid"
    assert torch.all(idx.round() >= 0) and torch.all(idx.round() <= n_levels - 1)
    assert torch.all(q2d.float().abs() <= s2d * (1 + 1e-5) + 1e-7)


def _sr_kwargs(step=0, layer=3, seed=0, bits=1, block_size=0, direction=FORWARD_DIRECTION):
    return dict(
        layer_idx=layer,
        global_step=step,
        base_seed=seed,
        bits=bits,
        direction=direction,
        block_size=block_size,
        rounding="sr",
    )


# --------------------------------------------------------------------------- #
# PRF uniform: scalar-reference equivalence + direction subkey
# --------------------------------------------------------------------------- #
def _scalar_uniform(sid, pos, ch, *, layer, step, seed, direction):
    """Scalar reference for one (token, channel, direction) uniform draw."""
    h = _derive_seed((seed, layer, step, sid, pos, ch, direction))
    return (h >> 11) / float(1 << 53)


def test_prf_uniform_matches_scalar_reference():
    """The vectorized PRF uniform is bit-identical to the documented scalar key."""
    sid = torch.tensor([5, 9, 0])
    pos = torch.tensor([0, 3, 7])
    for direction in (FORWARD_DIRECTION, BACKWARD_DIRECTION):
        u = prf_token_uniform(
            sid, pos, layer_idx=2, global_step=4, base_seed=1, hidden_size=6, direction=direction, device=CPU
        )
        for t in range(3):
            for ch in range(6):
                expect = _scalar_uniform(int(sid[t]), int(pos[t]), ch, layer=2, step=4, seed=1, direction=direction)
                assert float(u[t, ch]) == expect


def test_prf_uniform_direction_subkey_is_fresh():
    """Forward and backward wires draw from distinct (but reproducible) subkeys."""
    sid, pos = _ids_for(4, 8)
    kw = dict(layer_idx=3, global_step=1, base_seed=0, hidden_size=32, device=CPU)
    u_fwd = prf_token_uniform(sid, pos, direction=FORWARD_DIRECTION, **kw)
    u_bwd = prf_token_uniform(sid, pos, direction=BACKWARD_DIRECTION, **kw)
    assert not torch.equal(u_fwd, u_bwd)
    assert torch.equal(u_bwd, prf_token_uniform(sid, pos, direction=BACKWARD_DIRECTION, **kw))
    with pytest.raises(ValueError):
        prf_token_uniform(sid, pos, direction=2, **kw)


# --------------------------------------------------------------------------- #
# unbiasedness: E[q] = h (bits=1 whole-token, bits=2, blockwise)
# --------------------------------------------------------------------------- #
def _assert_unbiased(h, *, bits, block_size, n_steps=2000, sigma_mult=5.0):
    """Average q over n_steps fresh global_steps and bound |mean - h| by CLT.

    Per element the SR draw is two-point on {lo, lo+D} with P(up) = frac, so
    var = D^2 frac (1-frac) and the mean of n_steps draws deviates by more than
    sigma_mult standard errors only with negligible probability; the PRF is
    deterministic, so a passing tolerance is stable across runs.
    """
    H = h.shape[-1]
    sid, pos = _ids_for(h.shape[0], 1) if h.dim() == 2 else _ids_for(h.shape[0], h.shape[1])
    m = h.reshape(-1, H).float()
    s, _ = _block_scales(m, hidden_size=H, block_size=block_size)
    n_levels = 2**bits
    spacing = 2.0 * s / (n_levels - 1)
    k = torch.floor((m + s) / spacing).clamp(0.0, float(n_levels - 2))
    lo = -s + k * spacing
    frac = ((m - lo) / spacing).clamp(0.0, 1.0)
    sigma_mean = spacing * (frac * (1.0 - frac) / n_steps).sqrt()

    total = torch.zeros(m.shape, dtype=torch.float64)  # fp64 accumulator: no drift on deterministic elements
    for step in range(n_steps):
        q = sr_quantize(h, sid, pos, **_sr_kwargs(step=step, bits=bits, block_size=block_size))
        total += q.reshape(-1, H).double()
    mean_q = total / n_steps
    err = (mean_q - m.double()).abs()
    tol = sigma_mult * sigma_mean.double() + 1e-6
    assert torch.all(err <= tol), f"max unbiasedness violation {(err - tol).max().item():.3e}"


def test_unbiasedness_bits1_whole_token_scale():
    torch.manual_seed(0)
    h = torch.randn(64, 32)  # 64 tokens x 32 dims
    _assert_unbiased(h, bits=1, block_size=0)


def test_unbiasedness_bits2_four_levels():
    """bits=2 => 4 uniform levels; unbiasedness still holds."""
    torch.manual_seed(1)
    h = torch.randn(32, 16)
    _assert_unbiased(h, bits=2, block_size=0)
    # And the level set genuinely has 4 levels: some q strictly inside (-s, s).
    sid, pos = _ids_for(32, 1)
    q = sr_quantize(h, sid, pos, **_sr_kwargs(step=0, bits=2)).reshape(-1, 16)
    s, _ = _block_scales(h.float(), hidden_size=16, block_size=0)
    _grid_check(q, s, bits=2)
    interior = (q.abs() < s * 0.99) & (q.abs() > s * 0.01)
    assert bool(interior.any())


def test_unbiasedness_blockwise_scale_block8():
    """Blockwise scales (block_size=8): unbiasedness holds per element."""
    torch.manual_seed(2)
    h = torch.randn(16, 32)
    _assert_unbiased(h, bits=1, block_size=8)


# --------------------------------------------------------------------------- #
# determinism, step freshness, level membership
# --------------------------------------------------------------------------- #
def test_same_key_bit_identical_and_step_fresh():
    torch.manual_seed(0)
    h = torch.randn(4, 8, 16)
    sid, pos = _ids_for(4, 8)
    a = sr_quantize(h, sid, pos, **_sr_kwargs(step=7))
    b = sr_quantize(h, sid, pos, **_sr_kwargs(step=7))
    c = sr_quantize(h, sid, pos, **_sr_kwargs(step=8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_level_membership_whole_token():
    torch.manual_seed(0)
    H = 32
    h = torch.randn(8, H)
    sid, pos = _ids_for(8, 1)
    for bits in (1, 2, 3):
        q = sr_quantize(h, sid, pos, **_sr_kwargs(step=1, bits=bits)).reshape(-1, H)
        s, _ = _block_scales(h.float(), hidden_size=H, block_size=0)
        _grid_check(q, s, bits=bits)


def test_level_membership_per_block():
    """Blockwise scales: each channel's levels come from ITS block's absmax."""
    torch.manual_seed(0)
    H, block = 32, 8
    # Give each block a wildly different magnitude so grids are distinct.
    h = torch.randn(6, H) * torch.tensor([0.01, 1.0, 100.0, 5.0]).repeat_interleave(block)
    sid, pos = _ids_for(6, 1)
    q = sr_quantize(h, sid, pos, **_sr_kwargs(step=3, bits=1, block_size=block)).reshape(-1, H)
    s, n_blocks = _block_scales(h.float(), hidden_size=H, block_size=block)
    assert n_blocks == 4
    _grid_check(q, s, bits=1)
    # bits=1: q is exactly {-s_block, +s_block} per element.
    assert torch.all((q.abs() - s).abs() < 1e-4 * s)
    # Distinct blocks genuinely use distinct scales (not one token-wide absmax).
    per_block_s = s.reshape(-1, 4, block)
    assert not torch.allclose(per_block_s[:, 0], per_block_s[:, 2])


def test_bits1_reduces_to_sign_with_prob_shift():
    """b=1: q in {-s, +s}; over many steps P(+s) tracks (h/s + 1)/2."""
    torch.manual_seed(0)
    H = 16
    h = torch.randn(4, H)
    sid, pos = _ids_for(4, 1)
    s, _ = _block_scales(h.float(), hidden_size=H, block_size=0)
    n_steps = 2000
    up = torch.zeros(4, H)
    for step in range(n_steps):
        q = sr_quantize(h, sid, pos, **_sr_kwargs(step=step)).reshape(-1, H)
        assert torch.all((q.abs() - s).abs() < 1e-4 * s)  # exactly two levels
        up += (q > 0).float()
    p_up = up / n_steps
    expect = (h.float() / s + 1.0) / 2.0
    assert torch.all((p_up - expect).abs() <= 5.0 * (expect * (1 - expect) / n_steps).sqrt() + 1e-6)


def test_all_zero_token_is_finite_and_tiny():
    """An all-zero token (s clamped to 1e-8) yields |q| <= 1e-8 and no NaN."""
    h = torch.zeros(3, 16)
    sid, pos = _ids_for(3, 1)
    for bits in (1, 2):
        q = sr_quantize(h, sid, pos, **_sr_kwargs(step=0, bits=bits))
        assert torch.isfinite(q).all()
        assert torch.all(q.abs() <= 1e-8 + 1e-12)


def test_bf16_input_computes_in_fp32_and_casts_back():
    h = torch.randn(4, 16, dtype=torch.bfloat16)
    sid, pos = _ids_for(4, 1)
    q = sr_quantize(h, sid, pos, **_sr_kwargs(step=0))
    assert q.dtype == torch.bfloat16
    assert torch.isfinite(q.float()).all()


def test_cross_packing_consistency():
    """A token keyed by (sid, pos) gets the SAME draw under any token ordering."""
    torch.manual_seed(0)
    sid_a = torch.tensor([0, 0, 1, 1, 2])
    pos_a = torch.tensor([0, 1, 0, 1, 0])
    perm = torch.tensor([3, 0, 4, 1, 2])
    h = torch.randn(5, 24)
    qa = sr_quantize(h, sid_a, pos_a, **_sr_kwargs(step=11))
    qb = sr_quantize(h[perm], sid_a[perm], pos_a[perm], **_sr_kwargs(step=11))
    assert torch.equal(qa[perm], qb)


# --------------------------------------------------------------------------- #
# round-to-nearest ablation mode
# --------------------------------------------------------------------------- #
def test_rn_mode_deterministic_idempotent_on_grid():
    """rn: no PRF draw (step-invariant), idempotent, level-exact: and biased."""
    torch.manual_seed(0)
    H = 32
    h = torch.randn(8, H)
    sid, pos = _ids_for(8, 1)

    def rn(x, step, block=0, bits=2):
        return sr_quantize(
            x, sid, pos, layer_idx=3, global_step=step, base_seed=0, bits=bits, block_size=block, rounding="rn"
        )

    q0 = rn(h, step=0)
    q1 = rn(h, step=1)
    assert torch.equal(q0, q1)  # no step dependence: no PRF draw at all
    # Idempotent: bit-exact at bits=1 (D = 2s is float-exact), ulp-exact at
    # bits=2 (D = 2s/3 rounds, so the regenerated grid can shift by ulps).
    q_b1 = rn(h, step=0, bits=1)
    assert torch.equal(rn(q_b1, step=5, bits=1), q_b1)
    assert torch.allclose(rn(q0, step=5), q0, rtol=1e-5, atol=1e-6)
    # Level membership on the same grid as sr.
    s, _ = _block_scales(h.float(), hidden_size=H, block_size=0)
    _grid_check(q0, s, bits=2)
    # Biased in general: rn rounds toward the nearer level, so q != h.
    assert not torch.allclose(q0, h)
    # rn works without per-token identity (no PRF key needed).
    q_no_ids = sr_quantize(h, None, None, layer_idx=3, global_step=0, base_seed=0, bits=2, rounding="rn")
    assert torch.equal(q_no_ids, q0)
    # blockwise rn is deterministic and idempotent too (bit-exact at bits=1).
    qb = rn(h, step=0, block=8, bits=1)
    assert torch.equal(rn(qb, step=9, block=8, bits=1), qb)


def test_sr_mode_requires_ids():
    h = torch.randn(4, 8)
    with pytest.raises(RuntimeError):
        sr_quantize(h, None, None, layer_idx=0, global_step=0, base_seed=0, bits=1, rounding="sr")


def test_bad_args_raise():
    h = torch.randn(2, 8)
    sid, pos = _ids_for(2, 1)
    with pytest.raises(ValueError):
        sr_quantize(h, sid, pos, layer_idx=0, global_step=0, base_seed=0, bits=0)
    with pytest.raises(ValueError):
        sr_quantize(h, sid, pos, layer_idx=0, global_step=0, base_seed=0, bits=1, block_size=-1)
    with pytest.raises(ValueError):
        sr_quantize(h, sid, pos, layer_idx=0, global_step=0, base_seed=0, bits=1, rounding="nearest")


# --------------------------------------------------------------------------- #
# backward wire: quantized gradient
# --------------------------------------------------------------------------- #
def test_backward_returns_sr_quantized_gradient():
    """backward(g) == sr_quantize(g, direction=1): exact, on-grid, own scale."""
    torch.manual_seed(0)
    h = torch.randn(2, 8, 16, requires_grad=True)
    sid, pos = _ids_for(2, 8)
    out = BoundarySRQuant.apply(h, sid, pos, 3, 7, 0, 1, 0, "sr")
    g = torch.randn_like(out)
    out.backward(g)
    assert h.grad is not None

    expected = sr_quantize(g, sid, pos, **_sr_kwargs(step=7, direction=BACKWARD_DIRECTION))
    assert torch.equal(h.grad, expected)
    # Level membership against g's OWN per-token absmax scale (not h's).
    s_g, _ = _block_scales(g.reshape(-1, 16).float(), hidden_size=16, block_size=0)
    _grid_check(h.grad.reshape(-1, 16), s_g, bits=1)
    # Fresh direction subkey: the backward draw differs from the forward-keyed
    # quantization of the same tensor.
    fwd_keyed = sr_quantize(g, sid, pos, **_sr_kwargs(step=7, direction=FORWARD_DIRECTION))
    assert not torch.equal(h.grad, fwd_keyed)

    # Deterministic per key: a second identical pass yields the identical grad.
    h2 = h.detach().clone().requires_grad_(True)
    out2 = BoundarySRQuant.apply(h2, sid, pos, 3, 7, 0, 1, 0, "sr")
    out2.backward(g)
    assert torch.equal(h.grad, h2.grad)


def test_backward_gradient_unbiased_over_steps():
    """E[g_hat] = g for the backward wire (direction=1), by CLT bound."""
    torch.manual_seed(3)
    g = torch.randn(16, 16)
    sid, pos = _ids_for(16, 1)
    H = 16
    m = g.float()
    s, _ = _block_scales(m, hidden_size=H, block_size=0)
    spacing = 2.0 * s  # bits=1
    frac = ((m + s) / spacing).clamp(0.0, 1.0)
    n_steps = 2000
    total = torch.zeros(m.shape, dtype=torch.float64)
    for step in range(n_steps):
        total += sr_quantize(g, sid, pos, **_sr_kwargs(step=step, direction=BACKWARD_DIRECTION)).double()
    err = (total / n_steps - m.double()).abs()
    tol = 5.0 * spacing.double() * (frac.double() * (1 - frac.double()) / n_steps).sqrt() + 1e-6
    assert torch.all(err <= tol)


# --------------------------------------------------------------------------- #
# hook lifecycle on a toy model
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


def test_register_installs_hooks_on_boundaries_only():
    model = _ToyDecoder(num_layers=16, d=32)
    quantizer = ActivationQuantizer(bits=1, base_seed=0, pp_size=8)
    quantizer.register(model)
    assert quantizer.boundary_indices == [1, 3, 5, 7, 9, 11, 13]
    assert quantizer.is_registered
    for i, layer in enumerate(model.layers):
        assert (len(layer._forward_hooks) > 0) == (i in quantizer.boundary_indices)
    quantizer.unregister()
    assert not quantizer.is_registered
    for layer in model.layers:
        assert len(layer._forward_hooks) == 0


def test_register_is_idempotent_and_unregister_leaves_forward_clean():
    torch.manual_seed(0)
    model = _ToyDecoder(num_layers=16, d=32)
    quantizer = ActivationQuantizer(bits=1, base_seed=0, pp_size=8)
    quantizer.register(model)
    n_handles = len(quantizer._handles)
    quantizer.register(model)
    assert len(quantizer._handles) == n_handles

    x = torch.randn(2, 4, 32)
    _set_ctx(quantizer, 2, 4)
    out_quant = model(x)
    quantizer.unregister()
    out_clean = model(x)
    assert not torch.allclose(out_quant, out_clean)
    assert torch.allclose(out_clean, model(x))


def test_quantizer_is_hooks_only_no_params_or_buffers():
    model = _ToyDecoder(num_layers=16, d=32)
    keys_before = set(model.state_dict().keys())
    quantizer = ActivationQuantizer(bits=1, base_seed=0, pp_size=8)
    quantizer.register(model)
    assert set(model.state_dict().keys()) == keys_before
    quantizer.unregister()


def test_missing_context_raises():
    quantizer = ActivationQuantizer(bits=1, base_seed=0, pp_size=8)
    with pytest.raises(RuntimeError):
        quantizer._make_hook(3)(nn.Identity(), (), torch.randn(2, 4, 16))


def test_token_axis_mismatch_raises():
    quantizer = ActivationQuantizer(bits=1, base_seed=0, pp_size=8)
    _set_ctx(quantizer, 2, 4)  # 8 tokens
    with pytest.raises(RuntimeError):
        quantizer._make_hook(3)(nn.Identity(), (), torch.randn(3, 4, 16))  # 12 tokens


def test_tuple_output_first_element_quantized():
    quantizer = ActivationQuantizer(bits=1, base_seed=0, pp_size=8)
    _set_ctx(quantizer, 2, 4)
    extra = torch.randn(2, 4, 16)
    out = quantizer._make_hook(3)(nn.Identity(), (), (torch.randn(2, 4, 16), extra))
    assert isinstance(out, tuple)
    assert torch.equal(out[1], extra)


def test_hook_is_in_graph():
    quantizer = ActivationQuantizer(bits=1, base_seed=0, pp_size=8)
    _set_ctx(quantizer, 2, 4)
    h = torch.randn(2, 4, 16, requires_grad=True)
    quantizer._make_hook(3)(nn.Identity(), (), h).sum().backward()
    assert h.grad is not None


def test_applications_counter_increments():
    class _FakeState:
        def __init__(self):
            self.mask_applications = 0

    state = _FakeState()
    quantizer = ActivationQuantizer(bits=1, base_seed=0, pp_size=8, state=state)
    hook = quantizer._make_hook(3)
    _set_ctx(quantizer, 2, 4)
    hook(nn.Identity(), (), torch.randn(2, 4, 16))
    hook(nn.Identity(), (), torch.randn(2, 4, 16))
    assert state.mask_applications == 2


def test_constructor_validation():
    with pytest.raises(ValueError):
        ActivationQuantizer(bits=0, base_seed=0, pp_size=8)
    with pytest.raises(ValueError):
        ActivationQuantizer(bits=1, base_seed=0, pp_size=8, block_size=-1)
    with pytest.raises(ValueError):
        ActivationQuantizer(bits=1, base_seed=0, pp_size=8, rounding="stochastic")


# =========================================================================== #
# path-isolation / contamination guard + cross-pass identity through the state
# =========================================================================== #
def _make_quant_state(
    bits=1,
    block_size=0,
    rounding="sr",
    pp_size=8,
    seed=0,
    mask_recompute=False,
    mask_reference=False,
    d=32,
    subset_k=0,
):
    cfg = SimpleNamespace(
        enabled=True,
        compression_type="sr_quant",
        mask=SimpleNamespace(
            enabled=False,
            p=0.95,
            seed=seed,
            pp_size=pp_size,
            mask_recompute=mask_recompute,
            mask_reference=mask_reference,
        ),
        quant=SimpleNamespace(bits=bits, block_size=block_size, rounding=rounding, subset_k=subset_k),
    )
    state = maybe_build_comm_eff_state(cfg)
    assert isinstance(state, CommEffState)
    model = _ToyDecoder(num_layers=16, d=d)
    state.build(model)
    assert state.quantizer is not None
    assert state.masker is None  # mutually exclusive: sr_quant builds no masker
    assert state.powersgd is None
    return state, model


def test_quant_fires_only_on_train_tag():
    state, model = _make_quant_state()
    state.set_path_tag(TRAIN_TAG)
    state.quantizer.register(model)
    _set_ctx(state.quantizer, 2, 4)
    model(torch.randn(2, 4, 32))
    state.quantizer.unregister()

    assert state.mask_applications > 0
    assert state.mask_applications_by_path[TRAIN_TAG] == state.mask_applications
    for tag in PATH_TAGS:
        if tag != TRAIN_TAG:
            assert state.mask_applications_by_path[tag] == 0


@pytest.mark.parametrize("bad_tag", [t for t in PATH_TAGS if t != TRAIN_TAG] + [None])
def test_quant_hook_asserts_on_non_train_path(bad_tag):
    state, _ = _make_quant_state()
    state.set_path_tag(bad_tag)
    _set_ctx(state.quantizer, 2, 4)
    hook = state.quantizer._make_hook(3)
    with pytest.raises(AssertionError):
        hook(nn.Identity(), (), torch.randn(2, 4, 32))
    for tag in PATH_TAGS:
        assert state.mask_applications_by_path[tag] == 0


def test_quant_eligibility_reuses_mask_recompute_and_reference_knobs():
    """sr_quant widens eligibility via mask.mask_recompute / mask.mask_reference
    WITHOUT requiring mask.enabled (the mask codec itself stays off)."""
    s_default, _ = _make_quant_state()
    s_recomp, _ = _make_quant_state(mask_recompute=True)
    s_ref, _ = _make_quant_state(mask_reference=True)
    s_both, _ = _make_quant_state(mask_recompute=True, mask_reference=True)
    assert mask_eligible_tags(s_default) == frozenset({TRAIN_TAG})
    assert mask_eligible_tags(s_recomp) == frozenset({TRAIN_TAG, OLD_LOGPROB_TAG})
    assert mask_eligible_tags(s_ref) == frozenset({TRAIN_TAG, REF_LOGPROB_TAG})
    assert mask_eligible_tags(s_both) == frozenset({TRAIN_TAG, OLD_LOGPROB_TAG, REF_LOGPROB_TAG})

    # old_logprob rejected when mask_recompute is off.
    hook = s_default.quantizer._make_hook(3)
    _set_ctx(s_default.quantizer, 2, 4)
    s_default.set_path_tag(OLD_LOGPROB_TAG)
    with pytest.raises(AssertionError):
        hook(nn.Identity(), (), torch.randn(2, 4, 32))


def test_cross_pass_identity_across_path_tags():
    """The load-bearing PPO-ratio property: old / train / reference forwards of
    one global_step produce BIT-IDENTICAL quantized activations."""
    torch.manual_seed(0)
    state, _ = _make_quant_state(mask_recompute=True, mask_reference=True)
    hook = state.quantizer._make_hook(3)
    h = torch.randn(2, 8, 32)
    outs = {}
    for tag in (TRAIN_TAG, OLD_LOGPROB_TAG, REF_LOGPROB_TAG):
        _set_ctx(state.quantizer, 2, 8, step=6)
        state.set_path_tag(tag)
        outs[tag] = hook(nn.Identity(), (), h.clone())
    assert torch.equal(outs[TRAIN_TAG], outs[OLD_LOGPROB_TAG])
    assert torch.equal(outs[TRAIN_TAG], outs[REF_LOGPROB_TAG])
    # Fresh draw on the next step.
    _set_ctx(state.quantizer, 2, 8, step=7)
    state.set_path_tag(TRAIN_TAG)
    assert not torch.equal(outs[TRAIN_TAG], hook(nn.Identity(), (), h.clone()))


def test_recompute_replay_bit_identical():
    """Same context (gradient-checkpoint recompute) => bit-identical output."""
    torch.manual_seed(0)
    state, _ = _make_quant_state()
    state.set_path_tag(TRAIN_TAG)
    hook = state.quantizer._make_hook(3)
    h = torch.randn(2, 4, 32)
    _set_ctx(state.quantizer, 2, 4, step=0)
    out_a = hook(nn.Identity(), (), h)  # original forward
    out_b = hook(nn.Identity(), (), h)  # recompute: same context, no set_context
    assert torch.equal(out_a, out_b)


# --------------------------------------------------------------------------- #
# bit-budget metric + per-path counters through comm_eff_metrics
# --------------------------------------------------------------------------- #
def test_logical_pp_bits_metric_whole_token_scale():
    """H*bits + 16 for block_size=0 (one fp16 scale per token per boundary)."""
    state, model = _make_quant_state(bits=1, block_size=0)
    state.set_path_tag(TRAIN_TAG)
    state.quantizer.register(model)
    _set_ctx(state.quantizer, 2, 4, step=1)
    model(torch.randn(2, 4, 32))
    state.quantizer.unregister()
    metrics = comm_eff_metrics(state)
    assert metrics["comm_eff/logical_pp_bits_sr_quant"] == 32 * 1 + 16
    assert metrics["comm_eff/logical_pp_bytes_sr_quant"] == (32 * 1 + 16) / 8.0
    for tag in PATH_TAGS:
        assert f"comm_eff/mask_applications/{tag}" in metrics
    nonzero = [k for k, v in metrics.items() if k.startswith("comm_eff/mask_applications/") and v > 0]
    assert nonzero == [f"comm_eff/mask_applications/{TRAIN_TAG}"]


def test_logical_pp_bits_metric_reflects_block_scale_overhead():
    """H*bits + (H/block)*16: blockwise fp16 scales enter the budget."""
    state, model = _make_quant_state(bits=1, block_size=8)
    state.set_path_tag(TRAIN_TAG)
    state.quantizer.register(model)
    _set_ctx(state.quantizer, 2, 4, step=1)
    model(torch.randn(2, 4, 32))
    state.quantizer.unregister()
    metrics = comm_eff_metrics(state)
    assert metrics["comm_eff/logical_pp_bits_sr_quant"] == 32 * 1 + (32 // 8) * 16
    # Real geometry: H=1536, bits=1, block 32 -> 1536 + 48*16 = 2304 bits (288 B);
    # block_size=0 -> 1536 + 16 = 1552 bits (194 B). Checked via the hook math.
    q = ActivationQuantizer(bits=1, base_seed=0, pp_size=8, block_size=32)
    sid = torch.zeros(2, dtype=torch.int64)
    pos = torch.arange(2)
    q.set_context(global_step=0, sample_ids=sid, position_ids=pos)
    q._make_hook(1)(nn.Identity(), (), torch.randn(1, 2, 1536))
    assert q.logical_pp_bits_sr_quant == 1536 + 48 * 16
    q0 = ActivationQuantizer(bits=1, base_seed=0, pp_size=8, block_size=0)
    q0.set_context(global_step=0, sample_ids=sid, position_ids=pos)
    q0._make_hook(1)(nn.Identity(), (), torch.randn(1, 2, 1536))
    assert q0.logical_pp_bits_sr_quant == 1536 + 16


# --------------------------------------------------------------------------- #
# config schema: sr_quant registration + guards
# --------------------------------------------------------------------------- #
def test_config_sr_quant_validation():
    from verl.workers.config.comm_eff import (
        CommEffAnchorConfig,
        CommEffConfig,
        CommEffQuantConfig,
    )

    # Defaults compose while disabled.
    cfg = CommEffConfig(compression_type="sr_quant")
    assert cfg.quant.bits == 1
    assert cfg.quant.block_size == 32
    assert cfg.quant.rounding == "sr"

    # Like prf_mask, sr_quant cannot anchor-own-Q once enabled.
    with pytest.raises(ValueError, match="owns_q"):
        CommEffConfig(enabled=True, compression_type="sr_quant")
    CommEffConfig(
        enabled=True,
        compression_type="sr_quant",
        anchor=CommEffAnchorConfig(owns_q=False),
    )

    # Knob bounds fail loud.
    with pytest.raises(ValueError):
        CommEffConfig(quant=CommEffQuantConfig(bits=0))
    with pytest.raises(ValueError):
        CommEffConfig(quant=CommEffQuantConfig(bits=32))
    with pytest.raises(ValueError):
        CommEffConfig(quant=CommEffQuantConfig(block_size=-1))
    with pytest.raises(ValueError):
        CommEffConfig(quant=CommEffQuantConfig(rounding="nearest"))
    with pytest.raises(ValueError):
        CommEffConfig(compression_type="not_a_codec")


def test_resolve_compression_type_sr_quant_wins():
    from verl.workers.comm_eff.state import resolve_compression_type

    assert resolve_compression_type(SimpleNamespace(compression_type="sr_quant")) == "sr_quant"
    # The mask fall-through is untouched.
    assert (
        resolve_compression_type(SimpleNamespace(compression_type="dense", mask=SimpleNamespace(enabled=True, p=0.5)))
        == "prf_mask"
    )


# =========================================================================== #
# subset mode (issue #93 I5): the byte-parity hybrid
# =========================================================================== #
def _subset_keep_mask(sid, pos, *, layer, step, seed, hidden_size, k):
    """The subset J exactly as the mask codec's order statistic draws it."""
    from verl.workers.comm_eff.activation_mask import prf_token_mask

    return prf_token_mask(
        sid,
        pos,
        layer_idx=layer,
        global_step=step,
        base_seed=seed,
        hidden_size=hidden_size,
        p=0.0,
        device=CPU,
        dtype=torch.float32,
        exact_k=True,
        exact_keep=k,
    ).bool()


def test_subset_zero_is_full_width_regression():
    """subset_k=0 (and the omitted default) is byte-identical to full width."""
    torch.manual_seed(0)
    h = torch.randn(4, 8, 16)
    sid, pos = _ids_for(4, 8)
    for rounding in ("sr", "rn"):
        for block_size in (0, 8):
            kw = _sr_kwargs(step=3, bits=1, block_size=block_size)
            kw["rounding"] = rounding
            base = sr_quantize(h, sid, pos, **kw)
            assert torch.equal(base, sr_quantize(h, sid, pos, **kw, subset_k=0))


def test_subset_keeps_exactly_k_zero_elsewhere_mask_keyed():
    """Support of q is EXACTLY the mask codec's exact-k order-statistic J."""
    torch.manual_seed(0)
    H, k = 32, 7
    h = torch.randn(6, H)
    sid, pos = _ids_for(6, 1)
    q = sr_quantize(h, sid, pos, **_sr_kwargs(step=5, bits=2, block_size=4), subset_k=k).reshape(-1, H)
    keep = _subset_keep_mask(sid, pos, layer=3, step=5, seed=0, hidden_size=H, k=k)
    # bits=2 levels exclude 0 on a clamped-positive scale, so support == J.
    assert torch.equal(q != 0, keep)
    assert int((q != 0).sum(dim=-1).unique()) == k


def test_subset_unbiased_through_subset_and_rounding():
    """E[q] = h over BOTH randomness sources: average over many PRF base seeds
    (each seed refreshes the subset J and the SR uniforms) and bound the error
    by the empirical CLT standard error per element."""
    torch.manual_seed(0)
    H, k, n_seeds = 32, 8, 3000
    h = torch.randn(8, H)
    sid, pos = _ids_for(8, 1)
    total = torch.zeros((8, H), dtype=torch.float64)
    total_sq = torch.zeros((8, H), dtype=torch.float64)
    for seed in range(n_seeds):
        q = (
            sr_quantize(h, sid, pos, **_sr_kwargs(step=0, seed=seed, bits=2, block_size=8), subset_k=k)
            .reshape(-1, H)
            .double()
        )
        total += q
        total_sq += q * q
    mean_q = total / n_seeds
    var_q = (total_sq / n_seeds - mean_q * mean_q).clamp_min(0.0)
    se = (var_q / n_seeds).sqrt()
    err = (mean_q - h.double()).abs()
    tol = 5.0 * se + 1e-6
    assert torch.all(err <= tol), f"max unbiasedness violation {(err - tol).max().item():.3e}"


def test_subset_pass_identity_and_step_freshness():
    """Same (step, sample): bit-identical (J and SR shared); new step: fresh."""
    torch.manual_seed(0)
    h = torch.randn(4, 8, 32)
    sid, pos = _ids_for(4, 8)
    a = sr_quantize(h, sid, pos, **_sr_kwargs(step=7, bits=2, block_size=8), subset_k=9)
    b = sr_quantize(h, sid, pos, **_sr_kwargs(step=7, bits=2, block_size=8), subset_k=9)
    c = sr_quantize(h, sid, pos, **_sr_kwargs(step=8, bits=2, block_size=8), subset_k=9)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_subset_forward_backward_share_j_fresh_sr_subkey():
    """The wires share ONE J (no direction in the subset key) while the SR
    draw itself uses the fresh backward subkey."""
    torch.manual_seed(0)
    H, k = 32, 9
    h = torch.randn(24, H)
    sid, pos = _ids_for(24, 1)
    kw = dict(bits=2, block_size=8)
    q_fwd = sr_quantize(h, sid, pos, **_sr_kwargs(step=2, direction=FORWARD_DIRECTION, **kw), subset_k=k)
    q_bwd = sr_quantize(h, sid, pos, **_sr_kwargs(step=2, direction=BACKWARD_DIRECTION, **kw), subset_k=k)
    assert torch.equal(q_fwd != 0, q_bwd != 0)  # same J
    assert not torch.equal(q_fwd, q_bwd)  # fresh SR subkey


def test_subset_boundary_function_backward_subsets_gradient():
    """BoundarySRQuant threads subset_k to the backward wire: h.grad is the
    subset-quantized upstream gradient (same J as the forward, BACKWARD SR
    subkey, H/k rescale: the exact adjoint of the forward's subset map plus
    the quantized backward wire)."""
    torch.manual_seed(0)
    H, k = 32, 9
    h = torch.randn(2, 8, H, requires_grad=True)
    sid, pos = _ids_for(2, 8)
    out = BoundarySRQuant.apply(h, sid, pos, 3, 4, 0, 2, 8, "sr", k)
    g = torch.randn_like(out)
    out.backward(g)
    expect = sr_quantize(
        g, sid, pos, **_sr_kwargs(step=4, direction=BACKWARD_DIRECTION, bits=2, block_size=8), subset_k=k
    )
    assert torch.equal(h.grad, expect)
    assert torch.equal(out != 0, h.grad != 0)  # shared J across the wires


def test_subset_cross_pass_identity_through_state():
    """old/train/reference forwards of one step stay bit-identical with the
    subset lever on (the PPO ratio identity survives I5)."""
    torch.manual_seed(0)
    state, _ = _make_quant_state(bits=2, block_size=8, mask_recompute=True, mask_reference=True, subset_k=9)
    assert state.quantizer.subset_k == 9  # build() plumbs quant.subset_k
    hook = state.quantizer._make_hook(3)
    h = torch.randn(2, 8, 32)
    outs = {}
    for tag in (TRAIN_TAG, OLD_LOGPROB_TAG, REF_LOGPROB_TAG):
        _set_ctx(state.quantizer, 2, 8, step=6)
        state.set_path_tag(tag)
        outs[tag] = hook(nn.Identity(), (), h.clone())
    assert torch.equal(outs[TRAIN_TAG], outs[OLD_LOGPROB_TAG])
    assert torch.equal(outs[TRAIN_TAG], outs[REF_LOGPROB_TAG])


def test_subset_byte_accounting_parity_arm():
    """The #93 4.3 parity arm: k=493, bits=2, block=32 on H=1536 =>
    493*2 + 493*16/32 = 1232.5 logical bits/token/boundary (ceil 1233, vs the
    prf exact-k incumbent's 77*16 = 1232); block_size=0 => k*bits + 16."""
    q = ActivationQuantizer(bits=2, base_seed=0, pp_size=8, block_size=32, subset_k=493)
    sid = torch.zeros(2, dtype=torch.int64)
    pos = torch.arange(2)
    q.set_context(global_step=0, sample_ids=sid, position_ids=pos)
    q._make_hook(1)(nn.Identity(), (), torch.randn(1, 2, 1536))
    assert q.logical_pp_bits_sr_quant == 493 * 2 + 493 * 16 / 32  # 1232.5
    q0 = ActivationQuantizer(bits=2, base_seed=0, pp_size=8, block_size=0, subset_k=493)
    q0.set_context(global_step=0, sample_ids=sid, position_ids=pos)
    q0._make_hook(1)(nn.Identity(), (), torch.randn(1, 2, 1536))
    assert q0.logical_pp_bits_sr_quant == 493 * 2 + 16
    # And through comm_eff_metrics on the toy state.
    state, model = _make_quant_state(bits=2, block_size=8, subset_k=16)
    state.set_path_tag(TRAIN_TAG)
    state.quantizer.register(model)
    _set_ctx(state.quantizer, 2, 4, step=1)
    model(torch.randn(2, 4, 32))
    state.quantizer.unregister()
    metrics = comm_eff_metrics(state)
    assert metrics["comm_eff/logical_pp_bits_sr_quant"] == 16 * 2 + 16 * 16 / 8


def test_subset_validation():
    with pytest.raises(ValueError):
        sr_quantize(torch.randn(2, 8), *_ids_for(2, 1), **_sr_kwargs(), subset_k=-1)
    with pytest.raises(ValueError, match="exceeds the hidden size"):
        sr_quantize(torch.randn(2, 8), *_ids_for(2, 1), **_sr_kwargs(), subset_k=9)
    # The subset draw needs per-token identity even under rn (J is PRF-keyed).
    kw = _sr_kwargs()
    kw["rounding"] = "rn"
    with pytest.raises(RuntimeError, match="identity"):
        sr_quantize(torch.randn(2, 8), None, None, **kw, subset_k=4)
    with pytest.raises(ValueError):
        ActivationQuantizer(bits=1, base_seed=0, pp_size=8, subset_k=-1)
    from verl.workers.config.comm_eff import CommEffConfig, CommEffQuantConfig

    with pytest.raises(ValueError, match="subset_k"):
        CommEffConfig(quant=CommEffQuantConfig(subset_k=-1))
    assert CommEffConfig(quant=CommEffQuantConfig(subset_k=493)).quant.subset_k == 493
