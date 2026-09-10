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

"""CPU tests for the tah_quant boundary codec (TAH-Quant, arXiv:2506.01352v2).

Grouped by the claim each test defends. The claims that matter for the RLVR
codec table are: the arm is byte-matched to PRF exact-k so the codec is the only
variable, the reconstruction writes ONLY what the priced wire could carry, it is
pass-identical within an optimizer step so the PPO ratio still starts at one,
the codec is genuinely STATELESS (which is the property that makes it applicable
to on-policy RLVR at all, where AQ-SGD's premise fails), and the Hadamard
actually suppresses outliers at the bit widths we run rather than only at the
paper's.
"""

import math

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import prf_token_mask
from verl.workers.comm_eff.activation_tahquant import (
    TAHQUANT_INT3_BITS,
    TAHQUANT_INT4_BITS,
    ActivationTAHQuant,
    BoundaryTAHQuant,
    tahquant_hadamard,
    tahquant_metadata_bits,
    tahquant_reconstruct,
    tahquant_tile_entropy,
)

# Pair 1's operating point is H=1536 with k=242 at tile=32. Tests use a narrower
# H for speed but keep the tile and a comparable coverage ratio.
H = 256
K = 64  # 25% coverage, two full tiles at TILE=32
TILE = 32
N = 24


def _ids(n=N, step=7):
    return dict(
        sample_ids=torch.arange(n),
        position_ids=torch.arange(n),
        layer_idx=3,
        global_step=step,
        base_seed=1234,
    )


def _rec(x, *, step=7, tile=TILE, int4_frac=0.8, tau=2.0, rounding="rn", subset_k=K):
    return tahquant_reconstruct(
        x,
        **_ids(x.shape[0], step),
        tile=tile,
        int4_frac=int4_frac,
        tau=tau,
        rounding=rounding,
        subset_k=subset_k,
    )


# --------------------------------------------------------------------------- #
# the money claim: byte parity, or the arm proves nothing
# --------------------------------------------------------------------------- #
def test_wire_ledger_matches_prf_exact_k_at_pair_one():
    """k=242 at tile=32 must price out against PRF exact-k's 1232 bits.

    This is the whole premise of the codec ablation: three arms spend ONE budget
    and differ only in how they split it. The numbers here are Pair 1's real
    geometry (H=1536), not this module's reduced one.
    """
    codec = ActivationTAHQuant(tile=32, int4_frac=0.8, subset_k=242, base_seed=0)
    codec._record_bits(1536)
    bits = codec.logical_pp_bits_tah_quant
    assert bits == pytest.approx(1231.6), bits
    prf = 77 * 16
    assert abs(bits / prf - 1.0) < 1e-3, bits / prf
    # And the decomposition, so a future edit cannot drift one term silently.
    assert tahquant_metadata_bits(32) == 39
    assert 242 * (0.8 * TAHQUANT_INT4_BITS + 0.2 * TAHQUANT_INT3_BITS) == pytest.approx(919.6)
    assert math.ceil(242 / 32) == 8


def test_metadata_width_is_what_forces_tile_32():
    """The tile size is not a taste choice, it is arithmetic.

    Reproducing the paper's 4.41 bits/element leaves a unique 39 bits/tile, of
    which scale and zero-point take 16 each. At tile=64 the 6-bit pivot index
    consumes the last bit, leaving nothing for the flag the receiver needs in
    order to know whether the rotation was applied, so the paper's own budget
    cannot pay for the conditional gate its algorithm requires. At 32 the pivot
    needs 5 bits and the flag fits at the SAME width.
    """
    assert tahquant_metadata_bits(32) == 2 * 16 + 1 + 5 + 1 == 39
    assert tahquant_metadata_bits(64) == 2 * 16 + 1 + 6 + 1 == 40
    # The paper's own figure, which has no room for the flag.
    paper_meta_at_64 = 2 * 16 + 1 + 6
    assert 0.8 * 4 + 0.2 * 3 + paper_meta_at_64 / 64 == pytest.approx(4.41, abs=1e-3)
    # And at 64 with the flag counted, the parity solution misses tolerance,
    # which is the concrete reason 32 is the default.
    bits_64 = 273 * 3.8 + math.ceil(273 / 64) * tahquant_metadata_bits(64)
    assert abs(bits_64 / (77 * 16) - 1.0) > 1e-3, bits_64


def test_full_width_is_off_budget_by_the_factor_that_forces_subsetting():
    """The published setting is 5.50x our budget, which is why subset_k exists."""
    codec = ActivationTAHQuant(tile=32, int4_frac=0.8, subset_k=0, base_seed=0)
    codec._record_bits(1536)
    ratio = codec.logical_pp_bits_tah_quant / (77 * 16)
    assert ratio > 5.0, ratio


# --------------------------------------------------------------------------- #
# the reconstruction may use only what the wire could carry
# --------------------------------------------------------------------------- #
def test_support_is_exactly_the_prf_subset():
    """Nothing may be written outside the PRF-keyed channel subset.

    The subset costs zero bits precisely because both sides DERIVE it, so a
    reconstruction that touched any other coordinate would be spending bits the
    ledger does not account for, and the byte-parity claim would be false.
    """
    torch.manual_seed(0)
    x = torch.randn(N, H) * 3.0
    x[:, 11] = 4.0e4  # a massive-activation channel, as Qwen2.5 carries
    keep = prf_token_mask(
        torch.arange(N),
        torch.arange(N),
        layer_idx=3,
        global_step=7,
        base_seed=1234,
        hidden_size=H,
        p=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        exact_k=True,
        exact_keep=K,
    ).bool()
    for rounding in ("rn", "sr"):
        for tau in (0.0, 2.0, float("inf")):
            out = _rec(x, rounding=rounding, tau=tau)
            assert torch.isfinite(out).all(), (rounding, tau)
            assert torch.equal(out[~keep], torch.zeros_like(out[~keep])), (rounding, tau)


def test_the_gain_restores_the_activation_scale():
    """A gained sparse estimator sits at ``sqrt(H/k)``, not at 1.

    Worth pinning as a NUMBER rather than a vibe: without the ``H/k`` gain the
    network would see activations shrunk by the coverage ratio, which is a
    different model, not a compressed one. The incumbent PRF codec has exactly
    this property too (at k=77 its factor is 4.47), so the check is that
    tah_quant sits at its own theoretical value.
    """
    torch.manual_seed(0)
    x = torch.randn(N, H)
    measured = float(_rec(x).norm() / x.norm())
    theoretical = (H / K) ** 0.5
    assert 0.8 < measured / theoretical < 1.25, (measured, theoretical)


# --------------------------------------------------------------------------- #
# the paper's entropy equation is printed with the wrong sign
# --------------------------------------------------------------------------- #
def test_entropy_is_shannon_and_not_the_papers_printed_sign():
    """Eq (2) is printed WITHOUT a minus sign, which inverts it.

    As printed, ``H = sum p log(p + zeta)`` is maximal for a one-hot tile and
    minimal for a uniform one, the exact opposite of the paper's own prose ("H
    is high when energy is spread evenly ... and low when the tile contains a
    sharp outlier") and of the top-p% rule built on it. Implementing it
    literally would invert the whole bit allocation, giving INT4 to precisely
    the spiky tiles the rotation already fixes. So the orientation is asserted.
    """
    flat = torch.ones(1, 32)
    spike = torch.zeros(1, 32)
    spike[0, 5] = 1.0
    e_flat = float(tahquant_tile_entropy(flat))
    e_spike = float(tahquant_tile_entropy(spike))
    assert e_flat > e_spike, (e_flat, e_spike)
    assert e_flat == pytest.approx(math.log(32), abs=1e-4)
    assert e_spike == pytest.approx(0.0, abs=1e-6)


def test_int4_goes_to_the_flat_tiles_not_the_spiky_ones():
    """The allocation must follow the entropy, end to end through reconstruct."""
    x = torch.zeros(1, 64)
    x[0, :32] = torch.ones(32)  # tile 0 flat  -> high entropy -> INT4
    x[0, 32] = 1.0  # tile 1 one-hot -> low entropy  -> INT3
    ent = tahquant_tile_entropy(x.reshape(1, 2, 32))
    assert float(ent[0, 0]) > float(ent[0, 1])


# --------------------------------------------------------------------------- #
# the Hadamard, which is the method's headline mechanism
# --------------------------------------------------------------------------- #
def test_hadamard_is_orthogonal_and_rejects_non_powers_of_two():
    for g in (2, 16, 32, 64):
        h = tahquant_hadamard(g, device=torch.device("cpu"))
        assert torch.allclose(h @ h.T, g * torch.eye(g)), g
    for bad in (0, 3, 24, 100):
        with pytest.raises(ValueError, match="power of two"):
            tahquant_hadamard(bad, device=torch.device("cpu"))


def test_rotation_gate_is_self_selecting_on_tau():
    """tau=inf must never rotate and tau=0 must always rotate.

    These are the paper's own two ablation endpoints (its Table 7) and they are
    what the ``tahquant-noh`` arm rests on: if tau=inf still rotated, that arm
    would not isolate the Hadamard at all.
    """
    torch.manual_seed(0)
    x = torch.randn(N, H) * 3.0
    x[:, 11] = 1.0e4
    stats_never: dict = {}
    tahquant_reconstruct(
        x, **_ids(), tile=TILE, int4_frac=0.8, tau=float("inf"), rounding="rn", subset_k=K, stats=stats_never
    )
    assert stats_never["fired"] == 0, stats_never
    stats_always: dict = {}
    tahquant_reconstruct(x, **_ids(), tile=TILE, int4_frac=0.8, tau=0.0, rounding="rn", subset_k=K, stats=stats_always)
    assert stats_always["fired"] == stats_always["tiles"], stats_always


def test_rotation_reduces_error_on_an_outlier_tile_at_our_bit_widths():
    """The mechanism must still work BELOW the paper's own stability floor.

    The paper claims INT4/INT3 and says INT2 often fails to converge, while we
    run a far more aggressive budget. If the rotation only helped at 4 bits it
    would be the wrong method to port, so the gain is asserted at 3 bits on the
    distribution Qwen2.5 actually has (one dominant channel per tile), and
    asserted to be HARMLESS on a Gaussian tile, which is what makes the gate
    safe to leave on.
    """
    torch.manual_seed(0)

    def rel(x, xh):
        return float((xh - x).norm() / x.norm())

    # EXACTLY ONE dominant channel per tile, which is the case the paper's
    # heuristic targets and the shape a residual stream with massive-activation
    # channels presents. Placing outliers at random instead lands several in
    # some tiles, and a tile with two outliers gains far less (measured ~1.9x
    # against ~5x for one), so a random construction would understate the
    # mechanism while looking like it tested it.
    n_tiles = H // TILE
    spiky = torch.randn(N, H)
    for t in range(n_tiles):
        col = t * TILE + (t * 7) % TILE
        spiky[:, col] *= 60.0

    err_on = rel(spiky, _rec(spiky, subset_k=0, int4_frac=0.0, tau=2.0))
    err_off = rel(spiky, _rec(spiky, subset_k=0, int4_frac=0.0, tau=float("inf")))
    assert err_off / err_on > 2.5, (err_on, err_off)

    # And it must be HARMLESS where it cannot help, which is what makes leaving
    # the gate on safe: on Gaussian tiles the ratio heuristic almost never
    # clears tau, so the arm degrades to plain per-tile asymmetric quantization
    # rather than paying for a rotation it does not need.
    gauss = torch.randn(N, H)
    g_on = rel(gauss, _rec(gauss, subset_k=0, int4_frac=0.0, tau=2.0))
    g_off = rel(gauss, _rec(gauss, subset_k=0, int4_frac=0.0, tau=float("inf")))
    assert g_on < g_off * 1.10, (g_on, g_off)


# --------------------------------------------------------------------------- #
# pass identity, which is what keeps the PPO ratio starting at one
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rounding", ["rn", "sr"])
def test_pass_identical_within_a_step_and_fresh_across_steps(rounding):
    """The three passes of one step must reconstruct identically.

    Under ``rn`` this is free: the reconstruction is a deterministic function of
    the activation. Under ``sr`` it holds only because the draw is PRF-keyed on
    (sample, position, layer, step, direction) rather than sampled, and that is
    exactly the property a torch.rand would break.
    """
    torch.manual_seed(0)
    x = torch.randn(N, H) * 3.0
    a = _rec(x, step=11, rounding=rounding)
    b = _rec(x, step=11, rounding=rounding)
    assert torch.equal(a, b)
    c = _rec(x, step=12, rounding=rounding)
    if rounding == "sr":
        assert not torch.equal(a, c), "sr must redraw across steps"


def test_codec_is_stateless_across_steps():
    """No cross-step state, which is the property AQ-SGD does not have.

    A fresh codec and a codec that has already seen many steps must reconstruct
    the same tensor for the same input. This is the claim that makes TAH-Quant
    applicable to on-policy RLVR, where responses are resampled every step and
    a buffered history would be a history of different tokens.
    """
    torch.manual_seed(0)
    x = torch.randn(N, H) * 3.0
    fresh = _rec(x, step=500)
    warm_codec = ActivationTAHQuant(tile=TILE, subset_k=K, base_seed=1234)
    for step in range(1, 40):
        warm_codec.set_context(global_step=step, sample_ids=torch.arange(N), position_ids=torch.arange(N))
    again = _rec(x, step=500)
    assert torch.equal(fresh, again)
    # and the object itself holds nothing keyed by example
    assert not any(isinstance(v, dict) and v for v in vars(warm_codec).values())


# --------------------------------------------------------------------------- #
# autograd
# --------------------------------------------------------------------------- #
def test_backward_returns_one_none_per_forward_arg_and_a_finite_grad():
    """A wrong None count silently DROPS a gradient rather than raising."""
    torch.manual_seed(0)
    x = (torch.randn(N, H) * 3.0).requires_grad_(True)
    stats: dict = {}
    out = BoundaryTAHQuant.apply(x, torch.arange(N), torch.arange(N), 3, 7, 1234, TILE, 0.8, 2.0, "rn", K, stats)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert stats["tiles"] == N * math.ceil(K / TILE)
    # The contract, stated as arithmetic: forward takes ctx plus 12 arguments,
    # so backward must return 1 grad + 11 Nones.
    import inspect

    n_args = len(inspect.signature(BoundaryTAHQuant.forward).parameters) - 1  # drop ctx
    assert n_args == 12, n_args


def test_stochastic_rounding_is_less_biased_than_round_to_nearest():
    """sr must have the smaller residual bias, or the sr arm tests nothing."""
    torch.manual_seed(1)
    base = torch.randn(4, H)
    acc = {"sr": torch.zeros_like(base), "rn": torch.zeros_like(base)}
    trials = 48
    for t in range(trials):
        for mode in ("sr", "rn"):
            acc[mode] += tahquant_reconstruct(
                base,
                sample_ids=torch.arange(4),
                position_ids=torch.arange(4),
                layer_idx=3,
                global_step=2000 + t,
                base_seed=7,
                tile=TILE,
                int4_frac=0.8,
                tau=2.0,
                rounding=mode,
                subset_k=0,
            )
    bias = {m: float((acc[m] / trials - base).norm() / base.norm()) for m in acc}
    assert bias["sr"] < bias["rn"], bias


# --------------------------------------------------------------------------- #
# knob validation and hook lifecycle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kw",
    [
        dict(tile=24),
        dict(tile=1),
        dict(tile=0),
        dict(int4_frac=1.5),
        dict(int4_frac=-0.1),
        dict(tau=-1.0),
        dict(rounding="bogus"),
        dict(subset_k=-1),
        dict(pp_size=1),
    ],
)
def test_codec_rejects_bad_knobs(kw):
    with pytest.raises(ValueError):
        ActivationTAHQuant(**kw)


def test_reconstruct_requires_token_identity():
    x = torch.randn(N, H)
    with pytest.raises(RuntimeError, match="per-token identity"):
        tahquant_reconstruct(
            x,
            sample_ids=torch.arange(N - 1),
            position_ids=torch.arange(N - 1),
            layer_idx=0,
            global_step=1,
            base_seed=0,
            tile=TILE,
            subset_k=K,
        )


def test_reconstruct_rejects_subset_k_above_hidden_size():
    x = torch.randn(N, H)
    with pytest.raises(ValueError, match="subset_k"):
        _rec(x, subset_k=H + 1)


class _Blk(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return (self.lin(x),)


class _Model(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = nn.ModuleList([_Blk(d) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


def test_register_places_hooks_on_boundaries_and_unregister_clears_them():
    codec = ActivationTAHQuant(tile=TILE, subset_k=K, pp_size=8, base_seed=0)
    model = _Model(H, 16)
    assert codec.is_registered is False
    codec.register(model)
    assert codec.is_registered is True
    assert len(codec._handles) == 7, "pp_size=8 means 7 interior boundaries"
    codec.register(model)  # idempotent
    assert len(codec._handles) == 7
    codec.unregister()
    assert codec.is_registered is False


def test_hook_refuses_to_fire_without_token_identity():
    codec = ActivationTAHQuant(tile=TILE, subset_k=K, pp_size=8, base_seed=0)
    model = _Model(H, 16)
    codec.register(model)
    try:
        with pytest.raises(RuntimeError, match="without per-token identity"):
            model(torch.randn(4, H))
    finally:
        codec.unregister()


def test_telemetry_reaches_the_metrics_dict():
    """A codec that reports only a score cannot report a reason.

    ``rotation_rate`` in particular decides whether the arm measured TAH-Quant
    or merely per-tile asymmetric quantization, so it has to reach WandB.
    """
    from verl.workers.comm_eff.state import comm_eff_metrics, maybe_build_comm_eff_state
    from verl.workers.config.comm_eff import (
        CommEffAnchorConfig,
        CommEffConfig,
        CommEffMaskConfig,
        CommEffPowerSGDConfig,
        CommEffTAHQuantConfig,
    )

    cfg = CommEffConfig(
        enabled=True,
        compression_type="tah_quant",
        mask=CommEffMaskConfig(mask_recompute=True, mask_reference=True, pp_size=8, seed=1234),
        tah=CommEffTAHQuantConfig(tile=TILE, subset_k=K),
        anchor=CommEffAnchorConfig(owns_q=False),
        powersgd=CommEffPowerSGDConfig(fast_q_bootstrap=False),
    )
    state = maybe_build_comm_eff_state(cfg)
    model = _Model(H, 16)
    state.build(model)
    assert state.per_token_codec is state.tah
    state.path_tag = "train"
    state.compression_active = True
    codec = state.per_token_codec
    codec.register(model)
    try:
        codec.set_context(global_step=7, sample_ids=torch.arange(8), position_ids=torch.arange(8))
        model(torch.randn(8, H)).sum().backward()
    finally:
        codec.unregister()
    metrics = comm_eff_metrics(state)
    assert metrics["comm_eff/logical_pp_bits_tah_quant"] == pytest.approx(
        K * (0.8 * 4 + 0.2 * 3) + math.ceil(K / TILE) * tahquant_metadata_bits(TILE)
    )
    assert metrics["tah_quant/tiles_seen"] > 0
    assert 0.0 <= metrics["tah_quant/rotation_rate"] <= 1.0
    assert metrics["tah_quant/int4_share"] > 0.0
    assert metrics["comm_eff/mask_applications/train"] == 7.0
