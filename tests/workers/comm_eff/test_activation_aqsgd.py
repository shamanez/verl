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

"""CPU tests for the aq_sgd boundary codec (AQ-SGD, Wang et al. NeurIPS 2022).

Grouped by the claim each test defends. The claims that matter for the RLVR
codec table are: a warm buffer reconstructs exactly, a cold row is
BIT-IDENTICAL to the sr_quant subset codec so delta coding is the single
variable between the two arms, the reconstruction is pass-identical within one
optimizer step so the PPO ratio still starts at one, and differencing against
an unrelated buffer inflates the delta norm by sqrt(2), which is why the scope
knob exists.
"""

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_aqsgd import (
    AQSGD_FIRST_VISIT_MODES,
    AQSGD_SCOPES,
    ActivationAQSGD,
    AQSGDBuffer,
    aqsgd_example_ids,
    aqsgd_reconstruct,
)
from verl.workers.comm_eff.activation_quant import sr_quantize

# Pair 1's operating point: H=1536, and bits=2 with k=493 costs
# 493*2 + 493*16/32 = 1232.5 bits/token/boundary against PRF exact-k's
# 77*16 = 1232. Tests use a narrower H for speed but keep the k/H ratio.
H = 256
K = 82  # 32.03% coverage, matching k=493 of 1536
BITS = 2
BLK = 32
N = 48


def _ids(n=N, step=7):
    return dict(
        sample_ids=torch.arange(n),
        position_ids=torch.arange(n),
        layer_idx=3,
        global_step=step,
        base_seed=1234,
    )


def _rec(x, m, warm, *, step=7, rounding="sr", first_visit="rescaled", subset_k=K):
    return aqsgd_reconstruct(
        x,
        m,
        warm,
        **_ids(x.shape[0], step),
        bits=BITS,
        block_size=BLK,
        subset_k=subset_k,
        rounding=rounding,
        first_visit=first_visit,
    )


# --------------------------------------------------------------------------- #
# the buffer's defining property: an exact history reconstructs exactly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rounding", ["sr", "rn"])
@pytest.mark.parametrize("subset_k", [0, K])
def test_exact_buffer_reconstructs_exactly(rounding, subset_k):
    """m == x makes the delta zero, so Q(0) = 0 and x_hat == x.

    This is what AQ-SGD buys and what a memoryless codec cannot do at any bit
    width: once the buffer has caught up, the wire cost of staying caught up
    is the cost of encoding zero.
    """
    torch.manual_seed(0)
    x = torch.randn(N, H)
    warm = torch.ones(N, dtype=torch.bool)
    xh = _rec(x, x.clone(), warm, rounding=rounding, subset_k=subset_k)
    assert torch.allclose(xh, x, atol=1e-6), (xh - x).abs().max()


def test_massive_activation_channel_survives_the_buffer_round_trip():
    """The regression test for the nan that killed the first three aq_sgd arms.

    The chain, which no single-step test could see: a COLD row under
    ``rescaled`` reconstructs as ``(H/k) * scatter_J(Q(x_J))``, so what
    ``_stage_buffer`` writes is inflated by the gain (3.1156 at Pair 1). Qwen2.5
    carries massive-activation channels in the residual stream, and the gain
    lifts them over fp16's 65504 ceiling at a true magnitude of only ~21k. The
    entry saturates to inf, and NOTHING GOES WRONG UNTIL THE FIRST WARM READ,
    which at Pair 1 is one full epoch later at step 59: the delta against inf
    has an infinite block absmax, spacing goes with it, and
    ``floor((m + s) / spacing)`` is ``floor(nan)``. One saturated channel
    returns nan for all 32 channels of its block, on every row sharing it.

    So the assertion is on the WHOLE chain (cold -> store -> warm), not on the
    reconstruction alone, and it sweeps magnitudes that bracket the fp16 limit.
    """
    from verl.workers.comm_eff.activation_aqsgd import _BUFFER_DTYPE

    torch.manual_seed(0)
    for peak in (1.0e3, 1.0e4, 3.0e4, 1.0e5, 1.0e6):
        x = torch.randn(N, H) * 3.0
        x[:, 7] = peak  # one massive-activation channel, as in a real residual stream
        x_hat = _rec(x, None, None, step=1)
        assert torch.isfinite(x_hat).all(), f"cold path went non-finite at peak={peak:g}"
        assert x_hat.abs().max() >= peak, "the rescaled cold path must carry the H/k gain"

        # the buffer round trip, at the dtype the store actually uses
        stored = x_hat.detach().to(_BUFFER_DTYPE).to(torch.float32)
        assert torch.isfinite(stored).all(), (
            f"buffer dtype {_BUFFER_DTYPE} saturated at peak={peak:g} "
            f"(|x_hat|max={float(x_hat.abs().max()):.4g}); a warm read against it returns nan"
        )

        warm = torch.ones(N, dtype=torch.bool)
        x2 = x + torch.randn(N, H) * 0.05
        out = _rec(x2, stored, warm, step=59)
        assert torch.isfinite(out).all(), f"warm read went non-finite at peak={peak:g}"


def test_buffer_dtype_has_the_range_the_gain_demands():
    """State the headroom as a number, so a dtype change cannot quietly undo the fix.

    fp16 tops out at 65504, which the H/k gain reaches from a true activation of
    only ~21k. bf16 reaches 3.4e38. The buffer must be a dtype whose finite max
    clears the gain-inflated scale by a wide margin.
    """
    from verl.workers.comm_eff.activation_aqsgd import _BUFFER_DTYPE, _BUFFER_ITEMSIZE

    finfo = torch.finfo(_BUFFER_DTYPE)
    gain = 1536.0 / 493.0  # Pair 1
    # Qwen2.5 residual-stream outliers are observed in the 1e4-1e5 band; the
    # buffer has to hold gain * that with room to spare.
    assert finfo.max / gain > 1.0e6, (
        f"{_BUFFER_DTYPE} tolerates true activations only up to "
        f"{finfo.max / gain:.4g}; fp16's 65504 is what produced the step-59 nan"
    )
    assert _BUFFER_ITEMSIZE == torch.empty(0, dtype=_BUFFER_DTYPE).element_size()
    assert _BUFFER_ITEMSIZE == 2, "the 16 GiB capacity budget assumes two bytes per channel"


def test_nonfinite_buffered_row_is_demoted_to_cold_and_counted():
    """The guard behind the dtype fix: a bad row degrades the codec, never the loss.

    If a non-finite value ever reaches the buffer again, the row must fall back
    to the sr_quant cold path rather than return nan, and the demotion must be
    COUNTED - a silent demotion is indistinguishable from a low hit rate.
    """
    codec = ActivationAQSGD(scope="all", bits=BITS, block_size=BLK, subset_k=K, base_seed=1234)
    n_pos = 4
    good = torch.randn(n_pos, H)
    bad = good.clone()
    bad[2, 5] = float("inf")
    codec.buffer.put(11, 3, good, step=1)
    codec.buffer.put(22, 3, bad, step=1)

    example_ids = torch.tensor([11] * n_pos + [22] * n_pos)
    position_ids = torch.cat([torch.arange(n_pos), torch.arange(n_pos)])
    in_scope = torch.ones(2 * n_pos, dtype=torch.bool)
    m, warm = codec._gather_buffer(3, example_ids, position_ids, in_scope, H, torch.device("cpu"), torch.float32)

    assert warm[:n_pos].all(), "the finite example must stay warm"
    assert bool(warm[n_pos + 2]) is False, "the row holding inf must be demoted to cold"
    assert warm[n_pos + 0] and warm[n_pos + 1] and warm[n_pos + 3], "only the bad ROW is demoted"
    assert torch.isfinite(m).all(), "a demoted row must leave a finite m"
    assert codec.buffer.nonfinite_rows == 1
    assert codec.buffer.telemetry()["aq_sgd/nonfinite_rows"] == 1.0

    x = torch.randn(2 * n_pos, H)
    out = aqsgd_reconstruct(
        x,
        m,
        warm,
        sample_ids=torch.arange(2 * n_pos),
        position_ids=position_ids,
        layer_idx=3,
        global_step=59,
        base_seed=1234,
        bits=BITS,
        block_size=BLK,
        subset_k=K,
        rounding="sr",
        first_visit="rescaled",
    )
    assert torch.isfinite(out).all()


def test_cold_row_is_bit_identical_to_sr_quant():
    """A cold row IS the sr_quant subset codec, which is what isolates the variable.

    The aq_sgd arm and an sr_quant arm at the same (bits, subset_k,
    block_size) therefore differ in exactly one thing: whether a buffered
    history is differenced against. Any gap between those two arms is
    attributable to delta coding and nothing else.
    """
    torch.manual_seed(1)
    x = torch.randn(N, H)
    aq = _rec(x, None, None, rounding="sr")
    srq = sr_quantize(
        x,
        torch.arange(N),
        torch.arange(N),
        layer_idx=3,
        global_step=7,
        base_seed=1234,
        bits=BITS,
        block_size=BLK,
        subset_k=K,
    )
    assert torch.equal(aq, srq)


def test_warm_refinement_reaches_the_untouched_share():
    """With J held fixed, refinement converges to sqrt(1 - k/H) exactly.

    A fixed subset can only ever correct its own k channels, so the residual
    floor is the norm share of the H-k it never sends. Hitting that floor
    proves the kept channels are corrected COMPLETELY and the rest are left at
    m, rather than being zeroed or rescaled.
    """
    torch.manual_seed(2)
    x = torch.randn(N, H)
    warm = torch.ones(N, dtype=torch.bool)
    m = _rec(x, None, None, rounding="rn")
    for _ in range(8):
        m = _rec(x, m, warm, rounding="rn")
    rel = float((m - x).norm() / x.norm())
    floor = (1.0 - K / H) ** 0.5
    assert rel == pytest.approx(floor, rel=0.02), (rel, floor)


def test_unrelated_buffer_inflates_the_delta_by_sqrt_two():
    """The reason scope='prompt' exists, measured rather than asserted.

    For roughly independent x and m, E||x - m||^2 = ||x||^2 + ||m||^2, so the
    delta carries sqrt(2) times the norm of the value. At a fixed bit budget
    that is a coarser grid and a LARGER error than quantizing the value
    directly, which is what buffering a resampled RLVR response token does.
    """
    torch.manual_seed(3)
    x = torch.randn(N, H)
    m = torch.randn(N, H)
    ratio = float(((x - m).norm(dim=1) / x.norm(dim=1)).mean())
    assert ratio == pytest.approx(2.0**0.5, rel=0.05), ratio

    # The consequence, measured LIKE FOR LIKE at full width so the only
    # difference is what got quantized. Warm rows quantize the delta, whose
    # norm sets the blockwise absmax grid; cold rows quantize the value. The
    # error therefore scales with the norm being encoded, so an unrelated
    # buffer costs a factor of about sqrt(2) at the same bit width.
    #
    # Note this is NOT the same as saying a warm row is always worse overall:
    # in SUBSET mode an unrelated buffer still beats the cold path, because
    # holding a wrong-but-correctly-scaled value on the H-k un-sent channels
    # beats zeroing them and rescaling the rest by H/k. The penalty is
    # specifically in the grid, which is what this measures.
    warm = torch.ones(N, dtype=torch.bool)
    err_warm = float((_rec(x, m, warm, rounding="sr", subset_k=0) - x).norm())
    err_cold = float((_rec(x, None, None, rounding="sr", subset_k=0) - x).norm())
    assert err_warm / err_cold == pytest.approx(2.0**0.5, rel=0.10), (err_warm, err_cold)


# --------------------------------------------------------------------------- #
# path identity: the PPO ratio must still start each step at one
# --------------------------------------------------------------------------- #
def test_pass_identical_within_a_step_and_fresh_across_steps():
    torch.manual_seed(4)
    x = torch.randn(N, H)
    m = torch.randn(N, H) * 0.1
    warm = torch.ones(N, dtype=torch.bool)
    a = _rec(x, m, warm, step=11)
    b = _rec(x, m, warm, step=11)
    assert torch.equal(a, b), "two passes of one step must agree bit-for-bit"
    c = _rec(x, m, warm, step=12)
    assert not torch.equal(a, c), "a new step must draw a fresh subset J"


def test_first_visit_dense_passes_x_through_untouched():
    """The faithful first message, off-budget by construction."""
    torch.manual_seed(5)
    x = torch.randn(N, H)
    xh = _rec(x, None, None, first_visit="dense")
    assert torch.equal(xh, x)


def test_first_visit_dense_leaves_warm_rows_compressed():
    """Only COLD rows bypass the codec; a warm row still sends a delta."""
    torch.manual_seed(6)
    x = torch.randn(N, H)
    warm = torch.zeros(N, dtype=torch.bool)
    warm[: N // 2] = True
    m = torch.randn(N, H) * 0.1
    xh = _rec(x, m, warm, first_visit="dense")
    assert torch.equal(xh[N // 2 :], x[N // 2 :]), "cold half is dense"
    assert not torch.equal(xh[: N // 2], x[: N // 2]), "warm half is compressed"


# --------------------------------------------------------------------------- #
# the backward wire is direct quantization, with no buffer
# --------------------------------------------------------------------------- #
def test_backward_is_direct_and_unbiased_under_sr():
    """AQ-SGD quantizes the gradient directly: a gradient has no history.

    Averaged over the PRF draw the returned gradient is the true one, so the
    backward wire adds variance and no bias.
    """
    torch.manual_seed(7)
    from verl.workers.comm_eff.activation_aqsgd import BoundaryAQSGD

    g_true = torch.randn(N, H)
    acc = torch.zeros(N, H)
    trials = 400
    for t in range(trials):
        h = torch.randn(N, H, requires_grad=True)
        out = BoundaryAQSGD.apply(
            h,
            None,
            None,
            torch.arange(N),
            torch.arange(N),
            3,
            1000 + t,
            1234,
            BITS,
            BLK,
            "sr",
            K,
            "rescaled",
        )
        out.backward(g_true)
        acc += h.grad
    mean = acc / trials
    rel = float((mean - g_true).norm() / g_true.norm())
    assert rel < 0.12, rel


# --------------------------------------------------------------------------- #
# the persistent example identity
# --------------------------------------------------------------------------- #
def test_example_ids_are_persistent_ordered_and_nonnegative():
    prompts = torch.tensor([[5, 9, 2, 7], [5, 9, 2, 7], [1, 4, 1, 4], [2, 9, 5, 7]])
    mask = torch.ones_like(prompts)
    ids = aqsgd_example_ids(prompts, mask)
    assert ids[0] == ids[1], "the same prompt must hash the same, that is the point"
    assert ids[0] != ids[2], "different prompts must differ"
    assert ids[0] != ids[3], "a permutation must differ: position enters the key"
    assert bool((ids >= 0).all()), "ids index a dict and must be non-negative"


def test_example_ids_ignore_padding():
    """Left padding must not change a prompt's identity across batches."""
    a = aqsgd_example_ids(torch.tensor([[5, 9, 2]]), torch.tensor([[1, 1, 1]]))
    b = aqsgd_example_ids(torch.tensor([[0, 0, 5, 9, 2]]), torch.tensor([[0, 0, 1, 1, 1]]))
    # Position is folded in, so equality holds only when the real tokens sit at
    # the same positions; this asserts the masked slots contribute nothing.
    c = aqsgd_example_ids(torch.tensor([[5, 9, 2, 123]]), torch.tensor([[1, 1, 1, 0]]))
    assert a[0] == c[0], "a masked trailing slot must not change the id"
    assert a[0] != b[0], "a shifted prompt is a different key by design"


def test_example_ids_rejects_bad_shapes():
    with pytest.raises(ValueError):
        aqsgd_example_ids(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        aqsgd_example_ids(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long))


# --------------------------------------------------------------------------- #
# the buffer: capacity, accounting, and the idempotent write
# --------------------------------------------------------------------------- #
def test_buffer_lru_respects_capacity():
    slab = torch.zeros(4, 64)  # 4*64*2 = 512 bytes per entry
    buf = AQSGDBuffer(capacity_bytes=512 * 3, device="cpu")
    for i in range(6):
        buf.put(i, 0, slab, step=i)
    assert buf.bytes_used <= 512 * 3
    assert buf.evictions >= 3
    assert buf.get(0, 0) is None, "the oldest entry must have been evicted"
    assert buf.get(5, 0) is not None, "the newest entry must survive"


def test_buffer_hit_rate_is_counted_in_token_positions():
    buf = AQSGDBuffer(capacity_bytes=1 << 20)
    buf.hits, buf.misses = 3, 1
    assert buf.hit_rate == pytest.approx(0.75)
    assert buf.telemetry()["aq_sgd/hit_rate"] == pytest.approx(0.75)


def test_buffer_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        AQSGDBuffer(capacity_bytes=0)


# --------------------------------------------------------------------------- #
# construction guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kw",
    [
        {"bits": 0},
        {"block_size": -1},
        {"rounding": "nonsense"},
        {"subset_k": -1},
        {"scope": "nonsense"},
        {"first_visit": "nonsense"},
        {"max_positions": -1},
    ],
)
def test_codec_rejects_bad_knobs(kw):
    with pytest.raises(ValueError):
        ActivationAQSGD(**kw)


def test_scope_and_first_visit_vocabularies_are_closed():
    assert AQSGD_SCOPES == ("prompt", "all")
    assert AQSGD_FIRST_VISIT_MODES == ("rescaled", "dense")


def test_reconstruct_rejects_subset_k_above_hidden_size():
    with pytest.raises(ValueError):
        aqsgd_reconstruct(
            torch.randn(4, 8),
            None,
            None,
            torch.arange(4),
            torch.arange(4),
            layer_idx=0,
            global_step=0,
            base_seed=0,
            bits=2,
            subset_k=9,
        )


def test_reconstruct_requires_token_identity():
    with pytest.raises(RuntimeError):
        aqsgd_reconstruct(
            torch.randn(4, 8),
            None,
            None,
            None,
            None,
            layer_idx=0,
            global_step=0,
            base_seed=0,
            bits=2,
            subset_k=4,
        )


# --------------------------------------------------------------------------- #
# the wire ledger, which is what makes an arm byte-matched
# --------------------------------------------------------------------------- #
def test_wire_ledger_matches_prf_exact_k_at_pair_one():
    """bits=2, k=493 at H=1536 must price out against PRF's 1232 bits."""
    codec = ActivationAQSGD(bits=2, block_size=32, subset_k=493)
    codec._record_bits(1536)
    assert codec.logical_pp_bits_aq_sgd == pytest.approx(1232.5)
    prf_bits = 77 * 16  # k=77 coordinates at fp16
    assert codec.logical_pp_bits_aq_sgd / prf_bits == pytest.approx(1.0, abs=0.001)


def test_wire_ledger_full_width_counts_every_channel_plus_scales():
    codec = ActivationAQSGD(bits=2, block_size=32, subset_k=0)
    codec._record_bits(1536)
    assert codec.logical_pp_bits_aq_sgd == pytest.approx(1536 * 2 + (1536 // 32) * 16)


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
    def __init__(self, num_layers=16, d=H):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(d) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _State:
    """The minimal CommEffState surface the hook's confinement guard reads."""

    def __init__(self, tag="train"):
        self.path_tag = tag
        self.mask_applications = 0
        self.config = None


def test_register_places_hooks_on_boundaries_and_unregister_clears_them():
    model = _ToyDecoder(num_layers=16)
    codec = ActivationAQSGD(pp_size=8, subset_k=K, bits=BITS, state=_State())
    codec.register(model)
    assert codec.is_registered
    assert len(codec.boundary_indices) == 7, codec.boundary_indices
    codec.unregister()
    assert not codec.is_registered


def test_passes_of_one_step_are_identical_and_the_stage_publishes_at_the_boundary():
    """The regression this codec's whole path-identity argument rests on.

    A second forward at the same step must reconstruct IDENTICALLY. A plain
    "skip the second write" guard does not achieve that: the second pass would
    read what the first pass published and diverge. Publication is therefore
    deferred to the step boundary, and this test pins both halves of that -
    passes within a step agree, and the buffer appears only once the step
    advances.
    """
    torch.manual_seed(8)
    model = _ToyDecoder(num_layers=16)
    state = _State(tag="train")
    codec = ActivationAQSGD(pp_size=8, subset_k=K, bits=BITS, block_size=BLK, scope="prompt", state=state)
    codec.register(model)
    try:
        n_tokens, prompt_len = 12, 5
        x = torch.randn(n_tokens, H)

        def ctx(step):
            return dict(
                global_step=step,
                sample_ids=torch.zeros(n_tokens, dtype=torch.long),
                position_ids=torch.arange(n_tokens),
                example_ids=torch.full((n_tokens,), 999, dtype=torch.long),
                prompt_lens=torch.full((n_tokens,), prompt_len, dtype=torch.long),
            )

        # Step 3, pass 1 (the train forward): stages, publishes nothing.
        codec.set_context(**ctx(3))
        first = model(x)
        assert len(codec.buffer) == 0, "nothing may be published mid-step"
        assert len(codec._pending) == 7, "one stage per boundary"

        # Step 3, pass 2 (the gradient-checkpoint recompute): must reproduce.
        codec.set_context(**ctx(3))
        second = model(x)
        assert torch.equal(first, second), "the recompute must reproduce the forward"

        # Step 4 opens: the stage from step 3 is published and readable.
        codec.set_context(**ctx(4))
        assert len(codec.buffer) == 7, "one entry per boundary after the boundary"
        entry = codec.buffer.get(999, codec.boundary_indices[0])
        assert entry is not None and entry[1] == 3, "published under the step that staged it"
        assert entry[0].shape[0] == prompt_len, "scope='prompt' bounds the slab"

        # Step 4 now reads a WARM buffer, so it must differ from the cold step 3.
        third = model(x)
        assert not torch.equal(first, third), "step 4 reads the published buffer"
        assert codec.buffer.hit_rate > 0.0, "the prefix positions scored hits"
    finally:
        codec.unregister()


def test_hook_refuses_to_fire_without_an_example_id():
    """Silently degenerating into memoryless quantization is the failure to avoid."""
    model = _ToyDecoder(num_layers=16)
    codec = ActivationAQSGD(pp_size=8, subset_k=K, bits=BITS, state=_State())
    codec.register(model)
    try:
        codec.set_context(
            global_step=1,
            sample_ids=torch.zeros(4, dtype=torch.long),
            position_ids=torch.arange(4),
        )
        with pytest.raises(RuntimeError, match="example_ids"):
            model(torch.randn(4, H))
    finally:
        codec.unregister()


def test_reference_pass_reads_the_buffer_but_never_writes_it():
    """The reference forward runs different weights, so its activation is not this visit's."""
    torch.manual_seed(9)
    model = _ToyDecoder(num_layers=16)
    state = _State(tag="ref_logprob")
    codec = ActivationAQSGD(pp_size=8, subset_k=K, bits=BITS, scope="prompt", state=state)
    # The confinement guard consults mask_eligible_tags(state); a config whose
    # mask flags open the reference path is what makes this legal.
    from verl.workers.config.comm_eff import (
        CommEffAnchorConfig,
        CommEffAQSGDConfig,
        CommEffConfig,
        CommEffMaskConfig,
        CommEffPowerSGDConfig,
        CommEffSpectralConfig,
    )

    state.config = CommEffConfig(
        enabled=True,
        compression_type="aq_sgd",
        mask=CommEffMaskConfig(mask_recompute=True, mask_reference=True, pp_size=8),
        aq_sgd=CommEffAQSGDConfig(bits=BITS, subset_k=K, block_size=BLK),
        anchor=CommEffAnchorConfig(enabled=False, owns_q=False, lookahead_mode="disabled", lookahead_min_snapshots=-1),
        spectral=CommEffSpectralConfig(enabled=False),
        powersgd=CommEffPowerSGDConfig(enabled=False, fast_q_bootstrap=False),
    )
    codec.register(model)
    try:
        n = 8
        codec.set_context(
            global_step=5,
            sample_ids=torch.zeros(n, dtype=torch.long),
            position_ids=torch.arange(n),
            example_ids=torch.full((n,), 7, dtype=torch.long),
            prompt_lens=torch.full((n,), 4, dtype=torch.long),
        )
        model(torch.randn(n, H))
        assert len(codec.buffer) == 0, "the reference pass must not advance the buffer"
    finally:
        codec.unregister()


def test_buffer_capacity_default_agrees_at_every_layer():
    """The fanout's host-RAM gate reserves a fixed per-arm budget, so a layer
    that defaulted higher on its own would silently overrun it.

    The gate reserves 32 GB + AQ_CAPACITY_GB per codec arm. If the dataclass,
    the module, Hydra or the engine script defaulted to more than the launchers
    do, an arm started without the env set would exceed the reservation, and
    three such arms would exceed it by three times that. This pins all of them
    to one figure.
    """
    import re
    from pathlib import Path

    from verl.workers.comm_eff.activation_aqsgd import _DEFAULT_CAPACITY_BYTES
    from verl.workers.config.comm_eff import CommEffAQSGDConfig

    expected = 16 * (1024**3)
    root = Path(__file__).resolve().parents[3]

    def _grep(rel, pattern):
        m = re.search(pattern, (root / rel).read_text())
        assert m, f"{pattern!r} not found in {rel}"
        return int(m.group(1))

    layers = {
        "dataclass": CommEffAQSGDConfig().capacity_bytes,
        "module": _DEFAULT_CAPACITY_BYTES,
        "codec object": ActivationAQSGD().buffer.capacity_bytes,
        "actor.yaml": _grep("verl/trainer/config/actor/actor.yaml", r"capacity_bytes: (\d+)"),
        "generated.yaml": _grep("verl/trainer/config/_generated_ppo_trainer.yaml", r"capacity_bytes: (\d+)"),
        "engine.sh": _grep(
            "examples/grpo_trainer/vast_comm_eff_engine_grpo.sh",
            r"COMM_EFF_AQ_SGD_CAPACITY_BYTES:-(\d+)",
        ),
    }
    assert set(layers.values()) == {expected}, layers

    # The enumeration above is REMEMBERED, and a remembered list certifies only
    # the layers its author thought of: state.py's own getattr fallback was
    # missing from it and had drifted to 24 GiB. So DERIVE the check too, by
    # scanning the package for any numeric capacity_bytes default. A new layer
    # is then covered the moment it is written, without anyone updating a tuple.
    derived = {}
    for path in sorted((root / "verl").rglob("*.py")):
        for m in re.finditer(
            r"capacity_bytes[\s\S]{0,40}?(\d+\s*\*\s*\(1024\*\*3\)|\d{6,})",
            path.read_text(),
        ):
            derived[f"{path.relative_to(root)}:{m.group(1)}"] = eval(m.group(1))  # noqa: S307
    assert derived, "the capacity_bytes scan matched nothing, so the regex has rotted"
    assert set(derived.values()) == {expected}, derived

    # The launcher-side knob is expressed in GiB and must be the same figure.
    for rel in (
        "examples/grpo_trainer/run_compass_rlvr_ablations_fsdp.sh",
        "examples/grpo_trainer/run_compass_rlvr_ablations_fanout_fsdp.sh",
    ):
        assert _grep(rel, r"AQ_CAPACITY_GB:-(\d+)") * 1024**3 == expected, rel


def test_aqsgd_telemetry_reaches_the_metrics_dict():
    """The hit rate is what EXPLAINS the arm, so it must reach the logger.

    comm_eff_metrics() had a quant_metrics() branch for sr_quant and none for
    aq_sgd, so the codec computed hit_rate and delta_ratio and then threw them
    away. Without them the ablation can report an accuracy number but not a
    reason, and the paper's appendix promises the reason.
    """
    import torch.nn as nn

    from verl.workers.comm_eff.state import comm_eff_metrics, maybe_build_comm_eff_state
    from verl.workers.config.comm_eff import (
        CommEffAnchorConfig,
        CommEffAQSGDConfig,
        CommEffConfig,
        CommEffMaskConfig,
        CommEffPowerSGDConfig,
        CommEffSpectralConfig,
    )

    cfg = CommEffConfig(
        enabled=True,
        compression_type="aq_sgd",
        mask=CommEffMaskConfig(enabled=False, mask_recompute=True, mask_reference=True, pp_size=8),
        aq_sgd=CommEffAQSGDConfig(bits=2, subset_k=K, block_size=BLK),
        anchor=CommEffAnchorConfig(enabled=True, owns_q=False, cadence=20, delay_K=20, replay_paired_batch=True),
        spectral=CommEffSpectralConfig(enabled=True),
        powersgd=CommEffPowerSGDConfig(enabled=False, fast_q_bootstrap=False),
    )

    class _Blk(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, x):
            return self.lin(x)

    class _Dec(nn.Module):
        def __init__(self, n=16, d=H):
            super().__init__()
            self.layers = nn.ModuleList([_Blk(d) for _ in range(n)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    state = maybe_build_comm_eff_state(cfg)
    model = _Dec()
    state.build(model)
    assert state.aqsgd is not None, "the aq_sgd codec must be the one built"
    state.aqsgd.register(model)
    state.set_path_tag("train")
    state.compression_active = True
    try:
        n = 16
        for step in (1, 2):
            state.global_step = step
            state.aqsgd.set_context(
                global_step=step,
                sample_ids=torch.zeros(n, dtype=torch.long),
                position_ids=torch.arange(n),
                example_ids=torch.full((n,), 42, dtype=torch.long),
                prompt_lens=torch.full((n,), 8, dtype=torch.long),
            )
            with torch.no_grad():
                model(torch.randn(n, H))
    finally:
        state.aqsgd.unregister()
        state.set_path_tag(None)

    mets = comm_eff_metrics(state)
    for key in (
        "aq_sgd/hit_rate",
        "aq_sgd/delta_ratio",
        "aq_sgd/buffer_gib",
        "aq_sgd/evictions",
        "comm_eff/logical_pp_bits_aq_sgd",
    ):
        assert key in mets, f"{key} missing from comm_eff_metrics; it would never reach WandB"
        assert isinstance(mets[key], float), key

    # Step 1 is cold and step 2 is warm over the same 8 prefix positions, so the
    # hit rate over the run is exactly one half. This pins that the counters
    # measure token POSITIONS rather than entries or fires.
    assert mets["aq_sgd/hit_rate"] == pytest.approx(0.5), mets["aq_sgd/hit_rate"]
    # The ledger is computed from THIS module's narrow geometry (H, K), not the
    # Pair 1 one; test_wire_ledger_matches_prf_exact_k_at_pair_one pins 1232.5.
    assert mets["comm_eff/logical_pp_bits_aq_sgd"] == pytest.approx(K * BITS + K * 16 / BLK)


# --------------------------------------------------------------------------- #
# integration gates: the two blockers a hand-called register() cannot catch
# --------------------------------------------------------------------------- #
def _pair1_aqsgd_config():
    from verl.workers.config.comm_eff import (
        CommEffAnchorConfig,
        CommEffAQSGDConfig,
        CommEffConfig,
        CommEffMaskConfig,
        CommEffPowerSGDConfig,
        CommEffSpectralConfig,
    )

    return CommEffConfig(
        enabled=True,
        compression_type="aq_sgd",
        mask=CommEffMaskConfig(enabled=False, mask_recompute=True, mask_reference=True, pp_size=8),
        aq_sgd=CommEffAQSGDConfig(bits=2, subset_k=K, block_size=BLK),
        anchor=CommEffAnchorConfig(enabled=True, owns_q=False, cadence=20, delay_K=20, replay_paired_batch=True),
        spectral=CommEffSpectralConfig(enabled=True),
        powersgd=CommEffPowerSGDConfig(enabled=False, fast_q_bootstrap=False),
    )


def test_engine_registration_gate_opens_for_aq_sgd():
    """The blocker: the gate enumerated masker/quantizer and not aqsgd.

    Consequence, measured on a live box before this test existed: three aq_sgd
    arms reported comm_eff/mask_applications = 0 while the matched sr_quant arm
    reported 10458. No hook was ever installed, so the arms were bit-for-bit
    DENSE while every pre-flight gate printed "resolved codec OK: aq_sgd".
    Four GPU-days would have produced dense runs labelled AQ-SGD.

    This asserts the gate expression the engine actually evaluates, so it fails
    if anyone re-enumerates codecs by hand at that site.
    """
    import torch.nn as nn

    from verl.workers.comm_eff.state import maybe_build_comm_eff_state

    class _Blk(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, x):
            return self.lin(x)

    class _Dec(nn.Module):
        def __init__(self, n=16, d=H):
            super().__init__()
            self.layers = nn.ModuleList([_Blk(d) for _ in range(n)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    state = maybe_build_comm_eff_state(_pair1_aqsgd_config())
    # The codec object is created in build(), not the constructor, so the gate
    # is only meaningful after the engine has built the state.
    state.build(_Dec())
    assert state.masker is None and state.quantizer is None, (
        "aq_sgd populates neither of the two the old gate looked at, which is the trap"
    )
    assert state.aqsgd is not None
    assert state.per_token_codec is state.aqsgd, "the gate must resolve the live codec"

    # The gate's own predicate, verbatim from transformer_impl._comm_eff_mask_active.
    assert getattr(state, "per_token_codec", None) is not None, "registration gate would refuse aq_sgd"


def test_every_engine_codec_site_uses_per_token_codec():
    """No site may enumerate codecs by hand; that is what caused both blockers.

    Two sites were blind: the registration gate (silent dense) and the teardown
    in forward_backward_batch's finally (which, once the gate was fixed, would
    have raised AttributeError on None.unregister). A third, the per-tag metric,
    left the counter ABSENT rather than zero.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]

    # DERIVE the file list, do not remember it. On the first pass this tuple was
    # hand-written, it omitted engine_workers.py, and that omission is exactly
    # why two more sites survived: boundary_codec was resolved
    # masker-or-quantizer there, so compression_active was never set on the
    # reference and old-logprob paths and the codec fired on the train forward
    # alone. A list an author maintains certifies only the files that author
    # thought of, so scan for anything that reaches a codec slot.
    # A file "touches the codec" if it reads a slot by name OR routes through
    # the helper. Both belong in the scan: the first is the shape that was
    # wrong, the second is where a future author will add the next site.
    # Matches attribute access (``state.masker``) AND the getattr-by-string
    # form (``getattr(comm_eff_state, "per_token_codec", None)``), which is how
    # engine_workers.py actually reaches it. Over-matching is safe here: an
    # extra file just gets checked against the negative patterns below, whereas
    # under-matching is the failure this whole test exists to prevent.
    slot_ref = re.compile(r"""(?:\.|["'])(?:masker|quantizer|aqsgd|per_token_codec)\b""")
    codec_modules = {
        "verl/workers/comm_eff/activation_mask.py",
        "verl/workers/comm_eff/activation_quant.py",
        "verl/workers/comm_eff/activation_aqsgd.py",
    }
    scanned = []
    for path in sorted((root / "verl").rglob("*.py")):
        rel_path = str(path.relative_to(root))
        if rel_path in codec_modules:
            continue
        if slot_ref.search(path.read_text()):
            scanned.append(rel_path)
    assert scanned, "the codec-slot scan matched no files, so the regex has rotted"
    for expected in (
        "verl/workers/engine/fsdp/transformer_impl.py",
        "verl/workers/comm_eff/state.py",
        "verl/workers/engine_workers.py",
    ):
        assert expected in scanned, f"{expected} reaches a codec slot but the scan missed it"

    for rel in scanned:
        text = (root / rel).read_text()
        # The exact shape that was wrong: masker and quantizer named together
        # with no aqsgd and no per_token_codec on the same logical line.
        for pat in (
            # "." excludes newlines by default, so these stay within one line.
            r"state\.masker.{0,40}is None and.{0,40}state\.quantizer.{0,20}is None",
            r"self\.masker is None and self\.quantizer is None",
            r"_codec = self\._comm_eff_state\.masker",
            r"boundary_codec = getattr\(comm_eff_state, \"masker\"",
        ):
            hits = [m for m in re.finditer(pat, text)]
            assert not hits, (
                f"{rel} still enumerates codecs by hand ({pat!r}); use state.per_token_codec, "
                "which enumerates masker/quantizer/aqsgd"
            )


def test_per_token_codec_covers_every_slot_build_assigns():
    """The positive half, because absence of old regexes is not presence of the fix.

    Every site above now routes through ``state.per_token_codec``, which makes
    the ONE tuple inside it a single point of failure for all of them: add a
    fourth codec, forget that line, and every site fails together while this
    module's negative scan still passes green.

    So derive the slots from ``build()``, the one place an author cannot forget,
    because a codec that is never constructed is never used at all.
    """
    import inspect
    import re

    from verl.workers.comm_eff.state import CommEffState

    build_src = inspect.getsource(CommEffState.build)
    assigned = set(re.findall(r"self\.(\w+) = Activation\w+\(", build_src))
    assert assigned, "no codec assignment found in build(), so the pattern has rotted"

    enumerated = set(re.findall(r"self\.(\w+)", inspect.getsource(CommEffState.per_token_codec.fget)))
    missing = assigned - enumerated
    assert not missing, (
        f"build() assigns codecs to {sorted(missing)} but per_token_codec does not enumerate them, "
        "so every engine site that routes through it would be blind to those codecs"
    )


def test_register_does_not_discard_the_staged_buffer():
    """The second blocker: register() cleared _pending.

    The engine registers and unregisters once per eligible PASS, not once per
    run, so clearing the stage in register() destroyed the train pass's staged
    buffer before the next step could publish it. The hit rate would have read
    0 for the whole run even with the codec firing, and the ablation's entire
    explanation is the hit rate.
    """
    import torch.nn as nn

    class _Blk(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, x):
            return self.lin(x)

    class _Dec(nn.Module):
        def __init__(self, n=16, d=H):
            super().__init__()
            self.layers = nn.ModuleList([_Blk(d) for _ in range(n)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = _Dec()
    state = _State(tag="train")
    codec = ActivationAQSGD(pp_size=8, subset_k=K, bits=BITS, block_size=BLK, scope="prompt", state=state)
    n = 12

    def ctx(step):
        return dict(
            global_step=step,
            sample_ids=torch.zeros(n, dtype=torch.long),
            position_ids=torch.arange(n),
            example_ids=torch.full((n,), 5, dtype=torch.long),
            prompt_lens=torch.full((n,), 6, dtype=torch.long),
        )

    torch.manual_seed(0)
    x = torch.randn(n, H)

    # Step 1: register / forward / unregister, exactly as the engine does per pass.
    codec.register(model)
    codec.set_context(**ctx(1))
    model(x)
    codec.unregister()
    assert len(codec._pending) == 7, "the train pass must stage one slab per boundary"

    # Step 2: the engine registers AGAIN for the next pass. That must not discard
    # the stage, and set_context must then publish it.
    codec.register(model)
    assert len(codec._pending) == 7, "register() must not discard the staged buffer"
    codec.set_context(**ctx(2))
    assert len(codec.buffer) == 7, "the step boundary must publish the stage"
    model(x)
    codec.unregister()
    assert codec.buffer.hit_rate > 0.0, "the second visit must score hits"
