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

"""CPU tests for the srparity arm's exact codec point (arm M).

The arm is the issue #93 cell ``a3-srq-parity-k493`` translated to Qwen3-4B:
sr_quant, 2 bits, ``subset_k=800`` of hidden 2560, 32-channel absmax blocks,
stochastic rounding. a3 was ranked stability 1 of 12 and never validated or run
past 120 steps, so this arm finishes it at horizon.

What is pinned here, and why each matters:

* the WIRE, 800*2 + 25*16 = 2000 bits per token per boundary, at or under the
  PRF arms' 128*16 = 2048. The arm's whole claim is more coordinates at equal
  or lower bandwidth, so a regression that quietly raises the payload would
  confound the codec change with a bandwidth change;
* COVERAGE, 31.25% of channels against the PRF mask's 5.0%, matching a3's
  32.10% at hidden 1536;
* UNBIASEDNESS through both the subset draw and the rounding. This is the
  load-bearing property of the family: a1 (stochastic) finished with the
  tightest optimizer in the program while a2 (identical but round-to-nearest,
  biased) was killed at step 60 with a run-minimum grad_norm 6.9x a1's
  120-step maximum.
"""

import pytest
import torch

from verl.workers.comm_eff.activation_quant import sr_quantize

H = 2560  # Qwen3-4B hidden size
BITS = 2
SUBSET_K = 800
BLOCK = 32
PRF_BUDGET_BITS = 128 * 16  # the PRF exact-k arms' payload at this hidden size


def wire_bits(k=SUBSET_K, bits=BITS, block=BLOCK):
    """Logical PP bits per token per boundary, the implementation's formula."""
    eff = k if (block <= 0 or block >= k) else block
    return k * bits + k * 16.0 / eff


def _ids(n_tokens):
    sid = torch.zeros(n_tokens, dtype=torch.long)
    pos = torch.arange(n_tokens, dtype=torch.long)
    return sid, pos


def _kwargs(step=0, seed=0, layer=3):
    # hidden_size is inferred from the tensor's last axis, not passed.
    return dict(
        layer_idx=layer,
        global_step=step,
        base_seed=seed,
        bits=BITS,
        block_size=BLOCK,
        rounding="sr",
    )


def test_wire_is_at_or_under_the_prf_budget():
    # 800*2 payload + 25 fp16 block scales = 1600 + 400 = 2000 <= 2048.
    assert wire_bits() == pytest.approx(2000.0)
    assert wire_bits() <= PRF_BUDGET_BITS
    # And it is a real saving, not a rounding artifact.
    assert PRF_BUDGET_BITS - wire_bits() == pytest.approx(48.0)


def test_coverage_matches_the_a3_cell_it_replicates():
    # a3: 493 of 1536 = 32.10%. This arm: 800 of 2560 = 31.25%.
    a3_coverage = 493 / 1536
    coverage = SUBSET_K / H
    assert coverage == pytest.approx(0.3125)
    assert abs(coverage - a3_coverage) < 0.01  # same codec point, new surface
    # 6.25x the PRF mask's coordinate coverage.
    assert coverage / (128 / H) == pytest.approx(6.25)


def test_blocks_divide_evenly_no_ragged_tail():
    # 800 / 32 = 25 whole blocks, so the pro-rata tail branch never engages and
    # the wire number above is exact rather than an upper bound.
    assert SUBSET_K % BLOCK == 0
    assert SUBSET_K // BLOCK == 25


def test_keeps_exactly_subset_k_channels_per_token():
    torch.manual_seed(0)
    n_tokens = 4
    h = torch.randn(n_tokens, H)
    sid, pos = _ids(n_tokens)
    q = sr_quantize(h, sid, pos, **_kwargs(), subset_k=SUBSET_K).reshape(n_tokens, H)
    # Exactly SUBSET_K channels may be nonzero per token. A quantized value can
    # legitimately land on zero, so this is an upper bound plus a zero count.
    for row in range(n_tokens):
        nonzero = int((q[row] != 0).sum())
        assert nonzero <= SUBSET_K
        assert int((q[row] == 0).sum()) >= H - SUBSET_K


def test_unbiased_through_the_subset_draw_and_the_rounding():
    # E[q] = h through BOTH stochastic stages. Each step refreshes the subset J
    # and the rounding uniforms, so averaging over many steps must converge to
    # the input. This is the property whose loss killed a2 at step 60.
    torch.manual_seed(0)
    n_tokens = 2
    h = torch.randn(n_tokens, H) * 0.5
    sid, pos = _ids(n_tokens)
    trials = 400
    acc = torch.zeros_like(h)
    for step in range(trials):
        acc += sr_quantize(h, sid, pos, **_kwargs(step=step), subset_k=SUBSET_K).reshape(n_tokens, H)
    mean = acc / trials
    # Monte Carlo over 400 draws at 31.25% keep rate: the per-coordinate
    # standard error is sizeable, so bound the AGGREGATE bias rather than each
    # coordinate. A biased codec (rounding="rn") fails this by a wide margin.
    rel = (mean - h).abs().mean() / h.abs().mean()
    assert rel < 0.10, f"aggregate relative bias {rel:.4f} is too large for an unbiased codec"
    # The signed mean error must straddle zero rather than sit to one side.
    assert abs(float((mean - h).mean())) < 0.02


def test_round_to_nearest_is_measurably_biased_at_this_point():
    # The control that justifies pinning rounding=sr in the launcher: at the
    # same bit depth and subset, the deterministic rule does NOT average back
    # to the input.
    torch.manual_seed(0)
    n_tokens = 2
    h = torch.randn(n_tokens, H) * 0.5
    sid, pos = _ids(n_tokens)
    kw = _kwargs()
    kw["rounding"] = "rn"
    trials = 200
    acc = torch.zeros_like(h)
    for step in range(trials):
        kw["global_step"] = step
        acc += sr_quantize(h, sid, pos, **kw, subset_k=SUBSET_K).reshape(n_tokens, H)
    rn_rel = ((acc / trials) - h).abs().mean() / h.abs().mean()

    acc_sr = torch.zeros_like(h)
    for step in range(trials):
        acc_sr += sr_quantize(h, sid, pos, **_kwargs(step=step), subset_k=SUBSET_K).reshape(n_tokens, H)
    sr_rel = ((acc_sr / trials) - h).abs().mean() / h.abs().mean()

    assert rn_rel > sr_rel, f"rn {rn_rel:.4f} should be more biased than sr {sr_rel:.4f}"


def test_same_key_is_bit_identical_and_step_refreshes():
    torch.manual_seed(0)
    h = torch.randn(3, H)
    sid, pos = _ids(3)
    a = sr_quantize(h, sid, pos, **_kwargs(step=7), subset_k=SUBSET_K)
    b = sr_quantize(h, sid, pos, **_kwargs(step=7), subset_k=SUBSET_K)
    c = sr_quantize(h, sid, pos, **_kwargs(step=8), subset_k=SUBSET_K)
    assert torch.equal(a, b)  # one step's draw is shared across the passes
    assert not torch.equal(a, c)  # and refreshes across steps


def test_config_accepts_the_arm_and_rejects_a_basis_owning_anchor():
    from verl.workers.config.comm_eff import (
        CommEffAnchorConfig,
        CommEffConfig,
        CommEffQuantConfig,
        CommEffSpectralConfig,
    )

    spectral = CommEffSpectralConfig(correction_mode="delayed_ef", delayed_ef_lambda=1.0, beta_anc=0.0, cadence=20)
    cfg = CommEffConfig(
        enabled=True,
        compression_type="sr_quant",
        quant=CommEffQuantConfig(bits=BITS, block_size=BLOCK, rounding="sr", subset_k=SUBSET_K),
        anchor=CommEffAnchorConfig(owns_q=False, cadence=20),
        spectral=spectral,
    )
    assert cfg.quant.subset_k == SUBSET_K and cfg.quant.bits == BITS

    # sr_quant carries no PowerSGD basis, so an anchor that owns Q is rejected.
    with pytest.raises(ValueError, match="owns_q"):
        CommEffConfig(
            enabled=True,
            compression_type="sr_quant",
            quant=CommEffQuantConfig(bits=BITS, block_size=BLOCK, rounding="sr", subset_k=SUBSET_K),
            anchor=CommEffAnchorConfig(owns_q=True, cadence=20),
            spectral=spectral,
        )


def test_subset_k_cannot_exceed_the_hidden_size():
    torch.manual_seed(0)
    h = torch.randn(2, H)
    sid, pos = _ids(2)
    with pytest.raises(ValueError, match="subset_k"):
        sr_quantize(h, sid, pos, **_kwargs(), subset_k=H + 1)
