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

"""comm-eff delayed_ef ``delayed_ef`` merger + the ``none`` inert mode — CPU unit tests.

Plan Correctness invariants covered here:

* limiting-case identity: ``delayed_ef`` with λ=0 yields ``G_corr == G_comp``
  EXACTLY (same tensor object — bitwise); ``blend`` with η=0 reduces to
  ``G_comp`` (regression);
* δ pairing semantics: refreshed at fire-aligned ticks from the exact
  ``G_comp_ring(t−K)`` entry, HELD between fires, shape-keyed reset;
* cold guards: unwarmed M / missing ring + no held δ ⇒ G_comp unchanged,
  counted — never a silent grad change;
* ``correction_mode=none`` through the core loop: grads bitwise untouched,
  zero corrections, zero writebacks (the geometry-probe inert posture);
* core-loop end-to-end: delayed_ef corrects per the formula, pushes the raw
  pre-correction G_comp into the fire-aware ring, pops the consumed entry.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _stub_parent_packages():
    import types

    for pkg in ("verl", "verl.workers", "verl.workers.comm_eff"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_parent_packages()
_sf = _load("verl.workers.comm_eff.spectral_filter", "verl/workers/comm_eff/spectral_filter.py")
_st = _load("verl.workers.comm_eff.state", "verl/workers/comm_eff/state.py")

SpectralFilter = _sf.SpectralFilter
apply_spectral_correction_to_params = _sf.apply_spectral_correction_to_params
FastGradRing = _st.FastGradRing

_NAME = "layers.0.q_proj.weight"


# --------------------------------------------------------------------------- #
# delayed_ef_matrix — limiting case, refresh/hold, cold guards, shape reset.
# --------------------------------------------------------------------------- #
def test_delayed_ef_lambda_zero_exact_identity():
    f = SpectralFilter(beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=0.0)
    g = torch.randn(4, 4)
    out = f.delayed_ef_matrix(_NAME, g, ring_grad=torch.randn(4, 4))
    assert out is g, "λ=0 must return G_comp EXACTLY (the same object — bitwise identity)"
    assert f.merger_coldM_fallbacks == 0
    assert not f._delayed_ef_delta, "λ=0 must build NO residual state"


def test_delayed_ef_cold_m_fallback():
    f = SpectralFilter(beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0)
    g = torch.randn(4, 4)
    out = f.delayed_ef_matrix(_NAME, g, ring_grad=torch.ones(4, 4))
    assert torch.equal(out, g), "cold M must return G_comp unchanged (never zero/alter it)"
    assert f.merger_coldM_fallbacks == 1


def test_delayed_ef_refresh_then_hold():
    f = SpectralFilter(beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0, ema_device="cpu")
    m_rep = torch.full((3, 3), 5.0)
    f.update_anchor(_NAME, m_rep)  # β=0 ⇒ M == m_rep exactly
    ring = torch.full((3, 3), 2.0)
    g1 = torch.full((3, 3), 1.0)
    # Fire-aligned tick: δ refreshed from the exact pair ⇒ δ = 5 − 2 = 3.
    out1 = f.delayed_ef_matrix(_NAME, g1, ring_grad=ring)
    assert torch.allclose(out1, torch.full((3, 3), 4.0))  # 1 + 1.0*3
    assert f.delayed_ef_refreshed == 1 and f.delayed_ef_held == 0
    # In-between tick (no ring entry): the HELD δ is re-applied.
    g2 = torch.full((3, 3), 10.0)
    out2 = f.delayed_ef_matrix(_NAME, g2, ring_grad=None)
    assert torch.allclose(out2, torch.full((3, 3), 13.0))  # 10 + 3
    assert f.delayed_ef_held == 1
    # λ scales the injection: λ=0.5 ⇒ 10 + 1.5.
    f.delayed_ef_lambda = 0.5
    out3 = f.delayed_ef_matrix(_NAME, g2, ring_grad=None)
    assert torch.allclose(out3, torch.full((3, 3), 11.5))


def test_delayed_ef_no_pair_no_held_falls_back():
    f = SpectralFilter(beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0)
    f.update_anchor(_NAME, torch.ones(3, 3))  # M warm
    g = torch.randn(3, 3)
    out = f.delayed_ef_matrix(_NAME, g, ring_grad=None)  # pre-first-fire warmup
    assert torch.equal(out, g)
    assert f.merger_coldM_fallbacks == 1


def test_delayed_ef_shape_change_resets_held_delta():
    f = SpectralFilter(beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0)
    f.update_anchor(_NAME, torch.ones(3, 3))
    f.delayed_ef_matrix(_NAME, torch.ones(3, 3), ring_grad=torch.zeros(3, 3))  # held δ = 1s
    assert _sf._canon(_NAME) in f._delayed_ef_delta
    # The same name arrives with a NEW logical shape: the stale held δ must be
    # RESET (counted), the stale-shaped M trips the cold guard, and G_comp comes
    # back unchanged — no cross-shape leak, no silent grad change.
    g_new = torch.ones(2, 2)
    out = f.delayed_ef_matrix(_NAME, g_new, ring_grad=None)
    assert f.residual_reset_on_shape_mismatch == 1
    assert f.merger_coldM_fallbacks == 1  # stored M is stale-shaped ⇒ cold fallback
    assert _sf._canon(_NAME) not in f._delayed_ef_delta, "stale δ must not survive a shape change"
    assert torch.equal(out, g_new), "after a shape reset with no valid pair, fall back to G_comp"


def test_blend_eta_zero_reduces_to_g_comp():
    f = SpectralFilter(beta_anc=0.0, correction_mode="blend", blend_eta=0.0)
    f.update_anchor(_NAME, torch.randn(4, 4))  # warm so the cold guard is NOT the reason
    g = torch.randn(4, 4)
    out = f.blend_matrix(_NAME, g)
    assert torch.equal(out, g), "blend η=0 must reduce to G_comp exactly"


# --------------------------------------------------------------------------- #
# Core loop — mode none is inert; delayed_ef corrects + manages the ring.
# --------------------------------------------------------------------------- #
class _TinyDecoderLayer(torch.nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.o_proj = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class _Duck:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _mk_state(correction_mode, delayed_ef_lambda=1.0):
    cfg = _Duck(
        enabled=True,
        compression_type="dense",
        mask=None,
        clean_cadence=0,
        anchor=_Duck(enabled=True, cadence=5, delay_K=5, owns_q=False, replay_paired_batch=True, snapshot_device="cpu"),
        spectral=_Duck(
            enabled=True,
            beta_anc=0.0,
            cadence=1,
            correction_mode=correction_mode,
            inject_gamma=1.0,
            blend_eta=0.3,
            signed_ema_alpha=0.0,
            ef_decay=0.0,
            ef_clip=0.0,
            delayed_ef_lambda=delayed_ef_lambda,
            ema_device="cpu",
            max_targets=-1,
        ),
        capture=None,
        probe=_Duck(geometry_enabled=False, out_dir="", rank0_only=True, m4_lags=5, per_target_sidecar=True),
    )
    state = _st.CommEffState(cfg)
    state.build(None)
    return state


def _backward(module, d=8):
    x = torch.randn(4, d)
    module(x).pow(2).mean().backward()


def _full_grad_of(grad):
    return grad, {"grad_container_type": type(grad).__name__, "is_dtensor": "False"}


def _writeback(grad, g_proj):
    grad.copy_(g_proj.to(grad.dtype))


_TARGETS = ("q_proj", "o_proj")
_META = {"fsdp_version": "test", "module_is_FSDP1": "False"}


def test_core_loop_mode_none_is_inert():
    torch.manual_seed(0)
    module = _TinyDecoderLayer()
    _backward(module)
    state = _mk_state("none")
    before = {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}
    writebacks = []

    corrected = apply_spectral_correction_to_params(
        module.named_parameters(),
        spectral=state.spectral,
        target_substrs=_TARGETS,
        max_targets=-1,
        state=state,
        discovery_meta=_META,
        full_grad_of=_full_grad_of,
        writeback=lambda g, p: writebacks.append(1),
    )
    assert corrected == 0
    assert not writebacks, "mode=none must NEVER write back"
    assert state.spectral_corrections == 0
    for n, p in module.named_parameters():
        if p.grad is not None:
            assert torch.equal(p.grad, before[n]), f"mode=none mutated the grad of {n}"


def test_core_loop_delayed_ef_corrects_and_manages_ring():
    torch.manual_seed(1)
    module = _TinyDecoderLayer()
    _backward(module)
    state = _mk_state("delayed_ef", delayed_ef_lambda=1.0)
    assert state.fast_grad_ring is not None  # armed by build for delayed_ef
    spectral = state.spectral

    # Warm M_rep (β=0 ⇒ exactly the fed gradient) + seed the t−K ring entry.
    state.anchor_step = 10  # fire-aligned tick (cadence 5, delay_K 5 ⇒ pair at 5)
    ring_entry, ring_norms, m_rep, expected = {}, {}, {}, {}
    for n, p in module.named_parameters():
        if p.grad is None or p.grad.dim() != 2:
            continue
        cn = _sf._canon(n)
        rg = torch.zeros_like(p.grad)
        ring_entry[cn] = rg
        ring_norms[cn] = 0.0
        m = p.grad.detach().clone() * 3.0
        m_rep[cn] = m
        spectral.update_anchor(n, m)
        expected[cn] = p.grad.detach().clone() + 1.0 * (m - rg)  # G + λ(M − ring)
    assert state.fast_grad_ring.push(5, ring_entry, ring_norms) is True

    corrected = apply_spectral_correction_to_params(
        module.named_parameters(),
        spectral=spectral,
        target_substrs=_TARGETS,
        max_targets=-1,
        state=state,
        discovery_meta=_META,
        full_grad_of=_full_grad_of,
        writeback=_writeback,
    )
    assert corrected == 2
    assert spectral.delayed_ef_refreshed == 2 and spectral.merger_coldM_fallbacks == 0
    for n, p in module.named_parameters():
        if p.grad is None or p.grad.dim() != 2:
            continue
        assert torch.allclose(p.grad, expected[_sf._canon(n)], atol=1e-6), f"delayed_ef formula mismatch on {n}"
    # Ring management: tick 10 (retained) pushed with the RAW pre-correction
    # G_comp; the consumed tick-5 entry popped.
    assert state.fast_grad_ring.get(5) is None, "consumed ring entry must be popped"
    pushed = state.fast_grad_ring.get(10)
    assert pushed is not None, "the raw pre-correction G_comp must be pushed at retained ticks"
    # Raw means BEFORE correction: pushed != corrected grad (which now includes δ).
    g_pushed, _ = pushed
    for n, p in module.named_parameters():
        if p.grad is None or p.grad.dim() != 2:
            continue
        cn = _sf._canon(n)
        assert not torch.allclose(g_pushed[cn], p.grad.to(torch.float32)), (
            "ring must hold the RAW pre-correction G_comp, not the merged gradient"
        )


def test_core_loop_delayed_ef_lambda_zero_bitwise_unchanged():
    torch.manual_seed(2)
    module = _TinyDecoderLayer()
    _backward(module)
    state = _mk_state("delayed_ef", delayed_ef_lambda=0.0)
    state.anchor_step = 10
    before = {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}

    apply_spectral_correction_to_params(
        module.named_parameters(),
        spectral=state.spectral,
        target_substrs=_TARGETS,
        max_targets=-1,
        state=state,
        discovery_meta=_META,
        full_grad_of=_full_grad_of,
        writeback=_writeback,
    )
    for n, p in module.named_parameters():
        if p.grad is not None:
            assert torch.equal(p.grad, before[n]), (
                f"delayed_ef λ=0 must leave {n}'s grad bitwise unchanged (limiting-case identity)"
            )


def test_spectral_filter_rejects_unknown_mode():
    with pytest.raises(AssertionError):
        SpectralFilter(correction_mode="delayed_EF")  # typo must be loud


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
