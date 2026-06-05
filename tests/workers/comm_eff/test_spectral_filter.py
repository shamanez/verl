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

"""Unit tests for the comm_eff anchor-guided gradient corrector.

These cover formula-correctness invariants without a GPU or distributed runtime.
The filter operates on logical 2D matrices; FSDP unsharding is the engine's job.
The live correction is the signed-EMA merger (``correction_mode="signed_ema"``);
``inject`` and ``blend`` are alternate anchor combiners. All consult only the
anchor-gradient EMA ``M_anchor`` (no SVD / no basis cache — the dead
reweight/SVD/Tikhonov/seeded path was removed in EXP-25).

* anchor EMA cold-starts at zeros and moves under update_anchor
* signed_ema: alpha=1 => G_noisy unchanged; alpha=0 => |G_noisy|*sign(M);
  cold-M => G_noisy unchanged (NOT zeroed) + merger_coldM_fallbacks bumped
* inject / blend fire across the FSDP name infix
* ema_device=cpu round-trip equals on-device
* _canon collapses the FSDP wrap infix to one EMA key
"""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

# Load spectral_filter by FILE PATH so the heavy verl.__init__ chain (tensordict
# / vllm / ray, absent on the CPU dev box) is not imported — same harness as
# tests/workers/comm_eff/test_grad_correction_hook.py. The module under test is
# pure-torch and FSDP-agnostic by design, so this is sufficient and runs on CPU.
_REPO = pathlib.Path(__file__).resolve().parents[3]
for _pkg in ("verl", "verl.workers", "verl.workers.comm_eff"):
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = []
        sys.modules[_pkg] = _m
_spec = importlib.util.spec_from_file_location(
    "verl.workers.comm_eff.spectral_filter", _REPO / "verl/workers/comm_eff/spectral_filter.py"
)
_sf = importlib.util.module_from_spec(_spec)
sys.modules["verl.workers.comm_eff.spectral_filter"] = _sf
_spec.loader.exec_module(_sf)

SpectralFilter = _sf.SpectralFilter
_canon = _sf._canon

TOL = 1e-6


# --------------------------------------------------------------------------- #
# anchor EMA: cold-starts at zeros and moves under update_anchor
# --------------------------------------------------------------------------- #
def test_anchor_cold_starts_at_zeros():
    """ensure_anchor cold-starts M_anchor at zeros (no seeded basis)."""
    f = SpectralFilter(beta_anc=0.5)
    g = torch.randn(8, 6, dtype=torch.float32)
    anc = f.ensure_anchor("w", g)
    assert anc.shape == g.shape
    assert torch.count_nonzero(anc).item() == 0, "cold start must be all-zeros"


def test_ema_update_moves_anchor():
    f = SpectralFilter(beta_anc=0.5)
    g_anchor = torch.ones(4, 4, dtype=torch.float32)
    a0 = f.ensure_anchor("w", g_anchor).clone()  # zeros (cold start)
    a1 = f.update_anchor("w", g_anchor)
    # beta=0.5: M <- 0.5*0 + 0.5*1 = 0.5
    assert torch.allclose(a1, torch.full((4, 4), 0.5), atol=TOL)
    assert not torch.allclose(a0, a1)


def test_ema_device_cpu_roundtrip_equals_on_device():
    # On CPU-only CI both "gpu" and "cpu" storage resolve to CPU tensors, so the
    # EMA arithmetic must be identical; the test guards the offload-move logic
    # (anchor_on / store-back) does not corrupt values. Same seeds, same grads.
    g1 = torch.randn(8, 6, generator=torch.Generator().manual_seed(7), dtype=torch.float32)
    g2 = torch.randn(8, 6, generator=torch.Generator().manual_seed(8), dtype=torch.float32)

    f_gpu = SpectralFilter(beta_anc=0.9, ema_device="gpu")
    f_cpu = SpectralFilter(beta_anc=0.9, ema_device="cpu")

    for f in (f_gpu, f_cpu):
        f.update_anchor("w", g1)
        f.update_anchor("w", g2)

    m_gpu = f_gpu.anchor_on("w", torch.device("cpu"))
    m_cpu = f_cpu.anchor_on("w", torch.device("cpu"))
    assert torch.allclose(m_gpu, m_cpu, atol=TOL), "cpu-offloaded EMA diverged from on-device EMA"
    # The cpu-storage EMA tensor actually lives on CPU between refreshes.
    assert f_cpu._anchor["w"].device.type == "cpu"


# =========================================================================== #
# signed_ema merger (EXP-25/R3): G_corr = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M)
# =========================================================================== #
def test_signed_ema_alpha_one_returns_g_noisy_exactly():
    """alpha=1 => G_corr == G_noisy verbatim, regardless of the (warm) anchor."""
    f = SpectralFilter(beta_anc=0.0, correction_mode="signed_ema", signed_ema_alpha=1.0)
    gen = torch.Generator().manual_seed(31)
    # beta=0 => M_anchor == g_anchor exactly (warm, nonzero).
    f.update_anchor("w", torch.randn(12, 7, generator=gen, dtype=torch.float32))
    g_noisy = torch.randn(12, 7, generator=gen, dtype=torch.float32)
    out = f.signed_ema_matrix("w", g_noisy.clone())
    assert torch.allclose(out, g_noisy, atol=TOL), "alpha=1 must return G_noisy unchanged"
    assert f.merger_coldM_fallbacks == 0, "anchor was warm => no cold-M fallback"


def test_signed_ema_alpha_zero_is_magnitude_g_sign_m():
    """alpha=0 => G_corr == |G_noisy| * sign(M_anchor) elementwise."""
    f = SpectralFilter(beta_anc=0.0, correction_mode="signed_ema", signed_ema_alpha=0.0)
    gen = torch.Generator().manual_seed(32)
    m_anchor = torch.randn(10, 10, generator=gen, dtype=torch.float32)
    f.update_anchor("w", m_anchor)  # beta=0 => M == m_anchor
    g_noisy = torch.randn(10, 10, generator=gen, dtype=torch.float32)
    out = f.signed_ema_matrix("w", g_noisy.clone()).to(torch.float32)
    expected = g_noisy.abs() * torch.sign(m_anchor)
    assert torch.allclose(out, expected, atol=1e-5), "alpha=0 must be |G_noisy|*sign(M)"


def test_signed_ema_cold_M_returns_g_noisy_and_counts_fallback():
    """COLD-M guard: when M_anchor is unwarmed (zeros), the merger must return
    G_noisy UNCHANGED (NOT silently zeroed) and bump merger_coldM_fallbacks.
    This is the silent grad-zeroing guard — at alpha=0 a cold M would otherwise
    give |G|*sign(0)=0."""
    f = SpectralFilter(correction_mode="signed_ema", signed_ema_alpha=0.0)
    g_noisy = torch.randn(8, 8, generator=torch.Generator().manual_seed(33), dtype=torch.float32)
    # No update_anchor => M cold (zeros).
    out = f.signed_ema_matrix("w", g_noisy.clone())
    assert torch.allclose(out, g_noisy, atol=TOL), "cold M must return G_noisy UNCHANGED, not zeroed"
    assert f.merger_coldM_fallbacks == 1, "cold-M fallback must be counted"
    # After M warms, the fallback must NOT fire again for that matrix.
    f.update_anchor("w", torch.ones(8, 8, dtype=torch.float32))
    out2 = f.signed_ema_matrix("w", g_noisy.clone())
    # alpha=0, M all-positive => |G_noisy| * (+1) = |G_noisy|.
    assert torch.allclose(out2.to(torch.float32), g_noisy.abs(), atol=1e-5)
    assert f.merger_coldM_fallbacks == 1, "warm M must not increment the fallback counter"


def test_signed_ema_finds_anchor_across_fsdp_infix():
    """Feed M_anchor under the CLONE (non-infixed) name, merge under the LIVE
    (infixed) name — the merger must see the warmed anchor (no cold-M fallback)."""
    f = SpectralFilter(beta_anc=0.0, correction_mode="signed_ema", signed_ema_alpha=0.0)
    gen = torch.Generator().manual_seed(34)
    g_anchor = torch.randn(16, 16, generator=gen, dtype=torch.float32)
    f.update_anchor(CLONE_NAME, g_anchor)  # feed under clone (non-infixed) name
    g_noisy = torch.randn(16, 16, generator=gen, dtype=torch.float32)
    out = f.signed_ema_matrix(LIVE_NAME, g_noisy.clone()).to(torch.float32)
    assert f.merger_coldM_fallbacks == 0, "M_anchor must be found warm across the FSDP infix"
    expected = g_noisy.abs() * torch.sign(g_anchor)
    assert torch.allclose(out, expected, atol=1e-5)
    # Exactly one canonical EMA entry (no divergent live/clone buffers).
    assert list(f._anchor.keys()) == [CLONE_NAME]
    assert LIVE_NAME not in f._anchor


# =========================================================================== #
# Anchor-circuit name-key consistency across FSDP wrap infixes.
#
# The anchor EMA is FED from the anchor CLONE's named_parameters() (which, when
# build_anchor_module's deepcopy fails and the config-rebuild fallback runs, are
# NON-infixed: "model.layers.0.self_attn.q_proj.weight") and READ back via the
# LIVE FSDP module's summoned names (per-layer-wrapped => carry the
# "._fsdp_wrapped_module." infix). These tests assert the feed-key and read-key
# resolve to the same EMA entry.
# =========================================================================== #
CLONE_NAME = "model.layers.0.self_attn.q_proj.weight"                      # fallback clone (non-infixed)
LIVE_NAME = "model.layers.0._fsdp_wrapped_module.self_attn.q_proj.weight"  # live FSDP per-layer-wrapped


def test_canon_strips_fsdp_infix():
    """_canon collapses the live (infixed) and clone (non-infixed) names to one key."""
    assert _canon(LIVE_NAME) == CLONE_NAME
    assert _canon(CLONE_NAME) == CLONE_NAME  # already canonical => no-op (deepcopy-success safe)
    # leading root-wrap (no dot prefix) is also stripped
    assert _canon("_fsdp_wrapped_module.model.embed_tokens.weight") == "model.embed_tokens.weight"
    # multiple nested wraps collapse
    assert _canon("a._fsdp_wrapped_module.b._fsdp_wrapped_module.c") == "a.b.c"


def test_inject_finds_anchor_across_fsdp_infix():
    """Feed under the clone name, inject under the live name, and ensure it fires."""
    f = SpectralFilter(
        beta_anc=0.5, correction_mode="inject", inject_gamma=1.0,
    )
    gen = torch.Generator().manual_seed(7)
    # A clean anchor gradient (the K-stale unmasked G_anchor) fed under the clone name.
    g_anchor = torch.randn(16, 16, generator=gen, dtype=torch.float32)
    f.update_anchor(CLONE_NAME, g_anchor)
    # The masked fast-path gradient, corrected under the LIVE (infixed) name.
    g_mask = torch.randn(16, 16, generator=gen, dtype=torch.float32)
    g_corr = f.inject_matrix(LIVE_NAME, g_mask.clone())
    # Injection fired => corrected grad differs from the masked grad.
    diff = torch.linalg.norm(g_corr.to(torch.float32) - g_mask).item()
    assert diff > 1e-6, (
        "inject_matrix returned g_mask unchanged — M_anchor read as zero across "
        "the FSDP infix (the EXP-18 bug). The injection must fire."
    )


def test_anchor_ema_shared_entry_across_infix():
    """The clone-name feed and the live-name read address the SAME _anchor entry
    (exactly one key, the canonical one) — not two divergent buffers."""
    f = SpectralFilter(beta_anc=0.5, correction_mode="inject")
    g_anchor = torch.ones(8, 8, dtype=torch.float32)
    f.update_anchor(CLONE_NAME, g_anchor)          # feed under clone name
    # Exactly one EMA entry, keyed canonically.
    assert list(f._anchor.keys()) == [CLONE_NAME]
    assert CLONE_NAME in f._anchor and LIVE_NAME not in f._anchor
    # The live (infixed) name resolves to that very same tensor object.
    anc_via_live = f.anchor_on(LIVE_NAME, g_anchor.device)
    anc_via_clone = f.anchor_on(CLONE_NAME, g_anchor.device)
    assert anc_via_live.data_ptr() == anc_via_clone.data_ptr()
    # beta=0.5 over a zero start with all-ones grad => EMA == 0.5 everywhere.
    assert torch.allclose(anc_via_live, torch.full((8, 8), 0.5), atol=TOL)


# =========================================================================== #
# Convex blend: G_corr = (1-eta)*G_mask + eta*scale*M_anchor.
# eta=0 => G_mask exactly; eta=1 => scale-matched M_anchor; the blend must fire
# across the FSDP name infix (the same key-consistency the inject path needs).
# =========================================================================== #
def test_blend_eta_zero_returns_g_mask_exactly():
    """eta=0 => the convex blend collapses to G_mask verbatim (the floor)."""
    f = SpectralFilter(
        beta_anc=0.5, correction_mode="blend", blend_eta=0.0
    )
    gen = torch.Generator().manual_seed(21)
    # A nonzero anchor so the no-op short-circuit (anc_norm<=eps) is NOT what
    # produces the result — eta=0 itself must give G_mask.
    f.update_anchor("w", torch.randn(16, 16, generator=gen, dtype=torch.float32))
    g_mask = torch.randn(16, 16, generator=gen, dtype=torch.float32)
    out = f.blend_matrix("w", g_mask.clone())
    assert torch.allclose(out, g_mask, atol=TOL), "eta=0 must return G_mask exactly"


def test_blend_eta_one_is_scale_matched_anchor():
    """eta=1 => G_corr ≈ scale*M_anchor with scale=||G_mask||/||M_anchor||, i.e.
    a vector PARALLEL to M_anchor whose magnitude equals ||G_mask||."""
    gen = torch.Generator().manual_seed(22)
    # beta_anc=0 => M_anchor == m_anchor exactly (a real, warm anchor).
    f2 = SpectralFilter(
        beta_anc=0.0, correction_mode="blend", blend_eta=1.0
    )
    m_anchor = torch.randn(16, 16, generator=gen, dtype=torch.float32)
    f2.update_anchor("w", m_anchor)  # beta=0 => M_anchor == m_anchor exactly
    g_mask = torch.randn(16, 16, generator=gen, dtype=torch.float32)
    out = f2.blend_matrix("w", g_mask.clone()).to(torch.float32)
    gm_norm = torch.linalg.norm(g_mask.to(torch.float32))
    anc_norm = torch.linalg.norm(m_anchor)
    expected = (gm_norm / anc_norm) * m_anchor  # scale-matched anchor
    assert torch.allclose(out, expected, atol=1e-4), "eta=1 must return scale*M_anchor"
    # Magnitude is matched to ||G_mask|| and direction is M_anchor's.
    assert abs(torch.linalg.norm(out).item() - gm_norm.item()) < 1e-3
    cos = (out * m_anchor).sum() / (torch.linalg.norm(out) * anc_norm)
    assert cos.item() > 1.0 - 1e-4, "eta=1 output must be parallel to M_anchor"


def test_blend_magnitude_stable_at_eta_0p7():
    """The convex blend keeps ||G_corr||/||G_mask|| <= 1.
    For orthogonal terms it equals sqrt((1-eta)^2 + eta^2) < 1; for any anchor it
    is bounded by 1 under the convex blend (triangle ineq with scale-match)."""
    f = SpectralFilter(
        beta_anc=0.0, correction_mode="blend", blend_eta=0.7
    )
    gen = torch.Generator().manual_seed(23)
    f.update_anchor("w", torch.randn(32, 32, generator=gen, dtype=torch.float32))
    g_mask = torch.randn(32, 32, generator=gen, dtype=torch.float32)
    out = f.blend_matrix("w", g_mask.clone()).to(torch.float32)
    ratio = (torch.linalg.norm(out) / torch.linalg.norm(g_mask.to(torch.float32))).item()
    assert ratio <= 1.0 + 1e-5, f"||G_corr||/||G_mask|| must be <= 1; got {ratio}"


def test_blend_finds_anchor_across_fsdp_infix():
    """Key-consistency for the blend path (mirrors test_inject_finds_anchor...):
    feed M_anchor under the CLONE (non-infixed) name, blend under the LIVE
    (infixed) name. The blend MUST fire (result != G_mask) AND must NOT equal the
    eta=0 floor — proving M_anchor was found nonzero under the canonical key."""
    f = SpectralFilter(
        beta_anc=0.5, correction_mode="blend", blend_eta=0.7
    )
    gen = torch.Generator().manual_seed(24)
    g_anchor = torch.randn(16, 16, generator=gen, dtype=torch.float32)
    f.update_anchor(CLONE_NAME, g_anchor)  # feed under clone (non-infixed) name
    g_mask = torch.randn(16, 16, generator=gen, dtype=torch.float32)
    g_corr = f.blend_matrix(LIVE_NAME, g_mask.clone())  # blend under live (infixed) name
    diff = torch.linalg.norm(g_corr.to(torch.float32) - g_mask).item()
    assert diff > 1e-6, (
        "blend_matrix returned g_mask unchanged — M_anchor read as zero across "
        "the FSDP infix (the EXP-18 bug). The blend must fire."
    )
    # Exactly one canonical EMA entry (no divergent live/clone buffers).
    assert list(f._anchor.keys()) == [CLONE_NAME]
    assert LIVE_NAME not in f._anchor


# =========================================================================== #
# relative_change is faithful: 0 when G_corr == G_noisy, > 0 when it differs.
# =========================================================================== #
def test_relative_change_zero_when_unchanged_and_positive_when_changed():
    f = SpectralFilter(beta_anc=0.0, correction_mode="signed_ema", signed_ema_alpha=0.0)
    g = torch.randn(8, 8, generator=torch.Generator().manual_seed(41), dtype=torch.float32)
    assert f.relative_change(g, g.clone()) == 0.0
    f.update_anchor("w", -torch.ones(8, 8, dtype=torch.float32))  # flip all signs
    out = f.signed_ema_matrix("w", g.clone())
    assert f.relative_change(g, out) > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
