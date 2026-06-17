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

"""CPU regression tests for the spectral grad-correction hook.

No GPU, no torch.distributed, no FSDP runtime. Loads the (lightweight)
``spectral_filter`` and ``state`` modules by file path so the heavy
``verl.__init__`` import chain (tensordict, vllm, ...) is not required.

These tests cover the CPU-checkable core loop: original 2D params are corrected,
near-zero-but-present gradients still fire the hook, and reducible metrics stay
numeric. Actual FSDP1/FSDP2 grad containers, ``summon_full_params``, and
distributed writeback must be covered by the real multi-GPU probe instead of
mocked here.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _stub_parent_packages():
    """Register empty stub packages for verl.* so a deferred
    `from verl.workers.comm_eff.spectral_filter import ...` inside state.build()
    does NOT execute the real verl/__init__ (which imports tensordict/ray/vllm,
    absent on the CPU dev box). Only does this if the real packages are not
    already importable, so a full-env run is unaffected."""
    import types

    for pkg in ("verl", "verl.workers", "verl.workers.comm_eff"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []  # mark as a package
            sys.modules[pkg] = m


def _load(mod_name, rel_path):
    """Import a module by file path, bypassing verl's package __init__."""
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_parent_packages()


# Register the spectral_filter module under BOTH a private name and the real
# dotted path, so state.build()'s deferred `from verl.workers.comm_eff.
# spectral_filter import SpectralFilter` resolves to our file-loaded module
# instead of triggering the heavy verl package __init__ (tensordict/vllm/ray).
_sf = _load("verl.workers.comm_eff.spectral_filter", "verl/workers/comm_eff/spectral_filter.py")
_st = _load("_comm_eff_state_grad_hook", "verl/workers/comm_eff/state.py")

SpectralFilter = _sf.SpectralFilter
apply_spectral_correction_to_params = _sf.apply_spectral_correction_to_params


# --------------------------------------------------------------------------- #
# Fixtures: a tiny decoder-shaped module + a minimal enabled CommEffState
# --------------------------------------------------------------------------- #
class _TinyDecoderLayer(torch.nn.Module):
    """Linears named like a transformer decoder block so the target substrings
    (q_proj/k_proj/v_proj/o_proj/...) match real param names."""

    def __init__(self, d=8):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.k_proj = torch.nn.Linear(d, d, bias=False)
        self.v_proj = torch.nn.Linear(d, d, bias=False)
        self.o_proj = torch.nn.Linear(d, d, bias=False)
        self.norm = torch.nn.LayerNorm(d)  # 1-D params; must NOT be targeted

    def forward(self, x):
        h = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        return self.o_proj(self.norm(h))


class _TinyDecoder(torch.nn.Module):
    def __init__(self, d=8, n_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyDecoderLayer(d) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _MinimalConfig:
    """Duck-typed comm_eff config with .enabled and a .spectral sub-config."""

    class _Spectral:
        enabled = True
        beta_anc = 0.95
        correction_mode = "signed_ema"
        signed_ema_alpha = 0.0

    enabled = True
    spectral = _Spectral()
    mask = None


def _build_state():
    state = _st.CommEffState(_MinimalConfig())
    state.build(None)  # constructs the SpectralFilter from the spectral sub-config
    return state


def _run_backward_nonzero(module, d=8):
    """Real forward+backward producing NONZERO finite grads on every Linear."""
    x = torch.randn(4, d, dtype=torch.float32)
    out = module(x)
    loss = out.pow(2).mean()
    loss.backward()
    return loss


# Identity unshard + plain in-place writeback for the CPU (non-DTensor) path.
def _full_grad_of(grad):
    return grad, {"grad_container_type": type(grad).__name__, "is_dtensor": "False"}


def _writeback(grad, g_proj):
    grad.copy_(g_proj.to(grad.dtype))


def _warm_anchor_sign_flipped(module, spectral, target_substrs):
    """Warm the signed_ema anchor EMA so the cold-M guard does NOT fire and the
    merger provably CHANGES the grad.

    The real engine warms ``M_anchor`` via the anchor circuit (``update_anchor``)
    before the fast-path corrector runs; the cold-M guard is a no-op only until
    that happens. Feeding the NEGATED current grad makes ``sign(M)`` oppose
    ``sign(G_noisy)`` everywhere, so at ``signed_ema_alpha=0`` the merger output
    ``|G_noisy|*sign(M) = -|G_noisy|*sign(G_noisy)`` differs from ``G_noisy``
    wherever a coordinate is nonzero (rel_change > 0).
    """
    for name, p in module.named_parameters():
        if p.grad is None or p.grad.dim() != 2:
            continue
        if any(s in name for s in target_substrs):
            spectral.update_anchor(name, -p.grad.detach().clone())


_DISCOVERY_META = {"fsdp_version": "test", "module_is_FSDP1": "False"}


# --------------------------------------------------------------------------- #
# Fixed core: hook fires, discovery is recorded once, corrections > 0.
# --------------------------------------------------------------------------- #
def test_hook_fires_records_discovery_and_corrections():
    module = _TinyDecoder()
    _run_backward_nonzero(module)
    state = _build_state()
    assert state.spectral is not None
    _TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Warm the signed_ema anchor (the engine does this via the anchor circuit
    # before the corrector); without it the cold-M guard makes the merger a no-op.
    _warm_anchor_sign_flipped(module, state.spectral, _TARGETS)

    corrected = apply_spectral_correction_to_params(
        module.named_parameters(),
        spectral=state.spectral,
        target_substrs=_TARGETS,
        max_targets=4,
        state=state,
        discovery_meta=_DISCOVERY_META,
        full_grad_of=_full_grad_of,
        writeback=_writeback,
    )

    # discovery recorded exactly once, on a real 2-D target
    assert state.fsdp_grad_repr, "FSDP-DISCOVERY was never recorded"
    assert "proj" in state.fsdp_grad_repr["target_name"]
    assert state.fsdp_grad_repr["logical_2d_shape"] == "(8, 8)"
    assert state.fsdp_grad_repr["fsdp_version"] == "test"  # discovery_meta merged

    # corrections actually fired
    assert corrected > 0
    assert state.spectral_corrections == corrected
    assert corrected == 4  # capped by max_targets

    # rel_change > 0 for every corrected target (the merger actually changed the
    # grad — not a silent no-op). With sign-flipped M and alpha=0 the merger
    # negates each nonzero coordinate, so rel_change is strictly positive.
    assert state.spectral_rel_change, "no rel_change recorded"
    for name, rel in state.spectral_rel_change.items():
        assert rel > 0.0, f"{name}: rel_change={rel} — correction was a silent no-op"
    # No cold-M fallbacks fired (the anchor was warmed for every target).
    assert state.spectral.merger_coldM_fallbacks == 0


def test_norm_and_non_target_params_are_skipped():
    module = _TinyDecoder()
    _run_backward_nonzero(module)
    state = _build_state()
    apply_spectral_correction_to_params(
        module.named_parameters(),
        spectral=state.spectral,
        target_substrs=("q_proj", "k_proj", "v_proj", "o_proj"),
        max_targets=-1,  # no cap
        state=state,
        discovery_meta=_DISCOVERY_META,
        full_grad_of=_full_grad_of,
        writeback=_writeback,
    )
    # 2 layers * 4 proj matrices = 8 targets; the 1-D LayerNorm weights/biases
    # carry "norm" not a proj substring and are correctly skipped.
    assert state.spectral_corrections == 8
    assert all("norm" not in n for n in state.spectral_rel_change)


def test_writeback_mutates_the_grad_in_place():
    module = _TinyDecoder()
    _run_backward_nonzero(module)
    before = {n: p.grad.clone() for n, p in module.named_parameters() if "q_proj" in n}
    state = _build_state()
    # Warm the anchor so the signed_ema merger provably changes the grad.
    _warm_anchor_sign_flipped(module, state.spectral, ("q_proj",))
    apply_spectral_correction_to_params(
        module.named_parameters(),
        spectral=state.spectral,
        target_substrs=("q_proj",),
        max_targets=-1,
        state=state,
        discovery_meta=_DISCOVERY_META,
        full_grad_of=_full_grad_of,
        writeback=_writeback,
    )
    for n, p in module.named_parameters():
        if "q_proj" in n:
            # grad was corrected in place => changed, finite, same shape
            assert p.grad.shape == before[n].shape
            assert torch.isfinite(p.grad).all()
            assert not torch.allclose(p.grad, before[n])


def test_hook_fires_on_near_zero_grad():
    """The hook must fire regardless of gradient magnitude — only grad=None is
    skipped. A present-but-~0 grad still records discovery and runs a
    correction. The hook itself must not silently no-op on small grads."""
    module = _TinyDecoder()
    _run_backward_nonzero(module)
    # Crush every grad to ~0 but keep it present (not None).
    for p in module.parameters():
        if p.grad is not None:
            p.grad.mul_(1e-12)
    state = _build_state()
    corrected = apply_spectral_correction_to_params(
        module.named_parameters(),
        spectral=state.spectral,
        target_substrs=("q_proj", "k_proj", "v_proj", "o_proj"),
        max_targets=4,
        state=state,
        discovery_meta=_DISCOVERY_META,
        full_grad_of=_full_grad_of,
        writeback=_writeback,
    )
    assert state.fsdp_grad_repr, "discovery must fire even on a near-zero grad"
    assert corrected > 0
    assert state.spectral_corrections > 0


def test_spectral_metrics_are_all_numeric():
    """Every value surfaced into the trainer metrics dict must be numeric.

    reduce_metrics() does np.mean() on EVERY metric value; a string value
    (e.g. a flattened FSDP-discovery field) raises UFuncNoLoopError and crashes
    the step's metric reduction before global_step is logged. The string-valued
    FSDP discovery must live only on state.fsdp_grad_repr / the stdout line,
    never in the reducible metrics dict.
    """
    import numbers

    module = _TinyDecoder()
    _run_backward_nonzero(module)
    state = _build_state()
    apply_spectral_correction_to_params(
        module.named_parameters(),
        spectral=state.spectral,
        target_substrs=("q_proj", "k_proj", "v_proj", "o_proj"),
        max_targets=4,
        state=state,
        discovery_meta=_DISCOVERY_META,
        full_grad_of=_full_grad_of,
        writeback=_writeback,
    )
    # The discovery log is populated...
    assert state.fsdp_grad_repr
    # ...but it must NOT leak string values into the reducible metrics.
    metrics = _st.comm_eff_metrics(state)
    assert metrics, "expected non-empty comm_eff metrics"
    for k, v in metrics.items():
        assert isinstance(v, numbers.Number) and not isinstance(v, bool), (
            f"metric {k!r}={v!r} is not numeric; reduce_metrics() would crash on np.mean"
        )
    # And spectral metrics specifically carry rel_change but no fsdp_grad_repr keys.
    sm = state.spectral_metrics()
    assert any("rel_change" in k for k in sm)
    assert not any("fsdp_grad_repr" in k for k in sm)


def test_none_grad_is_skipped():
    module = _TinyDecoder()  # no backward => all grads None
    state = _build_state()
    corrected = apply_spectral_correction_to_params(
        module.named_parameters(),
        spectral=state.spectral,
        target_substrs=("q_proj", "k_proj", "v_proj", "o_proj"),
        max_targets=4,
        state=state,
        discovery_meta=_DISCOVERY_META,
        full_grad_of=_full_grad_of,
        writeback=_writeback,
    )
    assert corrected == 0
    assert state.spectral_corrections == 0
    assert not state.fsdp_grad_repr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
