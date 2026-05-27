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

"""CPU reproduction + regression for the EXP-7 spectral grad-correction hook.

No GPU, no torch.distributed, no FSDP runtime. Loads the (lightweight)
``spectral_filter`` and ``state`` modules by file path so the heavy
``verl.__init__`` import chain (tensordict, vllm, ...) is not required.

This test ENCODES the two EXP-7 defects the first Vast run hit:

* DEFECT 2 — the grad-correction hook body never executed. The original engine
  iterated ``module._fsdp_wrapped_module.named_parameters()`` directly, which
  under FSDP1 (``use_orig_params=false``) yields a 1-D ``_flat_param`` whose
  name has no ``q_proj``/etc. substring and whose grad is not 2-D, so EVERY
  param was skipped: no ``[FSDP-DISCOVERY]`` line, ``spectral_corrections=0``.
  The fix exposes the original 2-D named params + grads and routes them through
  ``apply_spectral_correction_to_params``. The first three tests assert the
  fixed core: discovery recorded once, ``spectral_corrections > 0``,
  ``rel_change`` in ``(0, 1]``. ``test_legacy_flatparam_iteration_reproduces_bug``
  reproduces the ORIGINAL failure (flat-param iteration => zero corrections).

* The hook must fire REGARDLESS of gradient magnitude — only ``grad is None``
  is skipped. ``test_hook_fires_on_near_zero_grad`` proves a ~0 (but present)
  grad still triggers discovery + a correction.

It also covers DEFECT 3 — cross-process anchor determinism — by simulating two
processes with different ``PYTHONHASHSEED`` salts and asserting identical seeded
anchors for the same parameter name.
"""

import importlib.util
import os
import pathlib
import subprocess
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
_st = _load("_exp7_state", "verl/workers/comm_eff/state.py")

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
        alpha = 0.3
        tau = 1e-3
        beta_anc = 0.95
        seed_anchor_cache = True
        anchor_seed = 0

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


_DISCOVERY_META = {"fsdp_version": "test", "module_is_FSDP1": "False"}


# --------------------------------------------------------------------------- #
# DEFECT 2 — fixed core: hook fires, discovery recorded once, corrections > 0
# --------------------------------------------------------------------------- #
def test_hook_fires_records_discovery_and_corrections():
    module = _TinyDecoder()
    _run_backward_nonzero(module)
    state = _build_state()
    assert state.spectral is not None

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

    # discovery recorded exactly once, on a real 2-D target
    assert state.fsdp_grad_repr, "FSDP-DISCOVERY was never recorded"
    assert "proj" in state.fsdp_grad_repr["target_name"]
    assert state.fsdp_grad_repr["logical_2d_shape"] == "(8, 8)"
    assert state.fsdp_grad_repr["fsdp_version"] == "test"  # discovery_meta merged

    # corrections actually fired
    assert corrected > 0
    assert state.spectral_corrections == corrected
    assert corrected == 4  # capped by max_targets

    # rel_change strictly in (0, 1] for every corrected target at alpha=0.3
    assert state.spectral_rel_change, "no rel_change recorded"
    for name, rel in state.spectral_rel_change.items():
        assert 0.0 < rel <= 1.0, f"{name}: rel_change={rel} not in (0, 1]"


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
    correction (this is exactly the degenerate-loss situation DEFECT 1 created;
    the hook itself must not silently no-op on small grads)."""
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


# --------------------------------------------------------------------------- #
# DEFECT 2 — reproduce the ORIGINAL bug: FSDP1-style flat-param iteration.
# This is what the engine did before the fix and why nothing fired.
# --------------------------------------------------------------------------- #
class _FakeFlatParam:
    """Mimics an FSDP1 use_orig_params=false FlatParameter: a single 1-D param
    named `_flat_param` whose grad is the concatenation of all matrix grads."""

    def __init__(self, module):
        flat = torch.cat([p.grad.reshape(-1) for p in module.parameters() if p.grad is not None])
        self.grad = flat


def _flat_named_params(module):
    # Exactly the shape of what `module._fsdp_wrapped_module.named_parameters()`
    # yields under FSDP1 use_orig_params=false: a 1-D `_flat_param`.
    yield "_flat_param", _FakeFlatParam(module)


def test_legacy_flatparam_iteration_reproduces_bug():
    """With FSDP1 flat-param iteration the core fires ZERO corrections — the
    name has no proj substring and the grad is 1-D. This is the first-run
    failure; the engine fix (summon_full_params -> original 2-D params) is what
    avoids feeding this iterator in the real path."""
    module = _TinyDecoder()
    _run_backward_nonzero(module)
    state = _build_state()
    corrected = apply_spectral_correction_to_params(
        _flat_named_params(module),
        spectral=state.spectral,
        target_substrs=("q_proj", "k_proj", "v_proj", "o_proj"),
        max_targets=4,
        state=state,
        discovery_meta=_DISCOVERY_META,
        full_grad_of=_full_grad_of,
        writeback=_writeback,
    )
    assert corrected == 0  # the bug: nothing matched
    assert state.spectral_corrections == 0
    assert not state.fsdp_grad_repr  # no FSDP-DISCOVERY line => exactly the symptom


# --------------------------------------------------------------------------- #
# DEFECT 3 — cross-process seeded-anchor determinism (stable hash).
# --------------------------------------------------------------------------- #
def test_seeded_anchor_identical_across_processes():
    """Two SpectralFilter instances in DIFFERENT processes (different
    PYTHONHASHSEED salts) must build the IDENTICAL seeded anchor for the same
    (name, anchor_seed). The builtin hash() this replaced was salted per
    process => divergent anchors per FSDP rank => corrupted correction."""
    name = "model.layers.3.self_attn.q_proj.weight"
    code = (
        "import importlib.util, sys, pathlib, torch\n"
        f"repo = pathlib.Path({str(_REPO)!r})\n"
        "spec = importlib.util.spec_from_file_location('_sf', repo/'verl/workers/comm_eff/spectral_filter.py')\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['_sf']=m; spec.loader.exec_module(m)\n"
        "f = m.SpectralFilter(seed_anchor_cache=True, anchor_seed=7)\n"
        "g = torch.zeros(8, 6, dtype=torch.float32)\n"
        f"a = f.ensure_anchor({name!r}, g)\n"
        "import hashlib;print(hashlib.sha256(a.numpy().tobytes()).hexdigest())\n"
    )

    def _digest(hashseed):
        env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    # Two different hash salts -> identical anchor digest under the sha256 seed.
    d0 = _digest(0)
    d1 = _digest(12345)
    assert d0 == d1, f"seeded anchor diverged across PYTHONHASHSEED salts: {d0} != {d1}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
