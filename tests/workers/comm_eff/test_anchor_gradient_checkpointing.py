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

"""The anchor clone must inherit the live module's gradient checkpointing.

``build_anchor_module`` returns a plain (non-FSDP) clone that the anchor circuit
runs a full dense forward/backward through. The engine only ever calls
``gradient_checkpointing_enable`` on the LIVE module, so the clone's activation
policy was whatever its construction path happened to leave behind:

* the ``copy.deepcopy`` path carries the flag across on current transformers,
  but incidentally — nothing enforces it and no log line recorded it;
* the "cannot pickle" config-rebuild fallback builds a FRESH model and loses it
  outright. That is the case this test pins (it fails without the guard).

Why the guard is worth having at all — run 90 (Qwen2.5-Math-1.5B, anchor cadence
20, ``research/runs/90-prf-exactk-600/metrics/prf_train.log``):
``actor/perf/max_memory_allocated_gb`` is 46.067 GiB flat for steps 3-19 and
109.011 GiB at step 20, the first anchor fire. +62.9 GiB on a 1.5B model.
"""

import copy

import pytest
import torch

from verl.workers.comm_eff.anchor import build_anchor_module

transformers = pytest.importorskip("transformers")


def _tiny_causal_lm():
    """A 2-layer HF causal LM, small enough to build in well under a second."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return AutoModelForCausalLM.from_config(cfg)


def _gc_modules(module: torch.nn.Module) -> int:
    return sum(1 for m in module.modules() if getattr(m, "gradient_checkpointing", False))


def test_anchor_clone_inherits_gradient_checkpointing():
    """deepcopy path: source has GC on ⇒ clone must have GC on."""
    src = _tiny_causal_lm()
    src.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert _gc_modules(src) > 0, "test setup: the source model did not enable gradient checkpointing"

    clone = build_anchor_module(src)

    assert _gc_modules(clone) > 0, (
        "anchor clone has gradient checkpointing OFF while the live module has it ON — "
        "the anchor's dense replay forward will materialize every activation "
        "(+62.9 GiB measured at the first anchor fire of run 90)"
    )
    assert _gc_modules(clone) == _gc_modules(src)
    # The clone must still be isolated: no shared parameter storage with the live module.
    src_ptrs = {p.data_ptr() for p in src.parameters()}
    assert not any(p.data_ptr() in src_ptrs for p in clone.parameters())


def test_anchor_clone_inherits_gradient_checkpointing_on_rebuild_path(monkeypatch):
    """config-rebuild fallback: the same guarantee must hold when deepcopy fails."""
    src = _tiny_causal_lm()
    src.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    real_deepcopy = copy.deepcopy

    def _boom_on_modules(obj, *args, **kwargs):
        # Only the module deepcopy fails, mirroring the real monkey-patched-model
        # case; the config deepcopy the rebuild path itself performs must still work.
        if isinstance(obj, torch.nn.Module):
            raise TypeError("cannot pickle 'module' object")
        return real_deepcopy(obj, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", _boom_on_modules)
    clone = build_anchor_module(src)

    assert _gc_modules(clone) > 0, (
        "anchor clone built through the config-rebuild fallback has gradient checkpointing OFF"
    )


def test_anchor_clone_leaves_gradient_checkpointing_off_when_source_is_off():
    """No unilateral enable: a dense (non-checkpointed) live module stays that way."""
    src = _tiny_causal_lm()
    assert _gc_modules(src) == 0

    clone = build_anchor_module(src)

    assert _gc_modules(clone) == 0
