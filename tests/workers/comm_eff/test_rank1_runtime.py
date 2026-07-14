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

"""CPU contracts for rank1_relex Q-only warmup and correction readiness."""

import weakref
from types import SimpleNamespace

import torch

from verl.workers.comm_eff.lookahead import validate_rank1_broadcast_receipts
from verl.workers.comm_eff.powersgd_activation import PowerSGDActivationCompressor
from verl.workers.comm_eff.state import CommEffState
from verl.workers.engine.fsdp import transformer_impl as fsdp_impl


class _TinyModel(torch.nn.Module):
    """Four decoder-like blocks; pp_size=4 selects the first three boundaries."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(4)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _PowerSGDState:
    def __init__(self):
        self.path_tag = None
        self.anchor_q_updates = 0
        self.anchor_q_broadcasts = 0
        self.anchor_q_activations = 0
        self.anchor_q_stage_overwrites = 0
        self.powersgd_applications = 0
        self.powersgd_basis_updates = 0

    def note_powersgd_application(self):
        self.powersgd_applications += 1

    def note_powersgd_basis_update(self):
        self.powersgd_basis_updates += 1


def test_q_only_autograd_forward_refreshes_q_without_backward_or_m():
    torch.manual_seed(0)
    model = _TinyModel()
    state = _PowerSGDState()
    spectral = SimpleNamespace(_anchor={})
    compressor = PowerSGDActivationCompressor(
        rank=8,
        base_seed=0,
        pp_size=4,
        update_cadence=1,
        warm_start=True,
        compress_recompute=True,
        sync_basis=False,
        qr_dtype="fp32",
        reortho_eps=1e-6,
        anchor_owns_q=True,
        q_basis="act",
        q_basis_passive=[],
        anchor_cadence=20,
        state=state,
    )
    compressor.register(model)
    try:
        compressor.set_context(global_step=10)
        compressor.set_anchor_sketch_mode(True)
        output = model(torch.randn(12, 32))

        # The forward retained its graph so the activation hooks could harvest V,
        # but q_only deliberately never calls backward.
        assert output.requires_grad
        assert all(parameter.grad is None for parameter in model.parameters())
        assert set(compressor._sketch) == set(compressor.boundary_indices)
        q_before = {layer: basis.clone() for layer, basis in compressor._basis.items()}
        compressor.set_anchor_sketch_mode(False)
        probe_input = torch.randn(5, 32)
        with torch.no_grad():
            projected_before_stage = model(probe_input)

        assert compressor.anchor_update_basis(staged=True)
        q_receipts = compressor.broadcast_basis(staged=True)
        validate_rank1_broadcast_receipts(
            q_receipts=q_receipts,
            m_receipts=None,
            q_only=True,
            dp_multi=False,
            spectral_enabled=True,
        )

        assert state.anchor_q_updates == 1
        # The slow-net candidate is ready, but the old/current policy pair keeps
        # the exact live Q until the complete actor update commits.
        assert all(torch.equal(compressor._basis[layer], q_before[layer]) for layer in q_before)
        assert any(not torch.equal(compressor._pending_anchor_basis[layer], q_before[layer]) for layer in q_before)
        assert compressor.anchor_basis_generation == 0
        with torch.no_grad():
            projected_while_staged = model(probe_input)
        torch.testing.assert_close(projected_while_staged, projected_before_stage, rtol=0, atol=0)

        # A second anchor fire in the same actor transaction is explicitly
        # last-candidate-wins, while live Q remains frozen.
        first_candidate = {layer: basis.clone() for layer, basis in compressor._pending_anchor_basis.items()}
        compressor.set_context(global_step=11)
        compressor.set_anchor_sketch_mode(True)
        model(torch.randn(12, 32))
        compressor.set_anchor_sketch_mode(False)
        assert compressor.anchor_update_basis(staged=True)
        compressor.broadcast_basis(staged=True)
        assert state.anchor_q_updates == 2
        assert state.anchor_q_stage_overwrites == 1
        assert any(
            not torch.equal(compressor._pending_anchor_basis[layer], first_candidate[layer])
            for layer in first_candidate
        )
        assert all(torch.equal(compressor._basis[layer], q_before[layer]) for layer in q_before)

        assert compressor.activate_staged_anchor_basis()
        assert any(not torch.equal(compressor._basis[layer], q_before[layer]) for layer in q_before)
        assert compressor._pending_anchor_basis == {}
        assert compressor.anchor_basis_generation == 1
        assert state.anchor_q_activations == 1
        assert not compressor.activate_staged_anchor_basis()
        with torch.no_grad():
            projected_after_activation = model(probe_input)
        assert not torch.equal(projected_after_activation, projected_before_stage)

        live_after_activation = {layer: basis.clone() for layer, basis in compressor._basis.items()}
        compressor._pending_anchor_basis = {
            layer: torch.flip(basis, dims=(0,)) for layer, basis in live_after_activation.items()
        }
        assert compressor.discard_staged_anchor_basis()
        assert compressor._pending_anchor_basis == {}
        assert all(
            torch.equal(compressor._basis[layer], live_after_activation[layer]) for layer in live_after_activation
        )
        assert not compressor.discard_staged_anchor_basis()

        compressor.reset_basis_runtime()
        assert compressor._basis == {}
        assert compressor._pending_anchor_basis == {}
        assert compressor.anchor_basis_generation == 0
        assert all(parameter.grad is None for parameter in model.parameters())
        assert spectral._anchor == {}
        assert compressor._sketch == {}
    finally:
        compressor.set_anchor_sketch_mode(False)
        compressor.clear_family_harvest()
        compressor.unregister()


def test_forward_inner_run_backward_false_retains_autograd_without_gradients(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor(3.0))
    graph_roots = []

    class _FakeEngine:
        ulysses_sequence_parallel_size = 1
        scaler = None

        def get_data_parallel_group(self):
            return None

        def get_data_parallel_size(self):
            return 1

        def forward_step(self, micro_batch, loss_function, forward_only):
            if graph_roots:
                # collect_outputs=False must release the preceding microbatch's
                # graph before beginning this one.
                assert all(ref() is None for ref in graph_roots[-1])
            loss = loss_function(parameter)
            model_output = loss * 2.0
            graph_roots.append((weakref.ref(loss), weakref.ref(model_output)))
            return loss, {"model_output": model_output}

    data = {"loss_mask": torch.ones(1)}
    monkeypatch.setattr(fsdp_impl.tu, "assign_non_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(fsdp_impl.torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(fsdp_impl, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(fsdp_impl, "prepare_micro_batches", lambda **kwargs: ([data, data, data], [0, 1, 2]))
    monkeypatch.setattr(fsdp_impl, "postprocess_batch_func", lambda **kwargs: kwargs["output_lst"])

    result = fsdp_impl.FSDPEngine._forward_backward_batch_inner(
        _FakeEngine(),
        data,
        lambda value: value.square(),
        forward_only=False,
        run_backward=False,
        collect_outputs=False,
    )
    assert result == []
    assert all(ref() is None for ref in graph_roots[-1])
    assert parameter.grad is None


def test_correction_bypass_advances_cadence_and_preserves_fast_grad_bits():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    parameter.grad = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    grad_before = parameter.grad.clone()

    class _Rank1WarmupState:
        enabled = True
        spectral = SimpleNamespace(correction_mode="signed_ema")
        spectral_step = 0
        spectral_corrections = 0
        rank1_m_ready = False
        rank1_correction_bypass_ticks = 0

        @staticmethod
        def rank1_relex_active():
            return True

        def should_run_spectral_correction(self):
            return self.spectral_step % 2 == 0

    state = _Rank1WarmupState()
    engine = SimpleNamespace(_comm_eff_state=state, module=torch.nn.Module())
    engine.module.register_parameter("weight", parameter)

    for _ in range(4):
        fsdp_impl.FSDPEngine._maybe_comm_eff_grad_correction(engine)

    assert state.spectral_step == 4
    assert state.rank1_correction_bypass_ticks == 2
    assert state.spectral_corrections == 0
    assert torch.equal(parameter.grad, grad_before)


def test_same_worker_resume_clears_rank1_history_q_and_m():
    config = SimpleNamespace(
        enabled=True,
        anchor=SimpleNamespace(lookahead_anchor=True, lookahead_mode="rank1_relex"),
    )
    state = CommEffState(config)
    state._rank1_history = object()
    state._rank1_projector = object()
    state._rank1_base_batch = object()
    state._rank1_projection_probe = object()
    state._anchor_replay_ring = object()
    state.rank1_m_ready = True
    state.rank1_fires = 3
    state.rank1_probe_predictions = 3
    state.rank1_probe_resolutions = 2
    state.spectral_step = 17
    state.spectral = SimpleNamespace(
        _anchor={"layers.0.q_proj.weight": torch.ones(2, 2)},
        _ef_residual={"layers.0.q_proj.weight": torch.ones(2, 2)},
        _delayed_ef_delta={"layers.0.q_proj.weight": torch.ones(2, 2)},
        _delta_momentum={"layers.0.q_proj.weight": torch.ones(2, 2)},
        _delta_momentum_last_step={"layers.0.q_proj.weight": 17},
        _adaptive_lambda_hist={"layers.0.q_proj.weight": [0.5]},
        _subbasis_energy_ratios=[1.0],
        current_step=17,
    )
    state.powersgd = SimpleNamespace(
        _basis={0: torch.ones(4, 2)},
        _pending_anchor_basis={0: torch.ones(4, 2)},
        _sketch={0: torch.ones(4, 2)},
        clear_family_harvest=lambda: None,
    )

    state.reset_rank1_runtime()

    assert not hasattr(state, "_rank1_history")
    assert not hasattr(state, "_rank1_projector")
    assert not hasattr(state, "_rank1_base_batch")
    assert not hasattr(state, "_rank1_projection_probe")
    assert not hasattr(state, "_anchor_replay_ring")
    assert state.spectral._anchor == {}
    assert state.spectral._ef_residual == {}
    assert state.spectral._delayed_ef_delta == {}
    assert state.spectral._delta_momentum == {}
    assert state.spectral._delta_momentum_last_step == {}
    assert state.spectral._adaptive_lambda_hist == {}
    assert state.spectral._subbasis_energy_ratios == []
    assert state.spectral.current_step == 0
    assert state.powersgd._basis == {}
    assert state.powersgd._pending_anchor_basis == {}
    assert state.powersgd._sketch == {}
    assert state.spectral_step == 0
    assert not state.rank1_m_ready
    assert state.rank1_fires == 0
    assert state.rank1_probe_predictions == 0
    assert state.rank1_probe_resolutions == 0
