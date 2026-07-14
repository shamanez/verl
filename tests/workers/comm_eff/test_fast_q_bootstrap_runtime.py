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

"""Runtime ordering contracts for the opt-in pre-old-logprob fast-Q bootstrap."""

from types import SimpleNamespace

import pytest

from verl.workers.comm_eff.state import OLD_LOGPROB_TAG, TRAIN_TAG
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _FakeBootstrapCompressor:
    """Transaction spy matching the small compressor surface the engine drives."""

    def __init__(self):
        self.live_q = "Q0"
        self.pending_q = None
        self.observing = False
        self.done = False
        self.unregistered = 0
        self.aborted = 0
        self.fast_q_bootstrap_observations = 0
        self.fast_q_bootstrap_updates = 0
        self.fast_q_bootstrap_activations = 0
        self.fast_q_bootstrap_dense_observation_elements = 64.0
        self.fast_q_bootstrap_sync_elements = 16.0

    def fast_q_bootstrap_needed(self):
        return not self.done

    def begin_fast_q_bootstrap_observation(self):
        assert not self.done and not self.observing
        self.observing = True

    def finish_fast_q_bootstrap_observation(self):
        assert self.observing
        self.observing = False
        self.fast_q_bootstrap_observations += 1

    def stage_fast_q_bootstrap_basis(self):
        assert not self.observing and self.live_q == "Q0"
        self.pending_q = "Q1"
        self.fast_q_bootstrap_updates += 1
        return True

    def verify_fast_q_bootstrap_basis_across_ranks(self):
        assert self.pending_q == "Q1" and self.live_q == "Q0"
        return 0.0

    def activate_staged_fast_q_bootstrap_basis(self):
        assert self.pending_q == "Q1" and self.live_q == "Q0"
        self.live_q = self.pending_q
        self.pending_q = None
        self.done = True
        self.fast_q_bootstrap_activations += 1
        return True

    def abort_fast_q_bootstrap(self):
        had_state = self.observing or self.pending_q is not None
        self.observing = False
        self.pending_q = None
        self.aborted += 1
        return had_state

    def unregister(self):
        self.unregistered += 1


class _FakeEngine:
    def __init__(self, *, fail_observation=False):
        self.compressor = _FakeBootstrapCompressor()
        self._comm_eff_state = SimpleNamespace(
            enabled=True,
            path_tag=OLD_LOGPROB_TAG,
            powersgd=self.compressor,
        )
        self._comm_eff_global_step = 1
        self.events = []
        self.fail_observation = fail_observation

    def _comm_eff_mask_active(self, forward_only):
        return False

    def _comm_eff_powersgd_active(self, forward_only):
        return True

    def _comm_eff_register_powersgd_hooks(self):
        return True

    def _forward_backward_batch_inner(
        self,
        data,
        loss_function,
        forward_only=False,
        run_backward=True,
        collect_outputs=True,
    ):
        if self.compressor.observing:
            self.events.append(("discarded_dense_observation", self.compressor.live_q))
            if self.fail_observation:
                raise RuntimeError("observation failed")
            return []
        tag = "old_logprob" if forward_only else "current_train"
        self.events.append((tag, self.compressor.live_q))
        return [tag]


def test_bootstrap_activates_before_real_oldlog_and_stays_frozen_for_current():
    engine = _FakeEngine()

    old_output = FSDPEngine.forward_backward_batch(engine, data={}, loss_function=None, forward_only=True)
    assert old_output == ["old_logprob"]
    assert engine.events == [
        ("discarded_dense_observation", "Q0"),
        ("old_logprob", "Q1"),
    ]
    assert engine.compressor.fast_q_bootstrap_observations == 1
    assert engine.compressor.fast_q_bootstrap_updates == 1
    assert engine.compressor.fast_q_bootstrap_activations == 1

    engine._comm_eff_state.path_tag = TRAIN_TAG
    train_output = FSDPEngine.forward_backward_batch(engine, data={}, loss_function=None, forward_only=False)
    assert train_output == ["current_train"]
    assert engine.events[-1] == ("current_train", "Q1")
    # A later old-logprob call is one pass only: no second observation/sync.
    engine._comm_eff_state.path_tag = OLD_LOGPROB_TAG
    FSDPEngine.forward_backward_batch(engine, data={}, loss_function=None, forward_only=True)
    assert engine.events[-1] == ("old_logprob", "Q1")
    assert sum(event[0] == "discarded_dense_observation" for event in engine.events) == 1


def test_bootstrap_prepass_exception_aborts_without_policy_forward_or_q_swap():
    engine = _FakeEngine(fail_observation=True)
    with pytest.raises(RuntimeError, match="observation failed"):
        FSDPEngine.forward_backward_batch(engine, data={}, loss_function=None, forward_only=True)
    assert engine.events == [("discarded_dense_observation", "Q0")]
    assert engine.compressor.live_q == "Q0"
    assert engine.compressor.pending_q is None
    assert not engine.compressor.done
    assert engine.compressor.aborted == 1
    assert engine.compressor.unregistered == 1
