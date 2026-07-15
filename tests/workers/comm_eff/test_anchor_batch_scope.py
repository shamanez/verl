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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.comm_eff.anchor import clone_batch_for_replay, select_anchor_batch_for_scope
from verl.workers.config.comm_eff import CommEffAnchorConfig, CommEffConfig
from verl.workers.engine_workers import TrainingWorker


def _worker(scope: str, dp_size: int = 2):
    anchor = SimpleNamespace(enabled=True, batch_scope=scope)
    state = SimpleNamespace(config=SimpleNamespace(anchor=anchor))
    engine = SimpleNamespace(
        _comm_eff_state=state,
        get_data_parallel_size=lambda: dp_size,
    )
    return SimpleNamespace(engine=engine)


def _batch(rows: int = 6):
    data = TensorDict({"payload": torch.arange(rows * 2).reshape(rows, 2)}, batch_size=[rows])
    # The trainer initially stamps the PPO mini-batch global size. Full scope
    # must replace this value on its private clone without mutating the source.
    tu.assign_non_tensor(data, global_batch_size=4)
    return data


def test_anchor_batch_scope_config_enum_and_default():
    assert CommEffAnchorConfig().batch_scope == "ppo_minibatch"
    assert CommEffConfig(anchor=CommEffAnchorConfig(batch_scope="rollout_batch")).anchor.batch_scope == "rollout_batch"
    with pytest.raises(ValueError, match="batch_scope"):
        CommEffConfig(anchor=CommEffAnchorConfig(batch_scope="fullish"))


def test_rollout_batch_context_clones_full_batch_and_cleans_up_on_error():
    worker = _worker("rollout_batch")
    data = _batch()
    original = data["payload"].clone()

    with pytest.raises(RuntimeError, match="sentinel"):
        with TrainingWorker._comm_eff_anchor_batch_context(worker, data, batch_size_per_dp=6):
            full = worker.engine._comm_eff_rollout_batch
            assert full.shape[0] == 6
            assert full["payload"].device.type == "cpu"
            assert tu.get(full, "global_batch_size") == 12
            assert tu.get(full, "comm_eff_update_sequences_global") == 12
            data["payload"].zero_()
            torch.testing.assert_close(full["payload"], original)
            raise RuntimeError("sentinel")

    assert not hasattr(worker.engine, "_comm_eff_rollout_batch")


def test_ppo_minibatch_context_stamps_denominator_without_full_clone():
    worker = _worker("ppo_minibatch")
    data = _batch()

    with TrainingWorker._comm_eff_anchor_batch_context(worker, data, batch_size_per_dp=6):
        assert not hasattr(worker.engine, "_comm_eff_rollout_batch")
        assert tu.get(data, "comm_eff_update_sequences_local") == 6
        assert tu.get(data, "comm_eff_update_sequences_global") == 12
        # Historical mini-batch normalization remains untouched.
        assert tu.get(data, "global_batch_size") == 4


def test_anchor_batch_source_selection_fails_closed_without_full_context():
    current = _batch(rows=2)
    full = _batch(rows=6)

    assert select_anchor_batch_for_scope("ppo_minibatch", current, full) is current
    assert select_anchor_batch_for_scope("rollout_batch", current, full) is full
    with pytest.raises(RuntimeError, match="no full update batch"):
        select_anchor_batch_for_scope("rollout_batch", current, None)
    with pytest.raises(RuntimeError, match="unsupported"):
        select_anchor_batch_for_scope("all_the_data", current, full)


def test_retained_full_batches_do_not_alias_context_or_each_other():
    source = _batch(rows=6)
    retained_a = clone_batch_for_replay(source, device=torch.device("cpu"))
    retained_b = clone_batch_for_replay(source, device=torch.device("cpu"))
    expected = source["payload"].clone()

    source["payload"].zero_()
    torch.testing.assert_close(retained_a["payload"], expected)
    torch.testing.assert_close(retained_b["payload"], expected)
    retained_a["payload"].fill_(-1)
    torch.testing.assert_close(retained_b["payload"], expected)


def test_full_batch_token_normalization_matches_microbatch_accumulation_and_dp_mean():
    from verl.trainer.ppo.core_algos import agg_loss

    rank0_loss = torch.tensor([[1.0, 3.0], [2.0, 5.0]])
    rank0_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    rank1_loss = torch.tensor([[7.0, 11.0], [13.0, 17.0]])
    rank1_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    global_tokens = rank0_mask.sum() + rank1_mask.sum()

    def _rank_accum(loss, mask):
        total = torch.zeros(())
        for i in range(loss.shape[0]):
            total = total + agg_loss(
                loss_mat=loss[i : i + 1],
                loss_mask=mask[i : i + 1],
                loss_agg_mode="token-mean",
                dp_size=2,
                batch_num_tokens=global_tokens,
                global_batch_size=4,
                loss_scale_factor=None,
            )
        return total

    # FSDP/DDP averages the two already-DP-scaled rank gradients. This must be
    # identical to one global full-batch token mean, even though each rank ran
    # multiple dynamic microbatches.
    distributed_loss = (_rank_accum(rank0_loss, rank0_mask) + _rank_accum(rank1_loss, rank1_mask)) / 2
    expected = ((rank0_loss * rank0_mask).sum() + (rank1_loss * rank1_mask).sum()) / global_tokens
    torch.testing.assert_close(distributed_loss, expected)
