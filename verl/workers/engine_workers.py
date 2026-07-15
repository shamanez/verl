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
import functools
import logging
import os
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Optional

import psutil
import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, get_torch_device, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.comm_eff import maybe_build_comm_eff_state
from verl.workers.comm_eff.state import comm_eff_metrics
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import ppo_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _with_routing_replay_flag(enabled: bool):
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


class TrainingWorker(Worker, DistProfilerExtension):
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)
        self.engine_config.use_fused_kernels = self.model_config.get("use_fused_kernels", False)

        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=self.profiler_tool_config)
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        if hasattr(self.model_config, "hf_config"):
            self.flops_counter = FlopsCounter(self.model_config.hf_config)
        else:
            self.flops_counter = None

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().item()
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # log memory
        final_metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        final_metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        final_metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None and self.flops_counter is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @contextmanager
    def _comm_eff_anchor_batch_context(self, data: TensorDict, batch_size_per_dp: int):
        """Expose one immutable pre-split actor batch to an opt-in full anchor.

        ``train_mini_batch`` owns the only point at which the complete
        worker-local update batch still exists.  The FSDP anchor normally sees
        only one iterator-produced PPO mini-batch.  Under
        ``anchor.batch_scope=rollout_batch`` we deep-clone the complete batch to
        CPU before that split and lend it to the engine for this update only.
        Paired replay/rank1 warmup retain independent deep clones of this
        private full-update source, so no later mini-batch metadata mutation or
        transient-context cleanup can break the retained checkpoint/batch
        association.

        The scope is shared by the anchor-owned Q observation and dense M
        backward.  Cleanup is fail-closed and exception-safe: a stale context
        may never leak into the next actor update.
        """
        state = getattr(self.engine, "_comm_eff_state", None)
        anchor_cfg = getattr(getattr(state, "config", None), "anchor", None)
        if state is None or anchor_cfg is None or not bool(getattr(anchor_cfg, "enabled", False)):
            yield
            return

        dp_size = int(self.engine.get_data_parallel_size())
        update_sequences_global = int(batch_size_per_dp) * dp_size
        # These stamps ride mini-batch clones into delayed replay.  In
        # particular, they let telemetry report the honest fraction of the
        # complete update even after the source batch has become stale.
        tu.assign_non_tensor(
            data,
            comm_eff_update_sequences_local=int(batch_size_per_dp),
            comm_eff_update_sequences_global=update_sequences_global,
        )

        scope = str(getattr(anchor_cfg, "batch_scope", "ppo_minibatch"))
        if scope == "ppo_minibatch":
            yield
            return
        if scope != "rollout_batch":  # validated earlier; retain a local fail-closed guard
            raise RuntimeError(f"unsupported comm_eff anchor batch_scope={scope!r}")

        attr = "_comm_eff_rollout_batch"
        if hasattr(self.engine, attr):
            raise RuntimeError("comm_eff rollout-batch anchor context leaked across actor updates")

        from verl.workers.comm_eff.anchor import clone_batch_for_replay

        full_batch = clone_batch_for_replay(data, device=torch.device("cpu"))
        # The trainer's inherited global_batch_size is the PPO mini-batch size.
        # Replace it on the private full clone so every aggregation mode (not
        # only the locked token-mean mode) normalizes over this complete batch.
        tu.assign_non_tensor(full_batch, global_batch_size=update_sequences_global)
        if int(full_batch.shape[0]) != int(batch_size_per_dp):
            raise RuntimeError(
                "comm_eff rollout-batch clone changed the worker-local row count: "
                f"expected={batch_size_per_dp} got={full_batch.shape[0]}"
            )
        object.__setattr__(self.engine, attr, full_batch)
        try:
            yield
        finally:
            if hasattr(self.engine, attr):
                delattr(self.engine, attr)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        dataloader = tu.make_iterator(
            data,
            mini_batch_size=mini_batch_size_per_gpu,
            epochs=epochs,
            seed=seed + self.engine.get_data_parallel_rank(),
            dataloader_kwargs=dataloader_kwargs,
        )

        with (
            self._comm_eff_anchor_batch_context(data, batch_size_per_dp),
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # update
            output_lst = []
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

            for batch_idx, mini_batch_td in enumerate(dataloader):
                # add global token num
                if "input_ids" in mini_batch_td:
                    global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                    # allgather from dp rank
                    global_token_num_output = [None] * torch.distributed.get_world_size(
                        self.engine.get_data_parallel_group()
                    )
                    torch.distributed.all_gather_object(
                        global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                    )
                    global_token_num = [x for xs in global_token_num_output for x in xs]
                else:
                    global_token_num = None

                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                actor_output = self.train_batch(mini_batch_td)
                output_lst.append(actor_output)

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="train_batch")
    def train_batch(self, data: TensorDict) -> TensorDict:
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role
        self.actor: TrainingWorker = None
        self.ref: TrainingWorker = None
        self.rollout: BaseRollout = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        self.enable_routing_replay = (
            self.config.actor.strategy == "megatron" and self.config.actor.megatron.router_replay.mode != "disabled"
        )

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type=ref_config.model_config.get("model_type", "language_model"),
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
            )

            # assign engine configs
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.ref.ppo_micro_batch_size_per_gpu
            )
            ref_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            self.ref = TrainingWorker(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type=actor_config.model_config.get("model_type", "language_model"),
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if self.distillation_enabled:
                comm_eff = actor_config.comm_eff
                anchor = getattr(comm_eff, "anchor", None)
                if bool(getattr(comm_eff, "enabled", False)) and bool(getattr(anchor, "enabled", False)):
                    raise ValueError(
                        "comm_eff anchor objective parity does not yet support "
                        "distillation_ppo_loss. Disable the anchor or implement and test "
                        "an explicit ratio-one mapping for every distillation term."
                    )
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor = TrainingWorker(config=actor_training_config)
            self.actor.reset()
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            # If custom_backend_module is set, import it so plugins can register
            # in CheckpointEngineRegistry before the backend is instantiated.
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        # Reference-policy log-prob is an RL-measurement path. The path tag keeps
        # actor-train compression confined even if this path shares the engine.
        with self._comm_eff_path("ref_logprob"):
            output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        # With compress_recompute enabled, old-policy log-prob uses the same
        # frozen PowerSGD basis as the paired actor-train forward.
        with self._comm_eff_path("old_logprob"):
            comm_eff_state = self._maybe_comm_eff_state()
            self._comm_eff_thread_global_step(data, comm_eff_state)
            stamped_compression_active = False
            prev_compression_active = False
            if comm_eff_state is not None:
                ps_cfg = getattr(comm_eff_state.config, "powersgd", None)
                ps_recompute = bool(getattr(ps_cfg, "compress_recompute", False)) if ps_cfg is not None else False
                if getattr(comm_eff_state, "powersgd", None) is not None and ps_recompute:
                    prev_compression_active = bool(getattr(comm_eff_state, "compression_active", False))
                    comm_eff_state.compression_active = True
                    stamped_compression_active = True
            try:
                output = self.actor.infer_batch(data)
            finally:
                if stamped_compression_active and comm_eff_state is not None:
                    comm_eff_state.compression_active = prev_compression_active

        return output.cpu() if output is not None else None

    def _maybe_comm_eff_state(self):
        """Return this worker's comm_eff state, building it once on first use.

        Disabled is the strict no-op path: ``maybe_build_comm_eff_state`` returns
        ``None`` without drawing RNG, allocating buffers or registering hooks, so
        dense GRPO remains unaffected. The result is cached so repeated
        ``update_actor`` calls do not re-read the config each time.
        """
        state = getattr(self, "_comm_eff_state", None)
        if state is None and not getattr(self, "_comm_eff_state_built", False):
            comm_eff_cfg = self.config.actor.get("comm_eff", None)
            state = maybe_build_comm_eff_state(comm_eff_cfg)
            # object.__setattr__ avoids any frozen-config interplay; these are
            # plain worker attributes, not config fields.
            object.__setattr__(self, "_comm_eff_state", state)
            object.__setattr__(self, "_comm_eff_state_built", True)
            if state is None and not getattr(self, "_comm_eff_marker_logged", False):
                logger.info("comm_eff: disabled (no-op) — dense GRPO path unchanged")
                object.__setattr__(self, "_comm_eff_marker_logged", True)
            if state is not None:
                # Build the circuits and attach their shared state to
                # the actor engine. The worker scopes compression_active around
                # paired forwards; the engine owns hook and correction lifetimes.
                engine = getattr(getattr(self, "actor", None), "engine", None)
                if engine is not None:
                    state.build(getattr(engine, "module", None))
                    object.__setattr__(engine, "_comm_eff_state", state)
                    logger.info("comm_eff: enabled — pipeline attached to actor train engine")
                    # Bind the actor DP group so the PowerSGD basis all-reduce
                    # pools sketches over exactly the data-parallel ranks.
                    powersgd = getattr(state, "powersgd", None)
                    if powersgd is not None and hasattr(engine, "get_data_parallel_group"):
                        try:
                            dp_group = engine.get_data_parallel_group()
                            powersgd.set_dp_group(dp_group)
                            try:
                                import torch.distributed as _dist

                                ws = _dist.get_world_size()
                                dp_ws = _dist.get_world_size(group=dp_group) if dp_group is not None else ws
                                logger.info(
                                    "comm_eff.powersgd: basis sync bound to DP group "
                                    "(dp_world_size=%s, global_world_size=%s, sync_basis=%s)",
                                    dp_ws,
                                    ws,
                                    getattr(powersgd, "sync_basis", None),
                                )
                                print(
                                    f"[comm_eff] basis-sync DP group: dp_world_size={dp_ws} "
                                    f"global_world_size={ws} sync_basis={getattr(powersgd, 'sync_basis', None)}",
                                    flush=True,
                                )
                            except Exception:  # pragma: no cover - logging only
                                pass
                        except Exception as e:  # pragma: no cover - defensive
                            logger.warning("comm_eff.powersgd: could not bind DP group (%s); using world group", e)
        return getattr(self, "_comm_eff_state", None)

    def _comm_eff_thread_global_step(self, data: TensorDict, state) -> Optional[int]:
        """Thread the trainer step onto the comm_eff state.

        The trainer stamps ``batch.meta_info["comm_eff_global_step"]`` before each
        ``update_actor`` / ``compute_log_prob`` call; ``DataProto.to_tensordict``
        carries ``meta_info`` into the worker ``data`` as non-tensor entries, so
        we read it back here with ``tu.get``. The value is stored on the shared
        state and mirrored onto the train engine for PowerSGD cadence, anchor
        timing, and rank-1 RELEX history.

        We use a comm_eff-private meta_info key (``comm_eff_global_step``) rather
        than the bare ``global_steps``: the vLLM rollout already emits
        ``global_steps`` as a per-sample batch column (``extra_fields`` in
        vllm_async_server.py), so adding ``global_steps`` to ``meta_info`` would
        collide in ``to_tensordict``'s "meta key must not be a batch column"
        assert. A private key cannot collide with any batch column.

        Returns ``None`` when ``state`` is absent or the caller did not stamp a
        step; in either case existing runtime state is left untouched.
        """
        if state is None:
            return None
        gs = tu.get(data, key="comm_eff_global_step", default=None)
        if gs is None:
            return None
        gs = int(gs)
        state.global_step = gs
        engine = getattr(getattr(self, "actor", None), "engine", None)
        if engine is not None:
            object.__setattr__(engine, "_comm_eff_global_step", gs)
        return gs

    @contextmanager
    def _comm_eff_path(self, tag: str):
        """Stamp the comm_eff execution-path ``tag`` for the wrapped forward.

        PowerSGD is allowed only on the actor-train path and, when explicitly
        configured, the paired old-logprob recompute. The prior tag is restored
        on exit so nested operations cannot leak compression into inference,
        validation, reference-policy, or checkpoint paths.
        """
        state = self._maybe_comm_eff_state()
        if state is None:
            yield
            return
        prev = getattr(state, "path_tag", None)
        state.set_path_tag(tag)
        try:
            yield
        finally:
            state.set_path_tag(prev)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        # comm_eff guard. When disabled (default) this resolves to None with zero
        # side effects (no hook, no buffer, no RNG) and the dense GRPO update runs
        # without modification. The compressed circuits are entered only when
        # comm_eff.enabled=true; the disabled path never touches the gradient, so
        # the no-op parity holds.
        #
        # Optimizer-step ordering. Per
        # actor train_batch (reached via train_mini_batch -> engine.train_batch):
        #   [anchor: DENSE K-stale fwd/bwd -> RAW G_anchor -> EMA, NO step]
        #   -> compressed fwd/bwd -> FSDP reduction -> signed EMA -> AdamW.
        # Building the state here attaches the SpectralFilter (+ the anchor's
        # staleness queue, lazily) to the actor train engine; the engine's
        # _maybe_comm_eff_anchor_refresh hook fires at the TOP of
        # BaseEngine.train_batch (before the compressed path; G_anchor is read
        # raw before any correction) and the _maybe_comm_eff_grad_correction hook
        # fires AFTER backward (grads FSDP-reduced) and BEFORE optimizer_step.
        # The anchor cadence (comm_eff.anchor.cadence) gates per-step firing.
        comm_eff_state = self._maybe_comm_eff_state()

        # Thread the trainer step into the communication-efficient circuits.
        global_step = self._comm_eff_thread_global_step(data, comm_eff_state)

        # Scope projection hooks to the actor-train forward/backward. Other paths
        # never set this flag and therefore remain dense.
        if comm_eff_state is not None:
            comm_eff_state.compression_active = True
            comm_eff_state.set_path_tag("train")
        actor_update_succeeded = False
        try:
            output = self.actor.train_mini_batch(data=data)
            actor_update_succeeded = True
        finally:
            if comm_eff_state is not None:
                comm_eff_state.compression_active = False
                comm_eff_state.set_path_tag(None)
                # PowerSGD block-power-iteration basis update. Runs
                # ONCE per trainer step, AFTER all PPO mini-batch forwards/backwards
                # of this update_actor have run (so the sketch V has folded in
                # every gradient-bearing actor-train forward) and AFTER the
                # gradient-bearing work (so Q was frozen for both paired GRPO
                # forwards this step). Sets Q_t -> Q_{t+1} for the NEXT
                # step. Non-cadence steps are skipped inside maybe_update_basis.
                # Strict no-op for dense/disabled runs (powersgd is None there).
                powersgd = getattr(comm_eff_state, "powersgd", None)
                # When the anchor owns Q, the fast net is a pure read-only
                # consumer. An anchor fire occurs inside train_mini_batch, after
                # this batch's old_log_probs were already recomputed. The anchor
                # therefore stages Q_{t+1}; publish it only after ALL PPO
                # minibatches sharing those old_log_probs have completed. The
                # next global step's old-logprob and train forwards then see the
                # same new Q. The fast-side sketch accumulation stays gated off.
                anchor_owns_q = bool(getattr(powersgd, "anchor_owns_q", False)) if powersgd is not None else False
                if powersgd is not None and anchor_owns_q:
                    if actor_update_succeeded:
                        # Every rank executes this handoff, even ranks on which
                        # train_mini_batch returned no metrics. The compressor
                        # candidate was already broadcast/verified collectively
                        # at the anchor fire, so this outer exception boundary is
                        # a local atomic handoff (no deadlock-prone collective).
                        try:
                            did_activate = powersgd.activate_staged_anchor_basis()
                        except Exception:
                            powersgd.discard_staged_anchor_basis()
                            raise
                        if did_activate:
                            print(
                                f"[comm_eff][q-stage] activated after update_actor global_step={global_step} "
                                f"generation={powersgd.anchor_basis_generation} "
                                f"activations={getattr(comm_eff_state, 'anchor_q_activations', 0)}",
                                flush=True,
                            )
                    else:
                        # A candidate was derived from an update that did not
                        # commit. Never let it leak into a later policy pair.
                        powersgd.discard_staged_anchor_basis()
                fast_owns_q = not anchor_owns_q
                if powersgd is not None and fast_owns_q:
                    did_update = powersgd.maybe_update_basis()
                    # After the first basis update, verify Q is bit-identical on
                    # every DP rank. This gate is symmetric across ranks and
                    # raises on a real divergence.
                    if did_update and not getattr(comm_eff_state, "_powersgd_q_agreement_checked", False):
                        try:
                            dev = powersgd.verify_basis_agreement_across_ranks()
                            object.__setattr__(comm_eff_state, "_powersgd_q_agreement_checked", True)
                            object.__setattr__(comm_eff_state, "_powersgd_q_agreement_dev", dev)
                            if dev is not None:
                                logger.info(
                                    "comm_eff.powersgd: cross-rank Q agreement verified "
                                    "(max_rel_dev=%.3e, sync_basis=%s)",
                                    dev,
                                    getattr(powersgd, "sync_basis", None),
                                )
                                print(
                                    f"[comm_eff] cross-rank Q agreement: max_rel_dev={dev:.3e} "
                                    f"sync_basis={getattr(powersgd, 'sync_basis', None)}",
                                    flush=True,
                                )
                        except RuntimeError:
                            # A genuine divergence must fail the training step.
                            raise

        # Finalize this train_batch's measured inter-stage comm
        # volume. The powersgd hook accumulated Σ N·r (compressed Y) and Σ N·H
        # (dense-equiv) over the fast-train forward; add the amortized per-tick
        # Q-broadcast term (H·r/cadence per boundary) and snapshot into last_elems_*
        # so powersgd_metrics() surfaces comm/bytes_compressed + comm/bytes_dense_equiv
        # + comm/bytes_ratio. No-op for dense/disabled runs.
        if comm_eff_state is not None:
            _ps = getattr(comm_eff_state, "powersgd", None)
            if _ps is not None and hasattr(_ps, "add_amortized_q_broadcast_bytes"):
                _ps.add_amortized_q_broadcast_bytes()
                _ps.last_elems_compressed = float(getattr(_ps, "tick_elems_compressed", 0.0))
                _ps.last_elems_dense_equiv = float(getattr(_ps, "tick_elems_dense_equiv", 0.0))

        # Surface the comm_eff operation counters into training metrics. When
        # disabled we emit explicit zeros for the communication-efficient circuits
        # so the no-op is machine-checkable; emitting a constant metric is not a
        # numerical side effect on training. `output` is
        # None on non-output ranks (train_mini_batch only populates metrics on the
        # mp-src rank), in which case there is nothing to annotate.
        if output is not None:
            if comm_eff_state is None:
                counters = {
                    "comm_eff/anchor_backwards": 0,
                    "comm_eff/spectral_corrections": 0,
                    # Anchor counters: explicit zeros on the disabled path so
                    # the no-op stays machine-checkable.
                    "comm_eff/anchor_grad_corrected": 0,
                    "comm_eff/anchor_rollouts_generated": 0,
                    "comm_eff/anchor_rewards_recomputed": 0,
                    "comm_eff/anchor_optimizer_steps": 0,
                    "comm_eff/anchor_batch_fraction": 0.0,
                    "comm_eff/anchor_batch_sequences_global": 0,
                    "comm_eff/anchor_update_sequences_global": 0,
                    "comm_eff/anchor_batch_prompt_equivalents_global": 0,
                    "comm_eff/anchor_update_prompt_equivalents_global": 0,
                    "comm_eff/anchor_rollout_n": 0,
                    "comm_eff/anchor_batch_scope_rollout": 0,
                    # Explicit zeros on the disabled path for PowerSGD counters.
                    "comm_eff/powersgd_applications": 0,
                    "comm_eff/powersgd_basis_updates": 0,
                    # Explicit zeros on the disabled path for anchor-owned-Q counters.
                    "comm_eff/anchor_q_updates": 0,
                    "comm_eff/anchor_q_broadcasts": 0,
                    "comm_eff/anchor_q_activations": 0,
                    "comm_eff/anchor_q_stage_overwrites": 0,
                    "comm_eff/fast_q_bootstrap_done": 0,
                    "comm_eff/fast_q_bootstrap_observations": 0,
                    "comm_eff/fast_q_bootstrap_updates": 0,
                    "comm_eff/fast_q_bootstrap_activations": 0,
                    "comm_eff/fast_q_bootstrap_dense_observation_elements": 0.0,
                    "comm_eff/fast_q_bootstrap_sync_elements": 0.0,
                    "comm_eff/merger_coldM_fallbacks": 0,
                    "comm_eff/anchor_replay_fires": 0,
                }
            else:
                counters = comm_eff_metrics(comm_eff_state)
            metrics = tu.get(output, "metrics", default=None)
            if isinstance(metrics, dict):
                metrics.update(counters)

        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        # A checkpoint load may run an actor forward; keep it outside compression.
        with self._comm_eff_path("ckpt"):
            self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)
        state = self._maybe_comm_eff_state()
        if state is not None and hasattr(state, "reset_rank1_runtime") and state.rank1_relex_active():
            state.reset_rank1_runtime()
            print(
                "[comm_eff][rank1_relex] checkpoint load reset local history/M/Q; "
                "resume will seed a fresh base and rewarm",
                flush=True,
            )
        elif state is not None:
            # Staging is shared by every anchor-owned-Q arm. Even modes that do
            # not own rank1 trajectory history must never activate a candidate
            # computed from weights that preceded this checkpoint load.
            powersgd = getattr(state, "powersgd", None)
            if powersgd is not None and bool(getattr(powersgd, "anchor_owns_q", False)):
                state.reset_anchor_q_runtime()
                print(
                    "[comm_eff] checkpoint load reset non-checkpointed anchor-owned Q runtime",
                    flush=True,
                )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        # Checkpoint save must not carry a live compressed-forward context.
        with self._comm_eff_path("ckpt"):
            self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.

        LoRA handling: when model.lora.merge=True (peft_merge), LoRA is merged into
        base weights before sync. The engine returns full HF-keyed params with
        peft_config=None, so the rollout receives a standard weight update.

        Args:
            global_steps: Current global training step count, passed to rollout for logging/tracking.
            mode: Weight update strategy. Supported values:
                - ``"auto"``: Automatically resolve to the backend configured in
                  ``config.rollout.checkpoint_engine.backend`` (default).
                - ``"naive"``: Direct in-process weight sync between colocated trainer
                  and rollout. Used for synchronous training where both share the same
                  process. Rollout must be in sleep mode before this call.
                - Any other value: Delegates to
                  :meth:`checkpoint_engine.send_weights` for asynchronous weight
                  transfer via checkpoint engine, suitable for disaggregated
                  trainer/rollout deployments.
        """

        # Resolve mode: "auto" falls back to config, explicit values take precedence
        effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if effective_mode != "naive":
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(per_tensor_param)
            return

        set_expandable_segments(False)
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # 2. determine if we need a base weight sync (adapter path only)
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done

        # 3. sync weights: For SGLang, we need base first (when needed), then adapter/merged
        if do_lora_base_sync:
            per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(
                per_tensor_param_base, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
            )

        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
        )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
