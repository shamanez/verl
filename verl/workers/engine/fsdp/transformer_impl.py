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
"""
The concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP)
"""

import gc
import logging
import os
import warnings
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    merged_lora_context,
    normalize_peft_param_name,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.utils.padding import build_attention_mask_from_nested

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class FSDPEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP).

    Supports model sharding, activation/optimizer offloading, LoRA, and sequence parallelism.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the FSDPEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with FSDP and model settings.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None

        self.rank = torch.distributed.get_rank()

        # Apply NPU patches for FSDP backend
        from .utils import apply_npu_fsdp_patches

        apply_npu_fsdp_patches(self.model_config)

        # build device mesh for Ulysses Sequence Parallel

        self.use_remove_padding = self.model_config.use_remove_padding

        self._init_device_mesh()

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0
        # Set in _build_fsdp_module when FSDP2 CPUOffloadPolicy is configured (see #5995).
        self._uses_fsdp2_cpu_offload_policy = False

        # Defaults for mixed-precision state. _build_fsdp_module overrides these when it
        # runs; subclasses that bypass _build_fsdp_module (e.g. VeOmniEngine) keep the
        # defaults so forward_step / optimizer_step can still read them safely.
        self._autocast_dtype = torch.bfloat16
        self.scaler = None

        # QAT (Quantization-Aware Training)
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            logger.info(f"QAT enabled: mode={self._qat_config.mode}, group_size={self._qat_config.group_size}")

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile  #  use torch compile by default
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        # This is used to import external_lib into the huggingface systems
        self._build_model_optimizer()

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _init_device_mesh(self):
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.engine_config.fsdp_size

        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_parallel_group = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size
        dp_size = self.get_data_parallel_size()
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp_size, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
            self.ulysses_parallel_group = self.ulysses_device_mesh["sp"].get_group()

        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def _build_module(self):
        from verl.utils.model import get_hf_auto_model_class
        from verl.utils.torch_dtypes import PrecisionType

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            # if it is training, we force torch_dtype to fp32
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_config.model_type == "language_model":
                auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)

                module = auto_class.from_pretrained(
                    pretrained_model_name_or_path=self.model_config.local_path,
                    torch_dtype=torch_dtype,
                    config=self.model_config.hf_config,
                    trust_remote_code=self.model_config.trust_remote_code,
                )
            else:
                from verl.utils.model import load_valuehead_model

                assert self.model_config.model_type == "value_model", (
                    f"Unsupported model type: {self.model_config.model_type}"
                )
                self.model_config.hf_config.num_labels = 1
                self.model_config.hf_config.classifier_dropout = 0.0
                self.model_config.hf_config.hidden_dropout = "0"
                self.model_config.hf_config.summary_dropout_prob = 0.0
                module = load_valuehead_model(
                    local_path=self.model_config.local_path,
                    torch_dtype=torch_dtype,
                    model_config=self.model_config.hf_config,
                    trust_remote_code=self.model_config.trust_remote_code,
                )

            use_liger = self.model_config.use_liger
            # Apply Liger kernel; disable fused_linear_cross_entropy (conflicts with verl's forward patching)
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(
                    model=module,
                    fused_linear_cross_entropy=False,
                    swiglu=True,
                )

            fused_kernel_options = self.model_config.fused_kernel_options
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            use_fused_kernels = self.model_config.use_fused_kernels
            apply_monkey_patch(
                model=module,
                use_remove_padding=self.use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype
            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_lora_module(self, module):
        module.enable_input_require_grads()

        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            from peft import PeftModel

            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")
            # Copy adapter to local if needed
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            module = PeftModel.from_pretrained(module, local_adapter_path, is_trainable=True)
            peft_config = module.peft_config["default"]
            # Ensure task_type is TaskType enum, not string
            if isinstance(peft_config.task_type, str):
                peft_config.task_type = TaskType.CAUSAL_LM
        else:
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "target_parameters": convert_to_regular_types(self.model_config.target_parameters),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module = get_peft_model(module, LoraConfig(**lora_config))

        return module

    def _build_fsdp_module(self, module):
        # TODO(ziheng): need to improve
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        self._autocast_dtype = param_dtype
        # fp16 training requires loss scaling to avoid gradient underflow. Mirror the pattern
        # landed in #4036 for the legacy dp_actor path. bf16 / fp32 do not need a scaler.
        if param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self.model_config.lora_rank > 0,
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh, zero3_enable=self.engine_config.reshard_after_forward)

        # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
        if self.engine_config.strategy == "fsdp":
            # cpu_offload:
            # - actor: None
            # - critic: None
            # - ref: CPUOffload(offload_params=True)

            # We force reference policy to use CPUOffload to save memory.
            # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
            cpu_offload = None
            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False

            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=self.engine_config.use_orig_params,
                cpu_offload=cpu_offload,
            )
        elif self.engine_config.strategy == "fsdp2":
            # - actor: offload_policy
            # - critic: offload_policy
            # - ref: CPUOffloadPolicy(pin_memory=True)
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)
                self._uses_fsdp2_cpu_offload_policy = True

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }
            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

        if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        return module

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        return build_optimizer(module.parameters(), self.optimizer_config)

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        optim_config = self.optimizer_config

        total_steps = optim_config.total_training_steps
        num_warmup_steps = optim_config.lr_warmup_steps
        lr_scheduler_type = optim_config.lr_scheduler_type
        min_lr_ratio = optim_config.min_lr_ratio
        num_cycles = optim_config.num_cycles
        zero_indexed_step = optim_config.zero_indexed_step
        if num_warmup_steps <= 0:
            num_warmup_steps_ratio = optim_config.lr_warmup_steps_ratio
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        if lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
                zero_indexed_step=zero_indexed_step,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
        return lr_scheduler

    def _apply_qat(self, module):
        """Apply QAT transformations to the model before FSDP wrapping."""
        from verl.utils.qat.core import apply_qat, enable_qat_fuse

        module = apply_qat(
            module,
            {
                "enable": self._qat_config.enable,
                "mode": self._qat_config.mode,
                "group_size": self._qat_config.group_size,
                "ignore_patterns": list(self._qat_config.ignore_patterns),
                "activation_observer": self._qat_config.activation_observer,
            },
        )
        enable_qat_fuse(module)

        if self._qat_config.mode == "w4a4":
            self._restore_w4a4_input_scales(module, self.model_config.local_path)

        return module

    def _restore_w4a4_input_scales(self, model, model_path):
        """Restore input_global_scale and input_amax from checkpoint for W4A4 mode."""
        import glob

        from safetensors import safe_open

        safetensor_files = glob.glob(f"{model_path}/model*.safetensors")
        loaded_count = 0

        for sf_path in safetensor_files:
            with safe_open(sf_path, framework="pt") as f:
                for key in f.keys():
                    if "input_global_scale" in key:
                        module_path = key.replace(".input_global_scale", "")
                        amax_key = f"{module_path}.input_amax"

                        module = model
                        for part in module_path.split("."):
                            module = module[int(part)] if part.isdigit() else getattr(module, part)

                        scale_val = f.get_tensor(key)
                        val = scale_val.item() if scale_val.numel() == 1 else scale_val.max().item()
                        module.input_global_scale.fill_(val)

                        amax_val = f.get_tensor(amax_key)
                        amax = amax_val.item() if amax_val.numel() == 1 else amax_val.max().item()
                        module.input_amax.fill_(amax)
                        loaded_count += 1

        logger.info(f"[QAT W4A4] Restored {loaded_count} input_global_scale/input_amax from {model_path}")

    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size

        # Load base model with specified configuration and dtype
        module = self._build_module()
        # Apply LoRA adapters if low-rank adaptation is enabled
        if self._is_lora:
            module = self._build_lora_module(module)

        # Apply QAT before FSDP wrapping (training only)
        if self._qat_enabled and not self.engine_config.forward_only:
            module = self._apply_qat(module)

        # Synchronize all distributed processes before proceeding
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        # Wrap model with FSDP for distributed training (sharding, mixed precision, etc.)
        log_gpu_memory_usage("Before FSDP", logger=None)
        module = self._build_fsdp_module(module)
        log_gpu_memory_usage("After FSDP", logger=None)

        if not self.engine_config.forward_only:
            # Initialize optimizer with model parameters and config settings
            optimizer = self._build_optimizer(module)
            # Create learning rate scheduler with warmup and decay settings
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_mode(self, **kwargs):
        """
        Return a context manager that switches to training mode with FSDP-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Return a context manager that switches to evaluation mode with FSDP-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size

    def get_data_parallel_group(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def get_model_parallel_group(self):
        raise NotImplementedError

    def get_context_parallel_group(self):
        raise NotImplementedError

    def _comm_eff_powersgd_active(self, forward_only: bool) -> bool:
        """True iff the PowerSGD projection hooks should be live for this forward.

        The projector is confined to the actor-train forward/backward
        (``path_tag == "train"``,
        ``forward_only=False``); when ``powersgd.compress_recompute=true``, to
        the old-policy log-prob recompute (``path_tag == "old_logprob"``,
        ``forward_only=True``); and when ``powersgd.compress_reference=true``, to
        the frozen reference-policy log-prob forward (``path_tag == "ref_logprob"``,
        ``forward_only=True``). All of a step's forwards see the same
        anchor-owned ``Q_t``; the basis only advances after the gradient-bearing
        work (the anchor stages ``Q_{t+1}`` during update_actor and it is activated
        afterward). Returns False (strict no-op) unless an enabled state with the
        powersgd codec and a live ``compression_active`` flag is attached.
        """
        from verl.workers.comm_eff.state import OLD_LOGPROB_TAG, REF_LOGPROB_TAG, TRAIN_TAG

        state = getattr(self, "_comm_eff_state", None)
        if state is None or not getattr(state, "enabled", False):
            return False
        if getattr(state, "powersgd", None) is None:
            return False
        if not getattr(state, "compression_active", False):
            return False
        tag = getattr(state, "path_tag", None)
        if tag == TRAIN_TAG:
            compressor = state.powersgd
            if hasattr(compressor, "fast_q_bootstrap_needed") and compressor.fast_q_bootstrap_needed():
                raise RuntimeError(
                    "comm_eff fast-Q bootstrap is still pending at the first compressed train forward. "
                    "The old-logprob recompute must run first (compress_recompute=true; rollout-correction "
                    "bypass mode is incompatible) so Q1 is frozen across the complete old/current PPO pair."
                )
            return not forward_only
        if tag == OLD_LOGPROB_TAG:
            # The worker stamps this path only when recompute compression is on.
            return forward_only
        if tag == REF_LOGPROB_TAG:
            # Reference-KL forward. Compress it through the SAME live anchor-owned
            # Q_t as this step's paired old/current policy forwards so KL(current||
            # ref) is measured on one shared basis. forward_only keeps grad
            # disabled, so the hook returns M_hat but never folds the sketch or
            # advances Q. Gated on compress_reference AND on Q being established:
            # before the fast-Q bootstrap commits there is no calibrated Q, so fall
            # back to a DENSE reference rather than compress against a random basis.
            # Unlike the train path this NEVER raises (it is a measurement path).
            if not forward_only:
                return False
            ps_cfg = getattr(getattr(state, "config", None), "powersgd", None)
            if not bool(getattr(ps_cfg, "compress_reference", False)):
                return False
            compressor = state.powersgd
            if not (hasattr(compressor, "reference_basis_ready") and compressor.reference_basis_ready()):
                self._comm_eff_note_ref_dense_once()
                return False
            return True
        return False

    def _comm_eff_note_ref_dense_once(self) -> None:
        """Emit a one-time note that the reference forward ran dense (Q not ready)."""
        if getattr(self, "_comm_eff_ref_dense_logged", False):
            return
        object.__setattr__(self, "_comm_eff_ref_dense_logged", True)
        print(
            "[comm_eff] ref forward dense (Q not yet bootstrapped); "
            "reference-KL will compress once the anchor-owned Q is established",
            flush=True,
        )

    def _comm_eff_register_powersgd_hooks(self) -> bool:
        """Register the PowerSGD projection hooks on the boundary blocks.

        The boundary activation token axis is what
        Ulysses SP>1 slices across ranks (out of scope) — refuse it loudly. The
        per-forward context (global_step + the generation bump that dedupes the
        basis sketch against grad-ckpt recompute) is set per micro-batch in
        ``prepare_model_inputs``. Returns True iff hooks were registered.
        """
        if getattr(self, "ulysses_sequence_parallel_size", 1) and self.ulysses_sequence_parallel_size > 1:
            raise NotImplementedError(
                "comm_eff powersgd does not support "
                f"ulysses_sequence_parallel_size>1 (got {self.ulysses_sequence_parallel_size}); "
                "set ulysses_sequence_parallel_size=1 for this codec."
            )
        state = self._comm_eff_state
        compressor = state.powersgd
        compressor.register(self.module)
        return compressor.is_registered

    def _comm_eff_maybe_set_powersgd_context(self, micro_batch: TensorDict, input_ids) -> None:
        """Bump the PowerSGD forward generation and stamp global_step.

        No-op unless the projection hooks are live. The only per-micro-batch
        state is the generation counter (which deduplicates gradient-checkpoint
        recompute) and the trainer step.

        rmpad guard: the boundary activation ``M`` the projector compresses is
        the rmpad (nested / no-padding) token axis. If a caller ever ran without
        ``use_remove_padding=True`` the
        activation would be a PADDED ``(B, T, H)`` block and the projector +
        basis sketch ``V`` would silently fold PAD tokens into ``M`` and into the
        codebook — corrupting both the reconstruction metric and the learned
        basis. Refuse it loudly here rather than produce a quietly-wrong codec.
        This guard prevents padded inputs from silently changing the projected
        token population.
        """
        state = getattr(self, "_comm_eff_state", None)
        if state is None:
            return
        compressor = getattr(state, "powersgd", None)
        if compressor is None or not compressor.is_registered:
            return
        if not getattr(input_ids, "is_nested", False):
            raise NotImplementedError(
                "comm_eff powersgd requires rmpad (nested / no-padding) inputs "
                "(use_remove_padding=True); padded forwards would fold PAD tokens "
                "into the projected activation M and the basis sketch V."
            )
        compressor.set_context(global_step=int(getattr(self, "_comm_eff_global_step", 0)))

    def _comm_eff_mask_active(self, forward_only: bool) -> bool:
        """True iff the boundary-codec (prf_mask / sr_quant) hooks should be live.

        The prf_mask masker and the sr_quant quantizer share one lifecycle:
        both are confined to the actor-train forward/backward by default,
        additionally to the old-policy log-prob recompute when
        ``comm_eff.mask.mask_recompute=true``, and additionally to the frozen
        reference-policy forward when ``comm_eff.mask.mask_reference=true``
        (sr_quant reuses the mask eligibility knobs). Returns False (strict
        no-op) unless an enabled state with a built boundary codec (``masker``
        or ``quantizer``), a live ``compression_active`` flag, and an eligible
        ``path_tag`` is attached. The gate mirrors
        ``_comm_eff_powersgd_active``; because the masker, the quantizer and
        the PowerSGD compressor are mutually exclusive (state.build constructs
        exactly one), only one of the ``*_active`` gates can be positive for
        any forward. The anchor pass (``path_tag=None``) never matches an
        eligible tag, so anchors stay uncompressed.
        """
        from verl.workers.comm_eff.state import OLD_LOGPROB_TAG, REF_LOGPROB_TAG, TRAIN_TAG, mask_eligible_tags

        state = getattr(self, "_comm_eff_state", None)
        if state is None or not getattr(state, "enabled", False):
            return False
        if getattr(state, "masker", None) is None and getattr(state, "quantizer", None) is None:
            return False
        if not getattr(state, "compression_active", False):
            return False
        tag = getattr(state, "path_tag", None)
        eligible = mask_eligible_tags(state)
        if tag not in eligible:
            return False
        if tag == TRAIN_TAG:
            # Train forward MUST be the backward-bearing pass.
            return not forward_only
        if tag == OLD_LOGPROB_TAG:
            # old_logprob recompute is forward-only by construction
            # (compute_log_prob -> infer_batch -> forward_only=True).
            return forward_only
        if tag == REF_LOGPROB_TAG:
            # Reference-KL forward. Masked with the SAME within-step
            # (sample_id, position_id) key as the paired policy forwards so
            # KL(current || ref) is measured codec-vs-codec (masked-current vs
            # masked-reference). Eligibility above already gated on
            # mask.mask_reference. The reference forward is forward_only by
            # construction (compute_ref_log_prob -> infer_batch), and grad stays
            # disabled, so this is a pure measurement path.
            return forward_only
        return False

    def _comm_eff_register_mask_hooks(self) -> bool:
        """Register the boundary-codec (mask or quant) hooks on the boundary blocks.

        The per-token PRF context (``global_step`` + token-aligned
        ``sample_ids`` / ``position_ids``) is set per micro-batch in
        ``prepare_model_inputs``, since the stable ids are only known once the
        micro-batch is packed. SP guard mirrors the PowerSGD path: the key is
        aligned to the rmpad token axis, which Ulysses SP>1 slices/pads across
        ranks (out of scope); refuse it loudly. Returns True if hooks were
        registered. The prf_mask masker and the sr_quant quantizer expose the
        identical register/set_context/unregister surface, so one call site
        serves both codecs.
        """
        if getattr(self, "ulysses_sequence_parallel_size", 1) and self.ulysses_sequence_parallel_size > 1:
            raise NotImplementedError(
                "comm_eff per-element masking does not support "
                f"ulysses_sequence_parallel_size>1 (got {self.ulysses_sequence_parallel_size}); "
                "set ulysses_sequence_parallel_size=1 for this codec."
            )
        state = self._comm_eff_state
        masker = state.masker if state.masker is not None else state.quantizer
        masker.register(self.module)
        return masker.is_registered

    def _comm_eff_maybe_set_mask_context(self, micro_batch: TensorDict, input_ids) -> None:
        """Set the per-token PRF context for this micro-batch's masked forward.

        No-op unless mask hooks are live. Builds, in the packed order of
        ``input_ids.values()`` (the activation token axis under SP=1):
        ``sample_ids`` (each row's ``comm_eff_sample_id`` repeated across its
        tokens) and ``position_ids`` (position within each sequence, from the
        rmpad offsets). Keying on these stable ids makes the mask
        packing-invariant across the differently-packed forwards.

        The within-step ``global_step`` folded into the PRF key is read from the
        SHARED comm_eff state, not the per-engine ``_comm_eff_global_step``
        attribute. The worker stamps that attribute only on the actor engine; a
        fused ``actor_rollout_ref`` worker runs the reference forward on a
        DISTINCT engine object that shares the same state, so
        ``state.global_step`` is the only value guaranteed to equal the step the
        paired train / old-logprob forwards used. This keeps the masked
        reference forward (mask_reference) bit-identical to the policy mask so
        KL(current || ref) is a true codec-vs-codec quantity. It is
        byte-identical for the train / old-logprob paths, where the shared
        ``state.global_step`` already equals the actor-engine attribute.
        """
        state = getattr(self, "_comm_eff_state", None)
        if state is None:
            return
        masker = getattr(state, "masker", None)
        if masker is None:
            # The sr_quant quantizer shares the masker's context surface
            # (set_context keyed on the same stable per-token ids).
            masker = getattr(state, "quantizer", None)
        if masker is None or not masker.is_registered:
            return
        if getattr(masker, "_anchor_sketch_mode", False):
            # Anchor-owned-Q harvest (issue #93). The FRLR sketch needs the step
            # and a fresh forward-generation (the grad-checkpoint dedupe key),
            # but NO per-token PRF key: no mask is drawn on the anchor pass, the
            # hook returns the raw activation. So skip the rmpad and
            # comm_eff_sample_id requirements below rather than making the
            # anchor's replayed batch satisfy them.
            _gstep = getattr(state, "global_step", None)
            if _gstep is None or int(_gstep) < 0:
                _gstep = getattr(self, "_comm_eff_global_step", 0)
            masker.set_context(global_step=int(_gstep), sample_ids=None, position_ids=None)
            return
        if not getattr(input_ids, "is_nested", False):
            raise NotImplementedError(
                "comm_eff per-element masking requires rmpad (nested / no-padding) "
                "inputs; padded forwards are out of scope for the per-element mask."
            )
        sample_id_per_row = micro_batch.get("comm_eff_sample_id", None)
        if sample_id_per_row is None:
            raise RuntimeError(
                "comm_eff_sample_id missing from the micro-batch while mask hooks "
                "are live; the worker must stamp a stable per-row id on the batch "
                "before micro-batching (engine_workers.update_actor / compute_log_prob "
                "/ compute_ref_log_prob when mask_reference)."
            )
        device = input_ids.values().device
        offsets = input_ids.offsets().to(device=device)  # (nseq+1,)
        seqlens = offsets.diff()  # (nseq,)
        sample_id_per_row = sample_id_per_row.reshape(-1).to(device=device, dtype=torch.int64)
        # per-token stable sample id: repeat each row's id across its tokens.
        sample_ids = torch.repeat_interleave(sample_id_per_row, seqlens)  # (total_nnz,)
        # per-token position within its sequence: flat_index - sequence_start.
        total = int(offsets[-1].item())
        flat = torch.arange(total, device=device)
        starts = torch.repeat_interleave(offsets[:-1], seqlens)
        position_ids = flat - starts  # (total_nnz,)
        # Prefer the shared-state step (authoritative on every engine sharing the
        # state, including the reference engine); fall back to the per-engine
        # attribute only if the state has no valid step yet.
        gstep = getattr(state, "global_step", None)
        if gstep is None or int(gstep) < 0:
            gstep = getattr(self, "_comm_eff_global_step", 0)
        masker.set_context(
            global_step=int(gstep),
            sample_ids=sample_ids,
            position_ids=position_ids,
        )

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
        # Register PowerSGD only around the paired old/current actor forwards;
        # later inference, reference, validation, and anchor passes stay dense.
        _powersgd_hooks_live = self._comm_eff_powersgd_active(forward_only=forward_only)
        if _powersgd_hooks_live:
            _powersgd_hooks_live = self._comm_eff_register_powersgd_hooks()
        # Register the prf_mask / sr_quant boundary codec around the same eligible
        # forwards. The mask, quantizer and PowerSGD compressors are mutually
        # exclusive, so at most one codec is live per forward.
        _mask_hooks_live = self._comm_eff_mask_active(forward_only=forward_only)
        if _mask_hooks_live:
            _mask_hooks_live = self._comm_eff_register_mask_hooks()
        _ce_state = getattr(self, "_comm_eff_state", None)
        # Reset the per-tick comm-volume accumulators before the real
        # fast-train forward so the powersgd hook accumulates THIS tick's Y/dense
        # element counts cleanly (snapshotted into last_elems_* in engine_workers
        # after backward). No-op unless the powersgd codec is live.
        if not forward_only and _ce_state is not None and getattr(_ce_state, "enabled", False):
            _ps = getattr(_ce_state, "powersgd", None)
            if _ps is not None and hasattr(_ps, "reset_tick_comm_counters"):
                _ps.reset_tick_comm_counters()
        try:
            # Optional one-time fast-network Q calibration.  It runs only on the
            # first compressed old-logprob call.  The discarded prepass
            # returns raw boundary activations while collecting a private V; the
            # complete DP-consensus candidate is verified and atomically activated
            # before this method computes any real old_log_probs.  Consequently
            # the actual old-policy forward and every subsequent current-policy
            # minibatch share exactly Q1, and the anchor remains the sole Q writer
            # after this one-time handoff.
            if _powersgd_hooks_live:
                _bootstrap_state = getattr(self, "_comm_eff_state", None)
                _bootstrap_compressor = getattr(_bootstrap_state, "powersgd", None)
                _bootstrap_tag = getattr(_bootstrap_state, "path_tag", None)
                from verl.workers.comm_eff.state import OLD_LOGPROB_TAG

                if (
                    forward_only
                    and _bootstrap_tag == OLD_LOGPROB_TAG
                    and _bootstrap_compressor is not None
                    and hasattr(_bootstrap_compressor, "fast_q_bootstrap_needed")
                    and _bootstrap_compressor.fast_q_bootstrap_needed()
                ):
                    _bootstrap_compressor.begin_fast_q_bootstrap_observation()
                    try:
                        self._forward_backward_batch_inner(
                            data,
                            loss_function,
                            forward_only=True,
                            run_backward=False,
                            collect_outputs=False,
                        )
                        _bootstrap_compressor.finish_fast_q_bootstrap_observation()
                        if not _bootstrap_compressor.stage_fast_q_bootstrap_basis():
                            raise RuntimeError("comm_eff fast-Q bootstrap produced no candidate")
                        _bootstrap_dev = _bootstrap_compressor.verify_fast_q_bootstrap_basis_across_ranks()
                        if not _bootstrap_compressor.activate_staged_fast_q_bootstrap_basis():
                            raise RuntimeError("comm_eff fast-Q bootstrap candidate did not activate")
                        print(
                            "[comm_eff][fast-q-bootstrap] activated before real old_logprob "
                            f"global_step={getattr(self, '_comm_eff_global_step', 0)} "
                            f"observations={_bootstrap_compressor.fast_q_bootstrap_observations} "
                            f"updates={_bootstrap_compressor.fast_q_bootstrap_updates} "
                            f"activations={_bootstrap_compressor.fast_q_bootstrap_activations} "
                            f"dense_observation_elements="
                            f"{_bootstrap_compressor.fast_q_bootstrap_dense_observation_elements:.0f} "
                            f"sync_elements={_bootstrap_compressor.fast_q_bootstrap_sync_elements:.0f} "
                            f"cross_rank_max_rel_dev={_bootstrap_dev}",
                            flush=True,
                        )
                    except Exception:
                        # No candidate from a partial/malformed prepass may leak
                        # into a retry or the actual old-policy forward.  Live Q0
                        # remains untouched unless the all-boundary commit above
                        # completed successfully.
                        _bootstrap_compressor.abort_fast_q_bootstrap()
                        raise
            return self._forward_backward_batch_inner(data, loss_function, forward_only=forward_only)
        finally:
            if _powersgd_hooks_live:
                self._comm_eff_state.powersgd.unregister()
            if _mask_hooks_live:
                _codec = self._comm_eff_state.masker
                if _codec is None:
                    _codec = self._comm_eff_state.quantizer
                _codec.unregister()

    def _forward_backward_batch_inner(
        self,
        data: TensorDict,
        loss_function: Callable,
        forward_only: bool = False,
        run_backward: bool = True,
        collect_outputs: bool = True,
    ) -> list[TensorDict]:
        # note that the global_batch_size should include data on all the dp
        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        # getattr fallback: some subclasses (e.g. VeOmniEngine) bypass FSDPEngine.__init__
        # and _build_fsdp_module, so self.scaler may not be set.
        scaler = getattr(self, "scaler", None)

        for micro_batch in micro_batches:
            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)

                if not forward_only and run_backward:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            if collect_outputs:
                output_lst.append(meta_info)
            else:
                # Q-only rank1 warmup needs autograd enabled so the forward
                # activation hooks see the real graph, but it deliberately
                # never backpropagates or consumes model outputs. Drop both
                # graph roots before the next microbatch so memory stays bounded
                # to one microbatch instead of accumulating the whole batch.
                del loss, meta_info

        # postprocess and return
        if not collect_outputs:
            return []
        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def _comm_eff_target_names(self, spec_cfg) -> tuple:
        """Substrings selecting which named 2D params receive spectral correction."""
        substrs = getattr(spec_cfg, "target_substr", None)
        if substrs is None:
            return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        return tuple(substrs)

    def _comm_eff_target_scope(self, spec_cfg) -> str:
        """M/correction selector policy; defaults preserve decoder matrices."""
        return str(getattr(spec_cfg, "target_scope", "decoder_matrices"))

    def _dp_all_reduce_anchor_grads(self, anchor_grads: dict) -> dict:
        """All-reduce(MEAN) ``G_anchor`` across the actor DP group.

        The anchor backward runs on a per-rank deep-copy clone with NO FSDP hooks,
        so each rank's ``p.grad`` is the gradient of ITS OWN 1/dp_size data shard —
        NOT the global gradient. Before feeding ``M_anchor``'s EMA we therefore
        all-reduce(MEAN) each per-target grad over ``get_data_parallel_group()`` so
        ``M_anchor`` is the GLOBAL stale gradient, bit-identical on every rank.

        **MEAN is correct (not SUM).** The clean-PG ``agg_loss`` already scales by
        ``dp_size`` (core_algos.py:1173, ``masked_sum/batch_num_tokens * dp_size``),
        which cancels FSDP's mean gradient reduction; the anchor clone has no FSDP
        reduction, so all-reduce-MEAN of the per-rank shard gradients reproduces
        the true full-batch token-mean gradient. all-reduce-SUM would over-count by
        ``dp_size`` and silently inflate ``sign(M)`` magnitude bookkeeping.

        **Collective safety (deadlock guard).** Build the GLOBAL union of target
        names via ``all_gather_object`` over the DP group, walk it in FIXED sorted
        order, and contribute a correctly-shaped ZERO for any target a rank lacks,
        so every rank issues the IDENTICAL collective sequence (mirrors the
        PowerSGD sketch-sync discipline). The clone arch + target_substrs are
        identical across ranks and the DP shards are symmetric, so the union ==
        every rank's local set by construction (the coverage assert enforces
        set-equality at full coverage) and this adds no values on the normal path;
        the union + zero-fill is the genuine collective-safety guard, not merely a
        comment — it makes the
        sequence symmetric even if a rank were pathologically missing a target.

        Reduces on the GRAD's device (GPU) regardless of the EMA storage device
        (cpu) — the EMA feed moves it to the storage device afterward. Logs each
        target's pre/post-reduce norm (cheap mean-vs-sum proxy: a SUM bug shows
        ~dp_size× inflation; a correct MEAN keeps the norm O(1)×).

        Returns the in-place-reduced ``anchor_grads`` dict.
        """
        if not torch.distributed.is_initialized():
            return anchor_grads
        group = self.get_data_parallel_group()
        try:
            dp_world = torch.distributed.get_world_size(group=group)
        except Exception:
            dp_world = 1
        if dp_world <= 1:
            return anchor_grads
        # Cross-rank UNION of target names (collective-safety). All-gather each
        # rank's {name: (shape, dtype)} so every rank walks the IDENTICAL sorted
        # target list and contributes a correctly-shaped ZERO for any target it
        # lacks. Normally the union == this rank's local set (identical by
        # construction), so this is a no-op on values; it only guarantees the
        # collective sequence can never go asymmetric (deadlock guard).
        local_meta = {name: (tuple(g.shape), g.dtype) for name, g in anchor_grads.items()}
        gathered_meta: list = [None] * dp_world
        torch.distributed.all_gather_object(gathered_meta, local_meta, group=group)
        union_meta: dict = {}
        for meta in gathered_meta:
            if not meta:
                continue
            for name, shape_dtype in meta.items():
                union_meta.setdefault(name, shape_dtype)
        # Reference device for zero-fill: a local grad's device, else the current GPU.
        ref_device = next(iter(anchor_grads.values())).device if anchor_grads else get_device_id()
        # FIXED sorted order over the UNION so every rank issues the same sequence.
        names = sorted(union_meta.keys())
        norm_pre = {}
        norm_post = {}
        for name in names:
            if name in anchor_grads:
                g = anchor_grads[name]
                gd = g.to(torch.float32)
                out_dtype = g.dtype
            else:
                # Target absent on this rank — contribute a correctly-shaped ZERO so
                # the all_reduce stays symmetric (cannot happen on the normal path).
                shape, out_dtype = union_meta[name]
                gd = torch.zeros(shape, dtype=torch.float32, device=ref_device)
            norm_pre[name] = float(torch.linalg.norm(gd).item())
            # all-reduce(SUM) then divide by dp_world == MEAN. (ReduceOp.AVG is not
            # available on every backend; SUM+/dp_world is portable + exact.)
            torch.distributed.all_reduce(gd, op=torch.distributed.ReduceOp.SUM, group=group)
            gd /= float(dp_world)
            norm_post[name] = float(torch.linalg.norm(gd).item())
            anchor_grads[name] = gd.to(out_dtype)
        if names:
            # Mean pre/post ratio across targets — a MEAN reduce keeps it ~O(1);
            # a SUM bug would show ~dp_world. Greppable scale falsifier.
            import statistics as _stats

            ratios = [norm_post[n] / norm_pre[n] for n in names if norm_pre[n] > 0]
            ratio_mean = _stats.fmean(ratios) if ratios else 0.0
            print(
                f"[comm_eff][dp-reduce] anchor G_anchor all-reduced(MEAN) over DP "
                f"dp_world={dp_world} targets={len(names)} "
                f"||G||_post/||G||_pre_mean={ratio_mean:.4f} "
                f"(MEAN ⇒ ~O(1) per-rank-shard-dependent; a SUM bug ⇒ ~{dp_world}x)",
                flush=True,
            )
        return anchor_grads

    def _broadcast_anchor_M(self, spectral, anchor_grads: dict, *, src: int = 0) -> dict:
        """``dist.broadcast`` the anchor EMA ``M`` to every DP rank.

        After the DP-reduce + EMA feed, ``M_anchor`` is already bit-identical
        across ranks (the all-reduce made ``G_anchor`` identical, and the EMA is
        deterministic). This broadcast is the positive-receipt mechanism: it
        proves every fast/DP rank holds the anchor's ``M``
        (a wrong process group / dropped collective surfaces as recv != src). The
        merger reads ``sign(M)`` on the fast path, so a stale/cold M on any rank
        would silently break the correction there.

        Walks the FIXED ``sorted(anchor_grads.keys())`` order (== the EMA's covered
        targets) so every rank issues the identical collective sequence. Operates
        on ``spectral._anchor`` (keyed by canonical name, the EMA store). Brings a
        CPU-offloaded M onto the grad device for the broadcast, then restores it to
        the EMA storage device. Returns a per-target receipt dict.
        """
        if not torch.distributed.is_initialized():
            return {}
        group = self.get_data_parallel_group()
        try:
            dp_world = torch.distributed.get_world_size(group=group)
        except Exception:
            dp_world = 1
        if dp_world <= 1:
            return {}
        from verl.workers.comm_eff.spectral_filter import _canon

        receipts: dict = {}
        for name in sorted(anchor_grads.keys()):
            cname = _canon(name)
            m = spectral._anchor.get(cname)
            if m is None:
                continue
            # Broadcast on a contiguous fp32 copy on the grad's compute device.
            dev = anchor_grads[name].device
            m_dev = m.detach().to(device=dev, dtype=torch.float32).contiguous()
            pre = float(torch.linalg.norm(m_dev).item())
            torch.distributed.broadcast(m_dev, src=src, group=group)
            post = float(torch.linalg.norm(m_dev).item())
            # Store back on the EMA storage device (re-pin if CPU-offloaded).
            store_dev = spectral._ema_storage_device(dev)
            stored = m_dev.to(store_dev)
            if store_dev.type == "cpu" and dev.type == "cuda":
                stored = stored.pin_memory()
            spectral._anchor[cname] = stored
            receipts[name] = {"pre_norm": pre, "post_norm": post, "changed": bool(abs(post - pre) > 0.0)}
        return receipts

    def _verify_anchor_M_dp_identical(self, spectral, anchor_grads: dict, *, step: int, atol: float = 1e-6) -> None:
        """Assert ``M_anchor`` is bit-identical across DP.

        All-gathers a per-target fp64 checksum of the EMA ``M`` over the DP group
        and asserts the max cross-rank relative deviation is ``<= atol``. After the
        all-reduce(MEAN) of ``G_anchor`` the EMA is deterministic, so ``M`` MUST be
        identical on every rank; a non-zero deviation proves the DP-reduce did not
        run or used the wrong group. Symmetric collective
        (FIXED sorted target order, same-length vector on every rank). No-op when
        single-rank / distributed unavailable.
        """
        if not torch.distributed.is_initialized():
            return
        group = self.get_data_parallel_group()
        try:
            world = torch.distributed.get_world_size(group=group)
        except Exception:
            world = 1
        if world <= 1:
            return
        from verl.workers.comm_eff.spectral_filter import _canon

        names = sorted(anchor_grads.keys())
        dev = get_device_id()
        ramp_sums = []
        for name in names:
            m = spectral._anchor.get(_canon(name))
            if m is None:
                ramp_sums.append(0.0)
                continue
            md = m.detach().to(torch.float64).reshape(-1)
            # A deterministic ramp-weighted sum (sign/permutation/value sensitive).
            ramp = torch.arange(1, md.numel() + 1, dtype=torch.float64, device=md.device)
            ramp_sums.append(float((md * ramp).sum().item()))
        vec = torch.tensor(ramp_sums, dtype=torch.float64, device=dev)
        gathered = [torch.zeros_like(vec) for _ in range(world)]
        torch.distributed.all_gather(gathered, vec, group=group)
        ref = gathered[0]
        max_abs = 0.0
        for g in gathered[1:]:
            max_abs = max(max_abs, float((g - ref).abs().max().item()))
        scale = float(ref.abs().max().item()) or 1.0
        max_rel = max_abs / scale
        print(
            f"[comm_eff][M-dp-identical] step={step} targets={len(names)} "
            f"cross_rank_max_rel_dev={max_rel:.3e} (0 ⇒ M is the GLOBAL DP-reduced gradient)",
            flush=True,
        )
        assert max_rel <= atol, (
            f"comm_eff anchor M DIVERGED across DP ranks (max_rel_dev={max_rel:.3e} > atol={atol:.1e}); "
            "the all-reduce(MEAN) of G_anchor did not make M identical "
            "(wrong process group / reduce never ran)."
        )

    def _build_anchor_pg_loss(self, fast_path_loss_function, anchor_pg_loss):
        """Bind the clean policy-gradient loss for the anchor pass.

        The anchor must NOT reuse the fast-path PPO ratio/clip loss: its
        ``old_log_probs`` come from the compressed fast circuit, so their ratio
        against the dense anchor forward is not the declared anchor objective.
        Instead we run ``anchor_pg_loss`` (ratio ≡ 1) bound to the SAME actor
        ``config`` the fast path carries, so ``agg_loss`` normalizes identically
        and ``M_anchor`` is the clean true gradient at the same scale.

        ``fast_path_loss_function`` is ``functools.partial(ppo_loss,
        config=actor_config)``; we read ``config`` off ``.keywords`` and rebind.
        This touches ONLY the anchor pass — the fast path keeps its real loss.
        """
        from functools import partial

        from verl.workers.utils.losses import ppo_loss

        fast_callable = getattr(fast_path_loss_function, "func", None)
        if fast_callable is not ppo_loss:
            callable_name = getattr(fast_callable, "__qualname__", repr(fast_callable))
            raise RuntimeError(
                "comm_eff anchor objective parity currently supports the plain "
                "verl.workers.utils.losses.ppo_loss fast objective only; "
                f"got {callable_name}. Distillation or another wrapped/additive "
                "objective needs an explicit ratio-one anchor mapping before it can run."
            )

        config = None
        kw = getattr(fast_path_loss_function, "keywords", None)
        if kw is not None:
            config = kw.get("config")
        if config is None:
            raise RuntimeError(
                "comm_eff anchor: could not read 'config' off the fast-path "
                "loss_function (expected functools.partial(ppo_loss, config=...)). "
                "The clean-gradient anchor loss needs the actor config for "
                "agg_loss normalization."
            )
        return partial(anchor_pg_loss, config=config)

    def _maybe_comm_eff_anchor_refresh(self, data, loss_function) -> None:
        """FSDP anchor-circuit refresh: dense K-stale GRPO actor loss
        fwd/bwd -> RAW G_anchor -> spectral anchor EMA, NO optimizer step.

        Runs at the top of ``BaseEngine.train_batch`` before the compressed fast
        path). The six non-negotiable invariants this enforces:

        1. **Clean dense policy-gradient loss.** The anchor
           uses ``anchor_pg_loss`` (ratio ≡ 1, no clip, no ``old_log_probs``)
           over THIS rollout-expanded batch — its gradient is the CLEAN true
           policy gradient ``-(A·∇logπ)`` at the K-stale weights. It is
           bound to the SAME actor config the fast path carries (so ``agg_loss``
           normalizes identically). It is NOT a supervised next-token loss, and
           it is NOT the fast-path PPO loss: reusing the fast path's ``ppo_loss``
           here would feed current-path ``old_log_probs`` against the anchor's
           stale forward, making the importance ratio ≠ 1 and letting the PPO
           clip corrupt ``G_anchor``. The fast-path loss
           is left UNTOUCHED — this is anchor-pass-only.
        2. **No rollout / no reward recompute.** It only re-forwards ``data``
           (which already carries ``responses``/``old_log_probs``/``advantages``);
           rollout generation + reward scoring live upstream in the trainer and
           are never invoked here. The ``anchor_rollouts_generated`` /
           ``anchor_rewards_recomputed`` counters stay 0 structurally.
        3. **No optimizer step.** The snapshot is detached clones OFF the
           optimizer's param group; this method never calls ``optimizer_step``.
        4. **enabled=false ⇒ no-op.** Gated on ``state.enabled`` AND
           ``anchor.enabled``; the cadence predicate gates per-step firing.
        5. **Uncompressed.** The pass clears ``state.compression_active`` and
           its path tag, so PowerSGD hooks cannot enter the anchor clone pass.
        6. **Uncorrected.** ``G_anchor`` is read RAW and fed to
           ``SpectralFilter.update_anchor`` (the EMA) BEFORE any fast-path
           corrector; ``anchor_grad_corrected`` stays 0.
        """
        state = getattr(self, "_comm_eff_state", None)
        if state is None or not getattr(state, "enabled", False):
            return
        anchor_cfg = getattr(state.config, "anchor", None)
        spectral = getattr(state, "spectral", None)
        if anchor_cfg is None or not bool(getattr(anchor_cfg, "enabled", False)):
            return
        # Anchor-owned Q is independent of the merger: Q <- orth(V) from the
        # anchor's stale forward and broadcast can run even when no spectral
        # filter exists. Only the M-EMA feed/broadcast needs `spectral`, so the
        # spectral-dependent steps below each guard on `spectral is not None`.
        _anchor_owns_q_pre = bool(getattr(anchor_cfg, "owns_q", False)) and getattr(state, "powersgd", None) is not None
        if spectral is None and not _anchor_owns_q_pre:
            return

        from verl.workers.comm_eff.anchor import (
            AnchorStalenessQueue,
            anchor_pg_loss,
            anchor_should_fire,
            assert_anchor_module_isolated,
            build_anchor_module,
            clone_batch_for_replay,
            extract_target_grads,
            feed_anchor_grads_into_ema,
            maybe_build_replay_ring,
            select_anchor_batch_for_scope,
            snapshot_canary,
            snapshot_named_params,
            verify_canary_on_module,
            verify_canary_on_snapshot,
        )
        from verl.workers.comm_eff.lookahead import (
            Rank1ProjectionError,
            Rank1RelexProjector,
            Rank1SnapshotHistory,
            lookahead_history_mode,
            lookahead_max_snapshots,
            lookahead_min_points,
            rank1_relex_enabled,
            resolve_lookahead_rollout_source,
            validate_rank1_broadcast_receipts,
        )

        # Canonicalize FSDP wrap-infix so the (possibly fallback non-infixed)
        # anchor clone matches the live module's per-layer-wrapped snapshot keys.
        from verl.workers.comm_eff.spectral_filter import _canon

        cadence = int(getattr(anchor_cfg, "cadence", 20))
        delay_K = int(getattr(anchor_cfg, "delay_K", 20))
        _rank1_mode = rank1_relex_enabled(anchor_cfg)

        # Advance the trainer-step counter the cadence is keyed on (1-based).
        state.anchor_step += 1
        step = state.anchor_step

        # Paired replay mode. Snapshot storage is always on CPU: it moves the
        # delay_K+1 full bf16 snapshots off HBM in BOTH modes (numerics-neutral:
        # the clone load casts back via .to(p.device, p.dtype), a byte-preserving
        # round trip).
        replay_mode = bool(getattr(anchor_cfg, "replay_paired_batch", False))
        anchor_batch_scope = str(getattr(anchor_cfg, "batch_scope", "ppo_minibatch"))
        _current_anchor_batch = select_anchor_batch_for_scope(
            anchor_batch_scope,
            current_batch=data,
            rollout_batch=getattr(self, "_comm_eff_rollout_batch", None),
        )

        def _retain_current_anchor_batch():
            # Give every long-lived replay/base entry its own CPU deep clone.
            # rollout_batch is already a private pre-split CPU clone, but a
            # second clone here prevents the transient update context, rank1
            # base, and replay ring from aliasing tensor storage across their
            # different lifetimes. Retention happens only on replayable ticks.
            return clone_batch_for_replay(_current_anchor_batch, device=torch.device("cpu"))

        _snap_dev = torch.device("cpu")
        # Diagnostics only controls the canary log. The canary assertion remains
        # load-bearing regardless of this flag.
        _spec_cfg_for_diag = getattr(state.config, "spectral", None)
        comm_eff_spectral_diagnostics = (
            bool(getattr(_spec_cfg_for_diag, "diagnostics", True)) if _spec_cfg_for_diag is not None else True
        )

        # Lazily build the staleness queue on the state (survives across steps).
        # CommEffState is a plain class with a __dict__, so a direct setattr is
        # correct; it is the single object shared with the worker. In replay
        # mode the per-tick queue is never built (the per-global-step
        # generator ring replaces it — maybe_build_replay_ring below).
        queue = getattr(state, "_anchor_queue", None)
        if queue is None and not replay_mode:
            queue = AnchorStalenessQueue(delay_K=delay_K)
            state._anchor_queue = queue
        ring = maybe_build_replay_ring(state, anchor_cfg, delay_K, cadence=cadence)

        spec_cfg = getattr(state.config, "spectral", None)
        target_substrs = self._comm_eff_target_names(spec_cfg)
        target_scope = self._comm_eff_target_scope(spec_cfg)
        # Default to full coverage (-1); max_targets caps both anchor extraction
        # and merger writeback.
        max_targets = int(getattr(spec_cfg, "max_targets", -1)) if spec_cfg is not None else -1

        # Anchor-owns-Q: when on, the anchor's stale forward also
        # harvests slow-net activations into the PowerSGD sketch V, computes
        # Q ← orth(V), and broadcasts Q (and M) to every DP rank.
        anchor_owns_q = bool(getattr(anchor_cfg, "owns_q", False))
        powersgd = getattr(state, "powersgd", None)
        do_anchor_q = anchor_owns_q and powersgd is not None
        # Anchor-owns-Q for the FRLR codec (issue #93). Same governance, other
        # compressor: the mask codec's FRLR basis is harvested and refreshed on
        # the anchor's clean stale-weight forward and NEVER on the fast path, so
        # Q moves only when the anchor fires (like PowerSGD) and its side channel
        # rides the slow circuit. The two are mutually exclusive by construction
        # (state.build() creates either masker or powersgd, never both).
        _mask_codec = getattr(state, "masker", None)
        do_anchor_frlr_q = (
            anchor_owns_q and _mask_codec is not None and bool(getattr(_mask_codec, "anchor_owns_q", False))
        )

        use_orig = bool(getattr(self.engine_config, "use_orig_params", False))
        module_is_fsdp1 = isinstance(self.module, FSDP)
        module_is_fsdp2 = isinstance(self.module, FSDPModule)

        def _inner_named_params():
            return getattr(self.module, "_fsdp_wrapped_module", self.module).named_parameters()

        # --- snapshot THIS step's (full) weights into the staleness ring -------
        # Snapshot OFF the optimizer's param group (plain detached clones) so no
        # accidental optimizer step can ever touch them. For FSDP1
        # we summon the full params to clone the logical matrices; FSDP2 keeps
        # original names (DTensor) — we clone the local shard's full_tensor.
        def _summon_ctx():
            if module_is_fsdp1 and not module_is_fsdp2:
                if not use_orig:
                    raise RuntimeError(
                        "comm_eff anchor circuit under FSDP1 requires "
                        "actor_rollout_ref.actor.fsdp_config.use_orig_params=true "
                        "(FSDP.summon_full_params(with_grads=True) is unsupported with "
                        "use_orig_params=false). Set it in the launcher."
                    )
                return FSDP.summon_full_params(self.module, with_grads=True, writeback=True)
            return nullcontext()

        if replay_mode:
            # : ONE generator snapshot per GLOBAL STEP, taken at its first
            # train_batch tick (before any optimizer tick of this global step) —
            # exactly the weights vLLM held when it generated this step's
            # rollouts. The gs boundary is detected as "the ring has no snapshot
            # for this gs yet" (engine_workers stamps _comm_eff_global_step
            # before the mini-batch loop, so every tick of a global step sees
            # the same gs). The push-time canary (fp32-on-CPU norm+sum of 2
            # target matrices) is verified BITWISE off the clone at fire time.
            _gs_now = int(getattr(self, "_comm_eff_global_step", 0))
            if not ring.has_snapshot(_gs_now):
                with _summon_ctx():
                    gen_snapshot = snapshot_named_params(
                        _inner_named_params(), target_substrs=None, device=_snap_dev, detach=True
                    )
                _push_canary = snapshot_canary(gen_snapshot, target_substrs=target_substrs)
                ring.push_snapshot(_gs_now, gen_snapshot, canary=_push_canary, tick=step)
                if _rank1_mode:
                    rank1_history = getattr(state, "_rank1_history", None)
                    if rank1_history is None:
                        rank1_min_snapshots = lookahead_min_points(anchor_cfg)
                        rank1_history = Rank1SnapshotHistory(
                            window_snapshots=int(getattr(anchor_cfg, "lookahead_window_snapshots", 4)),
                            min_snapshots=rank1_min_snapshots,
                            history_mode=lookahead_history_mode(anchor_cfg),
                            max_snapshots=lookahead_max_snapshots(anchor_cfg),
                        )
                        state._rank1_history = rank1_history
                        # RELEX weight projection covers every unique floating
                        # named parameter. The spectral selector remains scoped
                        # independently to decoder-gradient correction.
                        state._rank1_projector = Rank1RelexProjector(
                            anchor_cfg,
                            min_snapshots=rank1_min_snapshots,
                        )
                    if rank1_history.seed_base(step, gen_snapshot):
                        # Reuse the exact generator snapshot object already
                        # allocated for replay: no second model copy. Retain the
                        # matching first batch only so the first Q-only fire can
                        # use the tick-1 base pair rather than replay warmup's
                        # too-fresh current fallback.
                        state._rank1_base_batch = _retain_current_anchor_batch()
                        state._rank1_base_canary = _push_canary
                        state.rank1_history_checkpoints = 1
                        state.rank1_history_deltas = 0
                        print(
                            f"[comm_eff][rank1_relex] seeded local base tick={step} gs={_gs_now} "
                            f"window={rank1_history.window_snapshots} (reused generator snapshot)",
                            flush=True,
                        )
                print(
                    f"[comm_eff][stale-replay] snapshot_push gs={_gs_now} tick={step} "
                    f"device=cpu snapshots_retained={len(ring.snapshot_steps)} "
                    f"canary_targets={sorted(_push_canary.keys())}",
                    flush=True,
                )
            # Deep-clone THIS tick's batch into the ring (CPU — ~0 HBM; the
            # batch TensorDict is already CPU-resident at train_batch time).
            # Deep clone at store time is REQUIRED: _forward_backward_batch_inner
            # mutates the live batch in place right after this hook returns.
            # Fire-aware retention: only ticks a future fire can request
            # (tick ≡ −delay_K mod cadence) are cloned + stored at all.
            if ring.tick_retained(step):
                ring.push_batch(step, _retain_current_anchor_batch(), _gs_now)
        else:
            with _summon_ctx():
                cur_snapshot = snapshot_named_params(
                    _inner_named_params(), target_substrs=None, device=_snap_dev, detach=True
                )
            queue.push(step, cur_snapshot)
            # Queue mode with CPU-resident snapshots: record the push-time
            # canary so the fire-time load is value-verified (bounded dict,
            # same retention as the queue).
            if _snap_dev is not None:
                from collections import OrderedDict as _ODict

                _canaries = getattr(state, "_anchor_canary_by_tick", None)
                if _canaries is None:
                    _canaries = _ODict()
                    state._anchor_canary_by_tick = _canaries
                _canaries[step] = snapshot_canary(cur_snapshot, target_substrs=target_substrs)
                while len(_canaries) > delay_K + 1:
                    _canaries.popitem(last=False)

        # Periodic full-fidelity step (issue #93): suppress the anchor on a
        # bypassed step. The anchor exists to correct a COMPRESSED gradient, and
        # on a dense step there is no compression error to correct, so firing it
        # would inject a correction against an error that is not there. Read the
        # global step fresh rather than relying on _gs_now, which is bound inside
        # the snapshot branch above and is not guaranteed to be in scope here.
        # NOTE this gates the anchor REFRESH only. The signed-EMA `M` accumulated
        # by earlier fires is applied by the spectral circuit on its own cadence
        # and is deliberately left alone: zeroing it would change the optimizer
        # state itself rather than just skipping one correction.
        _mask_codec_dense = getattr(state, "masker", None)
        if _mask_codec_dense is not None and _mask_codec_dense.is_dense_step(
            int(getattr(self, "_comm_eff_global_step", 0))
        ):
            print(
                f"[comm_eff][dense-step] anchor refresh SUPPRESSED at "
                f"gs={int(getattr(self, '_comm_eff_global_step', 0))} tick={step} "
                f"(dense_every={_mask_codec_dense.dense_every})",
                flush=True,
            )
            return

        if not anchor_should_fire(step, cadence, True):
            return

        # --- fetch the stale (weights, batch) to forward from -------------------
        _fire_canary = None  # push-time canary to verify off the clone (replay / cpu-snapshot modes)
        _replay_batch = None
        if replay_mode:
            # Replay the PAIRED (batch[t-delay_K], generator-snapshot).
            _rep = ring.get_replay(step, delay_K)
            if _rep is None:  # pragma: no cover - ring always has >=1 after push
                return
            _used_step, _replay_batch, _batch_gs, stale, _fire_canary, _snap_tick, _warm_fb = _rep
            # By construction the snapshot is fetched under the BATCH's gs key
            # (push_batch asserts the snapshot exists for that gs), so
            # batch_gs == snapshot_gs always; the load-bearing runtime checks are
            # the exact data staleness and the weights-never-fresher-than-K bound.
            _data_delay = int(step) - int(_used_step)
            _realized_weight_delay = int(step) - int(_snap_tick)
            print(
                f"[comm_eff][stale-replay] step={step} delay_K={delay_K} used_tick={_used_step} "
                f"batch_gs={_batch_gs} snapshot_gs={_batch_gs} snapshot_tick={_snap_tick} "
                f"data_delay={_data_delay} realized_step_delay={_realized_weight_delay} "
                f"warmup_fallback={_warm_fb} "
                f"ring_batches={len(ring)} ring_snapshots={len(ring.snapshot_steps)}",
                flush=True,
            )
            # 1-based ticks: the t-delay_K batch only exists for step > delay_K
            # because tick numbering starts at one.
            if int(step) > int(delay_K):
                assert (not _warm_fb) and _used_step == int(step) - int(delay_K), (
                    f"comm_eff stale-replay: post-warmup step={step} must replay the exact "
                    f"t-delay_K batch (expected tick {int(step) - int(delay_K)}, got {_used_step}, "
                    f"warmup_fallback={_warm_fb}). push_batch runs every tick + the ring retains "
                    f"delay_K+1 batches, so the paired batch MUST be available — a mismatch means "
                    f"eviction broke or the push cadence changed."
                )
                # The generator snapshot sits at the FIRST tick of the batch's
                # global step, so the realized weight staleness is >= delay_K
                # (K or K+1 on the 2-tick-per-step substrate) — the anchor's
                # weights are never FRESHER than the contract.
                assert _realized_weight_delay >= int(delay_K), (
                    f"comm_eff stale-replay: realized weight staleness {_realized_weight_delay} < "
                    f"delay_K={delay_K} at step={step} (snapshot_tick={_snap_tick}) — the generator "
                    f"snapshot is too fresh; the gs-boundary detection mis-keyed the snapshot."
                )
            state.anchor_replay_fires += 1
        else:
            stale = queue.get_stale(step, delay_K)
            if stale is None:  # pragma: no cover - queue always has >=1 after push
                return

            # Log the actual stale snapshot step used and realized delay.
            # get_stale() falls back to the OLDEST retained snapshot while warming up
            # (step < delay_K — the t-delay_K snapshot has not been taken yet), so the
            # first refresh can be near-current; that is finite/expected. Post-warmup
            # (step >= delay_K) push-runs-every-step + queue maxlen=delay_K+1 guarantee
            # t-delay_K is still retained, so the realized delay MUST equal delay_K —
            # hard-assert it so a silently-too-fresh anchor cannot pass unnoticed. (This
            # mirrors get_stale's own fallback logic via the public .steps property.)
            _req_step = int(step) - int(delay_K)
            _avail_steps = queue.steps
            _used_step = _req_step if _req_step in _avail_steps else (_avail_steps[0] if _avail_steps else int(step))
            _realized_delay = int(step) - _used_step
            print(
                f"[comm_eff][stale] step={step} delay_K={delay_K} requested_step={_req_step} "
                f"used_step={_used_step} realized_delay={_realized_delay} "
                f"warmup_fallback={_used_step != _req_step}",
                flush=True,
            )
            # 1-based steps (no step 0): the t-delay_K snapshot only becomes available
            # at step == delay_K + 1 (at step == delay_K the request is step 0, which
            # never existed). Therefore the post-warmup guarantee holds for
            # step > delay_K, not step >= delay_K.
            if int(step) > int(delay_K):
                assert _used_step == _req_step, (
                    f"comm_eff anchor staleness: post-warmup step={step} requested the t-delay_K "
                    f"snapshot step={_req_step} (delay_K={delay_K}) but used step={_used_step} "
                    f"(realized_delay={_realized_delay}). push runs every step + the queue retains "
                    f"delay_K+1 snapshots, so t-delay_K MUST be available once step>=delay_K; a "
                    f"mismatch means the snapshot was evicted or the push cadence changed."
                )
            # CPU-resident snapshots: pull the push-time canary recorded
            # for the snapshot tick actually used (verified off the clone below).
            if _snap_dev is not None:
                _fire_canary = getattr(state, "_anchor_canary_by_tick", {}).get(_used_step)

        # --- RELEX rank-1 weight projection -----------------------------------
        # The history admits only the first local generator base plus later exact delayed transfers;
        # warmup fallbacks and current target snapshots never enter its window.
        _la_active = False  # theta_hat actually loaded THIS fire
        _la_info = None
        _rank1_fire = False
        _rank1_q_only = False
        _rank1_q_only_batch = None
        _rank1_warm_correct = False
        _rank1_warm_correct_batch = None
        load_weights = stale  # what the clone receives (raw stale by default)
        _src_tick = int(_snap_tick) if replay_mode else int(_used_step)
        _la_target_tick = int(step)
        if replay_mode:
            _gs_gen_tick = ring.snapshot_tick(_gs_now)
            if _gs_gen_tick >= 0:
                _la_target_tick = _gs_gen_tick

        if _rank1_mode:
            rank1_history = getattr(state, "_rank1_history", None)
            rank1_projector = getattr(state, "_rank1_projector", None)
            if rank1_history is None or rank1_projector is None:
                raise RuntimeError("rank1_relex history/projector was not seeded from the first generator snapshot")

            admitted = False
            if not bool(_warm_fb):
                admitted = rank1_history.admit_exact(_src_tick, stale)
                if not admitted:
                    print(
                        f"[comm_eff][rank1_relex] step={step} excluded duplicate/out-of-order "
                        f"exact transfer tick={_src_tick} history={rank1_history.ticks}",
                        flush=True,
                    )
            state.rank1_history_checkpoints = rank1_history.total_retained()
            state.rank1_history_deltas = max(0, rank1_history.total_retained() - 1)
            state.rank1_window_span = (
                rank1_history.ticks[-1] - rank1_history.ticks[0] if rank1_history.total_retained() > 1 else 0
            )

            _rank1_snaps, _rank1_ticks = rank1_history.sources()
            if _rank1_snaps is not None and not admitted:
                # A duplicate/out-of-order exact transfer is not a new
                # checkpoint. Never extrapolate the old window toward a newer
                # target and pretend a fresh rank1 fire occurred. Once M is
                # ready, merely returning would let the correction hook apply
                # the previous M on this skipped tick. Abort before clone/Q/M
                # mutation instead; the normal aligned schedule always admits.
                raise Rank1ProjectionError(
                    f"rank1_relex ready fire has no new exact checkpoint: "
                    f"transfer_tick={_src_tick} history={rank1_history.ticks} step={step}"
                )
                return
            if _rank1_snaps is not None:
                # Finish every tensor projection before loading the clone or
                # touching M/Q. Rank1ProjectionError therefore aborts the run
                # without contaminating the anchor EMA.
                theta_hat, _la_info = rank1_projector.project(
                    _rank1_snaps,
                    ticks=_rank1_ticks,
                    target_tick=_la_target_tick,
                )
                load_weights = theta_hat
                _la_active = True
                _rank1_fire = True
                state.rank1_fires += 1
                state.rank1_history_checkpoints = int(_la_info["checkpoint_count"])
                state.rank1_history_deltas = int(_la_info["delta_count"])
                state.rank1_window_span = int(_la_info["window_span"])
                state.rank1_prediction_horizon = int(_la_info["prediction_horizon"])
                state.rank1_evr_mean = float(_la_info["evr_mean"])
                state.rank1_r2_mean = float(_la_info["r2_mean"])
                state.rank1_zero_motion_tensors = int(_la_info["zero_motion_tensors"])
                print(
                    f"[comm_eff][rank1_relex] step={step} target_tick={_la_target_tick} "
                    f"history_ticks={list(_la_info['history_ticks'])} "
                    f"history_mode={_la_info['history_mode']} "
                    f"checkpoints={_la_info['checkpoint_count']} deltas={_la_info['delta_count']} "
                    f"fit={_la_info['fit_kind']} "
                    f"window_span={_la_info['window_span']} horizon={_la_info['prediction_horizon']} "
                    f"parameter_tensors={_la_info['targets_projected']} "
                    f"nonfloating_passthrough={_la_info['nonfloating_tensors_passthrough']} "
                    f"zero_motion={_la_info['zero_motion_tensors']} "
                    f"evr_mean={_la_info['evr_mean']:.6f} evr_min={_la_info['evr_min']:.6f} "
                    f"r2_mean={_la_info['r2_mean']:.6f} r2_min={_la_info['r2_min']:.6f}",
                    flush=True,
                )
            elif str(getattr(anchor_cfg, "warmup_mode", "")) == "q_only":
                latest_snapshot, latest_tick = rank1_history.latest()
                if bool(_warm_fb):
                    load_weights = latest_snapshot
                    _rank1_q_only_batch = getattr(state, "_rank1_base_batch", None)
                    _fire_canary = getattr(state, "_rank1_base_canary", None)
                    _rank1_q_source_tick = latest_tick
                else:
                    # Even when this exact generator snapshot is a duplicate
                    # (multiple optimizer batches can share one rollout
                    # generator), its replayed trajectories remain correctly
                    # paired with `stale`. Exclude it from history but still run
                    # the required observation-only Q refresh.
                    load_weights = stale
                    _rank1_q_only_batch = _replay_batch
                    _rank1_q_source_tick = _src_tick
                if _rank1_q_only_batch is None:
                    raise RuntimeError("rank1_relex q_only warmup has no paired trajectory batch")
                _rank1_q_only = True
                state.rank1_q_only_fires += 1
                print(
                    f"[comm_eff][rank1_relex] step={step} Q_ONLY history_ticks={rank1_history.ticks} "
                    f"source_tick={_rank1_q_source_tick} checkpoints={rank1_history.total_retained()}/"
                    f"{rank1_history.window_snapshots} M=disabled correction=disabled",
                    flush=True,
                )
            elif str(getattr(anchor_cfg, "warmup_mode", "")) == "stale_correct":
                # Replay warmup normally falls back to the current fire batch
                # because t-K does not exist yet. For rank1, fire 1 instead uses
                # the explicitly retained tick-1 generator checkpoint and its
                # paired tick-1 rollout batch. That gives M a real initial policy
                # point and lets the fast correction consume it on this same
                # optimizer tick. Later pre-readiness fires use their exact
                # delayed checkpoint/trajectory pair.
                latest_snapshot, latest_tick = rank1_history.latest()
                if bool(_warm_fb):
                    load_weights = latest_snapshot
                    _rank1_warm_correct_batch = getattr(state, "_rank1_base_batch", None)
                    _fire_canary = getattr(state, "_rank1_base_canary", None)
                    _rank1_warm_source_tick = latest_tick
                else:
                    load_weights = stale
                    _rank1_warm_correct_batch = _replay_batch
                    _rank1_warm_source_tick = _src_tick
                if _rank1_warm_correct_batch is None:
                    raise RuntimeError("rank1_relex stale_correct warmup has no exact paired trajectory batch")
                _rank1_warm_correct = True
                state.rank1_warmup_correction_fires += 1
                print(
                    f"[comm_eff][rank1_relex] step={step} WARM_CORRECT "
                    f"history_ticks={rank1_history.ticks} source_tick={_rank1_warm_source_tick} "
                    f"checkpoints={rank1_history.total_retained()}/{rank1_history.window_snapshots} "
                    f"M=dense correction=same_tick",
                    flush=True,
                )
            else:  # validated config permits only q_only or stale_correct
                raise RuntimeError(f"unsupported rank1_relex warmup_mode={anchor_cfg.warmup_mode!r}")

        # Keep the anchor clone outside the fast compressed-forward lifecycle.
        prev_compression_active = getattr(state, "compression_active", False)
        prev_path_tag = getattr(state, "path_tag", None)
        state.compression_active = False
        if hasattr(state, "set_path_tag"):
            state.set_path_tag(None)
        opt_steps_before = int(getattr(state, "anchor_optimizer_steps", 0))

        # Shallow-copy the batch so the anchor fwd/bwd never mutates the
        # TensorDict the compressed fast path reuses immediately after. Replay mode
        # consumes the ring's stored t-delay_K batch instead of the current tick's
        # batch; the inner loop's in-place non-tensor stamps must never mutate the
        # stored clone, which a warmup fallback may replay twice.
        #
        # Rollout-source resolution (anchor.lookahead_rollout_source): when the
        # look-ahead projector actually fired (_la_active) and the resolved
        # source is "current_step", the anchor consumes THIS tick's batch
        # instead of the stale t-delay_K replay batch. Those trajectories are
        # time-aligned with theta_hat[t]'s forecast target, but the live fast
        # actor—not theta_hat[t]—generated them; this reduces temporal batch
        # staleness without claiming an exact weight-policy pair. "auto"
        # resolves to current_step whenever the projector is on;
        # "stale_paired" (or projector OFF, or a warmup-fallback fire) keeps
        # the exact checkpoint/replay pairing.
        # Non-replay mode consumes the current tick's batch. The ring's batch
        # pushes are retained for warmup and holdover fires.
        _rollout_source = resolve_lookahead_rollout_source(anchor_cfg)
        if _rank1_q_only:
            anchor_data = _rank1_q_only_batch.copy() if hasattr(_rank1_q_only_batch, "copy") else _rank1_q_only_batch
            _batch_choice = "rank1_delayed_pair"
        elif _rank1_warm_correct:
            anchor_data = (
                _rank1_warm_correct_batch.copy()
                if hasattr(_rank1_warm_correct_batch, "copy")
                else _rank1_warm_correct_batch
            )
            _batch_choice = "rank1_exact_warm_pair"
        elif replay_mode:
            if _la_active and _rollout_source == "current_step":
                anchor_data = (
                    _current_anchor_batch.copy() if hasattr(_current_anchor_batch, "copy") else _current_anchor_batch
                )
                _batch_choice = "current_step"
            else:
                anchor_data = _replay_batch.copy() if hasattr(_replay_batch, "copy") else _replay_batch
                _batch_choice = "stale_paired"
        else:
            anchor_data = (
                _current_anchor_batch.copy() if hasattr(_current_anchor_batch, "copy") else _current_anchor_batch
            )
            _batch_choice = "current_step"
        # comm_eff: anchor_data is a copy of a replay-ring batch and does not
        # carry the dynamic micro-batch packing keys that engine_workers injects
        # onto the live update batch. Without them prepare_micro_batches() asserts
        # on a missing max_token_len_per_gpu when use_dynamic_bsz is True (the
        # actor default). Propagate the SAME packing budget the fast update path
        # uses; this only affects micro-batch grouping, never the anchor
        # loss/gradient (token-mean normalization is recomputed independently).
        for _mb_key, _mb_default in (
            ("use_dynamic_bsz", self.engine_config.use_dynamic_bsz),
            ("max_token_len_per_gpu", self.engine_config.max_token_len_per_gpu),
            ("micro_batch_size_per_gpu", self.engine_config.micro_batch_size_per_gpu),
        ):
            if _mb_key not in anchor_data.keys():
                tu.assign_non_tensor(
                    anchor_data,
                    **{_mb_key: tu.get_non_tensor_data(data=data, key=_mb_key, default=_mb_default)},
                )
        if _la_active:
            print(
                f"[comm_eff][lookahead] step={step} fire pairing: weights=projected(t={_la_target_tick}) "
                f"batch={_batch_choice}"
                + (f"(tick={_used_step})" if _batch_choice == "stale_paired" else f"(tick={step})"),
                flush=True,
            )

        # Validate the requested scope before the clone forward can mutate Q or
        # M. Full-scope clones carry corrected global_batch_size metadata; the
        # inner loop below independently recomputes the global valid-token count
        # used by token-mean normalization.
        _anchor_sequences_global = int(
            tu.get(
                anchor_data,
                key="global_batch_size",
                default=int(anchor_data.shape[0]) * self.get_data_parallel_size(),
            )
        )
        _update_sequences_global = int(
            tu.get(
                anchor_data,
                key="comm_eff_update_sequences_global",
                default=_anchor_sequences_global,
            )
        )
        if _anchor_sequences_global <= 0 or _update_sequences_global <= 0:
            raise RuntimeError(
                "comm_eff anchor batch telemetry received a non-positive batch size: "
                f"anchor={_anchor_sequences_global} update={_update_sequences_global}"
            )
        if _anchor_sequences_global > _update_sequences_global:
            raise RuntimeError(
                "comm_eff anchor batch exceeds its source rollout batch: "
                f"anchor={_anchor_sequences_global} update={_update_sequences_global}"
            )
        if anchor_batch_scope == "rollout_batch" and _anchor_sequences_global != _update_sequences_global:
            raise RuntimeError(
                "comm_eff rollout_batch anchor did not consume the complete update: "
                f"anchor={_anchor_sequences_global} update={_update_sequences_global}"
            )
        _fast_loss_keywords = getattr(loss_function, "keywords", None) or {}
        _fast_actor_config = _fast_loss_keywords.get("config")
        _rollout_n = int(getattr(_fast_actor_config, "rollout_n", 1))
        if _rollout_n < 1:
            raise RuntimeError(f"comm_eff anchor received invalid actor rollout_n={_rollout_n}")
        if _anchor_sequences_global % _rollout_n or _update_sequences_global % _rollout_n:
            raise RuntimeError(
                "comm_eff anchor batch does not contain an integral number of rollout_n response blocks: "
                f"anchor_sequences={_anchor_sequences_global} "
                f"update_sequences={_update_sequences_global} rollout_n={_rollout_n}"
            )
        # These are prompt-equivalent row counts (responses / rollout_n), not a
        # claim that a shuffled PPO mini-batch preserved every prompt group.
        # rollout_batch contains the complete update and therefore does preserve
        # all groups regardless of PPO iterator ordering.
        _anchor_prompt_equivalents_global = _anchor_sequences_global // _rollout_n
        _update_prompt_equivalents_global = _update_sequences_global // _rollout_n

        def _record_anchor_batch_telemetry(signal_role: str):
            state.anchor_batch_sequences_global = _anchor_sequences_global
            state.anchor_update_sequences_global = _update_sequences_global
            state.anchor_batch_prompt_equivalents_global = _anchor_prompt_equivalents_global
            state.anchor_update_prompt_equivalents_global = _update_prompt_equivalents_global
            state.anchor_rollout_n = _rollout_n
            state.anchor_batch_fraction = _anchor_sequences_global / _update_sequences_global
            state.anchor_batch_scope_rollout = int(anchor_batch_scope == "rollout_batch")
            print(
                f"[comm_eff][anchor-batch] scope={anchor_batch_scope} "
                f"sequences_global={_anchor_sequences_global} "
                f"update_sequences_global={_update_sequences_global} "
                f"prompt_equivalents_global={_anchor_prompt_equivalents_global} "
                f"update_prompt_equivalents_global={_update_prompt_equivalents_global} "
                f"rollout_n={_rollout_n} "
                f"fraction={state.anchor_batch_fraction:.6f} signal={signal_role}",
                flush=True,
            )

        anchor_grads = {}
        # The anchor's loss.backward() MUST NOT
        # trigger the live FSDP1 module's `_post_backward_hook` (which would
        # call `_check_grad_to_accumulate(flat_param._saved_grad_shard.shape)`
        # outside the fast-path window where `_saved_grad_shard is None` →
        # `AttributeError: 'NoneType' object has no attribute 'shape'`).
        #
        # Mechanism: deep-copy the underlying nn.Module (after summoning full
        # FSDP1 params so the copy receives full unsharded weights), load
        # the K-stale snapshot into the clone, then run fwd/bwd on the CLONE.
        # The clone has no FSDP _handles, no FlatParameters, no post-backward
        # hooks → the autograd-hook chain is broken by construction.
        #
        # `assert_anchor_module_isolated` is a cheap runtime guard: any future
        # refactor that lets the clone alias live optimizer/FSDP params will
        # fire this assertion before we touch the GPU.
        live_module_swap = None
        try:
            with _summon_ctx():
                inner = getattr(self.module, "_fsdp_wrapped_module", self.module)
                # Cache the anchor clone across refreshes to avoid repeated
                # full-model allocation. The K-stale snapshot is loaded into the
                # cached clone below.
                cached_anchor = getattr(self, "_anchor_module_cache", None)
                if cached_anchor is None:
                    anchor_module = build_anchor_module(inner)
                    # Cache it for subsequent refreshes.
                    self._anchor_module_cache = anchor_module
                else:
                    anchor_module = cached_anchor

            # Belt-and-braces: the clone's params share NO id() with either
            # the live optimizer's param_groups OR the live FSDP module. Cheap
            # Runtime guard against future aliasing drift.
            assert_anchor_module_isolated(anchor_module, optimizer=self.optimizer, fsdp_module=inner)

            # Move the clone to the live module's device + dtype so its
            # forward/backward runs on the same accelerator.
            try:
                live_p = next(inner.parameters())
                anchor_module.to(device=live_p.device, dtype=live_p.dtype)
            except StopIteration:
                pass

            # Load the K-stale snapshot weights into the clone (NOT into the
            # live module — the live optimizer's params remain untouched).
            # With rank-1 RELEX active, `load_weights` is projected theta_hat for
            # every unique named parameter tensor. Otherwise it is the raw stale
            # snapshot.
            # The snapshot is keyed by the live module's
            # (FSDP per-layer-wrapped) names — those carry the
            # `._fsdp_wrapped_module.` infix — while the clone (when the deepcopy
            # path fell back to a plain config-rebuild) has NON-infixed names. A
            # raw `n in stale` lookup then never matches → the clone keeps RANDOM
            # init weights → G_anchor is garbage. Match by canonical (infix-
            # stripped) key so the clone receives the REAL delay_K-stale weights.
            stale_canon = {_canon(k): v for k, v in load_weights.items()}
            with torch.no_grad():
                loaded = 0
                for n, p in anchor_module.named_parameters():
                    s = stale_canon.get(_canon(n))
                    if s is not None and s.shape == p.shape:
                        p.copy_(s.to(p.device, p.dtype))
                        loaded += 1
            total = sum(1 for _ in anchor_module.named_parameters())
            print(
                f"[comm_eff][anchor-load] loaded {loaded}/{total} "
                f"{'PROJECTED theta_hat' if _la_active else 'stale'} params into clone (canon-matched)",
                flush=True,
            )
            # Fail-closed: a partial load means the canonical key-match failed and
            # the clone kept random init for the unmatched params, so G_anchor and
            # M are garbage. The stale snapshot covers all params, so every clone
            # param must canon-match a shape-equal snapshot entry.
            assert loaded == total, (
                f"comm_eff anchor clone load INCOMPLETE: loaded {loaded}/{total} stale "
                f"params (canon-matched). A partial load leaves the clone on RANDOM init "
                f"for the unmatched params ⇒ G_anchor/M are invalid. The "
                f"snapshot covers ALL params (target_substrs=None), so this MUST be full; a "
                f"mismatch means the _canon key normalization regressed."
            )

            # Value-level staleness canary: the clone must now hold exactly the
            # weights recorded at push time. Both record
            # and verify reduce in fp32 ON CPU (bf16->cpu->device is a
            # byte-preserving round trip), so the match is bitwise. Scalar-only
            # and always-on in replay / cpu-snapshot modes. A mismatch means the loaded clone is
            # NOT the recorded snapshot — hard fail.
            if _fire_canary:
                # LOAD-BEARING: the verify + assert below are the bitwise
                # staleness guard and run UNCONDITIONALLY. Only the stdout echo
                # is diagnostic and is gated by spectral.diagnostics.
                # Look-ahead fires verify the SOURCE snapshot dict instead of
                # the clone: the clone holds the PROJECTED theta_hat, but the
                # staleness guarantee lives in the recorded source weights,
                # which must still round-trip bitwise.
                if _la_active:
                    _can_ok, _can_got = verify_canary_on_snapshot(stale, _fire_canary)
                else:
                    _can_ok, _can_got = verify_canary_on_module(anchor_module, _fire_canary, canon=_canon)
                if comm_eff_spectral_diagnostics:
                    print(
                        f"[comm_eff][anchor-canary] step={step} match={_can_ok} "
                        f"verified_on={'source-snapshot' if _la_active else 'clone'} "
                        + " ".join(
                            f"{n}: push(norm={_fire_canary[n][0]!r},sum={_fire_canary[n][1]!r}) "
                            f"got(norm={_can_got[n][0]!r},sum={_can_got[n][1]!r})"
                            for n in sorted(_fire_canary.keys())
                        ),
                        flush=True,
                    )
                assert _can_ok, (
                    f"comm_eff anchor-canary MISMATCH at step={step} "
                    f"(verified on {'the SOURCE snapshot (look-ahead fire)' if _la_active else 'the clone'}): "
                    f"the weights differ from the values recorded at snapshot-push time "
                    f"(push={_fire_canary} got={_can_got}). The bf16 snapshot round trip must be "
                    f"byte-preserving — a mismatch means storage corruption, a lossy device cast, "
                    f"or a mis-keyed snapshot."
                )

            # Swap `self.module` to point at the clone for the duration of
            # _forward_backward_batch_inner — that method calls self.module(...)
            # inside forward_step. After the anchor backward, restore.
            live_module_swap = self.module
            self.module = anchor_module

            # Zero any stray grads on the clone before backward (it just got
            # built — there should be none, but be defensive).
            for p in anchor_module.parameters():
                if p.grad is not None:
                    p.grad = None

            # Anchor-owns-Q: register the PowerSGD projection hooks
            # on the clone so the anchor's dense stale-weight forward folds its
            # slow-net boundary activations into the SAME compressor's sketch V
            # (V += Aᵀ(AQ)). The forward hook's sketch gate is routed by
            # _anchor_sketch_mode (set True here) — it accumulates regardless of
            # path_tag (None on the anchor pass) and is deduped per forward-
            # generation against grad-ckpt recompute. The fast-path sketch
            # accumulation stays gated OFF. We unregister + clear the mode in the
            # finally so the live fast path is untouched.
            if do_anchor_q:
                powersgd.set_anchor_sketch_mode(True)
                powersgd.register(self.module)  # self.module is the clone now
            # Same handoff for the FRLR basis. The mask hooks are registered and
            # unregistered inside forward_backward_batch, and the anchor refresh
            # runs at the TOP of train_batch (see engine/base.py), so the live
            # codec is NOT registered here and register() will not no-op on its
            # idempotence guard. In sketch mode the hook returns the RAW
            # activation, so the anchor forward stays dense and its gradient
            # stays clean.
            if do_anchor_frlr_q:
                _mask_codec.set_anchor_sketch_mode(True)
                _mask_codec.register(self.module)  # self.module is the clone now

            # Dense forward/backward on the clone. No FSDP hooks fire because the
            # clone is a plain module. forward_only=False populates .grad on its
            # Parameters.
            #
            # Clean anchor gradient. The anchor uses a plain
            # policy-gradient loss (ratio ≡ 1, no clip, no old_log_probs) instead
            # of the fast-path PPO loss. Reusing the batch's old_log_probs against
            # this stale-weight forward makes
            # the GRPO importance ratio ≠ 1, so PPO clipping would distort
            # G_anchor. anchor_pg_loss instead evaluates the ratio-one advantage
            # gradient at the stale
            # weights, and the configured reference-policy KL term is retained.
            # The fast-path loss_function (real ratio/clip) is UNTOUCHED — this
            # swap is anchor-pass-only. We bind the SAME actor config the fast
            # path uses (read off the partial) so aggregation and KL settings
            # remain identical.
            anchor_loss_function = self._build_anchor_pg_loss(loss_function, anchor_pg_loss)
            self._forward_backward_batch_inner(
                anchor_data,
                anchor_loss_function,
                forward_only=False,
                run_backward=not _rank1_q_only,
                collect_outputs=not _rank1_q_only,
            )

            if _rank1_q_only:
                dirty_grads = [name for name, p in anchor_module.named_parameters() if p.grad is not None]
                assert not dirty_grads, (
                    "rank1_relex q_only forward populated clone gradients despite run_backward=false: "
                    f"{dirty_grads[:8]}"
                )

            # Read G_anchor RAW per target (NO correct_matrix) off
            # the clone. full_grad_of is the identity — the clone is a plain
            # nn.Module so its p.grad is already a full logical tensor.
            def _full_grad_of(grad):
                return grad, {"grad_container_type": type(grad).__name__, "is_dtensor": str(isinstance(grad, DTensor))}

            if not _rank1_q_only:
                anchor_grads = extract_target_grads(
                    anchor_module.named_parameters(),
                    target_substrs=target_substrs,
                    max_targets=max_targets,
                    full_grad_of=_full_grad_of,
                    target_scope=target_scope,
                )
        finally:
            # Anchor-owns-Q: tear down the anchor's PowerSGD hooks on the clone and
            # clear the sketch-harvest mode so the live fast path is untouched.
            # The sketch V (just harvested) PERSISTS on the compressor — consumed
            # by anchor_update_basis below. Q/M broadcasts also happen below.
            if do_anchor_q:
                try:
                    powersgd.unregister()
                finally:
                    powersgd.set_anchor_sketch_mode(False)
            # Same teardown for the FRLR codec. Its sketch V also persists past
            # this block and is consumed by anchor_update_basis below.
            if do_anchor_frlr_q:
                try:
                    _mask_codec.unregister()
                finally:
                    _mask_codec.set_anchor_sketch_mode(False)
            # Restore self.module to the live FSDP-wrapped actor.
            if live_module_swap is not None:
                self.module = live_module_swap
            # Keep the cached clone alive (reused next refresh)
            # but zero its grads + clear the reference held by the local name so
            # the next refresh re-loads the K-stale snapshot into clean params.
            # Empty PyTorch's CUDA cache so transient allocations from the
            # per-param copy + fwd/bwd are released back to CUDA before vLLM's
            # sleep_replicas runs (its freed_bytes>=0 assertion otherwise trips).
            try:
                for _p in anchor_module.parameters():
                    if _p.grad is not None:
                        _p.grad = None
                # Park the cached clone on HOST between fires. The cache lookup
                # above happens INSIDE `summon_full_params(with_grads=True)`, so
                # a clone left resident on device sits alongside the unsharded
                # summon and the live static state for the whole run — a full
                # extra copy of the model (~30 GiB at 8B/bf16) held permanently
                # for a circuit that fires once per `cadence` optimizer ticks.
                # Numerics are UNCHANGED: the next fire moves it back with
                # `anchor_module.to(device=live_p.device, dtype=live_p.dtype)`
                # and then overwrites EVERY parameter from the K-stale snapshot
                # (guarded by the `loaded == total` assert), and a same-dtype
                # device<->host round trip is byte-preserving. The cost is one
                # H2D copy per fire, paid outside the summon.
                anchor_module.to("cpu")
            except UnboundLocalError:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            # Restore the prior compression/path state regardless of outcome.
            state.compression_active = prev_compression_active
            if hasattr(state, "set_path_tag"):
                state.set_path_tag(prev_path_tag)
            # The live optimizer's param.grads were NEVER touched by the
            # anchor pass (we ran fwd/bwd on the clone), so the compressed fast
            # path that follows starts from whatever grads were there at
            # entry.

        # The anchor took no optimizer step.
        assert int(getattr(state, "anchor_optimizer_steps", 0)) == opt_steps_before, (
            "comm_eff anchor pass took an optimizer step (anchor_optimizer_steps "
            "must stay 0; snapshot is OFF the optimizer's param group)."
        )

        if _rank1_q_only:
            # Observation-only warmup: the autograd-enabled forward harvested V
            # for Q, but no backward/gradient/M work is allowed below this
            # barrier. Stage Q_{t+1} and broadcast that candidate without
            # changing the live Q_t used by this update_actor's already-
            # recomputed old_log_probs or any of its train minibatches. The
            # worker activates it only after every minibatch finishes.
            assert getattr(powersgd, "_sketch", None), (
                "rank1_relex q_only forward harvested an empty PowerSGD activation sketch"
            )
            q_updated = powersgd.anchor_update_basis(staged=True)
            q_receipts = powersgd.broadcast_basis(src=0, staged=True)
            m_receipts = None
            assert q_updated, "rank1_relex q_only anchor_update_basis() did not refresh Q"
            _dp_multi = False
            if torch.distributed.is_initialized():
                try:
                    _dp_multi = torch.distributed.get_world_size(group=self.get_data_parallel_group()) > 1
                except Exception:
                    _dp_multi = False
            validate_rank1_broadcast_receipts(
                q_only=True,
                dp_multi=_dp_multi,
                q_receipts=q_receipts,
                m_receipts=m_receipts,
                spectral_enabled=spectral is not None,
            )
            qdev = powersgd.verify_basis_agreement_across_ranks(staged=True)
            object.__setattr__(state, "_powersgd_q_agreement_checked", True)
            object.__setattr__(state, "_powersgd_q_agreement_dev", qdev)
            changed_q = sum(1 for receipt in (q_receipts or {}).values() if receipt.get("changed"))
            print(
                f"[comm_eff][rank1_relex][q_only] step={step} Q staged={q_updated} "
                f"broadcast_boundaries={len(q_receipts or {})} changed={changed_q} "
                f"cross_rank_max_rel_dev={qdev if qdev is not None else 'n/a'} "
                f"M_receipts=0 clone_grads=0 anchor_backwards={state.anchor_backwards}",
                flush=True,
            )
            _record_anchor_batch_telemetry("Q")
            return

        # Coverage set-equality: the anchor M must cover every
        # tensor the merger corrects (set-equal, NOT 4 / NOT boundary-only). Build
        # the expected merger set from the SAME scope selector the merger
        # uses, over the live module's named_parameters (architecture == the
        # clone), and assert set(anchor_grads canon) == set(expected canon) when
        # uncapped. A mismatch emits the count plus the
        # symmetric difference so it is greppable. (Only meaningful when uncapped:
        # max_targets<0; a diagnostic cap deliberately narrows both.)
        from verl.workers.comm_eff.spectral_filter import _canon as _canon_cov
        from verl.workers.comm_eff.spectral_filter import is_spectral_target

        try:
            with _summon_ctx():
                _inner_cov = getattr(self.module, "_fsdp_wrapped_module", self.module)
                expected = {
                    _canon_cov(n)
                    for n, p in _inner_cov.named_parameters()
                    if is_spectral_target(
                        n,
                        p,
                        target_substrs=target_substrs,
                        target_scope=target_scope,
                    )
                    and p.requires_grad
                }
        except Exception as _cov_exc:  # pragma: no cover - defensive
            expected = set()
            print(f"[comm_eff][coverage] WARN could not enumerate expected set: {_cov_exc!r}", flush=True)
        got = {_canon_cov(k) for k in anchor_grads.keys()}
        if expected:
            missing = expected - got
            extra = got - expected
            print(
                f"[comm_eff][coverage] anchor_targets={len(got)} merger_expected={len(expected)} "
                f"set_equal={got == expected} missing={sorted(missing)[:6]}{'...' if len(missing) > 6 else ''} "
                f"extra={sorted(extra)[:6]}{'...' if len(extra) > 6 else ''}",
                flush=True,
            )
            if max_targets < 0:
                assert got == expected, (
                    f"comm_eff anchor coverage MISMATCH: anchor covers {len(got)} targets but the "
                    f"merger corrects {len(expected)}; missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}. "
                    "set(anchor M) MUST == set(merger targets) at full coverage (max_targets=-1)."
                )

        # All-reduce(MEAN) G_anchor across the DP group so
        # M_anchor is the GLOBAL stale gradient (bit-identical across ranks, at the
        # correct mean scale), BEFORE the EMA. The anchor clone had no FSDP
        # reduction, so without this M is each rank's local-shard gradient.
        anchor_grads = self._dp_all_reduce_anchor_grads(anchor_grads)

        # Feed RAW (now DP-reduced) grads into the EMA (update_anchor, NEVER
        # correct_matrix). Skip the M-EMA feed when there is
        # no spectral filter (plain PowerSGD with anchor-owns-Q but no merger);
        # the anchor still runs to update Q below. With spectral on,
        # the feed runs exactly as before.
        deltas = {}
        if spectral is not None:
            deltas = feed_anchor_grads_into_ema(anchor_grads, spectral, state=state)
        # Anchor-sourced optimizer-moment EMAs (anchor.opt_reset): fold the SAME
        # DP-reduced RAW G_anchor tensors that feed M into fp32 CPU m/v states.
        # The overwrite itself runs at the end of the optimizer tick, in
        # _maybe_comm_eff_opt_reset (called from optimizer_step).
        _opt_reset_cfg = getattr(anchor_cfg, "opt_reset", None)
        if _opt_reset_cfg is not None and bool(getattr(_opt_reset_cfg, "enabled", False)):
            from verl.workers.comm_eff.opt_reset import AnchorOptMoments

            _opt_moments = getattr(state, "_opt_reset_moments", None)
            if _opt_moments is None:
                _opt_moments = AnchorOptMoments(
                    beta1=float(getattr(_opt_reset_cfg, "beta1", 0.8)),
                    beta2=float(getattr(_opt_reset_cfg, "beta2", 0.95)),
                )
                state._opt_reset_moments = _opt_moments
            _opt_moments.update(anchor_grads)
        state.anchor_backwards += 1
        if do_anchor_q or do_anchor_frlr_q:
            _signal_role = "Q+M" if spectral is not None else "Q"
        else:
            _signal_role = "M"
        _record_anchor_batch_telemetry(_signal_role)

        # Anchor-owned FRLR basis. The clean anchor forward has folded slow-net
        # activations into the FRLR sketch V; refresh Q ← orth(V) here, once per
        # anchor fire. The fast path is gated off as a Q writer, so this is the
        # sole refresh and Q stays bitwise frozen across every step in between.
        if do_anchor_frlr_q:
            # Fail-closed must-fire invariant, same reasoning as the PowerSGD
            # branch below: an EMPTY sketch means the mask hooks never fired on
            # the clone (find_decoder_layers returned None / register no-op'd), and
            # Q would then silently never move for the whole run while the arm
            # still reported as anchor-owned.
            assert getattr(_mask_codec, "_frlr_sketch", None), (
                "comm_eff anchor-owns-Q (FRLR): the anchor clone forward harvested an EMPTY "
                "sketch V — the mask hooks did not fire on the clone (find_decoder_layers/"
                "register no-op?). Q would never refresh. Refusing to continue."
            )
            _frlr_dp_group = None
            try:
                _frlr_dp_group = self.get_data_parallel_group()
            except Exception:
                _frlr_dp_group = None
            # STAGED, not live. The anchor fires here, at the top of train_batch,
            # AFTER this step's old_log_probs were recomputed. Publishing Q now
            # would make the old-logprob and train forwards of the same step see
            # different bases, so the PPO ratio would deviate from 1 for a reason
            # that is not a policy change and PPO would clip it (measured at
            # pg_clipfrac 0.19-0.37 on exactly the anchor steps, against
            # identically 0 for the fast-Q arms). engine_workers publishes the
            # candidate after all PPO minibatches, as it already does for PowerSGD.
            _frlr_q_updated = _mask_codec.anchor_update_basis(staged=True, dp_group=_frlr_dp_group)
            assert _frlr_q_updated, (
                "comm_eff anchor-owns-Q (FRLR): anchor_update_basis() did NOT refresh Q "
                "(orth(V) produced nothing). Q must refresh every anchor cadence."
            )
            print(
                "[comm_eff][frlr-anchor-q] refreshed global_step="
                f"{getattr(state, 'global_step', -1)} anchor_step={step} "
                f"boundaries={len(_mask_codec.boundary_indices)} "
                f"refreshes={_mask_codec.frlr_q_refreshes}",
                flush=True,
            )

        # Anchor-owned Q. Now that the slow-net activations are
        # harvested into V (during the clean anchor forward above), compute
        # Q_{t+1} ← orth(V) on the ANCHOR (DP-synced), stage/broadcast that
        # candidate, and broadcast freshly EMA'd M to every DP rank with a
        # positive receipt. Live Q_t remains frozen through every minibatch in
        # this update_actor; engine_workers activates the candidate afterward.
        # The fast net's local Q-update is gated OFF, so the anchor is the sole
        # candidate writer. All ranks reach this in lockstep.
        if do_anchor_q:
            # Fail-closed must-fire invariant: the anchor clone's clean forward MUST have
            # harvested slow-net activations into the sketch V (it persists past the
            # finally above and is consumed here). An EMPTY sketch means the PowerSGD
            # projection hooks never fired on the clone (find_decoder_layers returned
            # None / register() no-op'd) — and under sync_basis anchor_update_basis
            # would then silently ZERO-fill Q instead of failing. Assert the hooks
            # fired rather than trusting register()'s log-only warning path.
            assert getattr(powersgd, "_sketch", None), (
                "comm_eff anchor-owns-Q: the anchor clone forward harvested an EMPTY sketch V "
                "— PowerSGD projection hooks did not fire on the clone (find_decoder_layers/"
                "register no-op?). Q would be silently zero-filled under sync_basis. Refusing "
                "to continue."
            )
            q_updated = powersgd.anchor_update_basis(staged=True)
            q_receipts = powersgd.broadcast_basis(src=0, staged=True)
            # The M broadcast applies only when a merger reads sign(M). When the
            # merger is disabled there is no M, so only Q is broadcast. The Q
            # update and broadcast above are unconditional because the anchor owns
            # Q regardless of the merger.
            m_receipts = self._broadcast_anchor_M(spectral, anchor_grads, src=0) if spectral is not None else None
            # Fail-closed must-fire invariant: the anchor is the sole Q/M writer, so each of
            # these MUST do real work every refresh. q_updated False = orth(V) produced
            # no Q; empty receipts on a genuinely multi-rank DP group = the broadcast
            # never propagated, leaving fast/DP ranks on a cold/stale Q,M the merger then
            # reads. The [comm_eff][bcast] prints below are gated on non-empty receipts and
            # would otherwise vanish SILENTLY — hard-assert instead. (Receipts are a
            # legitimate no-op when dp_world<=1, so the receipt asserts are gated on a
            # genuinely multi-rank DP group; q_updated holds even single-rank.)
            assert q_updated, (
                "comm_eff anchor-owns-Q: anchor_update_basis() did NOT update Q (orth(V) "
                "produced nothing). Q must refresh every anchor cadence."
            )
            _dp_multi = False
            if torch.distributed.is_initialized():
                try:
                    _dp_multi = torch.distributed.get_world_size(group=self.get_data_parallel_group()) > 1
                except Exception:
                    _dp_multi = False
            if _rank1_fire or _rank1_warm_correct:
                validate_rank1_broadcast_receipts(
                    q_only=False,
                    dp_multi=_dp_multi,
                    q_receipts=q_receipts,
                    m_receipts=m_receipts,
                    spectral_enabled=spectral is not None,
                )
            if _dp_multi:
                assert q_receipts, (
                    "comm_eff anchor-owns-Q: broadcast_basis() returned NO receipts on a "
                    "multi-rank DP group — the anchor Q broadcast did not fire; every fast/DP "
                    "rank would keep a stale/cold Q."
                )
                assert m_receipts or spectral is None, (
                    "comm_eff anchor-owns-Q: _broadcast_anchor_M() returned NO receipts on a "
                    "multi-rank DP group — the anchor M broadcast did not fire; sign(M) the "
                    "merger reads could be stale/cold on other ranks. "
                    "This check is skipped when the merger is disabled because no M exists."
                )
            # Cross-rank consensus guard (must not raise): the anchor-owned Q must
            # be identical on every DP rank + both boundary sides.
            try:
                qdev = powersgd.verify_basis_agreement_across_ranks(staged=True)
                object.__setattr__(state, "_powersgd_q_agreement_checked", True)
                object.__setattr__(state, "_powersgd_q_agreement_dev", qdev)
            except RuntimeError:
                raise
            if q_receipts:
                changed_q = sum(1 for r in q_receipts.values() if r.get("changed"))
                print(
                    f"[comm_eff][bcast] step={step} Q staged={q_updated} broadcast boundaries={len(q_receipts)} "
                    f"changed={changed_q} cross_rank_max_rel_dev={qdev if qdev is not None else 'n/a'} "
                    f"anchor_q_updates={getattr(state, 'anchor_q_updates', 0)} "
                    f"anchor_q_broadcasts={getattr(state, 'anchor_q_broadcasts', 0)} "
                    # Fast-net basis-update counter must stay 0 in anchor-owns-Q
                    # mode because the fast maybe_update_basis path is gated off.
                    f"powersgd_basis_updates={getattr(state, 'powersgd_basis_updates', 0)}",
                    flush=True,
                )
            if m_receipts:
                changed_m = sum(1 for r in m_receipts.values() if r.get("changed"))
                print(
                    f"[comm_eff][bcast] step={step} M broadcast targets={len(m_receipts)} changed={changed_m} "
                    f"(sign(M) is what the merger reads; receipt proves every DP rank holds the anchor M)",
                    flush=True,
                )

        # Prove M is the global gradient: bit-identical
        # across DP ranks. All-gather a per-target M checksum over the DP group and
        # assert the max cross-rank deviation is ~0 (the all-reduce(MEAN) of
        # G_anchor made M identical on every rank). A non-zero deviation means the
        # DP-reduce did not happen or used the wrong group.
        # Only run this when a merger consumes M.
        if spectral is not None:
            self._verify_anchor_M_dp_identical(spectral, anchor_grads, step=step)

        # EMA-evolution log line. Static merger settings are logged once at build.
        # `deltas` is the MERGER's per-target EMA delta dict — populated ONLY when
        # `spectral is not None` (the merger maintains M); see the `if spectral is
        # not None: deltas = feed_anchor_grads_into_ema(...)` feed above. On a
        # disabled-merger path has no M, so `deltas` is always empty. The
        # anchor's own success is logged elsewhere, so gate this merger-EMA log
        # block on `spectral is not None`. With a real merger present, an empty
        # `deltas` is a genuine coverage bug and still surfaces the warning.
        if spectral is not None:
            if deltas:
                mean_delta = sum(deltas.values()) / len(deltas)
                max_delta = max(deltas.values())
                print(
                    f"[comm_eff] anchor refresh step={step} fired backward "
                    f"(cadence={cadence} delay_K={delay_K}) targets={len(deltas)} "
                    f"||dM_anchor||_mean={mean_delta:.6e} ||dM_anchor||_max={max_delta:.6e} "
                    f"anchor_backwards={state.anchor_backwards} "
                    f"anchor_grad_corrected={state.anchor_grad_corrected} "
                    f"anchor_optimizer_steps={state.anchor_optimizer_steps} "
                    f"anchor_batch_fraction={state.anchor_batch_fraction} "
                    f"anchor_backward_isolation_mode=clone "
                    # Anchor uses clean policy-gradient loss: ratio = 1, no clip,
                    # no old_log_probs.
                    f"anchor_loss=clean_pg anchor_ratio=1.0",
                    flush=True,
                )
            else:
                print(
                    f"[comm_eff] anchor refresh step={step} produced NO target grads "
                    f"(targets matched=0); check target_substr / use_orig_params",
                    flush=True,
                )

        if _rank1_fire or _rank1_warm_correct:
            assert spectral is not None and anchor_grads and deltas, (
                "rank1_relex full anchor fire completed without a populated M_anchor update"
            )
            state.rank1_m_ready = True
            print(
                f"[comm_eff][rank1_relex] step={step} M_READY=true "
                f"source={'warm_exact_pair' if _rank1_warm_correct else 'projected'} "
                f"anchor_backwards={state.anchor_backwards} targets={len(anchor_grads)} "
                f"correction_enabled_same_tick=true",
                flush=True,
            )

    def _maybe_comm_eff_grad_correction(self) -> None:
        """FSDP spectral gradient-correction hook.

        Runs in ``BaseEngine.train_batch`` AFTER the actor backward and BEFORE
        ``optimizer_step`` (which is where gradient clipping happens). Under
        FSDP2 (``fully_shard``) the backward has already reduced gradients
        across the data-parallel mesh by the time control reaches here, so this
        correction is applied **after FSDP gradient reduction** and **before
        gradient clipping** — a fact this method discovers empirically and logs
        rather than assumes.

        Strict no-op when comm_eff is disabled or no spectral filter is attached
        (the dense path is untouched, no collective is issued, no grad
        is read).

        On the first correction it logs, for >=1 target
        matrix, ``type(p.grad)``, the grad container shape, the logical 2D matrix
        shape, the FSDP wrapping/version, the DTensor placements/mesh, and the
        correction point relative to FSDP reduction and gradient clipping. The
        log lands in ``state.fsdp_grad_repr`` (surfaced into metrics) and in the
        training log via ``logger``.
        """
        state = getattr(self, "_comm_eff_state", None)
        if state is None or not getattr(state, "enabled", False):
            return
        spectral = getattr(state, "spectral", None)
        if spectral is None:
            return

        # Spectral-correction cadence gate. Advance the 1-based optimizer-step
        # counter and fire only on cadence steps. If spectral and anchor cadence
        # match, correction uses the anchor EMA refreshed earlier in this
        # train_batch. Dense/disabled runs never advance this counter.
        state.spectral_step += 1
        if not state.should_run_spectral_correction():
            return

        # rank1_relex warmup has an explicit safety barrier. Do not rely on the
        # merger's cold-M fallback: that path still traverses matrices, creates
        # anchor entries, and increments correction counters. Cadence advances
        # above, but the optimizer receives the untouched fast gradient until a
        # ready projected anchor has successfully populated M.
        if (
            hasattr(state, "rank1_relex_active")
            and state.rank1_relex_active()
            and not bool(getattr(state, "rank1_m_ready", False))
        ):
            state.rank1_correction_bypass_ticks += 1
            print(
                f"[comm_eff][rank1_relex] correction BYPASS tick={state.spectral_step} "
                f"rank1_m_ready=false spectral_corrections={state.spectral_corrections}",
                flush=True,
            )
            return

        spec_cfg = getattr(state.config, "spectral", None)
        target_substrs = self._comm_eff_target_names(spec_cfg)
        target_scope = self._comm_eff_target_scope(spec_cfg)
        # Full coverage default (-1). max_targets caps the merger too, so a
        # residual cap would silently skip matrices the merger should correct.
        max_targets = int(getattr(spec_cfg, "max_targets", -1)) if spec_cfg is not None else -1

        fsdp_ver = None
        try:
            fsdp_ver = fsdp_version(self.module)
        except Exception:  # pragma: no cover - defensive
            fsdp_ver = "unknown"
        module_is_fsdp1 = isinstance(self.module, FSDP)
        module_is_fsdp2 = isinstance(self.module, FSDPModule)

        # Ordering / wrapping facts logged once with the first correction. These
        # are facts (not assumptions): this hook is invoked by
        # BaseEngine.train_batch after forward_backward_batch (FSDP backward =>
        # grads already reduced) and before optimizer_step (where clip runs).
        discovery_meta = {
            "fsdp_version": str(fsdp_ver),
            "module_is_FSDP1": str(module_is_fsdp1),
            "module_is_FSDPModule_FSDP2": str(module_is_fsdp2),
            "correction_point": "after_actor_backward__before_optimizer_step",
            "relative_to_fsdp_reduction": "AFTER (FSDP backward reduces grads before this hook)",
            "relative_to_grad_clipping": "BEFORE (clip_grad_norm_ runs inside optimizer_step)",
            "world_size": str(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1),
        }

        # Expose original named params and grads. FSDP1 can flatten wrapped
        # units into 1-D FlatParameters, so materialize original unflattened
        # params/grads via summon_full_params. FSDP2 keeps original names with
        # DTensor grads and can be iterated directly.
        if module_is_fsdp1 and not module_is_fsdp2:
            # with_grads=True surfaces the unsharded .grad on each original
            # param inside the context; writeback=True copies edits back into
            # the FlatParameter shard on exit. summon_full_params all-gathers,
            # so this is the unsharded full-tensor view the filter needs.
            # NOTE: with_grads=True is ONLY supported when the module was wrapped
            # with use_orig_params=True (the launcher sets this for the
            # spectral correction). Guard so a misconfigured run fails loudly with a
            # clear message rather than a cryptic FSDP internal assert.
            use_orig = bool(getattr(self.engine_config, "use_orig_params", False))
            if not use_orig:
                raise RuntimeError(
                    "comm_eff spectral correction under FSDP1 requires "
                    "actor_rollout_ref.actor.fsdp_config.use_orig_params=true "
                    "(FSDP.summon_full_params(with_grads=True) is unsupported with "
                    "use_orig_params=false — grads live on a 1-D FlatParameter, "
                    "not the original 2D matrices). Set it in the launcher."
                )
            with FSDP.summon_full_params(self.module, with_grads=True, writeback=True):
                inner = getattr(self.module, "_fsdp_wrapped_module", self.module)
                self._apply_spectral_correction_core(
                    inner.named_parameters(),
                    spectral=spectral,
                    target_substrs=target_substrs,
                    target_scope=target_scope,
                    max_targets=max_targets,
                    state=state,
                    discovery_meta=discovery_meta,
                )
        else:
            # FSDP2 (DTensor grads) or non-FSDP (plain tensors). Original names
            # are intact; the core all-gathers DTensors via full_tensor().
            inner = getattr(self.module, "_fsdp_wrapped_module", self.module)
            self._apply_spectral_correction_core(
                inner.named_parameters(),
                spectral=spectral,
                target_substrs=target_substrs,
                target_scope=target_scope,
                max_targets=max_targets,
                state=state,
                discovery_meta=discovery_meta,
            )

    def _apply_spectral_correction_core(
        self,
        named_params,
        *,
        spectral,
        target_substrs,
        target_scope,
        max_targets,
        state,
        discovery_meta,
    ) -> int:
        """FSDP-agnostic core of the spectral gradient-correction hook.

        Iterates ``named_params`` (an iterator of ``(name, param)`` where each
        ``param`` exposes a full logical ``.grad`` — a plain ``Tensor`` or a
        ``DTensor``), and for every configured target tensor:

        * logs the FSDP gradient representation once, **regardless of gradient
          magnitude** — it fires on the
          first target with a non-``None`` grad even if that grad is ~0, so a
          degenerate-loss step still proves the hook ran;
        * applies the spectral filter and records the per-target
          ``||G_corr - G_comp|| / ||G_comp||`` ratio;
        * writes the corrected full tensor back into the (possibly sharded)
          grad in place and bumps ``state.spectral_corrections``.

        The iteration/discovery/correction loop itself lives in
        :func:`verl.workers.comm_eff.spectral_filter.apply_spectral_correction_to_params`.
        This method only supplies the two FSDP-specific callables — the DTensor unshard
        (``full_grad_of``) and the in-place writeback. Returns the number of
        matrices corrected.
        """
        from verl.workers.comm_eff.spectral_filter import apply_spectral_correction_to_params

        def full_grad_of(grad):
            # Present a full logical tensor to the (FSDP-agnostic) filter.
            # FSDP2 shards weights as DTensors; full_tensor() all-gathers the
            # logical matrix. The logical shape is the DTensor's global .shape.
            # An FSDP1-summoned grad (or CPU/non-FSDP) is already the full tensor.
            is_dtensor = isinstance(grad, DTensor)
            full = grad.full_tensor() if is_dtensor else grad
            placements = None
            mesh_shape = None
            if is_dtensor:
                try:
                    placements = str(grad.placements)
                    mesh_shape = str(tuple(grad.device_mesh.shape))
                except Exception:  # pragma: no cover
                    placements = "unavailable"
            meta = {
                "grad_container_type": type(grad).__name__,
                "grad_container_shape": str(tuple(grad.shape)),
                "is_dtensor": str(is_dtensor),
                "dtensor_placements": str(placements),
                "dtensor_mesh_shape": str(mesh_shape),
            }
            return full, meta

        def writeback(grad, g_proj):
            # For a DTensor: redistribute the corrected full tensor to the
            # original mesh/placements and copy the LOCAL shard back in place,
            # preserving the sharded layout the optimizer/clip expect. For a
            # plain Tensor (FSDP1-summoned full grad / CPU / non-FSDP): copy in
            # place directly.
            if isinstance(grad, DTensor):
                from torch.distributed.tensor import distribute_tensor

                redist = distribute_tensor(g_proj.to(grad.dtype), grad.device_mesh, grad.placements)
                grad.to_local().copy_(redist.to_local())
            else:
                grad.copy_(g_proj.to(grad.dtype))

        return apply_spectral_correction_to_params(
            named_params,
            spectral=spectral,
            target_substrs=target_substrs,
            target_scope=target_scope,
            max_targets=max_targets,
            state=state,
            discovery_meta=discovery_meta,
            full_grad_of=full_grad_of,
            writeback=writeback,
        )

    def _opt_reset_fsdp1_shard_infos(self) -> dict:
        """Map ``id(orig_param) -> _ShardParamInfo`` for FSDP1 use_orig_params.

        Outside ``summon_full_params`` each orig param (and hence its lazily
        created optimizer state) is the rank-local 1-D flat-param slice, so the
        anchor's FULL logical moment tensors must be sliced per param before
        the writeback. Empty for non-FSDP1 modules.
        """
        infos: dict = {}
        if not isinstance(self.module, FSDP):
            return infos
        for fsdp_mod in FSDP.fsdp_modules(self.module):
            handle = getattr(fsdp_mod, "_handle", None)
            flat_param = getattr(handle, "flat_param", None) if handle is not None else None
            if flat_param is None:
                continue
            params = getattr(flat_param, "_params", None)
            shard_infos = getattr(flat_param, "_shard_param_infos", None)
            if params is None or shard_infos is None:
                continue
            for param, info in zip(params, shard_infos, strict=False):
                infos[id(param)] = info
        return infos

    def _opt_reset_reduce_sq_sum(self, local_sq: float) -> float:
        """all-reduce(SUM) a local sum-of-squares over the DP/sharding group.

        Optimizer-state shards are disjoint slices of the logical tensors, so
        SUM of per-rank local sq-sums is the exact global L2^2 (the same
        geometry FSDP1's clip_grad_norm_ relies on). Mirrors the collective
        pattern of _dp_all_reduce_anchor_grads.
        """
        if not torch.distributed.is_initialized():
            return local_sq
        group = self.get_data_parallel_group()
        try:
            dp_world = torch.distributed.get_world_size(group=group)
        except Exception:
            dp_world = 1
        if dp_world <= 1:
            return local_sq
        total = torch.tensor([local_sq], dtype=torch.float32, device=get_device_id())
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM, group=group)
        return float(total.item())

    def _maybe_comm_eff_opt_reset(self) -> None:
        """Anchor-sourced optimizer-state reset (comm_eff.anchor.opt_reset).

        Runs at the END of the optimizer tick, from ``optimizer_step`` AFTER
        ``self.optimizer.step()`` — and therefore also after any anchor fire
        scheduled on the same tick (the anchor hook runs at the top of
        ``train_batch``). The cadence counts the same ``state.anchor_step``
        optimizer ticks the anchor cadence counts. Strict no-op while
        disabled: no state is read, no tensor is touched.
        """
        state = getattr(self, "_comm_eff_state", None)
        if state is None or not getattr(state, "enabled", False):
            return
        anchor_cfg = getattr(state.config, "anchor", None)
        opt_cfg = getattr(anchor_cfg, "opt_reset", None) if anchor_cfg is not None else None
        if opt_cfg is None or not bool(getattr(opt_cfg, "enabled", False)):
            return
        cadence = int(getattr(opt_cfg, "cadence", 50))
        tick = int(getattr(state, "anchor_step", 0))
        if tick <= 0 or tick % cadence != 0:
            return
        moments = getattr(state, "_opt_reset_moments", None)
        if moments is None or int(getattr(moments, "fires", 0)) <= 0:
            print(
                f"[comm_eff][opt_reset] SKIP tick={tick} cadence={cadence}: the anchor has never "
                "fired, so no clean moments exist yet (optimizer state untouched)",
                flush=True,
            )
            return

        from verl.workers.comm_eff.opt_reset import reset_optimizer_moments

        mode = str(getattr(opt_cfg, "mode", "anchor_moments"))
        scale_match = bool(getattr(opt_cfg, "scale_match", True))
        shard_infos = self._opt_reset_fsdp1_shard_infos()

        def writeback(state_tensor, param, full):
            # Full logical fp32 -> this rank's optimizer-state layout: DTensor
            # (FSDP2) redistributes like the spectral writeback; an FSDP1
            # use_orig_params state tensor is the 1-D flat-param slice its
            # _ShardParamInfo describes; a plain tensor is the full shape.
            if isinstance(state_tensor, DTensor):
                from torch.distributed.tensor import distribute_tensor

                redist = distribute_tensor(
                    full.to(state_tensor.dtype), state_tensor.device_mesh, state_tensor.placements
                )
                state_tensor.to_local().copy_(redist.to_local())
                return
            info = shard_infos.get(id(param))
            if info is not None and tuple(state_tensor.shape) != tuple(full.shape):
                if not bool(getattr(info, "in_shard", True)):
                    return
                start = int(info.intra_param_start_idx)
                end = int(info.intra_param_end_idx)
                src = full.reshape(-1)[start : end + 1]
            else:
                src = full.reshape(state_tensor.shape) if tuple(state_tensor.shape) != tuple(full.shape) else full
            state_tensor.copy_(src.to(device=state_tensor.device, dtype=state_tensor.dtype))

        def sq_sum_of(state_tensor):
            local = state_tensor.to_local() if isinstance(state_tensor, DTensor) else state_tensor
            local32 = local.detach().to(torch.float32)
            return float(torch.sum(local32 * local32).item())

        rho = reset_optimizer_moments(
            self.optimizer,
            list(self.module.named_parameters()),
            moments=moments,
            mode=mode,
            scale_match=scale_match,
            writeback=writeback,
            sq_sum_of=sq_sum_of,
            reduce_sq_sum=self._opt_reset_reduce_sq_sum,
        )
        state.opt_reset_count = int(getattr(state, "opt_reset_count", 0)) + 1
        if rho is not None:
            state.opt_reset_last_rho = float(rho)
        print(
            f"[comm_eff][opt_reset] FIRED tick={tick} cadence={cadence} mode={mode} "
            f"scale_match={scale_match} rho={rho if rho is not None else 'n/a'} "
            f"anchor_fires_folded={moments.fires} count={state.opt_reset_count}",
            flush=True,
        )

    def optimizer_zero_grad(self):
        """
        Zero gradients and enforce FSDP grad-clipping logic.
        """
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
        """
        assert self.optimizer_config.clip_grad is not None

        # getattr fallback: some subclasses (e.g. VeOmniEngine) bypass FSDPEngine.__init__.
        scaler = getattr(self, "scaler", None)

        # Unscale gradients before clip so the clip threshold is applied to true gradient
        # magnitudes, not scaled ones. scaler.step() will skip the update if any grad is inf/nan.
        if scaler is not None:
            scaler.unscale_(self.optimizer)

        if isinstance(self.module, FSDP):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        elif isinstance(self.module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), max_norm=self.optimizer_config.clip_grad
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        if scaler is not None:
            # scaler handles inf/nan skipping internally via _check_inf_per_device.
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # if grad_norm is not finite, skip the update
            if not torch.isfinite(grad_norm):
                print(f"WARN: grad_norm is not finite: {grad_norm}")
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()

        if self._qat_enabled:
            from verl.utils.qat.core import invalidate_all_scales

            invalidate_all_scales(self.module)

        # End-of-tick anchor-sourced optimizer-state reset. Placed AFTER the
        # step so the reset lands on the post-step moments of tick T, and after
        # any anchor fire of the same tick (the anchor hook ran at the top of
        # train_batch). No-op unless comm_eff.anchor.opt_reset is enabled.
        self._maybe_comm_eff_opt_reset()

        return grad_norm.item()

    def lr_scheduler_step(self):
        """
        Advance FSDP scheduler and return updated learning rate.
        """
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]  # only return the first group
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move FSDP model and/or optimizer to CPU or GPU with offload support.
        Note that this function executes irrespective of offload config. It serves as manual control
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            # force cpu_offload
            return

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_fsdp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                offload_fsdp_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save FSDP checkpoint, handling parameter offload as needed.
        """
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """
        Load FSDP checkpoint, restoring parameters and optimizer state.
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

        # FSDP2 CPUOffloadPolicy owns CPU<->GPU placement; calling model.to(device) here
        # leaves the module half-moved and crashes state_dict() below (#5995). The
        # per-DTensor .to(device).full_tensor() below still produces GPU tensors.
        if not self._uses_fsdp2_cpu_offload_policy:
            load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        merge_lora = self.model_config.lora.get("merge", False)

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            if not merge_lora:
                peft_config = peft_model.peft_config.get("default", None)
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                )
                if not base_sync_done:
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            else:  # merge lora
                with merged_lora_context(self.module, backup_adapters=True):
                    params = self.module.state_dict()
                    params = normalize_peft_param_name(params)
        else:
            params = self.module.state_dict()

        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params.items()
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )

        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer
            from verl.utils.torch_dtypes import PrecisionType

            mixed_precision_config = self.engine_config.mixed_precision
            if mixed_precision_config is not None:
                param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            else:
                param_dtype = torch.bfloat16

            quantizer = QATQuantizer(
                mode=self._qat_config.mode,
                group_size=self._qat_config.group_size,
                ignore_patterns=list(self._qat_config.ignore_patterns),
                device=torch.device(get_device_id()),
                param_dtype=param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=torch.device("cpu"),
            )

        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return per_tensor_param, peft_config_dict

    def disable_adapter(self) -> ContextManager:
        return self.module.disable_adapter()


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.engine.engine_config.fsdp_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithLMHead(FSDPEngine):
    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        temperature_item = temperature
        if use_fused_kernels:
            assert not isinstance(temperature, torch.Tensor), (
                "use_fused_kernels does not support per sample temperature yet"
            )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]

        # args used to get outputs
        output_args = {}

        if use_remove_padding:
            # Bump the PowerSGD forward generation and stamp the step before the
            # boundary projection hooks fire (no-op unless powersgd is live).
            self._comm_eff_maybe_set_powersgd_context(micro_batch, input_ids)
            # Set the per-token PRF context before the boundary mask/quant hooks
            # fire (no-op unless the prf_mask or sr_quant codec is live).
            self._comm_eff_maybe_set_mask_context(micro_batch, input_ids)
            # support per sample temperature
            # temperature (bsz,)
            # input_ids (bsz, j1)
            temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()  # (total_nnz,)
            temperature_rmpad = temperature_rmpad.unsqueeze(0)  # (1, total_nnz)

            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
                if position_ids.dim() == 3:
                    position_ids_rmpad = position_ids.values().unsqueeze(1)  # (4, 1, total_nnz)
                else:
                    position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.use_ulysses_sp:
                is_vlm_model = hasattr(getattr(self.module, "module", self.module).config, "vision_config")
                if is_vlm_model:
                    # vlm model's inputs will be sliced after embedding
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                else:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                        skip_position_ids_rmpad=getattr(self, "_veomni_handles_position_ids", False),
                    )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled,
                    position_ids_rmpad=None,
                    sp_size=self.ulysses_sequence_parallel_size,
                )

                temperature_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                    temperature_rmpad, position_ids_rmpad=None, sp_size=self.ulysses_sequence_parallel_size, pad_value=1
                )

                output_args["pad_size"] = pad_size

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
            temperature_rmpad = temperature_rmpad.squeeze(0)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
            output_args["temperature_rmpad"] = temperature_rmpad

            # only pass input_ids and position_ids to enable flash_attn_varlen

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = int(seq_len_effective.max().item())

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
                # we store the per sample temperature
                output_args["temperature"] = temperature

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                if position_ids.dim() == 3:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                    ).transpose(0, 1)  # (4, batch_size, max_seq_len)
                else:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, max_seq_len)
                    )

                attention_mask = build_attention_mask_from_nested(
                    input_ids=micro_batch["input_ids"], max_seq_len=max_seq_len
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }

            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        extra_args = {}
        if use_fused_kernels:
            extra_args["temperature"] = temperature_item
            extra_args["return_dict"] = True
            if use_remove_padding:
                # We have already computed `input_ids_rmpad_rolled` from the *full*
                # global sequence and (when SP>1) SP-sliced it. Pass it into the model
                # so the fused forward uses these labels verbatim instead of redoing
                # `torch.roll` on the local SP shard, which would wrap around the
                # shard boundary rather than the global sequence (issue #6068). This
                # mirrors what the veomni engine already does for fused kernels.
                extra_args["shift_labels"] = output_args["input_ids_rmpad_rolled"].unsqueeze(0)

        model_inputs.update(multi_modal_inputs)
        model_inputs.update(extra_args)

        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(
            data=micro_batch, key="calculate_sum_pi_squared", default=False
        )
        distillation_use_topk = tu.get_non_tensor_data(data=micro_batch, key="distillation_use_topk", default=False)

        if calculate_sum_pi_squared and use_fused_kernels:
            raise NotImplementedError(
                "calculate_sum_pi_squared=True is not supported with use_fused_kernels=True: "
                "fused kernels do not materialize the full logits tensor needed for Σπ²."
            )

        model_output = {}

        input_ids = micro_batch["input_ids"]

        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            temperature_rmpad = output_args["temperature_rmpad"]

            if use_fused_kernels:
                # temperature is singleton
                log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
            else:
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                # compute entropy
                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )

                # compute sum_pi_squared (Σπ²) for optimal-baseline advantage estimators
                if calculate_sum_pi_squared:
                    sum_pi_squared_rmpad = verl_F.calculate_sum_pi_squared_from_logits(logits_rmpad)

                # logits_processor_func return tensors with shape (1, total_nnz/sp_size)
                if distillation_use_topk:
                    outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
                    cu_seqlens = input_ids.offsets()
                    for k, v in outputs.items():
                        v = v.squeeze(0)
                        assert v.shape == log_probs.shape, f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}"
                        if self.use_ulysses_sp:
                            pad_size = output_args["pad_size"]
                            v = gather_outputs_and_unpad(v, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)

            # gather log_prob if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]

                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(
                    log_probs,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
                if calculate_entropy:
                    entropy_rmpad = gather_outputs_and_unpad(
                        entropy_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                if calculate_sum_pi_squared:
                    sum_pi_squared_rmpad = gather_outputs_and_unpad(
                        sum_pi_squared_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                if calculate_sum_pi_squared:
                    sum_pi_squared = torch.nested.nested_tensor_from_jagged(sum_pi_squared_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:  # not using rmpad and no ulysses sp
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:
                logits = output.logits  # (bsz, response_length, vocab_size)
                temperature = output_args["temperature"]  # (bsz,)
                temperature = temperature.unsqueeze(-1).unsqueeze(-1)
                logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if calculate_sum_pi_squared:
                    sum_pi_squared = verl_F.calculate_sum_pi_squared_from_logits(logits)

                if pad_mode == DatasetPadMode.NO_PADDING:
                    cu_seqlens = input_ids.offsets()
                    seq_lengths = cu_seqlens.diff()
                    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                    logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                    logits_rmpad = torch.cat([t for t in logits.unbind()])
                    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                    # Mirror the use_remove_padding=True branch (see verl#6293).
                    # No Ulysses SP gather here: this branch is the no-SP path
                    # (log_probs is also not gathered) and pad_size is only
                    # populated in output_args along the use_remove_padding=True
                    # path of prepare_model_inputs.
                    if distillation_use_topk:
                        outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
                        for k, v in outputs.items():
                            v = v.squeeze(0)
                            assert v.shape == log_probs.shape, (
                                f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}"
                            )
                            model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)

                    # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                    if calculate_entropy:
                        entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                        entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                        entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = torch.nested.narrow(
                            sum_pi_squared, 1, starts, seq_lengths, layout=torch.jagged
                        )
                        sum_pi_squared_rmpad = torch.cat([t for t in sum_pi_squared.unbind()])
                        sum_pi_squared = torch.nested.nested_tensor_from_jagged(sum_pi_squared_rmpad, cu_seqlens)
                else:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy
        if calculate_sum_pi_squared:
            model_output["sum_pi_squared"] = sum_pi_squared

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        # actually, we should avoid assigning like this...
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        # Honor mixed_precision.param_dtype resolved during FSDP setup. When dtype is fp32,
        # autocast is a no-op at best and a footgun at worst, so skip it entirely.
        # getattr fallback: some subclasses (e.g. VeOmniEngine) bypass FSDPEngine.__init__
        # and _build_fsdp_module, so self._autocast_dtype may not be set.
        autocast_dtype = getattr(self, "_autocast_dtype", torch.bfloat16)
        autocast_ctx: ContextManager = (
            nullcontext()
            if autocast_dtype == torch.float32
            else torch.autocast(device_type=device_name, dtype=autocast_dtype)
        )
        with autocast_ctx:
            raw_output = self.module(
                **model_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch, logits_processor_func=loss_function
            )

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithValueHead(FSDPEngineWithLMHead):
    """
    The only difference between critic and actor is how the raw model output is processed
    """

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)

        input_ids = micro_batch["input_ids"]
        if use_remove_padding:
            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values_rmpad = output[2].squeeze(0)
            else:
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz, 1)
                # critic model arch is like Qwen3ForTokenClassfication and num_labels=1
                # so we squeeze the last dimension here to get the value for each token
                values_rmpad = values_rmpad.squeeze(-1)

            # gather output if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]
                values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values = output[2]
            else:
                values = output.logits.squeeze(-1)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                seq_lengths = cu_seqlens.diff()
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                values = torch.nested.narrow(values, 1, starts, seq_lengths, layout=torch.jagged)
                values_rmpad = torch.cat([t for t in values.unbind()])
                # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        return {"values": values}
