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

        optimizer = build_optimizer(module.parameters(), self.optimizer_config)

        return optimizer

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

    def _comm_eff_mask_active(self, forward_only: bool) -> bool:
        """True iff the activation-mask hooks should be live for this forward.

        Masking is confined to the actor-train forward/backward by default, and
        additionally to the old-policy log-prob recompute when
        ``comm_eff.mask.mask_recompute=true``. This returns False
        (strict no-op) unless ALL of:
          * an enabled ``CommEffState`` is attached,
          * the worker has set ``state.mask_active`` (set only around
            ``update_actor`` and, with mask_recompute, around
            ``compute_log_prob``; cleared everywhere else),
          * the path_tag is eligible per ``mask_eligible_tags(state)``,
          * a masker was constructed (mask sub-config enabled, ``p > 0``),
          * the pass-type matches the path:
              - ``train``        ⇒ requires ``forward_only=False`` (the
                                     gradient-bearing actor train forward),
              - ``old_logprob``  ⇒ requires ``forward_only=True`` AND
                                     ``mask.mask_recompute=true`` (the
                                     compute_log_prob infer pass; consumes
                                     pipeline-boundary bandwidth, no backward
                                     against this forward — but the recomputed
                                     old_logp ENTERS the next train forward via
                                     the PPO importance ratio).
        Anchor pass (``path_tag=None``) never enters this method positively; the
        anchor uses ``state.mask_active=False``.
        """
        # Import locally to keep the engine's import surface unchanged.
        from verl.workers.comm_eff.state import OLD_LOGPROB_TAG, TRAIN_TAG, mask_eligible_tags

        state = getattr(self, "_comm_eff_state", None)
        if state is None or not getattr(state, "enabled", False):
            return False
        if not getattr(state, "mask_active", False):
            return False
        if getattr(state, "masker", None) is None:
            return False
        tag = getattr(state, "path_tag", None)
        eligible = mask_eligible_tags(state)
        if tag not in eligible:
            return False
        # Pass-type / path consistency:
        if tag == TRAIN_TAG:
            # Train forward MUST be the bwd-bearing pass.
            return not forward_only
        if tag == OLD_LOGPROB_TAG:
            # old_logprob recompute is forward-only by construction
            # (compute_log_prob → infer_batch → forward_only=True). Refuse a
            # backward-bearing pass stamped old_logprob — that would be a
            # mis-wired entrypoint.
            return forward_only
        # Any other eligible tag is unexpected here; bail safely (no mask).
        return False

    def _comm_eff_register_mask_hooks(self) -> bool:
        """Register the per-element mask hooks on the boundary blocks.

        The per-token PRF context (``global_step`` + token-aligned
        ``sample_ids`` / ``position_ids``) is set per micro-batch in
        ``prepare_model_inputs``, since the stable ids are only known once the
        micro-batch is packed. SP guard: the key is aligned to the rmpad token
        axis, which Ulysses SP>1 slices/pads across ranks (out of scope) — refuse
        it loudly. Returns True if hooks were registered.
        """
        if getattr(self, "ulysses_sequence_parallel_size", 1) and self.ulysses_sequence_parallel_size > 1:
            raise NotImplementedError(
                "comm_eff per-element masking does not support "
                f"ulysses_sequence_parallel_size>1 (got {self.ulysses_sequence_parallel_size}); "
                "the launcher runs with SP=1."
            )
        state = self._comm_eff_state
        masker = state.masker
        masker.register(self.module)
        return masker.is_registered

    def _comm_eff_maybe_set_mask_context(self, micro_batch: TensorDict, input_ids) -> None:
        """Set the per-token PRF context for this micro-batch's masked forward.

        No-op unless mask hooks are live. Builds, in the packed order of
        ``input_ids.values()`` (the activation token axis under SP=1):
        ``sample_ids`` (each row's ``comm_eff_sample_id`` repeated across its
        tokens) and ``position_ids`` (position within each sequence, from the
        rmpad ``cu_seqlens``). Keying on these stable ids makes the mask
        packing-invariant.
        """
        state = getattr(self, "_comm_eff_state", None)
        if state is None:
            return
        masker = getattr(state, "masker", None)
        if masker is None or not masker.is_registered:
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
                "before micro-batching (engine_workers.update_actor / "
                "compute_log_prob)."
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
        masker.set_context(
            global_step=int(getattr(self, "_comm_eff_global_step", 0)),
            sample_ids=sample_ids,
            position_ids=position_ids,
        )

    def _comm_eff_powersgd_active(self, forward_only: bool) -> bool:
        """True iff the PowerSGD projection hooks should be live for this forward.

        The projector is confined to the actor-train forward/backward
        (``path_tag == "train"``,
        ``forward_only=False``) and, when ``powersgd.compress_recompute=true``, to
        the old-policy log-prob recompute (``path_tag == "old_logprob"``,
        ``forward_only=True``). Both paired forwards see the same frozen ``Q_t``;
        the basis only advances after the gradient-bearing work. Returns
        False (strict no-op) unless an enabled state with the powersgd codec and a
        live ``mask_active`` flag is attached.
        """
        from verl.workers.comm_eff.state import OLD_LOGPROB_TAG, TRAIN_TAG

        state = getattr(self, "_comm_eff_state", None)
        if state is None or not getattr(state, "enabled", False):
            return False
        if getattr(state, "powersgd", None) is None:
            return False
        # `mask_active` is the shared "compressed-forward is live" flag the worker
        # sets around update_actor / (recompute) compute_log_prob; it gates BOTH
        # codecs (the name is historical). A clean step clears it ⇒ dense forward.
        if not getattr(state, "mask_active", False):
            return False
        tag = getattr(state, "path_tag", None)
        if tag == TRAIN_TAG:
            return not forward_only
        if tag == OLD_LOGPROB_TAG:
            # Only when compress_recompute=true did the worker stamp mask_active
            # on the recompute; the recompute is forward_only by construction.
            return forward_only
        return False

    def _comm_eff_register_powersgd_hooks(self) -> bool:
        """Register the PowerSGD projection hooks on the boundary blocks.

        SP guard mirrors the mask: the boundary activation token axis is what
        Ulysses SP>1 slices across ranks (out of scope) — refuse it loudly. The
        per-forward context (global_step + the generation bump that dedupes the
        basis sketch against grad-ckpt recompute) is set per micro-batch in
        ``prepare_model_inputs``. Returns True iff hooks were registered.
        """
        if getattr(self, "ulysses_sequence_parallel_size", 1) and self.ulysses_sequence_parallel_size > 1:
            raise NotImplementedError(
                "comm_eff powersgd does not support "
                f"ulysses_sequence_parallel_size>1 (got {self.ulysses_sequence_parallel_size}); "
                "the launcher runs with SP=1."
            )
        state = self._comm_eff_state
        compressor = state.powersgd
        compressor.register(self.module)
        return compressor.is_registered

    def _comm_eff_maybe_set_powersgd_context(self, micro_batch: TensorDict, input_ids) -> None:
        """Bump the PowerSGD forward generation and stamp global_step.

        No-op unless the projection hooks are live. Unlike the mask, PowerSGD does
        not key on token identity (its basis is shared across all tokens), so the
        only per-micro-batch state is the generation counter (dedupes the sketch
        against gradient-checkpoint recompute) and the trainer step.

        rmpad guard: the boundary activation ``M`` the projector compresses is
        the rmpad (nested / no-padding) token axis. If a caller ever ran without
        ``use_remove_padding=True`` the
        activation would be a PADDED ``(B, T, H)`` block and the projector +
        basis sketch ``V`` would silently fold PAD tokens into ``M`` and into the
        codebook — corrupting both the reconstruction metric and the learned
        basis. Refuse it loudly here rather than produce a quietly-wrong codec.
        The launcher runs rmpad + SP=1, so this never fires in the sanctioned
        config; it only catches a future mis-launch.
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
                "into the projected activation M and the basis sketch V. The "
                "launcher runs rmpad + SP=1."
            )
        compressor.set_context(global_step=int(getattr(self, "_comm_eff_global_step", 0)))

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
        # comm_eff activation-mask hook lifecycle: register hooks on entry to the
        # train forward/backward and remove them on exit, so a later log-prob /
        # infer / ref / validation forward on the same module is clean. When
        # disabled (default) or not on the actor-train path, nothing is registered
        # and no RNG is drawn, so the pass is byte-identical to dense GRPO.
        #
        # The PowerSGD codec uses the same register/unregister lifecycle.
        # The two codecs are mutually exclusive (state.build constructs exactly
        # one), so at most one branch fires.
        _mask_hooks_live = False
        _powersgd_hooks_live = False
        if self._comm_eff_mask_active(forward_only=forward_only):
            _mask_hooks_live = self._comm_eff_register_mask_hooks()
        elif self._comm_eff_powersgd_active(forward_only=forward_only):
            _powersgd_hooks_live = self._comm_eff_register_powersgd_hooks()
        try:
            return self._forward_backward_batch_inner(data, loss_function, forward_only=forward_only)
        finally:
            if _mask_hooks_live:
                self._comm_eff_state.masker.unregister()
            if _powersgd_hooks_live:
                self._comm_eff_state.powersgd.unregister()

    def _forward_backward_batch_inner(
        self, data: TensorDict, loss_function: Callable, forward_only=False
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

                if not forward_only:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            output_lst.append(meta_info)

        # postprocess and return
        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def _comm_eff_target_names(self, spec_cfg) -> tuple:
        """Substrings selecting which named 2D params receive spectral correction."""
        substrs = getattr(spec_cfg, "target_substr", None)
        if substrs is None:
            return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        return tuple(substrs)

    def _dp_all_reduce_anchor_grads(self, anchor_grads: dict) -> dict:
        """EXP-25 (R1, FIX 5): all-reduce(MEAN) ``G_anchor`` across the actor DP group.

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

        **Collective safety (deadlock guard).** Walk a FIXED ``sorted(keys)`` order
        and contribute a correctly-shaped ZERO for any target a rank lacks, so
        every rank issues the IDENTICAL collective sequence (mirrors the PowerSGD
        sketch-sync discipline). The clone arch + target_substrs are identical
        across ranks and the DP shards are symmetric, so the key set is identical
        by construction; the zero-fill is belt-and-braces against any future asym.

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
        # FIXED sorted order so every rank issues the same collective sequence.
        names = sorted(anchor_grads.keys())
        norm_pre = {}
        norm_post = {}
        for name in names:
            g = anchor_grads[name]
            gd = g.to(torch.float32)
            norm_pre[name] = float(torch.linalg.norm(gd).item())
            # all-reduce(SUM) then divide by dp_world == MEAN. (ReduceOp.AVG is not
            # available on every backend; SUM+/dp_world is portable + exact.)
            torch.distributed.all_reduce(gd, op=torch.distributed.ReduceOp.SUM, group=group)
            gd /= float(dp_world)
            norm_post[name] = float(torch.linalg.norm(gd).item())
            anchor_grads[name] = gd.to(g.dtype)
        if names:
            # Mean pre/post ratio across targets — a MEAN reduce keeps it ~O(1);
            # a SUM bug would show ~dp_world. Greppable scale falsifier.
            import statistics as _stats
            ratios = [norm_post[n] / norm_pre[n] for n in names if norm_pre[n] > 0]
            ratio_mean = _stats.fmean(ratios) if ratios else 0.0
            print(
                f"[comm_eff][EXP-25][dp-reduce] anchor G_anchor all-reduced(MEAN) over DP "
                f"dp_world={dp_world} targets={len(names)} "
                f"||G||_post/||G||_pre_mean={ratio_mean:.4f} "
                f"(MEAN ⇒ ~O(1) per-rank-shard-dependent; a SUM bug ⇒ ~{dp_world}x)",
                flush=True,
            )
        return anchor_grads

    def _broadcast_anchor_M(self, spectral, anchor_grads: dict, *, src: int = 0) -> dict:
        """EXP-25 (R2): ``dist.broadcast`` the anchor EMA ``M`` to every DP rank.

        After the DP-reduce + EMA feed, ``M_anchor`` is already bit-identical
        across ranks (the all-reduce made ``G_anchor`` identical, and the EMA is
        deterministic). This broadcast is the POSITIVE-RECEIPT mechanism the (R2)
        invariant requires: it proves every fast/DP rank holds the anchor's ``M``
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
        """EXP-25 (R1, checklist #2): assert ``M_anchor`` is bit-identical across DP.

        All-gathers a per-target fp64 checksum of the EMA ``M`` over the DP group
        and asserts the max cross-rank relative deviation is ``<= atol``. After the
        all-reduce(MEAN) of ``G_anchor`` the EMA is deterministic, so ``M`` MUST be
        identical on every rank; a non-zero deviation proves the DP-reduce did not
        run / used the wrong group (the R1 FIX-5 falsifier). Symmetric collective
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
            f"[comm_eff][EXP-25][M-dp-identical] step={step} targets={len(names)} "
            f"cross_rank_max_rel_dev={max_rel:.3e} (0 ⇒ M is the GLOBAL DP-reduced gradient)",
            flush=True,
        )
        assert max_rel <= atol, (
            f"comm_eff anchor M DIVERGED across DP ranks (max_rel_dev={max_rel:.3e} > atol={atol:.1e}); "
            "the all-reduce(MEAN) of G_anchor did not make M identical — R1 FIX-5 broken "
            "(wrong process group / reduce never ran)."
        )

    def _build_anchor_pg_loss(self, fast_path_loss_function, anchor_pg_loss):
        """Bind the clean policy-gradient loss for the anchor pass.

        The anchor must NOT reuse the fast-path PPO ratio/clip loss (its
        ``old_log_probs`` are from the MASKED path; the ratio against the
        anchor's UNMASKED forward ≠ 1, so the clip corrupts ``G_anchor``).
        Instead we run ``anchor_pg_loss`` (ratio ≡ 1) bound to the SAME actor
        ``config`` the fast path carries, so ``agg_loss`` normalizes identically
        and ``M_anchor`` is the clean true gradient at the same scale.

        ``fast_path_loss_function`` is ``functools.partial(ppo_loss,
        config=actor_config)``; we read ``config`` off ``.keywords`` and rebind.
        This touches ONLY the anchor pass — the fast path keeps its real loss.
        """
        from functools import partial

        config = None
        kw = getattr(fast_path_loss_function, "keywords", None)
        if kw is not None:
            config = kw.get("config")
        if config is None:
            raise RuntimeError(
                "comm_eff anchor C4: could not read 'config' off the fast-path "
                "loss_function (expected functools.partial(ppo_loss, config=...)). "
                "The clean-gradient anchor loss needs the actor config for "
                "agg_loss normalization."
            )
        return partial(anchor_pg_loss, config=config)

    def _maybe_comm_eff_anchor_refresh(self, data, loss_function) -> None:
        """FSDP anchor-circuit refresh: unmasked K-stale GRPO-actor-loss
        fwd/bwd -> RAW G_anchor -> spectral anchor EMA, NO optimizer step.

        Runs at the TOP of ``BaseEngine.train_batch`` (before the masked fast
        path). The six non-negotiable invariants this enforces:

        1. **Clean unmasked policy-gradient loss.** The anchor
           uses ``anchor_pg_loss`` (ratio ≡ 1, no clip, no ``old_log_probs``)
           over THIS rollout-expanded batch — its gradient is the CLEAN true
           policy gradient ``-(A·∇logπ_unmasked)`` at the K-stale weights. It is
           bound to the SAME actor config the fast path carries (so ``agg_loss``
           normalizes identically). It is NOT a supervised next-token loss, and
           it is NOT the fast-path PPO loss: reusing the fast path's ``ppo_loss``
           here would feed the MASKED-path ``old_log_probs`` against the anchor's
           UNMASKED forward, making the importance ratio ≠ 1 and letting the PPO
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
        5. **Unmasked.** The pass runs with ``state.mask_active=False``
           and ``path_tag != "train"`` so the mask hooks are NOT registered;
           ``anchor_mask_applications`` is recorded as the (asserted-zero) delta
           of ``state.mask_applications`` around the pass.
        6. **Uncorrected.** ``G_anchor`` is read RAW and fed to
           ``SpectralFilter.update_anchor`` (the EMA) BEFORE any fast-path
           corrector; ``anchor_grad_corrected`` stays 0.
        """
        state = getattr(self, "_comm_eff_state", None)
        if state is None or not getattr(state, "enabled", False):
            return
        anchor_cfg = getattr(state.config, "anchor", None)
        spectral = getattr(state, "spectral", None)
        if anchor_cfg is None or not bool(getattr(anchor_cfg, "enabled", False)) or spectral is None:
            return

        from verl.workers.comm_eff.anchor import (
            AnchorStalenessQueue,
            anchor_pg_loss,
            anchor_should_fire,
            assert_anchor_module_isolated,
            build_anchor_module,
            extract_target_grads,
            feed_anchor_grads_into_ema,
            snapshot_named_params,
        )
        # Canonicalize FSDP wrap-infix so the (possibly fallback non-infixed)
        # anchor clone matches the live module's per-layer-wrapped snapshot keys.
        from verl.workers.comm_eff.spectral_filter import _canon

        cadence = int(getattr(anchor_cfg, "cadence", 20))
        delay_K = int(getattr(anchor_cfg, "delay_K", 20))

        # Advance the trainer-step counter the cadence is keyed on (1-based).
        state.anchor_step += 1
        step = state.anchor_step

        # Lazily build the staleness queue on the state (survives across steps).
        # CommEffState is a plain class with a __dict__, so a direct setattr is
        # correct; it is the single object shared with the worker.
        queue = getattr(state, "_anchor_queue", None)
        if queue is None:
            queue = AnchorStalenessQueue(delay_K=delay_K)
            setattr(state, "_anchor_queue", queue)

        spec_cfg = getattr(state.config, "spectral", None)
        target_substrs = self._comm_eff_target_names(spec_cfg)
        # EXP-25: default to FULL coverage (-1) — the merger corrects ALL 196
        # matrices, and max_targets caps BOTH the anchor extraction and the merger.
        max_targets = int(getattr(spec_cfg, "max_targets", -1)) if spec_cfg is not None else -1

        # EXP-25 (R2): anchor-owns-Q — when on, the anchor's stale forward also
        # harvests slow-net activations into the PowerSGD sketch V, computes
        # Q ← orth(V), and broadcasts Q (and M) to every DP rank.
        anchor_owns_q = bool(getattr(anchor_cfg, "owns_q", False))
        powersgd = getattr(state, "powersgd", None)
        do_anchor_q = anchor_owns_q and powersgd is not None

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

        with _summon_ctx():
            cur_snapshot = snapshot_named_params(
                _inner_named_params(), target_substrs=None, device=None, detach=True
            )
        queue.push(step, cur_snapshot)

        if not anchor_should_fire(step, cadence, True):
            return

        # --- fetch the t-K stale snapshot to forward from ----------------------
        stale = queue.get_stale(step, delay_K)
        if stale is None:  # pragma: no cover - queue always has >=1 after push
            return

        # Ensure masking is off for the anchor pass and the path
        # tag is NOT "train" (the mask hook requires both to fire). We measure
        # mask applications as a delta so a leak is a loud failure.
        prev_mask_active = getattr(state, "mask_active", False)
        prev_path_tag = getattr(state, "path_tag", None)
        state.mask_active = False
        if hasattr(state, "set_path_tag"):
            state.set_path_tag(None)
        mask_apps_before = int(getattr(state, "mask_applications", 0))
        opt_steps_before = int(getattr(state, "anchor_optimizer_steps", 0))

        # Shallow-copy the batch so the anchor fwd/bwd never mutates the
        # TensorDict the masked fast path reuses immediately after.
        anchor_data = data.copy() if hasattr(data, "copy") else data

        anchor_grads = {}
        # The anchor's loss.backward() MUST NOT
        # trigger the live FSDP1 module's `_post_backward_hook` (which would
        # call `_check_grad_to_accumulate(flat_param._saved_grad_shard.shape)`
        # outside the fast-path window where `_saved_grad_shard is None` →
        # `AttributeError: 'NoneType' object has no attribute 'shape'`).
        #
        # Mechanism: deep-copy the underlying nn.Module (after summoning full
        # FSDP1 params so the deepcopy captures full unsharded weights), load
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
                # Cache anchor clone across refreshes so we do NOT
                # allocate a fresh Qwen2ForCausalLM (~3 GB) every step — that was
                # tripping vLLM v1's sleep_replicas memory assertion at step 2.
                # The K-stale snapshot is loaded INTO the cached clone below.
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
            assert_anchor_module_isolated(
                anchor_module, optimizer=self.optimizer, fsdp_module=inner
            )

            # Move the clone to the live module's device + dtype so its
            # forward/backward runs on the same accelerator.
            try:
                live_p = next(inner.parameters())
                anchor_module.to(device=live_p.device, dtype=live_p.dtype)
            except StopIteration:
                pass

            # Load the K-stale snapshot weights into the clone (NOT into the
            # live module — the live optimizer's params remain untouched).
            # The `stale` snapshot is keyed by the live module's
            # (FSDP per-layer-wrapped) names — those carry the
            # `._fsdp_wrapped_module.` infix — while the clone (when the deepcopy
            # path fell back to a plain config-rebuild) has NON-infixed names. A
            # raw `n in stale` lookup then never matches → the clone keeps RANDOM
            # init weights → G_anchor is garbage. Match by canonical (infix-
            # stripped) key so the clone receives the REAL delay_K-stale weights.
            stale_canon = {_canon(k): v for k, v in stale.items()}
            with torch.no_grad():
                loaded = 0
                for n, p in anchor_module.named_parameters():
                    s = stale_canon.get(_canon(n))
                    if s is not None and s.shape == p.shape:
                        p.copy_(s.to(p.device, p.dtype))
                        loaded += 1
            print(
                f"[comm_eff][EXP-18][anchor-load] loaded {loaded}/{sum(1 for _ in anchor_module.named_parameters())} "
                f"stale params into clone (canon-matched)",
                flush=True,
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

            # EXP-25 (R2): anchor-owns-Q. Register the PowerSGD projection hooks
            # ON THE CLONE so the anchor's UNMASKED stale-weight forward folds its
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

            # UNMASKED forward/backward on the CLONE. No FSDP hooks fire (the
            # clone has none). mask_active=False ⇒ no mask hooks fire on the
            # clone either (the masker is registered on self.module — now the
            # clone — but mask_active gates the work). forward_only=False
            # populates .grad on the clone's plain Parameters.
            #
            # Clean anchor gradient. The anchor uses a plain
            # policy-gradient loss (ratio ≡ 1, no clip, no old_log_probs) instead
            # of the fast-path PPO loss. The batch's old_log_probs came from the
            # MASKED fast path; reusing them against this UNMASKED forward makes
            # the GRPO importance ratio ≠ 1 → the PPO clip mangles G_anchor →
            # M_anchor was never the clean true gradient. anchor_pg_loss fixes
            # that (gradient = -(A·∇logπ_unmasked) at the stale weights). The
            # fast-path loss_function (real ratio/clip) is UNTOUCHED — this swap
            # is anchor-pass-only. We bind the SAME actor config the fast path
            # uses (read off the partial) so agg_loss normalizes identically.
            anchor_loss_function = self._build_anchor_pg_loss(loss_function, anchor_pg_loss)
            self._forward_backward_batch_inner(anchor_data, anchor_loss_function, forward_only=False)

            # Read G_anchor RAW per target (NO correct_matrix) off
            # the clone. full_grad_of is the identity — the clone is a plain
            # nn.Module so its p.grad is already a full 2D tensor.
            def _full_grad_of(grad):
                return grad, {"grad_container_type": type(grad).__name__, "is_dtensor": str(isinstance(grad, DTensor))}

            anchor_grads = extract_target_grads(
                anchor_module.named_parameters(),
                target_substrs=target_substrs,
                max_targets=max_targets,
                full_grad_of=_full_grad_of,
            )
        finally:
            # EXP-25 (R2): tear down the anchor's PowerSGD hooks on the clone and
            # clear the sketch-harvest mode so the live fast path is untouched.
            # The sketch V (just harvested) PERSISTS on the compressor — consumed
            # by anchor_update_basis below. Q/M broadcasts also happen below.
            if do_anchor_q:
                try:
                    powersgd.unregister()
                finally:
                    powersgd.set_anchor_sketch_mode(False)
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
            except UnboundLocalError:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            # Restore the prior mask/path state regardless of outcome.
            state.mask_active = prev_mask_active
            if hasattr(state, "set_path_tag"):
                state.set_path_tag(prev_path_tag)
            # The live optimizer's param.grads were NEVER touched by the
            # anchor pass (we ran fwd/bwd on the clone), so the masked fast
            # path that follows starts from whatever grads were there at
            # entry.

        # The anchor pass must have fired zero mask hooks.
        mask_apps_after = int(getattr(state, "mask_applications", 0))
        anchor_mask_delta = mask_apps_after - mask_apps_before
        state.anchor_mask_applications += max(0, anchor_mask_delta)
        assert anchor_mask_delta == 0, (
            f"comm_eff anchor pass fired {anchor_mask_delta} mask hooks "
            "(anchor_mask_applications must be 0 — the anchor runs UNMASKED on "
            "the actor-train path; GUARD 5 violated)."
        )
        # The anchor took no optimizer step.
        assert int(getattr(state, "anchor_optimizer_steps", 0)) == opt_steps_before, (
            "comm_eff anchor pass took an optimizer step (anchor_optimizer_steps "
            "must stay 0; snapshot is OFF the optimizer's param group)."
        )

        # EXP-25 (R1, FIX 7): COVERAGE SET-EQUALITY — the anchor M must cover EVERY
        # matrix the merger corrects (set-equal, NOT 4 / NOT boundary-only). Build
        # the expected merger set from the SAME substring+2D selector the merger
        # uses, over the live module's named_parameters (architecture == the
        # clone), and assert set(anchor_grads canon) == set(expected canon) when
        # uncapped. A mismatch is the EXP-23 coverage bug; emit the count + the
        # symmetric difference so it is greppable. (Only meaningful when uncapped:
        # max_targets<0; a diagnostic cap deliberately narrows both.)
        from verl.workers.comm_eff.spectral_filter import _canon as _canon_cov
        try:
            with _summon_ctx():
                _inner_cov = getattr(self.module, "_fsdp_wrapped_module", self.module)
                expected = {
                    _canon_cov(n)
                    for n, p in _inner_cov.named_parameters()
                    if any(s in n for s in target_substrs) and getattr(p, "ndim", p.dim()) == 2
                }
        except Exception as _cov_exc:  # pragma: no cover - defensive
            expected = set()
            print(f"[comm_eff][EXP-25][coverage] WARN could not enumerate expected set: {_cov_exc!r}", flush=True)
        got = {_canon_cov(k) for k in anchor_grads.keys()}
        if expected:
            missing = expected - got
            extra = got - expected
            print(
                f"[comm_eff][EXP-25][coverage] anchor_targets={len(got)} merger_expected={len(expected)} "
                f"set_equal={got == expected} missing={sorted(missing)[:6]}{'...' if len(missing) > 6 else ''} "
                f"extra={sorted(extra)[:6]}{'...' if len(extra) > 6 else ''}",
                flush=True,
            )
            if max_targets < 0:
                assert got == expected, (
                    f"comm_eff anchor coverage MISMATCH (R1 FIX 7): anchor covers {len(got)} targets but the "
                    f"merger corrects {len(expected)}; missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}. "
                    "set(anchor M) MUST == set(merger targets) at full coverage (max_targets=-1)."
                )

        # EXP-25 (R1, FIX 5): all-reduce(MEAN) G_anchor across the DP group so
        # M_anchor is the GLOBAL stale gradient (bit-identical across ranks, at the
        # correct mean scale), BEFORE the EMA. The anchor clone had no FSDP
        # reduction, so without this M is each rank's local-shard gradient.
        anchor_grads = self._dp_all_reduce_anchor_grads(anchor_grads)

        # Feed RAW (now DP-reduced) grads into the EMA (update_anchor, NEVER correct_matrix).
        deltas = feed_anchor_grads_into_ema(anchor_grads, spectral, state=state)
        state.anchor_backwards += 1
        # anchor_batch_fraction: this implementation consumes the WHOLE batch.
        state.anchor_batch_fraction = 1.0

        # EXP-25 (R2): anchor-owned Q. Now that the slow-net activations are
        # harvested into V (during the clean anchor forward above), compute
        # Q ← orth(V) on the ANCHOR (DP-synced) and BROADCAST both Q and the freshly
        # EMA'd M to every DP rank with a positive receipt. The fast net's local
        # Q-update is gated OFF (engine_workers.py), so the anchor is the SOLE Q
        # writer. All ranks reach this in lockstep (the anchor fired on all ranks).
        if do_anchor_q:
            q_updated = powersgd.anchor_update_basis()
            q_receipts = powersgd.broadcast_basis(src=0)
            m_receipts = self._broadcast_anchor_M(spectral, anchor_grads, src=0)
            # Cross-rank consensus guard (must not raise): the anchor-owned Q must
            # be identical on every DP rank + both boundary sides.
            try:
                qdev = powersgd.verify_basis_agreement_across_ranks()
            except RuntimeError:
                raise
            if q_receipts:
                changed_q = sum(1 for r in q_receipts.values() if r.get("changed"))
                print(
                    f"[comm_eff][bcast] step={step} Q updated={q_updated} broadcast boundaries={len(q_receipts)} "
                    f"changed={changed_q} cross_rank_max_rel_dev={qdev if qdev is not None else 'n/a'} "
                    f"anchor_q_updates={getattr(state, 'anchor_q_updates', 0)} "
                    f"anchor_q_broadcasts={getattr(state, 'anchor_q_broadcasts', 0)}",
                    flush=True,
                )
            if m_receipts:
                changed_m = sum(1 for r in m_receipts.values() if r.get("changed"))
                print(
                    f"[comm_eff][bcast] step={step} M broadcast targets={len(m_receipts)} changed={changed_m} "
                    f"(sign(M) is what the merger reads; receipt proves every DP rank holds the anchor M)",
                    flush=True,
                )

        # EXP-25 (R1, checklist #2): prove M is the GLOBAL gradient — bit-identical
        # across DP ranks. All-gather a per-target M checksum over the DP group and
        # assert the max cross-rank deviation is ~0 (the all-reduce(MEAN) of
        # G_anchor made M identical on every rank). A non-zero deviation means the
        # DP-reduce did not happen / used the wrong group. Greppable falsifier.
        self._verify_anchor_M_dp_identical(spectral, anchor_grads, step=step)

        # EMA-evolution log line. String discovery (ema_device/correction_mode)
        # is logged once at build, never here.
        if deltas:
            mean_delta = sum(deltas.values()) / len(deltas)
            max_delta = max(deltas.values())
            print(
                f"[comm_eff][EXP-12] anchor refresh step={step} fired backward "
                f"(cadence={cadence} delay_K={delay_K}) targets={len(deltas)} "
                f"||dM_anchor||_mean={mean_delta:.6e} ||dM_anchor||_max={max_delta:.6e} "
                f"anchor_backwards={state.anchor_backwards} "
                f"anchor_mask_applications={state.anchor_mask_applications} "
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
                f"[comm_eff][EXP-12] anchor refresh step={step} produced NO target grads "
                f"(targets matched=0); check target_substr / use_orig_params",
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

        spec_cfg = getattr(state.config, "spectral", None)
        target_substrs = self._comm_eff_target_names(spec_cfg)
        # EXP-25: full coverage default (-1). max_targets caps the merger too, so a
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

        # Expose original 2D named params and grads. FSDP1 can flatten wrapped
        # units into 1-D FlatParameters, so materialize original unflattened
        # params/grads via summon_full_params. FSDP2 keeps original names with
        # DTensor grads and can be iterated directly.
        if module_is_fsdp1 and not module_is_fsdp2:
            # with_grads=True surfaces the unsharded .grad on each original
            # param inside the context; writeback=True copies edits back into
            # the FlatParameter shard on exit. summon_full_params all-gathers,
            # so this is the unsharded full-matrix view the filter needs.
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
        max_targets,
        state,
        discovery_meta,
    ) -> int:
        """FSDP-agnostic core of the spectral grad-correction hook (CPU-testable).

        Iterates ``named_params`` (an iterator of ``(name, param)`` where each
        ``param`` exposes a full logical-2D ``.grad`` — a plain ``Tensor`` or a
        ``DTensor``), and for every targeted 2D matrix:

        * logs the FSDP gradient representation once, **regardless of gradient
          magnitude** — it fires on the
          first target with a non-``None`` grad even if that grad is ~0, so a
          degenerate-loss step still proves the hook ran;
        * applies the spectral filter and records the per-target
          ``||G_proj - G_mask|| / ||G_mask||`` ratio;
        * writes the corrected full matrix back into the (possibly sharded)
          grad in place and bumps ``state.spectral_corrections``.

        The iteration/discovery/correction loop itself lives in
        :func:`verl.workers.comm_eff.spectral_filter.apply_spectral_correction_to_params`
        (FSDP-agnostic, no torch.distributed) so it is exercised on CPU with no
        distributed runtime (see
        ``tests/workers/comm_eff/test_grad_correction_hook.py``). This method
        only supplies the two FSDP-specific callables — the DTensor unshard
        (``full_grad_of``) and the in-place writeback. Returns the number of
        matrices corrected.
        """
        from verl.workers.comm_eff.spectral_filter import apply_spectral_correction_to_params

        def full_grad_of(grad):
            # Present a full logical 2D matrix to the (FSDP-agnostic) filter.
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
            max_targets=max_targets,
            state=state,
            discovery_meta=discovery_meta,
            full_grad_of=full_grad_of,
            writeback=writeback,
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
            # comm_eff: set the per-token mask context for this micro-batch before
            # the forward fires the boundary hooks (no-op unless masking is live).
            self._comm_eff_maybe_set_mask_context(micro_batch, input_ids)
            # Bump the PowerSGD forward generation and stamp the step before the
            # boundary projection hooks fire (no-op unless powersgd is live).
            self._comm_eff_maybe_set_powersgd_context(micro_batch, input_ids)
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
