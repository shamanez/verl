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

"""Decoder discovery and pipeline-boundary selection helpers."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

__all__ = ["decoder_boundary_indices", "find_decoder_layers"]


def decoder_boundary_indices(num_layers: int, pp_size: int) -> list[int]:
    """Return the last decoder-block index in every non-final pipeline shard."""

    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1; got {num_layers}")
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1; got {pp_size}")

    # Cap pp_size at num_layers so we never produce an empty shard.
    effective_pp_size = min(pp_size, num_layers)
    if effective_pp_size == 1:
        return []

    base = num_layers // effective_pp_size
    remainder = num_layers % effective_pp_size
    boundaries: list[int] = []
    cursor = 0
    for shard_index in range(effective_pp_size):
        shard_length = base + (1 if shard_index < remainder else 0)
        cursor += shard_length
        if shard_index < effective_pp_size - 1:
            boundaries.append(cursor - 1)
    return boundaries


def find_decoder_layers(module: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the decoder-block ``ModuleList`` inside a wrapped causal LM."""

    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.ModuleList) and len(submodule) > 1:
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("layers", "h", "blocks", "decoder"):
                candidates.append((name, submodule))

    if not candidates:
        candidates = [
            (name, submodule)
            for name, submodule in module.named_modules()
            if isinstance(submodule, nn.ModuleList) and len(submodule) > 1
        ]
    if not candidates:
        return None

    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    return candidates[0][1]
