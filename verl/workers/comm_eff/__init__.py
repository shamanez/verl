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

"""Communication-efficient GRPO worker package.

The retained pipeline combines rank-77 PowerSGD activation projection, a
delayed paired dense anchor, signed-EMA gradient correction, and rank-1 RELEX
weight projection. It is disabled by default; see
``verl.workers.config.comm_eff.CommEffConfig``.

Only ``state`` is imported eagerly. Runtime kernels are imported lazily and
run only when ``comm_eff.enabled=true``.
"""

from .state import (
    comm_eff_metrics,
    maybe_build_comm_eff_state,
)

__all__ = [
    "comm_eff_metrics",
    "maybe_build_comm_eff_state",
]
