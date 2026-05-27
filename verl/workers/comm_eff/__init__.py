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

"""Communication-efficient pipeline-adaptation worker package.

The two-circuit compression method (activation masking + async unmasked anchor
circuit + spectral gradient correction). Disabled by default; see
``verl.workers.config.comm_eff.CommEffConfig``.

Only ``state`` is imported eagerly because it is cheap (no torch heavy lifting
at import time). The actual masking / anchor / spectral kernels are imported
lazily by ``CommEffState.maybe_build`` and run **only** when
``comm_eff.enabled=true``; the disabled path never reaches them.
"""

from .state import CommEffState, maybe_build_comm_eff_state

__all__ = ["CommEffState", "maybe_build_comm_eff_state"]
