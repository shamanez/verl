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

"""Per-worker state object for the communication-efficient compression method.

The integration contract is deliberately asymmetric so the disabled path is a
**strict no-op**:

* ``maybe_build_comm_eff_state(config)`` returns ``None`` when
  ``config.enabled`` is false (or the config is absent). No object is
  constructed, **no RNG is drawn**, no buffer is allocated, no forward hook is
  registered. The actor therefore holds ``self._comm_eff_state = None`` for a
  dense GRPO run, and every hook below short-circuits on the ``None`` check.

* Only when ``config.enabled`` is true is a ``CommEffState`` constructed; that
  is where mask RNG, anchor EMA buffers and the spectral workspace get
  allocated (lazily, by ``build()``, which later M2 work fills in).

Because construction is gated, a dense GRPO run with this scaffolding merged
consumes the exact same RNG sequence and issues the exact same collective ops
as one without it — the criterion-7 rel-tol-1e-4 parity check holds.

The instrumented counters (``mask_applications``, ``anchor_backwards``,
``spectral_corrections``) live on the state object. When disabled there is no
state object, so the counters are *absent* rather than zero — which the
analyst treats as equivalent to ``== 0`` (no comm_eff op fired). When enabled
they start at 0 and increment per fired op.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid an import cycle at runtime; only needed for type hints
    from verl.workers.config.comm_eff import CommEffConfig

logger = logging.getLogger(__name__)

__all__ = ["CommEffState", "maybe_build_comm_eff_state", "comm_eff_metrics"]


def _is_enabled(config: Any) -> bool:
    """Read the ``enabled`` flag from a comm_eff config that may be a dataclass,
    an OmegaConf node, a plain dict, or ``None``. Pure read — no side effects."""
    if config is None:
        return False
    if isinstance(config, dict):
        return bool(config.get("enabled", False))
    return bool(getattr(config, "enabled", False))


class CommEffState:
    """Per-worker communication-efficient compression state.

    Constructed **only** when ``comm_eff.enabled=true``. Holds the operation
    counters and (once ``build()`` is implemented by later M2 work) the mask
    RNG generator, anchor EMA buffers and spectral workspace. The disabled path
    never instantiates this class — see ``maybe_build_comm_eff_state``.
    """

    def __init__(self, config: "CommEffConfig"):
        # Invariant: never construct a disabled state. The factory enforces it;
        # this assert catches a future caller that forgets to go through it.
        assert _is_enabled(config), (
            "CommEffState must not be constructed when comm_eff.enabled=false; "
            "go through maybe_build_comm_eff_state() so the disabled path stays a no-op."
        )
        self.config = config
        self.enabled = True
        self._built = False

        # Operation counters surfaced into training metrics under comm_eff/*.
        self.mask_applications = 0
        self.anchor_backwards = 0
        self.spectral_corrections = 0

        # The activation masker (first circuit). Constructed in build(); None
        # when the mask sub-config is disabled.
        self.masker = None

        # Whether masking is currently active. Set True only on entry to the
        # actor-train forward/backward (around update_actor) and cleared on
        # exit, so log-prob / ref / infer / val / checkpoint forwards stay clean.
        self.mask_active = False

        # Monotonic optimizer-substep counter (microbatch identity for the PRF
        # key). A trainer step reuses one rollout batch over multiple PPO
        # mini-batches, so this advances per actor optimizer substep, giving
        # each substep a distinct mask even within the same trainer step.
        self.substep = 0

    def build(self, module: Any) -> None:
        """Construct the activation masker for the enabled mask circuit.

        Idempotent. When ``comm_eff.mask.enabled`` is true this constructs an
        ``ActivationMasker`` (no hooks registered yet — the engine registers
        them only on entry to the train forward and removes them on exit). Anchor
        / spectral workspace allocation is deferred to later M2 work.
        """
        if self._built:
            return
        mask_cfg = getattr(self.config, "mask", None)
        mask_enabled = bool(getattr(mask_cfg, "enabled", False)) if mask_cfg is not None else False
        if mask_enabled and float(getattr(mask_cfg, "p", 0.0)) > 0.0:
            # Imported lazily so the disabled path never pays the import cost.
            from verl.workers.comm_eff.activation_mask import ActivationMasker

            self.masker = ActivationMasker(
                p=float(mask_cfg.p),
                base_seed=int(getattr(mask_cfg, "seed", 0)),
                pp_size=int(getattr(mask_cfg, "pp_size", 8)),
                state=self,
            )
        self._built = True

    def mask_ratio_metrics(self) -> dict:
        """Return the most-recently-measured masked fraction per boundary layer.

        Surfaced as ``comm_eff/mask_ratio`` (mean across boundaries) plus a
        per-boundary breakdown. Empty when no mask fired this step.
        """
        if self.masker is None or not self.masker.last_mask_ratio:
            return {}
        ratios = self.masker.last_mask_ratio
        mean_ratio = sum(ratios.values()) / len(ratios)
        out = {"comm_eff/mask_ratio": mean_ratio}
        for idx, r in sorted(ratios.items()):
            out[f"comm_eff/mask_ratio/layer_{idx}"] = r
        return out

    def metrics(self) -> dict:
        """Return the comm_eff operation counters for logging."""
        return {
            "comm_eff/mask_applications": self.mask_applications,
            "comm_eff/anchor_backwards": self.anchor_backwards,
            "comm_eff/spectral_corrections": self.spectral_corrections,
        }


def maybe_build_comm_eff_state(config: Any) -> Optional[CommEffState]:
    """Construct a ``CommEffState`` iff comm_eff is enabled, else return ``None``.

    This is the single gate that guarantees the disabled path is inert: when
    ``config.enabled`` is false (or ``config`` is ``None``/absent) it returns
    ``None`` **without drawing RNG, allocating buffers or registering hooks**.
    Callers store the result and guard every comm_eff op behind a ``None`` /
    ``state.enabled`` check.
    """
    if not _is_enabled(config):
        return None
    state = CommEffState(config)
    logger.info("comm_eff: enabled — constructed CommEffState")
    return state


def comm_eff_metrics(state: Optional[CommEffState]) -> dict:
    """Return comm_eff counters for ``state``, or an empty dict when disabled.

    Centralises the "disabled means no counters" convention so call sites do
    not each re-derive it. Includes the measured ``comm_eff/mask_ratio`` when a
    mask fired this step.
    """
    if state is None:
        return {}
    out = state.metrics()
    out.update(state.mask_ratio_metrics())
    return out
