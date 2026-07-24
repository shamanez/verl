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

"""Dense-view probe metrics + adaptive reference-KL coefficient controller.

Issue #93 sections 4.5-4.6 (I3). The trainer periodically reruns one step's
log-probability computations with every comm_eff codec silent (the dense
view), turns the result into ``probe/*`` metrics, and optionally closes the
loop: a projected dual-ascent controller in log space retunes the
reference-KL coefficient ``beta`` toward a dense-KL setpoint derived from the
finished dense-control run.

This module is import-light on purpose: ``verl.workers.config.comm_eff``
validates the setpoint table through :func:`parse_kl_target_table`, so heavy
imports (torch, core_algos) stay inside the functions that need them.
"""

import math

__all__ = [
    "parse_kl_target_table",
    "interp_kl_target_table",
    "should_probe",
    "DenseKLCoefController",
    "LRBrakeDetector",
    "compute_probe_metrics",
]


def should_probe(probe_every: int, global_step: int) -> bool:
    """True iff the dense-view probe fires at this trainer step.

    ``probe_every <= 0`` is the strict off default: the trainer's probe hook
    returns before touching the batch, the workers, or the metrics dict.
    """
    probe_every = int(probe_every or 0)
    return probe_every > 0 and global_step % probe_every == 0


def parse_kl_target_table(spec: str) -> list[tuple[int, float]]:
    """Parse a ``"step:value,step:value"`` dense reference-KL table.

    Returns ``[(step, value), ...]`` sorted by step. An empty/whitespace spec
    returns ``[]`` (no table; the setpoint floor applies alone). Raises
    ``ValueError`` on malformed entries, duplicate steps, negative steps, or
    non-finite/negative values.
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    table: list[tuple[int, float]] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            raise ValueError(f"empty entry in kl target table {spec!r}")
        step_s, sep, value_s = entry.partition(":")
        if not sep:
            raise ValueError(f"kl target table entry {entry!r} is not 'step:value'")
        try:
            step = int(step_s)
            value = float(value_s)
        except ValueError:
            raise ValueError(f"kl target table entry {entry!r} is not 'int:float'") from None
        if step < 0:
            raise ValueError(f"kl target table step must be >= 0; got {step}")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"kl target table value must be finite and >= 0; got {value}")
        table.append((step, value))
    table.sort(key=lambda sv: sv[0])
    for (a, _), (b, _) in zip(table, table[1:], strict=False):
        if a == b:
            raise ValueError(f"duplicate step {a} in kl target table {spec!r}")
    return table


def interp_kl_target_table(table: list[tuple[int, float]], step: int) -> float:
    """Linear interpolation with edge clamping; 0.0 for an empty table."""
    if not table:
        return 0.0
    if step <= table[0][0]:
        return table[0][1]
    if step >= table[-1][0]:
        return table[-1][1]
    for (s0, v0), (s1, v1) in zip(table, table[1:], strict=False):
        if s0 <= step <= s1:
            frac = (step - s0) / float(s1 - s0)
            return v0 + frac * (v1 - v0)
    raise AssertionError("unreachable: table sorted with clamped edges")


class DenseKLCoefController:
    """Projected dual ascent in log space with proportional damping (#93 4.6).

    Once per probe::

        e_k      = (probe_kl_dense - c_k) / c_k
        beta_new = clip(beta * exp(ki * e_k + kp * (e_k - e_prev)),
                        beta_min, beta_max)

    with setpoint ``c_k = max(c_floor, gain * table(step))``. ``beta`` itself
    is the integral (dual) state; anti-windup is the standard
    conditional-integration form: while ``beta`` is pinned at a bound AND the
    error would drive it further into that bound, the integral term ``ki*e_k``
    is skipped, so only the proportional term acts and the first error-sign
    flip pulls ``beta`` off the bound immediately (no accumulated saturation
    to unwind). ``e_prev`` always tracks the latest error so the damping term
    stays well defined; it starts at 0.0 (first update is purely integral for
    an on-setpoint start).
    """

    # Relative tolerance for "pinned at a bound" (survives exp/log round-trips).
    _BOUND_RTOL = 1e-9

    def __init__(
        self,
        beta0: float,
        ki: float = 0.3,
        kp: float = 0.1,
        beta_min: float = 2.0e-4,
        beta_max: float = 0.05,
        c_floor: float = 0.005,
        gain: float = 2.0,
        table: list[tuple[int, float]] | None = None,
    ):
        if not 0.0 < beta_min <= beta_max:
            raise ValueError(f"require 0 < beta_min <= beta_max; got [{beta_min}, {beta_max}]")
        if c_floor <= 0.0:
            raise ValueError(f"require c_floor > 0 (setpoint divides the error); got {c_floor}")
        self.ki = float(ki)
        self.kp = float(kp)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.c_floor = float(c_floor)
        self.gain = float(gain)
        self.table = list(table or [])
        self.beta = min(max(float(beta0), self.beta_min), self.beta_max)
        self.e_prev = 0.0

    @property
    def at_max(self) -> bool:
        return self.beta >= self.beta_max * (1.0 - self._BOUND_RTOL)

    @property
    def at_min(self) -> bool:
        return self.beta <= self.beta_min * (1.0 + self._BOUND_RTOL)

    def setpoint(self, step: int) -> float:
        return max(self.c_floor, self.gain * interp_kl_target_table(self.table, step))

    def update(self, kl_dense: float, step: int) -> float:
        """One dual-ascent update; returns the new beta.

        A non-finite or non-positive probe reading is a measurement failure,
        not a control signal: the state (beta, e_prev) is left untouched.
        """
        kl_dense = float(kl_dense)
        if not math.isfinite(kl_dense) or kl_dense <= 0.0:
            return self.beta
        c = self.setpoint(step)
        e = (kl_dense - c) / c
        integral = self.ki * e
        if (self.at_max and e > 0.0) or (self.at_min and e < 0.0):
            integral = 0.0
        proportional = self.kp * (e - self.e_prev)
        # math.exp overflows past ~709 nats; the usable beta range spans only
        # ln(beta_max/beta_min) nats, so this clamp is pure overflow armor and
        # never changes the projected result.
        exponent = min(max(integral + proportional, -60.0), 60.0)
        beta_new = self.beta * math.exp(exponent)
        self.beta = min(max(beta_new, self.beta_min), self.beta_max)
        self.e_prev = e
        return self.beta


class LRBrakeDetector:
    """Dormant LR emergency brake: DETECTION ONLY in this build (#93 4.6).

    Logs ``probe/lr_brake_triggered = 1.0`` when either

    * kl_dense doubled across two consecutive probes while beta is pinned at
      beta_max (the controller has no authority left), or
    * the gap_dense least-squares slope over the last ``window`` probes
      exceeds ``slope_ratio`` x the (positive) slope over the previous
      ``window`` probes (creep acceleration).

    It deliberately does NOT mutate the LR: halving the LR mid-run changes
    the optimizer trajectory under test, so round B first measures how often
    the brake WOULD fire before any mutation is allowed (explicit opt-in in a
    later build). Non-finite readings are recorded as gaps in the history
    (they reset consecutiveness) rather than fabricated values.
    """

    def __init__(self, double_ratio: float = 2.0, window: int = 4, slope_ratio: float = 3.0):
        self.double_ratio = float(double_ratio)
        self.window = int(window)
        self.slope_ratio = float(slope_ratio)
        self.kl_history: list[float] = []
        self.gap_history: list[float] = []

    @staticmethod
    def _ls_slope(values: list[float]) -> float:
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den

    def observe(self, kl_dense: float, gap_dense: float, beta_at_max: bool) -> bool:
        kl_dense = float(kl_dense)
        gap_dense = float(gap_dense)
        if math.isfinite(kl_dense):
            self.kl_history.append(kl_dense)
        else:
            self.kl_history.clear()
        if math.isfinite(gap_dense):
            self.gap_history.append(gap_dense)
        else:
            self.gap_history.clear()

        doubled = (
            len(self.kl_history) >= 2
            and self.kl_history[-2] > 0.0
            and self.kl_history[-1] >= self.double_ratio * self.kl_history[-2]
        )
        if doubled and beta_at_max:
            return True

        w = self.window
        if len(self.gap_history) >= 2 * w:
            recent = self._ls_slope(self.gap_history[-w:])
            previous = self._ls_slope(self.gap_history[-2 * w : -w])
            # Ratio tests need a positive baseline: a flat/negative previous
            # slope makes "3x" meaningless and would trip on noise.
            if previous > 0.0 and recent > self.slope_ratio * previous:
                return True
        return False


def compute_probe_metrics(
    *,
    dense_log_probs,
    dense_ref_log_probs,
    rollout_log_probs,
    response_mask,
    kl_loss_last,
    kl_loss_type: str = "low_var_kl",
    loss_agg_mode: str = "token-mean",
    loss_scale_factor=None,
) -> dict:
    """Turn one dense-view probe pass into the ``probe/*`` metrics dict.

    All tensors are padded ``(bsz, response_length)`` driver-side tensors.
    ``dense_ref_log_probs`` / ``rollout_log_probs`` may be ``None`` (no
    reference policy / ``calculate_log_probs`` off); the dependent metrics
    are then logged as NaN so the WandB panel keeps one schema.

    NOTE the one-step offset: the trainer probes AFTER this step's actor
    update, so probe/kl_dense measures theta_{t+1} while the actor/kl_loss
    it is ratioed against in probe/kl_gain was measured on theta_t during
    the update. At probe cadences >= 10 steps the offset is well inside the
    curve-reading noise; it is documented rather than corrected.

    probe/kl_dense uses the same estimator + aggregation as actor/kl_loss
    (``kl_penalty`` with the actor's ``kl_loss_type`` then ``agg_loss`` with
    the actor's ``loss_agg_mode``), so kl_gain is an apples-to-apples G(t).
    probe/gap_dense = token-mean(rollout_log_probs - dense actor log probs),
    the dense-view train-inference gap, directly comparable to the dense
    control's settled floor.
    """
    import torch

    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
    from verl.utils.torch_functional import masked_mean

    nan = float("nan")
    response_mask = response_mask.to(bool)

    kl_dense = nan
    if dense_ref_log_probs is not None:
        with torch.no_grad():
            kld = kl_penalty(logprob=dense_log_probs, ref_logprob=dense_ref_log_probs, kl_penalty=kl_loss_type)
            kl_dense = float(
                agg_loss(
                    loss_mat=kld,
                    loss_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                    loss_scale_factor=loss_scale_factor,
                ).item()
            )

    kl_gain = nan
    if kl_loss_last is not None and math.isfinite(kl_dense) and kl_dense > 0.0:
        kl_loss_last = float(kl_loss_last)
        if math.isfinite(kl_loss_last):
            kl_gain = kl_loss_last / kl_dense

    gap_dense = nan
    if rollout_log_probs is not None:
        with torch.no_grad():
            gap_dense = float(masked_mean(rollout_log_probs - dense_log_probs, response_mask).item())

    return {
        "probe/kl_dense": kl_dense,
        "probe/kl_gain": kl_gain,
        "probe/gap_dense": gap_dense,
    }
