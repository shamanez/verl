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

"""Bounded, causal diagnostics for rank-1 weight forecasts.

The training worker may already have a newer generator snapshot in a replay
ring when it forecasts ``theta_hat[target_tick]``.  Reading that snapshot here
would make the diagnostic acausal, even if the values were never fed back into
training.  This module therefore has a deliberately narrow two-stage API:

``record_prediction``
    Retains a few deterministic scalar samples from the projected checkpoint
    and from the newest *admitted exact* checkpoint used as its stale baseline.

``resolve_exact``
    Compares those samples only when an exact delayed transfer carrying the
    identical target tick is handed to the probe later.

No full tensor, tensor view, RNG state, optimizer object, or model reference is
retained.  The probe is telemetry-only; callers decide whether and where to
surface the returned numeric record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import torch

__all__ = [
    "PROBE_ROLES",
    "Rank1ProjectionProbe",
    "Rank1ProjectionProbeError",
    "canonical_parameter_name",
    "deterministic_sample_indices",
    "projection_sample_metrics",
    "sample_tensor_values",
    "select_representative_qwen_tensors",
]


PROBE_ROLES = ("embedding", "decoder", "layer_norm", "final_norm")
_MAX_SAMPLES_PER_TENSOR = 64
_SAMPLE_HASH_DOMAIN = "rank1-projection-probe-v1"
_LAYER_Q_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.q_proj\.weight$")
_LAYER_NORM_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.input_layernorm\.weight$")


class Rank1ProjectionProbeError(RuntimeError):
    """A malformed or causally invalid rank-1 projection probe operation."""


def _nonnegative_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Rank1ProjectionProbeError(f"{label} must be an integer; got {value!r}")
    value = int(value)
    if value < 0:
        raise Rank1ProjectionProbeError(f"{label} must be >= 0; got {value}")
    return value


def _positive_int(value, label: str) -> int:
    value = _nonnegative_int(value, label)
    if value < 1:
        raise Rank1ProjectionProbeError(f"{label} must be >= 1; got {value}")
    return value


def canonical_parameter_name(name: str) -> str:
    """Remove FSDP wrapping infixes without changing the logical name."""

    if not isinstance(name, str) or not name:
        raise Rank1ProjectionProbeError(f"parameter name must be a non-empty string; got {name!r}")
    canonical = name.replace("._fsdp_wrapped_module", "")
    if canonical.startswith("_fsdp_wrapped_module."):
        canonical = canonical[len("_fsdp_wrapped_module.") :]
    return canonical


def _names_from(names_or_mapping: Mapping | Iterable[str]) -> list[str]:
    names = list(names_or_mapping.keys()) if isinstance(names_or_mapping, Mapping) else list(names_or_mapping)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise Rank1ProjectionProbeError("representative-tensor selection requires non-empty string parameter names")
    if len(names) != len(set(names)):
        raise Rank1ProjectionProbeError("representative-tensor selection received duplicate parameter names")
    return names


def _suffix_match(canonical: str, suffix: str) -> bool:
    return canonical == suffix or canonical.endswith(f".{suffix}")


def select_representative_qwen_tensors(names_or_mapping: Mapping | Iterable[str]) -> dict[str, str]:
    """Select stable embedding/decoder/norm representatives from Qwen names.

    The decoder and layer-norm representatives come from the median numbered
    decoder layer.  Qwen2.5-1.5B has layers 0..27, so this selects layer 14.
    Returned values are the original keys (including any FSDP infix) so they can
    index the exact snapshot dictionaries directly.
    """

    names = _names_from(names_or_mapping)
    canonical_to_raw: dict[str, str] = {}
    for raw_name in names:
        canonical = canonical_parameter_name(raw_name)
        if canonical in canonical_to_raw:
            raise Rank1ProjectionProbeError(
                f"multiple raw parameter names canonicalize to {canonical!r}: "
                f"{canonical_to_raw[canonical]!r}, {raw_name!r}"
            )
        canonical_to_raw[canonical] = raw_name

    embedding = sorted(
        (canonical, raw) for canonical, raw in canonical_to_raw.items() if canonical.endswith("embed_tokens.weight")
    )
    q_by_layer: dict[int, tuple[str, str]] = {}
    norm_by_layer: dict[int, tuple[str, str]] = {}
    final_norm = sorted(
        (canonical, raw) for canonical, raw in canonical_to_raw.items() if _suffix_match(canonical, "model.norm.weight")
    )
    for canonical, raw in canonical_to_raw.items():
        q_match = _LAYER_Q_RE.search(canonical)
        if q_match is not None:
            q_by_layer[int(q_match.group(1))] = (canonical, raw)
        norm_match = _LAYER_NORM_RE.search(canonical)
        if norm_match is not None:
            norm_by_layer[int(norm_match.group(1))] = (canonical, raw)

    missing = []
    if not embedding:
        missing.append("embedding (*embed_tokens.weight)")
    if not q_by_layer:
        missing.append("decoder (*.layers.N.self_attn.q_proj.weight)")
    if not norm_by_layer:
        missing.append("layer_norm (*.layers.N.input_layernorm.weight)")
    if not final_norm:
        missing.append("final_norm (*model.norm.weight)")
    if missing:
        raise Rank1ProjectionProbeError("missing representative Qwen parameter categories: " + ", ".join(missing))

    decoder_layers = sorted(q_by_layer)
    middle_layer = decoder_layers[len(decoder_layers) // 2]
    if middle_layer in norm_by_layer:
        norm_layer = middle_layer
    else:
        # A defensive fallback for architecture variants: stay as close as
        # possible to the selected decoder layer, with the lower layer winning
        # an exact-distance tie.
        norm_layer = min(norm_by_layer, key=lambda layer: (abs(layer - middle_layer), layer))

    # Multiple matches for an architecture-defining singleton are ambiguous;
    # silently choosing one could compare a wrapper alias rather than the
    # parameter the projected checkpoint actually loaded.
    if len(embedding) != 1:
        raise Rank1ProjectionProbeError(
            f"expected one Qwen embedding parameter; found {[name for name, _ in embedding]}"
        )
    if len(final_norm) != 1:
        raise Rank1ProjectionProbeError(
            f"expected one Qwen final norm parameter; found {[name for name, _ in final_norm]}"
        )

    return {
        "embedding": embedding[0][1],
        "decoder": q_by_layer[middle_layer][1],
        "layer_norm": norm_by_layer[norm_layer][1],
        "final_norm": final_norm[0][1],
    }


def deterministic_sample_indices(name: str, shape: Sequence[int], sample_count: int = 16) -> tuple[int, ...]:
    """Return stable, unique flat indices without touching any RNG."""

    canonical = canonical_parameter_name(name)
    sample_count = _positive_int(sample_count, "sample_count")
    if sample_count > _MAX_SAMPLES_PER_TENSOR:
        raise Rank1ProjectionProbeError(f"sample_count must be <= {_MAX_SAMPLES_PER_TENSOR}; got {sample_count}")
    try:
        clean_shape = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise Rank1ProjectionProbeError(f"shape must be an integer sequence; got {shape!r}") from exc
    if any(dim < 0 for dim in clean_shape):
        raise Rank1ProjectionProbeError(f"shape dimensions must be >= 0; got {clean_shape}")
    numel = math.prod(clean_shape)
    if numel < 1:
        raise Rank1ProjectionProbeError(f"cannot sample an empty tensor with shape {clean_shape}")

    count = min(sample_count, numel)
    payload = f"{_SAMPLE_HASH_DOMAIN}|{canonical}|{clean_shape}".encode()
    digest = hashlib.sha256(payload).digest()
    offset = int.from_bytes(digest[:8], "little") % numel
    if numel == 1:
        return (0,)
    stride = int.from_bytes(digest[8:16], "little") % numel
    if stride == 0:
        stride = 1
    while math.gcd(stride, numel) != 1:
        stride += 1
        if stride == numel:
            stride = 1
    indices = tuple((offset + i * stride) % numel for i in range(count))
    assert len(indices) == len(set(indices))
    return indices


def sample_tensor_values(tensor: torch.Tensor, indices: Sequence[int]) -> tuple[float, ...]:
    """Copy only ``indices`` from ``tensor`` into an immutable CPU tuple."""

    if not isinstance(tensor, torch.Tensor):
        raise Rank1ProjectionProbeError(f"sample source must be a torch.Tensor; got {type(tensor).__name__}")
    if not torch.is_floating_point(tensor):
        raise Rank1ProjectionProbeError(f"sample source must be floating point; got dtype={tensor.dtype}")
    if not tensor.is_contiguous():
        raise Rank1ProjectionProbeError("sample source must be contiguous; refusing a model-sized reshape copy")
    clean_indices = tuple(_nonnegative_int(index, f"sample index[{i}]") for i, index in enumerate(indices))
    if not clean_indices:
        raise Rank1ProjectionProbeError("at least one sample index is required")
    if len(clean_indices) != len(set(clean_indices)):
        raise Rank1ProjectionProbeError(f"sample indices must be unique; got {clean_indices}")
    if max(clean_indices) >= tensor.numel():
        raise Rank1ProjectionProbeError(
            f"sample index {max(clean_indices)} is out of bounds for tensor with {tensor.numel()} elements"
        )

    with torch.no_grad():
        flat = tensor.detach().view(-1)
        index_tensor = torch.tensor(clean_indices, dtype=torch.long, device=flat.device)
        sampled = torch.index_select(flat, 0, index_tensor).to(device="cpu", dtype=torch.float32)
        values = tuple(float(value) for value in sampled.tolist())
    if any(not math.isfinite(value) for value in values):
        raise Rank1ProjectionProbeError("sampled tensor values must all be finite")
    return values


def _finite_values(values: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        clean = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise Rank1ProjectionProbeError(f"{label} must contain numeric values") from exc
    if not clean or any(not math.isfinite(value) for value in clean):
        raise Rank1ProjectionProbeError(f"{label} must be non-empty and finite")
    return clean


def projection_sample_metrics(
    projected: Sequence[float],
    latest_exact: Sequence[float],
    actual: Sequence[float],
) -> dict[str, float | int | bool]:
    """Compare sampled forecast error against the no-projection stale baseline."""

    projected = _finite_values(projected, "projected samples")
    latest_exact = _finite_values(latest_exact, "latest-exact samples")
    actual = _finite_values(actual, "actual samples")
    if not (len(projected) == len(latest_exact) == len(actual)):
        raise Rank1ProjectionProbeError(
            "projected/latest-exact/actual sample counts must match; "
            f"got {len(projected)}, {len(latest_exact)}, {len(actual)}"
        )

    projected_error = tuple(pred - truth for pred, truth in zip(projected, actual, strict=True))
    stale_error = tuple(stale - truth for stale, truth in zip(latest_exact, actual, strict=True))
    predicted_update = tuple(pred - stale for pred, stale in zip(projected, latest_exact, strict=True))
    actual_update = tuple(truth - stale for truth, stale in zip(actual, latest_exact, strict=True))

    n = len(actual)
    projected_sse = math.fsum(value * value for value in projected_error)
    stale_sse = math.fsum(value * value for value in stale_error)
    actual_energy = math.fsum(value * value for value in actual)
    predicted_update_energy = math.fsum(value * value for value in predicted_update)
    actual_update_energy = math.fsum(value * value for value in actual_update)
    projected_rmse = math.sqrt(projected_sse / n)
    stale_rmse = math.sqrt(stale_sse / n)
    actual_l2 = math.sqrt(actual_energy)
    relative_floor = 1e-12

    if stale_sse == 0.0:
        # There is no stale error to improve upon. A zero-error projection ties
        # the baseline; a nonzero forecast is strictly worse. Keep the metric
        # finite for JSON/W&B rather than emitting an undefined ratio.
        skill = 0.0 if projected_sse == 0.0 else -1.0
    else:
        skill = 1.0 - projected_sse / stale_sse

    if predicted_update_energy == 0.0 and actual_update_energy == 0.0:
        direction_cos = 1.0
    elif predicted_update_energy == 0.0 or actual_update_energy == 0.0:
        direction_cos = 0.0
    else:
        dot = math.fsum(pred * truth for pred, truth in zip(predicted_update, actual_update, strict=True))
        direction_cos = dot / math.sqrt(predicted_update_energy * actual_update_energy)
        direction_cos = max(-1.0, min(1.0, direction_cos))

    metrics: dict[str, float | int | bool] = {
        "sample_count": n,
        "projected_rmse": projected_rmse,
        "stale_rmse": stale_rmse,
        "projected_mae": math.fsum(abs(value) for value in projected_error) / n,
        "stale_mae": math.fsum(abs(value) for value in stale_error) / n,
        "projected_relative_l2": math.sqrt(projected_sse) / max(actual_l2, relative_floor),
        "stale_relative_l2": math.sqrt(stale_sse) / max(actual_l2, relative_floor),
        "predicted_update_l2": math.sqrt(predicted_update_energy),
        "actual_update_l2": math.sqrt(actual_update_energy),
        "skill": skill,
        "direction_cos": direction_cos,
        "projection_beats_stale": projected_sse < stale_sse,
    }
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise Rank1ProjectionProbeError(f"projection sample metric {key!r} is non-finite")
    return metrics


@dataclass(frozen=True)
class _TensorPrediction:
    role: str
    name: str
    canonical_name: str
    shape: tuple[int, ...]
    indices: tuple[int, ...]
    projected: tuple[float, ...]
    latest_exact: tuple[float, ...]


@dataclass(frozen=True)
class _PendingPrediction:
    fire_step: int
    target_tick: int
    source_tick: int
    history_ticks: tuple[int, ...]
    tensors: tuple[_TensorPrediction, ...]


class Rank1ProjectionProbe:
    """Retain bounded scalar forecasts and resolve them on delayed arrival."""

    schema = "rank1_projection_probe/v1"

    def __init__(
        self,
        *,
        samples_per_tensor: int = 16,
        max_pending: int = 2,
        out_path: Optional[str | os.PathLike] = None,
        writer: bool = True,
    ):
        self.samples_per_tensor = _positive_int(samples_per_tensor, "samples_per_tensor")
        if self.samples_per_tensor > _MAX_SAMPLES_PER_TENSOR:
            raise Rank1ProjectionProbeError(
                f"samples_per_tensor must be <= {_MAX_SAMPLES_PER_TENSOR}; got {self.samples_per_tensor}"
            )
        self.max_pending = _positive_int(max_pending, "max_pending")
        if not isinstance(writer, bool):
            raise Rank1ProjectionProbeError(f"writer must be a bool; got {writer!r}")
        self.writer = writer
        self.out_path = os.fspath(out_path) if out_path is not None else None
        self._pending: OrderedDict[int, _PendingPrediction] = OrderedDict()
        self.predictions_recorded = 0
        self.resolutions_completed = 0
        self.last_record: Optional[dict] = None

    @property
    def pending_ticks(self) -> tuple[int, ...]:
        return tuple(self._pending)

    @property
    def retained_scalar_count(self) -> int:
        """Number of forecast/baseline scalars retained across all targets."""

        return sum(
            len(tensor.projected) + len(tensor.latest_exact)
            for pending in self._pending.values()
            for tensor in pending.tensors
        )

    def status(self) -> dict[str, int]:
        return {
            "predictions_recorded": int(self.predictions_recorded),
            "resolutions_completed": int(self.resolutions_completed),
            "pending": len(self._pending),
            "retained_scalars": self.retained_scalar_count,
        }

    def record_prediction(
        self,
        *,
        fire_step: int,
        target_tick: int,
        source_tick: int,
        history_ticks: Sequence[int],
        projected: Mapping[str, torch.Tensor],
        latest_exact: Mapping[str, torch.Tensor],
    ) -> dict:
        """Store tiny forecast and stale-baseline samples for a future check."""

        fire_step = _nonnegative_int(fire_step, "fire_step")
        target_tick = _nonnegative_int(target_tick, "target_tick")
        source_tick = _nonnegative_int(source_tick, "source_tick")
        clean_history = tuple(_nonnegative_int(tick, f"history_tick[{i}]") for i, tick in enumerate(history_ticks))
        if not clean_history or any(
            right <= left for left, right in zip(clean_history, clean_history[1:], strict=False)
        ):
            raise Rank1ProjectionProbeError(
                f"history_ticks must be non-empty and strictly increasing; got {clean_history}"
            )
        if clean_history[-1] != source_tick:
            raise Rank1ProjectionProbeError(
                f"source_tick={source_tick} must equal newest exact history tick={clean_history[-1]}"
            )
        if target_tick <= source_tick:
            raise Rank1ProjectionProbeError(f"target_tick={target_tick} must be newer than source_tick={source_tick}")
        if target_tick in self._pending:
            raise Rank1ProjectionProbeError(f"target_tick={target_tick} already has a pending forecast")
        if len(self._pending) >= self.max_pending:
            raise Rank1ProjectionProbeError(
                f"pending forecast bound reached ({len(self._pending)}/{self.max_pending}); "
                f"unresolved_ticks={list(self._pending)}"
            )
        if not isinstance(projected, Mapping) or not projected:
            raise Rank1ProjectionProbeError("projected checkpoint must be a non-empty mapping")
        if not isinstance(latest_exact, Mapping) or not latest_exact:
            raise Rank1ProjectionProbeError("latest_exact checkpoint must be a non-empty mapping")

        selected = select_representative_qwen_tensors(projected)
        tensor_predictions = []
        selected_summary = {}
        for role in PROBE_ROLES:
            name = selected[role]
            predicted_tensor = projected.get(name)
            stale_tensor = latest_exact.get(name)
            if not isinstance(predicted_tensor, torch.Tensor) or not isinstance(stale_tensor, torch.Tensor):
                raise Rank1ProjectionProbeError(
                    f"representative {role} parameter {name!r} must exist as a tensor in both checkpoints"
                )
            if predicted_tensor.shape != stale_tensor.shape:
                raise Rank1ProjectionProbeError(
                    f"representative {role} parameter {name!r} shape mismatch: "
                    f"projected={tuple(predicted_tensor.shape)} latest_exact={tuple(stale_tensor.shape)}"
                )
            shape = tuple(predicted_tensor.shape)
            indices = deterministic_sample_indices(name, shape, self.samples_per_tensor)
            canonical = canonical_parameter_name(name)
            tensor_predictions.append(
                _TensorPrediction(
                    role=role,
                    name=name,
                    canonical_name=canonical,
                    shape=shape,
                    indices=indices,
                    projected=sample_tensor_values(predicted_tensor, indices),
                    latest_exact=sample_tensor_values(stale_tensor, indices),
                )
            )
            selected_summary[role] = canonical

        self._pending[target_tick] = _PendingPrediction(
            fire_step=fire_step,
            target_tick=target_tick,
            source_tick=source_tick,
            history_ticks=clean_history,
            tensors=tuple(tensor_predictions),
        )
        self.predictions_recorded += 1
        return {
            "fire_step": fire_step,
            "target_tick": target_tick,
            "source_tick": source_tick,
            "history_ticks": list(clean_history),
            "selected": selected_summary,
            "samples_per_tensor": self.samples_per_tensor,
            "pending": len(self._pending),
        }

    def resolve_exact(
        self,
        *,
        resolve_step: int,
        exact_tick: int,
        exact_snapshot: Mapping[str, torch.Tensor],
    ) -> Optional[dict]:
        """Resolve only a forecast whose exact delayed target just arrived."""

        resolve_step = _nonnegative_int(resolve_step, "resolve_step")
        exact_tick = _nonnegative_int(exact_tick, "exact_tick")
        overdue = [target for target in self._pending if target < exact_tick]
        if overdue:
            raise Rank1ProjectionProbeError(
                f"exact transfer tick={exact_tick} skipped unresolved forecast target(s) {overdue}"
            )
        if exact_tick not in self._pending:
            return None
        if not isinstance(exact_snapshot, Mapping) or not exact_snapshot:
            raise Rank1ProjectionProbeError("exact_snapshot must be a non-empty mapping")

        pending = self._pending[exact_tick]
        if resolve_step <= pending.fire_step:
            raise Rank1ProjectionProbeError(
                f"target_tick={exact_tick} must arrive after prediction fire_step={pending.fire_step}; "
                f"got resolve_step={resolve_step}"
            )

        tensor_records = []
        aggregate_projected: list[float] = []
        aggregate_latest: list[float] = []
        aggregate_actual: list[float] = []
        for tensor_prediction in pending.tensors:
            exact_tensor = exact_snapshot.get(tensor_prediction.name)
            if not isinstance(exact_tensor, torch.Tensor):
                raise Rank1ProjectionProbeError(
                    f"exact checkpoint tick={exact_tick} is missing representative parameter {tensor_prediction.name!r}"
                )
            if tuple(exact_tensor.shape) != tensor_prediction.shape:
                raise Rank1ProjectionProbeError(
                    f"exact checkpoint tick={exact_tick} parameter {tensor_prediction.name!r} shape mismatch: "
                    f"expected={tensor_prediction.shape} got={tuple(exact_tensor.shape)}"
                )
            actual = sample_tensor_values(exact_tensor, tensor_prediction.indices)
            metrics = projection_sample_metrics(
                tensor_prediction.projected,
                tensor_prediction.latest_exact,
                actual,
            )
            aggregate_projected.extend(tensor_prediction.projected)
            aggregate_latest.extend(tensor_prediction.latest_exact)
            aggregate_actual.extend(actual)
            tensor_records.append(
                {
                    "role": tensor_prediction.role,
                    "name": tensor_prediction.canonical_name,
                    "shape": list(tensor_prediction.shape),
                    "flat_indices": list(tensor_prediction.indices),
                    "projected": list(tensor_prediction.projected),
                    "latest_exact": list(tensor_prediction.latest_exact),
                    "actual": list(actual),
                    "metrics": metrics,
                }
            )

        aggregate = projection_sample_metrics(aggregate_projected, aggregate_latest, aggregate_actual)
        record = {
            "schema": self.schema,
            "prediction_fire_step": pending.fire_step,
            "resolution_step": resolve_step,
            "wait_steps": resolve_step - pending.fire_step,
            "target_tick": pending.target_tick,
            "exact_transfer_tick": exact_tick,
            "source_tick": pending.source_tick,
            "prediction_horizon": pending.target_tick - pending.source_tick,
            "history_ticks": list(pending.history_ticks),
            "tensor_count": len(tensor_records),
            "samples_per_tensor": self.samples_per_tensor,
            "aggregate": aggregate,
            "tensors": tensor_records,
        }
        # Prove serializability and finiteness even when file output is off.
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        if self.writer and self.out_path is not None:
            parent = os.path.dirname(os.path.abspath(self.out_path))
            os.makedirs(parent, exist_ok=True)
            with open(self.out_path, "a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        del self._pending[exact_tick]
        self.resolutions_completed += 1
        self.last_record = record
        return record
