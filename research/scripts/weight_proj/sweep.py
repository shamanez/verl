#!/usr/bin/env python3
"""weight_proj/sweep.py — grouping + the (family x order x coeff x Delta x h) driver.

ONE streaming pass discipline (r2-access-pattern-for-analysis.md): blocks/layers on
the OUTSIDE, ticks on the INSIDE. In this implementation we do the dual that is
strictly-bounded AND single-pass: iterate ticks ONCE (inside), loading each .pt
exactly once and fanning its per-matrix slices into ALL grouping accumulators; then
run the (family x coeff x Delta x h) predictor cross-product on the accumulated
per-group history vectors. Because the model is 1.5B params, the whole per-tick
fp32 state (~6 GB) fits in RAM for the sampled tick window, and we only ever hold
the requested matrices — never the full 494 GB on disk.

Grouping (## Success criteria):
  * per-matrix: 338 tensors
  * per-block : 11 families {q_proj,k_proj,v_proj,o_proj, gate/up/down_proj,
                input_layernorm, post_attention_layernorm, embed, norm}
                (q/k/v carry weight+bias; o weight only; embed tied to lm_head)
  * per-layer : 28 decoder layers (0..27)
The three aggregations each PARTITION the 338 matrices with none dropped/double-counted.
"""
from __future__ import annotations

import re
import numpy as np
import torch

from . import metrics as M
from . import noise_floor as NF


# =============================================================================
# Grouping labels derived from manifest matrix names
# =============================================================================
_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)")


def block_family(name: str) -> str:
    """11-family block label. q/k/v/o proj, gate/up/down proj, 2 layernorms, embed, norm."""
    if name == "model.embed_tokens.weight":
        return "embed"
    if name == "model.norm.weight":
        return "norm"
    m = _LAYER_RE.match(name)
    assert m, f"ungrouped matrix name: {name}"
    tail = m.group(2)  # e.g. self_attn.q_proj.weight / mlp.down_proj.weight / input_layernorm.weight
    if tail.startswith("self_attn.q_proj"):
        return "q_proj"
    if tail.startswith("self_attn.k_proj"):
        return "k_proj"
    if tail.startswith("self_attn.v_proj"):
        return "v_proj"
    if tail.startswith("self_attn.o_proj"):
        return "o_proj"
    if tail.startswith("mlp.gate_proj"):
        return "gate_proj"
    if tail.startswith("mlp.up_proj"):
        return "up_proj"
    if tail.startswith("mlp.down_proj"):
        return "down_proj"
    if tail.startswith("input_layernorm"):
        return "input_layernorm"
    if tail.startswith("post_attention_layernorm"):
        return "post_attention_layernorm"
    raise AssertionError(f"unrecognized block tail: {tail} (from {name})")


def layer_index(name: str):
    """Decoder layer index 0..27, or None for embed/norm (layer-agnostic groups)."""
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def build_grouping(matrix_names: list[str]) -> dict:
    """Return {'matrix':{name:[name]}, 'block':{fam:[names]}, 'layer':{idx:[names]}} +
    a partition-integrity report proving each aggregation covers all 338 exactly once."""
    per_matrix = {n: [n] for n in matrix_names}
    per_block: dict[str, list] = {}
    per_layer: dict = {}
    for n in matrix_names:
        per_block.setdefault(block_family(n), []).append(n)
        li = layer_index(n)
        key = li if li is not None else ("embed" if n == "model.embed_tokens.weight" else "norm")
        per_layer.setdefault(key, []).append(n)
    # partition integrity
    def _covers(groups):
        seen = []
        for names in groups.values():
            seen.extend(names)
        return len(seen) == len(matrix_names) and set(seen) == set(matrix_names) and len(seen) == len(set(seen))
    integrity = {
        "n_matrices": len(matrix_names),
        "n_blocks": len(per_block),
        "n_layers": len([k for k in per_layer if isinstance(k, int)]),
        "matrix_partition_ok": _covers(per_matrix),
        "block_partition_ok": _covers(per_block),
        "layer_partition_ok": _covers(per_layer),
        "block_families": sorted(per_block.keys()),
    }
    return {"matrix": per_matrix, "block": per_block, "layer": per_layer, "integrity": integrity}


# =============================================================================
# Group-vector accumulation across a sampled tick window (streamed once)
# =============================================================================
def concat_group(sd: dict, names: list[str]) -> torch.Tensor:
    """Concatenate the flattened fp32 vectors of `names` into ONE group vector."""
    return torch.cat([sd[n].to(torch.float32).reshape(-1) for n in names if n in sd])


def stream_group_histories(stream, ticks: list[int], names_needed: list[str]) -> dict:
    """ONE streaming pass: load each tick once, keep only `names_needed` as fp32.
    Returns {tick: {name: fp32 tensor}} for the sampled window (bounded, RAM-held)."""
    hist = {}
    for t in ticks:
        hist[t] = stream.load(t, names_needed)   # downloads, slices, deletes .pt
    return hist


# =============================================================================
# The predictor cross-product on accumulated per-group history
# =============================================================================
def score_family_on_group(fam, group_vectors_by_tick: dict, ticks_sorted: list[int],
                          delta: int, h: int, floor: float | None = None) -> dict | None:
    """Score ONE family at (Delta=delta, horizon=h) on ONE group's history.

    History = the `order+1`-ish snapshots ending at the anchor (theta_stale); the
    scoring point is `h` sampling-steps after the anchor. Learnable/regression
    families FIT on a strictly-earlier retrospective split (leakage guard) then
    score on the held-out later point. Returns a full metric row or None if the
    window is too short.

    NOISE FLOOR (EXP-44 correction). The floor is the bf16 DIFFERENCED-noise floor
    of THIS predictor's residual e = (sum_j c_j theta_j) - theta_now — computed here
    from the family's own linear coefficients and the actual history + theta_now (see
    noise_floor.differenced_floor). The old passed-in `||theta||`-scaled `floor` arg
    is IGNORED (a category error; kept only for call-site compatibility). Because it
    is per-element at each element's magnitude and propagated through the coefficients,
    an unchanging tensor floors to ~0 and a moving block clears it at h>=5.
    """
    from .predictors import fit_score_split
    n = len(ticks_sorted)
    # anchor position = the score position minus h; need enough history behind it.
    order = max(fam.order, 1) if fam.order > 0 else 3
    need_hist = order + 1
    # scoring point index (last tick), anchor = score - h
    score_pos = n - 1
    anchor_pos = score_pos - h
    if anchor_pos < need_hist - 1:
        return None
    hist_positions = list(range(anchor_pos - (need_hist - 1), anchor_pos + 1))
    history = [(ticks_sorted[p], group_vectors_by_tick[ticks_sorted[p]]) for p in hist_positions]
    theta_now = group_vectors_by_tick[ticks_sorted[score_pos]]
    theta_stale = group_vectors_by_tick[ticks_sorted[anchor_pos]]

    if getattr(fam, "needs_fit", False):
        # LEAKAGE GUARD: fit on a retrospective point strictly BEFORE the score point.
        # Fit target = the anchor extrapolated to a PAST held-out point (anchor_pos)
        # from an even-earlier history — score stays the strictly-later real point.
        fit_len = need_hist
        fit_idx, score_idx = fit_score_split(list(range(n)), anchor_pos, fit_len)
        # build fit history ending strictly before anchor, fit toward the anchor truth
        fit_hist_pos = list(range(max(0, anchor_pos - need_hist), anchor_pos))
        if len(fit_hist_pos) < need_hist:
            return None
        fit_history = [(ticks_sorted[p], group_vectors_by_tick[ticks_sorted[p]]) for p in fit_hist_pos]
        fit_truth = group_vectors_by_tick[ticks_sorted[anchor_pos]]
        assert max(fit_hist_pos) < score_pos, "leakage: fit history overlaps score point"
        try:
            fam.fit(fit_history, fit_truth, h=1)
        except Exception:
            return None

    theta_hat = fam.predict(history, h)
    # CORRECTED differenced-noise floor for THIS predictor's residual (family coeffs).
    coeffs = fam.linear_coeffs(len(history), h)
    hist_vectors = [th for _, th in history]
    diff_floor = NF.differenced_floor(coeffs, hist_vectors, theta_now)
    row = M.full_metric_row(theta_hat, theta_now, theta_stale, diff_floor)
    row.update({"family": fam.name, "coeff_source": fam.coeff_source,
                "order": fam.order, "delta": delta, "h": h, "floor": diff_floor})
    return row
