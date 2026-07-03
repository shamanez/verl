#!/usr/bin/env python3
"""weight_proj/structure.py — #45 structure-axes layer (matrix_name -> layer_idx / block_type / super_block).

ADDITIVE module (plan 45 `## Code change`). The name parsers (`_LAYER_RE`,
`block_family` / `layer_index`) are defined locally below (vendored verbatim from
the retired sweep engine) and remapped to the sharpened MOAT taxonomy:

  1. any name ending `.bias`                 -> block_type=bias,  super_block=bias
     (this MOVES the 84 q/k/v biases OUT of q/k/v_proj — sweep.block_family folds
     them in via startswith)
  2. input_layernorm / post_attention_layernorm / model.norm.weight
                                             -> block_type=norm,  super_block=norm
  3. q/k/v/o `.weight`                       -> block_type=<x>_proj, super_block=attention
  4. gate/up/down `.weight`                  -> block_type=<x>_proj, super_block=mlp
  5. model.embed_tokens.weight               -> block_type=embed, super_block=embed
  6. lm_head is TIED (Qwen2.5-1.5B tie_word_embeddings=true; the 338-matrix manifest
     has NO standalone lm_head.weight) — the scorecard synthesizes an explicit
     `tied=true, tied_to=embed` lm_head row via `synthesize_tied_lm_head` so #56
     sees an lm_head row rather than assuming exclusion.
  7. anything unmatched                      -> other/other (partition gate FAILS if >0)

338-matrix accounting (Qwen2.5-1.5B-Instruct, 28 decoder layers): per layer q/k/v
`.weight`+`.bias` = 6, o `.weight` = 1, gate/up/down `.weight` = 3, 2 layernorms = 2
-> 12/layer x 28 = 336, plus embed_tokens.weight + final norm.weight = 338.
"""
from __future__ import annotations

import re
from collections import Counter

N_LAYERS = 28  # Qwen2.5-1.5B-Instruct decoder depth (the fixed control model)

# =============================================================================
# Name parsers (vendored verbatim from the retired weight_proj.sweep engine).
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

DECODER_WEIGHT_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")
BLOCK_TYPES = DECODER_WEIGHT_TYPES + ("norm", "bias", "embed", "lm_head", "other")
SUPER_BLOCKS = ("attention", "mlp", "norm", "bias", "embed", "lm_head", "other")

_SB_OF_BT = {
    "q_proj": "attention", "k_proj": "attention", "v_proj": "attention",
    "o_proj": "attention",
    "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp",
    "norm": "norm", "bias": "bias", "embed": "embed",
    "lm_head": "lm_head", "other": "other",
}

# the tied-lm_head descriptor (rule 6) — consumed by synthesize_tied_lm_head
TIED_LM_HEAD = {"block_type": "lm_head", "super_block": "lm_head",
                "tied": True, "tied_to": "embed"}


def classify(name: str) -> dict:
    """matrix_name -> {matrix_name, layer_idx, block_type, super_block, special}.

    layer_idx comes from layer_index (local); block routing applies the sharpened
    taxonomy above ON TOP of block_family (bias split FIRST, since block_family folds
    q/k/v biases into q/k/v_proj). `special` is the special-group label
    (embed/norm/bias) for layer-agnostic reporting, else None.
    """
    layer_idx = layer_index(name)
    if name.endswith(".bias"):
        bt = "bias"
    elif name == "model.embed_tokens.weight":
        bt = "embed"
    else:
        try:
            fam = block_family(name)
        except AssertionError:
            fam = None
        if fam in ("input_layernorm", "post_attention_layernorm", "norm"):
            bt = "norm"
        elif fam in DECODER_WEIGHT_TYPES:
            bt = fam
        elif fam == "embed":
            bt = "embed"
        else:
            bt = "other"
    sb = _SB_OF_BT[bt]
    special = bt if bt in ("embed", "norm", "bias") else None
    return {"matrix_name": name, "layer_idx": layer_idx,
            "block_type": bt, "super_block": sb, "special": special}


def canonical_matrix_names(n_layers: int = N_LAYERS) -> list[str]:
    """The 338 canonical Qwen2.5-1.5B-Instruct state-dict matrix names, in dump order.

    These are FULLY determined by the architecture (they are the real HF state-dict
    keys, not a fixture): embed + final norm + per-layer {q,k,v}_proj .weight/.bias,
    o_proj .weight, {gate,up,down}_proj .weight, and the two layernorms. lm_head is
    TIED (no standalone key). Used by the offline harness's self-test to exercise the
    partition gate WITHOUT the ~1 TB trace/manifest on disk (lightweight-testing path);
    the on-box run still validates against the REAL manifest names. partition() on this
    list returns ok=True with exactly expected_counts()."""
    names = ["model.embed_tokens.weight"]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        for x in ("q_proj", "k_proj", "v_proj"):     # Qwen2.5 attn has .weight AND .bias
            names.append(f"{p}self_attn.{x}.weight")
            names.append(f"{p}self_attn.{x}.bias")
        names.append(f"{p}self_attn.o_proj.weight")
        for x in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"{p}mlp.{x}.weight")
        names.append(f"{p}input_layernorm.weight")
        names.append(f"{p}post_attention_layernorm.weight")
    names.append("model.norm.weight")
    return names


def expected_counts(n_layers: int = N_LAYERS) -> tuple[dict, dict]:
    """EXACT expected per-group counts for a tied-embedding Qwen2-style decoder."""
    bt = {t: n_layers for t in DECODER_WEIGHT_TYPES}
    bt.update({"norm": 2 * n_layers + 1, "bias": 3 * n_layers,
               "embed": 1, "lm_head": 0, "other": 0})
    sb = {"attention": 4 * n_layers, "mlp": 3 * n_layers,
          "norm": 2 * n_layers + 1, "bias": 3 * n_layers,
          "embed": 1, "lm_head": 0, "other": 0}
    return bt, sb


def partition(names: list[str], n_layers: int = N_LAYERS) -> dict:
    """Classify every name; verify the partition-integrity gate (plan step 2).

    Each name maps to exactly ONE block_type and ONE super_block by construction
    (classify is a pure function); the gate is that the COUNTS are exactly the
    expected ones, both partitions sum to len(names) == 12*n_layers + 2, and
    count(other) == 0. Returns a machine-checkable report; `ok` is the hard gate.
    """
    rows = [classify(n) for n in names]
    bt_counts = Counter(r["block_type"] for r in rows)
    sb_counts = Counter(r["super_block"] for r in rows)
    exp_bt, exp_sb = expected_counts(n_layers)
    n_expected = 12 * n_layers + 2
    failures: list[str] = []
    if len(names) != len(set(names)):
        failures.append(f"duplicate matrix names: {len(names) - len(set(names))}")
    if len(names) != n_expected:
        failures.append(f"n_matrices {len(names)} != expected {n_expected}")
    for k in BLOCK_TYPES:
        if bt_counts.get(k, 0) != exp_bt[k]:
            failures.append(f"block_type[{k}] = {bt_counts.get(k, 0)} != {exp_bt[k]}")
    for k in SUPER_BLOCKS:
        if sb_counts.get(k, 0) != exp_sb[k]:
            failures.append(f"super_block[{k}] = {sb_counts.get(k, 0)} != {exp_sb[k]}")
    if sum(bt_counts.values()) != len(names):
        failures.append("block_type partition does not cover all names")
    if sum(sb_counts.values()) != len(names):
        failures.append("super_block partition does not cover all names")
    other_names = [r["matrix_name"] for r in rows if r["block_type"] == "other"]
    if other_names:
        failures.append(f"partition leak: count(other) = {len(other_names)}: "
                        f"{other_names[:5]}")
    return {
        "rows": rows,
        "block_type_counts": {k: bt_counts.get(k, 0) for k in BLOCK_TYPES},
        "super_block_counts": {k: sb_counts.get(k, 0) for k in SUPER_BLOCKS},
        "expected_block_type": exp_bt,
        "expected_super_block": exp_sb,
        "n_matrices": len(names),
        "other_names": other_names,
        "failures": failures,
        "ok": not failures,
    }


def synthesize_tied_lm_head(embed_row: dict) -> dict:
    """Clone an emitted embed row into the explicit TIED lm_head row (rule 6).

    lm_head shares storage with embed (tie_word_embeddings=true), so its metrics ARE
    embed's metrics; the row exists so #56 sees lm_head explicitly (tied=true,
    tied_to=embed) instead of silently dropping it.
    """
    row = dict(embed_row)
    row.update({
        "group_key": "lm_head",
        "block_type": TIED_LM_HEAD["block_type"],
        "super_block": TIED_LM_HEAD["super_block"],
        "special": "lm_head",
        "tied": True,
        "tied_to": TIED_LM_HEAD["tied_to"],
    })
    return row
