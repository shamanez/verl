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
"""Geometry monitor (issue #93 I7): compare two merged HF checkpoints.

Per 2D weight matrix it reports
  * bf16-changed fraction  - fraction of elements whose bf16 bit pattern
    differs (the refresh-compression estimate, M1); fp32 sub-ULP drift that
    a bf16 wire format would never ship counts as unchanged.
  * NSS (normalized spectral shift) - sigma_max(W1 - W0) / sigma_max(W0).
  * top-k principal angles (degrees) between the top-k left singular
    subspaces of W0 and W1 (k = --topk, default 8).
  * relative Frobenius change ||W1 - W0||_F / ||W0||_F.

1D tensors (norms, biases) get the bf16 fraction + relative change only.

Usage:
  python geometry_monitor.py --base <hf_dir_or_hub_id> --other <hf_dir> \
      [--topk 8] [--substr layers.] [--out geometry.json] [--fast]

Aggregates are grouped by transformer layer index (model.layers.<i>.) so an
early-layer shift (the #93 conditional-protection signature) is visible in
one table.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def _resolve_model_dir(spec: str) -> Path:
    p = Path(spec).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(spec, allow_patterns=["*.safetensors", "*.json"])
    )


def load_state_dict(spec: str) -> dict:
    """Load all safetensors shards of a merged HF checkpoint on CPU."""
    from safetensors.torch import load_file

    d = _resolve_model_dir(spec)
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {d}")
    sd = {}
    for s in shards:
        sd.update(load_file(str(s), device="cpu"))
    return sd


def bf16_changed_fraction(a, b) -> float:
    import torch

    ab = a.to(torch.bfloat16).view(torch.int16)
    bb = b.to(torch.bfloat16).view(torch.int16)
    return (ab != bb).float().mean().item()


def top_sigma(w, q: int = 6, niter: int = 4) -> float:
    import torch

    if min(w.shape) <= q:
        return torch.linalg.svdvals(w.float()).max().item()
    _, s, _ = torch.svd_lowrank(w.float(), q=q, niter=niter)
    return s[0].item()


def principal_angles_deg(w0, w1, k: int, niter: int = 8) -> list[float]:
    """Angles between the spans of the top-k left singular vectors.

    The randomized factorizations share one RNG seed so identical inputs give
    identical subspaces (angle 0), not sketch noise.
    """
    import torch

    q = min(k + 4, min(w0.shape))
    torch.manual_seed(0)
    u0, _, _ = torch.svd_lowrank(w0.float(), q=q, niter=niter)
    torch.manual_seed(0)
    u1, _, _ = torch.svd_lowrank(w1.float(), q=q, niter=niter)
    u0, u1 = u0[:, :k], u1[:, :k]
    s = torch.linalg.svdvals(u0.T @ u1).clamp(-1.0, 1.0)
    return [math.degrees(math.acos(v.item())) for v in s]


_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def layer_of(name: str) -> int:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="reference checkpoint (HF dir or hub id)")
    ap.add_argument("--other", required=True, help="checkpoint to compare against --base")
    ap.add_argument("--topk", type=int, default=8, help="principal-angle subspace size")
    ap.add_argument("--substr", default="", help="only tensors whose name contains this")
    ap.add_argument("--out", default="", help="write the full JSON report here")
    ap.add_argument("--fast", action="store_true", help="bf16 fraction + Frobenius only (M1 mode)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))
    sd0, sd1 = load_state_dict(args.base), load_state_dict(args.other)
    names = sorted(set(sd0) & set(sd1))
    missing = sorted(set(sd0) ^ set(sd1))
    if args.substr:
        names = [n for n in names if args.substr in n]

    rows = []
    for n in names:
        w0, w1 = sd0[n], sd1[n]
        if w0.shape != w1.shape:
            rows.append({"name": n, "error": f"shape {tuple(w0.shape)} vs {tuple(w1.shape)}"})
            continue
        f0 = w0.float()
        diff = w1.float() - f0
        fro0 = f0.norm().item()
        row = {
            "name": n,
            "layer": layer_of(n),
            "numel": w0.numel(),
            "bf16_changed_frac": bf16_changed_fraction(w0, w1),
            "rel_fro_change": diff.norm().item() / fro0 if fro0 > 0 else 0.0,
        }
        if w0.dim() == 2 and not args.fast:
            s0 = top_sigma(f0)
            row["nss"] = top_sigma(diff) / s0 if s0 > 0 else 0.0
            row["principal_angles_deg"] = [
                round(a, 3) for a in principal_angles_deg(f0, w1.float(), args.topk)
            ]
        rows.append(row)

    by_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "error" in r:
            continue
        by_layer[r["layer"]]["bf16"].append(r["bf16_changed_frac"])
        by_layer[r["layer"]]["fro"].append(r["rel_fro_change"])
        if "nss" in r:
            by_layer[r["layer"]]["nss"].append(r["nss"])
            by_layer[r["layer"]]["angle1"].append(r["principal_angles_deg"][-1])

    total = sum(r["numel"] for r in rows if "error" not in r)
    changed = sum(r["bf16_changed_frac"] * r["numel"] for r in rows if "error" not in r)
    print(f"tensors compared: {len(rows)}  (name mismatches: {len(missing)})")
    print(f"GLOBAL bf16-changed fraction: {changed / total:.6f}  over {total:,} params")
    hdr = f"{'layer':>5} {'bf16_frac':>10} {'rel_fro':>10} {'nss_max':>9} {'max_angle_deg':>14}"
    print(hdr)
    for layer in sorted(by_layer):
        m = by_layer[layer]
        nss = f"{max(m['nss']):.4f}" if m.get("nss") else "-"
        ang = f"{max(m['angle1']):.2f}" if m.get("angle1") else "-"
        print(
            f"{layer:>5} {sum(m['bf16']) / len(m['bf16']):>10.5f} "
            f"{max(m['fro']):>10.5f} {nss:>9} {ang:>14}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "base": args.base,
                    "other": args.other,
                    "topk": args.topk,
                    "global_bf16_changed_fraction": changed / total,
                    "tensors": rows,
                    "name_mismatches": missing,
                },
                indent=1,
            )
        )
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
