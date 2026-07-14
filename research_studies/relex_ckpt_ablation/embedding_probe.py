"""Embedding-only forecast skill for a few key combos (H6).

The tied embedding (~233M params, ~15% of the model) is excluded from the main
grid because loading 10 fp32 copies (9.3 GB) plus per-combo delta matrices spikes a
26 GB laptop into swap. Here we score just that one tensor for the headline combos,
loading only the steps a given combo needs, so memory stays bounded.

Usage: python embedding_probe.py --ckpt_dir <ckpts> --gap 10 --anchors 40,50,60 \
           --windows 2,4 --horizon 1
"""

from __future__ import annotations

import argparse
import os
import re

import harness_projector as hp
import torch
from run_forecast_ablation import tensor_type, whole_tensor_metrics
from safetensors import safe_open


def find_embed_name(ckpt_dir, step):
    d = os.path.join(ckpt_dir, f"global_step_{step}")
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), framework="pt", device="cpu") as h:
                for k in h.keys():
                    if tensor_type(k) == "embed":
                        return k
    return None


def load(ckpt_dir, step, name):
    d = os.path.join(ckpt_dir, f"global_step_{step}")
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), framework="pt", device="cpu") as h:
                if name in h.keys():
                    return h.get_tensor(name).to(torch.float32)
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--anchors", default="40,50,60")
    ap.add_argument("--windows", default="2,4")
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    avail = set(int(m.group(1)) for m in (re.match(r"global_step_(\d+)$", d) for d in os.listdir(args.ckpt_dir)) if m)
    name = find_embed_name(args.ckpt_dir, sorted(avail)[0])
    print(f"embedding tensor: {name}")
    anchors = [int(x) for x in args.anchors.split(",")]
    windows = [int(x) for x in args.windows.split(",")]

    from collections import defaultdict

    agg = defaultdict(list)
    for W in windows:
        for anchor in anchors:
            src = [anchor - (W - 1 - i) * args.gap for i in range(W)]
            target = anchor + args.horizon * args.gap
            if not (set(src) | {target}) <= avail:
                continue
            snaps = [load(args.ckpt_dir, s, name) for s in src]
            actual = load(args.ckpt_dir, target, name)
            proj, st = hp.project_rank1_tensor(snaps, list(src), target, strength=1.0, rank=1)
            m = whole_tensor_metrics(proj, snaps[-1], actual)
            agg[W].append(m["skill"])
            print(
                f"W={W} anchor={anchor} tgt={target} -> skill={m['skill']:+.4f} "
                f"cos={m['direction_cos']:+.4f} evr={st['evr']:.4f}"
            )
            del snaps, actual, proj
    print("\n== embedding skill (mean over anchors) ==")
    import statistics as sstat

    for W in windows:
        v = agg[W]
        if v:
            mean = sstat.mean(v)
            std = sstat.pstdev(v) if len(v) > 1 else 0.0
            print(f"  W={W}: skill = {mean:+.4f} +/- {std:.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
