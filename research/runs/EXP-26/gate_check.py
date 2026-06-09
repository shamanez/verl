#!/usr/bin/env python3
"""EXP-26 ACCEPTANCE GATE: is the parallel uncompressed G_dense probe faithful on
a codec-ON arm? Compares G_dense against the TRUSTED G_fresh_anchor@delay_K=0
(a realism-GREEN full uncompressed backward at the SAME weights/batch) per target.

GATE (machine-checkable): on a comm-eff arm, post-Q-warm,
  cos(G_dense, G_fresh_anchor) >= 0.95  AND  norm_ratio in [0.8, 1.25].
Two full backwards at the same weights on the same batch MUST be cos~1.0.

Usage: python gate_check.py <arm_capture_dir>   (dir containing rank0/manifest.jsonl)
"""
import json, sys, statistics
from pathlib import Path
import torch

def _canon(n):
    n = n.replace("._fsdp_wrapped_module", "")
    return n[len("_fsdp_wrapped_module."):] if n.startswith("_fsdp_wrapped_module.") else n

root = Path(sys.argv[1]) / "rank0"
rows = [json.loads(l) for l in open(root / "manifest.jsonl")]
def idx(role):
    d = {}
    for r in rows:
        if r["role"] == role:
            d[(r["global_step"], r["optimizer_tick"], _canon(r["target_name"]))] = r
    return d
gd = idx("G_dense"); ga = idx("G_fresh_anchor"); gan = idx("G_anchor"); gc = idx("G_comp")
print(f"roles: G_dense={len(gd)} G_fresh_anchor={len(ga)} G_anchor={len(gan)} G_comp={len(gc)}")

def compare(a_idx, b_idx, label):
    common = sorted(set(a_idx) & set(b_idx))
    coss, ratios = [], []
    for k in common:
        A = torch.load(root / a_idx[k]["path"]).float().flatten()
        B = torch.load(root / b_idx[k]["path"]).float().flatten()
        if A.shape != B.shape or A.norm() == 0 or B.norm() == 0:
            continue
        coss.append(float((A @ B) / (A.norm() * B.norm())))
        ratios.append(float(A.norm() / B.norm()))
    if not coss:
        print(f"  {label}: NO common (gs,tick,target) keys"); return None
    mc, mr = statistics.median(coss), statistics.median(ratios)
    print(f"  {label}: n={len(coss)} median_cos={mc:.4f} median_norm_ratio={mr:.4f} "
          f"cos_range=[{min(coss):.3f},{max(coss):.3f}]")
    return mc, mr

print("=== ACCEPTANCE GATE: cos(G_dense, G_fresh_anchor) >= 0.95 AND norm_ratio in [0.8,1.25] ===")
r_fa = compare(gd, ga, "G_dense vs G_fresh_anchor (delay_K=0, TRUSTED)")
r_an = compare(gd, gan, "G_dense vs G_anchor (K-stale)")
if gc: compare(gd, gc, "G_dense vs G_comp (H1 — for reference)")
if r_fa:
    mc, mr = r_fa
    ok = mc >= 0.95 and 0.8 <= mr <= 1.25
    print(f"\nGATE: {'PASS' if ok else 'FAIL'} (cos={mc:.4f} need>=0.95 ; norm_ratio={mr:.4f} need[0.8,1.25])")
    sys.exit(0 if ok else 1)
sys.exit(2)
