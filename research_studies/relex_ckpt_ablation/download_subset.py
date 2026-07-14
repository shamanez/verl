"""Download ONLY the RELEX trajectory steps the cadence study needs — to a temp dir.

The released trajectory ``relex-rlvr/RLVR-Qwen2.5-Math-1.5B`` exposes every step as
its own Hub branch (``revision="step_N"``), so we fetch a SPARSE set of revisions
instead of the whole 500-step trajectory. Each checkpoint is ~3.1 GB (1.5B params,
fp16 safetensors), so downloading 10-12 steps is ~35 GB — laptop-feasible.

Never writes into the repo: pass an absolute ``--output_dir`` under a temp/scratch
mount (e.g. $TMPDIR or an external SSD). Idempotent: skips steps already present.

Example (the Tier-1 core set: base + multiples of 10 up to 100):
  python download_subset.py \
      --steps 10,20,30,40,50,60,70,80,90,100 \
      --output_dir "$TMPDIR/relex_ckpt_study" \
      --with_base
"""

import argparse
import glob
import os
import sys

from huggingface_hub import snapshot_download

ALLOW = ["*.safetensors", "*.json", "*.txt", "*.jinja"]
HUB_REPO = "relex-rlvr/RLVR-Qwen2.5-Math-1.5B"
BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"


def _present(dst):
    return os.path.exists(os.path.join(dst, "config.json")) and (
        bool(glob.glob(os.path.join(dst, "*.safetensors"))) or bool(glob.glob(os.path.join(dst, "pytorch_model*.bin")))
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", required=True, help="Comma list of step indices, e.g. 10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--output_dir", required=True, help="ABSOLUTE temp dir (NOT inside the repo).")
    ap.add_argument("--hub_repo", default=HUB_REPO)
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument(
        "--with_base",
        action="store_true",
        help="Also download the base model theta_0 (needed for from-base deltas / RELEX contrast).",
    )
    args = ap.parse_args()

    out = os.path.abspath(args.output_dir)
    if "/verl" in out and "/verl-" not in out and "relex_ckpt" not in out:
        # Loud guard against accidentally writing into a verl checkout.
        print(f"[warn] output_dir={out} looks like it may be inside a repo. Use a temp/scratch path.", file=sys.stderr)
    os.makedirs(out, exist_ok=True)
    steps = [int(s) for s in args.steps.split(",") if s.strip()]

    if args.with_base:
        dst = os.path.join(out, "base_theta0")
        if not _present(dst):
            print(f"[base] {args.base_model} -> {dst}", flush=True)
            snapshot_download(args.base_model, local_dir=dst, allow_patterns=ALLOW)
        else:
            print(f"[base] present, skip {dst}")

    for n in steps:
        dst = os.path.join(out, f"global_step_{n}")
        if _present(dst):
            print(f"[step_{n}] present, skip", flush=True)
            continue
        print(f"[step_{n}] {args.hub_repo}@step_{n} -> {dst}", flush=True)
        snapshot_download(args.hub_repo, revision=f"step_{n}", local_dir=dst, allow_patterns=ALLOW)

    # Report disk footprint.
    total = 0
    for root, _d, files in os.walk(out):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print(f"\nDone. {len(steps)} steps{' + base' if args.with_base else ''} at {out}/  ({total / 1e9:.1f} GB on disk)")


if __name__ == "__main__":
    main()
