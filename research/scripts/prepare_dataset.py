"""Skeleton dataset-prep helper.

Real experiments fill in the dataset specifics per plan. This skeleton:
- Accepts --hf-dataset, --output-dir, --max-rows, --tokenizer.
- Downloads or loads from cache (placeholder — uses `datasets` if available).
- Shards into N parquet files in output_dir for fast rsync to Vast.ai nodes.

The experiment-runner agent invokes this with parameters lifted from the plan's
`## Notes for runner` section when the plan declares dataset prep.

This script is intentionally minimal — extend per experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-dataset", help="HuggingFace dataset id (e.g. gsm8k)")
    ap.add_argument("--local-path", type=Path,
                    help="alternative: local parquet/jsonl path to shard")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--tokenizer", default=None,
                    help="optional HF tokenizer id (e.g. Qwen/Qwen2.5-7B); skips if unset")
    args = ap.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.hf_dataset and not args.local_path:
        print("prepare_dataset: must pass --hf-dataset or --local-path", file=sys.stderr)
        return 2

    # Lazy imports so the script's --help and arg-parsing work without heavy deps.
    try:
        if args.hf_dataset:
            from datasets import load_dataset  # type: ignore[import-not-found]
            ds = load_dataset(args.hf_dataset, split="train")
        else:
            from datasets import load_dataset  # type: ignore[import-not-found]
            ds = load_dataset("parquet", data_files=str(args.local_path), split="train")
    except ImportError:
        print("prepare_dataset: install `datasets` (uv pip install datasets) or extend this script.",
              file=sys.stderr)
        return 1

    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    rows_per_shard = max(1, len(ds) // args.shards)
    written = []
    for i in range(args.shards):
        start = i * rows_per_shard
        end = len(ds) if i == args.shards - 1 else start + rows_per_shard
        shard = ds.select(range(start, end))
        shard_path = out / f"shard-{i:04d}.parquet"
        shard.to_parquet(str(shard_path))
        written.append({"shard": i, "rows": len(shard), "path": str(shard_path)})

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({
        "source": args.hf_dataset or str(args.local_path),
        "total_rows": sum(s["rows"] for s in written),
        "shards": written,
    }, indent=2))
    print(f"prepare_dataset: wrote {len(written)} shards to {out}, manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
