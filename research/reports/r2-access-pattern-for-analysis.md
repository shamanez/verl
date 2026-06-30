# R2 weight-trace access pattern — MANDATORY for every M4 weight-projection analysis (EXP-43 trace)

> Canonical constraint for issues #44-#56. The dense weight trajectory produced by EXP-43 (#43) lives in
> Cloudflare R2 and is far too large to pull down whole. Any analysis MUST stream + reduce it
> layer-wise / block-wise, then combine, then render HTML. Bulk-download = guaranteed out-of-disk error.

## Trace location (canonical)
- `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/` — ~160 per-tick snapshots, each a
  FULL bf16 model state_dict (~338 matrices, ~3.08 GB/snapshot). **Total ~492 GB.**
- Manifests at `.../weights/`: `full_manifest.jsonl` (what was dumped: global_step + tick + per-matrix
  names/shapes/dtype/fp32-norms) and `r2_manifest.jsonl` (what was verified-uploaded: keys + sizes).
- Resolution: per-tick (~160) subsamples to per-step (~80) by taking the FIRST tick of each `global_step`
  (every manifest row carries both `global_step` and `tick`).

## DO NOT bulk-download (~492 GB will exhaust laptop/box disk → out-of-disk-space error)
Stream and reduce incrementally; bound the local working set to a few GB:

1. **Layer-wise / block-wise reduction.** Each `.pt` is a whole-model snapshot, so loop on the OUTSIDE
   over the 28 decoder layers (or weight blocks: attn `q/k/v/o_proj`, mlp `gate/up/down_proj`, `embed_tokens`,
   `norm`), and on the INSIDE over only the ticks you need:
   - Need ONE layer/block across all ticks: stream tick-by-tick — `torch.load(map_location="cpu")` ONE
     snapshot, extract ONLY that layer/block's tensor(s), accumulate the partial result, then DELETE the
     local `.pt` before loading the next.
   - Need ALL layers independently: outer-loop layers/blocks, inner-loop ticks, freeing tensors between
     iterations (or load each snapshot once, fan its per-layer slices into per-layer accumulators, delete).
2. **Bound local footprint.** Stage at most a handful of in-flight snapshots (~a few GB), mirroring the
   collection's upload-then-delete discipline. Delete each `.pt` right after extracting what you need; watch
   `df` and never approach disk capacity.
3. **Combine THEN render.** Reduce to small per-layer/per-block intermediates (e.g. per-layer `.npz`/parquet
   of the metric-vs-horizon arrays), combine those into the final arrays, and ONLY THEN build the HTML.
   Never hold the full trajectory in RAM or on disk at once.

## Credentials for ad-hoc aws -> R2
The repo scripts (`verl/workers/comm_eff/r2_sink.py`, `research/scripts/verify_full_weight_dump.py`) map
`R2_* -> AWS_*` internally, so they work as-is. For a MANUAL `aws` call, export the mapping first:
`export AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY AWS_DEFAULT_REGION=auto`
and pass `--endpoint-url "$R2_ENDPOINT"`. Bucket = `shamane-pluralis` ONLY. Copy one object at a time and
remove it after use (never `aws s3 cp --recursive` the whole prefix to local).

This pattern is part of the analysis contract for every issue that consumes the EXP-43 weight trace.
