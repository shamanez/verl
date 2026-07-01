# Weight-trace access pattern — analysis runs on a PRE-DOWNLOADED local trace (fp32 EXP-57)

> Canonical access note for the M4 weight-projection analyses (issues #45–#56). **Updated
> 2026-07-02:** analysis has moved to the **fp32 EXP-57** trace and to a **download-everything-first**
> model on a cheap, big-disk, GPU-free box. The old "stream + reduce, never bulk-download, delete
> each `.pt`, bound to a few GB" discipline is **RELEASED for analysis**. It still governs
> **collection** (`research/scripts/weight_traj_run_cell.sh`, which keeps its upload-then-delete path).

## Trace locations (canonical)
- **fp32 (primary — what all analysis now uses):**
  `s3://shamane-pluralis/verl-research/EXP-57/regimeA/weights/full/tick_<N>/tick_<N>.pt` —
  160 per-tick snapshots (N=0..159), each a FULL fp32 model state_dict (338 matrices).
  **~6.17 GB/snapshot ⇒ ~987 GB (~1 TB) for all 160.** This ~1 TB is the **intended on-disk size**
  of the analysis box, not a hazard.
- **bf16 (legacy reference only):** `.../EXP-43/regimeA/weights/full/…` — ~3.08 GB/snapshot, ~492 GB.
  Kept as the **communication-regime** reference; not the substrate for the weight-space science.

## The model: download the whole trace once, then analyse from local disk
Analysis runs on a **cheap, GPU-free, big-disk (~1 TB+), ample-RAM** Vast box (the projection study is
GPU-free). Provision the cheapest 1× low-end GPU offer with a big disk + fast inet (the locked image is
GPU-oriented, so target a small GPU rather than truly zero-GPU), e.g. via the `vast-provision` skill:
`--query 'num_gpus=1 gpu_ram>=8 disk_space>=1100 reliability>0.95 inet_down>=500' --disk-gb 1100 --max-price 0.6`.

1. **One-shot fetch (the ONE place a bulk pull is allowed).**
   ```bash
   set -a; . ~/.config/verl-research/secrets.env; set +a          # R2_* -> AWS_* internally
   python scripts/weight_proj_fetch_trace.py --experiment EXP-57 --dest /workspace/trace/EXP-57
   ```
   Pulls all 160 fp32 snapshots into `<dest>/full/tick_<N>/tick_<N>.pt` (resumable, `--jobs` parallel).
   Add `--cadence per-step` for the ~80-tick / ~494 GB subsample if you don't need per-tick Δ resolution.

2. **Synthesize the EXP-57 manifests (once).** EXP-57 shipped without manifests; the engine needs them:
   ```bash
   python scripts/synth_exp57_manifests.py --trace-root /workspace/trace/EXP-57
   ```
   Writes `runs/EXP-57/regimeA/weights/{full_manifest,r2_manifest}.jsonl` — matrix structure reused from
   EXP-43 (identical model), per-matrix **fp32** fro-norms **recomputed** from one real EXP-57 snapshot.

3. **Analyse from local disk.**
   ```bash
   python scripts/weight_proj_sweep.py runs/EXP-57/regimeA/weights/full_manifest.jsonl \
       --trace-root /workspace/trace/EXP-57 --cadence per-tick --families all \
       --emit-report reports/<name>.html
   ```
   `--trace-root` reads snapshots **in place** (no download, no delete, no `df` guard). fp32 is
   auto-detected from the manifest's `dump_dtype`, so the **bf16 quantization-noise floor gate is OFF** —
   reliability is **projection accuracy + linearity (directedness)**, not a floor.

## Streaming is still available (secondary path)
Omit `--trace-root` and the engine streams from R2 one `.pt` at a time (the original bounded-footprint
reader, unchanged). This is the natural fit for a **few-snapshot** pass that does not want a ~1 TB local
trace — e.g. the GPU-gated functional tier (#46), which needs only a handful of snapshots per horizon.
Set `WP_R2_EXPERIMENT=EXP-57` (or `WP_R2_PREFIX=…`) so the streamer targets the fp32 trace.

## Credentials for ad-hoc `aws` → R2
Repo scripts (`verl/workers/comm_eff/r2_sink.py`, `research/scripts/verify_full_weight_dump.py`,
`weight_proj/r2_stream.py`) map `R2_* → AWS_*` internally, so they work as-is. For a MANUAL `aws` call:
`export AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY AWS_DEFAULT_REGION=auto`
and pass `--endpoint-url "$R2_ENDPOINT"`. Bucket = `shamane-pluralis` only. Secret VALUES are never logged.
