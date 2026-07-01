# Research Runs Summary

Durable record (full run dirs de-bloated; provenance = this file + W&B + git
history + merged code). North-star + "done": `../.claude/GOAL.md`. The two
active fronts: `../reports/priority-1-anchor-staleness-k-collapse.html` and
`../reports/priority-2-compression-train-inference-mismatch.html`.

## Current baseline (the problem state)

The baseline every future test compares against is
`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`:

- **merger** — `signed_ema` (α=0.25, β_anc=0.50)
- **substrate** — locked PowerSGD r=77 anchor circuit (anchor owns `Q`, clean=0,
  paired replay, `disable_custom_all_reduce`)
- **surface** — resp 1024, dynamic-bsz, rollout TP=1, gpu_mem 0.55, 50 steps, val@25/50, diagnostics off
- **anchor latency** — `cadence`/`delay_K` = **20/20** — the **k-collapse regime**

The baseline deliberately runs at high latency, where the method **fails** (see
below). That failure is the problem Priority 1 targets. Exact values:
`FIXED_CONTROL_SURFACE.md` (not duplicated here).

## Settled background (locked — do not relitigate)

- **Substrate locked** — PowerSGD r=77 on the mandatory anchor circuit reaches
  dense parity at ~5% gradient comm. The anchor is mandatory and is the only
  thing that updates `Q`; the two-circuit structure is mandatory.
- **Stable/parity holds only at LOW anchor latency.** At `cadence`/`delay_K` = 5/5
  the comm-eff `signed_ema` run reached **val@50 ≈ 0.736** vs a dense control
  **≈ 0.766** on the older 2K surface (n=1 each; rollout nondeterminism ≈ ±0.024/draw)
  — i.e. parity-band at ~5% gradient comm. **Goals 1–3 (stable / parity / savings)
  are met at low latency; Goal 4 (one canonical surpass launcher) is open.**
- **Reference floors** — no-merger PowerSGD ≈ 0.63; dense full-gradient band ≈ 0.75–0.78.

## The k-collapse finding (why the baseline sits at 20/20)

Anchor latency is the failure knob. At **20/20** the method breaks; at 5/5 it is
stable/near-dense. The stale anchor gradient rotates ~orthogonal to the live
gradient by k≈10–20 (cos 0.51→0.18@k5→~0@k10→−0.01@k20, norm preserved ⇒ pure
rotation). Two collapse symptoms, one cause (off-policy staleness `K>τ`):

| anchor latency | merger | outcome |
|---|---|---|
| 5/5 | signed_ema | STABLE, near-dense |
| 20/20 | signed_ema | terminal collapse ~step 61 (entropy collapse + length explosion) |
| 20/20 | (additive mergers) | stalls (grad_norm grows, sub-baseline plateau) |
| dense control | — | sails through to ≈0.78 ⇒ compression-specific, not an epoch effect |

Full argument: `../reports/priority-1-anchor-staleness-k-collapse.html`.

## Bottom line

The baseline is a comm-eff trainer that is **stable at low anchor latency but
collapses at the realistic high latency** (single slow anchor serving a fast
swarm). The two open priorities both target the anchor ↔ fast-circuit coupling:
**(1)** fix the k-collapse by extrapolating the anchor's weights forward
(Priority 1), and **(2)** reduce the compression-induced train–inference
mismatch (Priority 2). Both next steps are **GPU-free offline kill-gates** — see
the two reports. Stay on EMA-family mergers; everything else is locked
(`FIXED_CONTROL_SURFACE.md`).

## Milestone M4

The M4 weight-projection collection unit has PASSed. Roll-up:

- **EXP-43 — shared dense FULL-weight per-tick trajectory to R2 (collection unit, PASS).**
  Success criteria checked: all five acceptance gates on the artifact (160/160
  full-model bf16 snapshots in R2 with n_matrices=338 real shapes, not a reduced/subset dump;
  >=80 steps no NaN; r2_manifest 160/160 verified:true + sample download verify PASS;
  comm_eff counters all 0 = codec OFF; local staging near-empty = upload-then-delete).
  Key values: verify_full_weight_dump.py --r2 PASS, max_rel_norm_err=0.0001 (<= 0.01 tol);
  final GSM8K val acc 0.7809 (dense-control band 0.75-0.78, provenance only); WandB
  a51waqza; run cost ~$18.60. This raw full-weight trace is the ground-truth substrate
  the analysis spine (#44-#56) consumes; the offline weight-projection sweep (projection
  ratio / direction cosine vs horizon, crossover h*, RLVR-linearity, damped-alpha) is
  computed on it directly — never on any lossy or subset instrument.

**M4 weight-proj spine root (published, canonical).** Every downstream M4 weight-proj
issue (#44 through #56) reads the EXP-43 R2 trace:

  `s3://shamane-pluralis/verl-research/EXP-43/regimeA/weights/full/`

Key form `tick_<N>/tick_<N>.pt`, 160 bf16 full-model snapshots (n_matrices=338), one
per optimizer tick over 80 steps (~492 GB, R2 ONLY — do not pull to the laptop). The
per-step movement trajectory is the first tick of each global_step (ticks 0,2,...,158
-> steps 1..80). Heavy .pt live in R2; local artifacts are the two small manifests +
the internal log. Downstream differencing of consecutive snapshots must account for
bf16 ~4e-3 relative error (or commission a future fp32 run, dump_dtype=fp32, ~984 GB).

**Index + access (what every downstream issue reads).** The trace is self-describing: the
index manifests live BOTH in R2 (`.../weights/full_manifest.jsonl` = per-tick per-matrix
names/shapes/fp32-norms; `.../weights/r2_manifest.jsonl` = keys + verified sizes) AND tracked
in-repo at `runs/EXP-43/regimeA/weights/{full_manifest,r2_manifest}.jsonl`. MANDATORY access
discipline (do NOT bulk-download ~492 GB, it will exhaust disk): stream layer-wise/block-wise,
reduce to small per-layer/per-block intermediates, combine, THEN render the HTML. Full pattern:
`research/reports/r2-access-pattern-for-analysis.md` (also commented on every issue #44-#56).
Ad-hoc aws->R2 needs `R2_*->AWS_*` (region=auto); the repo scripts map it internally.

**Next (M4 analysis phase).** #43 (collection) is DONE and closed. The entry point is **#44**
(extend the offline weight-projection sweep engine: orders>=2, damped-alpha, learnable-at-every-order,
general regression, EMA/momentum, new GPU-free metrics) - the shared engine the rest of #45-#56
build on. All downstream M4 issues are `kind:analysis` (GPU-free) and depend only on this trace.
