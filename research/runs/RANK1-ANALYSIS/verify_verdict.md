# RANK1-ANALYSIS independent verification — 2026-07-03

## VERDICT: PASS

Independent recompute (plain numpy + torch.load on raw fp32 snapshots, weight_proj NOT
imported) reproduces every sampled row, invariant, and headline conclusion. No anomalies.

## Task 1 — direct recompute vs rows.jsonl (tol abs|rel 1e-5)
| # | method / matrix / spec | metric | recomputed | reported | max |Δ| |
|---|---|---|---|---|---|
| 1a | rank1_traj / L13.q_proj / w16 a119 h20 | ratio,dir_cos,skill,evr1,coef_r2_1 | 1.2744858 / 0.91592538 / -0.62431405 / 0.99617271 / 0.99087257 | identical | 4.7e-15 |
| 1b | rank1_anchored / L27.down_proj / w8 a79 h5 | ratio,dir_cos,skill | 0.98566615 / 0.97892012 / 0.028462232 | identical | 3.8e-14 |
| 1c | naive_last2 / L7.input_layernorm / a119 h40 | ratio,dir_cos,skill | 1.6262842 / 0.24920489 / -1.6448003 | identical | 1.3e-15 |
| 1d | rank2_traj / L0.o_proj / w8 a79 h10 | ratio,dir_cos,skill,evr1,coef_r2_1 | 0.98365093 / 0.39414628 / 0.032430843 / 0.99896619 / 0.99963377 | identical | 2.3e-14 |

All four PASS at machine precision (~1e-14), far inside the 1e-5 gate.

## Task 2 — invariants on rows.jsonl
- Total rows = 19800  → PASS.
- hold_stale: 900 finite rows, worst |ratio−1| = 0 (≤1e-9)  → PASS.
- No negative weight_proj_ratio; no NaN e2/b2 with finite ratio  → PASS.
- Additivity rank1_traj[16] a119 h20: global e2/b2/eb = sum of 61 matrix rows exactly (rel = 0.00e0)  → PASS.
- Additivity down_proj block_type group = sum of its 5 member matrices exactly (rel = 0.00e0)  → PASS.
- Row-count derivation: 61 matrices × 2 anchors × (2 rank-free methods × 6h  +  5 window methods × 4 windows × 6h)
  = 61 × 264 = 16104 matrix rows (+ global/super_block/block_type groups) = 19800 total. Observed split:
  matrix 16104, global 264, super_block 1056, block_type 2376; per-method matrix counts hold_stale/naive_last2=732,
  {two_point_window,rank1_traj,rank1_anchored,rank2_traj,rank2_anchored}=2928  → PASS (matches prompt formula exactly).

## Task 3 — conclusion checks (global ratio table rebuilt independently; 0 mismatches vs summary.json, >1e-9)
- (i) rank1_anchored ∈ [0.95,1.10] at ALL h for window 8 & 16: observed 0.9911–1.0333  → CONFIRMED.
- (ii) rank1_traj ≥ 2 at h=1 for every window_spec: 8=3.94, 16=6.76, 32=11.11, prefix=21.72  → CONFIRMED.
- (iii) naive_last2 < 0.5 at h=1 (0.4634) and > 1.15 at h=20 (1.1935)  → CONFIRMED.

## Task 4 — paper-consistency spot check (L13.mlp.gate_proj, prefix, anchor 119)
Reported evr1=0.917522, coef_r2_1=0.850813. Subsampled-x4 recompute (30 of 119 window ticks):
evr1=0.917704 (Δ=0.0002), coef_r2_1≈0.845741 (Δ=0.0051). Both Δ ≤ 0.02  → CONSISTENT.

## Task 5 — run log
`/workspace/rank1_emit2.log` (45 lines): gram cache HIT, 61/61 matrices swept, hold_stale identity gate
PASS (2.22e-16 / 900 rows), 19800 rows written. No fail/error/retry/warn/nan/inf/skip/traceback. Internal
audit (3 matrices) PASS. Concurrent `rank1e` session absent from `tmux ls`; no interference. Box load ~3.3.

## Honest note (not a defect in the emitted numbers)
The tool's own diagnostic reports coef_r2_1 median = 0.9793 with only 48.98% of matrices > 0.98 — i.e. it
does NOT clear the paper's ">0.98" bar, and the log states this plainly. This is a faithfully reported
finding, not a computational error; the emitted values are correct.

## Scope
All recomputes ran on the box (16-core CPU); scratch only under /workspace/tmp_rank1_verify/. Read-only
everywhere else. This verdict is the sole local write (runs/RANK1-ANALYSIS/).
