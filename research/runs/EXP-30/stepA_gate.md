# EXP-30 Step-A gate evaluation

source: metrics/stepA_fires.jsonl · fires total=8 post-warmup=7
rules: .claude/plans/30.md §Pre-registered gates (VERBATIM; thresholds untouched)

## GATE-B1 (blend, cell B1): **CLOSED**

- median-over-fires m1_matrix_median = 0.0121  (>= 0.10? False)
- median m2 (old-M null) = 0.0036; median(m1) >= 2*median(m2)? True
- paired per-fire m1 >= 2*m2 fraction = 0.57  (>= 0.80? False)

## GATE-B2 (delayed_ef, cell B2): **OPEN**

- median-over-fires m5_ratio_matrix_median = 1.0528  (in [0.1, 1.5]? True)
- max loss_mismatch_nats = 0.0103  (<= 0.02? True)

## H_decorr context (m4 lag-autocorrelation medians, post-warmup)

j=1 0.0864 · j=2 0.2002 · j=3 0.1147 · j=4 0.2945 · j=5 0.1685

## Per-fire table (post-warmup)

| step | tick | m1 | m2 | m3 | m4_j4 | m4_j5 | m5_ratio | m5_cos | m6 | m7_srank | m7_top1% | loss_mismatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 10 | 0.0147 | 0.0035 | 0.6170 | 0.1042 | 0.1322 | 1.3851 | -0.7218 | 0.9999 | 1.96 | 0.6002 | 0.0103 |
| 8 | 15 | 0.0046 | 0.0133 | 0.5852 | 0.2945 | 0.1685 | 1.1375 | -0.8802 | 0.6169 | 2.03 | 0.5931 | 0.0094 |
| 10 | 20 | 0.0121 | 0.0036 | 0.6216 | 0.1486 | 0.2000 | 1.0373 | -0.9620 | 0.5856 | 2.05 | 0.5792 | 0.0097 |
| 13 | 25 | 0.0253 | 0.0133 | 0.6392 | 0.3467 | 0.6747 | 1.0176 | -0.9809 | 0.6217 | 1.90 | 0.5899 | 0.0097 |
| 15 | 30 | 0.0148 | 0.0266 | 0.6114 | 0.0579 | 0.1266 | 1.0528 | -0.9504 | 0.6283 | 2.02 | 0.5983 | 0.0095 |
| 18 | 35 | -0.0029 | -0.0031 | 0.7575 | 0.6392 | -0.0001 | 1.0735 | -0.9236 | 0.6220 | 1.88 | 0.6079 | 0.0097 |
| 20 | 40 | 0.0045 | -0.0010 | 0.7335 | 0.3728 | 0.3570 | 1.0307 | -0.9727 | 0.7509 | 1.77 | 0.6061 | 0.0088 |

(m6 at fire 2 shares the tick-5 replay pair with fire 1 — structural artifact, real cross-pair values start fire 3.)
