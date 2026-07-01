"""weight_proj — GPU-free offline weight-projection sweep engine (EXP-44).

Package layout (architect §4.1):
  metrics.py      SOLE owner of the metric math (boundary B1 with #45)
  r2_stream.py    bounded-footprint streaming reader for the EXP-43 R2 trace
  predictors.py   family-pluggable predictor zoo + reconstruction/leakage contracts
  noise_floor.py  the bf16 round-trip noise-floor gate (replaces on-box parity)
  sweep.py        grouping + the (family x order x coeff x Delta x h) driver
  report.py       per-block intermediates -> combine -> self-test HTML
The CLI entry point is research/scripts/weight_proj_sweep.py.
"""
