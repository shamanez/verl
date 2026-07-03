"""weight_proj — GPU-free offline weight-projection sweep engine (EXP-44).

Package layout (architect §4.1):
  metrics.py      SOLE owner of the metric math (boundary B1 with #45)
  r2_stream.py    bounded-footprint streaming reader for the EXP-43 R2 trace
  predictors.py   family-pluggable predictor zoo + reconstruction/leakage contracts
  sampling.py     paper-style per-matrix coordinate sampling
  structure.py    matrix_name -> layer_idx / block_type / super_block taxonomy
  tick_select.py  tick/cadence selection helpers
  rank1_traj.py   RELEX-style rank-1 trajectory-SVD family (Gram-domain math)
The CLI entry points are research/scripts/moat_scorecard.py and
research/scripts/rank1_scorecard.py (the rank-1 trajectory lane).
"""
