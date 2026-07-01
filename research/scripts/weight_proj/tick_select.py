#!/usr/bin/env python3
"""weight_proj/tick_select.py — per-step vs per-tick tick selection (## Notes for runner).

The EXP-43 trace is PER-TICK: 2 ticks per global_step, ticks 0..159, global_step 1..80.
Per the plan (`## Notes for runner` "Per-step vs per-tick"): the MAIN predictor families
run PER-STEP, and the per-step trajectory is subsampled by taking the FIRST tick of each
global_step — ticks 0,2,4,...,158 (the even ticks). Per-tick (all ticks) is the finer-Δ,
NOISIER (catastrophic-cancellation) regime reserved for the floor sensitivity study.

`select_ticks(cadence, want)` returns the first `want` ticks at the requested cadence:
  cadence="per-step" -> [0,2,4,...]   (first tick of each global_step; the default)
  cadence="per-tick" -> [0,1,2,...]   (every tick; noisier single-tick deltas)
"""
from __future__ import annotations


def select_ticks(cadence: str, want: int, max_tick: int = 159) -> list[int]:
    """First `want` ticks at the given cadence, clipped to the available trace."""
    if cadence == "per-tick":
        stride = 1
    elif cadence == "per-step":
        stride = 2                      # first tick of each global_step = even ticks
    else:
        raise ValueError(f"unknown cadence {cadence!r} (want per-step | per-tick)")
    ticks = list(range(0, max_tick + 1, stride))
    return ticks[:want]
