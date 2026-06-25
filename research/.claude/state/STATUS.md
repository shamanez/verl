# Research Status — idle

No experiment in flight. The orchestrator rewrites this file each tick when running.

## Focus

Two priorities, both **GPU-free offline kill-gates** (no Vast.ai run until a gate passes):

1. **Solve the k-collapse by projecting the weights** (Priority 1 / milestone M4) —
   `reports/priority-1-anchor-staleness-k-collapse.html`.
2. **Reduce the compression-induced train–inference mismatch (Gap A)** (Priority 2 / milestone M6) —
   `reports/priority-2-compression-train-inference-mismatch.html`.

## Baseline (the problem state)

`signed_ema` (α=0.25, β_anc=0.50) on the fast 1K surface at HIGH anchor latency
(cadence/delay_K = 20/20, the k-collapse regime), PowerSGD r=77 anchor substrate.
Values: `runs/FIXED_CONTROL_SURFACE.md`.

## Issue pipeline

(none in flight)

## Last tick
(idle) · running=[] · analyzing=[] · logging=[] · blocked=[]
