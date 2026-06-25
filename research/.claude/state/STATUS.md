# Research Status — 2026-06-25

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 41 | M4 look-ahead anchor (delay_K=20, fixed-linear) | DONE | external 4×H200 i_42465843 (team) TORN_DOWN | STOP | probe PASSED 10/10 hard invariants (code correct); cell A 5/5 ref clean (val@100=0.7066); cell B collapsed (val@100=0.0478, 8 resp_len breach steps, no catastrophic ignition); lift +0.0267 present but merger over-amplified; cell C gated off; WandB A=7tbzm9kl B=g6dt6bza |

## Last tick
2026-06-25 · running=[] · analyzing=[] · logging=[] · blocked=[]

## Pipeline state
Idle. No experiment in flight. EXP-41 is the most recent closed experiment (STOP).

## EXP-41 close-out summary
- ✅ Fire-forcing invariant probe (10/10 hard gates PASS, runs/EXP-41/probe-invariants.md)
- ✅ Cell A (onsurface 5/5 reference, 100 steps clean): val@25/50/75/100 = 0.6998/0.7255/0.7233/0.7066; raw-stale anchor_align_cos mean +0.0063
- ✅ Cell B (fixed_linear 20/20, 100 steps, no NaN): anchor_align_cos lift +0.0267 (6/8 fires positive, peak +0.131 @ step 60); BUT response_length/mean breached 2x threshold at 8 steps (peak 552 @ step 59); val crashed 0.498->0.115->0.048
- ✅ WandB backfilled both cells to step 100 (A=7tbzm9kl, B=g6dt6bza, project verl_compression_research)
- ✅ Box 42465843 torn down (team), 0 live instances verified
- VERDICT: STOP — no-collapse criterion FAILS, val@100 band criterion FAILS; cell C gated off per plan on_fail
- Deferred research direction (operator review): lower beta_anc (0.50->0.10-0.25) with fixed-linear held on at 20/20; merger over-amplification is now the suspect, not the projector

## Budget
All GPU spend complete. 0 live instances. No active spend.
