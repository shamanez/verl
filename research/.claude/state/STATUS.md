# Research Status — 2026-06-04T03:05:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 20 | PowerSGD-style PP activation compression (M6) | RUNNING — stalled→fixing | 1×4×H200 (i_39319060, $15.21/hr) | — | probe sub-1 PASSED GPU hard gate (q_cond≈1.0, rec_err 0.97→0.72, logical_pp_bytes=102, 4 GPUs live, no NaN/OOM/flat_param); chain STALLED on a launcher early-stop-watcher leak; M-capture verified correct; runner re-engaged to fix watcher + restart |
| 19 | M5 — surpass dense baseline (epic) | UNCLAIMED | — | — | labels `kind:experiment`+`milestone:M5` only; no `research:claim`/`status:*`/plan → awaiting triage |
| 18 | M4 curve-match | TORN_DOWN | — | — | torn down 2026-06-03, no-heartbeat-30min |
| 21 | reweight on fixed anchor | TORN_DOWN | — | — | torn down 2026-06-03, no-heartbeat-30min |

## Last tick
2026-06-04T03:05:00+10:00 · running=[20 stalled→fixing] · analyzing=[] · logging=[] · blocked=[] · unclaimed=[19]

## Active finding — EXP-20 launcher stall (runner fixing)
- **Symptom:** PowerSGD probe sub-1 ran clean, then the back-to-back chain hung; sub-probes 2-3 + the 50-step sweep never started; 4×H200 idle while tmux alive.
- **Root cause:** `vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh` early-stop watcher (`tail -n +1 -F | grep -m1 -nE "$EARLY_STOP_RE"`, ~L335-351). On a clean run grep never matches, `tail -F` follows forever, and the EXIT trap kills the subshell PID but not the child `tail` → pipe stays open → `launch.sh` blocks. **Re-hangs after EVERY clean cell** → needs a real launcher fix (`tail --pid=$TRAIN_PID` or kill the watcher pgroup), not a one-time `kill`.
- **Action:** runner re-engaged (continue_in_place_iteration) to fix the watcher on exp/20, restart the chain with the fixed launcher, re-confirm the full hard-gate probe, then run the sweep.

## M-capture correctness (operator's "wrong M" FSDP check) — VERIFIED CORRECT
- The mask's prior fix = rmpad **token-axis alignment** (build sample_ids/position_ids in packed `input_ids.values()` order via cu_seqlens; require `is_nested`) — a *per-token-keying* fix.
- PowerSGD **reuses** the mask's `find_decoder_layers` + `decoder_boundary_indices`, hooks the **same** boundary blocks, captures the **same** `output[0]`, and is **order-invariant** (shared basis; `M@Q` per-row, sketch `MᵀMQ` sums over tokens) → immune to that bug class.
- Run config confirmed: `use_remove_padding=True`, `ulysses_sequence_parallel_size=1`, powersgd-only codec fires (mask does not double-compress). M = packed real tokens.
- **One latent gap (hardening, not biting now):** PowerSGD lacks the mask's explicit rmpad (`is_nested`) guard, so it would *silently* compress padded M if run without rmpad. Asked the runner to add the assert. Not a current-run issue (rmpad is on).

## GPU-utilization watch (operator standing instruction)
- Treat sustained idle GPUs (all ≤~5-10% while tmux ALIVE) as a stall/failure — don't wait for the monitor timeout. This stall was exactly that signature. Saved to memory; folded into the runner re-engage + future monitor dispatches.

## Budget
$/hr now: $15.21 (1 instance, EXP-20) · max_dph $24 (OK) · max_gpu_hr 96 (OK) · idle-while-stalled cost is the thing to minimize → runner unblocking now
