# Research Status — 2026-06-04T03:18:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 20 | PowerSGD-style PP activation compression (M6) | RUNNING (rerun, monitor active) | 1×4×H200 (i_39319060, $15.21/hr) | — | launcher stall FIXED (9287994a); restarted in tmux `exp-20-rerun`, GPUs at 52%/59GB; full 3-sub-probe hard gate → mask → powersgd → opt dense; M-capture verified correct + rmpad guard added; monitor-exp20-rerun (bg) |
| 19 | M5 — surpass dense baseline (epic) | UNCLAIMED | — | — | labels `kind:experiment`+`milestone:M5` only; no `research:claim`/`status:*`/plan → awaiting triage |
| 18 | M4 curve-match | TORN_DOWN | — | — | torn down 2026-06-03, no-heartbeat-30min |
| 21 | reweight on fixed anchor | TORN_DOWN | — | — | torn down 2026-06-03, no-heartbeat-30min |

## Last tick
2026-06-04T03:18:00+10:00 · running=[20 rerun] · analyzing=[] · logging=[] · blocked=[] · unclaimed=[19]

## Resolved this cycle — EXP-20 launcher stall
- **Was:** PowerSGD probe sub-1 ran clean, then the chain hung (4×H200 idle, tmux alive) on the launcher early-stop watcher (`tail -F | grep -m1` orphaned a follower that blocked `launch.sh`); re-hangs every clean cell.
- **Fixed (commit 9287994a, pushed origin/exp/20-powersgd-activation):** (1) `tail --pid=$TRAIN_PID -F` so the follower dies when training exits; launcher `wait`s on training only + propagates `exit $TRAIN_RC` (a real arm failure now aborts rather than comparing on a broken baseline). (2) watcher under `setsid`; EXIT trap `kill -- -$PGID`. Verified control flow on the box.
- **Restarted:** box reset --hard to 9287994a (editable install → live), bundle refreshed, stale logs/flags cleared, tmux `exp-20-rerun`. Full ≤2-step probe (incl. rank-H lossless + off-path-parity sub-probes that never ran) now runs before the sweep.

## M-capture correctness (operator's FSDP check) — VERIFIED CORRECT / MOOT
- **Operator clarification:** the prior FSDP fix referred to was the **anchor-clone-on-random-weights** bug (nested-FSDP state-dict key mismatch — `._fsdp_wrapped_module` infix dropped — so the stale-anchor clone ran on uninitialized weights; confounded EXP-16). **Moot for EXP-20:** anchor + spectral are OFF (`anchor.enabled=false`, `spectral.enabled=false`), so NO model clone is created — that bug surface is absent. PowerSGD hooks the live model's real-weight boundary outputs; its random basis `Q=orth(randn)` is a deterministic codebook bootstrap (INF-13), NOT uninitialized weights.
- Independently verified the activation-side path too: PowerSGD reuses the mask's `find_decoder_layers`+`decoder_boundary_indices`, hooks the same boundary blocks, captures the same `output[0]`, and is order-invariant (shared basis) → immune to the per-token-keying/alignment bug class. `use_remove_padding=True`, SP=1, powersgd-only codec (no double-compress) confirmed. Hardening landed: PowerSGD now also asserts rmpad (`is_nested`).

## GPU-utilization watch (operator standing instruction — now applied)
- Saved to memory; baked into the runner's restart check (GPUs confirmed 52%/59GB) and the monitor dispatch (sustained idle GPUs while tmux ALIVE = stall, act immediately, don't wait for timeout).

## Watch (scientific, not a bug)
- Deeper boundary reconstruction error was 0.86–0.92 at the 2-step probe (vs 0.025 at the first boundary) — warm-start. If it stays ≥~0.9 with finite q_cond past warm-up in the 50-step sweep, that's the plan's REVISE-toward-r=205 signal.

## Budget
$/hr now: $15.21 (1 instance, EXP-20) · max_dph $24 (OK) · max_gpu_hr 96, full seq worst-case ~48 GPU-hr (OK) · stall idle cost ~$5, now computing
