# Research Status — 2026-06-04T03:55:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 20 | PowerSGD-style PP activation compression (M6) | RUNNING — restart pending (sync_basis fix) | 1×4×H200 (i_39319060, $15.21/hr) | — | launcher stall fixed; rerun progressed to rank-H sub-probe (GPUs 93-100%, launcher fix confirmed working). Now applying `sync_basis=true` (operator decision) + collective-safety hardening, then restart → full probe → 50-step sweep |
| 19 | M5 — surpass dense baseline (epic) | UNCLAIMED | — | — | no `research:claim`/`status:*`/plan → awaiting triage |
| 18 | M4 curve-match | TORN_DOWN | — | — | no-heartbeat-30min |
| 21 | reweight on fixed anchor | TORN_DOWN | — | — | no-heartbeat-30min |

## Last tick
2026-06-04T03:55:00+10:00 · running=[20 restart-pending] · analyzing=[] · logging=[] · blocked=[] · unclaimed=[19]

## Basis cross-DP sync — DECISION: apply sync_basis=true (runner applying)
- **Operator intent (clarified):** the PowerSGD basis `Q` is a single shared codebook that must differ ONLY per layer-boundary and be IDENTICAL across DP ranks. DP *training* is not modified — only the codebook is synchronized.
- **Why it was wrong:** `sync_basis=false` + the actor dispatch (`make_nd_compute_dataproto_dispatch_fn`, scatters a different data shard per rank) ⇒ each rank builds `V=(MᵀM)Q` from its own shard ⇒ the 7 per-boundary `Q`'s DIVERGE across the 4 ranks after the first update (the code comment's "lockstep under identical data ordering" is false under DP).
- **Fix:** `sync_basis=true` → all-reduce `V` across the DP group (H×r per boundary, once per non-clean update) → consensus `Q` identical on every rank. Makes the plan's hard-invariant #4 actually true (previously unverifiable). Runner also: (2) verify the all-reduce group is the DP group (default world group is correct for this FSDP/SP=1/no-TP-PP actor); (3) collective-safety — iterate fixed boundary list, zero-fill missing, so the all-reduce can't deadlock; (4) on-box assert `Q` bit-identical across ranks after one update.
- Restart from the probe (sweep hasn't started → cheap), then 50-step sweep.

## M-capture correctness (operator's FSDP checks) — VERIFIED CORRECT
- The "wrong M" FSDP fix the operator recalled = the **anchor-clone-on-random-weights** bug — moot here (anchor + spectral OFF, no clone created).
- Activation-side path verified: PowerSGD reuses the mask's boundary helpers, hooks the same blocks, captures the same `output[0]`, order-invariant; `use_remove_padding=True`, SP=1, powersgd-only codec (no double-compress). rmpad `is_nested` guard added.
- Basis update verified: init once/boundary (det. seed), no-gradient subspace update via block power iteration `V=(MᵀM)Q → orth(V)` (reduces reconstruction error, Eckart–Young); clean step every k disables BOTH forwards (`mask_active = not clean_step`, `is_clean_step = gs%cadence==0`) and skips the Q update.

## Monitoring
- `monitor-exp20-rerun` DIED on a harness watchdog ("no progress 600s") — a tooling stall, NOT a training failure (its last reading: GPUs 93-100%, chain healthy on the rank-H sub-probe). Will dispatch a fresh GPU-util-aware monitor once the runner hands back from the sync_basis restart. Next monitor: use bounded/timeout'd SSH snapshots (no blocking follows) to avoid the watchdog stall.

## Budget
$/hr now: $15.21 · max_dph $24 (OK) · max_gpu_hr 96 (OK) · restarting early to avoid a full sweep on divergent bases (cheap now)
