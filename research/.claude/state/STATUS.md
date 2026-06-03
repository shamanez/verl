# Research Status — 2026-06-04T04:30:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 20 | PowerSGD-style PP activation compression (M6) | RUNNING (sync_basis=true, monitor active) | 1×4×H200 (i_39319060, $15.21/hr) | — | committed `f748dbc1`; **Q bit-identical across all 4 DP ranks proven on-box (`q_cross_rank_max_rel_dev=0.0`, no deadlock)**; restarted tmux `exp-20-sync`, GPUs ramping; chain in probe (rank-H sub) → mask → powersgd → opt dense; 5-lens math-review team auditing in parallel |
| 19 | M5 — surpass dense baseline (epic) | UNCLAIMED | — | — | no research:claim/status/plan → awaiting triage |
| 18 / 21 | M4 / reweight | TORN_DOWN | — | — | no-heartbeat-30min |

## Last tick
2026-06-04T04:30:00+10:00 · running=[20] · math-review=[5 lenses] · analyzing=[] · unclaimed=[19]

## EXP-20 — sync_basis=true: DONE + VERIFIED (commit f748dbc1)
- All-reduce the RAW per-rank sketch `V` over the DP group then `orth` → consensus `Q` identical on every rank (orth scale-invariant). DP training untouched.
- DP group bound via `set_dp_group(get_data_parallel_group())` (==WORLD under SP=1; future SP>1/TP/PP reduces over DP subgroup only).
- Collective-safety: iterate fixed `sorted(boundary_indices)`, zero-fill missing → no deadlock.
- On-box proof of invariant #4: `verify_basis_agreement_across_ranks` all-gathers an fp64 Q checksum, RAISES on >1e-6 divergence → measured **0.0**. q_cond=1.0000003, recon=0.967<1, bytes=102, mask counters 0 (powersgd-only). +6 tests; 39 powersgd+config + 100 comm_eff tests pass.

## Math-validity agent team `powersgd-mathcheck` — COMPLETE ✅ (VERDICT: faithful + mathematically correct, NO INVALID; full report → runs/EXP-20/math_review_SYNTHESIS.md; HIGH-1 consensus verified on-disk max_rel_dev=0.0; open: HIGH-2 latent verifier-gate [non-blocking], q_cond/byte-metric MED [analyst], r=102 recon EMPIRICAL [→maybe REVISE r=205])
| Lens | Member | Focus | Status |
|---|---|---|---|
| Theory/core math | mathematical-checker | projector, power-iter→Eckart-Young, seed, byte budget, lossless, consensus | running |
| Autograd/graph | autograd-checker | no-STE backward, grad-ckpt recompute, no_grad V, FSDP flat-param | running |
| Distributed | distributed-correctness | DP-group, collective-safety/deadlock, consensus = orth(allreduce(V)) | running |
| RL objective | rl-grpo-checker | frozen-Q ρ≈1, clean-cadence, vanilla-GRPO invariance, train-inference gap | running |
| Numerics | numerics-stability | fp32-QR, q_cond, **activation-scale shrink ‖M_hat‖≤‖M‖ vs RMSNorm**, grad_norm warm-start | running |
- Each writes `runs/EXP-20/review_<lens>.md` + adversarially cross-flags the overlapping claims, then reports to lead. Lead synthesizes a combined verdict.

## Monitoring
- `monitor-exp20-sync` (bounded-SSH, GPU-util-aware) watching tmux `exp-20-sync`. Previous monitor died on a harness watchdog (tooling, not training).

## Budget
$/hr now: $15.21 · note: 5 Opus reviewers + monitor are token-heavy (operator opted in: "heavy issue, ADD ALL") · max_gpu_hr 96 OK
