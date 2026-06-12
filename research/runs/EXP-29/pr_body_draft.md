# [EXP-29] Anchor on-policy replay — pair stale weights with the trajectories they generated (+ CPU-resident snapshots)

Plan: `research/.claude/plans/29.md` (operator-directed; this PR is the record).

## Problem

The anchor refresh loads the `delay_K`-stale weight snapshot but forwards the **current**
tick's batch: `G_anchor = ∇θ[-A·logπ_θ]` at `θ_{t−K}` on samples from `π_current` with no
importance correction — neither the stale policy's true gradient nor the current one's. For
the decentralized-PP target, a stale worker computes gradients on trajectories generated
from **its own** weights. Additionally the legacy snapshot ring was GPU-resident
(`delay_K+1` ≈ 6 full bf16 snapshots ≈ ~18.6 GB HBM).

## What this adds

- `comm_eff.anchor.replay_paired_batch` (default **false** = byte-identical legacy path):
  per-tick deep-cloned batch ring (CPU) + ONE generator-weight snapshot per global step,
  taken at its first `train_batch` tick — exactly the weights vLLM generated that step's
  rollouts from. At fire time the anchor replays `(batch[t−delay_K], gen_snapshot)`.
  Hard asserts: exact `t−delay_K` pairing post-warmup; weights never fresher than `delay_K`.
- `comm_eff.anchor.snapshot_device` (default **gpu** = today's behavior): `cpu` moves the
  snapshots off HBM in both modes (byte-preserving bf16 round trip, numerics-neutral).
- `[anchor-canary]` value-level staleness verification: push-time fp32-on-CPU (norm, sum)
  of 2 target matrices, hard-asserted **bitwise** off the clone after every snapshot load.
- `comm_eff/anchor_replay_fires` counter/metric; `[stale-replay]` log lines with
  `used_tick / batch_gs / snapshot_gs / realized_step_delay / warmup_fallback`.

Off path (`replay_paired_batch=false` + `snapshot_device=gpu`): no ring constructed, no new
collective, no RNG — every existing line unchanged.

## Validation (plan EXP-29, all hard gates)

- **CPU suite** (laptop): `tests/workers/comm_eff/ + tests/workers/config/test_comm_eff_config.py` — **187 passed**.
- **GPU suite** (4×H200 box, torch 2.11+cu130): same suite — **187 passed**.
- **25-step GPU smoke** (`exp29_replay_smoke`, EXP-27 substrate + the two new knobs,
  capture OFF per diagnostics policy):
  - `[anchor-canary]` … `match=True` on every fire — RESULT_CANARY
  - `[stale-replay]` exact pairing post-warmup (`used_tick == step−5`, `warmup_fallback=False`) — RESULT_PAIRING
  - anchor isolation unchanged (clone load N/N, coverage set-equality, `anchor_optimizer_steps==0`, mask delta 0) — RESULT_ISOLATION
  - `perf/max_memory_allocated_gb` max = RESULT_MAXMEM (< 57.9 EXP-27 healthy-phase baseline)
  - `comm/bytes_ratio` = RESULT_BYTES (≈ 0.0505, codec untouched)
  - val@25 (`val-core/openai/gsm8k/acc/mean@1`) = RESULT_VAL (gate ≥ 0.60; EXP-27 ref 0.7134 — replay changes the science, this is a not-broken gate)
  - mean step time RESULT_STEPTIME (≤ EXP-27 + 1.5 s)
  - 25/25 steps, no NaN/OOM/E1 length-explosion flag

## Files

- `verl/workers/config/comm_eff.py` — the two knobs + validation
- `verl/workers/comm_eff/anchor.py` — `AnchorReplayRing`, `clone_batch_for_replay` (NJT-safe),
  `maybe_build_replay_ring`, `snapshot_canary` / `verify_canary_on_module`
- `verl/workers/engine/fsdp/transformer_impl.py` — gated replay branch in
  `_maybe_comm_eff_anchor_refresh` + canary assert
- `verl/workers/comm_eff/state.py` — `anchor_replay_fires`
- tests: new `test_anchor_replay_ring.py`; `test_anchor_pg_loss.py` replay-clone round-trip
  identity; `test_comm_eff_config.py` knob validation

🤖 Generated with [Claude Code](https://claude.com/claude-code)
