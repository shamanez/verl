# Verdict — EXP-29 anchor on-policy replay (25-step mechanism smoke)

VERDICT: PASS

- plan: .claude/plans/29.md (operator-directed; no GH issue — PR on shamanez/verl is the record)
- branch: exp/29-anchor-onpolicy-replay @ 67acf3707 (3 iterations: d311904 impl → 933e79a actor.yaml
  Hydra-struct hotfix → c512128 fire-aware retention → 67acf37 relevance probe)
- cell: exp29_replay_smoke on i_40676027 (4×H200, operator-provided), 25/25 steps,
  W&B verl_compression_research/eyguqjh4, done.flag rc=0 2026-06-12T08:24Z
- substrate: EXP-27 resolved params (powersgd r77 sync_basis, anchor owns_q cadence=5 delay_K=5,
  clean=0, ef_powersgd ef_clip=0.5 ef_decay=0.5); ONLY deltas: replay_paired_batch=true,
  snapshot_device=cpu, steps 100→25, test_freq=25, capture OFF (diagnostics policy)

## Hard gates (all green)

| invariant | evidence |
|---|---|
| off-path parity | CPU suite 195 passed (flag-OFF builds no ring; snapshot_device=gpu default untouched); GPU-box suite 195 passed at 67acf37 |
| stale-weight VALUE canary | 20/20 `[anchor-canary] match=True` (fires at ticks 5..50; push fp32 norm+sum == clone recompute BITWISE through bf16→cpu→device) |
| exact pairing post-warmup | 9/9 post-warmup fires: `used_tick == step−5`, `batch_gs == snapshot_gs`, `warmup_fallback=False`; realized weight delay alternates 5/6 (=K/K+1, the derived pattern); warmup fire (tick 5) self-pairs exactly (data_delay=0, own gs snapshot) |
| anchor isolation unchanged | clone loads 20/20 full; coverage `set_equal=True` ×20; `anchor_optimizer_steps=0`; `anchor_mask_applications=0`; `anchor_grad_corrected=0` |
| zero GPU-memory growth | max `perf/max_memory_allocated_gb` = 30.77 < 57.9 (EXP-27 healthy-phase) — the ~18.6 GB legacy GPU snapshot ring is gone (CPU-resident; box RAM 1.5 TB, cpu_memory_used ~237 GB total) |
| backend integration | first fire (tick 5) survived NJT clone + CPU→device load under FSDP1/use_orig_params/bf16/grad-ckpt; 0 OOM / 0 AssertionError / 0 NaN through step 25 |

## Soft gates (green)

- `comm/bytes_ratio` 0.05037–0.05053 ∈ [0.045, 0.056] (codec untouched)
- mean step time 84.8 s ≤ EXP-27 healthy-mean 110.5 + 1.5 (FASTER: one per-gs snapshot replaces the per-tick GPU clone)

## Success criteria

- [x] every hard-gate invariant green (table above)
- [x] 25/25 steps, no NaN/non-finite, no OOM; no E1 (steps 10–25 len/max = 1126 < 4000;
      single stray cap-pin at step 4 is outside the gate window, lengths decline after)
- [x] val@25 `val-core/openai/gsm8k/acc/mean@1` = 0.7005 ≥ 0.60 (informational vs EXP-27 0.7134 —
      replay changes the science; this is the not-broken gate)
- [x] `comm/bytes_ratio` ∈ [0.045, 0.056]
- [x] mean step time ≤ EXP-27 + 1.5 s
- [x] local CPU suite green (195 passed)

## Operator-added evidence (mid-experiment scope additions, both green)

1. **Fire-aware ring retention** (c512128): retain only ticks ≡ −delay_K (mod cadence);
   `ring_batches≤2, ring_snapshots≤2` on every fire line (bounds asserted in code);
   80% of per-tick deep clones skipped.
2. **Relevance probe** (67acf37): per-fire masked mean |logπ(loaded stale weights) −
   rollout_log_probs stored WITH the replayed trajectories| = 0.0083–0.0105, FLAT across
   all 10 fires (incl. independent snapshots gs∈{3,5,8,10,13,15,18,20,23}) — the loaded
   weights are the trajectories' generator at the value level (engine-noise floor;
   contrast: codec-affected old-vs-rollout diff ~0.61–0.84 in the same run).

## Notes

- anchor_replay_fires=10 == anchor_backwards=10 == anchor_q_updates=10 (every fire replayed).
- The 2 Tracebacks at run end are the known WandB atexit teardown noise ("Exception ignored"),
  post-training, not failures.
- PASS = mechanism correct + memory-clean. NO claim about parity/surpass-dense (25 steps cannot);
  the science of self-consistent M belongs to a successor on 50–100 steps with controls.
- owns-Q semantic shift (V harvested from STALE batch at stale weights) behaved as the plan
  predicted: reconstruction_rel_error warmed 0.975 → 0.024, q_cond ~1.0000003, no divergence.
