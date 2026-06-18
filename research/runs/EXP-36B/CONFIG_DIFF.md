# EXP-36B confirmation run — config diff vs EXP-35 C2

**Run:** `exp-36-c2eff-055-diag` (project `verl_compression_research_accel_rebaseline`)
**Purpose:** isolate whether `comm_eff.spectral.diagnostics=false` is neutral — i.e. does it reproduce EXP-35 C2's proven **val@50 = 0.7528**? Single-variable test.
**Reference:** EXP-35 C2 = signed_ema α=0.25 / β_anc=0.50, gpu_mem_util=0.55, no efficiency knobs, val@50=0.7528.
**Code:** `exp/spectral-diagnostics-knob` @ `3300cc61` (diagnostics=true default is byte-identical to EXP-35 pre-knob code — diff-audited + 293 comm_eff tests pass).

## Reverted back to EXP-35 state (were changed in the EXP-36 0.75 run → undone)
| Knob | EXP-36 0.75 run (val@50 0.7043) | reverted to (EXP-35) | reason |
|---|---|---|---|
| gpu_memory_utilization | 0.75 | **0.55** | zero measured speedup; non-bit-neutral |
| rollout.enable_chunked_prefill | true | **false** | numeric-only (alters rollout realization); ~no benefit |
| fsdp forward_prefetch (actor + ref) | true | **off** | report-flagged "verify bit-identical" — would confound the isolation; sub-min benefit |

## Changed vs EXP-35 (kept in the confirmation run)
| Knob | EXP-35 C2 | confirmation run | result-affecting? |
|---|---|---|---|
| comm_eff.spectral.diagnostics | (on / knob absent) | **false** | **YES — the variable under test** (the real speed lever: removes ~196 per-step .item() syncs + prints + relevance-probe forward) |
| trainer.save_freq | 50 | 0 | No — only skips the step-50 checkpoint write |
| HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE | unset | on | No — only network/init (model already cached) |

## Unchanged (identical to EXP-35 C2)
signed_ema · α=0.25 · β_anc=0.50 · update_weights_bucket=4096 (already 4096 in EXP-35) · accel surface (use_dynamic_bsz, max_response=2048, TP=1, ppo_max_token_len=24576) · train_batch=128 / mini=64 / n=8 · lr=1e-6 · total_steps=50 · test_freq=25 · val_before_train=False · ema_device=cpu · anchor.snapshot_device=cpu · PowerSGD r=77 + anchor (cadence=5/delay_K=5/owns_q/replay_paired_batch).

## Verdict rule
- val@50 ≈ 0.7528 → `diagnostics=false` confirmed NEUTRAL → promote it (the only validated speed lever) at 0.55 → push → teardown.
- val@50 ≈ 0.7043 (or materially off 0.7528) → the diagnostics gating (relevance-probe-forward RNG) or the accel surface is the real cause → do NOT promote → investigate → push data → teardown.

## Context (the comparison set, all on the accel surface)
- EXP-35 C2 (orig surface, 0.55, no eff-knobs): val@50 = 0.7528
- EXP-36 dense (0.75, comm_eff OFF): val@50 = 0.7695 @ 11:29  ← new accel-surface dense reference
- EXP-36 c2eff (0.75 + chunked_prefill + forward_prefetch + diag-off): val@50 = 0.7043 @ 24:45  ← the regressed/confounded run this test disambiguates
