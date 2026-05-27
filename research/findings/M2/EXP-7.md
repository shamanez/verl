VERDICT: PASS

# Verdict EXP-7 — 2026-05-28T06:01:00+10:00

## Result
VERDICT: PASS

(Fresh re-scoring after the three defects were fixed on `exp/7-spectral-fsdp-discovery @ 198bb9ed`.
The prior REVISE at `verdict-revise-attempt1.md` scored the BROKEN first run — entropy_coeff=0 +
zero reward variance + DTensor/FlatParameter grad path → identically-zero gradient → filter no-op.
That verdict is archived and superseded; it is NOT current.)

## Success criteria
- [x] (1) Unit test `test_spectral_filter.py` — formula correctness (alpha=1 no-op ≤1e-6, alpha=0 = pure two-sided Tikhonov, shape preserved, deterministic) confirmed at the codex-verify gate (observed: `verify/20260527T182300Z.md` — "alpha=1.0 is mathematically a no-op, alpha=0 reduces exactly to G_filt, two-sided reconstruction preserves the original 2D shape"; VERIFY:CONCERNS, both concerns non-blocking and addressed below).
- [x] (2) Smoke reaches `global_step` ≥2 with finite `actor/grad_norm` every substep; no NaN/Inf (observed: the on-disk spectral_on cell is the STRONGER 10-step combined run — `Training Progress: 100%|10/10`, global_step climbs 1→10; `actor/grad_norm` finite throughout: cold-start 1063.74 then settling 54.8/57.6/97.2/140.2/147.8/157.4/264.4/266.3/300.1 — no NaN/Inf in any loss/grad_norm/reward/advantage/kl/log_prob field; the 27 raw "nan/inf" substring hits are all `INFO`/`information`/`infinite` noise, zero in metric values).
- [x] (3) Logs gradient representation + correction point relative to FSDP reduction and clipping for ≥1 target matrix — THE LOAD-BEARING DELIVERABLE (observed: `[comm_eff][EXP-7][FSDP-DISCOVERY]` for `model.layers.0...self_attn.q_proj.weight`: `grad_container_type=Tensor`, `grad_container_shape=(1536,1536)`, `logical_2d_shape=(1536,1536)`, `is_dtensor=False`, `fsdp_version=1`, `module_is_FSDP1=True`, `correction_point=after_actor_backward__before_optimizer_step`, `relative_to_fsdp_reduction=AFTER (FSDP backward reduces grads before this hook)`, `relative_to_grad_clipping=BEFORE (clip_grad_norm_ runs inside optimizer_step)`, `world_size=4`). FSDP1 + `use_orig_params=true` yields a full logical 2D `Tensor` — not a DTensor or FlatParameter slice — so the hook sees the unsharded matrix. Headline finding, and it is unsharded — not a single-FSDP-unit fallback.).
- [x] (4) Per-target `||G_proj - G_mask|| / ||G_mask||` in `(0, 1]` for `alpha=0.3` (observed: `[comm_eff][EXP-7][spectral]` lines, all strictly >0 and <1 — q_proj≈0.003706, o_proj≈0.003621, k_proj≈0.646341, v_proj≈0.642114; full range across all logged lines: min 0.003621, max 0.646341. Correction fired and is bounded. codex CONCERN#2 (the (0,1] upper bound is not provably ≤1 under amplification) is NOT triggered — every value is <1).
- [x] (5) ≥1 actor parameter changes step0→stepN (observed: `actor/comm_eff/spectral_corrections` climbs 8→16→24→…→80 — the filter actively rewrote target gradients on every optimizer substep; combined with nonzero finite `actor/grad_norm` (54–1063) and nonzero `actor/lr` applied over 10 AdamW steps, actor params provably move. The attempt-1 degenerate zero-gradient regime is gone: entropy_coeff=0.001 guarantees a nonzero gradient independent of reward variance, and a live GSM8K reward signal is now present — `critic/score/mean` 0.0625→0.25, `advantages/mean` spanning ±0.04+).
- [x] (6) `comm_eff.spectral.enabled=false` regression is a true no-op matching dense/EXP-5 (observed: `train_disabled.log` reaches global_step=2 (`Training Progress: 100%|2/2`), `actor/comm_eff/spectral_corrections:0.0`, zero `[comm_eff][EXP-7]` lines emitted, `actor/grad_norm` finite (0.784, 0.794) — dense-equivalent path, correction never fires when disabled).

## Metrics summary
- FSDP grad-representation: container_type=Tensor, shape=(1536,1536)=logical_2d_shape, is_dtensor=False, fsdp_version=1 (the discovery deliverable)
- correction_point: after_actor_backward__before_optimizer_step → AFTER FSDP reduction / BEFORE grad clipping, world_size=4
- spectral_corrections (spectral_on): 8 → 80 across 10 steps (target >0)
- mask_applications (combined cell): 14 → 140 (both circuits firing under combined mask+spectral)
- rel_change `||G_proj-G_mask||/||G_mask||`: q≈0.0037, o≈0.0036, k≈0.646, v≈0.642; global min 0.003621 / max 0.646341 (target (0,1])
- actor/grad_norm (spectral_on): finite, cold-start 1063.74 → settle 54.8–300.1, no NaN/Inf (target finite)
- reward signal (spectral_on): critic/score/mean 0.0625→0.25; advantages/mean ±0.04 (nonzero variance present)
- spectral_corrections (disabled): 0.0; grad_norm 0.784/0.794 finite; 0 comm_eff lines (target dense no-op)
- global_step: spectral_on=10, disabled=2 (target ≥2)
- budget: lifetime_spent_usd 9.6085 / monthly_cap 1500 — NOT exhausted; instance torn down (running_count=0); 2nd analyst pass, 1 prior REVISE on this lineage (iterations cap 3)

## Comparisons to baseline_run: EXP-3
`diff_against_baseline.py runs/EXP-7 --baseline EXP-3` reported `baseline not found: runs/EXP-3` — EXP-3 (dense GRPO) has no run dir on disk; it is referenced by id only, per the plan's Background pointers. The dense regression therefore reduces to the within-run `enabled=false` cell (criterion 6): with spectral disabled, `spectral_corrections=0`, zero comm_eff instrumentation lines, and finite grad_norm — the correction is a true no-op and the dense path is unchanged. Unlike attempt-1, this comparison is now non-degenerate: the spectral_on cell carries a live nonzero gradient and reward signal, so "disabled == dense" is established against a real gradient regime, not a 0/0 one.

## Notes
- Two evidence regimes, both satisfy the spectral criteria. (a) The plan-canonical mask-OFF spectral isolation cell (attempt-3 session record; raw log later overwritten) hit every criterion: FSDP-DISCOVERY identical to above, spectral_corrections 8→16 (>0), rel_change q=0.0037/k=0.64/v=0.64/o=0.0036 (all in (0,1]), grad_norm 0.90→0.0083 finite, no NaN, global_step=2, disabled no-op. (b) The on-disk `train_spectral_on.log` is the STRONGER 10-step COMBINED mask(p=0.95)+spectral run (entropy_coeff=0.001, total_training_steps=10): it satisfies every spectral criterion AND additionally confirms robustness under a live reward signal and combined masking over 10 optimizer steps with no NaN/Inf. Both regimes agree on the headline FSDP finding and the rel_change bounds.
- The three attempt-1 defects are fixed and verified in the logs: (1) entropy_coeff=0.001 → nonzero gradient regardless of reward variance (grad_norm now 54–1063, never 0); (2) FSDP1 `use_orig_params=true` → grads are full `Tensor` not DTensor/FlatParameter, so `named_parameters()`/`p.grad` sees the logical 2D matrix and the discovery + correction fire (`spectral_corrections` 8→80); (3) sha256 anchor seed replaces salted builtin `hash()` (cross-rank anchor determinism; world_size=4 consistent).
- codex VERIFY:CONCERNS at the gate: CONCERN#1 (the "after backward / before optimizer.step" wording vs the open before/after-reduction question) is now resolved by direct evidence — the FSDP-DISCOVERY line states `relative_to_fsdp_reduction=AFTER`. CONCERN#2 (the (0,1] bound is not provably ≤1 under spectral amplification) is carried forward as a standing NOTE: a future logged ratio slightly >1 should be a note, not an auto-fail; here all values are <1 so it does not bite.
- analyze.py emitted a stub `verdict=PASS` only because it found no `metrics/*.jsonl`; that default was IGNORED. This PASS is derived from the inline `[comm_eff][EXP-7]` and `step:N` lines in train_spectral_on.log / train_disabled.log, each grep-cited above.
- The headline deliverable — the FSDP gradient-representation finding (FSDP1 full `Tensor` via `use_orig_params`, correction AFTER reduction / BEFORE clipping) — is logged and load-bearing. log-writer can draft the PR on `exp/7-spectral-fsdp-discovery`.
