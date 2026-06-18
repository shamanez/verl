# comm_eff.spectral.diagnostics — neutrality review (commit 3300cc61)

**VERDICT: NEUTRAL — the knob changes ZERO optimization math.** Independent adversarial static review (agent a59e9b1, 2026-06-18), read-only, `git show 3300cc61:<f>` (post) vs `3300cc61^:<f>` (parent).

## (a) default `diagnostics=true` == parent, byte-for-byte
Only semantic additions across the 5 files: the `diagnostics: bool = True` field + `__post_init__` bool-validation (comm_eff.py), the `diagnostics: true` actor.yaml mirror, the `SpectralFilter` kwarg + `self.diagnostics` (spectral_filter.py:125), the state.py:605 threading, the transformer_impl.py:1369-1371 diag-var, and 7 `if <diagnostics>:` guards wrapping pre-existing diagnostic blocks. No reordering; ring push/pop + cold-M counter sit ABOVE the gate in both parent and post; canary verify+assert straddle the gated print identically.

## (b) `diagnostics=false` skips ONLY diagnostics — optimizer path unconditional (ALL PASS)
writeback(grad,g_proj) [spectral_filter.py:1226-1227, inside no_grad, after the gate]; merger math g_corr for inject/blend/signed_ema/ef_powersgd/delayed_ef (signed_ema/ef/delayed matrix fns NOT in diff at all); anchor EMA M_anchor / beta_anc (untouched, 0 diff hits); PowerSGD V-sketch/orth(V)/Q/broadcasts (untouched); counters spectral_corrections/merger_coldM_fallbacks/residual_reset/anchor_backwards/anchor_q_updates/bytes_ratio (unconditional or untouched — anchor_backwards/bytes_ratio appear ONLY in comment prose); delayed_ef ring push/pop (above the gate); canary verify_canary_on_module + assert _can_ok (only stdout echo gated).

## (c) relevance-probe RNG analysis — DOES NOT DRAW RNG (the load-bearing finding)
The probe does NOT run an extra forward: `_wrap_anchor_loss_with_replay_relevance` reuses `model_output["log_probs"]` from the forward `_forward_backward_batch_inner` runs anyway; `replay_relevance_stats` = `(log_probs.detach().fp32 - ref.detach().fp32).abs()` masked-mean `.item()`. Grep `rand|dropout|sample|manual_seed|generator|normal_|uniform_|bernoulli|backward|copy_|add_|mul_` over the wrapper+stats = NOTHING. Wrapper RETURNS the loss bit-identical (probe is a try-wrapped scalar append side-effect; exception → WARN + untouched loss). With diagnostics=false the UNWRAPPED loss feeds the backward ⇒ anchor grad → G_anchor → M EMA bit-identical. Install is purely additive (gated at `replay_mode and diagnostics`).

## (d) residual notes (all benign)
- `state.spectral_rel_change` written only in the gated block, read only by `spectral_metrics` (W&B dict) — nothing in the optimizer path consumes it; diagnostics=false just drops the `comm_eff/spectral/rel_change_*` W&B keys (visibility, not optimization).
- `_relevance_acc` → geometry-probe stash only if the SEPARATE `comm_eff.probe.geometry_enabled` is on (off on production); telemetry-only, zero optimizer steps.
- `_subbasis_gamma()` moved inside the gate but is a pure function + only called when delta_subbasis_rank>0 (=0 default).
- Static review (no execution); rests on the merger/EMA/PowerSGD fns being absent from the diff + gated locals consumed only by gated prints/metrics.

**Conclusion:** default-true == parent byte-for-byte; diagnostics=false skips only diagnostic prints, the per-matrix rel_change GPU→CPU sync, and the probe's scalar re-score — no RNG, no extra forward, no shared-state mutation. Predicts the empirical confirmation run (exp-36-c2eff-055-diag @0.55) reproduces EXP-35 C2's 0.7528, and that the EXP-36 0.7043 came from the non-neutral knobs (0.75/chunked_prefill) or variance, NOT diagnostics.
