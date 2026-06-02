# Candidate 4 (C4) — CLEAN anchor gradient (ratio≡1) + blend η=0.7 — the first VALID test

> **Why this is not "just another iteration":** C1/C2/C3 all degraded the policy, and the
> root cause is code-confirmed — the anchor's GRPO loss reuses the batch's **masked**
> `old_log_probs` with an **unmasked** new forward, so its importance ratio ≠ 1 and the PPO
> clip mangles the gradient. `M_anchor` was therefore NEVER the clean unmasked true gradient
> the M4 method assumes — every prior candidate tested a corrupted signal (and pre-fix, a
> random-weight clone). C4 fixes the anchor to emit the CLEAN policy gradient, then runs the
> first valid test of the correction. Branch `exp/18-anchorcleangrad-c5d5` from
> `exp/18-anchorblend-c5d5` (inherits canon-naming fix, blend mode, OOM fixes).

## The fix — anchor computes a plain policy gradient (ratio≡1)
The anchor does ONE fwd/bwd per refresh, so the PPO importance ratio is unnecessary and
harmful here. Replace the loss the anchor uses (currently the fast-path PPO `loss_function`)
with a plain policy-gradient loss equal to what PPO reduces to at ratio≡1:
```
loss_anchor = -(advantages * logπ_unmasked * response_mask).sum() / response_mask.sum().clamp(min=1)
# grad = -(A · ∇logπ_unmasked) = the clean unmasked policy gradient at the stale weights θ_{t-5}
```
This is "the clean step's gradient, but computed at the delay_K=5 stale weights" — exactly the
M4 premise. Implementation (runner picks the cleanest that fits the code; keep it inside the
plan's target_modules — `transformer_impl.py` + `comm_eff/`):
- **Preferred:** define `anchor_pg_loss(output, data)` in `verl/workers/comm_eff/anchor.py`
  (mirror how the fast-path loss extracts per-token `log_probs` from the forward output, then
  do the PG formula above — NO ratio, NO clip, NO `old_log_probs`). In
  `_maybe_comm_eff_anchor_refresh`, pass `anchor_pg_loss` to `_forward_backward_batch_inner`
  instead of the fast-path `loss_function`.
- **Alt (if extracting log_probs cleanly is hard):** ratio≡1 trick — capture the anchor
  forward's `log_probs`, set `anchor_data["old_log_probs"] = log_probs.detach()` so the
  existing PPO loss sees ratio=exp(logπ−logπ.detach())≡1 → unclipped → grad = clean PG. (Costs
  a forward to get log_probs first; acceptable.)
The fast path (real PPO loss with multi-inner-step ratio/clip) MUST be untouched — the change
is anchor-only.

## CPU test (MUST pass before launch)
Add a test: with a tiny model, the `anchor_pg_loss` gradient equals `-(A·∇logπ)` (compare to a
hand-computed PG on a 2-token toy), and the anchor pass logs ratio≡1 (or no ratio). Existing
spectral/blend/canon tests still pass.

## Diagnostic to confirm the fix on the box
After the first anchor refresh, the log should show the anchor's effective ratio ≈ 1 (or the
PG path active). Compare `||dM_anchor||`/`cos(G_mask,M_anchor)` to C3's — the clean gradient
may have a different magnitude/cosine; that's expected (it's a different, correct vector).

## Launch (reuse box 39132674; inherits ALL prior fixes)
EXPERIMENT_NAME=`curvematch_cleangrad_blend_c5_d5`. Same env as C2/C3 (blend η=0.7,
ema_device=cpu, max_targets=-1, seed_anchor_cache=false, 18432, anchor c5/d5, clean=0) — the
ONLY change is the new clean-gradient anchor loss (code), correction_mode=blend, blend_eta=0.7.
Pins (INVALID if violated): ANCHOR_DELAY_K=5, CLEAN_CADENCE=0, ANCHOR_CADENCE=5, MAX_RESPONSE 16384.

## Read (the decisive experiment)
- **Reward LIFTS toward dense** (steps 5–20 climb off the ~0.13 floor) → the ratio-corruption
  WAS the blocker; the clean stale gradient drives learning. Then tune η/cadence (follow-on),
  likely PASS-path. **Core goal in reach.**
- **Reward still degrades/flat** → even the CLEAN delay_K=5 stale gradient cannot reproduce
  dense → STOP with a definitive negative finding: the realistic-staleness (delay_K=5) target
  is unreachable by forcing toward the stale true gradient; the clean@K existence proof does
  not survive staleness. (Then the next-cycle direction is a staleness-correction, e.g.
  gradient extrapolation/momentum-debiasing across refreshes — candidates.md C4 axis.)
