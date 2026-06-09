# STUCK: EXP-26 — parallel uncompressed G_dense backward does not compose with codec-ON

Date: 2026-06-10  ·  Branch: exp/26-geometry-audit-ef-powersgd @ 973e21f9d  ·  Box 40242796 (warm)

## Bottom line
The plan's NAMED highest-risk failure surface #1 — the parallel uncompressed
`G_dense` backward in `verl/workers/engine/fsdp/transformer_impl.py`
(`_maybe_comm_eff_capture_g_dense`) — **does not produce a true dense gradient on
codec-ON arms**, even after a focused fix. This blocks H1 (`cos(G_dense, G_comp)`)
and H2 (`Q_act` update-capture vs a trustworthy `G_dense`). Per the plan's Step-A
`on_fail` ("inconclusive on a probe → STUCK: EXP-26 (design call)") and the
operator's explicit directive (one focused fix, then STUCK — do NOT grind), the
runner halts for an FSDP/autograd design decision. **H3 is already cleanly
answered** (see below) so the audit is not a total loss.

## The acceptance gate (the operator's machine-checkable bar)
On a codec-ON arm (powersgd r77), post-Q-warm:
`cos(G_dense, G_fresh_anchor@delay_K=0) >= 0.95` AND `norm_ratio in [0.8, 1.25]`.
(`G_fresh_anchor` is the realism-GREEN full uncompressed backward at the SAME
weights/batch — the TRUSTED dense reference. Two full backwards at the same
weights MUST be cos≈1.)

| arm | code | cos(G_dense, G_fresh_anchor) | median norm_ratio | gate |
|---|---|---|---|---|
| A0_dense (codec OFF) | any | +0.99 | ~1.0 | PASS (faithful — analyst-validated) |
| A1_powersgd_r77 (codec ON) | OLD (f1ce9185, broken) | −0.02 .. 0.44 | 0.047 (= r/H) | FAIL |
| GATE_probe powersgd r77 (codec ON) | **NEW (973e21f9d, Defect-6 fix)** | **0.24** | **0.28** (27/84 targets at r/H≈0.05) | **FAIL** |

The Defect-6 fix IMPROVED it (norm 0.05→0.28, cos −0.02→0.24) but did NOT clear the gate.

## What was fixed (Defect 6) and PROVEN correct — yet insufficient
- `_maybe_comm_eff_capture_g_dense` reused the SHARED `_anchor_module_cache`, onto
  which the anchor registers PowerSGD projection forward-hooks; `copy.deepcopy`
  (in `build_anchor_module`) copies a module's `_forward_hooks`, and that helper
  cleared backward/FSDP sentinels but NOT forward hooks.
- FIX (973e21f9d): a DEDICATED `_g_dense_clone`; strip ALL `_forward_hooks` /
  `_forward_pre_hooks` (+with_kwargs) from the clone and every submodule; HARD
  ASSERT zero residual forward hooks before the backward; defensively unregister
  the live compressor.
- ISOLATION TEST (on box): `build_anchor_module` deepcopy DOES copy a projecting
  forward hook; the strip removes it; the clone's forward becomes UNprojected
  (norm 1.95 vs projected 1.64). The strip logic is correct.
- ON THE REAL RUN: the strip assert PASSES (0 residual forward hooks) — the clone
  is genuinely hook-free at backward time — **yet `||G_dense||` is still ~r/H of
  the true norm** (mean_norm 0.0097 vs G_fresh_anchor ~0.03).

## Why the focused fix is insufficient — the residual (STRUCTURAL) mechanism
The clone has NO forward hooks, so a plain forward would be uncompressed. The
projection signature (norm ≈ r/H, partial/heterogeneous across targets) persisting
points at a route the forward-hook strip cannot reach:

1. **Gradient checkpointing (`enable_gradient_checkpointing=True`, confirmed in the
   resolved config).** The boundary block's forward is RECOMPUTED inside the
   backward. The recompute path can re-introduce the projection (the codec is
   applied at the boundary output) in a way that is NOT a persistent
   `_forward_hooks` entry on the clone — e.g. via the checkpoint function closure
   captured when the graph was built, or via the saved-tensor hooks. Stripping
   `_forward_hooks` does not touch the checkpoint recompute closure.
2. **The fast-path `loss_function` partial.** `_maybe_comm_eff_capture_g_dense`
   runs the REAL PPO `loss_function` (bound to the worker / the live actor) on the
   clone. If that partial closes over the live module's compressed activations /
   old_log_probs / the codec state rather than re-deriving everything from the
   clone's forward, the clone's backward is coupled to the compressed graph.
   (The FAITHFUL `G_fresh_anchor` instead uses `anchor_pg_loss` on the anchor's
   own proven clone path — a different, isolated loss.)

Either mechanism is a structural interaction between the parallel backward, FSDP1
+ gradient checkpointing, and the codec graph — not a one-line bug. Resolving it
requires a design decision (see options) rather than more focused patching.

## Options for the operator's design call
- **(A) Drop the dedicated G_dense clone+PPO-loss; make G_dense the anchor's
  proven path at delay_K=0.** `G_fresh_anchor@delay_K=0` is ALREADY a faithful
  full uncompressed dense gradient (cos +0.93..0.99 vs G_anchor on every arm,
  realism-GREEN). The audit could simply USE `G_fresh_anchor` as the dense
  reference for H1/H2 instead of a separate `G_dense` probe — i.e. retire the
  `_maybe_comm_eff_capture_g_dense` clone entirely and point the audit's
  `cos(G_dense, G_comp)` at `cos(G_fresh_anchor, G_comp)`. (Caveat: clean-PG vs
  PPO loss; on the dense arm they agreed at cos 0.99, so likely fine at ratio≈1.)
  This is the LOWEST-RISK path and reuses a validated mechanism — recommended.
- **(B) Disable gradient checkpointing on the G_dense clone only** (the clone is
  off the optimizer; ckpt is a memory optimization, unnecessary for a single
  dump-only backward) and ensure the loss is re-derived purely from the clone's
  forward (use `anchor_pg_loss`, not the live PPO partial). Higher risk; needs to
  confirm the clone forward+backward is fully self-contained.
- **(C) An upstream/FSDP-level change** to make a second uncompressed backward
  compose with FSDP1 + grad-ckpt + the codec — out of scope for this experiment.

## What SURVIVES (do not lose)
- **H3 CONFIRMED, clean, no G_dense dependence:** magnitude-weighted
  sign-agreement(`G_fresh_anchor`, `G_comp`) = 0.490 at delay_K=0 (and 0.490 at
  delay_K=5) ∈ [0.45, 0.55] ⇒ sign-replacement is structurally a coin-flip EVEN
  with a fresh anchor ⇒ corroborates the #25 `retire_sign_replacement` direction.
  The successor `ef_powersgd` (Step B) is direction-preserving (no sign term), so
  H3 does not block it.
- **fp32 dump fidelity PASSES** (recon drift ~1e-4) — the capture WRITE path is
  correct; the defect is confined to what the parallel backward COMPUTES.
- **The substrate is correct** (resolved_params.txt: PowerSGD r77, anchor owns_q,
  delay_K=5, clean_cadence=0, GRPO no-KL/no-entropy).
- **Defects 1–5 are genuinely fixed** (anchor-owns-Q fires without a merger;
  no-merger G_comp captured; capture-tick unified; target-name canonicalised;
  rank0-only + post-warm min_tick disk/quality guards). Only the G_dense backward
  on codec-ON arms remains — and it is the design call above.

## Reproduce
- On box: `python /workspace/runs/EXP-26/gate_check.py /workspace/captures/GATE_probe`
- The gate probe: `runs/EXP-26/probe_gdense_gate.sh` (powersgd r77, cadence=1
  fast-warm, min_tick=2; 3 steps). The AUTHORITATIVE re-run (not run — blocked by
  this gate) would use the locked cadence=5/delay_K=5.
