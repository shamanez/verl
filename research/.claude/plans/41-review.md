# Adversarial review — Plan EXP-41 (M4 linear weight extrapolation, look-ahead anchor)

Reviewer: independent, code-grounded. Read the plan in full + every `target_module` + the
hook ordering + the verification scripts. All `file:line` below are real and were opened.

## 1. Verdict

**SOLID-WITH-FIXES** (leaning NEEDS-REWORK on the test gate). The implementation is feasible and
the insertion points the plan names are REAL and well-chosen — but the load-bearing PASS criterion
(the "anti-damping alignment lift" `cos(g(theta_hat), g_live)`) depends on telemetry that **does not
exist today, cannot be emitted at the point the plan implies, and structurally conflicts with the
plan's own merger config**. As written, a dead/damped run can pass the gate and a clean run can be
scored against a signal the runner never actually produced. The compute is approvable only after the
must-fixes below close the alignment-lift definition and the snapshot-retention gap.

## 2. Implementation soundness (REAL ⇄ PLAN, per target_module)

### `verl/workers/comm_eff/anchor.py` — MATCH, clean insertion point
- The plan's "feed `g(theta_hat[t])` into the existing EMA/merger path" is accurate. The raw anchor
  grad is read by `extract_target_grads` (anchor.py:601) and fed to the EMA by
  `feed_anchor_grads_into_ema` → `spectral.update_anchor` (anchor.py:1095-1122), which is exactly
  `M_anchor <- beta*M + (1-beta)*G_anchor`. Loading extrapolated weights does NOT touch this stage —
  it only changes WHICH weights the clone forwards from. So "feed into the existing path" is correct
  and requires NO surgery here. Good.
- The reusable helpers the new code needs all exist: `_build_anchor_pg_loss`
  (transformer_impl.py:1205), `_dp_all_reduce_anchor_grads` (:1004), `_comm_eff_target_names` (:997).

### `verl/workers/comm_eff/state.py` — PARTIAL MISMATCH (the snapshot the plan wants to REUSE cannot hold theta[t-40])
- This is the single biggest implementation gap and the plan glosses it.
- The launcher runs `replay_paired_batch=true` (launcher line 33), so the live path is the
  **`AnchorReplayRing`** (anchor.py:391), NOT the `AnchorStalenessQueue`. The ring stores
  `gs -> (snapshot, canary, tick)` and is FIRE-AWARE: it retains only
  `self._maxlen = delay_K // cadence + 1` snapshots (anchor.py:427) and aggressively evicts any gs
  "no retained batch references" on every push (anchor.py:453-455, 483-486). At cadence=delay_K=20
  that is `20//20 + 1 = 2` snapshot slots, and they are keyed to the single fire-aligned tick a fire
  consumes — i.e. `theta[t-20]`. **`theta[t-40]` is evicted long before the next fire.** Fixed-linear
  `2*theta[t-20] - theta[t-40]` therefore has NO second anchor point in the live ring today.
- The legacy `AnchorStalenessQueue` is no better: `_maxlen = delay_K + 1` (anchor.py:248), so it holds
  `theta[t-20]..theta[t]` — again NOT `theta[t-40]`. And it is not even built in replay mode
  (transformer_impl.py:1380 only builds it `if ... not replay_mode`).
- **Conclusion:** the plan's "reuse paired-replay snapshots where possible; build a compact
  fire-aligned look-ahead history" is NOT a reuse — it requires a NEW retention structure (a small
  fire-aligned ring holding the last 2–3 fire-tick snapshots: `theta[t-20], theta[t-40]`, and for the
  learned variant `theta[t-60]`). This is buildable (the ring scaffolding is good prior art) but it is
  net-new code with its own eviction-bound assert, not a config tweak. The memory-budget invariant's
  "reuse where possible" wording will mislead the runner into thinking the points are already on hand.
- Memory budget sanity: snapshots are CPU-resident bf16 full-param clones (`snapshot_device=cpu` →
  `_snap_dev=cpu` at transformer_impl.py:1361; `snapshot_named_params(..., device=cpu)`). 1.5B bf16 ≈
  3 GB/snapshot is right. 2 snapshots (fixed-linear) ≈ 6 GB CPU, 3 (learned) ≈ 9–12 GB CPU. Realistic
  on the box. The extrapolated `theta_hat` is materialized transiently into the EXISTING cached clone
  (it is loaded via `p.copy_` at transformer_impl.py:1646), so it adds ~0 steady HBM. Fine.

### `verl/workers/config/comm_eff.py` — MATCH; the yaml-mirror gotcha is REAL and named correctly
- The dataclass→yaml→CLI flow is exactly as the plan warns. A new knob (e.g.
  `CommEffAnchorConfig.lookahead_anchor` / `lookahead_mode`) must be added in THREE places or it
  silently no-ops:
  1. the dataclass field in `comm_eff.py` (CommEffAnchorConfig, comm_eff.py:97-139), with validation
     in `__post_init__`;
  2. the structured yaml block `verl/trainer/config/actor/actor.yaml` under `comm_eff.anchor:`
     (actor.yaml:355-380 — it explicitly mirrors every anchor field; `_target_: CommEffAnchorConfig`
     at :358 means OmegaConf rejects unknown keys, so a missing yaml field is a HARD parse error, but
     a field present in yaml + dataclass but NOT wired to a launcher override silently keeps its
     default);
  3. the launcher Hydra override line + the `${VAR:-default}` export in
     `vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh` (the `actor_rollout_ref.actor.comm_eff.anchor.*`
     block at launcher lines 448-453) AND the accel-base wrapper that sets it.
- The plan names actor.yaml correctly. Good. The disabled-path-parity invariant is enforceable here
  (a `lookahead_anchor=false` default makes the new branch inert) — consistent with the codebase's
  strict-no-op discipline (state.py:995, the `maybe_build` gate).

### `verl/workers/engine/fsdp/transformer_impl.py` — MATCH on the load point; extrapolation IS mechanically possible
- `_maybe_comm_eff_anchor_refresh` is at **transformer_impl.py:1269** exactly as claimed.
- The raw `theta[t-20]` load into the isolated clone is at **transformer_impl.py:1640-1663**: the
  stale snapshot is canon-keyed (`stale_canon`) and `p.copy_(s.to(p.device, p.dtype))` per param, with
  a fail-closed `assert loaded == total` (:1657). This is precisely where look-ahead must instead load
  `theta_hat[t]`.
- **Loading an extrapolated weight point that was never a real training state is mechanically fine.**
  The load is a plain per-element `copy_` into a deep-copied non-FSDP clone
  (`build_anchor_module`, anchor.py:649 — the clone is summoned-full / plain `nn.Module`, no
  FlatParameter). The extrapolation `2*A - B` is per-element linear, so it composes with sharding
  trivially **provided you compute it in the SAME space the snapshots are stored in**. Here snapshots
  are FULL logical tensors (taken inside `summon_full_params`, transformer_impl.py:1433-1436 /
  1454-1457), so the projector operates in full-param space on CPU — no cross-shard op, no norm, no
  identity ambiguity. The LayerNorm/embedding exclusion is doable per-name on these full snapshots
  (the keys are canonical param names; the existing `target_substr` selector already distinguishes
  decoder matrices from norms/embeds, comm_eff.py:228-238). The known deepcopy-fallback infix bug is
  already handled by `_canon` on both sides (anchor.py:106, spectral_filter.py:56,
  transformer_impl.py:1640) — the look-ahead code MUST route every new key through `_canon` too, but
  the machinery is there.
- One caveat the plan should state: the canary assert (transformer_impl.py:1672-1693) BITWISE-verifies
  the clone holds the *recorded snapshot*. An extrapolated `theta_hat` is NOT any recorded snapshot, so
  the look-ahead path must compute its OWN canary over `theta_hat` (or skip the snapshot-canary for the
  extrapolated load and instead canary the two SOURCE snapshots before combining). If the runner naively
  loads `theta_hat` and leaves the existing `_fire_canary` check live, it will hard-fail at :1687. This
  is a real, easy-to-hit trap; flag it.

### `verl/workers/comm_eff/lookahead.py` (NEW) — feasible; underspecified in two spots
- The per-block linear projector (fixed-linear + learned per-block residual coeffs, CPU-resident,
  cross-rank-deterministic, LayerNorm/embedding excluded) is cleanly implementable as a pure-CPU module
  mirroring the sibling pure-logic style (`anchor.py` helpers are explicitly "FSDP-agnostic,
  CPU-testable" — anchor.py:46-66). Good fit.
- Underspecified #1 (cross-rank determinism of the LEARNED coeffs): the plan asserts coefficients are
  "cross-rank-identical" but never says HOW. Fixed-linear is trivially identical (pure function of two
  DP-identical snapshots — the snapshots are DP-mean-reduced? NO: snapshots are raw per-rank weight
  clones, but FSDP weights ARE identical across DP ranks, so this is fine). The LEARNED residual is
  the risk: it updates from `theta_true[t_prev] - theta_hat[t_prev]`. `theta_true` (FSDP weights) is
  DP-identical, so the residual is DP-identical IF every rank computes it the same way — but the plan
  must assert this and the runner must add a cross-rank max-rel-dev check (there is prior art:
  `_powersgd_q_agreement_dev`, state.py:931). Without it, "multi-rank agreement" is aspirational prose.
- Underspecified #2 (where `theta_true[t_prev]` lives at the NEXT fire — the no-leakage retrospective
  target): see Testing §no-leakage. The fire lifecycle CAN supply it, but it requires retaining one
  extra past snapshot specifically for the residual target, which compounds the state.py retention gap.

## 3. Testing soundness — the gate has a hole

### 3a. The alignment-lift PASS bar (`cos(g(theta_hat), g_live)`) — NOT computable as written. THIS IS THE CRUX.
- `g_live` is defined by the plan as "a fresh fast-circuit gradient at sync." **No such tensor is
  captured anywhere today** (grep for g_live/fresh-fast/sync-grad: nothing except the unrelated
  `G_fresh_anchor` *probe*). It must be added as new telemetry.
- Worse, the timing is wrong by construction. Hook order in `BaseEngine.train_batch`
  (base.py:211-226) is: **(1)** `_maybe_comm_eff_anchor_refresh` (computes the anchor grad at the very
  TOP) → **(2)** `forward_backward_batch` (the live fast grad first lands on `p.grad`) → **(3)**
  `_maybe_comm_eff_grad_correction` (signed_ema **overwrites** `p.grad`) → **(4)**
  `_maybe_comm_eff_geometry_probe` → **(5)** `optimizer_step`. So at the moment the anchor grad exists
  (step 1) the live fast grad does not exist yet. Any `cos(g(theta_hat), g_live)` must be deferred to
  end-of-batch and must cache the anchor grad — extra code, not a free scalar.
- And under the plan's own merger (`signed_ema`), by the time you could read the live grad (step 4),
  the correction hook (step 3) has already REWRITTEN `p.grad` to the merged value. So "g_live" read at
  end-of-batch is the MERGED gradient, not the raw fast-circuit gradient. To get raw `g_live` you must
  stage it BEFORE step 3 (a new copy in `forward_backward_batch`).
- The one existing analog — the geometry probe's `m1 = cos(g_comp, rep)` (anchor.py:1012, computed in
  `_maybe_comm_eff_geometry_probe`, transformer_impl.py:2503) — is exactly "cos(live compressed grad,
  anchor replay grad)", i.e. the lift signal in spirit. BUT it is hard config-gated to be UNUSABLE
  here: `probe.geometry_enabled=true` REQUIRES `spectral.correction_mode=none`
  (comm_eff.py:864-870 and the runtime guard at transformer_impl.py:2408-2409). The plan runs
  `signed_ema`. **You cannot turn on the existing probe and the plan's merger at the same time.** So
  the plan cannot lean on m1 as-is; it needs a NEW, merger-compatible scalar.
- Net: the load-bearing PASS criterion depends on telemetry that (a) doesn't exist, (b) needs a raw
  `g_live` staged before the merger overwrites it, (c) needs the cached anchor grad carried to
  end-of-batch, and (d) cannot reuse the existing probe because of the merger conflict. The plan's
  "Notes for runner → Anti-damping telemetry is mandatory" acknowledges the scalars must be logged,
  but it does NOT acknowledge the merger/probe conflict, the hook-order timing, or that `g_live` is
  net-new capture. **As written the PASS bar is undefined and a `STOP`-vs-`PASS` call cannot be made
  from artifacts the run will produce.**

### 3b. "Off-diagonal" (genuine rotation, not the diagonal trap) — aspirational, no concrete check
- The criterion (success-criteria box + analyst notes) demands the lift be "off-diagonal: a genuine
  rotation, not a per-coordinate rescale." There is NO defined statistic for this anywhere in the plan
  or code. A bare `cos` rising does not distinguish rotation from rescale. The codebase has the raw
  material (per-target full 2D grads are staged CPU/fp32 in the probe stash, transformer_impl.py:2356-
  2370) but no off-diagonal metric is computed. This box is currently unfalsifiable prose. Either
  define it concretely (e.g. compare `cos(theta_hat-grad, live)` against the best achievable by a
  per-coordinate diagonal rescale of the stale grad — if the lift exceeds the diagonal bound it is
  off-diagonal) or demote it from a hard PASS gate to a reported diagnostic.

### 3c. The verification scripts EXIST — but don't do what the gate needs
- Contrary to the plan's hedge ("if a referenced helper is absent in the de-bloated tree"), all three
  scripts are present: `research/scripts/analyze.py`, `check_budget.py`, `diff_against_baseline.py`
  (verified on disk). BUT:
  - `analyze.py` emits `VERDICT: PENDING` for any real (non-smoke) experiment and only dumps a
    min/mean/max summary of `metrics/*.jsonl` (analyze.py:108-114, 42-55). It does NOT compute the
    alignment lift, the off-diagonal test, the collapse check, or the band comparison. The analyst
    hand-fills everything.
  - `diff_against_baseline.py` only diffs the FINAL-row numeric keys of `train.jsonl` (diff:78-93). It
    cannot express "lift vs cell A's raw stale-anchor baseline" (that is a different cell + a custom key).
  - So the gate reduces to the plan's `grep` fallback — which is fine IN PRINCIPLE, except the keys it
    greps do not exist (next point).

### 3d. The grepped metric keys DO NOT EXIST and must be created by the same code change
- The verification `grep -E "anchor_align_cos|lookahead_source_ticks|anchor_opt_steps|anchor_rollouts|
  anchor_reward_recompute|anchor_mask_apply|..."` references keys that are ALL absent from the source:
  - `anchor_align_cos` — 0 files. (the lift signal; net-new)
  - `lookahead_source_ticks` — 0 files. (the per-fire source-tick log; net-new)
  - `anchor_opt_steps` — 0 files. The REAL key is `anchor_optimizer_steps` (state.py:967).
  - `anchor_reward_recompute` — 0 files. REAL: `anchor_rewards_recomputed` (state.py:966).
  - `anchor_mask_apply` — 0 files. REAL: `anchor_mask_applications` (state.py:963).
  - `lookahead_mode` — 0 files. (net-new)
  So the success criteria depend on telemetry that the code change must ALSO add, and several greps as
  written would silently match NOTHING (returning "clean" by absence) — e.g. a grep for
  `anchor_opt_steps` finds zero lines and an unwary analyst reads "0 anchor optimizer steps = good"
  when in fact the real counter `anchor_optimizer_steps` was never inspected. **This is a false-pass
  hazard baked into the verification commands.** The plan does NOT acknowledge that its success
  criteria require new telemetry; it implies the keys already flow.

### 3e. What the gate DOES catch reliably (the good news)
- The anchor-isolation invariant is genuinely machine-checkable and already enforced HARD in code,
  independent of the plan: `anchor_mask_applications` delta asserted 0 (transformer_impl.py:1816),
  `anchor_optimizer_steps` unchanged asserted (:1822), and `anchor_rollouts_generated` /
  `anchor_rewards_recomputed` are structurally 0 (the anchor never generates/scores — state.py:331-332,
  metrics at state.py:965-967). These 4 counters DO get emitted. So "anchor invariant counters stay 0"
  is real — once the grep uses the correct key names.
- Disabled-path parity is enforceable (the strict-no-op discipline is real: state.py:995,
  the `lookahead_anchor=false` default branch).
- Collapse detection (NaN, response-length runaway, entropy) is checkable from standard metrics
  (`response_length/mean`, `entropy`, train-loss NaN via analyze.py:90-99). Fine.
- The no-leakage source-tick assert IS implementable (log the source ticks per fire; assert none `>= t`)
  — but the KEY must be created and the criterion currently grep-references a nonexistent
  `lookahead_source_ticks`.

### 3f. The 2-step probe vs the 8 hard invariants
- Most invariants are checkable in a 2-step probe AS STATED: disabled-path parity, fixed-linear identity
  (`theta_hat == 2A-B` within fp tol on the retained snapshots — but see §2 state.py, the second
  snapshot must first be retained), anchor isolation (the asserts already fire), backend integration
  (NaN/OOM in the probe), LayerNorm/embedding exclusion (log the excluded set).
- TWO are NOT meaningfully checkable in a 2-step probe:
  - "multi-rank determinism" — needs a concrete cross-rank max-rel-dev emission (prior art exists,
    state.py:931) and at delay_K=20 the FIRST fire is at tick 20 (≈step 10), so a 2-step probe never
    fires the anchor at all under cadence=20. The probe must FORCE a small cadence (e.g. cadence=delay_K=1)
    to exercise the look-ahead fire path, and the plan's "2-step probe with the cell's config" (cadence=20)
    would EXERCISE NOTHING — the look-ahead branch never runs in 2 steps at cadence 20. **This is a real
    hole: the cheap pre-run gate as specified does not actually fire the new code.** The probe must
    override cadence/delay_K down to fire within 2 steps, AND must additionally hold ≥`2*delay_K`-tick
    history to test fixed-linear identity, which at small delay_K is cheap.
  - "fixed-linear limiting-case identity" at the cell's delay_K=20 needs theta[t-40] retained (§2 gap).

### 3g. Cell A → Cell B snapshot chaining (the plan's "share one box, chain snapshots")
- The plan says cells chain snapshots on one box. This is a non-issue for CORRECTNESS but the plan
  slightly over-claims: each cell is a FRESH `python` training process (the launcher re-execs), so
  in-memory snapshot rings do NOT survive across cells — every cell rebuilds its own ring from its own
  run. Cell A (5/5) and cell B (20/20) have DIFFERENT fire-tick spacing and DIFFERENT retention bounds;
  A's ring is irrelevant to B regardless. What actually transfers between cells is only the laptop-side
  artifacts (val numbers, the raw-stale `cos` baseline cell A logs). So "chain snapshots" is harmless
  shorthand, NOT a dependency — the plan does not MISS anything here, but the wording could mislead a
  runner into thinking on-box state persists. Low severity.

## 4. Must-fix before approval (numbered, actionable)

1. **Define `g_live` and the alignment-lift computation concretely, accounting for hook order.** Specify:
   (a) stage a raw copy of the live fast-circuit per-target grad in `forward_backward_batch` BEFORE
   `_maybe_comm_eff_grad_correction` overwrites `p.grad` (base.py:212 vs :221); (b) cache the anchor
   grad from the top-of-batch refresh; (c) compute `cos(cached anchor grad, staged raw g_live)` at
   end-of-batch and emit it as a scalar `anchor_align_cos` (always-on safe). Without this the
   load-bearing PASS bar is uncomputable and the run is uninterpretable. (This also means cell A must
   emit the SAME scalar for the raw stale anchor as the lift baseline — confirm cell A's config path
   actually produces it; cell A is `lookahead_anchor=disabled`, so the staging+caching must NOT be gated
   behind the look-ahead flag.)

2. **Fix the snapshot retention gap (theta[t-40] / theta[t-60]).** State explicitly that a NEW
   fire-aligned look-ahead snapshot ring is required (the live `AnchorReplayRing` retains only
   `delay_K//cadence+1 = 2` slots at 20/20 and evicts everything but `theta[t-20]`; the legacy queue
   isn't built in replay mode). Specify its bound (2 for fixed-linear, 3 for learned, +1 if a separate
   retrospective-residual target is needed) and add the eviction-bound assert (mirror
   `FastGradRing`/`AnchorReplayRing` asserts). Remove the misleading "reuse paired-replay snapshots"
   framing from the memory invariant — they do not contain the second point.

3. **Correct the verification grep keys (false-pass hazard).** Replace `anchor_opt_steps →
   anchor_optimizer_steps`, `anchor_reward_recompute → anchor_rewards_recomputed`, `anchor_mask_apply →
   anchor_mask_applications`. State that `anchor_align_cos`, `lookahead_source_ticks`, `lookahead_mode`
   are NEW keys this code change must emit (they are absent today) — the success criteria depend on
   telemetry the patch must add, and grepping a misspelled/absent key returns "clean" by absence.

4. **Resolve the merger/probe conflict for the lift measurement.** The existing geometry probe (the only
   `cos(g_comp, anchor)` machinery) is hard-gated to `correction_mode=none` (comm_eff.py:864-870,
   transformer_impl.py:2408-2409) and cannot run with the plan's `signed_ema`. Either (a) implement the
   lift scalar as standalone always-on telemetry independent of the probe (preferred — see must-fix #1),
   or (b) add an explicitly-justified relaxation. Do NOT assume `probe.geometry_enabled=true` is
   available under signed_ema.

5. **Make the 2-step pre-run probe actually fire the look-ahead path.** At cadence=delay_K=20 the anchor
   never fires in 2 steps, so the probe exercises none of the new code. Specify a probe override
   (small cadence/delay_K, e.g. 1/1 or 2/2) that (i) triggers a look-ahead fire, (ii) retains enough
   fire-tick history to check fixed-linear identity `theta_hat == 2A-B`, and (iii) emits the cross-rank
   max-rel-dev for determinism. Keep the FULL cell at 20/20.

6. **Handle the canary against an extrapolated weight point.** `theta_hat` is not a recorded snapshot, so
   the existing bitwise `_fire_canary` assert (transformer_impl.py:1672-1693) will hard-fail if left live
   on the look-ahead load. Specify: canary the two/three SOURCE snapshots before combining, and either
   skip or replace the post-load canary for the extrapolated clone. Name this in "Notes for runner →
   Anchor clone load point."

7. **Give the "off-diagonal / not the diagonal trap" gate a concrete statistic or demote it.** As written
   it is unfalsifiable. Either define a computable test (e.g. lift must exceed the best per-coordinate
   diagonal-rescale cosine of the stale grad) or move it from a hard PASS gate to a reported diagnostic.

## 5. Nice-to-have

1. Add a cross-rank determinism emission for the learned coefficients (reuse the
   `_powersgd_q_agreement_dev` pattern, state.py:931) rather than asserting determinism in prose.
2. Spell out where the retrospective residual target `theta_true[t_prev]` is read in the fire lifecycle
   (it is the FSDP live weights at the next fire — DP-identical — but the plan should name the exact site
   so the runner doesn't re-snapshot redundantly).
3. The plan's "chain snapshots across cells" wording (Experiment sequence) should be softened — on-box
   in-memory rings do not survive the per-cell re-exec; only laptop artifacts transfer. No correctness
   impact, but it invites a wrong mental model.
4. Consider logging `cos(g(theta_hat[t]), g(theta[t-K]))` (look-ahead vs raw stale) as the plan's Notes
   already suggest — it is the cheapest "is the projector changing the direction at all" sanity scalar and
   needs no `g_live`.

## 6. Bottom line

**No — not as written; yes after must-fixes 1–6.** The integration points are real and the
extrapolation is mechanically sound, so this is NOT a dead route — but the PASS/STOP decision currently
rests on a signal (`cos(g(theta_hat), g_live)`, off-diagonal) that the run will not produce: the
telemetry doesn't exist, the hook order means `g_live` must be staged before the merger overwrites it,
the only existing analog conflicts with the plan's merger, the second extrapolation snapshot is evicted
before it's needed, and the verification greps reference nonexistent/misspelled keys (a silent false-pass
risk). All are fixable with bounded edits on the `exp/41` branch and none require a design pivot. Close
must-fixes 1–6 (7 can be a reported diagnostic) and the 96-GPU-hr spend is justified; approve then.
