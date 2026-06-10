# EXP-26 Step C1 — Q-family geometry screen report — 2026-06-10

> **STAGE C1 ranking, NOT the issue verdict.** This resolves the C1 passive
> Q-family screen and recommends the next dispatch per `STEP_C_SPEC.md` §Decision
> tree. The issue label stays `status:running`; no `verdict.md` is written (that path
> is reserved for the terminal whole-issue verdict + arms the teardown hook).
>
> Binding spec: `runs/EXP-26/STEP_C_SPEC.md` (§Judge metrics, §Decision tree).
> Step-A context: `stepA_decision.md` (UC_act=0.318, off-principal 0.682, act capture 0.9985).
> Audit invocation (recorded in `analysis.log`): `python3 research/scripts/stepC_screen_audit.py`
> — reuses the Step-A projection convention from `research/scripts/geometry_audit.py`
> (q_by_H setdefault by H, row/col projection, break-on-first-axis).

## Completion

`runs/EXP-26/captures/C1_screen/stepC1.done.flag` present (`2026-06-10T02:35:49Z done`);
`train_C1_screen.log` shows `Training Progress: 100%|██████████| 9/9` then a clean
final validation. The 3 Tracebacks at log lines 1228–1292 are **shutdown-time
artifacts** (a DataLoader worker `Killed` + a WandB `ConnectionResetError` in
`Tracking.__del__`) AFTER step-9 checkpoints saved and final val ran — NOT a
training divergence. Final-step counters all healthy: `anchor_backwards=3`,
`anchor_q_updates=3`, `anchor_q_broadcasts=3`, `family_screen_builds=3`,
`powersgd_basis_updates=0` (fast net never wrote Q — realism GREEN),
`spectral_corrections=0`, `clean_steps=0`, `residual_reset_on_shape_mismatch=0`,
`grad_norm=1.147` (finite), `recon_rel_error=0.0276` (healthy), `bytes_ratio=0.0504`.

## Method (matches Step A exactly)

- **Reference grad** `G_fresh_anchor@delay_K=0` (per weight matrix, 28 targets/tick),
  validated faithful in Step A. Here `cos(G_fresh_anchor, G_anchor)` median **1.0001**
  (range 0.983–1.008, n=28) — confirms the captured fresh grad IS the genuine full
  uncompressed gradient.
- **Post-warm ticks** = the latest 2 capture ticks that carry family bases: **(gs5,
  tick10)** and **(gs8, tick15)** — the two anchor-fire ticks (cadence=5).
- **Projection convention** identical to `geometry_audit.audit`: `q_by_H` collapses
  boundaries by `H=q.shape[0]=1536` (setdefault), each weight grad projected on its
  matching axis (`(g@q)q.T` for the 1536-column axis, `q(q.T g)` for the 1536-row
  axis), break on first matched axis.
- **Comparability anchor.** UC computed via the **live `Q` role** lands at **0.3042**
  (n=28) — close to Step-A's **0.318** (the residual gap is the different short run /
  ticks 5_10·8_15 vs Step-A's 10/15). Convention is therefore comparable. NOTE: the
  family **`Q_act`** is the *passively re-sketched* act basis built inside the anchor
  pass (stale weights, single power-iteration, no warm-start), so its UC reads
  **0.2010** — lower than the warm-started live `Q`. The family `Q_act` is the correct
  **apples-to-apples control** for ranking, because all 6 families were built by the
  identical passive procedure and differ ONLY in the statistic fed to power iteration.

## Per-family judge table (post-warm ticks 10/15, median over 28 targets)

| family | UC_f | OPP_f | AC_f | beats control? | UC per-tick (t10 / t15) | OPP per-tick (t10 / t15) |
|---|---|---|---|---|---|---|
| **act** (control) | **0.2010** | **~0.0000**¹ | 0.9993 | — (control) | 0.195 / 0.205 | ~0 / ~0 |
| grad | 0.0498 | 0.0518 | 0.0439 | no (UC≪) | 0.050 / 0.050 | 0.052 / 0.052 |
| adv | 0.2010 | ~0.0000 | 0.9993 | no (= act exactly) | 0.195 / 0.205 | ~0 / ~0 |
| tail | 0.0498 | 0.0518 | 0.0439 | no (UC≪; = grad exactly) | 0.050 / 0.050 | 0.052 / 0.052 |
| **hybrid** | **0.2496** | **0.0685** | 0.9987 | **YES** | 0.251 / 0.242 | 0.069 / 0.068 |
| **ticket** | **0.2555** | **0.1391** | 0.2933 ⚠ | **YES** | 0.259 / 0.237 | 0.143 / 0.138 |

¹ `OPP_act ≈ 1.4e-13` **by construction**: `G_off = G_fresh − proj_{Q_act}(G_fresh)`
is orthogonal to `Q_act`, so re-projecting it onto `Q_act` is ~0. So the spec's
`OPP_f > OPP_act` gate is trivially cleared by *any* family with nonzero off-principal
overlap — the meaningful comparison is the **absolute** OPP_f (reported above).

- Sanity (4): every family orthonormal — `||Q_f^T Q_f − I||_F ≤ 5.2e-6` (ticket exactly
  0). UC via live Q = 0.3042 ≈ Step-A 0.318 — convention validated.
- `AC_f < 0.9` flagged (guardrail, NOT a gate): **grad 0.044, tail 0.044, ticket 0.293**
  are forward-fidelity risks for a training arm (the spec anticipates this — grad/tail
  legitimately abandon activation reconstruction). hybrid 0.9987 and act/adv 0.999 are
  forward-safe.

## Winner decision (per §Decision tree)

**Winner rule** (spec): `f` beats control iff `UC_f > UC_act` AND `OPP_f > OPP_act`.
Against the family-internal control (UC_act=0.2010, OPP_act≈0):

```yaml
winners: [hybrid, ticket]      # both clear UC_f > 0.2010 AND OPP_f > 0 (absolute OPP: 0.069 / 0.139)
ranked_all (by UC_f):
  - ticket : UC 0.2555  OPP 0.1391  AC 0.293  -> BEATS (best off-principal pickup; AC-flagged)
  - hybrid : UC 0.2496  OPP 0.0685  AC 0.999  -> BEATS (forward-safe; modest off-principal pickup)
  - act    : UC 0.2010  OPP ~0      AC 0.999  -> control
  - adv    : UC 0.2010  OPP ~0      AC 0.999  -> act-DUPLICATE (not a real arm; see flag a)
  - grad   : UC 0.0498  OPP 0.0518  AC 0.044  -> FAILS (UC collapses far below act)
  - tail   : UC 0.0498  OPP 0.0518  AC 0.044  -> FAILS (= grad exactly; deflation is a no-op)
verdict: two_families_beat_act (hybrid, ticket) — proceed to C2 with the winner LIVE.
```

**Honest caveat on the win.** The "wins" are modest and the absolute capture stays
low. Hybrid/ticket beat act mainly because they *retain* act-subspace content (hybrid
keeps 39 act columns; ticket overlaps act 0.09–0.30) AND add a little off-principal
energy — they do not transform the picture. Critically, the grad-energy families
**fail their own thesis**: `grad`/`tail` capture only ~5% of `G_fresh_anchor`, and
even on their NATIVE operand `G_b` (the boundary activation gradient they were built
from) they capture only **0.0515** vs act's 0.0643 — the boundary activation gradient
is essentially **rank-diffuse**, so NO 77-dim subspace captures it. This means H2's
"a grad-content basis recovers the missing update energy" is **not supported by the
pure grad routes** at r=77; the recoverable gain is small and comes from the
axis-aligned (ticket) / blended (hybrid) constructions.

## Flag (a) — `adv` is an act DUPLICATE (RESOLVED)

`adv_weight=uniform` logged on every screen tick (`train_C1_screen.log` lines
1148/1174/1201). With uniform weights `w = diag(|a|/mean|a|) = I`, so the adv sketch
`(wM)^T(wM Q) = M^T(M Q)` **= the act sketch exactly**. Confirmed in the dumps:
`Q_adv` is **byte-identical** to `Q_act` at every tick/boundary (max-abs-diff = 0.0;
subspace overlap `||Q_adv^T Q_act||_F^2 / r = 1.00000`). **`adv` is NOT a real arm in
this screen** — it is act under a different label. To exercise the adv arm for real,
the advantage tensor must be plumbed into the compressor context with non-uniform
per-token weights (the spec's §Family-sketch note: "plumb the mini-batch `advantages`
into the compressor context, aligned to the rmpad row layout"); the screen logged
`adv_weight=uniform`, so that plumbing did not deliver real advantages. Recorded as a
finding; `adv` should be re-screened with real weights only if the activation-weighted
route is independently motivated (it is NOT a C2 candidate now).

## Flag (b) — `[EXP-12] anchor refresh produced NO target grads (targets matched=0)` (RESOLVED: stale-tag artifact, NOT a coverage gap)

The warning fires at steps 5/10/15 (`train_C1_screen.log` 1149/1175/1202). It is a
**stale-tag / sub-check artifact of the spectral-OFF C1 config, NOT a real anchor
coverage gap.** Evidence:

- The warning lives in `transformer_impl.py:1641-1646`, the **else** branch of the
  MERGER's EMA-delta log: it fires when `deltas` (the per-target `||dM_anchor||`
  *merger* EMA deltas) is empty. C1 runs spectral OFF (`spectral.enabled=false`,
  `spectral_corrections=0`, no `M` role dumped) ⇒ there is no merger maintaining M ⇒
  `deltas` is empty ⇒ the warning prints. It reports the **merger** matched no
  targets, which is *correct and expected* with no merger present.
- The **anchor's gradient capture is a separate code path and SUCCEEDED**:
  `anchor_backwards=3`, `anchor_q_updates=3`, `anchor_q_broadcasts=3`,
  `family_screen_builds=3`. The dumps confirm real, nonzero tensors:
  `G_anchor` 28/28 targets nonzero (norms 0.0035–0.101); `G_fresh_anchor` 28/28
  nonzero with `cos(G_fresh_anchor, G_anchor)=1.0001`; `G_b` nonzero at tick (5,10).
- `recon_rel_error` is stable ~0.030 across all 8 ticks (NOT the 0.976→0.026 cold
  trajectory — the screen started with an already-warm Q, so recon never dipped to the
  cold value). Q is warm and updating (live `Q` evolves across ticks; family `Q_act`
  vs live `Q` subspace overlap 0.52→0.76 confirms an independent warm sketch).

**The anchor's MERGER target set is NOT genuinely empty in the real (spectral-ON)
path** — it is empty here only because C1 deliberately runs no merger. Step B's
premise (the anchor feeds a populated merger target set under `ef_powersgd`) is
**NOT invalidated** by this warning. Recommend the runner gate the warning on
`spectral.enabled` so it stops crying wolf on merger-OFF cells.

> ⚠ One real data caveat (separate from flag b): **`G_b = 0` at tick (8,15)**, so the
> grad-derived family sketches (grad/tail/ticket) could not re-build at the latest tick
> and carry forward their tick-(5,10) values BYTE-IDENTICALLY (grad/tail max-abs-diff
> 0.0 across ticks). The grad-family ranking therefore rests effectively on the single
> tick (5,10). The `act`/`adv`/`hybrid` bases DID re-sketch at (8,15) (they draw on the
> activation M, not G_b). UC/OPP per-tick values are stable across both ticks regardless,
> so the ranking is robust — but the runner should fix the (8,15) `G_b` capture (likely
> the activation-grad hook missed the last anchor backward) before a C2 arm relies on
> live grad-family sketches.

## Recommendation for the next dispatch

**Route: C2 with `hybrid` LIVE (NOT ticket, NOT a straight-to-B-with-act).**

Per §Decision tree, "some family beats act → C2: ONE 50-step training arm, plain
PowerSGD r77 + winner `q_basis` LIVE." Two families beat the control; choose **hybrid**
as the C2 arm:

- **hybrid wins on the criterion that matters for a Q that does double duty.** It clears
  the gate (UC 0.250 > 0.201, OPP 0.069 > 0) **AND** keeps activation capture safe
  (**AC 0.9987**), so it will not wreck forward reconstruction in a live training arm.
- **ticket has the higher OPP (0.139) and UC (0.256) but is the riskier live arm:**
  AC 0.293 (≪ 0.9 guardrail) means the axis-aligned ticket basis would drop ~70% of the
  activation reconstruction energy — a real forward-fidelity hazard the screen flags. It
  is the comm-cheapest construction (indices only) and the best off-principal capturer,
  so keep it as the **fallback C2 arm** if hybrid fails its training gate.
- **Do not promote grad/tail/adv:** grad/tail collapse on UC (rank-diffuse boundary
  grad, the deflation is a no-op); adv is an act duplicate under uniform weights.

C2 gate (from spec): `val@50 >= 0.7414`, no length/clip collapse. If C2 passes → Step B
arms `{ef_powersgd + hybrid Q, plain PowerSGD r77 + act, dense}`. If C2 fails → Step B
with act (C recorded as a REVISE-grade finding, not a STOP). **Carry the honest caveat
forward:** the geometric edge is small and the pure grad-content thesis (H2's strongest
form) did not pan out at r=77 — C2 is testing whether a *modest* off-principal pickup
(hybrid) converts to a training-curve gain, not a dramatic one. Also fix the (8,15)
`G_b=0` capture and gate the EXP-12 merger warning on `spectral.enabled` before C2.

## Provenance

- Resolved substrate (`resolved_params.txt`, the Step-A capture; the C1 launch command
  in `train_C1_screen.log:35` carries the same substrate + family-screen flags):
  `compression_type=powersgd`, `powersgd.rank=77`, `powersgd.q_basis=act`,
  `sync_basis=true`, `anchor.enabled=true`, `anchor.owns_q=true`, `anchor.cadence=5`,
  `anchor.delay_K=5`, `clean_cadence=0`, `spectral.enabled=false`,
  `use_kl_loss=False` (the launcher's `use_kl_loss=True kl_loss_coef=0.001` is the dead
  default, overridden last-wins — vanilla GRPO no-KL confirmed), `train_batch_size=128`,
  `ppo_mini_batch_size=64`, `rollout.n=8`, `max_response_length=16384`. LOCKED control
  surface intact; the C1 variable is the passively-built family `q_basis` content.
- `check_budget.py`: running_count=1, running_dph=12.88 (≤ max_dph 24); lifetime/month
  $208.21 (cap $1500) — well under `max_gpu_hr=60`.
- All audit numbers reproducible: `python3 research/scripts/stepC_screen_audit.py`
  (output appended to `runs/EXP-26/analysis.log`).
