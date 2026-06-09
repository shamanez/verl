# Step A decision — EXP-26 — 2026-06-10T01:40:00+10:00

> This is the Step-A **stage gate**, NOT the terminal verdict. No `verdict.md` is
> written (it would trip the teardown-finished-runs Stop hook and destroy the warm
> box Step B must reuse). Issue label stays `status:running`.

## DECISION

```yaml
DECISION: STUCK
stuck_marker: "STUCK: EXP-26"
reason: "The parallel uncompressed G_dense probe (the H1/H2 reference) is broken on the codec-ON path. H1 and H2 are NOT MEASURABLE; only H3 is cleanly measured. Per the plan's Step-A on_fail ('inconclusive on a probe -> STUCK: EXP-26 (design call)') the analyst must NOT force a {go_B_skip_C | go_C_then_B | retire_sign_replacement} routing decision."
route: "DO NOT launch Step B/C yet. Fix the G_dense parallel backward on exp/26-* (re-probe until cos(G_dense, G_anchor) ~ 1 in a comm-eff arm), then RE-RUN Step A. The warm box may be reused for the re-run if it survives; otherwise reprovision."
```

## Step A — geometry audit boxes (all from REAL fp32 tensors, post-tick-3)

- [ ] `cos(G_dense, G_comp) >= 0.95` for plain PowerSGD (H1) — **NOT MEASURABLE.** A1 dumped no `G_comp` (plain PowerSGD = no merger). A2's `cos(G_dense, G_comp) = 0.0072`, but the `G_dense` reference is corrupt on the codec-ON path (see Root cause), so this cannot test compression. Observed (uninterpretable): 0.0072.
- [ ] `cos(G_dense, G_corr)` materially below plain-PowerSGD's `cos(G_dense, G_comp)` (H1) — **NOT MEASURABLE.** A2 `cos(G_dense, G_corr) = -0.0033`; both endpoints rest on the broken `G_dense`. (-0.0033 < 0.0072 is technically true but both are at the floor — not evidence of merger collapse, just a broken reference.)
- [ ] `Q_act` activation capture `>= 0.99` AND `Q_act` update-energy capture with off-principal share (H2) — **activation FAILS / update INCONCLUSIVE.** activation capture median = 0.525 (A1) / 0.525 (A2), well below 0.99. Update-energy `||QQᵀG||²/||G||²` median = 0.282 (A1) / 0.276 (A2), off-principal share ~0.72 — BUT this projects onto the broken `G_dense`, so the update-capture number is not trustworthy. (The 0.525 activation figure is also suspect — the stratified post-warmup recon was ~0.065, not the on-box ~0.024, because the subset spans pre-Q-warm boundary ticks.)
- [x] sign-agreement(`M`, `G_comp`) and sign-agreement(`G_anchor_fresh@delay_K=0`, `G_comp`), magnitude-weighted, at delay_K∈{0,5} (H3) — **CONFIRMED (clean — no G_dense dependence).** delay_K=5: 0.490. delay_K=0: 0.490. Both inside [0.45, 0.55] ⇒ ≈ coin-flip even at delay_K=0 ⇒ **sign-replacement is structurally unrecoverable (H3 confirmed)**.
- [ ] machine-readable DECISION emitted — **emitted as STUCK** (cannot be one of the three routing values while H1/H2 rest on a broken probe).

## H1 / H2 / H3 verdicts

| Hypothesis | Verdict | Basis |
|---|---|---|
| **H1** (compression direction-benign: `cos(G_dense, G_comp) >= 0.95`) | **INCONCLUSIVE** | `G_dense` probe corrupt on codec-ON path — cannot test |
| **H2** (`Q_act` misses off-principal GRPO update energy) | **INCONCLUSIVE** | update-capture projects onto the same corrupt `G_dense` |
| **H3** (sign-disagreement structural, ≈coin-flip at delay_K=0) | **CONFIRMED** | `sign0 = sign5 = 0.490` ∈ [0.45, 0.55]; uses M / G_fresh_anchor / G_comp only |

## Four cosines / ratios (as computed; H1/H2 ones flagged untrustworthy)

| quantity | A0_dense (control) | A1_powersgd_r77 | A2_signed_ema_a0p5 |
|---|---|---|---|
| `cos(G_dense, G_comp)` median | 0.9757 | n/a (no G_comp) | 0.0072 ⚠ broken probe |
| `cos(G_dense, G_corr)` median | 0.9757 | n/a | -0.0033 ⚠ broken probe |
| `Q_act` activation capture median | n/a (no codec) | 0.525 | 0.525 |
| `Q_act` update-capture `||QQᵀG||²/||G||²` median | n/a | 0.282 ⚠ (vs corrupt G) | 0.276 ⚠ (vs corrupt G) |
| off-principal update share | n/a | 0.718 ⚠ | 0.724 ⚠ |
| sign-agreement @delay_K=5 | 0.982 | n/a (no M) | **0.490** |
| sign-agreement @delay_K=0 | 0.999 | n/a | **0.490** |
| fp32 dump fidelity (max recon drift) | — | 1.0e-4 ✓ | 8.8e-5 ✓ |

## Root cause — why H1/H2 are not measurable (the design call)

`G_anchor` and `G_fresh_anchor` are GENUINE full uncompressed gradients (the three
realism invariants were verified GREEN on the on-box training path). Using them as
the dense ground truth, the **parallel uncompressed `G_dense` probe** behaves as:

| arm | median cos(G_dense, G_anchor) | median cos(G_dense, G_fresh_anchor) | norm(G_dense)/norm(G_anchor) |
|---|---|---|---|
| A0_dense (codec OFF) | **+0.928** | **+0.986** | 0.69 |
| A1_powersgd_r77 (codec ON) | **−0.153** | −0.157 | **0.051** |
| A2_signed_ema_a0p5 (codec ON) | **−0.021** | −0.024 | **0.062** |

In the **dense** arm the probe agrees with the true dense gradient (cos +0.93/+0.99,
comparable norm). In **both comm-eff arms** the `G_dense` probe is orthogonal/anti-
correlated to the true dense gradient and ~20× too small. A correct dense backward
over the same batch cannot be anti-correlated with the anchor's full uncompressed
backward. **The parallel uncompressed `G_dense` backward (`transformer_impl.py`, the
plan's NAMED highest-risk failure surface #1) did not compose with the codec-ON path —
it only produces a faithful gradient when the codec is OFF.** This is exactly the
`fail-fast` situation the `## Correctness invariants` were meant to catch but the
off-path-parity probe only exercised the dense arm.

The locked #25 result is that plain PowerSGD r77 **ties dense at 0.7415** (high cosine
expected). A near-zero `cos(G_dense, G_comp)` is therefore the broken probe, NOT
compression rotating the update — so H1 cannot be falsified or confirmed from this run.

This is **not** a science result and **not** a measurement that approves anything: it
is a probe defect. Per the `## Analyst predicate` and `## Experiment sequence`
Step-A `on_fail` ("if the audit runs clean but is inconclusive on a probe → STUCK:
EXP-26 (design call)"), and the `STUCK` / `RECON_REL_ERROR_DRIFT`-family rescue
triggers, the gate returns **STUCK**, not a routing DECISION.

## What survives the defect (do not lose this)

- **H3 is confirmed and clean**: sign-agreement(`G_fresh_anchor`, `G_comp`) = 0.490 at
  delay_K=0 (and 0.490 at delay_K=5) — magnitude-weighted, on real tensors with no
  `G_dense` dependence. Sign-replacement is structurally a coin-flip **even with a
  fresh (delay_K=0) anchor**, so staleness is NOT the cause; this corroborates the #25
  `retire_sign_replacement` direction independently of the broken probe. The successor
  `ef_powersgd` (Step B) is direction-preserving with NO sign term — H3 does not block it.
- **fp32 dump fidelity PASSES** (recon drift ~1e-4 < 1e-3): the dump *path* is correct;
  the defect is confined to what the `G_dense` *backward* computed, not how it was written.
- **Substrate is correct** (resolved_params.txt): PowerSGD r77, anchor owns_q, delay_K=5,
  clean_cadence=0, GRPO no-KL/no-entropy. Not the problem.

## Required fix before re-running Step A (for the runner, on exp/26-*)

1. Fix the parallel uncompressed `G_dense` backward in `transformer_impl.py` so it
   composes with the codec-ON path (FSDP1 + grad-ckpt + the compressed boundary). The
   symptom (norm ~5% of true, anti-correlated) points at either (a) `G_dense` capturing
   the *residual / off-subspace* component rather than the full grad when the codec is
   active, or (b) the second backward reading a grad buffer already overwritten by the
   compressed path. The anchor-clone `_canon` naming fix (memory) is relevant.
2. **Acceptance gate for the fix:** in a 1–3 step probe on a *comm-eff* arm,
   `cos(G_dense, G_anchor) >= ~0.95` and `norm(G_dense)/norm(G_anchor) ~ 1` — i.e. the
   dense probe must match the anchor's full uncompressed grad on the codec-ON path the
   way it already does in the dense arm. Only THEN are H1/H2 measurable.
3. While there: add `._fsdp_wrapped_module` canonicalization to
   `research/scripts/geometry_audit.py` so the cosines align without per-arm hand-work
   (current script mis-keys G_dense vs G_comp/G_corr → all-NaN cosines).
4. Re-run Step A (3 arms) and re-audit. H3 already answered; the re-run is to recover
   H1 (`cos(G_dense, G_comp)` for A1) and H2 (`Q_act` update-capture vs a TRUSTWORTHY
   `G_dense`), which together choose `go_B_skip_C` vs `go_C_then_B`. All paths still
   route to Step B (ef_powersgd) eventually.

## Budget / provenance

- `check_budget.py`: lifetime $55.02 / month $55.02 (cap $1500); running_count=1,
  running_dph=12.84 — within `max_dph=24` and well under `max_gpu_hr=60`. The Step-A
  re-run after the probe fix fits the remaining staged budget.
- `resolved_params.txt` + `resolved_cmd.txt` captured (95 params, 1 main_ppo invocation;
  the A1 arm's expansion). Substrate matches the locked control surface.
- `diff_against_baseline.py --baseline EXP-25` rc=0 (wrote baseline_diff.md;
  EXP-20/23 dirs cleared, references read from W&B per the issue — documented condition).
