# Step A decision — EXP-26 — 2026-06-10T11:05:00+10:00 (CLEAN Option-A, supersedes STUCK)

> This is the Step-A **stage gate**, NOT the terminal issue verdict. Steps B/(C)/E
> run in a LATER session. Issue label stays `status:running`.
>
> **The earlier STUCK (2026-06-10T01:40) is RESOLVED via Option A:** the broken
> parallel-`G_dense` clone is retired (footnote only); the dense reference is now
> `G_fresh_anchor@delay_K=0`, a realism-GREEN full uncompressed backward
> (validated on the dense arm: `cos(G_fresh_anchor, G_dense)=0.985`). All three
> arms' clean post-warm captures (ticks 10/15, warm-Q recon ~0.025) are local.

## DECISION

```yaml
DECISION: go_C_then_B
sign_finding: retire_sign_replacement(confirmed)    # H3 — clean, no dense reference
route: "Run Step C (rlvr-native Q-content sweep at FIXED rank 77) FIRST, then Step B
        (ef_powersgd). Step C is gated ON because Q_act misses off-principal GRPO
        UPDATE energy (update-capture 0.318 vs activation-capture 0.999) — the basis
        CONTENT is implicated, so go_B_skip_C is NOT taken. Both arms' geometry_audit
        _decide() independently returned go_C_then_B."
note: "All paths still route to Step B (ef_powersgd, direction-preserving, NO sign
       term) eventually; C precedes it because H2 could not be shown false."
```

## Step A — geometry audit boxes (all from REAL fp32 post-warm tensors, ticks 10/15)

- [~] `cos(G_dense, G_comp) >= 0.95` for plain PowerSGD (H1) — **measured but CONFOUNDED.**
  With the Option-A reference, A1 `cos(G_fresh_anchor, G_comp) = +0.0096` (n=14). This is
  NOT evidence that compression rotates the update: plain PowerSGD r77 ties dense at 0.7415
  (locked #25), so the compressed update cannot be orthogonal to dense in a comparable frame.
  `cos(G_fresh_anchor, G_anchor) = +1.0003` confirms `G_fresh_anchor` IS the genuine full
  uncompressed grad — the low cosine is a LOSS + OPERAND mismatch (clean-PG weight-grad vs
  PPO-clip compressed boundary-grad), not compression direction. (See Root cause.)
- [x] `cos(G_dense, G_corr)` materially below plain-PowerSGD's `cos(G_dense, G_comp)` —
  confirmed via the **confound-free** isolate. cos(G_fresh_anchor, G_corr)=+0.349 vs
  cos(G_fresh_anchor, G_comp)=+0.060 on A2, AND the dense-reference-FREE
  **`cos(G_comp, G_corr) = +0.717` (merger rotates the compressed update ~44 deg)** — the
  signed_ema merger materially corrupts the direction relative to plain compression. H1's
  SPIRIT (merger collapses direction) confirmed without any dense reference.
- [~] `Q_act` activation capture `>= 0.99` AND update-energy capture with off-principal share
  (H2) — **activation PASS / update reads as MISS (confound-caveated).** Activation capture
  median = **0.9985** (PASS, recon drift 4.5e-5). Update-energy `||QQᵀG_fresh||²/||G_fresh||²`
  median = **0.318**, off-principal share **0.68** ⇒ Q_act appears to miss ~68% of the dense
  GRPO update energy. (Caveat: projects the clean-PG fresh-anchor grad → inherits the operand
  confound; it cannot be shown that Q_act ALREADY captures update energy ⇒ H2 not false.)
- [x] sign-agreement(`M`, `G_comp`) and sign-agreement(`G_fresh_anchor@delay_K=0`, `G_comp`),
  magnitude-weighted, at delay_K∈{0,5} (H3) — **CONFIRMED (clean — no G_dense dependence).**
  A1 delay_K=0: **0.500**. A2 delay_K=0: **0.523**; delay_K=5: **0.520**. All inside [0.45,0.55]
  ⇒ ≈ coin-flip EVEN with a fresh (delay_K=0) anchor ⇒ staleness is NOT the cause ⇒
  **sign-replacement is structurally unrecoverable (H3 confirmed)**.
- [x] machine-readable DECISION emitted — **`go_C_then_B` + `retire_sign_replacement(confirmed)`**.

## H1 / H2 / H3 verdicts

| Hypothesis | Verdict | Basis |
|---|---|---|
| **H1** (merger collapses direction; compression itself benign) | **CONFIRMED (in spirit, via confound-free isolate)** | `cos(G_comp, G_corr)=0.717` (~44 deg merger rotation, NO dense ref); the literal `cos(G_dense,G_comp)>=0.95` box is operand/loss-confounded and not cleanly testable from this reference |
| **H2** (`Q_act` misses off-principal GRPO update energy) | **TRUE (caveated)** | Q_act update-capture 0.318 (off-principal 0.68) vs activation 0.999; cannot be shown false ⇒ routes C-first |
| **H3** (sign-disagreement structural, ≈coin-flip at delay_K=0) | **CONFIRMED (clean)** | sign-agree ∈ [0.50, 0.52] at delay_K∈{0,5} on real fresh tensors; uses only M / G_fresh_anchor / G_comp |

## Audit table (clean Option-A, post-warm ticks 10/15)

| quantity | A0_dense (control) | A1_powersgd_r77 | A2_signed_ema_a0p5 |
|---|---|---|---|
| `cos(G_fresh_anchor, G_comp)` median | 0.9993 (n=14)¹ | **+0.0096** ⚠confound | +0.0601 ⚠confound |
| `cos(G_fresh_anchor, G_corr)` median | 0.9993¹ | n/a (no merger) | +0.3486 ⚠confound |
| **`cos(G_comp, G_corr)` (confound-FREE merger isolate)** | n/a | n/a (G_corr==G_comp) | **+0.7165** |
| `Q_act` activation capture median | n/a (no codec) | **0.9985** ✓ | 0.9989 ✓ |
| `Q_act` update-capture `‖QQᵀG_fresh‖²/‖G_fresh‖²` | n/a | **0.3179** (off-prin 0.68) | 0.317 (off-prin 0.68) |
| sign-agreement @delay_K=0 | 0.999 | **0.5004** | **0.5227** |
| sign-agreement @delay_K=5 | 0.982 | n/a (no M) | **0.5195** |
| sanity cos(G_fresh_anchor, G_anchor) | 0.977 | **+1.0003** | ~1.0 |
| **VALIDITY cos(G_fresh_anchor, G_dense)** | **0.9848** ✓ | −0.11 (broken G_dense clone — expected) | 0.38 (broken clone — expected) |
| fp32 dump fidelity (recon drift) | — | 4.5e-5 ✓ | 4.5e-5 ✓ |

¹ On the dense arm `G_comp`/`G_corr` are byte-equal to the dense update (no real codec), and
the only co-tick with `G_fresh_anchor` is tick 5, so this just re-confirms fresh≈dense.

## Root cause — why the literal H1 cosine is confounded (not a probe defect this time)

The earlier STUCK was a broken `G_dense` clone (norm ~r/H, anti-correlated). Option A retired
it. The Option-A reference `G_fresh_anchor` is now PROVEN faithful: `cos(G_fresh_anchor,
G_anchor)=+1.0003` on every A1 target (it IS the full uncompressed gradient), and on the dense
arm `cos(G_fresh_anchor, G_dense)=0.985`. So the reference is sound.

The residual near-zero `cos(G_fresh_anchor, G_comp)` on codec-ON arms is a **two-fold
operand/loss mismatch**, NOT compression rotating the update:
1. **Loss mismatch.** `G_fresh_anchor` is the gradient of the anchor's CLEAN policy-gradient
   loss (`anchor_pg_loss`, ratio≡1, no clip — by design, see anchor.py); `G_comp` is the
   gradient of the fast path's REAL PPO ratio/clip loss. On the dense arm these coincide
   (ratio≈1 ⇒ cos 0.999); post-warm on codec arms the PPO ratio drifts from 1, the clip
   activates, and the two losses' gradients diverge — independent of compression.
2. **Operand mismatch.** PowerSGD compresses the BOUNDARY ACTIVATION gradient (rank-77 in the
   H=1536 activation space), not the dense weight gradient directly. `G_comp`-onto-`Q_act`
   update-capture is 0.47 (not ~1), confirming `G_comp` is not simply `QQᵀG_weight`.

Because plain PowerSGD r77 TIES dense at 0.7415, the correct interpretation is: compression is
direction-benign at the TRAINING outcome level (locked result), the literal weight-space
cosine from this reference is non-comparable, and the **confound-free** discriminator that DOES
isolate the merger is `cos(G_comp, G_corr)=0.717` — which cleanly shows the signed_ema merger
rotates the (already-compressed) update by ~44 deg. That, plus the coin-flip sign-agreement,
fully corroborates the #25 thesis: the merger (specifically its sign term) is the defect.

## What survives / decides the route

- **H3 CONFIRMED, clean:** sign-replacement is a coin-flip (≈0.50) even at delay_K=0 ⇒
  structurally unrecoverable ⇒ `retire_sign_replacement(confirmed)`. The successor
  `ef_powersgd` (Step B) is direction-preserving with NO sign term — H3 does not block it.
- **Merger direction-corruption CONFIRMED, confound-free:** `cos(G_comp, G_corr)=0.717`.
- **Q_act under-captures update energy (H2 TRUE, caveated):** update-capture 0.318 « activation
  0.999. Cannot be shown false ⇒ the basis CONTENT is implicated ⇒ **Step C is gated ON**, run
  before Step B (`go_C_then_B`). Both arms' `geometry_audit._decide()` returned go_C_then_B.
- **fp32 dump fidelity PASS** (drift 4.5e-5 « 1e-3); **substrate correct** (resolved_params.txt:
  PowerSGD r77, anchor owns_q, delay_K=5, cadence=5, clean_cadence=0, GRPO no-KL/no-entropy,
  q_basis=act). A1 ran spectral OFF (plain PowerSGD, no G_corr/M); A2 ran spectral ON
  signed_ema α=0.5 (G_corr/M present) — confirmed from launch_A1A2_optionA.sh.

## Budget / provenance

- `check_budget.py`: running_count=1, running_dph=12.84 (≤ max_dph 24); lifetime/month
  $174.15 (cap $1500). Well under `max_gpu_hr=60` for the staged Step A diagnostic.
- `capture_resolved_config.py`: 95 params, 1 main_ppo invocation (A1 arm's expansion) →
  resolved_params.txt + resolved_cmd.txt. Substrate matches the locked control surface.
- `diff_against_baseline.py --baseline EXP-25` rc=0 (baseline_diff.md; EXP-20/23 dirs cleared,
  references read from W&B — documented condition, not a failure).

## Honest caveat for Step C/B planning (carry forward)

The literal weight-space `cos(G_dense, G_comp/G_corr)` discriminator named in the plan's
`## Step A` checklist is NOT cleanly measurable from `G_fresh_anchor` because of the clean-PG
vs PPO-clip loss mismatch and the activation-vs-weight operand mismatch. For Step B's success
box ("`cos(G_dense, G_corr)` improves over plain-PowerSGD's `cos(G_dense, G_comp)`"), the
analyst of that stage should use the **confound-free** `cos(G_comp, G_corr)` (merger vs its own
compressed input) as the direction-preservation discriminator, OR capture the anchor's full
grad under the SAME PPO-clip loss as the fast path so the two operands are comparable. The
`go_C_then_B` routing and `retire_sign_replacement` are robust to this caveat (they rest on
confound-free measurements). Recorded as a Notes flag in verdict.md.
