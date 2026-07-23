# Verdict — 89-prf-codec-autoresearch-benign-kl

VERDICT: PASS

Autoresearch search loop ran to the money gate and shipped a **symmetric clean
negative** with a full 8-candidate trajectory and a gate-legal recommended
config. The plan's Hypothesis section explicitly declares this a PASS: "if no
candidate tried within the GPU-hour budget beats the constant-rescale incumbent
under that gate ... the incumbent stays PRF constant. Either way we ship the
trajectory and the final recommended config." No candidate reached the goal
(drive codec-view entropy toward the dropout ~0.13 floor while holding
`actor/kl_loss` at or below the incumbent); the one gate-legal accept (exact-k)
is within noise. Every number below is greppable from `runs/89-prf-codec-autoresearch-benign-kl/`.

Judged against the plan cache (`.claude/state/plan-cache/89.md`) as amended by
the operator on the issue (2026-07-21..23). Per-trial + Stage-4 validation and
the winner-projection-off pair were operator-CANCELLED; the 30-step horizon and
the step-21 read of candidate 8 are operator-sanctioned. These are amendments to
the predicate, not analyst-unmeasured criteria.

---

## Reference frame (Stage 2, fresh controls, this project) — `metrics/train_<cell>.log` @ step 40

| metric (WandB key) | dense-control | dropout p=0.1 | prf-constant-incumbent |
|---|---|---|---|
| `actor/entropy` | 0.18073 | 0.13702 | **7.8158** |
| `actor/kl_loss` | 0.0033028 | 0.12373 | 0.11658 |
| `actor/ppo_kl` | -2.95e-06 | 0.090155 | -2.27e-04 |
| `actor/comm_eff/mask_ratio` | n/a | n/a | 0.94995 |
| `critic/score/mean` | 0.64551 | 0.60425 | 0.63159 |
| `training/rollout_probs_diff_mean` | 0.0022582 | 0.0018277 | 0.90265 |
| `actor/perf/max_memory_allocated_gb` | 45.806 | 45.756 | 110.48 |

Reads confirm the three-divergence structure: PRF-constant reproduces
codec-view entropy ~7.82 (structural, from the `h*mask/(1-p)` 20x amplification)
with `ppo_kl` at machine-zero (within-step mask bit-identity intact) and a slow
`kl_loss` climb (0.032@5 to 0.117@40), while dropout is entropy ~0.14 with a flat
`kl_loss` ~0.12 plateau. Peak GPU mem 110.48 GB after the first anchor refresh
(two-circuit clone at ~step 10) — at the ~110 GB gate, well inside 141 GB. No OOM.

**Incumbent matched-step-30 reference** (trials ran to 30, so all candidates are
judged here): `entropy=7.8134 kl_loss=0.064933 ppo_kl=0.0029998 mask=0.94995
score=0.59399 rpd=0.89082`.

---

## Incumbent trajectory — 8 candidates vs incumbent @ matched step 30 (candidate 8 @ 21)

Amended 5-part gate: KEEP iff (`actor/entropy` reduced) AND (`actor/kl_loss` no
higher) AND (`actor/ppo_kl` < ~1e-2) AND (`mask_ratio` in [0.94,0.96]) AND
(`critic/score/mean` early slope not degraded). Entropy compared intra-PRF
(apples to apples per plan).

| # | candidate (lever) | entropy | Δent | kl_loss | Δkl | ppo_kl | mask | score | verdict | one-line diagnosis |
|---|---|---|---|---|---|---|---|---|---|---|
| — | prf-constant-incumbent | 7.8134 | — | 0.064933 | — | 0.0030 | 0.94995 | 0.59399 | incumbent | 20x constant rescale; structural entropy, machine-zero ppo_kl |
| 1 | prf-rms-match (`rescale_mode=rms_match`) | 5.8606 | -1.95 | 2.4678 | **+2.40 (38x)** | 0.0098 | 0.94995 | 0.59961 | REJECT | removing the variance blow-up drops entropy but re-aligns the surviving 5% across steps → coherent residual → KL explodes 38x |
| 2 | prf-exact-k (`exact_k=true`) | 7.8128 | -0.0006 | 0.061633 | -0.0033 | 0.0033 | 0.94995 | 0.59326 | ACCEPT (gate-legal, marginal) | fixing the per-token keep count to round((1-p)H) is within-noise on every axis — **count jitter is irrelevant**; kept only because it is harmless + a hair lower KL |
| 3 | prf-nonuniform-p (`p_by_boundary` mean 0.95) | 7.8030 | -0.010 | 0.073971 | +0.0090 | 0.0027 | 0.94999 | 0.60229 | REJECT | reallocating the byte budget across depth at constant average moves entropy negligibly and slightly WORSENS KL — **depth reallocation is low-value** |
| 4 | prf-antithetic (`antithetic=true`) | 7.8135 | +0.0001 | 0.060319 | -0.0046 (-7.1%) | 0.0022 | 0.94995 | 0.60693 | REJECT (entropy not reduced) | complementary cross-step draw leaves entropy untouched (structural) and buys only **~7% KL reduction** — temporal decorrelation is a weak lever |
| 5 | prf-frlr r32k44 (`frlr rank32 k44`, `rescale=none`) | 2.9141 | -4.90 | 0.32856 | +0.264 (5.1x) | 0.0050 | 0.94987 | 0.58984 | REJECT | NEW lever (seed queue exhausted): replace 20x amplification with a fast-refresh rank-32 residual correction. Crushes entropy 7.81→2.91 (**amplification DOES drive the inflation**) but re-injects coherent residual energy → KL 0.33 |
| 6 | prf-frlr r16k60 (`rank16 k60`) | 3.4259 | -4.39 | 0.45316 | +0.388 (7x) | 0.0065 | 0.94987 | 0.57251 | REJECT (KL higher; ppo_kl 0.0109@20) | shrinking correction rank 32→16 raises residual energy → higher KL; the r+k≈76 budget trades rank for worse alignment |
| 7 | prf-frlr slowq (`rank32 k44`, slow Q: 14 vs 203 refreshes) | 5.8597 | -1.95 | 1.0532 | +0.988 (16x) | 0.0049 | 0.94987 | 0.61035 | REJECT (KL higher, unstable) | slowing Q refresh makes the basis stale → KL both higher AND **sawtooths** (1.46→2.85→0.74→1.05); slope = Q adaptivity |
| 8 | prf-frlr r48k28 (`rank48 k28`) @ step 21 | 3.27433 | -4.5 | 0.05050636 | +0.010 vs incum@21 (0.04044), **climbing** | 0.0044 | 0.94987 | 0.48755 | REJECT (incomplete + KL climbing; **goal-nearest**) | highest correction rank held KL lowest of the FRLR family (0.003 through step 9 via fast-Q, 0.0505 by step 21) while crushing entropy to 3.27 — closest any candidate came to the goal, but KL had already crossed above the incumbent @21 and was rising ~0.023/step; never reached the step-30 horizon (cut by budget teardown) |

All `ppo_kl` < 1e-2 everywhere (within-step mask bit-identity preserved across
old/train/reference forwards on every codec) except a single transient
0.010888 @ step 20 on r16k60. `mask_ratio` in [0.94, 0.96] on every cell.

Diagnose-and-propose chain (loop never terminated on a reject): seed queue
rms-match → exact-k → nonuniform-p → antithetic exhausted with no goal hit; the
antithetic/nonuniform nulls plus the rms-match KL blow-up pointed at the
residual (not the count/schedule), so the loop proposed the NEW FRLR lever
(fast-refresh low-rank residual correction) and swept its rank/refresh budget
r32k44 → r16k60 → slowq → r48k28 until the money gate.

---

## Recommended config

**Shipped (gate-legal): PRF constant + exact-k.** The only lever that passes the
amended gate without cost; exact-k marginally lowers KL (0.0649→0.0616 @30) at
zero entropy or capability cost.

```
COMM_EFF_ENABLED=true
COMM_EFF_COMPRESSION_TYPE=prf_mask
COMM_EFF_MASK_ENABLED=true
COMM_EFF_MASK_P=0.95
COMM_EFF_MASK_RESCALE_MODE=constant
COMM_EFF_MASK_EXACT_K=true
COMM_EFF_MASK_RECOMPUTE=true
COMM_EFF_MASK_REFERENCE=true
COMM_EFF_MASK_PP_SIZE=8
COMM_EFF_ANCHOR_OWNS_Q=false
COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP=false
# anchor.lookahead_anchor=true (projection-off pair operator-CANCELLED; see #88 close-out)
```

**Goal-nearest, documented (NOT recommended): FRLR r48k28**
(`rescale_mode=none frlr=true frlr_rank=48 frlr_k=28`). Measured tradeoff:
entropy 7.81→3.27 (the largest inflation cut of any candidate at low-ish KL) but
KL 0.003→0.0505 by step 21 and still climbing past the incumbent; unmeasured at
the step-30 horizon. It is the direction to reopen if the search budget is
re-approved (larger rank + faster Q to hold the residual alignment), not a
shippable config today.

---

## Mechanism findings (all grounded in the trajectory reads above)

1. **Codec-view entropy is structural inflation, and it IS reducible.** The ~7.82
   entropy is the `h*mask/(1-p)` 20x variance amplification; touching the rescale
   (rms_match 5.86, FRLR 2.91–3.43) moves it directly. So the amplification is a
   real driver of the entropy inflation — refuting the "amplification is
   irrelevant" half of the null.
2. **But entropy and the reference-KL climb are COUPLED through the residual.**
   Every lever that cut entropy (rms_match, all FRLR) re-introduced coherence in
   the discarded/rescaled energy and paid for it in `kl_loss` (2.47, 0.33, 0.45,
   1.05). KL **level** = residual energy × alignment asymmetry; KL **slope** = Q
   adaptivity (fast-Q r48k28 held 0.003 to step 9; stale/slow-Q sawtooths). You
   cannot cheaply trade one for the other at this horizon — the benign-KL,
   low-entropy corner was not reachable within budget.
3. **Count jitter is irrelevant** (exact-k within noise on every axis).
4. **Depth reallocation is low-value** (nonuniform-p: negligible entropy move,
   slightly worse KL).
5. **Temporal decorrelation is weak** (antithetic: entropy untouched, only ~7.1%
   KL reduction: (0.064933−0.060319)/0.064933).

Net: the constant-rescale incumbent stays the recommended PRF codec; the
amplification is reducible but not without re-inflating the residual KL, i.e. the
residual is coherence/anchor-coupled (findings Section 8), consistent with the
plan's declared clean-negative conclusion.

---

## Success-criteria checklist (amended items marked)

- [✓] all 4 stage gates evaluated: Stage 1 (levers resolved in config-echoes;
  incumbent = canonical byte-identical PRF ⇒ off-path parity) ✓; Stage 2
  (3 controls to step 40, no OOM, anchor clone fits @110.48 GB, reference frame
  logged) ✓; Stage 3 (8-candidate loop, exited at money gate) ✓; Stage 4
  **AMENDED-CANCELLED** (val + projection pair cancelled by operator; recommended
  config + trajectory still shipped).
- [✓] incumbent trajectory emitted: 8 candidates in order with per-candidate Δ`entropy`/Δ`kl_loss`/`ppo_kl`/`mask_ratio`/`score` and verdict.
- [✓] every candidate judged by the amended 5-part gate at the matched horizon
  (30, operator-sanctioned via `step_target_fallback`); `mask_ratio` in
  [0.94,0.96] on every cell including the one accept (exact-k 0.94995).
- [~] **AMENDED** final recommended config shipped ✓; the ONE confirmation val
  (`val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1`) was operator-CANCELLED
  (no MATH val pass anywhere; capability guard = free `critic/score/mean` slope,
  which tracks dense: incumbent 0.34→0.63 vs dense 0.39→0.65).
- [~] **AMENDED** matched projection pair (`lookahead_anchor` true vs false)
  operator-CANCELLED (budget teardown + close-out-now); the lever is characterized
  in the #88 close-out. `lookahead_anchor=true` held on every cell.
- [✓] search loop exited ONLY at the money gate: harness budget guard hard-enforced
  the ledger cap at ~03:05Z 2026-07-23 (teardown_reason=budget-exceeded); operator
  amendment 3 declares this the plan's money-gate exit. Every rejection has a
  recorded diagnose-and-propose entry (seed queue → FRLR family).
- [✓] `rollout_probs_diff_mean` ~0.9 on all PRF cells (0.77–0.90) recorded as
  EXPECTED (rollout uncompressed; dense/dropout 0.0018–0.0027). Not a stop trigger.
- [✓] non-codec surface identical across all cells (config-echo: train_batch 512,
  ppo_mini 256, n=8, prompt/response 1024/3072, lr 1e-6, kl_loss_coef 0.001,
  cadence/delay 20/20); the sole sanctioned exception (projection-off pair) was
  cancelled, so no non-codec variance was ever introduced.

---

## Budget / teardown record

- Attach: POLL 0 @ 2026-07-21T10:45:38Z (dense START 2026-07-21T10:39:31Z; operator-noted attach 10:34Z).
- Last live poll: POLL 65 @ 2026-07-23T03:02:12Z; POLL 66 @ 03:06:01Z rc=255 (instance gone).
- Teardown: budget-exceeded ~03:05Z 2026-07-23 (harness ledger cap; = plan money-gate exit).
- ~40 GPU-hr, single H200, ~40.5 h wall, ~USD 160. 11 cells (3 controls + 8 trials). Zero env incidents.

---

## Notes

- **RESOLVED_CONFIG_MISSING**: `capture_resolved_config.py` returned rc=1 — the
  synced `metrics/train_<cell>.log` are the on-box metric TAIL only; the launcher
  `python3 -m verl.trainer.main_ppo` `set -x` trace was not synced, so there is no
  `resolved_cmd.txt`. Ground-truth codec + non-codec config was recovered instead
  from each cell's own Hydra config-echo (CommEffMaskConfig dump) →
  `resolved_params.txt`. No plan-vs-ran divergence found: the incumbent echo
  matches the run.json canonical PRF cell exactly (p=0.95, rescale_mode=constant,
  pp_size=8, exact_k/antithetic false, p_by_boundary=[]).
- `rollout_probs_diff_mean` ~0.9 on PRF cells is EXPECTED (rollout is
  uncompressed; the codec touches only the trainer forward) — explicitly not a
  failure and not a stop trigger.
- **No validation was run** (operator-cancelled): there is no MATH val number in
  this run. Capability was guarded during the search by the free
  `critic/score/mean` early slope only, which held at dense levels on every cell.
- Candidate 8 (prf-frlr-r48k28) has no on-box train log or done flag — the
  instance was destroyed mid-cell at the budget cap. Its step-1..21 curve is read
  from `monitor-detail.log` (cycle58–cycle62 SPLITMETRIC blocks); judged at step
  21 per operator decision.
- Backfill: one attempt; `scripts/backfill_wandb.py` is incompatible with this
  run's per-issue WandB project (it hardcodes `verl_compression_research` +
  `resume='must'` with the cell name as run id) and errors out. It parsed the
  local log correctly (score=0.631592 == step-40), confirming every final-step row
  is already local, so no curve is missing and backfill is not blocking.
- REVISE not considered (money gate exhausted, operator chose close-out); the
  clean negative is fully measured, so STOP (budget-exhausted-UNMEASURED) does not
  apply — this is the planned money-gate exit with a complete trajectory.
