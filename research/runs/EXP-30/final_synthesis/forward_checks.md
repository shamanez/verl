# Forward checks from the best solution (B2) — ranked, theory-grounded

> **This is a planning DOCUMENT, not an issue queue.** Nothing here is filed as a GitHub issue; the
> operator was explicit. Each item below is framed as *"what to run next on the best solution"* and is
> ranked by decision-value-per-GPU-hr. No threshold below softens a pre-registered rule; no cell here
> launches without the operator approving it.
>
> **Author:** forward-planner (team commeff-grpo-verdict, task #2) · 2026-06-13
> **Sources** (same as the synthesizer / task #1): `runs/EXP-30/verdict.md` (incl. the ext100 addendum),
> `runs/EXP-30/PATH_FORWARD.md`, `runs/EXP-30/PROGRESS.md`, `runs/EXP-30/pathforward/{mechanist,critic,strategist}.md`,
> `runs/EXP-30/resolved_params_B2.txt`, `runs/EXP-30/metrics/{stepA_fires.jsonl, B2_delta_per_fire.jsonl,
> ext100_delta_per_fire.jsonl}`, `runs/EXP-30/train_plain_replay_substrate.log` (C2), `.claude/GOAL.md`.
> Numbers are quoted/recomputed from these, not invented.

---

## 0. The best solution and the gap to GOAL (so the ranking is legible)

**Best solution = B2:** K-delayed *exact* codec residual. `G_corr(t) = G_comp(t) + λ·δ`,
`δ = G_anc_rep(t) − G_comp_ring(t−K)`, **λ=1, β_anc=0**, on the EXP-29 paired-replay substrate
(PowerSGD r=77 act-basis Q, anchor-owned Q updated by power iteration, cadence = delay_K = 5,
clean_cadence = 0, replay_paired_batch=true, snapshot_device=cpu). Knob ground truth:
`resolved_params_B2.txt`.

**GOAL scorecard for B2 today** (`.claude/GOAL.md` §"Done"):

| GOAL criterion | status on B2 | residual gap |
|---|---|---|
| 1. Stable | emission-free through **100 steps, seed 0** (ext100) | censored at 100; single seed; mild late val decay 0.7536@50 → 0.7400@100 |
| 2. Parity | val@50 = **0.7528**, dense-old 0.7536 (−0.0008 ≈ 1 problem) | **statistically unestablished** — at SE 0.0119 (n=1319) B2 is indistinguishable from dense AND ~1.9σ above floor; T2 |
| 3. Savings | fast-path bytes_ratio 0.0505 (≈19.8×) | **anchor M-allreduce + Q-broadcast traffic uncounted**; the honest inter-stage number is not 19.8× (T6) |
| 4. Reproducible | one launcher (`launch_B2.sh`), one seed | no seed replicate; no canonical `examples/` promotion yet |

So the frontier is no longer "does a correction help" (B2 says yes) but: **(a) is the win real past the
ignition horizon and across seeds; (b) which adjacent operators on B2 are now cheap to settle; (c) can
B2's near-parity be pushed to a *surpass* by attacking the root cause (basis mismatch, m7); (d) can the
savings be made strictly better (GOAL-3 upside) without losing val.** Every candidate below is judged on
how much of that surface it closes per GPU-hr, counting a clean negative (retires a sub-route) as value.

### Mechanism facts every candidate is grounded in (the toolbox)

- **F1 (weight-space geometry, the prize).** At identical (batch, θ): pooled `cos(G_anc_rep, G_comp_ring) = +0.007`
  (statistically ⊥; only 6.9% of 196 matrices have |cos|>0.2; per-target, not a median artifact — critic T5),
  settled `‖G_anc_rep‖/‖G_comp_ring‖ ≈ 0.29`. **The codec error is ~92% of the compressed gradient's energy;**
  the true gradient rides on it at ~⅓ norm, orthogonally. Weight-space confirmation of EXP-26's 0.318.
- **The residual-over-blend selector.** `m1 ≈ 0.012` (cross-batch TRUE-gradient cosine: **dead**) vs
  `m4 j4 ≈ 0.295` (within-circuit CODEC-ERROR lag-autocorrelation: **alive** out to j=5; medians
  j1 0.086 / j2 0.200 / j3 0.115 / j4 0.295 / j5 0.169). A K-delayed operator can only transport structure
  that survives the delay; the codec error does, the cross-batch true gradient does not. **λ=1 ring
  telescoping is exact subtraction of the persistent artifact.**
- **The telescoping is only *approximate*.** `G_corr(t) = G_anc_rep(t) + [G_comp(t) − G_comp_ring(t−K)]` —
  true gradient **plus a K-tick fast-circuit drift term**. m4<1 means the cancellation is imperfect; the
  drift term shrinks as K shrinks (mechanist §3, the explicit delay_K lever). It is *endogenous* (the
  circuit's own drift), which F4/§5e argues is why it does not pump length.
- **m7 (re-frames GOAL-3 and the basis route).** Valid PG gradient: stable rank ≈ **1.8–2.05** (ambient
  1536), top-1% coordinate mass ≈ **0.60**. r=77 is **over-provisioned** for a rank-~2 object — the defect
  is **basis MISMATCH** (act-basis Q ≠ the ~2 gradient directions), not capacity. Constraint: EXP-26 Step C
  falsified update-energy/hybrid Q (rollouts uncompressed ⇒ a gradient-energy Q breaks forward recon).
- **Carrier law, made quantitative (m6).** Cross-fire autocorrelation `m6 ≈ 0.62` ⇒ AR(1) per-tick
  ρ₁ ≈ 0.909 ⇒ injected-carrier τ ≈ **10.5 ticks ≈ 2× cadence** (range 9.4–17.5). EXP-27's *compounding*
  EMA carrier had τ ≈ 19.5 ticks **and integrated**, igniting at step ~61. B2 sits at τ/cadence ≈ 2:
  marginal, on the safer side, **not memoryless**. β_anc=0 stops compounding, not the intrinsic
  persistence. Ignition is emission-judged; all stability is **censored** (50→100 for seed 0).
- **δ lifecycle (from `B2_delta_per_fire.jsonl` / `ext100_delta_per_fire.jsonl`).** coldM-fallback ticks
  1–9 (= plain PowerSGD), first valid pair at tick 10, then **refresh-at-fire / hold-between**;
  coldM_fallbacks=0 post-warmup; `delta_ratio_median` declines 1.37 → 1.007 (bounded, no monotone climb).

### Latest control evidence (incorporate — it sharpens several rankings)

| control (W&B) | what it isolates | val@50 | reading |
|---|---|---|---|
| **C2 plain_replay_substrate** (`k6nmcuyd`) — DONE | PowerSGD+anchor-Q-updates, **correction_mode=none** (no merging) on the *byte-identical* B2 substrate | **0.6300** | the true **single-knob floor** for B2 (resolves critic T3); Q-update IS live here (`anchor_q_updates=20`, `anchor_replay_fires=20`) yet no merge ⇒ 0.6300 |
| **B1 blend** (operator paper run) | same valid anchor signal, **magnitude-matched blend** η=0.3 vs residual λ=1 | **0.7422** | **blend is NOT inert** once magnitude-matched — *overturns the GATE-B1-CLOSED "blend dead at any dose" prior*. Blend converts to 0.7422 < residual 0.7528 < dense 0.7536 |
| **B2 residual** (`b59ncque`) | the best solution | **0.7528** | +0.1228 over C2; +0.0106 over B1; the δ-correction is the dominant lever, the substrate is **not** (C2 proves it) |
| dense-old (`5e2jpho9`) | method OFF | 0.7536 | the parity bar; 50 steps only; **never re-run** (operator directive) |
| **C3 frozen-Q** | PowerSGD+anchor with **Q frozen after warm-up**, no power-iteration update | **PENDING** | **prices the power-iteration Q-update**: C2 − C3. If C3 ≪ 0.6300 the Q-update is load-bearing ⇒ Q-update cadence/rank become first-class knobs (candidate g) |

**Two corrections to the strategist's stance the latest controls force:**
1. **"Blend CLOSED at any dose" is too strong.** GATE-B1 measured the *geometry* (m1≈0.012) and predicted
   inert; the magnitude-matched B1 paper run converts to 0.7422. The correct statement is **"blend is a
   weaker converter than the residual (0.7422 vs 0.7528) and is geometrically dominated by it (F1: you can
   only *subtract* 92%-energy orthogonal contamination, not *offset* it) — so blend is not a candidate to
   *advance*, but it is not literally inert."** This does not resurrect blend as a forward check; it
   removes blend from the "falsified primitive" exclusion and reclassifies it as "dominated."
2. **C2=0.6300 is the clean single-knob attribution** (T3 resolved favourably): the δ-correction, not the
   replay machinery, drives the +0.12. Every candidate below inherits this de-confounded baseline.

---

## 1. Candidate evaluations (each: hypothesis · backing · knob · cost · decision value · exclusion check)

I evaluate all six requested candidates (a–f) plus two the controls open (g: Q-update cadence/rank;
h: honest-bytes accounting — required for GOAL-3), then rank.

### (a) Shorter delay_K (K < 5) — **tighten the codec-error cancellation**

- **Hypothesis (falsifiable).** Reducing K from 5 → 3 (and a 2nd point at K=2) **reduces the fast-circuit
  drift term** `‖G_comp(t) − G_comp_ring(t−K)‖`, moving `G_corr` closer to pure `G_anc_rep`, which raises
  val@50 toward/above dense **and/or** reduces late-run decay (the 0.7536→0.7400 ext100 slide), with **no
  increase in emission**. Falsified if val@50 does not improve within noise (SE 0.0119) AND decay is
  unchanged — i.e. if drift was already negligible at K=5.
- **Theoretical backing (cite).** Telescoping is exact only if `G_comp(t) = G_comp_ring(t−K)`; it is
  *approximate* by the residual term `[G_comp(t) − G_comp_ring(t−K)]` (mechanist §3). m4 quantifies the
  imperfection: lag-autocorrelation falls from j1≈0.086 region but the *level* is 0.1–0.3 across j — the
  circuit is only partially K-stationary, so the drift term is real and **monotone-ish increasing in K**.
  Smaller K ⇒ smaller drift ⇒ cleaner true-gradient injection. **Bounded below by m4**: at K=1 the ring
  pair is the immediately-preceding tick (cosine ~j1), so there is still nonzero drift; K cannot reach
  zero (that is the current-step EF variant, a different object — see (a-note)).
- **Specific knob/value.** `anchor.delay_K` 5 → **3**, then **2**. Hold cadence=5 (decouple from
  candidate d). One 50-step cell per value; bundle as a 2-point mini-sweep (this is **not** a sweep on an
  inert primitive — delayed_ef converts, so a K-sensitivity probe is the *legitimate* analogue of the
  EXP-23 lesson's exception). **Carrier check is mandatory:** shorter K refreshes the residual against a
  *fresher* ring entry — verify m6 does not rise (fresher pairs could be *more* autocorrelated, pushing
  τ/cadence up). Watch P1/P2 + m6<0.85 trip-wire.
- **Cost.** ~5 GPU-hr per K value → **~10 GPU-hr** for {3, 2}. (Each ≈ the 50-step B2 cost.)
- **Decision value.** **HIGH.** This is the single tweak the mechanist explicitly flagged as the lever on
  the *winning* operator, and it attacks both the parity-margin question (GOAL-2) and the late-decay
  question (GOAL-1/stability) with one knob. Cheap, on-mechanism, directly testable.
- **Exclusion check.** Not blocked. delayed_ef is **not inert** (the EXP-23 "no sweeps on inert
  primitives" rule does not apply — explicitly carved out in the task). Does not touch dense, signed_ema,
  hybrid-Q, or clean_cadence. Only live risk: K↓ could raise the carrier (m6) — gated by the trip-wire,
  and falsifiable in-run.
- **(a-note) K=1 vs current-step EF.** K=1 is the *shortest delayed* residual but is NOT the same as #28's
  current-step codec-internal EF (`e_{t+1} = u_t − C(u_t)`), which needs **no anchor backward** and is
  truly zero-byte. K=1 still pays the anchor replay backward. Treat the current-step EF as a *separate*
  lower-byte route (it belongs to the #28 program, gated on Q-rotation per (g)), not as the K→0 limit of (a).

### (b) Residual-COMPRESSION: transmit δ in ≪77 columns — **the GOAL-3 upside**

- **Hypothesis (falsifiable).** Because the *residual* δ inherits the valid gradient's concentration
  (m7: stable rank ≈ 2, top-1% mass ≈ 0.60), δ can be sketched/transmitted in **r_δ ≈ 8–16 columns**
  (or a top-k coordinate-sparse form) and re-injected, preserving B2-level val@50 (within SE 0.0119)
  **at a strictly lower inter-stage byte cost than B2's anchor channel.** Falsified if val drops below the
  0.7210 floor at r_δ=16, OR if the byte saving is illusory once the anchor M-allreduce is counted (it
  must beat the honest B2 number from (h), not the bare 0.0505).
- **Theoretical backing (cite).** m7 says the object being transmitted is rank-~2 — r=77 is
  ~38× over-provisioned for it. F1 says δ ≈ −G_comp + (small valid correction); the *correction* is what
  carries the conversion and it is exactly the concentrated, low-rank piece. So a low-rank/sparse δ codec
  should lose almost nothing. This is the **only candidate that can make GOAL-3 strictly *better* than the
  current best**, which the synthesizer flagged as the live upside (PATH_FORWARD §2 "compress the low-rank
  RESIDUAL"). Mechanist §4 closes with this as "a GOAL-3 lever that F3+F1 jointly open and nothing has
  falsified."
- **Specific knob/value.** A residual-codec rank `r_δ` ∈ {16, 8}, or a top-k% coordinate mask on δ
  (k ≈ 1–2%, matching the top-1% mass). **Requires a small code change** on the correction path (sketch δ
  before injection) on an `exp/<N>` branch — it is *not* config-only. Hold all else at B2. Keep the
  forward/recon act-basis Q **untouched** (this only compresses the *correction* δ, never the forward
  pass) — which keeps it clear of the EXP-26 Step-C corner by construction.
- **Cost.** ~1–2 GPU-hr code+CPU-test (the residual-codec is a small operator + a scale-consistency unit
  test mirroring #25's mean-vs-sum invariant) + ~5 GPU-hr per r_δ value → **~12 GPU-hr** for {16, 8}.
  Gate the run on a **zero-GPU pre-check**: recompute the per-target SVD energy of the *stored* δ from the
  Step-A sidecar to confirm a rank-16 truncation of δ retains ≥~95% of its energy *before* spending a cell.
- **Decision value.** **HIGH (ceiling).** It is the one route that converts "parity at 19.8× fast-path"
  into "parity at strictly-better honest savings" — i.e. it moves GOAL-3, the half of "done" that B2 has
  *not* banked. Pairs naturally with (h) (you need the honest denominator to claim the win).
- **Exclusion check.** Not blocked. Avoids EXP-26 Step C **by construction** (forward Q unchanged; only the
  correction δ is compressed). Does not touch dense/signed_ema/clean_cadence. Carrier: δ stays endogenous
  and K-delayed, so no new exogenous carrier; **but** re-verify m6 (a coarser δ could change its
  autocorrelation). Heavy-tailed-early-δ caveat (critic T4: ~43% of matrices have ‖δ‖/‖G‖>1.5 at the first
  fire) means an aggressive truncation could be roughest in the first few fires — keep the warmup.

### (c) λ sweep around 1.0 — **justified-but-low-priority; weak theory motivation**

- **Hypothesis (falsifiable).** A λ≠1 (e.g. 0.7 or 1.3) changes val@50 outside noise. **My prior: it does
  not help, and λ>1 is mildly dangerous.** Falsified-in-the-useful-direction only if λ<1 *reduces late
  decay* without losing parity (a damping read).
- **Theoretical backing (and the argument *against* spending much here).** The task asks whether λ≠1 has
  *any* theory motivation given exact-residual telescoping. **Answer: very little, and there is a positive
  reason to stay at 1.** At λ=1 the ring telescopes *exactly* (`G_corr ≈ G_anc_rep + drift`); this is the
  unique value that makes the correction an unbiased reconstruction of the true gradient. **λ<1** leaves a
  fraction (1−λ) of the 92%-energy codec artifact *uncancelled* (you are deliberately re-admitting the
  contamination F1 says is the whole problem) — pure downside on the conversion axis, with only a vague
  "less aggressive injection → gentler" hope. **λ>1** over-subtracts: it injects `G_anc_rep + (λ−1)·δ`,
  and since δ ≈ −G_comp, λ>1 *adds back* a scaled negative-compressed-gradient term — a persistent,
  larger-magnitude push that **raises the effective carrier** and is exactly the dose-escalation the
  carrier law warns against (EXP-27: dose-capping delays, not prevents; the symmetric risk is dose-*raising*
  ignites sooner). So the only *defensible* λ probe is **λ slightly below 1 (0.8) as a decay-damping
  test**, and even that is dominated by candidate (a), which reduces drift *without* re-admitting artifact.
- **Specific knob/value.** If run at all: a **single point λ=0.8** (decay-damping read), never λ>1, never a
  fine grid. `spectral.delayed_ef_lambda` 1.0 → 0.8.
- **Cost.** ~5 GPU-hr for the one point.
- **Decision value.** **LOW.** Telescoping math says λ=1 is special; the controls (C2/B1/B2) already place
  the conversion; a λ-grid would mostly re-measure a known optimum. The honest recommendation is **do not
  run a λ sweep**; if the operator wants one robustness point, make it λ=0.8 and read it as decay-damping,
  not optimization.
- **Exclusion check.** **Partially self-excluding.** It is *not* the EXP-23 inert-primitive exclusion
  (delayed_ef converts, so a λ check is admissible in principle — the task is right). But it **is**
  dose-chasing in spirit, and λ>1 is affirmatively carrier-risky. Net: admissible but low-value; cap at one
  λ<1 point or skip.

### (d) Cadence variation — **secondary; couples staleness, carrier, and anchor bytes**

- **Hypothesis (falsifiable).** Changing `anchor.cadence` (refresh interval for M/Q and the residual) from
  5 → 3 refreshes the residual more often (fresher δ, smaller hold-staleness) but **raises anchor-side
  bytes** (more M-allreduces + Q-broadcasts per step) and **may raise the carrier** (more frequent
  injection of an autocorrelated signal). Hypothesis: cadence=3 improves val/decay marginally at a
  measurable byte cost; cadence=8 saves bytes at a val cost. Falsified if val is flat across {3,5,8} (then
  cadence is a pure bytes/compute knob, set it as high as val tolerates for GOAL-3).
- **Theoretical backing (cite).** Cadence sets both (i) the **hold-staleness** of δ between fires — the
  `held` counter in `B2_delta_per_fire.jsonl` shows δ is held for cadence−1 ticks, so a stale δ is injected
  most ticks — and (ii) the **carrier refresh rate** that enters τ/cadence in the carrier law. m6≈0.62 was
  measured *at cadence=5*; the carrier-law denominator IS cadence, so changing cadence directly moves
  τ/cadence (shorter cadence ⇒ larger τ/cadence ⇒ *more* ignition risk, the opposite of intuition).
  Cadence also directly sets the GOAL-3 anchor-traffic term (more refreshes = more M/Q bytes), so it is
  entangled with (h).
- **Specific knob/value.** `anchor.cadence` ∈ {3, 8} (hold delay_K=5 fixed; note cadence and delay_K are
  **both in optimizer-tick units**, per the standing memory — cadence=5 = 2.5 global steps). One cell each.
- **Cost.** ~10 GPU-hr for {3, 8}.
- **Decision value.** **MEDIUM-LOW.** Mostly a GOAL-3 / staleness knob, and its val effect is likely small
  and confounded with the carrier. Lower priority than (a) (which isolates drift cleanly without moving the
  carrier-law denominator) and (b) (which moves GOAL-3 more directly). Run only *after* (a)+(h) tell us
  whether staleness/bytes are even binding.
- **Exclusion check.** Not blocked (delayed_ef converts). **Carrier caveat is load-bearing**: cadence=3
  *raises* τ/cadence and could ignite — mandatory m6<0.85 trip-wire + P1/P2 watch; do not read a clean
  short-cadence run as safe at >100 steps (same censoring caveat).

### (e) Rank r below 77 — **bytes lever on the forward codec; bounded by act-recon, not gradient rank**

- **Hypothesis (falsifiable).** Since the *gradient* is rank-~2 (m7), the forward codec rank can drop
  r=77 → 40 (and a 2nd point at 24) with **no val loss**, cutting fast-path bytes further (GOAL-3).
  **Important nuance / likely-falsified-as-stated:** m7's rank-2 is the *gradient's* rank, but r sizes the
  codec that reconstructs the **forward activations**, whose effective rank is set by activation
  covariance, NOT gradient covariance. So r↓ is bounded by the **activation reconstruction error**
  (`powersgd_reconstruction_rel_error`, currently ~0.02–0.05 at r=77), not by the gradient rank.
  Falsified if val drops or recon error blows up at r=40 — which I expect is the likely outcome, because
  EXP-20 chose r=77 to match the p=0.95 activation-mask byte budget (memory: `qwen25-1p5b-hidden-size-1536`),
  i.e. r=77 was sized for *activations*.
- **Theoretical backing (cite + the trap).** The seductive read is "m7 says rank-2, so r=77 is 38× too
  big." **The mechanist explicitly refutes this corner** (§4): "the failure is basis MISMATCH, not
  capacity" — r is large enough; the directions are wrong. Dropping r does **not** fix mismatch and **does**
  risk the forward recon the codec exists to serve. The right rank lever for GOAL-3 is candidate (b)
  (compress the *residual*, whose rank IS ~2), **not** (e) (shrink the forward codec, whose rank is set by
  activations). I include (e) to **rank it down with a reason**, because m7 makes it a tempting wrong turn.
- **Specific knob/value.** `powersgd.rank` 77 → 40. One cell. (Pre-gate cheaply: read the existing
  `powersgd_reconstruction_rel_error` per layer at r=77; if it's already near a knee, r=40 will overshoot —
  a zero-GPU check that likely kills this before a cell is spent.)
- **Cost.** ~5 GPU-hr (or **0** if the recon-error pre-gate kills it).
- **Decision value.** **LOW.** Likely a small bytes win on the fast path that is *dominated* by (b)
  (residual compression moves GOAL-3 more, and (b) is the rank lever that m7 actually licenses). Most
  likely outcome: val degrades (mismatch unchanged, recon worse). Keep as a cheap-to-pre-gate "rule it out"
  rather than a priority.
- **Exclusion check.** Not formally blocked, but it leans on the *exact mismatch-vs-capacity confusion the
  mechanist warns against*. Does not touch the falsified hybrid-Q corner (it changes r of the same act-Q,
  not the Q construction). Treat as low-priority with a zero-GPU recon-error pre-gate.

### (f) Additive off-principal sub-basis routing (R5 from PATH_FORWARD) — **highest ceiling, highest risk, attacks the root**

- **Hypothesis (falsifiable).** A small gradient sub-basis (rank ≈ 2–4) sketched from the EXP-29 replay
  gradients and routed **additively into the K-delayed, fire-refreshed, β_anc=0 correction term only** —
  leaving the act-basis forward/recon Q **unchanged** — injects the rank-~2 true direction the act-basis
  misses (F1 cos≈0 / m7 mismatch), converting B2's *near-parity into a clear surpass* (val@50 above dense
  beyond noise). Falsified if val does not exceed B2's 0.7528 beyond SE, OR if it ignites (the static
  sub-basis is a persistent direction → carrier-law-exposed).
- **Theoretical backing (cite).** This is the **only candidate that attacks the root cause** rather than
  correcting around it: F1 says the act-basis returns 92%-orthogonal codec error; m7 says the missing
  signal lives in ~2 directions outside the act-subspace. B2 already *proves those directions are
  recoverable and helpful when injected additively* (δ = A−C injects them). R5 makes that injection a
  *first-class, separately-sketched channel*. Strategist §3 establishes admissibility precisely: it is a
  **NEW corner** (not EXP-26 Step C) **iff** it (a) augments rather than replaces the forward Q (so it does
  not break activation recon — the Step-C failure mode), (b) is delivered K-delayed/fire-refreshed/β_anc=0
  (so it respects the carrier law), and (c) waits until B2's own horizon behavior is known.
- **Specific knob/value.** New code: a `q_basis_passive`-style additive correction sub-basis sketched from
  replay gradients, rank 2–4, injected only into the delayed_ef correction. `exp/<N>` branch; substantial
  code (the existing `powersgd.q_basis_passive=[]` hook suggests the plumbing for a passive/augmenting
  basis partly exists — verify before scoping). Hold forward Q = act, r=77.
- **Cost.** ~2–4 GPU-hr code+CPU-test + ~15–20 GPU-hr to run with proper horizon ⇒ **~20 GPU-hr**.
- **Decision value.** **HIGH ceiling, but gated.** It is the route to *surpass* dense (not just parity),
  which is the strongest possible GOAL-2 result. But it is the most expensive and the most carrier-exposed,
  and it should be **scoped now, run later** — specifically after (a)/(b)/(h) and after seed-replication
  establishes the B2 baseline it must beat (you cannot claim "surpass" against a single-seed 0.7528).
- **Exclusion check.** **Admissible only under the three strategist conditions.** A static gradient
  sub-basis injected every step **is** the persistent exogenous direction the carrier law convicts —
  so it is admissible **only** K-delayed/fire-refreshed/β_anc=0, and is subject to the same 100-step
  censoring caveat as B2. Must NOT replace the forward act-Q (that re-enters EXP-26 Step C). Not
  signed_ema, not blend, not clean_cadence.

### (g) Q-update cadence / rank — **opened by the C2 vs C3-frozen-Q control; gates the whole basis story**

- **Hypothesis (falsifiable).** The anchor's power-iteration Q-update is **load-bearing**: C3 (frozen Q)
  ≪ C2 (0.6300, Q-updated). If so, *how often* and *to what rank* Q is updated is a real knob, and a
  zero-GPU Q-rotation telemetry re-read of the existing B2/ext100 logs will show Q is (or is not) rotating
  in the gradient-relevant directions. Falsified-toward-irrelevance if C3 ≈ C2 (Q-update inert ⇒ Q is
  effectively frozen already ⇒ candidate (f)'s basis route is the *only* way to touch the basis, and #28
  Cell-A current-step EF is inert-by-Q-convergence).
- **Theoretical backing (cite).** The C2−C3 delta **prices the power-iteration Q-update** (task brief).
  EXP-25 measured recon-error flat ~0.024 (Q near-converged) and m7's rank-2 mismatch is the strongest
  signal yet that **Q barely rotates in the directions that matter** (strategist §2.1). This is the *gate*
  on two downstream routes: if Q is frozen, (f) is necessary (only an injected sub-basis reaches the
  missing directions) and #28 Cell-A is inert; if Q rotates, the simpler forms may suffice and (f) is
  unnecessary complexity. **Decide it before spending (f)'s ~20 GPU-hr.**
- **Specific knob/value.** **First, zero-GPU:** wait for C3 (already pending), compute C2−C3; and re-read
  per-refresh `‖Q_new − Q_old‖_F` / top principal angle from the existing logs (the metrics are present:
  `anchor_q_updates`, `powersgd_q_cond`, `powersgd_reconstruction_rel_error` per layer). **Only if Q
  rotates materially:** scope a Q-update-cadence probe (`powersgd.update_cadence` is currently 1).
- **Cost.** **~0 GPU-hr** for the C2−C3 read + telemetry re-read (C3 cost is already committed as a
  pending control). A follow-on Q-cadence cell, only if warranted, ~5 GPU-hr.
- **Decision value.** **HIGH per GPU-hr (≈ free).** It is the cheapest measurement that *gates* the two
  most expensive routes ((f) and #28 Cell-A). The synthesizer ranked the equivalent "R4 Q-rotation
  telemetry" highly for exactly this reason.
- **Exclusion check.** Not blocked. The telemetry re-read touches nothing. A Q-cadence cell changes the
  *act-Q update frequency*, not the Q *construction* — so it does not re-enter EXP-26 Step C. Note: do not
  conflate "update Q more often" with "build Q from gradients" (the falsified corner).

### (h) Honest inter-stage byte accounting — **required to claim GOAL-3 at all; mostly zero-GPU**

- **Hypothesis (falsifiable).** The anchor circuit's amortized M-allreduce + Q-broadcast traffic, added to
  the fast-path 0.0505, still yields a **materially-lower-than-dense** inter-stage ratio — reported as a
  single concrete number. Falsified if the honest ratio is not materially <1 (then B2's "savings" claim is
  wrong and GOAL-3 is unmet at this config).
- **Theoretical backing (cite).** GOAL-3 demands "a concrete number," and the bytes_ratio 0.0505 counts
  **only the fast compressed boundary** (`bytes_compressed/bytes_dense_equiv`, y-only logical PP bytes) —
  it omits the **anchor's full-H M-allreduce every cadence=5 ticks and the Q-broadcast each refresh**
  (critic T6, strategist §5). The standing program estimate is "amortized comm ~4×, not 20×" (memory:
  `clean-step-realism-confound`). B2's headline 19.8× is **provisional on this**. Critically, B2 adds **no
  new traffic** over the substrate (δ is built from quantities the anchor already transfers — critic T6),
  so the honest number is a property of the *substrate*, computable once and reused for B2, (a), (c), (d),
  (f).
- **Specific knob/value.** **Accounting pass, no training.** From existing telemetry + known sizes
  (196 matrices, r=77, H=1536, cadence=5): bytes-per-refresh for M-allreduce + Q-broadcast, amortized over
  the cadence interval, added to the fast-path bytes ⇒ one honest inter-stage ratio. Optional ~1–2 GPU-hr
  instrumented run with byte counters on the anchor collectives to validate the hand-count.
- **Cost.** **~0 GPU-hr** (≤2 to validate).
- **Decision value.** **HIGHEST per GPU-hr.** It is the one item that converts B2's "parity (point
  estimate)" into a GOAL-3-shaped "parity + measured savings" result, it prices every other candidate's
  byte cost (it is the *denominator* candidate (b) must beat), and it can never be wasted or ignite.
- **Exclusion check.** None. Pure accounting.

---

## 2. What is BLOCKED, and why (standing exclusions, re-checked against the latest evidence)

| route | status | reason (cite) |
|---|---|---|
| **small-β_anc EMA smoothing** | **BLOCKED** | m6 ≈ 0.62 was the *pre-registered safety number* and came back **unsafe**: τ ≈ 10.5 ticks ≈ 2× cadence (mechanist §5b; strategist §4). β_anc>0 *compounds* an already-persistent carrier → into the ignition regime. One *narrow* contingent opening: if a **seed-replicated** B2 holds emission-free well past 100 steps (de-censoring the endogenous-carrier hypothesis F4/§5e), a small-β_anc variant on the **δ-residual (endogenous) object only** could be *re-argued* with a fresh carrier-law budget. Not now. |
| **signed_ema (sign-replacement) on any M** | **DEAD** | EXP-25/26: structurally convicted (sign-disagreement ~50% is a coin-flip at delay 0), independent of M validity. `signed_ema_alpha=0.5` present-but-inert in B2's params is a leftover default, not live. |
| **update-energy / hybrid Q** (build/blend Q from gradient energy) | **DEAD** | EXP-26 Step C: anti-converts because rollouts are uncompressed — a gradient-energy Q breaks the forward activation recon the codec serves. **Any basis work (e, f, g) must avoid this exact corner** — which (f) does by *augmenting additively*, (g) by changing *update frequency* not *construction*, (e) by changing *rank* of the same act-Q. |
| **clean_cadence > 0** (periodic full dense step) | **EXCLUDED** | clean_cadence=0 is the locked substrate (GOAL §"Where we are": the anchor *replaces* the unrealistic periodic-dense-step); a clean step is full-H transfer and would itself be stale on a real PP link (`clean-step-realism-confound`). Do not reintroduce. |
| **dense re-run** | **FORBIDDEN** | operator directive 2026-06-10. dense-old `5e2jpho9` (50 steps) is the only bar; **no dense@100 exists** and val@75/100 are trajectory-shape reads vs B2's own curve + the plain@100 drift control, never vs a dense@100 number that does not exist. |
| **blend / convex-combination as a route to advance** | **NOT a forward candidate (but reclassified: "dominated," not "inert")** | B1 paper run = **0.7422** (NOT inert once magnitude-matched — overturns the GATE-B1 "dead at any dose" prior). F1 explains why it is *dominated*: 92%-energy orthogonal contamination can be **subtracted** (residual 0.7528) but only weakly **offset** by an orthogonal partner (blend 0.7422). So blend is not advanced, but the "blend is falsified/inert" framing should be retired in favour of "blend is the weaker, geometrically-dominated converter." |
| **λ sweep (grid) on delayed_ef** | **DISCOURAGED, not formally blocked** | delayed_ef converts, so a λ check is *admissible* (not the EXP-23 inert exclusion). But λ=1 is the unique exact-telescoping value (theory, candidate c); λ<1 re-admits the artifact, λ>1 raises the carrier. Cap at one λ=0.8 robustness point or skip. Not a grid. |

---

## 3. Ranked shortlist — decision-value per GPU-hr (what to run next on the best solution)

> Framed as "what to run next on B2," **explicitly not as issues.** Tiered: free/cheap measurements that
> *gate* the expensive cells come first, exactly the EXP-30 discipline (cheap gate before expensive build).

| rank | what to run on B2 | GOAL crit | cost (GPU-hr) | decision value | the one-line reason it ranks here |
|---|---|---|---|---|---|
| **1** | **(h) Honest inter-stage byte accounting** | **3** | ~0 (≤2 to validate) | turns "parity (point est.)" into "parity + a real savings number"; prices every cell below | the cheapest item that closes the *unmet* half of "done"; can never be wasted or ignite |
| **2** | **(g) C2−C3 + Q-rotation telemetry re-read** | gates (f), #28-A | ~0 (C3 already pending) | decides whether the basis is frozen → whether (f) is *necessary* and #28-A is *inert* | free disambiguator that gates ~20 GPU-hr of downstream work |
| **3** | **B2 seed replicate @50** (resolves critic T2) | 2 (parity), 1 (seed-generality) | ~5 | converts "parity unestablished at n=1" into a mean±range; *prerequisite* for any "surpass" claim from (f) | the binding *statistical* threat; you cannot beat a single-seed 0.7528 — establish it first |
| **4** | **(a) shorter delay_K {3, 2}** | 2 (margin), 1 (decay) | ~10 | the mechanist's named lever on the winning operator; attacks parity-margin AND late-decay with one knob | smallest tweak with the cleanest theory (drift↓), carrier-gated, on the converting primitive |
| **5** | **(b) residual-compression {r_δ 16, 8}** | **3 (upside)**, 2 | ~12 (1–2 code + zero-GPU SVD pre-gate) | the only route to *strictly better* savings at B2 val — moves GOAL-3 forward, not just defends it | m7 licenses it (residual is the rank-~2 object); pre-gate on stored-δ SVD before spending a cell |
| **6** | **(f) additive off-principal sub-basis** (scope now, run later) | 2 (**surpass**), 1 | ~20 | the only root-cause attack (F1/m7 mismatch) → the path to *exceed* dense | highest ceiling, but gated behind #3 (a baseline to beat) + #2 (is it even needed?) + horizon |
| **7** | **(d) cadence {3, 8}** | 3, 1 | ~10 | mostly a bytes/staleness knob; val effect likely small + carrier-confounded | run only after (a)+(h) show staleness/bytes are binding; cadence↓ *raises* carrier risk |
| **8** | **(c) λ=0.8 single point** (or skip) | 2 | ~5 (or 0) | one robustness/decay-damping read; theory says λ=1 is special | dominated by (a) for the decay question; do **not** grid; never λ>1 |
| **9** | **(e) rank r=40** (zero-GPU recon-error pre-gate first) | 3 | ~5 (or 0) | likely a wrong turn — m7's rank-2 is the *gradient's*, not the *activation* codec's | included to rule out the tempting mismatch-vs-capacity confusion; pre-gate likely kills it |

**Budget shape.** The free/cheap gating tier — **(h) + (g) + seed replicate = ~5 GPU-hr** — closes GOAL-3
accounting, prices the basis, and establishes the statistical baseline *before* any expensive build. Add
**(a) ~10** and you have the cleanest on-mechanism parity/decay read. **(b) ~12** is the GOAL-3 *upside*
cell. **(f) ~20** is the one ~surpass bet, deliberately last and gated. Everything fits inside ~2 budget
cycles, and the gate-it-out path (if (g) shows Q frozen and (h) shows savings hold) spends as little as
~15 GPU-hr to reach a defensible "parity + measured savings + seed-replicated" result on B2 — with (b)/(f)
as the optional push toward strictly-better savings / surpass.

**The one sentence.** Spend the next ~5 GPU-hr on **(h) honest bytes + (g) the C2−C3 Q-update price + one
B2 seed** (none of which can be wasted), then **(a) shorter delay_K** as the cheapest on-mechanism push,
keeping **(b) residual-compression** as the live GOAL-3 upside and **(f) the additive sub-basis** as the
gated, scope-now-run-later bet to surpass dense — and do not relitigate the BLOCKED routes (β_anc-EMA,
signed_ema, hybrid-Q, clean_cadence, dense re-run) or grid λ.
