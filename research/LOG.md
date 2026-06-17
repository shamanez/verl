# Research Log (newest first)

> **De-bloat note (2026-06-15):** older bulky run dirs were removed
> (~64 GB reclaimed); their record is folded into `runs/SUMMARY.md` + memory + git history + W&B. The
> `run dir:` / `verdict:` pointers in those entries below are **historical** (dirs no longer on disk).
> The SOTA = **B2**, whose ground truth was migrated to **`runs/EXP-31/B2_baseline/`**
> (`resolved_params_B2.txt` + `launch_B2.sh` + `verdict.md` + `metrics/`). Only the active
> **EXP-31** dir is retained.

> **Evidence-boundary note (2026-06-17):** EXP-29 paired replay is the validity
> boundary for anchor-gradient claims. EXP-20 remains clean-step history only
> (not a current floor). Current valid-M merger, floor, and mechanism claims use
> post-EXP-29 paired-replay evidence only.

## EXP-33 · 2026-06-17T04:54:20+10:00 · M6 · PASS
β_anc (anchor-gradient EMA) sweep on the SOTA B2 delayed_ef substrate — {0, 0.25, 0.5, 0.75, 1.0}, everything else locked

- hypothesis: On the corrected SOTA B2 substrate (delayed_ef λ=1, valid-M PowerSGD r=77 anchor circuit, replay_paired_batch=true), β_anc=0 (no EMA averaging, latest fire's M_rep) is weakly optimal — no β>0 beats C0 beyond ±0.024 rollout noise, and β=1 collapses to the plain-PowerSGD no-merger floor via cold-M freeze.
- result: PASS (measurement). Freshness-best hypothesis SUPPORTED. β→accuracy curve is a FLAT free-averaging region for β∈[0,0.75]: C0 β=0.00 → 0.73844 (control, B2 band); C1 β=0.25 → 0.73995 (+0.0015, TIE); C2 β=0.50 → 0.75284 (+0.0144, TIE — nominal peak, within ±0.024 noise); C3 β=0.75 → 0.72176 (−0.0167, TIE); C4 β=1.00 → cold-M collapse confirmed (merger_coldM_fallbacks=196/196 permanent → plain PowerSGD, val@25=0.44807 val@30=0.56406, climbing toward floor). Max gap C2 +0.0144 < falsification bar +0.024. No β>0 cell strictly beats β=0 beyond noise. bytes_ratio 0.0504–0.0506 identical across all 5 cells (β is comm-neutral). promote_launcher_as: none — B2 (=C0, β=0) stays the reference. Box i_41194490 torn down (0 live instances verified).
- run dir: runs/EXP-33/
- verdict: runs/EXP-33/verdict.md

## EXP-31 · 2026-06-16T04:20:00+10:00 · M6 · STOP (tournament phase)
4-lever anchor-signal-usage tournament — L4 perturbation / L2 δ-momentum / L3 adaptive dose / L1 control-variate

- hypothesis: At least one anchor-usage lever (L4 isotropic perturbation, L2 δ-momentum accumulation, L3 adaptive dose, L1 control-variate de-noising) lifts greedy val@50 above B2_live (0.7354) toward ≥0.78, on the locked B2 substrate (delayed_ef λ=1, PowerSGD r=77, anchor cadence/delay_K=5, replay, seed 0, 4×H200 i_41048644).
- result: STOP. All four levers are NULL — none clear B2_live beyond ±0.024 noise. B2_live (Cell A, reproduced): val@25=0.7202 / val@50=0.7354; dense-this-box=0.7506 (band 0.75–0.78). L4 σ=0.01: 0.7157 (parity). L2 μ=0.9: 0.5701 (REGRESS −0.15, over-smoothed); μ=0.5: 0.7089 (parity). L3 ratio κ=1.0: 0.7119 / cos κ=1.0: 0.7134 (both parity). L1: SKIPPED — gate F1 fails (cov(G_comp,M)≈0 ⇒ control variate has nothing to cancel; no L2/L3 surpass signal to gate on). 4 process criteria PASS (off-path parity bitwise, bytes_ratio∈[0.0504,0.0506]=B2, no ignition/OOM/divergence). Code verified GO (adversarial 8-agent workflow). Mechanistic takeaway: B2 caps at parity because δ reconstructs the dense gradient on stale data — you cannot beat dense by reweighting (L3), accumulating (L2), perturbing (L4), or de-noising (L1) a stale estimate of dense. To surpass, the anchor must provide signal dense genuinely lacks; no admissible lever does.
- WandB: B2_live fy920fty · L2_mom09 ybemd5ux · L2_mom05 knlzxh2x · L3_ratio_k10 kzohyuod · L3_cos_k10 wmpmmdj1 (project shamanework-pl/verl_compression_research)
- run dir: runs/EXP-31/
- verdict: runs/EXP-31/verdict.md

## EXP-31 · 2026-06-14T00:00:00+10:00 · M6 · PARITY (operator-accepted)
Surpass the dense baseline with a stale-anchor-gradient merger (PowerSGD-locked comm-eff GRPO)

- hypothesis: A rank-2 off-principal direction harvested from the stale anchor gradient and routed additively into the K-delayed correction term (forward Q untouched) lifts greedy val@50 strictly above the dense control.
- result: PARITY, not greedy-surpass. The most important finding is a dense-reference reframe: the dense bar on THIS config is 0.7506, not 0.7839 (the 0.7839 was a different box). Best comm-eff arm (B2/Cell A, delayed_ef λ=1, r=77 act) = 0.7400 vs dense-here 0.7506 — gap 0.011, within ±0.024 eval noise = statistical PARITY. Comm-efficient GRPO already matches dense at ~5% gradient-comm cost. The rank-2 tail sub-basis (88-90% off-principal energy captured) accelerates early learning (+0.036 at step 25 for r2 arm) but does NOT convert to a greedy surpass: constant full weight over-amplifies near convergence and regresses (r2: 0.7293@25 → 0.6983@50); γ-decay fixes the regression (0.7210@50) but also tempers the early gain, ending at parity-below-B2. hold25-decay25 val@50 lost to a Vast-side box stop (~23:13 UTC, box not destroyed). Seed bands (dense×3, B2×3) deferred — box stopped and would not restart; single-draw parity claim is qualitatively robust (all comm-eff draws 0.72–0.74 vs dense 0.75). Branch exp/31-subbasis-merger available (pushed, unmerged — no surpass justified promotion).
- run dir: runs/EXP-31/
- verdict: runs/EXP-31/verdict.md
- reframe (2026-06-15, operator): the sub-basis (amplification) bet is FROZEN as a parity-only null; B2 stands as the frozen comm-eff SOTA. Issue #31 re-scoped to an OPEN-ENDED **4-lever anchor-gradient-usage tournament** (L4 perturbation [already built] / L2 δ-momentum / L3 adaptive-dose / L1 control-variate), target val@50 → 0.80, dose λ/β_anc now tunable, length-ignition trip-wires back on. Async-realism constraint: anchor = single SLOW node serving a fast SWARM ⇒ always lagging, never leads (delay-compensation ruled out). Plan `.claude/plans/31.md` rewritten; awaiting `status:approved`.

## EXP-30 · 2026-06-13T01:40:00+10:00 · M6 · PASS
EXP-30: generator-consistent M (EXP-29 paired replay, β_anc=0) — geometry-gated re-test of linear merging + K-delayed codec residual

- hypothesis: On the locked substrate with replay_paired_batch=true, the valid anchor gradient G_anc_rep is materially aligned with the retained fast gradient enough for either a short-memory blend (B1, eta=0.3) or the K-delayed exact codec residual (B2, delta = G_anc_rep - G_comp_ring at identical (batch,theta), lambda=1) to train 50 steps with ZERO post-warmup emission and pass the validation gate.
- result: PASS via B2 (the residual route); the blend route was retired for free by the geometry gate. Step A (20 steps, 7 post-warmup fires, 196 targets, all probe gates green incl. canary 16/16 bitwise + anchor_grad_corrected=0): GATE-B1 CLOSED — valid M is near-orthogonal to the live compressed gradient (med m1 0.0121; m3 0.59-0.76 high) yet NOT the decorrelation null (m4 lag-autocorr j4=0.295, j5=0.169 nonzero ⇒ K-delayed signals not uniformly dead). GATE-B2 OPEN (med ||delta||/||G_comp_ring|| 1.0528 in [0.1,1.5]; loss-mismatch <= 0.0103). B2 (delayed_ef lambda=1, beta_anc=0): best val@50 = 0.7528; NEAR-PARITY not established (reaches old-code dense 0.7536 -0.0008 but -0.031 below the same-code same-config dense rerun 0.7839/73ntu76u; dense val@50 is a band ~0.75-0.78, rollout nondeterminism ±0.024; seed replicates binding — DENSE BASELINE CORRECTED 2026-06-13) — with ZERO post-warmup emission (no len/max>4000 in [10,50]; no P1; delta_ratio bounded 1.37->1.03 declining). FIRST correction-carrying valid-M cell in the program to convert without igniting. Discovery (F1): cos(delta, G_comp_ring) ~ -0.95 at ratio ~1.05 ⇒ at identical (batch,theta) the TRUE gradient is near-orthogonal to the compressed one with ||G_true|| ~ 0.33||G_comp|| — codec error DOMINATES the fast gradient. m6 ~ 0.62 cross-fire persistence (carrier-law risk stated; 50-step stability = CENSORED). m7: stable rank ~1.9/1536, top-1% mass ~0.60 — RLVR replay gradient is rank-~2; capacity is not the codec's problem, act-basis mismatch is.
- key metrics: B2 val 0.0864@0/0.7036@25/0.7528@50; emission 0; bytes_ratio A=0.0504 B2=0.05052; Step-A mean step 85.01s (gate 86.3); max_mem A=27.92 B2=28.66 (<30.77); spend ~9.2/24 GPU-hr; CPU suite 230 green
- deliverable: m1-m7 priors posted to #28 (comment 4693870612) — m5 = direct weight-space measurement of the codec error #28's EF feeds on; B2 = production proof the K-delayed telescoping EF mechanism works
- CONTROL DECOMPOSITION (operator-directed, current hyperparameters, val@50): dense same-config rerun (73ntu76u) 0.7839 [band 0.75-0.78, dense-old 0.7536] · B2 δ-residual 0.7528 (≈96% dense, near-parity NOT established) · B1 blend 0.7422 · C2 plain PowerSGD+Q-updated no-merge (k6nmcuyd) 0.6300 · C3 PowerSGD Q-FROZEN no-merge (djy4tog1, killed@25 no-improvement) 0.0925. ⇒ power-iteration Q-update is the DOMINANT lever (+~0.5: C2 0.6300 vs C3 0.0925 frozen-random-Q = no learning); δ-residual merger +0.123 (B2−C2). All 6 cells one-knob (token cap 18432 non-binding under micro_batch=1/static-batch). Hand-off issue #31 = surpass-dense program from B2 (parity-first). Box 40765004 TORN DOWN (strict, on done). Intermediate team docs consolidated into verdict.md (deleted; git history).
- follow-up DONE: operator-authorized 100-step B2 extension (exp30_B2_ext100, W&B b59ncque) DE-CENSORED stability for seed 0 — emission-free through 100 (two isolated 1/1024 cap-pins @94/99, benign, no P1/P2/P3); vals 0.7278@25 / 0.7536@50 (= dense ceiling) / 0.7475@75 / 0.7400@100 (mild late decay; plain@100 = the right comparator). B1 blend-on-valid-M paper run (operator-directed) on a fresh box completes the operator-ablation row. Incident: teardown hook reaped i_40697545 post-ext100 (heartbeat-path row-naming bug; zero science lost)
- code: PR shamanez/verl#17 MERGED to vast-ai-workload (ca5f4b002); branch exp/30-valid-m-geometry deleted (remote+local+worktree)
- run dir: removed (de-bloated 2026-06-15) — B2 ground truth migrated to `runs/EXP-31/B2_baseline/` (resolved_params_B2.txt + launch_B2.sh + verdict.md + metrics); box i_40697545 torn down
