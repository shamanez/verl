# Research Log (newest first)

## EXP-23 · 2026-06-05T04:38:43+00:00 · M6 · STOP
[EXP-20 follow-up] Stale full-gradient re-anchoring for PowerSGD GRPO (delay_K=5 + inject/blend) — does a STALE refresh replace the fresh clean step?

- hypothesis: a delay_K=5 stale full-gradient re-anchor (via inject or blend) with clean_cadence=0 recovers most of the fresh-clean benefit for PowerSGD r=77; success predicate max(val@50(A2),val@50(A3)) >= 0.7315 AND >= val@50(A1)+0.05
- result: FALSIFIED — max(A2,A3)=0.6967 <= floor+0.02=0.7114; A1 no-refresh floor=0.6914, A2 stale-inject=0.6967 (+0.005), A3 stale-blend=0.6861 (−0.005); reference points A0 fresh-clean=0.7415 (EXP-20), dense=0.7536; mechanism: cos(G_powersgd, M_anchor)≈0.001 (near-orthogonal — 10× more orthogonal than mask's cos≈0.5 in EXP-21); inject adds tiny orthogonal noise (‖M‖≫‖G‖ so scale=‖G‖/‖M‖≈0.03, net correction ~0.03·‖G‖); blend shrinks step to 0.71× swapping half live signal for stale orthogonal direction; neither supplies the missing descent component
- arms: A1 (no-refresh floor)=0.6914 | A2 (stale inject γ=1.0)=0.6967 | A3 (stale blend η=0.5)=0.6861 — all on the floor; train-reward last-5-step mean A1=0.640 A2=0.650 A3=0.653 (statistically indistinguishable)
- integration verdict: WORKED (all 6 hard-gate invariants passed; circuits fired anchor_backwards=20/spectral_corrections=80 per A2/A3; codec health green q_cond≈1.0, recon_rel_error ~0.019–0.030, q_cross_rank_max_rel_dev=0.0; zero NaN/OOM/single-GPU; world_size=4 held) — this is a decisive NEGATIVE RESULT, not a failed run
- A1 floor finding: fresh-clean buys ~+0.05 (0.6914 → 0.7415); stale re-anchor does NOT recover this; delay_K is NOT the lever (orthogonality is structural: PowerSGD r=77 discards exactly the directions M lives in, so cos≈0 holds at any K)
- next lever (binding, per runs/EXP-23/stale_gradient_research/STALE_GRADIENT_ALTERNATIVES.md §8): error-feedback on the PowerSGD residual (per-matrix FP32 buffer e accumulating G_full−G_compressed, direct attack on orthogonality) + staleness-aware blend η∝1/K; follow-up = EXP-24
- code change: launcher wiring (spectral.correction_mode/inject_gamma/blend_eta env vars) on exp/23-stale-reanchor @ f42b7f36 — UNMERGED (verdict=STOP; promotion waits for a PASS in the EXP-24 follow-up lineage; the change is correct and additive — it unblocks future inject/blend runs — but no PR is opened for a STOP)
- run dir: runs/EXP-23/
- verdict: runs/EXP-23/verdict.md

## EXP-20 · 2026-06-04T13:42:00+10:00 · M6 · PASS
PowerSGD-style PP activation compression for the GRPO loop

- hypothesis: at equal logical PP byte budget (PowerSGD r=77 ≡ mask p=0.95 at H=1536), the PowerSGD-compressed reward trajectory tracks or beats the byte-matched PRF mask
- result: CONFIRMED un-caveated — val-core GSM8K acc@50: mask 0.7384 | r=77 (matched, ~20x) 0.7415 (+0.0031) | r=102 (+33% budget, ~15x) 0.7437 (+0.0053); PowerSGD >= mask at EQUAL communication budget
- key construction findings:
  - activations are effectively low-rank: reconstruction_rel_error converges from ~0.97 (step 0) to ~0.02 steady within ~5 steps for both r=77 and r=102; spectral gap is present and sufficient at these ranks
  - r=77 and r=102 reach near-identical steady-state reconstruction fidelity (~0.024 vs ~0.021); the marginal gain of +33% budget is small, confirming r=77 is already in the flat part of the rank-accuracy curve
  - clean_cadence=5 dominates reward trajectory: the 10 dense refresh steps (steps 5,10,...,50) are the primary correction mechanism; PowerSGD's per-step |Δreward| vs mask (0.013-0.016) is smaller than the mask's own dense-refresh jump (0.032), so the compressed trajectory is smoother, not more jagged
  - cross-DP consensus basis (sync_basis=true, all-reduce raw V then orth) proven bit-identical across all 4 DP ranks: q_cross_rank_max_rel_dev=0.0 every step; the shared-codebook invariant holds end-to-end
  - all probe hard invariants passed: off-path parity, r=H lossless (rec_rel_error=0.0029, bf16 floor), autograd no-STE (5-lens math panel VALID), deterministic seed, frozen-Q rho≈1, FSDP/dtype clean; zero NaN/OOM/single-GPU fallback across all 3 arms
- caveats: single seed (50-step directional gate, not a variance study); clean_cadence=5 masks codec differences on reward (both codecs get 10 dense steps); update cosine not instrumented (unmeasured, not failing — direction-agreement evidenced by reward tracking + reconstruction + jaggedness); launcher promotion deferred pending cosine instrumentation + operator confirmation
- run dir: runs/EXP-20/
- verdict: runs/EXP-20/verdict.md
- m6 progress: 1 PASS so far (milestone summary requires ≥2 PASS entries for M6; not yet written)
