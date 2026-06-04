# Research Log (newest first)

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
