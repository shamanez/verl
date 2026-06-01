# Research Log (newest first)

## EXP-17 · 2026-06-01T06:25:39Z · M3 · PASS
M3 — Long-horizon lossy-compression GRPO: periodic full-gradient refresh (clean_cadence K=20) over 2 epochs — the CORE single-run diagnostic
- hypothesis: masked GRPO (p=0.9 rescale, clean_cadence=20, anchor+spectral OFF) over 2 epochs keeps learning and the periodic clean step keeps fully repairing the weights (no irreparable drift)
- result: PASS — final val 0.7354 ≈ dense parity 0.741 (−0.82%); all 5 clean steps fired at 20/40/60/80/100; clean-step grad_norm trends DOWN (0.426→0.360, slope −0.00078/step); train-inference gap is a clean-resettable sawtooth (not a ratchet); per-clean-step repair FULL and CONSTANT; learning speed cost: steps-to-reward>=0.5 = 44 (vs dense 6) but final quality exceeds K=5 and K=4 reference trajectories; 85.5% boundary-activation comm savings; exp branch exp/17-masked-clean-every20 is a pure audit anchor (no method patch, promote_launcher_as: none)
- run dir: runs/EXP-17/
- verdict: runs/EXP-17/verdict.md
