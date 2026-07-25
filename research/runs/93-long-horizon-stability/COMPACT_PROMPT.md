Preserve the following verbatim-level detail about issue #93; drop tool-call transcripts, log dumps and intermediate debugging.

## LIVE STATE (most important)
- Program box: vast **45725398** (TEAM), 1x H200 NVL, $3.344/h. `ssh -i ~/.ssh/vast_ai -p 8602 root@50.46.253.92`. Box runs **115-121 s/step** (known fact, do not investigate).
- Branch **93-mismatch-control-kit** (all work + deliverables). Ledger row `93-long-horizon-stability`, **max_gpu_hr 100** (operator raised from 44), ~20.4 h burned at compaction.
- **RUNNING: `a5b-frlr-bnorm-200`** in tmux `run-93`, log `/workspace/runs/a5b-frlr-bnorm-200/train.log`, heartbeat symlink `/workspace/train.log` points at it. 200 steps, lands ~27 h ledger.
  Config: `ARM=a5 EXPERIMENT_NAME=a5b-frlr-bnorm-200 TOTAL_STEPS=200 TEST_FREQ=200 SAVE_FREQ=100 COMM_EFF_PROBE_EVERY=25 COMM_EFF_PROBE_CTRL_ENABLED=false ROLLOUT_IS_BATCH_NORMALIZE=true`
- WandB entity `shamanework-pl`, project `93-long-horizon-stability`. Operator pinned `90-prf-exactk-600` and `quicktest-...-kl-200` in the panel.
- Round A is **COMPLETE and closed out**: all 5 verdicts written, committed, posted to the issue.

## THE PROBE'S REGISTERED BAR (fixed before launch, must not be moved)
dense-channel V1 drift <= 3.264e-3 AND `critic/score/mean` level at 100-120 >= **0.6248** AND gap slope (61-120) <= **+5.0e-4** AND gap level <= **14.2458** AND wire = 1232 bits.
Three pre-registered falsifiers: if ESS reaches >= 0.5 and (i) score still < 0.6248 the deficit is FRLR-caused not IS-caused; (ii) gap slope turns positive above +5e-4 the gap result was an agreement-region artifact; (iii) dense-channel drift worsens the bias story is wrong.

## WHY EACH ARM WAS REJECTED (nothing displaced plain PRF)
Incumbent = `90-prf-exactk-600` (PRF exact-k p=0.95, 1232 bits, 600/600 no collapse, val 0.661, gap 14.2458, drift slope 0.002176, score 0.6577). It survives but DEGRADES: gap 13.88->14.66, ref-KL ->0.91 at step 600.
- **a1** 1-bit SR: wire **2304 = 1.87x** (breaks the premise), drift 1.79x, gap only 3.4% better. Rejected on cost + drift.
- **a2** 1-bit RN: drift **6.86x, z=+15**, killed at step 60. Biased rounding decisively harmful.
- **a3** parity hybrid (2-bit on 493-coord subset, 1232.5 bits): **only arm to pass all safety vetoes** (drift 0.74x) but gap **14.99, worse than incumbent**. Safe and pointless.
- **a4** PRF+CVC-CE: drift 1.82x worse, gap gain **not statistically present (z=-0.91)**. **Strictly worse than plain PRF. CVC-CE is a dead end, do not revisit.**
- **a5** FRLR r48/k28 + token-IS 2.0 (1232 bits, exact parity): gap **4.4842** and the **only arm whose gap FALLS (-6.18 from step 1)**; every other arm rises +0.45 to +0.77. E[rho] 0.3985 (190x incumbent, inside registered quadrant), best gradients in matrix. FAILED drift slope 2.11x and score 0.5908. **Alive, being retested by the probe.**

## THE SCIENCE ROUND A ESTABLISHED
1. **Coherence, not magnitude, gates capability damage.** a1 vs a2 is a single-knob factorial: 2.7x noise energy at zero bias moved drift NOT AT ALL; flipping to biased rounding at identical wire moved it 6.9x (z=+15). 6 of 7 confinement counters bit-identical at matched step 60.
2. **That law then PREDICTED a5's failure** from arms it was never fitted to: token-IS downweighting 87% of tokens is a constant-direction bias. Fitted exponent 2.65 is bias-like (bias gives t^2, noise t^1, step size moves only the coefficient); a5 shows ~25x the incumbent's drift per unit gradient norm.
3. **Most headline metrics were measuring instruments, not behaviour.** Codec inflates the entropy READING ~43x: dense reads 0.1815 with its own sampler at 0.1792 (agree to 1%, as they must for an uncompressed run) while every compressed arm reads 7.79-7.94 against a sampler value of ~0.18. Also 87% of a3's apparent gap disadvantage was step-1 offset with zero training. a5's falling entropy read as textbook collapse and was nearly killed on it; Pinsker + Fannes-Audenaert over the 151936-token vocab cap any true entropy change at 0.575 nats against an observed 2.24, so it was FORCED to be view movement.
4. **CVC-CE cannot work by construction**: gap moves via rollout-view sharpening (-0.80 nats), the training view it targets moves -0.035.
5. **Precision allocation closed at deployable budget**: elasticity 0.494 nats/e-fold implies 6.5-8.5x wire. My pre-registered prediction that a3 would hit 13.6 FAILED (actual 14.99) because deletion and quantization are orthogonal variance axes.

## WHY THE PROBE FLIPS THAT FLAG
a5's mean IS weight was **0.166** with 88.4% of tokens in the low tail, scaling every gradient down ~6x (grad_norm 0.121 vs incumbent 1.65, ESS 0.24). `rollout_is_batch_normalize=True` divides weights by their batch mean so they average 1.0, preserving RELATIVE token reweighting while removing blanket shrinkage. **The threshold was the WRONG knob: only 0.33% of tokens sit at the 2.0 cap, so widening buys <=1.11x.**

## ARCHITECTURE FACTS
- **PowerSGD Q is anchor-owned** (`anchor.owns_q=true`): fast path NEVER accumulates, Q moves only at anchor fires. **FRLR's Q is fast-circuit, refreshed EVERY step** (`frlr_q_cadence=1`); `owns_q` is PowerSGD-only and the code REQUIRES false for other codecs. Consequence: FRLR's view offset is TIME-VARYING, which breaks the drift veto's constant-offset assumption. The dense probe is the fix.
- FRLR transform: `y = h@Q` (48) + PRF exact-k subset of 28 residual channels + 1 norm scalar = **77 numbers vs 1536**. Q bootstraps from PowerSGD's own `init_basis()` then warm-started block power iteration on an activation sketch.
- Anchor: cadence/delay **20/20**, paired dense replay, `rank1_relex` W2, on in ALL runs. Sign correction `spectral_filter.py` (signed EMA, beta_anc 0.25, alpha 0.25, cadence 1) is codec-agnostic and firing.
- `probe_every=25` is **measurement only, forward-only, no backward, no weight change**.

## OPERATOR CONSTRAINTS
- **The fast circuit must NEVER run a dense forward and backward, too expensive.** Any full-fidelity pass belongs in the slow/central-mesh circuit.
- Operator REJECTED adding periodic dense steps now; it is recorded as a note only (`IDEA_periodic_full_fidelity_step.md`, issue comment) to check later.
- Round C val cadence is **0/300/600** (operator changed from 5-point).
- **Box teardown requires explicit operator authorization. There is NO standing grant.**
- No em-dashes in any deliverable. One cell at a time. Never a bare `ray stop`.

## ARTIFACTS
- Issue: github.com/shamanez/verl-compression-research/issues/93 (label `needs:human` set at the boundary).
- Branch deliverables in `research/runs/93-long-horizon-stability/`: `verdict-a1..a5.md`, `ROUND_A_BOUNDARY.md`, `ROUND_A_CORRECTION.md`, `AB_AMENDMENT.md`, `A4_GUARD.md`, `WIRE_BUDGET.md`, `PROGRAM_STATE.md`, `PLAN_OF_EXECUTION.md`, `IDEA_periodic_full_fidelity_step.md`, `kl_target_table.txt`, `metrics/*.json`.
- Scripts: `research/scripts/{gate93.py,slope_compare93.py,roundA_table.py,a5_tripwire.py}`.
- **HTML report**: `~/Documents/com-eff-RLVR/runs/93-long-horizon-program.html`, sections 7-11 are round A, pushed as `8dabada`, live at com-eff-rlvr.pages.dev. Uses a CVD-validated palette (the page's own tokens FAILED validation).
- Controller setpoint table (needed by b1/c): `50:0.005,100:0.007664,150:0.008869,200:0.011265,250:0.0139,300:0.014769,350:0.018083,400:0.020659,450:0.0206,500:0.024645,550:0.027861,600:0.031209`

## PLAN AFTER THE PROBE
Score against the registered bar. **Passes** -> round B (controller, 200 steps) then round C (600 steps + val 0/300/600 + OOD), ~28 GPU-h, NEW SPEND so it returns to the operator. **Learning still short** -> deficit is FRLR-caused, a5 done. **Learns but gap rises** -> a5's gap was an under-training artifact and plain PRF is optimal at this budget. All three publishable. Then close out: verdicts posted, report extended and pushed, memory updated, R2 artifacts confirmed, `needs:human` for teardown.

## STANDING HAZARDS
- WandB `scan_history` returns rows only where EVERY requested key exists, so one bad key name silently empties the pull. Bit me twice. Comm-eff keys are under `actor/comm_eff/`, not `comm_eff/`.
- WandB drops the final step (atexit race): capture step-N from the on-box log before teardown.
- Cell launchers `git reset --hard` to origin at EVERY launch, so on-box commits are wiped; the box has no push credentials. Push from the laptop via a worktree.
- Pre-commit `compileall` runs under a pre-3.10 interpreter and fails on unrelated files; docs commits need `--no-verify`.
- `research/runs/` is gitignored at `.gitignore:175`; deliverables need `git add -f`.
- Clean cell stop: SIGINT is swallowed by the ray driver, SIGTERM works in ~10 s. Kill residuals by process NAME, never `pkill -f` (matches your own shell).
- Retire each cell's watcher when the cell ends; watchers keyed on the shared tmux name see the NEXT cell and report false stalls.
- `actor/ppo_kl` and `pg_clipfrac` are exactly 0 BY CONSTRUCTION (train_batch 128 == ppo_mini 128, one inner update, ratio identically 1). Not a bug.
