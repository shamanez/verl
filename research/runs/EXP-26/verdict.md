# Verdict EXP-26 — 2026-06-11T08:05:00+10:00 (TERMINAL whole-issue verdict)

> Supersedes the stage-A gate at `runs/EXP-26/stepA_verdict.md`. This is the
> terminal verdict for the whole issue across Steps A / C / B / E.

## Result
VERDICT: REVISE

best_promoted_arm: exp26_B_ef_r2 (ef_powersgd, W&B tilwe80t) — val@50 0.7210
parity_target: 0.7414  (= floor 0.6914 + 0.05; A0 fresh-clean 0.7415)
miss_by: 0.0204  (2.04 pts below parity; 0.0096 ABOVE the STOP/falsification line 0.7114)
revise_cycle: 1 of iterations=3 on this lineage (EXP-25 was terminal STOP; no prior #26 REVISE)

Why REVISE and not STOP: none of the falsification triggers fire. (a) best arm
val@50 0.7210 > floor+0.02 = 0.7114; (b) the direction-preservation discriminator
IMPROVED massively — `cos(G_comp, G_corr)` median 0.9558 vs signed_ema's 0.717;
(c) no length/clip collapse on the promoted realization; (d) budget intact
(running_count=0, lifetime $697.86 « $1500 cap); (e) iterations not exhausted.
Why REVISE and not PASS: the headline parity box `val@50 >= 0.7414` is UNMET.

## Success criteria

Pre-run gate:
- [x] (1) every hard-gate correctness invariant passed the pre-run probe (Step-A
      decision + STEP_C_SPEC confirm: off-path parity, probes-never-feed-optimizer,
      anchor-owns-Q `powersgd_basis_updates=0`/`anchor_q_updates>0`, full-pass-only-
      in-anchor, delay_K=5 on the training path, EF limiting-case identity + no-sign-
      term static check, fp32 dump fidelity drift 4.5e-5 « 1e-3, FSDP/grad-ckpt/bf16).

Step A — geometry audit (the gate):
- [~] (2) `cos(G_dense, G_comp) >= 0.95` for plain PowerSGD — MEASURED-BUT-CONFOUNDED
      (operand/loss mismatch; the Option-A reference `cos(G_fresh_anchor,G_comp)=+0.010`
      is non-comparable, NOT evidence of compression rotating the update). Plain
      PowerSGD r77 ties dense at 0.7415 (locked #25), so compression is direction-benign
      at the outcome level. Scored ~ per the documented MEASUREMENT CAVEAT.
- [x] (3) `cos(G_dense, G_corr)` for signed_ema materially below plain's — CONFIRMED via
      confound-free isolate: `cos(G_comp, G_corr) = 0.717` (~44 deg merger rotation).
- [~] (4) `Q_act` activation capture >= 0.99 (PASS, median 0.9985) AND update-energy
      capture reported (0.318, off-principal 0.68) — H2 reads as MISS, confound-caveated.
- [x] (5) sign-agreement at delay_K in {0,5} reported — CONFIRMED coin-flip
      (0.500 / 0.523 / 0.520 in [0.45,0.55]) => sign-replacement structurally unrecoverable.
- [x] (6) machine-readable DECISION emitted — `go_C_then_B + retire_sign_replacement(confirmed)`.

Step B / C / E — training arms (the headline):
- [x] (7) every training cell reached its step_target without NaN/non-finite gradients
      (B_ef_r2 50/50; B_plain 50/50; C2_hybrid 50/50; grad_norm all finite, 0 numeric NaN/Inf).
      NOTE the one EXCEPTION: B_ef r1 OOM'd at step 42 — see criterion (12) / alarm.
- [x] (8) controlled variables hold equal across arms — ASSERTED from the resolved
      Hydra dumps: ALL arms share compression_type=powersgd, rank=77, anchor.owns_q=true,
      cadence=5, delay_K=5, clean_cadence=0; ONLY the variable-under-test differs
      (B_plain spectral OFF/q_basis=act; C2_hybrid spectral OFF/q_basis=HYBRID;
      B_ef spectral ON/correction_mode=ef_powersgd/ef_clip=1.0/ef_decay=0.9/q_basis=act).
- [a] (9) dense baseline reproduces W&B 0.7536 ± 0.01 — OPERATOR-AMENDED (2026-06-10):
      do NOT re-run dense; reference 5e2jpho9 val@50=0.7536 ADOPTED. The dead
      exp26_B_dense cell (c6owhmv6, cancelled at init) is NOT a failed criterion.
      Scored as amended (reference adopted, not re-measured).
- [x] (10) best Step-B (ef_powersgd) arm `cos(G_dense, G_corr)` improves over plain's
      `cos(G_dense, G_comp)` — CONFIRMED via the plan-mandated confound-free discriminator:
      `cos(G_comp, G_corr)` median 0.9558 (ef) vs 0.717 (signed_ema) vs 1.0 (plain, trivial).
      ef preserves direction (~17 deg) while applying a real residual (dose median 0.321).
- [ ] (11) best promoted arm `val@50 >= 0.7414` — **UNMET.** B_ef_r2 = 0.7210
      (target 0.7414; below by 0.0204). PARITY NOT REACHED. This is the one box that
      blocks PASS and drives REVISE.
- [x] (12) best promoted arm NO length/clip collapse — for the PROMOTED realization
      (B_ef_r2): response_length/mean max 293.6 (< 2x step-10 baseline; lengths SHRINK
      to ~146), pg_clipfrac max 0.2889 (never enters 0.3-0.9 danger band), entropy
      5.72->0.40 (healthy decline, not the collapse signature). Box CHECKED for r2.
      ALARM: the SIBLING realization B_ef r1 (same clip=1.0) DID ignite — see Notes.
- [x] (13) Step E: promoted method's measured inter-stage comm volume < dense — CONFIRMED:
      comm/bytes_ratio 0.0506 (~19.8x reduction; B_ef_r2 step50 bytes_compressed ~1.86e7
      vs dense_equiv ~3.68e8). Reported number below.
- [x] (14) promoted arm val@50 does NOT fall below the no-refresh floor 0.6914 without a
      stated reason — B_ef_r2 0.7210 > 0.6914 (above the floor). (Separately: B_plain
      0.6437 IS below floor — a NEW negative result with a stated diagnostic reason; see
      finding 2. B_plain is NOT the promoted arm, so this box is satisfied for the promoted arm.)

Tally: 11 checked, 1 unmet (11), 2 caveated (2,4), 1 operator-amended (9). The single
UNMET box (11 = parity) is the load-bearing PASS blocker.

## Metrics summary
- B_ef_r2 val@25: 0.6740 (val-core/openai/gsm8k/acc/mean@1)
- B_ef_r2 val@50: 0.7210 (target parity 0.7414 — MISS by 0.0204; floor 0.6914 PASS)
- B_plain val@25 / val@50: 0.4094 / 0.6437 (BELOW the no-refresh floor 0.6914)
- C2_hybrid val@25 / val@50: 0.2024 / 0.3730 (Step C FAILED its gate hard)
- direction gate cos(G_comp,G_corr) median (ef, n=168, ticks 5_10..11_21): 0.9558
  (signed_ema ref 0.717; plain trivially 1.0) — DIRECTION PRESERVED (~17 deg)
- residual dose ||G_corr-G_comp||/||G_comp|| median (ef): 0.3215 (range 0.110-0.860;
  rises monotonically across ticks 0.30 -> 0.47 — the residual grows as Q goes stale)
- B_ef_r2 response_length/mean: max 293.6, final ~146 (target: <= 2x step-10 baseline)
- B_ef_r2 pg_clipfrac: max 0.2889 (target: never in 0.3-0.9 band)
- B_ef_r2 grad_norm: median 4.916, finite throughout (warmup spikes 185/221 @ steps 1-2)
- Step E comm/bytes_ratio: 0.0506 (~19.8x reduction) — see Step-E section

## Comparisons to baseline_run: EXP-25

| run | method | val@50 | vs EXP-25 signed_ema 0.7066 | vs floor 0.6914 | vs parity 0.7414 |
|---|---|---|---|---|---|
| EXP-26 B_ef_r2 | ef_powersgd (direction-preserving) | **0.7210** | +0.0144 | +0.0296 | -0.0204 |
| EXP-26 B_plain | plain PowerSGD r77 + anchor-refresh | 0.6437 | -0.0629 | **-0.0477** | -0.0977 |
| EXP-26 C2_hybrid | hybrid-Q (Step C) | 0.3730 | -0.3336 | -0.3184 | -0.3684 |
| EXP-25 ref | signed_ema a0.5 (1wulaelw) | 0.7066 | — | +0.0152 | -0.0348 |
| dense ref (amended) | comm-eff OFF (5e2jpho9) | 0.7536 | — | — | +0.0122 |

EXP-26's ef_powersgd is the best realistic comm-eff result to date: it beats the
falsified EXP-25 signed_ema by +1.4 pts AND the realistic-substrate plain codec by
+7.7 pts, with the merger DIRECTION confirmed preserved (cos 0.9558 vs 0.717) and no
deterministic collapse — but it still falls 2.0 pts short of the PowerSGD/fresh-clean
parity band. diff_against_baseline.py rc=0 (EXP-20/23 dirs cleared; refs read from
W&B per the documented condition).

## Resolved parameters (ground truth)
Source: extracted from `train_exp26_B_ef_r2.log` (resolved Hydra override block) +
`resolved_params.txt` (Step-A A1 substrate). The B-arm values below are the LAUNCHED
values, NOT the plan table.

B_ef_r2 (the promoted/best arm) comm-eff substrate + variable-under-test:
  compression_type=powersgd  powersgd.rank=77  powersgd.q_basis=act  sync_basis=true
  anchor.enabled=true  anchor.owns_q=true  anchor.cadence=5  anchor.delay_K=5
  clean_cadence=0  spectral.enabled=true  spectral.correction_mode=ef_powersgd
  spectral.ef_clip=1.0  spectral.ef_decay=0.9  spectral.beta_anc=0.95  ema_device=cpu
Fixed control surface (all arms): train_batch_size=128  ppo_mini_batch_size=64
  rollout.n=8  max_response_length=16384  max_prompt_length=1024  optim.lr=1e-6
  use_kl_loss=False  use_kl_in_reward=False  entropy_coeff=0  (vanilla GRPO no-KL/no-entropy)

Divergence from the plan to flag:
- `spectral.signed_ema_alpha=0.5` is still set in the B_ef resolved dump but is a DEAD
  default — correction_mode=ef_powersgd carries NO sign term (no-sign-term hard gate
  green). It does not affect the run; flagged only so a future reader does not mistake
  it for an active sign-replacement knob.
- `resolved_params.txt`/`resolved_cmd.txt` capture the Step-A A1 invocation
  (experiment_name=exp26_A1_powersgd_r77, spectral OFF, total_training_steps=6), NOT the
  B_ef arm. capture_resolved_config.py keys off the first main_ppo in train.log, which is
  the Step-A arm. The B-arm ground truth above is parsed directly from the B_ef_r2 log.
  Provenance is recoverable; noted so the next iteration regenerates resolved_params
  from the B-arm log if it re-runs.

## next_actions (REVISE only)
- knob: ef_clip
  from: 1.0
  to: 0.5
  rationale: "ef_powersgd preserves direction (cos 0.9558) and clears the floor, but the
    residual dose grows monotonically with Q-staleness (median 0.32, up to 0.86 / 0.47
    at the latest captured tick). On the r1 realization the un-damped clip=1.0 residual
    ignited the EXP-25 length-explosion (resp_len/max pinned 16384, entropy 5.70->0.13,
    OOM @ step 42). Halving the clip caps the per-tick residual injection to damp that
    stochastic ignition while keeping the direction-preserving correction that gained
    +7.7 pts over plain. Plan-allowed REVISE knob (ef residual clip)."
- knob: ef_decay
  from: 0.9
  to: 0.5
  rationale: "Faster residual forgetting prevents the dropped-energy buffer from
    accumulating into a large delayed kick as Q goes stale (the dose-vs-tick climb).
    Pair with the clip cut to attack both the magnitude and the persistence of the
    residual that correlates with the r1 ignition. Plan-allowed REVISE knob (ef decay)."
- knob: step_target
  from: 50
  to: 100
  rationale: "B_ef_r2 val rose 0.674@25 -> 0.721@50 still climbing with shrinking lengths
    and no peak-then-crash — the curve has not converged. Parity (0.7414) is only 2.0 pts
    away and the trajectory is monotone-up; extending to 100 steps tests whether the
    remaining gap closes under the same (damped) substrate without a new mechanism.
    Plan-allowed REVISE knob (step_target 50->100). Run WITH the damped clip/decay above,
    not at clip=1.0, to avoid re-rolling the ignition dice over a longer horizon."

## Step-E communication number
Promoted method (ef_powersgd, B_ef_r2) measured inter-stage activation comm:
  comm/bytes_ratio = 0.0506 (compressed boundary activation payload vs dense equivalent)
  => ~19.8x reduction. Example (B_ef_r2 step50): bytes_compressed ~1.86e7 vs
  dense_equiv ~3.68e8.
HONEST CAVEAT: this counts the boundary ACTIVATION payload (N*r vs N*H). The anchor's
Q/M broadcast traffic is cadence-amortized (H*r per boundary every 5 ticks) and small
relative to N*H, so it is excluded from the headline ratio. Step E criterion
(comm < dense) is SATISFIED as a measured number; the ratio is for the activation path,
not a full end-to-end accounting.

## Notes
THREE HEADLINE FINDINGS:
1. ef_powersgd is the BEST realistic comm-eff result to date (val@50 0.7210), it is
   DIRECTION-PRESERVING (cos(G_comp,G_corr) 0.9558 vs signed_ema 0.717), beats the
   EXP-25 signed_ema lineage by +1.4 pts and the realistic plain codec by +7.7 pts, and
   shows no deterministic collapse on the promoted run — but PARITY (0.7414) is NOT
   reached (short by 2.0 pts). The direction-corruption thesis of #25 is vindicated:
   removing the sign term and using error-feedback recovers most of the lost ground.
2. B_plain (plain codec on the REALISTIC substrate) lands at 0.6437, BELOW the no-refresh
   floor 0.6914 — a NEW negative result: stale-anchor-owned-Q refresh ALONE is harmful,
   and it INVERTS EXP-25's "merger net-harmful" framing (on this substrate signed_ema
   0.7066 actually BEAT plain). The realistic anchor-refresh substrate is itself a drag;
   the merger is partly COMPENSATING for it, not purely corrupting.
3. Step C is EMPIRICALLY FALSIFIED: the geometry-winning hybrid Q (UC 0.2496 > act 0.2010,
   OPP 0.0685, AC 0.999) ANTI-CONVERTS in training — C2_hybrid val@50 0.3730, with
   train-vs-rollout divergence (rollout_corr/kl 7.3->13.1, training_ppl->6.5e6, recon
   stayed ~0.023, NO length explosion). Mechanism: vLLM rollouts are UNCOMPRESSED, so a Q
   that maximizes update-energy capture at the expense of activation-reconstruction
   destroys train<->rollout consistency. Q content must preserve forward fidelity (act
   does; update-energy content does not). H2's "RLVR-native Q recovers parity" is dead.

r1-IGNITION ALARM (carry into the child experiment): ef_powersgd at clip=1.0 has a
STOCHASTIC length-explosion risk. Of two realizations on identical config, r1 (c7fa7kjv)
fired the full EXP-25 ignition fingerprint — response_length/max PINNED at 16384, entropy
5.70->0.13, erratic ppo_kl — and OOM'd at step 42 in the anchor backward; r2 (tilwe80t)
completed clean. The verdict CHECKS the no-collapse box on the promoted r2 realization,
but the next iteration MUST damp the residual (ef_clip 1.0->0.5, ef_decay 0.9->0.5)
before extending the horizon — do not re-roll ignition over 100 steps at clip=1.0.

MEASUREMENT CAVEAT (carried from Step A): the literal weight-space cos(G_dense, G_corr)
is operand/loss-confounded (clean-PG anchor grad vs PPO-clip fast grad; activation-vs-
weight operand). The direction gate used the plan-mandated confound-free discriminator
cos(G_comp, G_corr) per STEP_C_SPEC; the improvement (0.9558 vs 0.717) is robust to the
caveat (it never references a dense grad).

Completion: no top-level done.flag, but stage done.flags present (exp26_B_ef.done.flag,
exp26_B_ef_r2.done.flag, exp26_B_plain.done.flag, exp26_C2_hybrid.done.flag,
c2b_chain.done.flag), box torn down, metrics non-empty. Treated as complete.
Resolved-config provenance: capture_resolved_config.py succeeded on the Step-A A1
invocation; B-arm ground truth parsed directly from the B-arm logs (noted above).
