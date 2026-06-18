# Verdict EXP-35 — 2026-06-18T06:40:00+00:00

## Result
VERDICT: REVISE

EXP-35 is a fully interpretable, scientifically clean run: the C3 surface-validation
GATE passes, the one-knob invariant holds byte-for-byte, the acceleration stack is
validated compute-bound, and the α curve is well-resolved. Two of the plan's
`## Success criteria` boxes are unchecked — both for benign, addressable reasons
(an H100-specific memory band that the H200 fallback could not meet at the same
token budget, and a C1-endpoint clause falsified by a genuine positive finding).
Neither is a broken/un-interpretable run, so this is REVISE (2 ≤ iterations=3),
not STOP. The headline science (parity-band α ridge peaking at α=0.25, no surpass)
is consistent with the σ(M)-ceiling thesis and is the EXPECTED, hypothesis-confirming
shape.

## Success criteria
- [x] (dependency-gate clarification) operator resolved dispatch by treating EXP-35 as a fresh sweep whose C3 re-establishes the 0.7635 control inline (the run dispatched + completed behind the REVISE parent EXP-34 — de-facto resolution; C3 ∈ band, see below)
- [ ] (mem-calibration) anchor-firing step `perf/max_memory_reserved_gb` ∈ [68, 76] GB — **NOT met**: ran on the **4×H200 fallback** (143 GB/card) at `ppo_max_token_len_per_gpu=24576`, peaking at **~123–125 GB reserved** on anchor steps (observed: C1 124.30, C2 123.74, C3 124.66, C4 123.13, C5 123.84). The [68,76] band was calibrated for the **80 GB H100**; on H200 the same budget runs at ~87% with ~19 GB margin and **no OOM**. The token budget IS identical across C1–C5 (24576, verified in every `resolved_params.txt`), so the one-knob spirit holds — but the literal GB band is unmet because the box is H200, not H100.
- [x] (correctness) every cell reached 50 steps with no NaN/non-finite gradients (observed: step:50 training row present in all 5 train.logs; 0 NaN/non-finite hits; grad_norm finite, e.g. C1=1.53 @step50; no early-kill / EARLY_STOP_SIGNAL)
- [x] **(SURFACE-VALIDATION GATE)** C3 (α=0.5) `val@50 = 0.7415` ∈ [0.7395, 0.7875] (observed: 0.7414708, margin +0.0020 to floor). GATE PASSES — the absolute α curve is interpretable against the 0.7635 anchor (note: passes by a narrow 0.002 margin at the low edge).
- [x] (one-knob) diff of the comm_eff + acceleration block of all 5 `resolved_params.txt` yields ONLY `signed_ema_alpha ∈ {0.0,0.25,0.5,0.75,1.0}` + `experiment_name`; `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=24576`, `tensor_model_parallel_size=1`, `gpu_memory_utilization=0.55`, `max_response_length=2048` are byte-identical across all 5 (verified: normalized diff vs C3 = empty for C1/C2/C4/C5)
- [ ] (curve endpoints) C5 ≤ 0.66 AND (C1 ignites OR lands materially below interior) — **partially met**: C5 (α=1.0) `val@50 = 0.6475 ≤ 0.66` ✓ (no-merger PowerSGD floor). BUT C1 (α=0.0) **did NOT ignite** (completed 50 steps, response_length/mean ≈ 210, clip_ratio ≈ 0.001, entropy 5.76→0.93 graceful) AND **does NOT land materially below the interior cells** — at `val@50 = 0.7437` it BEATS C4 (α=0.75 = 0.7043). This compound clause is **not satisfied as written**; the failure is itself the headline finding (see Notes).
- [x] (α curve) val@50 tabulated vs α for all 5 cells; interior peak (among 0.25/0.5/0.75) identified = **α=0.25 (0.7528), BELOW 0.5**
- [x] (speed) every cell ≤ ~45 min/50 steps (steps-only wall: C1 24.9, C2 24.5, C3 24.7, C4 26.8, C5 25.8 min — under the 30 min stretch); `timing_s/update_actor` ≈ 36.4–36.9 s + `old_log_prob` ≈ 2.0–2.6 s dominate `timing_s/step` ≈ 49.6–53.4 s, a large reduction vs the ~2 hr/50-step locked-surface baseline
- [x] (GPU saturation) monitor confirms sustained util 92–100% with power 576–700 W (near H200 TDP) during `update_actor`/anchor steps; `max_memory_reserved_gb` driven high (~124 GB) — note: > 76 GB because H200, not H100; the < 76 GB sub-clause is the same H100-band miss as the mem-calibration box
- [x] (comm) `actor/comm/bytes_ratio` = 0.05050 / 0.05051 / 0.05052 / 0.05052 / 0.05053 across C1–C5 (≈ 0.05, PowerSGD r=77 active; logical_pp_bytes_powersgd_y_only=77.0)
- [x] (W&B routing) all 5 cells resolve `trainer.project_name=verl_compression_research_alpha_sweep_signed_ema`
- [x] (account) box handle `handles/41420622.json` stamps `vast_account=team`
- [x] (no code) no PR opened; `code_change=false`, `promote_launcher_as=none`

## Metrics summary
val@50 (authoritative from each cell's train.log `val-core/openai/gsm8k/acc/mean@1`; WandB step-50 rows were backfilled from train.log per async-flush bug):
- α=0.00 (C1): val@25 0.6967, **val@50 0.7437**
- α=0.25 (C2): val@25 0.7005, **val@50 0.7528**  ← curve peak
- α=0.50 (C3): val@25 0.7293, **val@50 0.7415**  (surface-validation control; GATE ∈ band)
- α=0.75 (C4): val@25 0.7149, **val@50 0.7043**
- α=1.00 (C5): val@25 0.5474, **val@50 0.6475**  (no-merger PowerSGD floor; α=1 ⇒ G_noisy passed through)
- bytes_ratio: ~0.0505 (all 5; target ≈ 0.05)
- anchor_backwards: 20 (all 5; = 100 ticks / cadence 5, correct); merger_coldM_fallbacks: 0 (all 5, M warm)
- max_memory_reserved_gb (anchor steps): ~123–125 (target [68,76] on H100; ran on H200 — unmet, benign)
- timing_s/step: 49.6–53.4 (update_actor ~36.5 s = ~73% slice); steps-only wall ~25 min/50 steps (target ≤ 45)
- GPU util/power during update_actor: 92–100% / 576–700 W (near H200 TDP — compute-bound confirmed)

## Comparisons to baseline_run: EXP-34
`diff_against_baseline.py` ran EXIT=0 but reported "no common numeric keys" (it reads `train.jsonl`, which is not synced — metrics live in per-cell `train.log` as the plan documents). Falling back to the recorded EXP-34 reference per the plan's provenance fallback: **EXP-34 (signed_ema α=0.5, β_anc=0.50) val@50 = 0.7635 on the LOCKED 16384-context surface.** EXP-35 C3 (the same point on the ACCELERATED surface, max_response=2048 / dynamic bsz / TP=1) lands at **0.7415**, i.e. **0.0220 below** the locked-surface control but **inside the ±0.024 surface-validation band** (margin +0.002 to the floor). CRITICAL: this is a CROSS-SURFACE comparison — the EXP-34 0.7635 (itself n=1), the B2 SOTA 0.7528, and the dense band ~0.75–0.78 were all measured on the original 16384 surface. No dense + B2 re-baseline on THIS accelerated surface has been run, so absolute cross-surface ranking is NOT apples-to-apples; the directly interpretable result is the curve's RELATIVE shape (monotone-ish ridge peaking at α=0.25, declining to the α=1.0 floor 0.6475). Consistent with the σ(M)-ceiling thesis: no cell decisively surpasses dense; the whole spread C5..C2 = 0.6475..0.7528 sits in / below the parity band — parity, not surpass, exactly as predicted.

## Resolved parameters (ground truth)
Source: `runs/EXP-35/<cell>/resolved_params.txt` (extracted from each cell's train.log `main_ppo` trace, NOT the plan).
Per-cell swept knob + held merger config (verbatim):
```
C1: spectral.correction_mode=signed_ema  spectral.signed_ema_alpha=0.0   spectral.beta_anc=0.50
C2: spectral.correction_mode=signed_ema  spectral.signed_ema_alpha=0.25  spectral.beta_anc=0.50
C3: spectral.correction_mode=signed_ema  spectral.signed_ema_alpha=0.5   spectral.beta_anc=0.50
C4: spectral.correction_mode=signed_ema  spectral.signed_ema_alpha=0.75  spectral.beta_anc=0.50
C5: spectral.correction_mode=signed_ema  spectral.signed_ema_alpha=1.0   spectral.beta_anc=0.50
```
Acceleration stack (byte-identical across all 5 cells):
```
actor.use_dynamic_bsz=True
actor.ppo_max_token_len_per_gpu=24576
rollout.log_prob_max_token_len_per_gpu=32768   ref.log_prob_max_token_len_per_gpu=32768
rollout.tensor_model_parallel_size=1
rollout.gpu_memory_utilization=0.55
data.max_response_length=2048   data.max_prompt_length=1024
data.train_batch_size=128   actor.ppo_mini_batch_size=64   rollout.n=8
trainer.total_training_steps=50   trainer.val_before_train=False
trainer.project_name=verl_compression_research_alpha_sweep_signed_ema
```
**Divergence from plan (a finding):** the plan/issue title specifies **4×H100** as primary; the run executed on the **4×H200 fallback** (handle `41420622.json`: gpu_name=H200, num_gpus=4, gpu_ram=143771, vast_account=team). The plan explicitly authorizes H200 as a drop-in fallback "with more headroom," and the runner kept the SAME `ppo_max_token_len_per_gpu=24576` rather than re-calibrating to the H200 — which is WHY the anchor-step memory (~124 GB) sits far outside the H100-calibrated [68,76] GB band and the < 76 GB sub-clause. This is the sole reason two boxes are unchecked; it is a provisioning/calibration divergence, not a substrate or science defect. No other plan-vs-launched divergence found (merger, β_anc, accel stack, run length, W&B project, account all match the plan exactly).

## next_actions (REVISE only)
- knob: gpu_filter / box_type
  from: 4×H200 fallback (143 GB) at ppo_max_token_len_per_gpu=24576 (~124 GB reserved, outside [68,76])
  to: 4×H100 (80 GB) as the plan's primary tier — re-run the step-0 mem-calibration so an anchor-firing step lands in [68,76] GB; OR, if H200 is retained, lower ppo_max_token_len_per_gpu (24576 → ~12288) so the anchor step fits the band on the actual box
  rationale: the mem-calibration + GPU-saturation < 76 GB sub-clauses are unchecked ONLY because the run used the H200 fallback at the H100 token budget. Bringing the box (or the budget) into line satisfies the literal band without touching the science; the α curve is already validated, so this is a provisioning fix, not a science re-run.
- knob: signed_ema_alpha curve resolution (low-α side)
  from: 5 coarse points {0.0, 0.25, 0.5, 0.75, 1.0}, single draw each
  to: replicate the peak region α ∈ {0.0, 0.125, 0.25, 0.375} at 2–3 seeds to (a) confirm the peak is genuinely at α=0.25 not noise, and (b) formally re-bin the "C1 lands materially below interior" criterion — C1 (α=0.0 = 0.7437) BEATING C4 (α=0.75 = 0.7043) without igniting is the run's headline finding and deserves replication before the curve-endpoint box can be marked satisfied or formally rewritten
  rationale: the curve-endpoint criterion as written assumed α=0.0 ignites or lands low (EXP-25-era sign-SGD prior); the valid-M / anchor-circuit substrate falsified that. A short low-α replicate resolves whether to check the box (if the rewritten "C1 ≈ parity, no ignition" holds) and confirms the α=0.25 peak is real, all within iterations budget.

## Notes
- **Headline finding — α=0.0 did NOT ignite (substrate-stabilization).** Contradicting the EXP-25 sign-SGD / entropy-collapse prediction, C1 (α=0.0, the degenerate `|G|·sign(M)` merger) completed all 50 steps cleanly: response_length/mean ~210 tok (≪ the recalibrated 1.5k ignition gate = 0.7×2048), max clip_ratio ~0.001, grad_norm 1.53 @step50, entropy declining gracefully 5.76→1.33→0.93 (no sharpening spiral), and val@50 = 0.7437 — second-highest in the sweep, ABOVE α=0.75. The valid-M / anchor-circuit substrate (post-#29) appears to have stabilized the historically-unstable degenerate endpoint. This both falsifies the curve-endpoint criterion-as-written AND is positive evidence the anchor circuit removed the EXP-25 ignition pathology. Recommend the planner update the EXP-25-era ignition prior for the valid-M substrate.
- **σ(M)-ceiling thesis confirmed (parity, not surpass).** Per the plan's STOP note, parity ≈ B2/dense is the EXPECTED PASS-worthy outcome for this hypothesis, NOT a falsification — α is a deterministic reweighting of (G, M) so the curve stays in/below the parity band, which it does (0.6475..0.7528). The sweep is internally valid (best α = 0.25 among 5, self-validated by C3 ∈ band). This is why the verdict is REVISE on housekeeping criteria, NOT STOP.
- **Curve shape is the deliverable, and it is non-degenerate.** A clear ridge: rising 0.7437 (α=0.0) → peak 0.7528 (α=0.25) → 0.7415 (α=0.5) → 0.7043 (α=0.75) → floor 0.6475 (α=1.0). The interior peak is BELOW 0.5 (at 0.25), refining the EXP-34 β-anc=0.50 / α=0.5 working point toward lower α. Single-draw per the plan's seed_replicates=1 (curve SHAPE, not a replicated margin, is the stated deliverable) — but see next_action 2 for peak-replication.
- **C3 GATE passes narrowly (+0.002).** The absolute curve is interpretable per the gate, but the 0.002 margin to the floor plus the cross-surface caveat means promotion of any α finding to the canonical ledger still REQUIRES a separate dense + B2 re-baseline on the accelerated surface first (plan keeps `promote_launcher_as=none`). Do NOT enter 0.7528 (α=0.25) into the SOTA card as comparable to the locked-surface 0.7635/0.7528 — they are different surfaces.
- **Verification-script note:** `analyze.py --emit verdict.md` produced only an M0 smoke skeleton (no top-level `metrics/*.jsonl`; metrics live in per-cell train.log as the plan anticipates) and `diff_against_baseline.py` found no common keys (reads unsynced `train.jsonl`). Both expected; this verdict is driven by direct grep of each cell's authoritative `train.log` rows, per the plan's "train.log console rows are the authoritative durable record" directive. Outputs captured in `runs/EXP-35/analysis.log`.
- **Box status:** monitor-detail.log shows tmux DEAD (normal exit), no stalls, all artifacts pulled. `check_budget.py` reported `running_count: 1` / `running_dph: 12.88` — likely a ledger/teardown lag for instance 41420622; flag to the orchestrator to confirm teardown (training is complete; teardown is the orchestrator's responsibility, not the analyst's).
- **rc.txt=1 on every cell is benign** (post-train WandB/DataLoader teardown), confirmed: each cell has its step:50 training row + val@50 in train.log + done.flag. C5 has no rc.txt but has done.flag + complete train.log — also complete.
