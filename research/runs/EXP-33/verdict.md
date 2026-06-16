# Verdict EXP-33 — 2026-06-16T18:51:48Z

## Result
VERDICT: PASS
note: measurement PASS — the β_anc→accuracy curve is captured cleanly with the
predicted endpoints; `promote_launcher_as: none` (C0/β=0 IS the existing B2 SOTA, so
nothing is promoted). `iterations: 0` ⇒ no REVISE branch; this resolved PASS vs STOP only.

## Success criteria
- [x] **off-axis parity** — every cell's resolved trace = B2 + exactly one varied key. Per-cell `resolved_params.txt` (captured from each `<cell>/train.log`) diffed against `runs/EXP-31/B2_baseline/resolved_params_B2.txt`: the ONLY differing comm-eff key is `spectral.beta_anc` (0.00/0.25/0.50/0.75/1.00, last-wins over the wrapper's hard `=0.0` export via the Hydra passthrough), plus the allowed run-id deltas (`experiment_name` b0p00..b1p00, `project_name=verl_compression_research_beta_sweep`, `total_training_steps` 55/55/55/55/**30**, `val_before_train=false` on C1–C4). One extra delta surfaced — `spectral.blend_eta`: B2=0.3 vs all-cells=0.5 — but it is **inert under `correction_mode=delayed_ef`** (consumed ONLY by `blend_matrix()`; `delayed_ef_matrix()` never reads it; **0 `[blend]` diagnostics in any cell**) AND identical across all 5 cells ⇒ no cross-cell confound. Substrate (PowerSGD r=77/q_basis=act/sync_basis=true, anchor cadence=delay_K=5/owns_q/replay_paired_batch=true/snapshot_device=cpu, delayed_ef λ=1.0, all other levers OFF, clean_cadence=0) is byte-identical to B2 on every cell. (observed: only `beta_anc`+run-id+inert-`blend_eta` differ)
- [x] **identical comm budget** — `actor/comm/bytes_ratio` per cell: C0 [0.05039,0.05056], C1 [0.05038,0.05054], C2 [0.05037,0.05055], C3 [0.05037,0.05055], C4 [0.05043,0.05049]. All within B2's gate [0.0500,0.0510] and B2's observed band 0.0504–0.0506; distributions overlap across all 5 cells. β is training-side; comm cost did not move. (observed: 0.0504–0.0506 all cells)
- [x] **recon unchanged** — steady-state `powersgd_reconstruction_rel_error` ≈ 0.024–0.028 (act band) on every cell. The only >0.5 reads are exactly the **first 2 chronological reads per cell** (~0.975 step-1 warm-up transient before `warm_start` Q converges), NOT the ~0.68 Step-C plateau. (observed: ss≈0.025; 2/55 warmup reads)
- [x] **completion + stability** — C0–C3 reached global_step 55 (val@50 captured + flushed via the 5-step buffer); C4 reached its degenerate gs30 bracket. No real NaN / non-finite gradient / OOM on any cell (grad_norm finite, 0.55–408, step-1 spikes only). NO length-hack ignition: response_length/mean DECLINES over training (270s → 190–230s) on every cell ⇒ P1 (no cap-pins), P2 (slope ≤ 0), P3 (no >2× growth), E1 (max mean ~310 ≪ 4k) all CLEAR. (observed: clean on all 5)
- [x] **READ (a) — control reproduces** — C0 (β=0) val@50 = **0.73844 ∈ [0.716, 0.774]**. The β=0 endpoint reproduces the B2 band; the box/seed is sound and box-controls C1–C4. (observed: 0.73844, target [0.716,0.774])
- [x] **READ (b) — degenerate bracket** — C4 (β=1) **cold-M collapse CONFIRMED**: `merger_coldM_fallbacks = 196/196` PERMANENTLY (ticks 21–40 all 196 = all matrices frozen) ⇒ `M_rep = 1.0·M_rep` stays at its `torch.zeros` cold-start ⇒ `delayed_ef` is a strict no-op ⇒ plain PowerSGD, exactly the predicted mechanism. (Contrast: C0's coldM = 196 for ticks 1–9 (delay_K warmup) then **drops to 0 at tick 10** and stays 0 — M warms normally.) C4 ran the degenerate 30-step bracket so it has NO val@50; its reads are val@25=0.44807 and val@30=0.56406, **climbing toward the no-merger floor band [0.606,0.654] from below** (only ~12 effective gradient steps under merger-free PowerSGD — not yet at the floor). The cold-M MECHANISM read is fully satisfied; the floor-band numeric read is partial-by-design (degenerate bracket never reaching gs50). (observed: 196/196 cold-M; val@30=0.56406, floor band [0.606,0.654])
- [x] **READ (c) — β→accuracy curve (HEADLINE)** — full curve tabulated below. `max(val@50[C1,C2,C3]) = 0.75284` (C2, β=0.50); gap vs C0 = **+0.01440 ≤ +0.024** ⇒ the freshness-best hypothesis is **SUPPORTED** (no interior β beats the β=0 SOTA beyond ±0.024 noise; none reaches the falsification bar 0.7624). (observed: gap +0.0144, target ≤ +0.024)

## Metrics summary
β→accuracy curve (`val-core/openai/gsm8k/acc/mean@1`; val `step:N` index = 2× global_step because batch128/mini64 = 2 optimizer ticks/step, so `step:50`=val@gs25, `step:100`=val@gs50, `step:110`=val@gs55, `step:60`=val@gs30):

| cell | β_anc | val@25 | **val@50 (read of record)** | val@55 (buffer) | gap vs C0 (0.73844) | vs ±0.024 noise |
|---|---|---|---|---|---|---|
| **C0 `b0p00`** | 0.00 | 0.71418 | **0.73844** | 0.73616 | — (control) | reference |
| **C1 `b0p25`** | 0.25 | 0.71418 | **0.73995** | 0.74450 | +0.00151 | TIE (within ±0.024) |
| **C2 `b0p50`** | 0.50 | 0.70811 | **0.75284** | 0.74147 | **+0.01440** | TIE / mild up (within ±0.024) |
| **C3 `b0p75`** | 0.75 | 0.70053 | **0.72176** | 0.72782 | −0.01668 | TIE / mild down (within ±0.024) |
| **C4 `b1p00`** | 1.00 | 0.44807 | — (30-step bracket; val@30=0.56406) | — | n/a (no merger) | below floor band, climbing |

- bytes_ratio (parity proof): all 5 cells 0.0504–0.0506 (gate [0.0500,0.0510]) — β is comm-neutral.
- recon_rel_error: steady-state ≈ 0.024–0.028 (act band) all cells; 2 warmup reads/cell ≈ 0.975.
- C4 merger_coldM_fallbacks: 196/196 permanent (cold-M collapse → plain PowerSGD).
- C0 val@0 = 0.08188 (from its original run; `val_before_train=false` on C1–C4 reuses it — measurement optimization, not an off-axis change; step-0 is the untrained base, identical across cells).

**Curve shape:** a FLAT free-averaging region across β∈[0,0.25,0.5,0.75] — all four interior/endpoint cells lie within ±0.024 of the C0 control (the non-monotonic spread is noise-dominated). β=0.50 (C2) is the nominal peak at +0.0144 but does NOT clear the +0.024 beyond-noise bar; β=1.00 (C4) is the degenerate no-merger cliff. **No β>0 cell strictly beats β=0 beyond noise.**

## Comparisons to baseline_run: EXP-31/B2_baseline
`diff_against_baseline.py` found no common `train.jsonl` numeric keys (this sweep logs to per-cell `train.log` + the `verl_compression_research_beta_sweep` WandB project, not a `metrics/train.jsonl`), so off-axis parity was verified by diffing each cell's captured `resolved_params.txt` against `resolved_params_B2.txt` directly (see Notes). Result: C0 (β=0) IS the in-sweep B2 reproduction — its val@50=0.73844 sits squarely inside the historical B2 band ([0.716,0.774]; B2 ref 0.7354 same-box / 0.7528 EXP-30). Every cell holds the full B2 substrate byte-identical except `spectral.beta_anc` (the swept axis) and the inert `blend_eta` (dead under delayed_ef). The same-box C0 is the gold-standard reference for the curve; the relative β-ordering is the robust signal.

## Resolved parameters (ground truth)
Source: per-cell `runs/EXP-33/<cell>/resolved_params.txt` (extracted from each cell's `train.log` `main_ppo` invocation — NOT the plan). The top-level `runs/EXP-33/resolved_params.txt` reflects only the LAST cell (b0p75, β=0.75), because the top-level `train.log` is the last resume cell; per-cell captures are authoritative.

The swept knob, per cell (verified last-wins over the wrapper's hard `export ...BETA_ANC=0.0`):
```
b0p00: actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.00
b0p25: actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.25
b0p50: actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.50
b0p75: actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.75
b1p00: actor_rollout_ref.actor.comm_eff.spectral.beta_anc=1.00
```
Substrate (byte-identical across all 5 cells AND to B2):
```
actor_rollout_ref.actor.comm_eff.compression_type=powersgd
actor_rollout_ref.actor.comm_eff.powersgd.rank=77   q_basis=act   sync_basis=true   warm_start=true   update_cadence=1
actor_rollout_ref.actor.comm_eff.anchor.enabled=true  owns_q=true  cadence=5  delay_K=5  replay_paired_batch=true  snapshot_device=cpu
actor_rollout_ref.actor.comm_eff.spectral.correction_mode=delayed_ef  delayed_ef_lambda=1.0  cadence=1  max_targets=-1  ema_device=cpu
actor_rollout_ref.actor.comm_eff.spectral.ef_decay=0.0  ef_clip=0.0  delta_subbasis_rank=0  perturb_sigma=0.0  delta_momentum_mu=0.0  adaptive_lambda_mode=off
actor_rollout_ref.actor.comm_eff.clean_cadence=0   mask.enabled=false
actor_rollout_ref.rollout.name=vllm  n=8  tensor_model_parallel_size=2
data.train_batch_size=128  ppo_mini_batch_size=64  max_response_length=16384
```
**DIVERGENCE from the B2 snapshot (a finding, not a confound):** `spectral.blend_eta=0.5` in all 5 EXP-33 cells vs `0.3` in `resolved_params_B2.txt`. This is the b2_sota wrapper's current default (0.5), differing from the value frozen in the EXP-31 B2 snapshot (0.3). It is **inert under `correction_mode=delayed_ef`** — `blend_eta` is read only inside `blend_matrix()` (spectral_filter.py:435), which is never invoked on the delayed_ef path (early-return dispatch at spectral_filter.py:1180); 0 `[blend]` diagnostics fired in any cell. Because it is also identical across all 5 cells, it cannot create a cross-cell confound. The β-sweep comparison is sound. (Flagged so the next planner knows the wrapper default drifted from the B2 snapshot; harmless here, but worth aligning if a `blend`-mode experiment reuses this wrapper.)

## Notes
- **Verdict mechanics.** PASS via the predicate: STOP path (i) control failure — NO (C0 reproduces B2); (ii) falsifying β>0 surpass — NO (max interior val@50=0.75284 < falsification bar 0.7624); (iii) ignition — NO (all trip-wires clear); (iv) budget — NO (box already torn down, science captured). All success boxes checked ⇒ PASS. The curve SUPPORTS freshness-best (gap +0.0144 ≤ +0.024) — the expected, non-promoting measurement outcome.
- **Science one-liner.** On the valid-M delayed_ef substrate, β_anc is a FLAT free-averaging region for β∈[0,0.75]: mild averaging is *free* (C1/C2/C3 all tie C0 within ±0.024; C2/β=0.5 is the noise-bounded nominal peak at +0.0144) but never *helpful* beyond noise, and β=1 collapses (cold-M no-op → plain PowerSGD). Consistent with the standing design invariant "freshness ≥ variance-reduction" and the prior `anchor-gradient-ema-beta0-grpo` finding (β=0 is the right default for comm-eff GRPO). This sweep is the first direct β-curve on the valid-M circuit and it HOLDS β=0 as the default — it does not justify changing it. `promote_launcher_as: none` — B2 (= C0, β=0) stays the reference.
- **Script/path provenance.** The plan's `research/scripts/X` + `runs/EXP-33` paths assume cwd=repo-root; actual layout has both `scripts/` and `runs/` under `research/`. Ran the exact same 4 scripts with absolute paths (logged in `analysis.log`). `analyze.py` emitted only an M0-smoke stub (it found no `metrics/*.jsonl` and parsed only the top-level train.log) — this verdict is the analyst's full hand-verified product. `diff_against_baseline.py` found no common train.jsonl keys (per-cell logging) → off-axis parity verified by per-cell `resolved_params.txt` diff instead (FALLBACK, stated above). `check_budget.py`: 0 running instances, $0 (box torn down). All reads cross-checked against the operator-banked values and match exactly.
- **Completion confirmed.** `runs/EXP-33/done.flag` present + all 5 cells have non-empty `train.log` + per-cell done flags. C0/C1 from the original run, C2/C3/C4 from the clean resume (the gs=4 C2 crash was re-run from scratch as planned).
- **Benign (NOT failures), as flagged.** C4 `ValueError: Cannot use run() inside async loop` during WandB `finish` at teardown — the Final validation metrics (val@30=0.56406) were computed and printed AFTER the traceback, so the read is intact (the EXP-32-style WandB-403/teardown hiccup). DataLoader SIGKILL / Ray-teardown tracebacks at each cell's end are normal worker shutdown. The bare-word `nan` token in the `[delayed_ef]` banner lines is a non-numeric log fragment — no grad_norm/loss/pg_loss/reward/acc value is NaN on any cell.
