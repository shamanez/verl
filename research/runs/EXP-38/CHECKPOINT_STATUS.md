# EXP-38 — Checkpoint (data collected; analysis HELD for operator go-ahead)

Revised sequence per operator (2026-06-20): collect + sync everything from the
75-step run, verify it is **sufficient to analyze the next comm-eff method**,
tear down the GPU, then **STOP and wait** for the command to start analysis.

## ✅ Done (data collection + instrumentation)

- [x] **Instrumentation built + merged-ready** on `exp/38-dense-drift-probe` (origin/shamanez/verl @ `ef0a3e7`):
      two default-OFF diagnostic gates (weight/grad/optimizer **drift probe** + capture-only
      **boundary-activation probe**) across 6 files + the Hydra schema (`actor.yaml`).
- [x] **Step-1 off-parity gate PASS** — both flags OFF == unmodified dense GRPO: comm_eff counters all 0,
      no drift dir written, no `[exp38]` lines, healthy loss/grad_norm; byte-faithful to pre-patch dense.
- [x] **On-path validated** — all 5 capture roles fire; FSDP full-param summon works; boundary `grad_h`
      captured via `h.register_hook` under gradient checkpointing; **max_mem 35.6 GB ≤ 36.3 GB baseline**
      (capture adds no net HBM); **zero "capture skipped" warnings** across the whole 75-step run.
- [x] **75-step dense capture COMPLETE** — `comm_eff.enabled=false`, both probes ON, 150 optimizer ticks.
      val@25/@50/@75 = **0.7521 / 0.7665 / 0.7688** (monotonic; @25/@50 match EXP-37D dense to 4 dp ⇒
      probes confirmed measurement-only; +0.008 at @75 = normal GRPO rollout variance). 0 NaN / 0 OOM.
- [x] **Capture bounded** — 16 GB on box (≤ 40 GB ceiling), rank0-only, N=5, token cap 2048.
- [x] **WandB backfilled** — run `mhegvmbs`; step-75 (val@75=0.7688 + 122 scalars) pushed past the async
      tail-drop. Authoritative metrics also in local `train.log` (514 KB, all 75 steps).
- [x] **SUFFICIENCY for feature-work analysis CONFIRMED** (manifest, 1071 rows):
      - 5 roles complete: theta/g_dense/update_vector = 315 each (15 matrices × 21 ticks);
        boundary_h/grad_h = 63 each (3 boundaries × 21 ticks).
      - **Lag axis k∈{1,2,5,10,20,40} all sampleable** — k=5 (stable 5/5): 2 pairs; **k=20 (broken 20/20): 13 pairs**;
        k=40 (beyond): 11 pairs. The headline staleness comparisons are covered.
      - **Epoch-2 boundary crossed** — captures at gs 60/65/70/75 (epoch-2 ≈ step 58).
      - Matrices span depth × type: layers {6,13,20} attention (q/k/v/o) + layer 13 MLP (gate/up/down).
      - GRPO signals (reward, response_length, entropy, pg_clipfrac, ppo_kl, advantages) present for all 75 steps.
- [ ] **All tensors downloaded to laptop** — IN PROGRESS (rsync of 16 GB to `runs/EXP-38/captures/`).
- [ ] **GPU torn down** — pending full download + completeness confirm.

## ⏸️ HELD for operator command (do NOT start until "go")

- [ ] Offline drift/rank/subspace/periodicity/GRPO-correlation analysis (laptop-only; pipeline written + validated).
- [ ] **Strong standalone HTML report** → `research/reports/comm-eff-grpo/exp38-dense-drift.html`
      (engine `research/scripts/exp38_drift_analysis.py` + `exp38_report.py`, validated end-to-end on real data).
- [ ] Analyst verdict + LOG/SUMMARY/STATUS + draft PR (exp/38 → vast-ai-workload).

## Early signal (from the validated pipeline on partial data — NOT the final analysis)

- Dense gradient is **very low-rank** (rank-for-90% ≈ 22 ≪ r=77).
- Boundary activation `h` is **~rank-1** (top singular dim ≈ 99% energy — a massive-activation channel).
- Boundary `grad_h` is **rank ≈ 124-128 > r=77** — the backward boundary traffic is the hard-to-compress side.

(These are previews to confirm the data is interpretable; the real numbers come from the held analysis.)
