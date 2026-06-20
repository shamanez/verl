# EXP-38 — CHECKPOINT  ·  dataset = **GSM8K**  ·  analysis HELD for operator go-ahead

> **This run is GSM8K ONLY.** Model Qwen2.5-1.5B-Instruct, accel surface, dense GRPO
> (comm_eff OFF), 75 global steps. A planned **Big-Math** sibling run is a SEPARATE
> experiment in a SEPARATE session — see "Two-dataset plan" at the bottom.
> **The two datasets must never be mixed.**

---

## ✅ FINISHED

| # | Step | Evidence |
|---|---|---|
| 1 | **Instrumentation built + pushed** — two default-OFF diagnostic gates (weight/grad/optimizer **drift probe** + capture-only **boundary-activation probe**) across 6 verl files + Hydra schema | branch `exp/38-dense-drift-probe` @ `ef0a3e7` on `shamanez/verl` (origin) |
| 2 | **Off-parity gate PASS** — both flags OFF == byte-identical dense GRPO | comm_eff counters all 0, no drift dir, no `[exp38]` lines, healthy loss/grad_norm |
| 3 | **On-path validated** — all 5 capture roles fire; FSDP summon adds no net HBM (35.6 ≤ 36.3 GB); `grad_h` via tensor-hook works under grad-checkpointing; **zero "capture skipped"** all run | 4-step ON validation + full run logs |
| 4 | **75-step dense capture COMPLETE** | val@25/50/75 = **0.7521 / 0.7665 / 0.7688** (monotonic; @25/@50 match EXP-37D dense → probes confirmed measurement-only) |
| 5 | **All artifacts synced + verified local** | **1071/1071 `.pt`, 0 missing, 16.15 GB**; manifest cross-check clean; integrity spot-loaded (5 roles × early/mid/late ticks) |
| 6 | **Sufficiency for feature-work analysis CONFIRMED** | lags k∈{1,2,5,10,20,40} all sampleable (**k5: 2 · k20: 13 · k40: 11 pairs**); epoch-2 crossed (gs 60/65/70/75); 15 matrices × 3 boundaries |
| 7 | **WandB complete** | run `mhegvmbs`; step-75 backfilled past async tail-drop |
| 8 | **GPU torn down** | team box `41763713` **API-verified destroyed (0 live)**; ledger `TORN_DOWN` |
| 9 | **Analysis engine written + validated end-to-end on real data** (dataset-tagged) | `research/scripts/exp38_drift_analysis.py` + `exp38_report.py` |

## ⏳ TO DO NEXT (held — do NOT start until operator says "go")

1. **Run the offline analysis** (laptop-only, free, ~3 min): drift / cosine-sign-normratio vs k / gradient effective-rank-over-time / boundary low-rank + subspace-overlap-vs-lag + periodicity / boundary grad_h rank / GRPO-signal correlations / locate the gradient-anchor + activation-codec staleness knees vs k≈5, k≈20, epoch-2.
2. **Produce the strong standalone HTML report** → `research/reports/comm-eff-grpo/exp38-dense-drift-gsm8k.html` (embedded plots, all deliverable questions, next-method recommendation).
3. **Analyst verdict** (`runs/EXP-38/verdict.md`) → **log-writer** (LOG.md / SUMMARY.md / STATUS.md) → **draft PR** (`exp/38` → `vast-ai-workload`).
4. *(operator's new idea)* Repeat the whole thing on **Big-Math** in a new session, then a **joint two-dataset report**.

### To run step 1+2 when unblocked (single command, dataset auto-detected from DATASET.json):
```bash
cd /Users/shamane/Documents/verl/research
python3 scripts/exp38_drift_analysis.py runs/EXP-38
#   → writes research/reports/comm-eff-grpo/exp38-dense-drift-gsm8k.html
#   → writes research/reports/comm-eff-grpo/exp38-dense-drift-gsm8k_findings.json
# GRPO signals are auto-read from runs/EXP-38/sidecar_grpo.jsonl (already fetched).
```

---

## 📂 WHERE THE DATA IS (crystal clear — all on THIS laptop)

Root: `/Users/shamane/Documents/verl/research/runs/EXP-38/`

| Path | What | Size |
|---|---|---|
| `captures/rank0/manifest.jsonl` | manifest — 1 row per dumped tensor (role, target, tick, global_step, shape, norm, path) | 1071 rows |
| `captures/rank0/tick_<gs>_<tick>/<role>/<name>.pt` | the fp32 tensors: roles `theta` / `g_dense` / `update_vector` (15 matrices) + `boundary_h` / `boundary_grad_h` (3 boundaries) | **16.15 GB** |
| `captures/sidecar_layernorms.jsonl` | per-layer weight+grad L2 norms, all 28 decoder layers (future-research bundle) | 21 rows |
| `sidecar_grpo.jsonl` | per-step GRPO signals (reward, response_length, entropy, pg_clipfrac, ppo_kl, advantages…) from WandB | 75 rows |
| `train.log` | authoritative training log — all 75 steps, resolved Hydra config, every scalar | 514 KB |
| `box_run/` | provenance copy of the on-box run dir (train.log, done.flag, boot.log) | small |
| `handles/41763713.json` | the (now destroyed) Vast box handle | — |
| `DATASET.json` | **the dataset stamp = gsm8k** | — |
| `CHECKPOINT_STATUS.md` | this file | — |

**⚠️ The 16 GB `captures/` are gitignored (local-only) and are now the SOLE copy** (the box is
destroyed). They live only at the path above. Do **not** `git clean -fdx` this tree, and consider a
backup before the analysis if the laptop is at risk. Everything else (scripts, this doc, ledger,
PROGRESS, the `exp/38` branch) is in git.

WandB (cloud, always available): `shamanework-pl/verl_compression_research_accel_rebaseline/runs/mhegvmbs`.

---

## 🧪 TWO-DATASET PLAN (operator's idea — keep STRICTLY separate)

- **GSM8K** — `openai/gsm8k` — **DONE (this run, EXP-38)**. Data under `runs/EXP-38/`, tagged `DATASET.json:gsm8k`.
- **Big-Math** — `gshasiri/Big-Math-RL-Verified-filtered` — **TODO, in a NEW session**: re-run the same
  75-step dense drift+boundary probe on Big-Math, download all its data into a **SEPARATE** experiment dir
  (e.g. `runs/EXP-39/` with its own `DATASET.json:big-math`), then run the dataset-tagged analysis.
- **Joint analysis** — once both exist, produce a combined/comparative report. The analysis engine stamps
  the dataset on the report title, header badge, findings JSON, and output filename
  (`exp38-dense-drift-<dataset>.html`) so the two are **never confused**.
- **HARD RULE (operator): never mix the two datasets' tensors or curves.** Separation is enforced at the
  run-dir level (one experiment dir per dataset) + the `DATASET.json` stamp + dataset-tagged outputs.
