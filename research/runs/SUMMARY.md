# Runs Summary (newest first)

Concise index of completed experiments. Full per-run detail in `runs/<ID>/`, the formal log in `LOG.md`, and analysis in the research-repo issues.

## EXP-23 (M6) — Stale full-grad re-anchor for PowerSGD (delay_K=5 · inject/blend) · STOP (FALSIFIED) · 2026-06-05
Follow-up to EXP-20: can a **stale** full-gradient re-anchor replace the fresh `clean@5` step for PowerSGD r=77? One warm 4×H200 box, 3 arms on the fixed surface, codec held at PowerSGD r=77, fresh clean step OFF.

| arm | refresh mechanism | val GSM8K acc@50 |
|---|---|---|
| A0 fresh-clean@5 (EXP-20 ref, not re-run) | fresh full-grad every 5 | 0.7415 |
| A1 no-refresh (floor) | none | **0.6914** |
| A2 stale inject (γ=1.0) | stale anchor + additive inject | 0.6967 |
| A3 stale blend (η=0.5) | stale anchor + convex blend | 0.6861 |

- **FALSIFIED:** `max(A2,A3)=0.6967 ≤ floor+0.02=0.7114`; neither additive mode recovers the ~0.05 fresh-clean benefit. A1 sits 0.05 below A0 → refresh *does* matter for r=77, but a stale one can't supply it.
- **Root cause (measured, all 50 steps, both arms):** `cos(G_powersgd, M_anchor) ≈ 0.001` — the rank-77 compression subspace is ~orthogonal to the stale full-grad *by construction* (10× more orthogonal than the mask's ~0.5). Inject adds an orthogonal vector (grad-clip dilutes it); blend just shrinks the step to `√0.5·‖G‖`. Integration was clean (circuits fired `anchor_backwards=20`/`spectral_corrections=80` per arm, codec green `q_cond≈1`/`q_cross_rank=0`, 0 NaN/OOM) — a decisive negative result, not a failed run.
- **ERRATUM (2026-06-05): the anchor results here are CONFOUNDED by a bug.** `M_anchor` was EMA-updated for only 4/196 targets (`max_targets=4`, layer-0 attention) AND was a per-rank local-shard gradient (no DP all-reduce), so the "stale full gradient" EXP-23 tested was a tiny slice, not a true full global gradient. The empirical inject/blend ≈ floor result holds for the AS-IMPLEMENTED anchor, but the `cos≈0.001` orthogonality is NOT a clean test of a correct stale anchor. Issue #25 fixes both bugs (full coverage + global DP reduce) as hard gates and re-tests. See `runs/EXP-23/verdict.md` ERRATUM.
- **Next lever (issue #24):** error-feedback on the PowerSGD residual + basis-aligned anchor (NOT `delay_K` — orthogonality won't yield to smaller K). Launcher-wiring PR shamanez/verl#14 **merged** → vast-ai-workload (squash `9edea6105`). Full analysis: `runs/EXP-23/verdict.md` + `runs/EXP-23/stale_gradient_research/STALE_GRADIENT_ALTERNATIVES.md`.

## EXP-20 (M6) — PowerSGD-style PP activation compression · PASS · 2026-06-04
Qwen2.5-1.5B-Instruct + GSM8K, vanilla GRPO (no-KL/no-entropy), `clean_cadence=5`, 50 steps, 4×H200. Codec is the only axis (fixed surface: `runs/FIXED_CONTROL_SURFACE.md`).

| arm | codec | val GSM8K acc@50 | wandb |
|---|---|---|---|
| dense control | comm-eff OFF | **0.7536** | `5e2jpho9` |
| PowerSGD r=102 (+33% budget) | powersgd | 0.7437 | `kqozxfr0` |
| PowerSGD r=77 (byte-matched) | powersgd | 0.7415 | `oquyeic3` |
| mask p=0.95 | prf_mask | 0.7384 | `3yxzzwn3` |

- **PowerSGD ≥ PRF mask at equal communication budget** (un-caveated, r=77); spread across the 3 compressed arms = 0.53 pp.
- **Dense ~1–1.5 pp above all compressed** → a small but consistent compression tax (measured with the `clean@5` crutch).
- Within each compressed run, **compressed steps carry 57–95% of the reward gain** — not the 10 clean steps; the clean step is a small full-rank bias-flush.
- Codec correctness: cross-DP consensus basis bit-identical (`q_cross_rank=0.0`), recon → ~0.02 in ~9 steps, no NaN/OOM.
- Codec PR shamanez/verl#13 **merged** → vast-ai-workload. Analysis: issue **#21**. Follow-up (clean-step realism/staleness): issue **#22**.

_Prior experiments (EXP-9/13/14/15/16/17/18/21) — see `LOG.md` + git history; their `runs/` dirs were de-bloated._
