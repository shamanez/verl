# Runs Summary (newest first)

Concise index of completed experiments. Full per-run detail in `runs/<ID>/`, the formal log in `LOG.md`, and analysis in the research-repo issues.

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
