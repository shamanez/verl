# Research Status — 2026-06-04

## Active
Nothing running. **0 live Vast instances** ($0 idle, verified).

## Latest — EXP-20 (M6) · DONE · PASS
PowerSGD-style PP activation compression vs PRF mask. Qwen2.5-1.5B-Instruct + GSM8K, vanilla GRPO (no-KL/no-entropy), `clean_cadence=5`, 50 steps, 4×H200. Codec is the only axis (`runs/FIXED_CONTROL_SURFACE.md`).

| arm | codec | val GSM8K acc@50 |
|---|---|---|
| dense (control) | comm-eff OFF | **0.7536** |
| PowerSGD r=102 (+33% budget) | powersgd | 0.7437 |
| PowerSGD r=77 (byte-matched) | powersgd | 0.7415 |
| mask p=0.95 | prf_mask | 0.7384 |

PowerSGD ≥ mask at **equal** communication budget (un-caveated, r=77). Dense ~1–1.5 pp above all compressed → a small compression tax. Codec PR shamanez/verl#13 **merged** → vast-ai-workload.

## Open issues (research repo)
- **#21** — how RL/GRPO grads behave under PP compression (EXP-20 analysis; "is it just the 10 clean steps?" → **No**, compressed steps carry 57–95% of the gain; + dense results appended). `kind:experiment`, M6.
- **#22** — clean-step realism confound: does it survive WITHOUT a fresh full-grad refresh? (amortized comm is ~4× not 20×; test staleness/`delay_K` + cadence→∞ + error-feedback + a harder task). `kind:experiment`, M6. **UNCLAIMED** — flip to `research:claim` to queue.
- **#19** — M5 epic, unclaimed.

## M6 progress
EXP-20 PASS (1 of ≥2 for the milestone summary). Likely next: the #22 staleness/clean-cadence sweep.
