# Research Log (newest first)

## EXP-25 · 2026-06-06T09:22Z · M6 · STOP
EXP-25: Make the ANCHOR CIRCUIT the default for comm-efficient RL (stale-M + sign-based grad merger + move Q ownership to the anchor) — prerequisite for #24

- hypothesis: On the fixed GSM8K surface, the anchor-default substrate (full-coverage stale-M R1 + anchor-owned Q R2 + signed_ema merger R3, α swept) recovers the comm-efficiency gap that EXP-23 inject/blend could not.
- result: FALSIFIED. Best-α=0.5 val@50=0.7066 ≤ floor+0.02=0.7114 (STOP threshold). Dose-response is monotonic: α=0 val@50=0.354 (catastrophic length-explosion collapse), α=0.3 val@50=0.616 (delayed collapse), α=0.5 val@50=0.7066 (stable but below target). Root cause: signed_ema sign-reversal (`|G|·sign(M)`) acts as magnitude-preserving sign-SGD; stale-anchor signs disagree with live grad on ~50% of coords each step (warm rel_change median ≈ √2), inducing a persistent policy-sharpening pressure that drives response-length explosion and entropy collapse in low-α arms. The signed_ema correction primitive is net-harmful across the entire swept grid.
- probe gates: id-0 (anchor M / R1) PASS + id-1 (anchor-owns-Q R2 + signed_ema R3) PASS — both hard invariant sets green; implementation is correct; this STOP is a training-dynamics result, not a broken-code artifact.
- dose-response: α=0.0 val@50=0.354 · α=0.3 val@50=0.616 · α=0.5 val@50=0.7066 (vs dense 0.7536, A0 fresh-clean 0.7415, no-refresh floor 0.6914)
- cross-issue: #24 stays BLOCKED — depends_on #25 PASS, which did not occur; correction primitive must be redesigned before #24 spends compute.
- NO PR drafted — code_change=true but verdict=STOP; PRs are only opened on PASS.
- run dir: runs/EXP-25/
- verdict: runs/EXP-25/verdict.md
- deep findings: runs/EXP-25/DEEP_FINDINGS.md
- standing entropy watch: research/diagnostics/ENTROPY_COLLAPSE_WATCH.md (T1–T7 triggers, reusable on every future run)

## In-container hotfixes
The following patch files were captured from on-box commits and are stored under `runs/EXP-25/hotfix-patches/`. Apply with `git am` onto `vast-ai-workload` before deploy (these fixes were already merged to `vast-ai-workload` via the autosave loop; the patch is a backup):
- `BACKUP-uncommitted-box-diff.patch` — device-mismatch fix (powersgd_activation.py orthonormalize CPU/GPU) + anchor staleness off-by-one (transformer_impl.py step>=delay_K → step>delay_K)
