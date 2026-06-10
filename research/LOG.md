# Research Log (newest first)

## EXP-26 · 2026-06-10T11:08:00+10:00 · M6 · PASS-STAGE-A (Step-A diagnostic gate only; Steps B/C/E pending)
EXP-26: Diagnose the SFT→GRPO merger mismatch with a real-gradient geometry audit, then test direction-preserving, RLVR-native compression — #25 follow-up

- hypothesis: The #25 signed_ema lag (0.047 below dense) is caused by the merger corrupting the live GRPO update direction, not by rank-77 PowerSGD compression; a direction-preserving ef_powersgd merger recovers val@50 >= 0.7414.
- result: PASS-STAGE-A (Step-A gate cleared). H3 CONFIRMED (sign-agreement ∈ [0.50, 0.52] at delay_K∈{0,5} — coin-flip even fresh → sign-replacement structurally unrecoverable; corroborates EXP-25 STOP). H1 confirmed in spirit via confound-free merger isolate: cos(G_comp,G_corr)=0.717 (signed_ema rotates the compressed update ~44 deg). H2 TRUE (Q_act activation capture 0.9985 PASS, but update-energy capture only 0.318 — off-principal share 0.682 → Q_act misses ~68% of GRPO update energy). Option-A validity: cos(G_fresh_anchor,G_dense)=0.985. DECISION=go_C_then_B + retire_sign_replacement(confirmed). 7 capture-instrumentation bugs fixed on exp/26-geometry-audit-ef-powersgd @ 5a35fa96c. Steps B/(C)/E deferred to a new session; box 40242796 preserved warm.
- key metrics: cos(G_comp,G_corr)=+0.717 (merger rotation ~44 deg); sign-agree A1@K0=0.500 / A2@K0=0.523 / A2@K5=0.520; Q_act update-capture=0.318 (off-principal 0.682); activation-capture=0.9985; validity cos(G_fresh_anchor,G_dense)=0.985; fp32 dump fidelity max_recon_drift=4.5e-5
- decision: go_C_then_B — Step C (rlvr-native Q-content sweep at fixed rank 77) runs BEFORE Step B (ef_powersgd) because H2 could not be shown false
- branch: exp/26-geometry-audit-ef-powersgd @ 5a35fa96c (7 hotfix commits; pushed)
- STAGE GATE ONLY — no PR drafted, no launcher promoted (promote_launcher_as=none; ef_powersgd method not yet validated; Steps B/C/E pending)
- run dir: runs/EXP-26/
- verdict: runs/EXP-26/verdict.md
- step-a decision: runs/EXP-26/stepA_decision.md

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
