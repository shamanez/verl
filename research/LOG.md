# Research Log (newest first)

## EXP-29 · 2026-06-12T18:35:00+10:00 · M6 · PASS
EXP-29: Anchor on-policy replay — pair the anchor's stale weights with the trajectories those weights generated (+ CPU-resident snapshots, fire-aware ring retention, value-level relevance verification)

- hypothesis: With anchor.replay_paired_batch=true + anchor.snapshot_device=cpu on the locked EXP-27 substrate (powersgd r77 sync_basis, anchor owns_q cadence=5 delay_K=5, clean=0, ef_powersgd clip/decay 0.5), a 25-step smoke (a) passes every correctness invariant, (b) keeps perf/max_memory_allocated_gb < 57.9 with step time within +1.5s of EXP-27, (c) reaches val@25 >= 0.60 with bytes_ratio ~0.0505.
- result: PASS on ALL gates. Canary 20/20 match=True bitwise (push fp32 norm+sum == clone recompute through bf16->cpu->device); 9/9 post-warmup fires exact (used_tick==step-5, batch_gs==snapshot_gs, realized weight delay alternates K/K+1 as derived); isolation unchanged (loads 20/20, coverage set-equal, anchor_optimizer_steps=0, mask delta 0); max_mem 30.77 GB << 57.9 (the ~18.6 GB legacy GPU snapshot ring eliminated); mean step 84.8s vs EXP-27 110.5s (FASTER — one per-gs snapshot replaces per-tick GPU clone); bytes_ratio 0.0504-0.0505; val@25 0.7005 (>=0.60 gate; EXP-27 ref 0.7134 — replay changes the science, not-broken gate only); no E1 (steps 10-25 len/max 1126). Operator scope additions both green: fire-aware retention (only ticks = -delay_K mod cadence stored; ring_batches<=2/ring_snapshots<=2 asserted+observed) and the relevance probe (per-fire masked mean |logp(loaded weights) - rollout_log_probs stored with the replayed trajectories| = 0.0083-0.0105 FLAT across 10 fires incl. 9 distinct snapshots => loaded weights ARE the trajectories' generator at the value level; codec-affected old-vs-rollout ~0.61-0.84 for contrast). 3 iterations on the branch: impl d311904 -> actor.yaml Hydra-struct hotfix 933e79a (first launch died on dataclass<->YAML drift; drift test added) -> fire-aware retention c512128 -> relevance probe 67acf37. Suites at final commit: CPU 195 passed, GPU-box 195 passed. WandB: eyguqjh4.
- key metrics: val@25=0.7005; max_memory_allocated_gb=30.77 (gate <57.9); mean step 84.8s (gate <=112.0); bytes_ratio 0.0504-0.0505; canary 20/20; anchor_replay_fires=10=anchor_backwards=anchor_q_updates; relevance MAD 0.0083-0.0105 flat; 0 OOM/NaN/AssertionError
- PASS = mechanism correct + memory-clean; NO parity/surpass claim (25 steps cannot). Science of self-consistent M -> successor on 50-100 steps with controls.
- code: PR #16 MERGED to vast-ai-workload (d26176b44); branch exp/29-anchor-onpolicy-replay deleted (remote+local+worktree)
- run dir: runs/EXP-29/ · verdict: runs/EXP-29/verdict.md · box i_40676027 HELD WARM (operator-provided; outside auto-teardown)

## EXP-27 · 2026-06-11T04:25:00+00:00 · M6 · STOP
EXP-26.1: REVISE child of EXP-26 — damped ef_powersgd (clip 0.5, decay 0.5) to 100 steps

- hypothesis: On the locked substrate (Qwen2.5-1.5B-Instruct GSM8K vanilla GRPO no-KL/no-entropy, PowerSGD r=77, anchor owns Q, cadence=5, delay_K=5, clean_cadence=0, q_basis=act), damping the ef_powersgd merger to ef_clip=0.5/ef_decay=0.5 run to 100 steps closes the parent's 2.0-pt gap to parity (best val >= 0.7414) with NO length/clip ignition; halving clip+decay caps the residual dose while keeping the direction-preserving correction (parent cos=0.9558) that gained +7.7 pts over plain.
- result: FALSIFIED on BOTH predicate STOP clauses. Damping capped the EF dose (rel_change 0.02-0.19 vs parent 0.30-0.47) and direction was preserved in the healthy phase — yet the arm still ignited at step ~66 (resp_len/mean 171→575 crossing the 509 alarm, max pinned 16384 for steps 61-68, entropy 0.34→0.079) and gained nothing (val@25=0.7134, best val@50=0.7202 <= falsify floor 0.7210). Damping only delayed ignition (~20 steps) without preventing it; the length-explosion is not driven by EF dose magnitude. ef_powersgd lineage terminates (revise cycle 2 of 3); EXP-26's REVISE findings stand as the M6 record. WandB: qa6sll3h.
- key metrics: val@25=0.7134, best val=0.7202@step50 (target>=0.7414; falsify floor<=0.7210 — STOP both ways); EF residual dose peak ~0.189 (capped vs parent 0.47); comm bytes_ratio 0.0505 (~19.8x); resp_len/mean at ignition step66=557.6; entropy at ignition step66=0.079; score during ignition 0.73-0.84 (length-hack, not reward collapse); max_memory_allocated_gb=123.3/~143 (OOM-imminent at kill); 0 NaN/inf
- cell killed at step ~66-68 on confirmed LENGTH_EXPLOSION rescue trigger; val@75/val@100 not measured; EARLY_KILL_LENGTH_EXPLOSION marker written
- run dir: runs/EXP-27/
- verdict: runs/EXP-27/verdict.md

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
