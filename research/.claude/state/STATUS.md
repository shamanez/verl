# Research Status — 2026-05-29 (EXP-14 CLOSED)

## EXP-14 — CLOSED (status:pass), box torn down
- Diagnosis complete: masked-GRPO grad_norm explosion ROOT CAUSE = mask magnitude collapse (h*mask no rescale → 0.32× RMS → OOD). rescale (test2_cellD) tames grad_norm 771→1.49.
- ⚠️ CORRECTION (operator learning-check): rescale is NOT a training fix — test2_cellD val acc FLAT 0.080→0.084, entropy frozen ~5.9 (vs dense 0.083→0.721, clean_cadence=2 0.080→0.672 in the same 10-step window). Pure masked GRPO at p=0.9 does NOT learn, w/ or w/o rescale (masked forward = near-random surrogate). Only clean_cadence learns, via full-bandwidth clean steps. Posted as correction to #14 + reframe to #15 (real bar = val/score learning, not grad_norm; needs a mask-rate sweep).
- Report posted to GitHub #14 (CLOSED); untested follow-ups (per-channel cellG validation, longer-horizon convergence, anchor+spectral, comm-eff savings) → GitHub #15.
- 8 cells run (wandb exp14-test1_cellA/B, test2_cellA/B/C/D/F, test3_cellB); metrics in runs/EXP-14/metrics/; verdict runs/EXP-14/verdict.md.
- Branches pushed: exp/14-clean-cadence (8c8196b4), exp/14-mask-pertoken-rescale (905f4742), exp/14-mask-per-channel (a3590173), exp/14-mask-consistent-rng (49363ca4).
- Instance 38370788 TORN_DOWN (session-complete-exp14). Ledger clean.
- BRANCH-HYGIENE NOTE: research/ state currently lives on exp/14-clean-cadence (worktree isolation didn't persist this session); reconcile onto vast-ai-workload as a cleanup before the next cycle.

---

## (prior) Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 14 | Comm-eff grad_norm explosion: peel-ablation + clean-step fix (clean_cadence) | RUNNING (relaunched) | 1×4H200 (i_38370788) @ $14.74/hr | — | OPERATOR CAP 10 steps/cell; bug fixed (global_steps collision → comm_eff_global_step @5ac7f9ea); fresh chain ALIVE cell1 step1/10; monitor-exp14b (bg); ledger RUNNING |
| 11 | M3 — 100-step M95+AP vs dense baseline | BACKLOG | — | — | M3, no status label + no plan; not eligible (awaiting triage/claim) |
| 10 | M3 — DP gradient compression (PowerSGD + Streaming-DiLoCo) scope | BACKLOG | — | — | M3, no status label + no plan; depends on M95+AP smoke |
| — | baseline (dense GRPO control) | DONE | — | PASS | permanent reference; LOG.md + runs/SUMMARY.md |
| — | communication-baseline (comm-eff smoke) | DONE | — | PASS | permanent reference; LOG.md + runs/SUMMARY.md |

## Last tick
2026-05-29T20:05 · running=[14 (test2_cellC root-cause-fix validation)] · analyzing=[deferred until cellC] · logging=[] · blocked=[] · skipped=[10,11]

### EXP-14 test1→test3 chain DONE (all cells clean, zero FSDP/NaN/OOM)
- TEST 1 GATE PASS: test1_cellA gn s1=0.3506 (≈dense 0.36), score 0.14→0.73; test1_cellB gn=0.3371 (≤1.0 scaffold-clean). 10/10 each.
- TEST 2 PEEL — explosion confirmed (mask-only): test2_cellA gn 771→838 all 10 steps, entropy 6.35, score flat (no learning). test2_cellB (recompute=false) gn ~53, ppo_kl 11.6 (ratio corrupted). Both ≫10, both FSDP-clean, neither learns.
- TEST 3 FIX (clean_cadence=10) WORKS: test3_cellB masked steps 1-9 gn 641-810; step-10 CLEAN step gn=0.472 (1700× lower), clean_steps=1, ppo_kl=5.0, FSDP-clean.

### ROOT CAUSE found + fixed (mask-investigator, branch exp/14-mask-consistent-rng @49363ca4, pushed)
- Bug: mask PRF key folded in a per-forward `substep` counter that never reset → old_logprob & train forwards at the same global_step drew DIFFERENT masks → corrupted IS ratio → the 771 explosion. Explains cellA(771, both masked diff substep) vs cellB(52.8, old clean).
- Fix: knob `comm_eff.mask.consistent_across_forwards` (default true) holds substep fixed → identical mask across all forwards of one update. 6/6 new unit tests + 100/100 comm_eff suite. YAML mirrored+roundtrip-verified.
- Audit items 1-3 ALL CLEAN: anchor inert (transformer_impl.py:791); NO FSDP hook / NO optimizer coupling (pure simulated PP, 1 optimizer step/trainer step); mask h*mask no-rescale, boundaries [1,3,5,7,9,11,13].

### Mask-only stabilization sweep (operator: stabilize mask-only BEFORE anchor+spectral; box kept WARM)
- test2_cellC (consistent_across_forwards=true): step-1 gn=881.9 ≈ cellA 771 → substep/IS theory REFUTED. (10-step trajectory rsynced.)
- ✅ test2_cellD (mask.rescale=true) — BREAKTHROUGH: step-1 gn=**1.491** (~550× collapse to dense order), ppo_kl~0, stable, zero FSDP/NaN. ROOT CAUSE = mask magnitude-collapse bias; inverted-dropout h*mask/(1-p) is the fix. Branch exp/14-mask-pertoken-rescale@905f4742. (Nuance: entropy still ~5.9 at step3 → stable but convergence needs a longer run.)
- ⏳ test2_cellE (queued): per-token-consistent mask (the operator-MUST default, investigator implementing) + rescale=true = canonical mask-only default candidate.
- ⏸ test2_cellF (clean_cadence=2): SUPERSEDED by rescale (held; rescale fixes the root cause with full bandwidth savings, no clean-step cost).
- Operator directives logged: per-token mask consistency = MUST + default for all experiments; settable mask.seed; anchor+spectral OFF until mask-only stable.
- Box 38370788 RUNNING, kept warm for cellE; analyst (full set) + teardown after the sweep. cellD finishing its 10 steps.

## EXP-14 detail
- Instance: 38370788, 4×H200 (143 GB ea), tier 0, $14.74/hr (cap $24), reliability 0.988. SSH 156.19.254.2:33404 · tmux exp-14-156_19_254_2 (recreated 09:05:36Z, ALIVE).
- OPERATOR CAP (2026-05-29): EVERY cell = 10 trainer steps (was 25/25/10/10/100). NO 100-step run. Proven via resolved set -x trace (total_training_steps=10, progress 1/10). WandB names exp14-<test>_<cell>.
- BUG FOUND + FIXED mid-run: original chain crashed every cell (rc=1, ~10 min) on a `global_steps` meta-key collision — clean_cadence patch stamped batch.meta_info["global_steps"] but vLLM rollout already emits global_steps as a batch column → to_tensordict() assert. Fixed by renaming threading key to comm_eff_global_step (emit ray_trainer.py + read engine_workers.py). Commit 5ac7f9ea, pushed origin/exp/14-clean-cadence, exp.bundle regenerated. Crash signature now 0 occurrences.
- Branch: exp/14-clean-cadence (tip 5ac7f9ea). MAIN CHECKOUT IS ON THIS EXP BRANCH (worktree isolation didn't persist); plan 14.md restored from vast-ai-workload. RECONCILE branch/origin divergence + return checkout to vast-ai-workload AFTER run completes.
- Cell chain (sequential, fanout=1, shared docker/verl/dataset cache), ALL 10 steps: test1_cellA(KL-on dense ref) → test1_cellB(KL-off scaffold-noop) → test2_cellA(mask recompute=true) → test2_cellB(mask recompute=false) → test3_cellA REUSES test2_cellA → test3_cellB(mask + clean_cadence=10; single clean step @ step10). Test 4 conditional, not launched.
- Liveness post-relaunch: cell1 (exp14-test1_cellA) training step1/10, actor/grad_norm logged, jsonl step-0(val)+step-1(train) rows, zero FSDP/OOM/Ray/collision matches. Stale prior-run artifacts cleaned off box.
- Budget caps: max_gpu_hr=96, max_dph=24.0. Stop hook backstop (verdict / >30min stale / budget / PROVISIONED-stale).

## Rescue-trigger watch (operator surfaces; none fired yet)
- FSDP/DTensor backend error on any cell (esp. test3_cellB, where zero is HARD) → codex-verify --mode code-rescue.
- OOM on a method cell → cause-G headroom miscalc (lean cells should NOT OOM on 4×H200).
- Test 1 Cell B grad_norm>1.0 → corrective backend patch (re-run gate before method tests).
- Test 3 clean step "no better than Cell A" → Phase B (Test 4) or shorter cadence (REVISE).

## Budget
$/hr now: $14.74 (one box) · within max_dph $24/instance · max_gpu_hr=96 cap (Stop-hook backstop)
