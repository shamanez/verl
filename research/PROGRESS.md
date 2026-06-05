# PROGRESS — append-only audit log

Live: **#25** (RES-133) — realistic anchor-circuit PowerSGD GRPO trainer. One line per action.

[2026-06-05] #25 planned + codex-reviewed; 4 silent landmines hardened into plan (see `runs/UNWANTED_HOOKS_AND_SILENT_FAILURES.md`).
[2026-06-05] #25 R1 (DP-reduce + full-coverage M) / R2 (anchor-owns-Q) / R3 (signed_ema merger) + dead-spectral cleanup built; 128 CPU tests green. Provisioning blocked by Vast SSH key-injection bug (operator-owned).
[2026-06-05] #25 consolidated onto `vast-ai-workload` (`exp/25-anchor-default` merged `--no-ff` + deleted); doc/flag cleanup pushed; stale run scaffold deleted → re-materialize from plan.
[2026-06-05] #25 ready to resume on existing box 39602487 (4×H200) — verify SSH, reuse, don't provision.
[2026-06-06T00:47:43+10:00] [orchestrator] #25 resumed on operator warm box 39613656 (4×H200, direct -p 40872 root@46.243.55.155). Box synced vast-ai-workload@107ca01 (R1/R2/R3 present, editable install), secrets.env pushed (HF+WANDB), RUNNING ledger row registered (budget clock start, 12h headroom). Dispatching experiment-runner for id-0/id-1 probes + α sweep. DO-NOT-PROVISION.
[2026-06-06T00:50:39+10:00] [orchestrator] tick: running=[25] analyzing=[] logging=[] blocked=[24 dep]. experiment-runner (bg) driving id-0/id-1 probes + α sweep on reused box 39613656.
[2026-06-06T01:01:46+10:00] [orchestrator] id-0 probe (exp25_id0_anchorM) CRASHED on first launch — 2 backend bugs caught from log: (A) device-mismatch cuda/cpu in powersgd_activation.orthonormalize:127 degenerate-col repair (q_fix on CPU vs q on GPU); (B) anchor staleness off-by-one transformer_impl:1262 (step>=delay_K should be > ; step0 never pushed). Diagnosis+exact fixes sent to experiment-runner for on-box hotfix. Flags all verified correct in launcher banner.
[2026-06-06T01:08:38+10:00] [orchestrator] id-0 RE-PROBE post-hotfix: anchor invariants GREEN — coverage 196/196 set_equal=True, M-dp cross_rank_dev=0, dp-reduce MEAN (ratio~0.71-0.79 not 4x), anchor-load 338/338 canon-matched, ||dM||>0, anchor clean (ratio=1/0-opt/0-mask/clone-iso/clean_pg), staleness realized_delay=1 from step2. Both hotfixes verified working. Box uncommitted diff backed up locally (BACKUP-uncommitted-box-diff.patch). Watching id-0 completion + id-1 launch.
[2026-06-06T01:13:14+10:00] [orchestrator] id-0 PASS: 3/3 steps complete (step:3 @line1463), done.flag written, NO 'Error executing job' marker (post-step tracebacks = benign DataLoader/UnixTransport teardown noise). All anchor-M invariants green. id-0 gate CLEARED → id-1 (R2/R3) next. NOTE: post-completion teardown tracebacks are benign; real crash = 'Error executing job'/'RayTaskError'.
[2026-06-06T01:28:49+10:00] [orchestrator] id-1 PASS: R2 anchor-owns-Q broadcast verified — Q updated+broadcast every refresh (anchor_q_updates 1→6, changed>0), cross_rank_max_rel_dev=0.0, fast net powersgd_basis_updates=0 (never updates Q), M bcast to 196 all ranks; 3/3 steps, done.flag, no crash. COLD-M fallback deferred to α=0 arm (cadence=1 warmed M too fast in probe). Runner died on API-529 after id-0; orchestrator took over directly. id-2 α-sweep LAUNCHED (tmux exp25-sweep, arm α=0.0 first).
[2026-06-06T02:13:12+10:00] [orchestrator] tick: running=[25] analyzing=[] logging=[] blocked=[24 dep]. id-0+id-1 probe gates PASS; id-2 α-sweep LIVE on pinned box 39613656 (α=0.0 @step20/50, healthy: 4×100% GPU, grad_norm 3.49, no errors). Sweep auto-chains α=0.3→0.5 in tmux exp25-sweep. training-log-monitor dispatched (bg). Issue #25 status:approved→status:running.
[2026-06-06T02:38:10+10:00] [sync] EXP-25 box drift detected (powersgd_activation.py, transformer_impl.py) but local vast-ai-workload already has these changes committed — no new commit needed. Patch refreshed (38 lines, no delta). Local ahead of origin by 3 commits (autosave loop handles push).
[2026-06-06T02:40:59+10:00] [sync] EXP-25 box drift (powersgd_activation.py, transformer_impl.py) unchanged from last sync — local already up-to-date, no commit.
[2026-06-06T02:50:40+10:00] [sync] no box source drift (diff unchanged from prior tick — powersgd_activation.py + transformer_impl.py still same hotfix, already local-committed)
[2026-06-06T03:00:33+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[EXP-25] monitor-session: arm exp25_alpha_0p0 at step 40/50 @17:00Z, HEALTHY, val@25=0.7180, grad_norm finite, anchor_backwards=16/expected, no errors — continuing
[2026-06-06T03:01:21+10:00] [orchestrator] EXP-25 UNIT NUANCE confirmed (operator-flagged): anchor.cadence/delay_K count OPTIMIZER-TICKS not global steps (base.py:177 one opt-step/train_batch; engine_workers.py:282-303 per-minibatch; counter transformer_impl.py:1182/1677). batch128/mini64=2 ticks/step ⇒ cadence5/delayK5 = refresh every ~2.5 global steps + 2.5-step staleness (verified anchor_q_updates=14@gstep37). Held-fixed across all arms ⇒ NO α-sweep confound, NOT a bug. NOT changing mid-run; documented for analyst (staleness=~2.5 steps, comm ~2× worse than 5-step assumption) + memory + plan. 4-circuit deep audit: BOTH circuits CLEAN.
[2026-06-06T03:10:43+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T03:20:32+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T03:30:37+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T03:40:34+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T03:50:33+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T04:00:12+10:00] [orchestrator] ENTROPY COLLAPSE found in EXP-25 α=0.0 arm (operator-flagged): entropy 5.69→0.06 monotonic, response_length explodes ~step30 (300→8600), reward peaks 0.79@28 then degrades to 0.32@45. Prime suspect = α=0 signed_ema |G|·sign(M) = magnitude-preserving sign-SGD w/ persistent EMA signs (no cancellation→sharpening), no-KL/no-entropy to arrest. Isolated by 4 anchor-OFF control runs (no collapse). Dedicated team 'entropy-collapse' (member entropy-analyst) writing runs/EXP-25/ENTROPY_COLLAPSE_FINDINGS.md + standing research/diagnostics/ENTROPY_COLLAPSE_WATCH.md. Prediction: α=0.3/0.5 arrest it (sweep tests H1). Entropy/length/IS now a STANDING per-run watch.
[2026-06-06T04:00:34+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)

## EXP-25 Monitor Session 2 (2026-06-05 17:15-18:05 UTC) — arm0 COMPLETE

- **exit_state**: DONE_1FLAG (exp25_alpha_0p0.arm-done written, arm1 starting)
- **val@25**: 0.7180 | **val@50**: 0.3541 — STOP (threshold ≤0.7114 triggered)
- **Errors**: 0 training errors; 5 benign post-teardown RuntimeError matches
- **Artifacts**: runs/EXP-25/logs/train.log (1994179b), exp25_alpha_0p0.arm-done
- **arm1 (α=0.3)**: started, loading weights at session end (cold-M check pending)
- **Recommendation**: STOP for α=0.0; dispatch analyst + dispatch new monitor for arm1 cold-M check
[2026-06-06T04:09:43+10:00] [orchestrator] EXP-25 α=0.0 DONE: val@50=0.354 — CATASTROPHIC CRASH from val@25=0.718 (≪ floor 0.6914, ≪ STOP 0.7114). Confirms entropy-collapse analysis: α=0 |G|·sign(M) sign-SGD destroyed the policy. .arm-done written, reached step50. Sweep HEALTHY + chaining: α=0.3 LAUNCHED (train.log live 18:08Z); α=0.5 pending. Member's 'α=0.3 missing' was WandB lag. Decision: let sweep run to completion (experiment-failure=keep box; need α=0.3 severity-order + α=0.5 knee test). α=0.5 is decisive for #25 PASS/STOP. Re-arming monitor w/ entropy T1-T7 panel on α=0.3.
[2026-06-06T04:10:34+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)

## 2026-06-05T18:12:58Z EXP-25 monitor: α=0.0 DONE val@50=0.3541 (vs val@25=0.7180, STOP-zone), α=0.3 launched step3 cold-M=0 (guard ok). GPU healthy, no errors. [continue_in_place_iteration]
[2026-06-06T04:20:37+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T04:30:36+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T04:38:00+10:00] [orchestrator] EXP-25 α=0.3 (step18/50): collapse ARRESTED — NOT like α=0. Entropy floor ~1.9 (oscillating, bounced @s16), reward RISING to 0.728, response_len SHRINKING 282→187 (clip_ratio=0, no length-hack), rel_change median 0.99 (vs α=0's √2). Cold-M guard fired 196→196→0 ✓. Confirms (2α−1) mechanism; phase transition is BELOW α=0.3 (strict 'still collapses slower' prediction refuted in the favorable direction). CAVEATS: only step18 (α=0 cascaded in epoch2 ~s25-30 — late-cascade watch on), entropy floor 1.9 < dense ~2.5 so may underperform→REVISE. val@25/@50 PENDING. Re-dispatched monitor for α=0.3 val + cascade-watch + α=0.5 (predicted near-inert/cleanest). Box healthy ~5GPU-hr/48.
[2026-06-06T04:40:36+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T04:50:33+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T05:00:33+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
[2026-06-06T05:10:35+10:00] [sync] no box source drift (diff unchanged — same 2 hotfix files, already local-committed)
