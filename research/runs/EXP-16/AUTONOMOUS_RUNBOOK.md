# EXP-16 autonomous runbook (operator away — 2026-05-30)

Operator directives (this session):
1. Use the temporary box, NOT Vast provisioning. **Nothing without grad rescaling.**
2. New WandB project: **`exp16_rescale_matrix`** (entity `shamanework-pl`).
3. Maximize GPU utilization (`use_dynamic_bsz=True` + big `max_token_len`), **no OOM**.
4. After ALL experiments done: **push everything → tear down → PR to `vast-ai-workload`**.
5. Never let GPUs sit idle; keep the loop running.

## The box (operator-provided, MANUAL lifecycle — NOT in runs.jsonl)
- `ssh -i ~/.ssh/vast_ai -p 33732 root@3.144.230.17 -L 8080:localhost:8080`
  (key `~/.ssh/vast_ai_name` also works.)
- Vast instance **38454090**, 4×B200 (183 GB each), $25.84/hr. cuda-13.2.0 base (no docker).
- verl checkout: `/workspace/verl` on branch `exp/16-short-run-stability-matrix` @ `c47e210c`
  + uncommitted `rescale_mode` patch (4 files: actor.yaml, activation_mask.py, state.py, comm_eff.py).
- Run root on box: `/workspace/runs/EXP-16`. Data prestaged at `~/data/gsm8k/`. flash_attn 2.8.3.
- The stale `23.127.144.217:12276` string the operator first gave is a DEAD host — ignore.

## What's running
- `tmux exp16-rescale` → `run_rescale_sequence.sh 2` → cells **2 → 4 → 5 → 6** (launch.sh; skips removed
  no-rescale cells 1,3). Sources `perf.env` (dynamic_bsz + 98304 tok/GPU). Stops on any hard failure.
- `tmux exp16-cell7-chain` → `_chainer.sh` → waits for `RESCALE_SEQUENCE_DONE`, then `launch_cell7.sh`
  (cell 7 = rescale + clean@5, **50 steps**). Exits without launching if sequencer dies w/o the flag.

### Cell map (all rescale=true=constant h·m/(1-p); cell 6 = dense no-op)
| launch.sh | name | config | steps |
|---|---|---|---|
| 2 | grpo_mask_channel_p0p9_rescale_10steps | mask p0.9 + rescale | 10 |
| 4 | grpo_mask_channel_p0p9_rescale_clean_every4_20steps | + clean@4 | 20 |
| 5 | grpo_mask_channel_p0p9_rescale_anchor2_spectral2_20steps | + anchor@2 + spectral@2 (needs T5) | 20 |
| 6 | dense_grpo_comm_eff_off_25step_reference | dense / strict-no-op proof | 25 |
| cell7 | grpo_mask_channel_p0p9_rescale_clean_every5_50steps | rescale + clean@5 | 50 |

Done sentinels: per-cell `metrics/<name>/done.flag`; sequence `RESCALE_SEQUENCE_DONE`; cell7 `CELL7_DONE`.

## Perf config (`/workspace/runs/EXP-16/perf.env`, sourced by both scripts)
`USE_DYNAMIC_BSZ=True`, `PPO_MAX_TOKEN_LEN_PER_GPU=LOG_PROB=REF=98304`.
Rationale: free_cache_engine=True frees vLLM KV during the actor update → training owns ~full 183GB.
Static micro_batch=1 gave 0.75% MFU (mean resp 277 tok). 98304 est. peak ~75GB. **Verify on step 1; if peak
< ~130GB there's room to push later cells higher; if near OOM, lower it.**

## First result (static-bsz cell 2, archived as `*.preDynBsz.*`): grad_norm **4.51** (≈12× dense 0.38),
pg_clipfrac 0.028 (not saturated), mask_ratio 0.900. Rescale stability CONFIRMED. (Tracebacks in that log were
benign DataLoader/wandb shutdown noise.) Re-running cell 2 under dynamic_bsz for a consistent matrix.

## COMPLETION FLOW (when `CELL7_DONE` exists — operator pre-authorized all of this)
1. **Rsync metrics off the box** → `runs/EXP-16/metrics/` on laptop (MUST happen before teardown).
   `rsync -avz -e 'ssh -i ~/.ssh/vast_ai -p 33732' root@3.144.230.17:/workspace/runs/EXP-16/metrics/ runs/EXP-16/metrics/`
2. **Push everything.** On box: `cd /workspace/verl && git add -A && git commit` the rescale_mode patch
   (msg: "EXP-16: switchable mask.rescale_mode {none|constant|rms_match}"). The box has NO GH push creds →
   `git bundle create /workspace/runs/EXP-16/exp_final.bundle exp/16-short-run-stability-matrix`, rsync to
   laptop, `git fetch <bundle> exp/16-...:exp/16-...` then `git push origin exp/16-short-run-stability-matrix`
   from the laptop (which has gh auth).
3. **Tear down** instance 38454090: `bash .claude/skills/vast-teardown/run.sh 38454090` (or `vastai destroy
   instance 38454090` if the skill needs a ledger row). Only AFTER push + metrics confirmed on laptop.
4. **Draft PR** head=`exp/16-short-run-stability-matrix` base=`vast-ai-workload` repo=`shamanez/verl`
   (use the `pr` skill or `gh pr create --draft`). Summarize T5 (spectral.cadence gate + anchor.delay_K +
   numeric metrics + early-stop) + rescale_mode patch + EXP-16 rescale-matrix results + the dynamic_bsz perf
   change. Report science verdict separately; PR lands the infra regardless.
5. Record findings (LOG/findings), update PROGRESS + memory, mark issue #16 label.

## Monitoring helpers on box
`run_rescale_sequence.log`, `chainer.log`, `metrics/<cell>/train.log`. Step metric line: `grep 'step:N '`.
