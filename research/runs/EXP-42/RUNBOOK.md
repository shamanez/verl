# EXP-42 runbook — GPU-on execution sequence

Status as of pause (GPU off): **all implementation + CPU/config probe done locally.**
Remaining steps need the GPU box ON. When the operator switches it back on, get the
(possibly NEW) SSH endpoint and run the steps below in order.

## Branch / code
- Branch `exp/42-lookahead-horizon` @ `eda0eaeb` (pushed to origin shamanez/verl).
  = look-ahead code (649594ae/08bc6d96) + the NEW grad-projection-accuracy instrument
  (9839c090) + adversarial-review fixes (eda0eaeb).
- Instrument knob: `comm_eff.probe.grad_proj_enabled` (+ `grad_proj_out_dir`).

## CPU probe — DONE (local, MacBook), ALL PASS
`PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES="" python runs/EXP-42/probe_cpu.py`
Covers inv2 (limiting-case identity + coeffs + weight_proj_ratio==1@strength0),
inv3 (learned cold-start==fixed), inv6 (cross-rank single-proc), and config
(replay required, snapshot_device=cpu required, geometry mutual-exclusion,
signed_ema NOT forced off, Hydra override accepted).

## GPU steps (need box ON) — STRICT ORDER

### 0. Reconnect
- Get the current ssh endpoint (Vast may reassign host/port on stop/start).
- `SSH="ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST>"`
- Confirm: `$SSH 'nvidia-smi -L'` shows 4 H200; `cd /workspace/verl && git log --oneline -1`.
  If the checkout is stale: `git fetch origin exp/42-lookahead-horizon && git checkout eda0eaeb`.
- Re-confirm auth files survived the restart: `~/.verl_auth.env`, `~/.netrc`,
  `~/.cache/huggingface/token` (rewrite via SSH stdin if missing — never echo).
- rsync scripts: `rsync -az -e "$SSH-form" runs/EXP-42/{run_cell.sh,smoke.sh,probe_cpu.py} root@HOST:/workspace/runs/EXP-42/`

### 1. GPU smoke probe (the remaining hard gates: inv 1,4,5,7) — GATE
- `$SSH 'bash /workspace/runs/EXP-42/smoke.sh fixed_linear'`  (~few min; reduced batch, real 1024 ctx)
- PASS iff: a `[grad-proj-probe] ... lookahead_active=True ... grad_proj_gain=... cross_rank_max_rel_dev≈0`
  line appears; NO Traceback/OOM/NaN/AssertionError; rc=0.
- Optionally also `smoke.sh off` to exercise run3's +1-backward path (gain≈0, weight_proj_ratio≈1).
- If FAIL: STOP, diagnose, fix on the branch (commit+push), re-probe. Do NOT launch on a failed probe.

### 2. The 3 runs — STRICTLY sequential on the one box (run1 -> run2 -> run3, never parallel)
For CELL in run1, run2, run3:
- register: `bash runs/EXP-42/register_run.sh $CELL RUNNING`   (laptop; teardown-safe empty handles)
- launch in tmux on box:
  `$SSH "tmux new -d -s exp42-$CELL 'bash /workspace/runs/EXP-42/run_cell.sh $CELL'"`
- monitor with a training-log-monitor subagent (30s cadence): tmux liveness, NaN/OOM/Traceback,
  per-fire grad_proj_gain (grep '\[grad-proj-probe\]'), val@25/50, response_length.
  Internal log: `/workspace/runs/EXP-42/$CELL/train_${CELL}_internal.log`.
- on finish: backfill final 1-2 steps from the authoritative train.log; rsync artifacts:
  `rsync -az -e "$SSH-form" root@HOST:/workspace/runs/EXP-42/$CELL/ runs/EXP-42/$CELL/`
- `bash runs/EXP-42/register_run.sh $CELL COMPLETE`   (flip to COMPLETE; keeps hook a no-op anyway)
- ONLY then start the next cell.
  Cells: run1=fixed_linear@0.50 ; run2=learned@0.50 ; run3=lookahead OFF (no projection control).

### 3. Analyst -> verdict (step 7)
- `python research/scripts/analyze.py runs/EXP-42 --emit verdict.md` (or dispatch the analyst agent).
- HEADLINE: median grad_proj_gain for run1 & run2. **STOP if ≤ 0 for BOTH** (premise falsified at the
  gradient level — the deepest finding). Secondary: val@50 vs run3 + collapse check
  (response_length/mean ≤ 2× its first-25-step mean).

### 4. Report + teardown
- Report to operator. **Do NOT tear down the box — ask first** (operator owns it).

## Teardown-avoidance (in effect)
- Box registered via `vast-attach --no-register` => NO box ledger row.
- Per-cell rows use empty `handles[]` => teardown Stop hook destroys nothing.
- Result: the box can NEVER be auto-torn-down by the hook. Teardown only on operator OK.
