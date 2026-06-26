# EXP-42 runbook — GPU-from-minute-1 execution

**State:** all implementation + CPU/config probe DONE locally (14/14 pass). Only the
GPU phases remain. They are pre-staged so that, once the box is on, **one command**
keeps all 4 GPUs busy from minute one — `drive.sh` chains smoke→gate→run1→run2→run3
with no idle gaps. The only legitimate idle is an aborted smoke gate (= fix-and-redrive).

- Branch: `exp/42-lookahead-horizon` @ `eda0eaeb` (only exp branch; pushed to origin).
- Instrument: `comm_eff.probe.grad_proj_enabled` (+ `grad_proj_out_dir`).

---
## STEP A — reconnect (seconds; the only manual setup)
Vast may reassign host/port on stop/start, so get the current ssh line from the operator, then:
```bash
SSH='ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST>'
$SSH 'nvidia-smi -L && cd /workspace/verl && git fetch origin exp/42-lookahead-horizon -q && git checkout -q eda0eaeb && git log --oneline -1'
# auth files survive a stop/start on the persistent disk; rewrite only if missing (over stdin, never echoed):
$SSH 'ls ~/.verl_auth.env ~/.netrc ~/.cache/huggingface/token 2>&1'
# push the latest scripts:
rsync -az -e "$SSH-without-the-cmd" runs/EXP-42/{drive.sh,run_cell.sh,smoke.sh,probe_cpu.py} root@<HOST>:/workspace/runs/EXP-42/
```

## STEP B — launch the whole pipeline (ONE command; GPU busy from here on)
```bash
$SSH "tmux new -d -s exp42 'bash /workspace/runs/EXP-42/drive.sh'"
```
`drive.sh` runs: **smoke (≈3 min, the FSDP/backend hard gate)** → if clean → **run1
(fixed_linear@0.50)** → **run2 (learned@0.50)** → **run3 (no projection)**, each 50
steps @ 1024 ctx, full batch, strictly sequential. Watch: `tail -f /workspace/runs/EXP-42/drive.status`.

## STEP C — monitor from the laptop (while drive.sh runs)
- training-log-monitor subagent, 30s cadence, watching `drive.status` + the current cell's
  `/workspace/runs/EXP-42/<cell>/train_<cell>_internal.log` for: tmux liveness, the smoke
  GATE_PASS/GATE_FAIL line, per-fire `[grad-proj-probe] ... grad_proj_gain=...`, val@25/50,
  `response_length/mean`, and crash signatures (Traceback/OOM/NaN/AssertionError).
- register each cell teardown-safe: `bash runs/EXP-42/register_run.sh <cell> RUNNING` → `COMPLETE` on its done.flag.
- as each cell finishes (its `done.flag` appears): backfill final 1-2 steps from the
  authoritative internal log; `rsync` `/workspace/runs/EXP-42/<cell>/` → `runs/EXP-42/<cell>/`.
- GATE_FAIL ⇒ STOP, inspect smoke logs, fix on the branch (commit+push), re-drive. Do NOT bypass.

## STEP D — analyst → verdict (step 7)
```bash
python research/scripts/analyze.py runs/EXP-42 --emit verdict.md   # or dispatch the analyst agent
```
- **HEADLINE:** median `grad_proj_gain` for run1 & run2. **STOP if ≤ 0 for BOTH** (the projection
  premise is falsified at the gradient level — the deepest finding; a clean STOP with a measured
  gain profile is a SUCCESSFUL outcome of this plan).
- Secondary (conversion): `val@50` of each vs run3 + collapse check (`response_length/mean ≤ 2×`
  its first-25-step mean).

## STEP E — report + teardown
Report to operator. **Do NOT tear down the box — ASK first** (operator owns it; it's OFF by default).

---
## Why this is GPU-efficient
- One command (`drive.sh`) → no human-in-the-loop gaps between smoke and the 3 runs.
- The smoke (cadence=2/delay_K=2) reaches a PROJECTING fire — the +2-backward path — in ~2 min,
  so the risky path is validated cheaply BEFORE the long runs (in a real run it would not project
  until ~step 20, wasting 10+ min before the gate could fire).
- The 3 real runs (full batch, 1024 ctx) saturate all 4 H200s back-to-back.

## Teardown-safety (in effect)
Box attached via `vast-attach --no-register` (no box ledger row) + per-cell rows use empty
`handles[]` ⇒ the teardown Stop hook can find no instance to destroy ⇒ the box is NEVER
auto-torn-down. Teardown happens only on explicit operator OK.
