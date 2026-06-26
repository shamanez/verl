# EXP-42 runbook — GPU-from-minute-1 execution (optimized, no-smoke)

**State:** all implementation + CPU/config probe DONE locally (14/14 pass). The run is
now **performance-optimized**: dynamic batching ON, initial validation OFF, **no smoke
phase** — `drive.sh` goes straight into the 3 training cells. The box is operator-provided
per session (OFF by default; teardown-safe registration — never auto-torn-down).

- Branch: `exp/42-lookahead-horizon` @ `eda0eaeb` (only exp branch; pushed to origin).
- Instrument: `comm_eff.probe.grad_proj_enabled` (+ `grad_proj_out_dir`).

---
## Performance-optimized surface (LOCKED 2026-06-26 — do NOT re-tune)
The first attempt ran **2.7× too slow** because the comm-eff launcher defaults to STATIC
batching (`USE_DYNAMIC_BSZ=False` ⇒ `micro_batch=1`/GPU, the token budget IGNORED, ~1%
MFU, ~17 GB HBM of 143, ~135 s/step). The fast reference run `hoasiw5u` (same signed_ema
method) ran 50 s/step with dynamic batching. The scaffold now pins:

- **`USE_DYNAMIC_BSZ=True`** — token-balanced micro-batches. The per-element mask is
  packing-invariant, so results are unchanged; ~2.7× faster.
- **`VAL_BEFORE_TRAIN=False`** — skip the step-0 validation pass (operator). Validation
  still runs at steps 25/50/75/100.
- **No smoke** — `drive.sh` launches run1→run2→run3 directly (the +2-backward FSDP path
  was already validated: prior smoke GATE_PASS + run1 ran healthy through step 7).
- **`PPO_MAX_TOKEN_LEN_PER_GPU`** and **`snapshot_device`/`ema_device`** — see the
  "Locked params" block at the bottom (finalized from the OOM/offload analysis).

These are baked into `run_cell.sh`; do not override them on the box.

---
## HF + WandB auth (REQUIRED — set BEFORE launch; NEVER echo secret values)
The launcher **sources `~/.config/verl-research/secrets.env` ON THE BOX** and FATALS if it
is missing OR if `VAST_API_KEY` leaks into it. A fresh template box does NOT have it.
Push everything over SSH stdin so values never hit a terminal/log:

1. **secrets.env (mandatory).** Push a STRIPPED copy — HF_TOKEN + WANDB_API_KEY only:
   ```bash
   grep -E '^(export +)?(HF_TOKEN|WANDB_API_KEY)=' ~/.config/verl-research/secrets.env \
     | sed -E 's/^(export +)?/export /' \
     | ssh <box> 'mkdir -p ~/.config/verl-research && cat > ~/.config/verl-research/secrets.env && chmod 600 ~/.config/verl-research/secrets.env'
   ```
   (The `grep` drops VAST_API_KEY — the launcher aborts if it sees it.)
2. **HF token file** (so `hf`/`huggingface_hub` auth even without env):
   ```bash
   source ~/.config/verl-research/secrets.env
   printf '%s' "$HF_TOKEN" | ssh <box> 'mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token'
   ```
3. **WandB `.netrc`** (optional — the exported `WANDB_API_KEY` already authenticates wandb):
   ```bash
   printf 'machine api.wandb.ai\n  login user\n  password %s\n' "$WANDB_API_KEY" \
     | ssh <box> 'cat > ~/.netrc && chmod 600 ~/.netrc'
   ```
4. **HF CLI gotcha:** `huggingface-cli` is **DEPRECATED and no longer works** — use `hf`
   (huggingface_hub ≥ 1.x):
   - verify: `hf auth whoami`  → expect "✓ Logged in", user **`gshasiri`**.
   - pre-download (background, so training does not stall on a cold pull):
     `nohup hf download Qwen/Qwen2.5-1.5B-Instruct > ~/hf_dl.log 2>&1 &`
5. The launcher re-exports the HF token under `HUGGING_FACE_HUB_TOKEN` +
   `HUGGINGFACE_HUB_TOKEN` (every name HF clients look for).

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (the launcher's `MODEL_PATH` default).
- **WandB:** project `verl_compression_research`; run names `exp42-run1` / `exp42-run2` /
  `exp42-run3` (set per cell by `run_cell.sh` via `EXPERIMENT_NAME`).
- Auth files live on the persistent disk and survive a Vast stop/start — rewrite only if
  `ls ~/.config/verl-research/secrets.env ~/.netrc ~/.cache/huggingface/token` shows missing.

---
## STEP A — reconnect + checkout (seconds)
Vast may reassign host/port on stop/start; take the current ssh line from the operator.
```bash
SSH='ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST>'
$SSH 'nvidia-smi -L'   # confirm 4..8 H100/H200 — ABORT if single-GPU
$SSH 'cd /workspace/verl && git fetch origin exp/42-lookahead-horizon -q && git checkout -B exp/42-lookahead-horizon FETCH_HEAD && git log --oneline -1'   # expect eda0eae
```
Then do the HF + WandB auth above (if missing), and push the scripts:
```bash
rsync -az -e "ssh -i ~/.ssh/vast_ai -p <PORT>" runs/EXP-42/{drive.sh,run_cell.sh,probe_cpu.py} root@<HOST>:/workspace/runs/EXP-42/
```
(`smoke.sh` is no longer part of the flow — kept on disk only as an optional manual gate.)

## STEP B — launch the 3 runs (ONE command; no smoke, GPU busy from here)
```bash
$SSH "tmux new -d -s exp42 'bash /workspace/runs/EXP-42/drive.sh'"
```
`drive.sh` runs run1 (fixed_linear@0.50) → run2 (learned@0.50) → run3 (no-projection),
each 100 steps @ 1024 ctx, anchor delay_K=cadence=10, dynamic batching, strictly sequential.
Watch: `tail -f /workspace/runs/EXP-42/drive.status`.

## STEP C — monitor from the laptop (training-log-monitor subagent, 30 s cadence)
Watch `drive.status` + the current cell's `/workspace/runs/EXP-42/<cell>/train_<cell>_internal.log`
for: tmux liveness, crash signatures (Traceback / CUDA OOM / NaN / AssertionError), per-fire
`[grad-proj-probe] ... grad_proj_gain=...`, val@25/50/75/100, and `response_length/mean`.
- **First few steps are the new risk surface** (dynamic batching + the +2-backward path
  were not smoke-gated) — confirm no OOM/NaN by ~step 10 (the first projecting fire).
- register each cell teardown-safe: `bash runs/EXP-42/register_run.sh <cell> RUNNING` → `COMPLETE` on its `done.flag`.
- as each cell finishes (its `done.flag` appears): backfill the final 1–2 steps from the
  authoritative internal log; `rsync` `/workspace/runs/EXP-42/<cell>/` → `runs/EXP-42/<cell>/`.

## STEP D — analyst → verdict (step 6)
```bash
python research/scripts/analyze.py runs/EXP-42 --emit verdict.md   # or dispatch the analyst agent
```
- **HEADLINE:** median `grad_proj_gain` for run1 & run2. **STOP if ≤ 0 for BOTH** (the
  projection premise is falsified at the gradient level — the deepest finding; a clean STOP
  with a measured gain profile is a SUCCESSFUL outcome of this plan).
- Secondary (conversion): final `val@100` of each vs run3 + collapse check
  (`response_length/mean ≤ 2×` its first-25-step mean).

## STEP E — report + teardown
Report to operator. **Do NOT tear down the box — ASK first** (operator owns it).

---
## Teardown-safety (in effect)
Box attached via `vast-attach --no-register` (no box ledger row) + per-cell rows use empty
`handles[]` ⇒ the teardown Stop hook finds no instance to destroy ⇒ the box is NEVER
auto-torn-down. Teardown happens only on explicit operator OK.

---
## Locked params (finalized from the OOM/offload analysis)
<!-- FINALIZED-BELOW: PPO_MAX_TOKEN_LEN_PER_GPU per GPU type + snapshot_device/ema_device decision -->
