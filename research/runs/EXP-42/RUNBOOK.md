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
## Locked params (finalized 2026-06-26 from the OOM/offload analysis + operator choice)

Operator choice: **ema stays on CPU, strictly no smoke** — trade ~0.5–1 s/step (the
per-step EMA H2D transfer) for zero added HBM and no OOM gate. The ONLY non-default knobs
are batching + initial-val; everything memory-related stays at the safe, already-run
defaults, so dynamic batching is the single new variable (well-trodden in verl; budget
18432 < the fast ref's 24576 ⇒ low OOM risk).

| Knob | Locked value | Env var | Note |
|---|---|---|---|
| batching | `True` | `USE_DYNAMIC_BSZ` | the 2.7× win; results-invariant (per-element mask packing-invariant) |
| initial val | `False` | `VAL_BEFORE_TRAIN` | skip step-0 eval; val@25/50/75/100 unaffected |
| token budget (4×H200) | `18432` | `PPO_MAX_TOKEN_LEN_PER_GPU` | launcher default; active under dynamic_bsz; **ceiling ~20480 with anchor ON** |
| snapshot_device | `cpu` | (default) | per-fire only; **required by the grad_proj guard** — do not move |
| ema_device | `cpu` | (default) | KEPT on CPU per operator (OOM-safe, no smoke); per-step transfer accepted |
| vLLM mem util | `0.4` | `ROLLOUT_GPU_MEM_UTIL` | launcher default; conservative (the prior healthy run used it) |

- These map to: `run_cell.sh` exports `USE_DYNAMIC_BSZ=True` + `VAL_BEFORE_TRAIN=False`;
  all memory knobs inherit the launcher defaults (cpu/cpu/18432/0.4). No other overrides.
- **Expected:** ~55–65 s/step (a touch above the 48–52 ema-on-GPU estimate, since the EMA
  stays on CPU), ~55–70 min/cell, **~3–3.5 h for all 3 cells** (vs ~12 h unoptimized).
- **8×H100 fallback** (only if no H200; anchor-ON is marginal there): set
  `PPO_MAX_TOKEN_LEN_PER_GPU=10240` (drop to 8192 if a fire-tick OOMs) and
  `ROLLOUT_GPU_MEM_UTIL=0.38`; keep ema/snapshot on CPU. **Do not mix GPU types within one
  3-cell run** (different budgets/devices would break cell-to-cell comparability).
- **No-smoke residual risk:** the dynamic-batching + grad-proj +2-backward path was never
  gate-validated; the training-log-monitor must confirm no OOM/NaN by ~step 10 (the first
  projecting fire). If it OOMs, lower `PPO_MAX_TOKEN_LEN_PER_GPU` (e.g. 16384) and relaunch.
