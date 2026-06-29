# EXP-42 runbook — weight-projection accuracy (single 1×H200, 2 regimes, sketch→MacBook)

> **REFRAMED 2026-06-29.** EXP-42 now measures **weight-projection accuracy** (does `θ̂` land
> closer to `θ_now` than raw-stale `θ[t−K]`), as a function of how many steps ahead we predict,
> at the **K=10** operating point, in two regimes (plain GRPO / GRPO+activation-compression). A
> *gradient* study is deferred to a separate future session (not planned here). Plan:
> [`.claude/plans/42.md`](../../.claude/plans/42.md). The old 3-cell gradient scaffold here is
> superseded (it was the prior gradient study).

**Strategy (operator: LIMIT Vast spend):** GPUs **collect everything** — the box trains and emits
a tiny per-tick **weight sketch** (~320 MB/regime) + exact on-box headline scalars at the
operating points. The rule is just: anything doable from downloaded data, do on the MacBook — so
we tear the H200 down the moment `weights/` is synced instead of renting it through the long
horizon sweep + report. **One 1×H200 box, two short sequential runs (Regime A → Regime B), then
tear down; the full sweep replays on the MacBook.**

- Branch: `exp/42-weight-accuracy` (NEW, off `vast-ai-workload`). Instrument:
  `comm_eff.probe.weight_traj` (per-tick count-sketch + per-matrix mean + exact calib scalars).
- Single-GPU is an **operator-authorized deviation** (2026-06-29) from the 4≤num_gpus≤8 fixed
  control — full permission to fit this to a single H200. **Try 1×H200 first; fall back to 1×B200
  (~192 GB) ONLY if H200 OOMs.** Prefer attaching an operator-provided box via `vast-attach`.

---
## HF + WandB auth (REQUIRED — set BEFORE launch; NEVER echo secret values)
The launcher **sources `~/.config/verl-research/secrets.env` ON THE BOX** and FATALS if it is
missing OR if `VAST_API_KEY` leaks into it. Push a STRIPPED copy over SSH stdin so values never hit
a terminal/log:

1. **secrets.env (mandatory)** — HF_TOKEN + WANDB_API_KEY only:
   ```bash
   grep -E '^(export +)?(HF_TOKEN|WANDB_API_KEY)=' ~/.config/verl-research/secrets.env \
     | sed -E 's/^(export +)?/export /' \
     | ssh <box> 'mkdir -p ~/.config/verl-research && cat > ~/.config/verl-research/secrets.env && chmod 600 ~/.config/verl-research/secrets.env'
   ```
2. **HF token file:**
   ```bash
   source ~/.config/verl-research/secrets.env
   printf '%s' "$HF_TOKEN" | ssh <box> 'mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token'
   ```
3. **WandB `.netrc`** (optional — exported `WANDB_API_KEY` already authenticates):
   ```bash
   printf 'machine api.wandb.ai\n  login user\n  password %s\n' "$WANDB_API_KEY" \
     | ssh <box> 'cat > ~/.netrc && chmod 600 ~/.netrc'
   ```
4. **HF CLI gotcha:** `huggingface-cli` is DEPRECATED — use `hf` (huggingface_hub ≥ 1.x):
   `hf auth whoami` → expect "✓ Logged in", user **`gshasiri`**. Pre-pull:
   `nohup hf download Qwen/Qwen2.5-1.5B-Instruct > ~/hf_dl.log 2>&1 &`
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`. **WandB:** project `verl_compression_research`, run
  names `exp42-regimeA` / `exp42-regimeB`. Auth files survive a Vast stop/start.

---
## STEP A — single-GPU launcher edit (one-time, on the exp branch)
The committed launchers hard-fail on 1 GPU (`(( DETECTED_GPUS < 4 || DETECTED_GPUS > 8 ))`,
`vast_comm_eff_baseline_*.sh:105`). On `exp/42-weight-accuracy`:
- relax the guard to allow `>= 1` GPU;
- ensure `ROLLOUT_TP=1` (the accel base already sets it).
Push the branch BEFORE launch (survives a laptop crash).

## STEP B — reconnect + checkout
```bash
SSH='ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST>'
$SSH 'nvidia-smi -L'    # confirm 1× H200 (this run is INTENTIONALLY single-GPU)
$SSH 'cd /workspace/verl && git fetch origin exp/42-weight-accuracy -q && git checkout -B exp/42-weight-accuracy FETCH_HEAD && git log --oneline -1'
```
Do the HF + WandB auth above (if missing), then push the (re-materialised) scaffold:
```bash
rsync -az -e "ssh -i ~/.ssh/vast_ai -p <PORT>" runs/EXP-42/{drive.sh,run_cell.sh,probe_cpu.py} root@<HOST>:/workspace/runs/EXP-42/
```

## STEP C — launch the 2 regimes (ONE command, sequential)
```bash
$SSH "tmux new -d -s exp42 'bash /workspace/runs/EXP-42/drive.sh'"
```
`drive.sh` runs **regimeA (COMM_EFF_ENABLED=false)** → **regimeB (enabled=true, powersgd r=77,
anchor+spectral OFF)**, each 80 steps (=160 ticks) @ resp=1024, dyn_bsz, `probe.weight_traj.enabled=true`.
Watch `drive.status`.

## STEP D — monitor (training-log-monitor subagent, 30 s cadence)
Watch `drive.status` + `<regime>/train_<regime>_internal.log` for: tmux liveness, crash signatures
(Traceback / CUDA OOM / NaN), per-tick `[weight-traj]` sketch writes, and the **codec-active check**
in Regime B (PowerSGD `reconstruction_rel_error` > 0 ⇒ codec is actually changing the gradient on 1
GPU — if it is identically 0 or the trajectory matches Regime A, STOP: the boundary codec is inactive
without pipeline/DP and Regime B is invalid).
- **First 10 steps are the risk surface** (single-GPU OOM + the new summon→sketch hook) — confirm no
  OOM/NaN and a growing `weights/manifest.jsonl`.
- as each regime finishes (`done.flag`), `rsync` only `<regime>/weights/` (sketches + manifest +
  calib) → `runs/EXP-42/<regime>/weights/`. **~320 MB/regime — tiny.**

## STEP E — analysis on the MacBook (after download, so the box can be torn down first)
```bash
python research/scripts/weight_proj_sweep.py runs/EXP-42 --emit report.html --calib-tol 0.05
python research/scripts/analyze.py runs/EXP-42 --emit verdict.md
```
- validates sketch vs the on-box EXACT calib scalars (<5%); if it fails ⇒ REVISE (bigger k / bf16
  full-dump), NOT a GPU re-run.
- emits `weight_proj_ratio` & `dir_cos` vs horizon h (ticks), fixed & learned, regime A & B, with
  per-matrix p10/p50/p90 and the crossover horizon h\*.
- **HEADLINE answer:** at h=K=10 (α=1; plus under/over-shoot h∈{5,20}), is median
  `weight_proj_ratio < 1` and `dir_cos > 0`? This decides whether a future gradient-accuracy study
  is worth planning.

## STEP F — report + teardown
Report to operator. **Do NOT tear down the box — ASK first** (operator owns it). After teardown,
de-bloat per the close-out duty (keep the report + verdict + SUMMARY entry; delete the sketches).

---
## Locked params
| Knob | Value | Note |
|---|---|---|
| GPUs | **1× H200** (1× B200 only on OOM) | operator-authorized; resp=1024 defuses the 16K headroom rule |
| base launcher | accel surface | resp=1024, USE_DYNAMIC_BSZ=True, ROLLOUT_TP=1, mem_util=0.55, token budget 24576 |
| operating point | **K=10** | anchor delay_K = cadence = 10 (test at 10, NOT 20) |
| regime A | `COMM_EFF_ENABLED=false` | byte-identical dense path |
| regime B | `enabled=true type=powersgd rank=77`, `ANCHOR_ENABLED=false SPECTRAL_ENABLED=false` | codec only (no anchor/merger — avoids circularity) |
| instrument | `probe.weight_traj.enabled=true` k=4096 | per-tick count-sketch + mean + exact on-box headline @ Δ=10, h∈{5,10,20} |
| steps | 80 (=160 ticks) | spans h≤30 ticks with ~115 anchor points |
| test_freq | 40 | val@40/80 = convergence sanity only; val is NOT the metric |
| snapshot/ema device | cpu | OOM-safe |

- **Expected wall-clock (rough):** single-H200 ≈ 3.5–4× slower/step than 4×H200 (~190 s/step) ⇒
  ~4 h/regime ⇒ **~8–9 h for both** + provisioning. Budget envelope 14 h. Regime B ≈ A (no anchor
  clone). **To fit H200 (full permission):** if OOM, lower `PPO_MAX_TOKEN_LEN_PER_GPU`
  (24576→16384→…) and/or `ROLLOUT_GPU_MEM_UTIL` (0.55→0.45); only if it STILL OOMs, provision 1×B200.

## Teardown-safety
Attach via `vast-attach --no-register` + empty `handles[]` ⇒ the teardown Stop hook finds no
instance to destroy ⇒ never auto-torn-down. Teardown only on explicit operator OK.
