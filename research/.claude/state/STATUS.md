# Research Status — 2026-06-05

## Active
**No live Vast.ai instances ($0/hr).** EXP-25 GPU work is **PAUSED by operator** (2026-06-05) pending an account-level Vast SSH fix the operator owns. Code is implemented + pushed; only on-GPU probes/sweep are blocked.

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 25 | Anchor-circuit default (stale-M + anchor-owns-Q + signed_ema merger) | **PAUSED (infra)** | 3× provisioned, all TORN_DOWN | — | R1/R2/R3 built + CPU-validated + pushed `exp/25-anchor-default`@`bf351c6e4`. Provision BLOCKED: Vast SSH key injection broken (`ssh_key_ids:null`, attach API SQL type-bug, both account keys `default:None`). GPU work paused per operator; SSH fix operator-owned. |
| 24 | Error-feedback on PowerSGD residual + basis-aligned anchor | BLOCKED | — | — | `depends_on: #25` — needs #25 VERDICT=PASS first. |

## Blocker (operator-owned — DO NOT auto-work)
Vast.ai SSH key injection is broken: every provisioned box comes up `ssh_key_ids:null` → `Permission denied (publickey)` for both registered keys. Root cause: no DEFAULT key set on the account (no CLI to set one) + `vastai attach ssh` server-side bug (`psycopg2 InvalidTextRepresentation: ... integer: "ssh-ed25519 ..."`). Likely fix: set a default SSH key / delete+re-add in the Vast web console. Operator is handling this.

## Resume path (once SSH works — operator signals)
Re-provision one warm box (4×H200 → 8×H100) → id-0 anchor-M probe (cadence=1/delay_K=1) → id-1 all-flags-ON probe → α∈{0.0,0.3,0.5} sweep (50 steps, delay_K=5/cadence=5, seed_anchor_cache=false, max_targets=-1, powersgd r=77) → analyst → log-writer. Scaffold ready in `runs/EXP-25/` (launch.sh, check_probe.sh, exp.bundle).

## Hyperparameters (FIXED control surface)
Qwen2.5-1.5B-Instruct + GSM8K, vanilla GRPO (no-KL/no-entropy), lr 1e-6, train_batch 128, ppo_mini 64, n=8, max_response 16384, seed 0. Run control: total_steps 50→100, val every 25, anchor refresh every 5 from θ_{t−5} stale, NO periodic clean step. Codec: PowerSGD r=77. Source: `runs/FIXED_CONTROL_SURFACE.md` + launcher `${VAR:-default}` + `project.yaml`.

## Last tick
2026-06-05 · running=[] · paused=[25 infra] · blocked=[24 dep] · offline-work=[Thread-B dead-code cleanup in progress]

## Budget
$0/hr now (0 live instances). Wasted spend this session: 3× 4×H200 up only minutes each (unreachable, torn down on detection). max_gpu_hr=48 cap untouched (no probe/arm ran).
