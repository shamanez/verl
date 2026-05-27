# Research Status — 2026-05-27T23:32:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 4 | M2 — comm_eff no-op scaffolding: dense GRPO parity smoke | VERIFIED (operator-cleared) | — | verify:PASS (operator) | Plan reverted to rev-1 (rel-tol parity smoke; bitwise/SHA-256 hardening judged out-of-scope for this M2 smoke). Operator **waived** the codex plan-gate (codex raw: FAIL rev-1 / CONCERNS rev-2, preserved in `runs/EXP-4/verify/`). FAIL demotion comment deleted from #4; `VERIFY: PASS` (operator) comment posted; label `status:approved`. **Plan-level gate satisfied — next session dispatches `experiment-runner` WITHOUT re-verifying the plan.** Code-level verify of the eventual `exp/4-commeff-noop` diff is a separate later gate, not waived. **No GPU spent; no runner dispatched this session (stopped per operator).** |
| 3 | M1 — Qwen2.5-1.5B GRPO baseline (GSM8K) | DONE | 1×4H200 torn down | PASS | val 0.087→0.789 (+0.702, 14× threshold) by step 100; HF step50+step100 (private); `runs/EXP-3/REPRODUCIBILITY.md` |
| 5–11 | M2/M3 backlog (PRF mask, contamination guard, spectral filter, anchor circuit, full M95+AP smoke, DP compression, 100-step) | BACKLOG | — | — | open, no `status:` label, no plan yet — not yet claimed for planning (triage owns) |

## Last tick
2026-05-27T23:32:00+10:00 · verify=[] · running=[] · analyzing=[] · logging=[] · blocked=[] · operator-cleared=[4] · NOTE: operator-directed intervention, not an autonomous tick — no runner dispatched (stop requested)

## Budget
$/hr now: $0.00 (no instances running) · spent this tick: $0 (EXP-4 still pre-launch) · EXP-3 historical: ~$12.76 instance total / ~$10.29 training · monthly cap remaining: tracked separately
