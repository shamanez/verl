# Research Status — 2026-05-27T23:00:42+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 4 | M2 — comm_eff no-op scaffolding: dense GRPO parity smoke | PLAN_READY (demoted from VERIFIED_FAIL) | — | verify:FAIL | codex pre-impl structural gate returned `VERIFY: FAIL` (wrapper exit 0 — genuine, not timeout). Label `status:approved`→`status:planned`; critique posted to #4. **No GPU spent.** Awaiting human plan tightening (exact-match scalars + state/param-hash vs unmodified dense reference + disabled-marker counter validation) then re-approval. |
| 3 | M1 — Qwen2.5-1.5B GRPO baseline (GSM8K) | DONE | 1×4H200 torn down | PASS | val 0.087→0.789 (+0.702, 14× threshold) by step 100; HF step50+step100 (private); `runs/EXP-3/REPRODUCIBILITY.md` |
| 5–11 | M2/M3 backlog (PRF mask, contamination guard, spectral filter, anchor circuit, full M95+AP smoke, DP compression, 100-step) | BACKLOG | — | — | open, no `status:` label, no plan yet — not yet claimed for planning (triage owns) |

## Last tick
2026-05-27T23:00:42+10:00 · verify=[4→FAIL] · running=[] · analyzing=[] · logging=[] · blocked=[] · demoted=[4]

## Budget
$/hr now: $0.00 (no instances running) · spent this tick: $0 (EXP-4 caught at the verify gate before provisioning) · EXP-3 historical: ~$12.76 instance total / ~$10.29 training · monthly cap remaining: tracked separately
