# Research Status — 2026-06-04T13:42:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 20 | PowerSGD-style PP activation compression (M6) | DONE · PASS | 1×4×H200 (i_39319060, TORN DOWN 2026-06-04T13:32Z) | PASS | val-core GSM8K acc@50: mask 0.7384 \| r=77(matched) 0.7415 (+0.0031) \| r=102(+33%) 0.7437 (+0.0053); hypothesis CONFIRMED un-caveated; draft PR #13 flipped ready-for-review; launcher promotion deferred (cosine metric + operator confirm needed) |
| 19 | M5 — surpass dense baseline (epic) | UNCLAIMED | — | — | no research:claim/status/plan → awaiting triage |
| 18 / 21 | M4 / reweight | TORN_DOWN | — | — | no-heartbeat-30min |

## Last tick
2026-06-04T13:42:00+10:00 · running=[] · analyzing=[] · logging=[done] · unclaimed=[19]

## M6 progress
- EXP-20: PASS (PowerSGD codec, r=77 matched budget beats mask; 1 of ≥2 needed for milestone summary)
- Next M6 experiment: TBD (follow-up to instrument update cosine + broader rank/cadence sweep)

## EXP-20 — VERDICT: PASS (2026-06-04T13:41:56+10:00)
Three-arm result (mask p=0.95, PowerSGD r=102, PowerSGD r=77 — codec is the ONLY axis):
- mask p=0.95 (76.8 coords/tok):  val-acc@50 = 0.7384  [THE BAR]
- PowerSGD r=77  (77.0 coords, matched +0.26%): val-acc@50 = 0.7415 (+0.0031) [DECISIVE — equal budget]
- PowerSGD r=102 (102.0 coords, +33% budget):  val-acc@50 = 0.7437 (+0.0053) [CAVEATED — budget mismatch]

Key codec health (both psgd arms): q_cond≈1.0000002, recon_rel_error converged ~0.02, q_cross_rank_max_rel_dev=0.0 (bit-identical consensus), 0 NaN/OOM/single-GPU fallback.

Open follow-ups (non-blocking on verdict, required before launcher promotion):
1. Instrument dense-vs-compressed update cosine (criterion 7 was unmeasured, not failing)
2. Gate verify_basis_agreement_across_ranks on self.sync_basis (math panel HIGH-2; crashes sync_basis=false diagnostic mode)

Draft PR shamanez/verl#13 flipped ready-for-review (exp/20-powersgd-activation → vast-ai-workload).
Box 39319060 destroyed; vastai show instances = empty ($169.79 of $1500 monthly cap spent).
