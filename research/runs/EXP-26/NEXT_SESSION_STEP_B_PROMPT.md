# EXP-26 — new-session handoff prompt (Step B → C → E)

Launch a fresh session in `/Users/shamane/Documents/verl/research`, then paste the block
below (e.g. as `/goal <block>`, or `/loop 30m <block>`). It resumes a HALF-FINISHED,
staged experiment: Step A is done; this session finishes the rest of the plan.

---

You are the orchestrator for EXP-26 (issue #26, repo shamanez/verl-compression-research).
Read `.claude/playbooks/orchestrator.md` and `.claude/plans/26.md` and execute them to drive
EXP-26 to FULL completion. **You are RESUMING a half-finished, STAGED experiment** — Step A
(the diagnostic gate) is DONE; Steps **B → (C, conditional) → E** remain. Your job is to
finish the ENTIRE plan and land a terminal verdict + LOG entry. Do not stop before then unless
a REVISE-child or STUCK operator gate is reached.

## Already done — Step A (do NOT redo any of this)
- **All code lives on branch `exp/26-geometry-audit-ef-powersgd` @ `5a35fa96c`** (8 fix commits,
  pushed to origin `shamanez/verl`, base `vast-ai-workload`). This branch ALREADY contains: the
  **`ef_powersgd` direction-preserving merger** (the Step-B method), the geometry-audit
  instrumentation (now bug-free after 7 capture fixes), the `comm_eff` config flags, and the
  launcher wiring. The experiment-runner for Step B branches from / reuses this commit — do NOT
  re-implement the merger or the capture layer.
- **Step A's outcome is recorded in `runs/EXP-26/stepA_decision.md` (+ `verdict.md`).** READ IT
  for the machine-readable DECISION ∈ `{go_B_skip_C | go_C_then_B}`. Headline science:
  - **H3 CONFIRMED** — `signed_ema` sign-replacement is a structural coin-flip (~49% sign-agreement
    even with a fresh anchor) → retired. `ef_powersgd` (no sign term) is the successor.
  - **H1/H2 recovered via "Option A"** — the dense reference is `G_fresh_anchor@delay_K=0`
    (validated `cos=0.985`); the broken parallel-`G_dense` clone was retired, not used.
  - Step-A fp32 captures are local under `runs/EXP-26/captures/{A0_dense,A1_powersgd_r77,A2_signed_ema_a0p5}/`.
- **The Step-A `verdict.md` is a STAGE-A-scope verdict — NOT the terminal issue verdict.** Do NOT
  treat it as "done" and skip to log-writer. The issue is complete only after Step E + the
  whole-issue analyst predicate. (If the orchestrator state machine reads it as VERDICT_PASS,
  override that: continue with Step B per `## Experiment sequence`.)

## Remaining — drive per plan `## Experiment sequence`
- **Step B (`id: B`):** `correction_mode=ef_powersgd` (NO sign term); arms `{ef_powersgd,
  plain-PowerSGD r77, dense}`; 50→100 steps, val@25; LOCKED substrate (anchor owns Q,
  `delay_K=5`, `clean_cadence=0`, `r=77`). PASS gate: ef_powersgd best `val@50 >= 0.7414` AND
  update cosine `cos(G_fresh_anchor, G_corr)` improves over plain-PowerSGD AND no length/clip
  collapse alarm.
- **Step C (`id: C`):** run ONLY if the DECISION is `go_C_then_B` (Step A found `Q_act`
  under-captures GRPO update energy). Q-content sweep at FIXED rank 77.
- **Step E (`id: E`):** after B (or C) yields a stable parity-recovering method — measure
  inter-stage activation comm volume vs dense; success = parity `val@50>=0.7414` at comm < dense.
- Then analyst writes the **TERMINAL `verdict.md`** (whole-issue PASS/REVISE/STOP per
  `## Analyst predicate`) and log-writer prepends the **LOG.md** entry. THAT is "done."

## The warm box (operator kept it — verify first)
- Box **`40242796`** (4×H200, `145.241.108.98:40280`); handle `runs/EXP-26/handles/40242796.json`.
- FIRST check it's still alive (`vastai show instances`). **If ALIVE:** reuse it for Step B
  (warm docker/verl/dataset cache) via the gpu-idle-box-reuse pattern — register Step B as a NEW
  RUNNING ledger row owning the same `instance_id`, on commit `5a35fa96c`. **If GONE:** provision
  fresh per the plan's `gpu_filter_chain` (4×H200 → 8×H100).
- **BUDGET RESET:** the original `max_gpu_hr=60` was ~78% spent on the Step-A diagnostic
  debugging. Step B needs a FRESH budget — set a new `max_gpu_hr` on the Step-B ledger row
  (Step B ≈ 3 arms × 50–100 steps; budget ~40–60 GPU-hr). Do not inherit the spent Step-A clock.

## Non-negotiable guardrails (from the plan)
- `ef_powersgd` is direction-PRESERVING: **NO sign term** (sign-replacement was the falsified
  defect — that's the entire thesis).
- Realism invariants on the TRAINING path, asserted every arm: `powersgd_basis_updates==0` on the
  fast net, `anchor_q_updates>0`, full uncompressed pass ONLY in the anchor, `delay_K>=5`,
  `clean_cadence=0`. These are NOT sweep knobs and may not be tuned/ablated.
- Monitor every RUNNING box immediately with `training-log-monitor` (background). Honor the
  standing ENTROPY-COLLAPSE / response-length / IS watch on every run.
- Gate analyst acceptance on real metrics (val@50, update cosine, collapse alarms) — not "looks
  done." Never tear down a healthy box except on a monitor `teardown_*` recommendation or budget
  breach.

## Definition of done
`runs/EXP-26/verdict.md` reflects the WHOLE-ISSUE outcome (after Step B/(C)/E) AND `LOG.md` has an
EXP-26 entry — OR a REVISE child is created / STUCK escalation is reached.
