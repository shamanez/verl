# EXP-26 — new-session handoff prompt (Step B → C → E)

Launch a fresh session in `/Users/shamane/Documents/verl/research`, then paste the block
below (e.g. as `/goal <block>`, or `/loop 30m <block>`). It resumes a HALF-FINISHED,
staged experiment: Step A is done; this session finishes the rest of the plan.

---

You are the orchestrator for EXP-26 (issue #26, repo shamanez/verl-compression-research).
Read `.claude/playbooks/orchestrator.md` and `.claude/plans/26.md` and execute them to drive
EXP-26 to FULL completion. **You are RESUMING a half-finished, STAGED experiment** — Step A
(the diagnostic gate) is DONE and its **DECISION is `go_C_then_B`**; so the remaining sequence is
**Step C → Step B → Step E** (Step C runs FIRST — see why below). Your job is to finish the ENTIRE
plan and land a terminal verdict + LOG entry. Do not stop before then unless a REVISE-child or
STUCK operator gate is reached. (This file is named …STEP_B… but per the DECISION, Step C precedes
Step B.)

## Already done — Step A (do NOT redo any of this)
- **All code lives on branch `exp/26-geometry-audit-ef-powersgd` @ `5a35fa96c`** (8 fix commits,
  pushed to origin `shamanez/verl`, base `vast-ai-workload`). This branch ALREADY contains: the
  **`ef_powersgd` direction-preserving merger** (the Step-B method), the geometry-audit
  instrumentation (now bug-free after 7 capture fixes), the `comm_eff` config flags, and the
  launcher wiring. The experiment-runner for Step B branches from / reuses this commit — do NOT
  re-implement the merger or the capture layer.
- **Step A's outcome is recorded in `runs/EXP-26/stepA_decision.md` (+ STAGE-A `verdict.md`).**
  **DECISION = `go_C_then_B`** (+ `retire_sign_replacement` confirmed). Headline science:
  - **H3 CONFIRMED** — `signed_ema` sign-replacement is a structural coin-flip (sign-agreement
    ~0.50 even at `delay_K=0`) → retired. `ef_powersgd` (no sign term) is the successor.
  - **H1 CONFIRMED** via the confound-free merger isolate `cos(G_comp, G_corr)=+0.717`
    (signed_ema rotates the already-compressed update ~44°). The dense reference is
    `G_fresh_anchor@delay_K=0` ("Option A", validated `cos(G_fresh_anchor,G_dense)=0.985`); the
    broken parallel-`G_dense` clone was retired, not used.
  - **H2 TRUE** — `Q_act` activation-capture 0.9985 but UPDATE-energy capture only **0.318**
    (off-principal share 0.682) ⇒ `Q_act` misses ~68% of GRPO update energy. THIS is why the
    DECISION is `go_C_then_B`: run the rlvr-native Q-content sweep (Step C) BEFORE `ef_powersgd`.
  - Step-A fp32 captures local: A0 `captures/A0_dense/rank0`, A1 `captures/A1_powersgd_r77/rank0`,
    A2 `captures/A2_signed_ema_a0p5/rank0_new_optA` (NOT `…/rank0`, which is stale re-run#1 data).
  - **LOAD-BEARING CAVEAT for Step B/C analysts:** the plan's literal `cos(G_dense, G_comp)`
    discriminator is NOT cleanly measurable through `G_fresh_anchor` (clean-PG-vs-PPO-clip loss +
    activation-vs-weight operand confound — that's why A1's `cos(G_fresh_anchor,G_comp)` came back a
    spurious +0.01). Use the confound-free **`cos(G_comp, G_corr)`** as the update-direction
    discriminator, OR capture the anchor grad under the SAME PPO-clip loss as the fast path. The
    routing + sign-retirement rest only on confound-free measurements and are robust.
- **The Step-A `verdict.md` is a STAGE-A-scope verdict — NOT the terminal issue verdict.** Do NOT
  treat it as "done" and skip to log-writer. The issue is complete only after Step E + the
  whole-issue analyst predicate. (If the orchestrator state machine reads it as VERDICT_PASS,
  override that: continue with Step B per `## Experiment sequence`.)

## Remaining — drive per plan `## Experiment sequence` (DECISION = go_C_then_B ⇒ C FIRST)
- **Step C (`id: C`) — RUN FIRST** (gated by the Step-A H2 finding: `Q_act` under-captures
  off-principal GRPO UPDATE energy, 0.318 / off-principal 0.682). Q-CONTENT sweep at FIXED rank 77;
  arms `q_basis ∈ {act(control), grad, adv, tail, hybrid, ticket}`. NOTE: in the Step-A branch the
  non-`act` Q families are scaffolded but intentionally FAIL-LOUD — the runner must implement the
  sketch construction on `exp/26-…` before launching C (this is expected code work). Judge by
  update cosine + off-principal preservation (confound-free), NOT activation reconstruction.
  Success: a Q family beats `Q_act` on update-capture + off-principal preservation AND its training
  arm `val@50 >= 0.7414` with no collapse.
- **Step B (`id: B`):** `correction_mode=ef_powersgd` (NO sign term); arms `{ef_powersgd,
  plain-PowerSGD r77, dense}`; 50→100 steps, val@25; LOCKED substrate (anchor owns Q,
  `delay_K=5`, `clean_cadence=0`, `r=77`); use the passing Step-C Q content. PASS gate: ef_powersgd
  best `val@50 >= 0.7414` AND update direction `cos(G_comp, G_corr)` improves over plain-PowerSGD
  (confound-free metric — see caveat above) AND no length/clip collapse alarm.
- **Step E (`id: E`):** after B (or C) yields a stable parity-recovering method — measure
  inter-stage activation comm volume vs dense; success = parity `val@50>=0.7414` at comm < dense.
- Then analyst writes the **TERMINAL `verdict.md`** (whole-issue PASS/REVISE/STOP per
  `## Analyst predicate`; this SUPERSEDES the current STAGE-A `verdict.md`) and log-writer prepends
  the **LOG.md** entry + completes the `## Milestone M6` section. THAT is "done."

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
