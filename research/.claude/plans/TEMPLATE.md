# Plan EXP-<ID>

The `research-planner` subagent fills this template by parsing the GitHub issue body. Humans may edit before flipping the label to `status:approved`. After approval, the orchestrator treats every field below as the contract for `experiment-runner`, `analyst`, and `log-writer`. Do not delete sections; mark unused ones with `(n/a)`.

## Experiment
- id:            EXP-<NUMBER>
- title:         <verbatim from issue title>
- issue:         <gh issue URL>
- kind:          experiment | ablation | implementation | brainstorm | literature   # routes state machine — see §"Kind" below
- milestone:     M<N>            # routing tag only; user-defined semantics
- created_at:    <ISO-8601>
- baseline_run:  EXP-<NN> | none           # for ablation, this is the parent experiment

## Kind (drives orchestrator routing)

| kind | code_change | runs on Vast | analyst | output |
|---|---|---|---|---|
| `experiment` (default) | maybe | yes | runs the plan's predicate | verdict + LOG entry; draft PR on PASS+code_change |
| `ablation` | maybe | yes | runs the plan's predicate | same as experiment; requires `depends_on:` parent EXP that PASSED |
| `implementation` | **true** (required) | NO | n/a | plan is the deliverable; draft PR after human approval |
| `brainstorm` | no | NO | n/a | plan is the deliverable; iterate as comments; promote to `experiment` later by editing kind |
| `literature` | no | NO | n/a | plan/issue is the deliverable; the operator handles any derivation review manually |

If `kind:` is missing in the issue body, the planner defaults to `experiment`.

**Routing trap — new code that must RUN to be validated.** Use `kind: experiment` (or `ablation`) **with `code_change: true`**, NOT `kind: implementation`. The `implementation` kind never launches on Vast — it only drafts a PR — so it cannot prove the patch actually *runs* (no NaN/OOM, correct numerics, clean integration with the training backend). Reserve `implementation` for changes whose correctness is fully established by review + local checks. **If the hypothesis can only be confirmed by a training run, the kind is `experiment`/`ablation` even when the patch is large.** Those plans set `code_change: true` and MUST fill `## Correctness invariants` below.

## Hypothesis
<One paragraph. Falsifiable. Contains numeric thresholds. Example: "At setting X with knob Y, observed metric M_target / M_baseline <= 0.10 within `wall_clock_hr` hours, while validation loss stays within 0.05 of baseline at step 5000.">

## Background pointers
- prior findings: findings/M<N>/EXP-NN.md, findings/M<N>/EXP-MM.md   # or "(none)"
- referenced docs: <optional paths/URLs the issue body called out; planner does NOT read any fixed background doc by default>

## Correctness invariants (pre-run gate — REQUIRED when `code_change: true`, else `(n/a)`)

Before an expensive sweep is worth a single GPU-hour, a code change must prove it is *correct*, not merely that it trains. List the cheap, machine-checkable invariants the runner/analyst verify **first**, in a 1–2 step probe; a `hard`-gate failure aborts before the sweep (see `## Analyst predicate`). Keep every entry generic and falsifiable — no "looks right".

```yaml
invariants:
  - name:  off-path parity
    check: "with the feature DISABLED, outputs/bytes/loss are identical to the baseline path"
    gate:  hard            # hard = abort the plan on failure; soft = record metric + continue
  - name:  limiting-case identity
    check: "at the setting where the method must reduce to a no-op / lossless / exact case, error ≈ 0"
    gate:  hard
  - name:  gradient / autograd check
    check: "backward matches the intended analytic form (finite-diff or a known operator); no straight-through unless declared"
    gate:  hard
  - name:  determinism / multi-rank agreement
    check: "same seed ⇒ identical derived state across ranks; no divergence under the parallelism actually used"
    gate:  hard
  - name:  backend integration
    check: "composes with the training backend in use (sharding / activation checkpointing / mixed precision) with no NaN/Inf/OOM in the probe"
    gate:  hard
```

The point is to **fail fast and cheap** on a broken implementation instead of paying for a full sweep that was never going to be interpretable. Expect the first launch of new code to surface backend-integration bugs the runner cannot self-resolve (it emits `STUCK:` and halts) — name the likely failure surfaces in `## Notes for runner` so the operator can triage quickly.

## Experiment sequence (REQUIRED when tuning / testing / comparing)

**If this plan tunes a training job, tests logic, or compares
configurations, list every run needed in execution order.** A single
hypothesis often requires multiple sequential runs (e.g. dense baseline →
method enabled → ablation → re-tightening). Spell them out here so the
operator can plan compute and the runner knows the contract. Free-form
single runs may use `(n/a — single cell, see ## Experiment design)`.

```yaml
sequence:
  - id:        1
    name:      <short slug, e.g. "dense-baseline-reproduce">
    goal:      <one sentence — what this run answers>
    config:    <key deltas from the plan defaults>
    success:   <metric threshold that gates step 2>
    on_fail:   stop  # or "retry with <knob change>", "skip step 2", etc.
  - id:        2
    name:      <e.g. "method-enabled-knob-A">
    depends_on: 1
    goal:      <…>
    config:    <…>
    success:   <…>
  - id:        3
    name:      <…>
    depends_on: 2
    goal:      <…>
    config:    <…>
    success:   <…>
```

Rules:
- Number runs 1, 2, 3, … so the operator and analyst can refer to them
  unambiguously.
- Every run after #1 declares `depends_on:` so the analyst can short-circuit
  the sequence on a hard failure of an earlier step.
- If two runs are truly parallelizable (no data dependency), say so:
  `parallel_with: [<id>]` instead of `depends_on:`. Then the runner may
  fan out across the same provisioned box (see Vast.ai utilization
  discipline below).
- `success:` for each run is a single line gating the next run — NOT the
  plan's overall ## Success criteria. The latter is the headline metric
  for the whole hypothesis.

## Experiment design
```yaml
sweep_grid:
  <knob_a>: [<v1>, <v2>, <v3>]
  <knob_b>: [<v1>, <v2>]
baselines:
  - dense                          # always include unless the plan justifies omission
  - <method>_only
ablations:
  - disable_<component>
seed_replicates:  3
fanout_max:       6                # cap on parallel cells across the sweep
controlled_variables:              # held FIXED across every arm so a confound can't masquerade as a result
  - <e.g. compute/communication budget, step count, dataset, seed — whatever is NOT the variable under test>
```

When arms differ by method, every *other* axis (budget, steps, data, seed) MUST be held fixed **and asserted** — add a machine-checkable box to `## Success criteria` (e.g. `budget(arm_A) == budget(arm_B)` within tolerance) so the comparison is fair by construction.

## Compute budget (HARD CAPS)
```yaml
gpu_count:        1                       # number of Vast.ai instances (single-node default)
gpu_filter_chain:                         # ONLY two sanctioned tiers: 4×H200 (preferred) or 8×H100 — runner tries each in order; first tier with ≥1 offer ≤ max_dph wins
  - "num_gpus=4 gpu_name=H200 gpu_ram>=140 reliability>=0.95 rentable=true verified=true"   # preferred: most VRAM per $
  - "num_gpus=8 gpu_name=H100 gpu_ram>=80 reliability>=0.95 rentable=true verified=true"
max_dph:          24.0                    # per-instance $/hr ceiling
max_gpu_hr:       96                      # total across all cells (runner aborts past this)
max_parallel:     1                       # how many cells may run concurrently
wall_clock_hr:    12                      # soft deadline for analyst's first read
iterations:       3                       # max REVISE child-experiment depth on this lineage
```

The chain above is the planner default. The runner walks it in order and stops at the first tier with ≥1 offer ≤ `max_dph`; it records the chosen tier in the PROVISIONED ledger row so the analyst can attribute results. `per_node_gpus` is implicit in each tier's `num_gpus=` clause — the runner reads the actual count from each handle JSON to set `NGPUS_PER_NODE`.

Per-experiment plans override only when justified — e.g. M0 smoke chains with cheaper SKUs (RTX 3090/4090), multi-node fan-out for a large sweep, or a single-tier chain when a specific GPU is the experimental variable.

## Vast.ai training footprint (KEEP MINIMAL — this is real money)

State the *smallest* run that can still falsify the hypothesis. Every training
step on the Vast.ai box burns GPU-hours; fewer cells × fewer steps is always
better. Count ONLY what actually launches on Vast — local assertions and
operator-invoked plan reviews are free and do not belong here.

```yaml
vast_cells:        <N>          # how many training cells actually launch on Vast
steps_per_cell:    <S>          # global_step target per cell (the smallest that answers the hypothesis)
total_train_steps: <N*S>        # cross-check against max_gpu_hr — if this climbs, re-justify
justification:     "<why this is the minimum number of cells × steps that still falsifies the hypothesis>"
```

### Vast.ai utilization discipline (HARD RULE)

Provisioned instances cost money every second they exist, whether they're
training or not. **Plans MUST NOT leave Vast.ai instances stale**, and MUST
maximize utilization while they're up.

- **One instance for the whole `## Experiment sequence`, not one per run.**
  If steps 1, 2, 3 are sequential on the same hardware tier, the runner
  chains them on the same provisioned box (back-to-back tmux sessions or
  cells, sharing the docker / verl checkout / dataset cache). Tearing down
  + reprovisioning between sequential runs is forbidden — the warm-up cost
  alone (vLLM init, weight load, dataset preprocess) is 5-8 min per
  re-provision and the provision itself takes 1-3 min.
- **Parallelize within the box.** If `## Experiment sequence` declares
  runs `parallel_with:` each other, launch them as concurrent tmux sessions
  on the same box, partitioning GPUs (TP/PP) explicitly. Don't waste idle
  GPUs while one cell is running on a subset.
- **Tear down the instant the science is captured.** When the last cell in
  the sequence writes its `done.flag` and metrics are rsynced to the
  laptop, the runner / Stop hook tears down within the next tick. The plan
  must NOT include "leave it up for the analyst to look at later" — the
  analyst reads metrics from the laptop, not the box.
- **No keep-alive between sessions.** If a session ends with an active
  instance, the Stop hook destroys it. There is no `--keep-warm` flag.
  Always design with "ephemeral box" in mind: state lives in
  `runs/EXP-<ID>/` on the laptop, not in `/workspace/` on the box.
- **GPU-stall watchdog**: the orchestrator's `training-log-monitor` exits
  with `teardown_only` if all GPUs sit ≤ 5% util for 4 consecutive 30-s
  polls while tmux is ALIVE. That budget assumes the operator did NOT
  design a known-idle phase into the plan. If the plan legitimately needs
  an idle gap (e.g. a long evaluation between training cells), say so
  explicitly in `## Notes for runner` so the monitor's thresholds get
  loosened for that window.

**What the plan does NOT specify.** The docker image, container `--shm-size` / `--cap-add`, onstart script (clone fork + pip-install verl `--no-deps`), recommended disk default, and CUDA driver filter all live in the locked Vast.ai Template referenced by `research/.claude/skills/vast-provision/templates.json`. The `vast-provision` skill auto-reads that file and pins the Template; plans MUST NOT name a `template_hash` or `image`. If a future plan needs a different runtime (new vllm major, different image), update `templates.json` and the Vast.ai Template record together — never bypass.

## Success criteria
- [ ] (code_change) every `hard`-gate box in `## Correctness invariants` passes the pre-run probe — this gates the sweep
- [ ] every sweep cell reaches `>= <step_target>` training steps without NaN or non-finite gradients
- [ ] (comparison) controlled variables hold equal across arms (e.g. `budget(arm_A) == budget(arm_B)` within tolerance)
- [ ] dense baseline reproduces published reference within `eval_<metric> <= <bound>`
- [ ] best cell satisfies <method-specific metric threshold>
- [ ] best cell `<quality_metric> - dense_<quality_metric> <= <delta>` at step `<step>`
- [ ] <method-specific staleness/communication/efficiency metric within target>

Criteria must be **machine-checkable** — no "looks good" or "directionally correct". The analyst greps `runs/EXP-<ID>/metrics/*.jsonl` for the values referenced here.

## Verification commands
The analyst runs exactly these commands and captures stdout to `runs/EXP-<ID>/analysis.log`.
```bash
python research/scripts/analyze.py runs/EXP-<ID> --emit verdict.md
python research/scripts/check_budget.py runs/EXP-<ID>
python research/scripts/diff_against_baseline.py runs/EXP-<ID> --baseline <baseline_run>
```

## Analyst predicate
- **PASS** iff every box in `## Success criteria` is checked.
- **REVISE** if at most `iterations` boxes are unchecked AND the analyst can name a concrete next-action knob change for each. Output `next_actions:` as a yaml list of `{knob, from, to, rationale}` objects.
- **STOP** if the hypothesis is falsified (e.g. method underperforms baseline on the headline metric) OR budget exhausted OR `iterations` REVISE cycles already consumed on this lineage.
- **Pre-run gate**: a `hard`-gate failure in `## Correctness invariants` is an automatic **STOP** for this run — the implementation is broken, so fix the code in a new `code_change` cycle rather than spending the sweep. A `soft`-gate failure is recorded and the sweep continues.

## Code change
```yaml
code_change: false                 # if true, runner branches exp/<ID>-<slug> from origin/vast-ai-workload (NEVER from main — main tracks upstream)
target_modules:                    # only meaningful when code_change=true
  - verl/<path>/<module>.py
promote_launcher_as: none          # on PASS, log-writer derives THIS canonical launcher from resolved_params.txt and opens a draft PR into examples/grpo_trainer/. `none` = no promotion (human promotes manually). Name e.g. vast_<scenario>_qwen25_1p5b_grpo_gsm8k.sh
```

If `code_change: true`, the human operator reviews the plan before flipping `status:planned → status:approved`. The runner is the only agent allowed to write under `verl/`, and only while on an `exp/*` branch inside its worktree.

`promote_launcher_as` is the stability valve (see `examples/grpo_trainer/VAST_README.md` §"Stability contract"): the experiment sandbox stays volatile, but a PASS auto-proposes the proven config — taken from `resolved_params.txt`, never from this plan's prose — into a named canonical launcher via a draft PR a human merges. Leave `none` for throwaway probes; set it once a scenario's config is meant to become the reference.

## Dependencies
```yaml
depends_on: [<EXP-N>, <EXP-M>]     # other experiments that must PASS or STOP first
```

The orchestrator refuses to dispatch the runner until every `depends_on` entry has VERDICT in {PASS, STOP}.

## Rescue triggers
```yaml
escalate_if:
  - "STUCK: EXP-<ID>"          # runner hit verl-internal/backend code it can't self-resolve (common for code_change plans)
  - <pattern the runner or analyst might emit in PROGRESS.md>
  - <another pattern>
```

The orchestrator surfaces these PROGRESS.md patterns in `STATUS.md` so the
human operator sees them; the operator decides whether to edit the plan,
abandon the experiment, or keep going.

## Notes for runner
<Anything the planner discovered that the runner should not have to rediscover: verl-internal API gotchas, dataset paths to rsync, environment variables to set on the Vast.ai node, etc. Free-form prose; keep it tight.>

## Notes for analyst
<Optional. Anything specific the analyst should weight more heavily — e.g. "p95 staleness is the proxy for the headline claim; eval_loss is secondary".>
