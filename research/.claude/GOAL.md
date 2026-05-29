# Project north-star — the big goal

> **Agent-readable restatement** of this project's objective. The authoritative
> source is the human-only `../major-goal/` (the research-goal paper +
> `Prompt.md` + `implementation-logic.md`), which **agents must not read**
> (hard rule, `CLAUDE.md §3`). This file is the operator-approved summary that
> agents *are* allowed to read, so every plan, verdict, and PR can be checked
> against one shared target. If this file and `major-goal/` ever conflict,
> `major-goal/` wins and the operator updates this file.

## The goal

**Train Qwen2.5-1.5B-Instruct on GSM8K with the communication-efficient method
ENABLED in verl — end-to-end and STABLE (no grad_norm explosion / NaN) —
reaching reward/accuracy PARITY with (or beating) the dense GRPO baseline,
while MEASURABLY reducing communication.**

**Deliverable:** a reproducible canonical launcher (under
`examples/grpo_trainer/`) that trains comm-efficient GRPO successfully, plus the
`findings/` curve that proves parity and the communication-savings number.

"Done" = all four hold:

1. **Runs** — comm-eff ENABLED trains end-to-end at paper scale with no
   grad_norm explosion, NaN, or divergence.
2. **Parity** — final reward / GSM8K accuracy ≥ the dense GRPO baseline
   (`runs/baseline/`), within noise.
3. **Savings** — communication volume is measured and is materially lower than
   dense, reported as a concrete number.
4. **Reproducible** — a promoted canonical launcher reproduces it (the
   auto-propose-on-PASS → DRAFT PR path in `project.yaml.launchers.promotion`).

## Why code changes are mandatory

The communication-efficient method lives **in the verl source of this fork**
(mask / anchor / spectral / FSDP integration — see `CODE_WALKTHROUGH.md`).
"Train comm-efficient GRPO *using verl*" is therefore not a config toggle alone:
reaching a *stable* run requires **patching that source**. **Code change is in
scope for the project** — landed as `code_change:true` experiments on
`exp/<N>-<slug>` branches — even though individual *diagnostic* issues are
deliberately `code_change:false` (they find the target before anyone patches it).

## Milestone chain to the goal

Where we are now → the goal, as a sequence of gated issues. Each step is gated
by the previous one's verdict; the operator flips `status:planned →
status:approved` at each human gate.

| State | Step | kind | code_change | Status |
|---|---|---|---|---|
| ✅ | Dense GRPO baseline (the control) | experiment | — | landed → `runs/baseline/` |
| ✅ | Comm-eff smoke baseline (method toggles on, small scale) | experiment | — | landed → `runs/communication-baseline/` (PASS) |
| 🔬 | **#13 — diagnose the paper-scale grad_norm explosion** | investigation → experiment | **false** | `status:planned` |
| 🔧 | **Fix issue (opened after #13)** — patch the `verl/` module #13 names; restore stable training at paper scale | implementation | **true** | not yet opened |
| 📈 | Parity run — comm-eff ENABLED vs dense baseline, full GSM8K reward/accuracy curve | experiment | maybe | future |
| 📉 | Communication-savings measurement + report | experiment | maybe | future |
| 🚀 | Promote the proven launcher (DRAFT PR, base `vast-ai-workload`) | — | — | future |

**The critical-path link the operator asked to make explicit:** #13 is
diagnostic *on purpose* — you cannot write the fix until the diagnosis names
*which* mechanism (candidates B–G) dominates the step-1 grad_norm and *which*
`verl/...` module + knob to patch. So the path is:

```
#13 (code_change:false)  →  names dominant cause + target module/knob
        ↓
NEW fix issue (code_change:true)  →  patches it on exp/13-<slug>, base vast-ai-workload
        ↓
stable comm-eff training at paper scale  →  parity run  →  savings report  →  promoted launcher
```

(#13's own *Code change* and *Notes for runner* sections already anticipate this
follow-on; that is why the plan is diagnostic-first rather than wrong.)

## Fixed control variables (do not change without separate justification)

From `CLAUDE.md §1`:

- **Model** — Qwen2.5-1.5B-Instruct (every `findings/` curve is anchored to it).
- **RL loss** — vanilla GRPO (not DAPO / GSPO).
- **Hardware** — multi-GPU only, **4 ≤ num_gpus ≤ 8**, Vast.ai H100/H200 via the
  locked `verl-research-vllm020` template.
- **Dataset** — GSM8K.

## Pointers

- Engineering map of the method → `CODE_WALKTHROUGH.md`
- Project config (authoritative for operating values) → `.claude/project.yaml`
- Dense control launcher → `examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- Comm-eff launcher → `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- Human-only goal source (**DO NOT READ** — hard rule) → `../major-goal/`
