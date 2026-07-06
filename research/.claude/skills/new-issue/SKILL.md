---
name: new-issue
description: File a research issue from a one-liner or a full spec, auto-labeled research:claim + kind:*. Stage 1 of the issue lifecycle. Use for "run this experiment", "try X", or any new research claim.
argument-hint: "<one-liner or full issue text> [kind:experiment|ablation|implementation|brainstorm|literature|analysis]"
allowed-tools: Bash, Read
---

# /new-issue — file a tracked research issue

Turn the operator's request into a `research:claim` issue on the research repo
(`gh repo set-default` already points there; issues only, never PRs).

## Steps

1. `source .claude/skills/_lib.sh && ensure_labels` (idempotent).
2. Derive from the input:
   - **title** — imperative, ≤ 80 chars.
   - **kind** — explicit `kind:` in the input wins; else infer:
     GPU training comparison → `experiment`; variation of a PASSed parent →
     `ablation`; code-only → `implementation`; GPU-free offline study →
     `analysis`; open-ended → `brainstorm`. Default `experiment`.
   - **hypothesis** — one falsifiable sentence WITH a numeric threshold. If the
     input has none, propose one from context (baseline ≈0.7657 dense /
     ≈0.7362 comm-eff on the fast surface) and mark it `(proposed — edit me)`.
   - **slug** — 3–40 char kebab-case, self-describing (this becomes the run
     dir, branch, and WandB group: `<N>-<slug>`). No `c1`/`armA` patterns.
3. Body template (short — the plan carries the detail later):
   ```
   kind: <kind>
   slug: <slug>
   hypothesis: <one falsifiable sentence with a number>
   baseline_run: <baseline | EXP-parent | none>
   depends_on: []
   <2-5 lines of free context: what to vary, what to hold fixed, any budget cap>
   ```
4. `gh issue create --title "<title>" --label research:claim --label "kind:<kind>" --body "<body>"`
5. Print the issue number + `Next: /plan <N>`.

## Rules

- One claim per issue. If the input contains two hypotheses, file two issues.
- Never label `status:*` here — that starts at `/plan`.
- Ambiguous input (interactive session): ask 1–2 questions BEFORE filing;
  planning is where questions belong. In an unattended session, file with
  `(proposed — edit me)` markers instead of blocking.
