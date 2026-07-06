---
name: build
description: "Phase-1 entry point — file the research issue (thin wrapper over /new-issue). One window per phase: /build files it, /plan (fresh window) plans + gates it, /execute (fresh window) runs it to done."
argument-hint: "<one-liner or full issue text> [kind:experiment|ablation|implementation|brainstorm|literature|analysis]"
allowed-tools: Bash, Read, Skill
---

# /build — file the issue (phase 1 of 3)

A UX alias for stage 1: execute the `new-issue` skill now (Skill: new-issue)
with the same arguments — it owns title/kind/slug derivation, labels, and the
one-claim-per-issue rule. Add nothing on top.

When it prints the issue number, end with exactly this hand-off:

```
Filed #<N>. Next: open a FRESH window → /plan <N>
```

Fresh-window-per-phase is deliberate: it is the context-control layer over
the 7 fine-grained stages (which stay untouched underneath — labels + ledger
carry all state between windows).
