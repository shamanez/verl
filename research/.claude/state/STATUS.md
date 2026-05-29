# Research Status — fresh cycle

No active experiment. The orchestrator rewrites this file each tick while its
loop is running.

Project state:
- **Baseline = dense GRPO == method OFF** — proven (the bar to match).
- **Comm-eff implementation correct; masking under test** — plain masked GRPO
  does not yet learn at high mask rates. Next: a mask-rate sweep. Anchor +
  spectral correction stay OFF until masking learns.

See `../../LOG.md` and `../../runs/SUMMARY.md`.
