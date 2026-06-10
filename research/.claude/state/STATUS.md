# Research Status — 2026-06-10T11:08:00+10:00

## Issue pipeline

| EXP | Title | State | Vast runs | Verdict | Notes |
|---|---|---|---|---|---|
| 26 | Geometry audit + ef_powersgd (#25 follow-up) | Step A COMPLETE — Step B/C/E pending (new session) | 4×H200 (i_40242796, warm+idle — DO NOT teardown) | PASS-STAGE-A | Step-A gate cleared: DECISION=go_C_then_B + retire_sign_replacement(confirmed). H3 confirmed (sign-agree 0.50–0.52 coin-flip at delay_K=0 and 5). H1 spirit confirmed: cos(G_comp,G_corr)=0.717 (~44 deg merger rotation). H2 TRUE (Q_act update-energy capture 0.318, off-principal 0.682). Option-A validity: cos(G_fresh_anchor,G_dense)=0.985. Branch exp/26-geometry-audit-ef-powersgd@5a35fa96c (7 hotfix commits). Steps B/C/E deferred to next session. Issue label stays status:running. |

## Last tick
2026-06-10T11:08:00+10:00 · logged=[26 PASS-STAGE-A] · running=[] · analyzing=[] · blocked=[]

## Budget
$/hr now: $12.84 (1×4H200 warm, box 40242796) · lifetime $174.15 · max_gpu_hr cap: 60 (46.6 used in Step A; Step B needs fresh allocation) · max_dph cap: $24

## Next action
New /loop session: dispatch runner for Step C (rlvr-native Q-content sweep, q_basis sweep at fixed rank 77) on warm box 40242796 (exp/26-geometry-audit-ef-powersgd@5a35fa96c), then Step B (ef_powersgd merger, direction-preserving, no sign term). DECISION=go_C_then_B is locked.
