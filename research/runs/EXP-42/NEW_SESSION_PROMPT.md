# EXP-42 follow-up: GPU-FREE deeper analysis of the DENSE run weight behavior

Open a fresh Claude session in `/Users/shamane/Documents/verl/research` and paste the block
below (everything inside the fenced `/goal ...` block). It runs NO GPU and NO training. It is
purely further analysis of the existing local data, to understand how the normal (dense) GRPO
weights behave.

Context a fresh session needs (already true, do not redo):
- The planned 196-matrix study is COMPLETE and decisive (see `.claude/plans/42.md` Progress,
  `runs/EXP-42/verdict.md`). The dense run is regime A (codec off), WandB `er0syc3n`, val@80 0.7695.
- Data we HAVE locally and GPU-free: the dense per-tick weight-trajectory sketch at
  `runs/EXP-42/regimeA/weights/` (196 decoder matrices, 160 ticks, count-sketch k=4096). Also the
  compressed regime B sketch at `runs/EXP-42/regimeB/weights/` if a clean-vs-compressed contrast helps.
- Data we do NOT have and are NOT collecting: the widened all-matrices run (embeddings, RMSNorm
  gains, biases). The box was torn down and the operator has decided NOT to run any more GPU
  training. So this session is analysis-only on the decoder weights we already have.
- Tooling already built and reusable: `runs/EXP-42/build_dense_report.py` (the dense report:
  drift, projectability, linearity R squared, attention vs MLP, layer-depth) and
  `research/scripts/weight_proj_sweep.py` (the horizon sweep + count-sketch re-impl + load_trace).
  The current dense report is `runs/EXP-42/report_dense.html`.

```
/goal A deeper GPU-FREE analysis of how the DENSE (normal GRPO) run's weights behave is COMPLETE. NO GPU, NO training, NO provisioning, NO Vast box at any point: this is purely further analysis of the local data at runs/EXP-42/regimeA/weights (196 decoder matrices, 160 ticks, count-sketch k=4096), building on runs/EXP-42/build_dense_report.py and research/scripts/weight_proj_sweep.py. Done means, as shown by a short checklist I print each tick: (1) I have extended the dense-run weight-behavior analysis beyond the current report_dense.html with these GPU-free studies, each computed from the local sketch trace, each with a plot and a plain description: (a) low-rank / effective-rank structure of the weight-DISPLACEMENT subspace per matrix (stack the per-tick displacement vectors, report the participation ratio or the number of components for 90 percent energy), to test the RLVR-literature claim that RLVR updates are low-rank; (b) the per-matrix projectability distribution and per-matrix crossover horizon (which matrices project furthest, the histogram of crossover h*); (c) an optimal-coefficient sweep: at each horizon find the alpha that minimizes the median weight_proj_ratio and compare it to the naive alpha = h/Delta, to see whether a better-than-naive linear extrapolation exists and how much it would help; (d) the correlation between a matrix's fine-scale linearity R squared and its projectability (does more-linear mean more-projectable); (e) the learned_linear vs fixed_linear residual effect, quantified. (2) All claims are computed from the data (no hand-set numbers); the count-sketch is linear and norm-preserving so displacement norms, cosines, and Gram matrices reconstruct from it; state the sketch caveat (rel std about 1/sqrt(k) about 1.6 percent) wherever it matters. (3) The findings are written into a single self-contained HTML report runs/EXP-42/report_dense_v2.html (extend build_dense_report.py or add a sibling builder; research/ is freely writable), strictly analysis, plots, and descriptions, and ALSO summarized into runs/EXP-42/verdict.md, runs/EXP-42/narrow_findings.md, and runs/SUMMARY.md. Do NOT use em-dashes anywhere. (4) If a study is not computable from the decoder-only sketch (for example anything needing the embeddings or norms, which we did not collect), say so explicitly and skip it rather than inventing data. Print the checklist each tick; stop when all studies are delivered in report_dense_v2.html and summarized. Stop after 80 turns.
```
