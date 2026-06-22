# Paper Bibliography

Fresh web search date: 2026-06-22. Labels are one of: directly useful, background, tempting but not applicable, contradicts or challenges the thesis.

## Delayed Gradients and Async Optimization

| Label | Source | Relevance |
| --- | --- | --- |
| background | [Niu et al., 2011, "HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent"](https://arxiv.org/abs/1106.5730) | Classic async SGD; works best under sparsity/fixed objective, so it is not a direct GRPO stale-trajectory fix. |
| directly useful | [Ho et al., 2013, "More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server"](https://www.cs.cmu.edu/~seunghak/SSPTable_NIPS2013.pdf) | Bounded staleness model; useful for anchor age gates and for distinguishing bounded from unbounded stale reuse. |
| directly useful | [Zhang et al., 2016, "Staleness-Aware Async-SGD for Distributed Deep Learning"](https://www.ijcai.org/Proceedings/16/Papers/335.pdf) | Modulates learning rate by staleness; supports age-decay rather than raw stale-gradient injection. |
| directly useful | [Zheng et al., 2017, "Asynchronous Stochastic Gradient Descent with Delay Compensation"](https://arxiv.org/abs/1609.08326) | DC-ASGD compensates the parameter-point delay using Taylor/Hessian approximations; it does not address policy rollout distribution drift. |
| background | [Lian et al., 2015, "Asynchronous Parallel Stochastic Gradient for Nonconvex Optimization"](https://arxiv.org/abs/1506.08272) | Nonconvex ASGD convergence theory; background for delay assumptions and speedup claims. |
| contradicts or challenges the thesis | [Stich and Karimireddy, 2019/2020, "The Error-Feedback Framework: Better Rates for SGD with Delayed Gradients and Compressed Communication"](https://arxiv.org/abs/1909.05350) | Shows delayed/compressed SGD can be robust under fixed-objective assumptions; challenges a blanket anti-staleness claim but not the GRPO distribution-gap thesis. |
| contradicts or challenges the thesis | [Mishchenko et al., 2022, "Asynchronous SGD Beats Minibatch SGD Under Arbitrary Delays"](https://openreview.net/forum?id=4XP0ZuQKXmV) | Strong theoretical challenge to delay pessimism; must be separated from on-policy RL stale trajectories. |
| background | [Assran et al., 2020, "Advances in Asynchronous Parallel and Distributed Optimization"](https://arxiv.org/abs/2006.13838) | Survey for async optimization vocabulary: delays, centralized/decentralized async, convergence-rate effects. |
| background | [Wu and Luo, 2026, "Optimal Asynchronous Stochastic Nonconvex Optimization under Heavy-Tailed Noise"](https://arxiv.org/abs/2601.19379) | Post-2025 optimization theory for async nonconvex training under heavy-tailed noise; not RL-specific. |

## Async Pipeline Parallelism

| Label | Source | Relevance |
| --- | --- | --- |
| background | [Huang et al., 2018/2019, "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"](https://arxiv.org/abs/1811.06965) | Synchronous pipeline baseline; useful contrast because it avoids async stale-gradient semantics with pipeline flush/accumulation. |
| directly useful | [Narayanan et al., 2019, "PipeDream: Generalized Pipeline Parallelism for DNN Training"](https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_pipedream.pdf) | Weight stashing teaches strict version matching between forward and backward passes; maps to snapshot/rollout pairing. |
| directly useful | [Yang et al., 2019/2021, "PipeMare: Asynchronous Pipeline Parallel DNN Training"](https://arxiv.org/abs/1910.05124) | Async pipeline method that tolerates stale updates; useful comparison for damping/velocity routes under fixed-data training. |
| directly useful | [Narayanan et al., 2021, "Memory-Efficient Pipeline-Parallel DNN Training" / PipeDream-2BW](https://proceedings.mlr.press/v139/narayanan21a.html) | Double-buffered weight versions; reinforces version accounting and limited stashing rather than arbitrary stale reuse. |
| directly useful | [Ajanthan et al., 2025, "Nesterov Method for Asynchronous Pipeline Parallel Optimization"](https://proceedings.mlr.press/v267/ajanthan25a.html) | Required 2025 paper. Nesterov look-ahead targets delayed gradients in async pipeline PP; strong fixed-data counterpoint to raw GRPO stale-gradient rejection. |
| directly useful | [PluralisResearch/AsyncPP official code](https://github.com/PluralisResearch/AsyncPP) | Official implementation for the Ajanthan et al. ICML 2025 method; useful if implementation details are later needed. |
| directly useful | [Ajanthan et al., 2026, "AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism"](https://arxiv.org/abs/2601.22442) | Post-2025 async data+pipeline method with look-ahead and sparse averaging; comm-efficient systems comparison. |
| background | [Zhang et al., 2026, "Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation"](https://arxiv.org/html/2602.03515v2) | Post-2025 pipeline staleness paper comparing PipeDream, PipeMare/PipeDream-LR, and Nesterov; possible subspace/curvature inspiration. |

## PPO, GRPO, Off-Policy RL, and Stale Trajectories

| Label | Source | Relevance |
| --- | --- | --- |
| directly useful | [Schulman et al., 2015, "Trust Region Policy Optimization" / TRPO](https://proceedings.mlr.press/v37/schulman15.html) | Trust-region foundation for old-policy trajectory reuse; supports KL/ratio gating. |
| directly useful | [Schulman et al., 2017, "Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347) | PPO alternates sampling with surrogate optimization and allows multiple minibatch epochs only with ratio/clipping safeguards. |
| directly useful | [Shao et al., 2024, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"](https://arxiv.org/abs/2402.03300) | Introduces GRPO as a PPO variant; baseline change does not remove stale rollout issues. |
| directly useful | [Espeholt et al., 2018, "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"](https://arxiv.org/abs/1802.01561) | Actor-learner policy lag is corrected by V-trace; directly relevant distributed RL precedent. |
| directly useful | [Liu et al., 2020, "Off-Policy Policy Gradient with State Distribution Correction"](https://proceedings.mlr.press/v115/liu20a.html) | Explicitly targets state-distribution mismatch, which is one of the stale GRPO gaps. |
| directly useful | [Metelli et al., 2020, "Importance Sampling Techniques for Policy Optimization"](https://jmlr.csail.mit.edu/papers/volume21/20-124/20-124.pdf) | Useful for IS variance/bias tradeoffs and the requirement that behavior and target policies stay close. |
| contradicts or challenges the thesis | [Yao et al., 2025/2026, "Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm"](https://arxiv.org/abs/2509.24203) | Challenges a strict "GRPO is only on-policy" statement; still demands policy regularization and data-distribution control. |
| tempting but not applicable | [Yan et al., 2025, "Learning to Reason under Off-Policy Guidance" / LUFFY](https://arxiv.org/abs/2504.14945) | Off-policy demonstrations can help RLVR, but this is not raw delayed anchor-gradient reuse from stale rollouts. |
| directly useful | [Tyurin et al., 2025/2026, "Asynchronous Policy Gradient Aggregation for Efficient Distributed Reinforcement Learning"](https://arxiv.org/abs/2509.24305) | Post-2025 async policy-gradient aggregation; useful background for distributed RL communication limits. |
| directly useful | [Xu et al., 2026, "GAC: Stabilizing Asynchronous RL Training for LLMs via Gradient Alignment Control"](https://arxiv.org/abs/2603.01501) | Diagnoses stale-aligned gradient overshoot in async LLM RL and proposes projection control. |
| directly useful | [Huang et al., 2026, "Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs" / VCPO](https://arxiv.org/abs/2602.17616) | Uses ESS and variance control under stale rollouts; supports adaptive dose/kill-gates. |
| directly useful | [Shen et al., 2026, "VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training"](https://arxiv.org/abs/2602.10693) | Post-2025 sequence-level IS reshaping for stale LLM RL; direct relevance to long autoregressive rollouts. |
| directly useful | [Lu et al., 2026, "GIPO: Gaussian Importance Sampling Policy Optimization"](https://arxiv.org/abs/2603.03955) | Smooth trust weighting for stale replay; relevant to age/dose-decay alternatives to hard clipping. |
| directly useful | [Guan et al., 2026, "Missing Old Logits in Asynchronous Agentic RL"](https://arxiv.org/abs/2605.12070) | Implementation-critical: off-policy correction needs the right historical logits/snapshots. |
| directly useful | [Lei et al., 2026, "A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization"](https://arxiv.org/abs/2601.22718) | Warns token-level approximations can be unstable under large off-policyness; useful for GRPO sequence-level gates. |
| directly useful | [Lu et al., 2026, "Rollout-Level Advantage-Prioritized Experience Replay for GRPO"](https://arxiv.org/html/2606.04560) | Very recent GRPO replay/staleness-control paper; use cautiously, but relevant to replay, age, and ratio correction. |

## Gradient Compression and Error Feedback

| Label | Source | Relevance |
| --- | --- | --- |
| directly useful | [Vogels, Karimireddy, and Jaggi, 2019, "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization"](https://arxiv.org/abs/1905.13727) | Primary PowerSGD source; supports communication-efficient low-rank compression and error-feedback design. |
| directly useful | [EPFL PowerSGD official repository](https://github.com/epfml/powersgd) | Official implementation reference for PowerSGD. |
| directly useful | [Karimireddy et al., 2019, "Error Feedback Fixes SignSGD and other Gradient Compression Schemes"](https://proceedings.mlr.press/v97/karimireddy19a.html) | Core EF result; directly useful but must be scoped to compression error, not stale policy-objective bias. |
| directly useful | [EPFL error-feedback-SGD official repository](https://github.com/epfml/error-feedback-SGD) | Implementation reference for EF-SGD. |
| contradicts or challenges the thesis | [Anonymous/Recent, 2025, "From PowerSGD to PowerSGD+: Low-Rank Gradient Compression..."](https://arxiv.org/pdf/2509.11254) | Challenges naive PowerSGD convergence; supports adding safeguards/subspace refresh if using low-rank anchor/codec routes. |

## Multi-Timescale, Slow/Fast, and Target Networks

| Label | Source | Relevance |
| --- | --- | --- |
| background | [Borkar and Konda, 1997, "The actor-critic algorithm as multi-time-scale stochastic approximation"](https://link.springer.com/article/10.1007/BF02745577) | Classic two-timescale actor-critic analysis; background for slow/fast learning rates. |
| background | [Mnih et al., 2013, "Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602) | DQN target-network lineage; slow targets stabilize bootstrapping but do not justify stale policy-gradient application. |
| tempting but not applicable | [Zhang et al., 2019, "Lookahead Optimizer: k steps forward, 1 step back"](https://arxiv.org/abs/1907.08610) | Slow/fast weights are tempting as an anchor analogy; however Lookahead averages optimizer trajectories on the same data objective. |
| tempting but not applicable | [Tarvainen and Valpola, 2017, "Mean Teachers are Better Role Models"](https://arxiv.org/abs/1703.01780) | EMA teacher/target analogy for slow anchors; not an optimizer-gradient reuse method. |
| background | [Asadi et al., 2021, "Faster Deep Reinforcement Learning with Slower Online Network"](https://arxiv.org/abs/2112.05848) | Shows slow-network proximity can improve RL robustness; supports slow anchor as regularizer/calibrator rather than stale-gradient source. |

## Search Terms Covered

Delayed gradients; stale gradients; async SGD; delay-compensated ASGD; bounded staleness; stale synchronous parallel; asynchronous pipeline parallelism; PipeDream; PipeMare; Nesterov asynchronous pipeline; off-policy policy gradient; PPO stale trajectories; GRPO stale trajectories; trust region policy optimization; importance sampling variance; actor-learner policy lag; IMPALA V-trace; PowerSGD; error feedback; compressed communication; multi-timescale stochastic approximation; slow/fast networks; target networks; post-2025 async LLM RL; stale replay; old logits; prefix importance ratio.
