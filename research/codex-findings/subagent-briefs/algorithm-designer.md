# Algorithm Designer Brief - Delay-Robust Comm-Efficient GRPO

## Confirmed Evidence Assumptions

- EXP-38 says stale dense gradients are weak-to-dead optimizer signals: GSM8K cosine `k1=.507`, `k5=.176`, `k10=.023`, `k20=-.008`; Big-Math is near-decorrelated even at `k1=.018`.
- This is mostly angular drift, not norm drift: norm ratios stay near 1, sign agreement goes to chance.
- Forward activation geometry is stable: `h` rank90 = 1, top-1 overlap ~= 1.0 through `k<=40`; rank-77 overlap is flat, not decaying.
- Backward path is harder: `grad_h` rank90 = 105 on GSM8K, 180 on Big-Math. Symmetric forward/backward codec rank is likely wrong.
- Therefore: never treat a large/variable-lag anchor gradient as a primary optimizer update unless an offline replay proves it can recover live-gradient direction.

## Practical Algorithm Ranking

### A. Demote Anchor To Q/Codec Calibrator

Most likely to survive large variable delay.

Use the slow anchor only to refresh codec bases, ranks, scales, and health stats. Fast ranks train with compressed activation/link updates; stale anchor gradient does not enter the optimizer except as diagnostic signal.

```python
for step in train:
    h = forward_boundary()
    y = compress_forward(h, Q_frozen_or_slow)
    grad_h_hat = decompress_backward(comm_grad_h, Qb_task_rank)
    update_actor(grad_h_hat)

    if anchor_ready():
        Qf = ema_orth(top_subspace(h_sketch))          # low rank: 1/few
        Qb_rank = choose_rank(boundary_grad_h_rank90)  # >= 105/180 class
        broadcast(Qf, Qb_rank, codec_stats)
```

Offline kill-gates from EXP-38:
- Pass if replayed forward reconstruction at rank `1..5` stays high and stable for `k<=40`.
- Pass if backward rank policy picks at least GSM8K `>=105` and Big-Math `>=180` or shows no quality loss at lower rank.
- Kill symmetric rank if backward reconstruction at `r=77` misses the `grad_h` rank90 target.

### B. Forward/Backward Codec Split

Concrete version of A. Forward link: rank-1/few, long-lived `Q`. Backward link: task-adaptive higher rank, fresher scale/stat calibration.

```python
rank_f = min_rank_for(h_energy >= .99)      # expect 1/few
rank_b = min_rank_for(grad_h_energy >= .90) # task-specific
send_forward = low_rank(h, rank_f, Qf)
send_backward = low_rank(grad_h, rank_b, Qb)
```

Offline gates:
- Sweep ranks on `boundary_h` and `boundary_grad_h` captures separately.
- Require forward rank-few to preserve `>=99%` energy on both tasks.
- Require backward codec not to regress Big-Math: if rank cap needed exceeds comm budget, do not run GPU until budget/routing changes.

### C. Age-Decayed Stale Correction

Safety belt if stale correction must remain enabled. Make stale dose decay with actual age, not configured cadence.

```python
age = live_tick - anchor_tick
lambda_age = lambda0 * exp(-age / tau)      # or lambda0 * mu**age
G = G_comp + lambda_age * delta_anchor
if age > age_max or stale_cos_proxy < floor:
    lambda_age = 0
```

Offline gates:
- Replay EXP-38 lag table: choose `tau` so weight at GSM8K `k10` and `k20` is near zero; Big-Math `k>=1` should already be heavily suppressed.
- Kill if integrated stale dose remains material where cosine is chance-level.
- This is stability-only; it cannot surpass dense because it discards bad signal rather than adding new information.

### D. Adaptive Lambda By Measured Staleness

Use measured residuals/health stats to bound the correction dose.

```python
rho = norm(delta_anchor) / (norm(G_comp) + eps)
age = live_tick - anchor_tick
lambda_eff = min(lambda_cap, lambda0 / (1 + kappa * rho))
lambda_eff *= exp(-age / tau)
G = G_comp + lambda_eff * delta_anchor
```

Offline gates:
- Estimate `rho`, lag cosine, sign agreement from paired `g_dense` tensors.
- Fit monotone `lambda_eff(k,rho)` that goes low for GSM8K `k>=10` and Big-Math `k>=1`.
- Kill if lambda remains high when sign agreement is ~= chance or if it improves norm metrics without cosine lift.

### E. Trust-Region / IS-Corrected Stale Data

Only plausible when stale samples carry enough policy-overlap. Treat stale anchor contribution as off-policy GRPO data with clipping and hard KL gates.

```python
ratio = exp(logp_live(tokens) - logp_anchor(tokens))
w = clip(ratio, 1 - eps, 1 + eps)
if mean_kl_live_anchor > kl_max or ess(ratio) < ess_min:
    stale_grad = 0
else:
    stale_grad = grpo_grad(tokens, adv_anchor, weight=w)
G = G_comp + lambda_eff * stale_grad
```

Offline gates:
- EXP-38 has aggregate GRPO drift/sidecars; use them first as a proxy. If per-sample logprobs are unavailable, this route is not GPU-ready.
- Kill when proxy KL/clipfrac reaches half-drift within ~1 global step, matching EXP-38 H2, unless per-sample replay shows ESS remains acceptable.
- Require stale-corrected gradient cosine lift over raw stale gradient on held-out lags; otherwise it is just expensive stale reuse.

### F. Learned Projection / Extrapolation

Speculative but testable without GPU. Learn `R(age, stats)` mapping stale low-rank gradient coordinates to live coordinates. Must be cross-rank-identical and variable-age conditioned.

```python
# offline train on captured pairs only
z_old = U.T @ g[t-age]
target = U.T @ g[t]
R = fit_regularized_map(features=[age, drift_stats], x=z_old, y=target)

# online
z_hat = R(age, stats) @ (U.T @ g_anchor)
g_hat = U @ z_hat
G = G_comp + lambda_proj * project_safe(g_hat - G_comp)
```

Offline gates:
- Train on EXP-38 early/mid held-in pairs, evaluate on held-out ticks and lags, separately per dataset.
- Minimum go gate: GSM8K `cos@k5 .176 -> >= .40`; any Big-Math success must show lift from `k1 .018`, not averaged with GSM8K.
- Diagonal trap: compare full map vs diagonal-only map. If lift is diagonal-only, label parity/stability, not surpass.
- Kill if learned map fails age interpolation or needs rank-local/private fitting.

### G. Cross-Rank Second Moment

Promising surpass route because it can use information outside stale `sigma(M)`: disagreement/noise geometry across ranks.

```python
for rank in ranks:
    g_i = local_compressed_or_sketch_grad()
mean_g = all_reduce_mean(g_i)
second = all_reduce_mean((g_i - mean_g) ** 2 or low_rank_outer(g_i))
precond = build_shared_preconditioner(second)
G = precond(mean_g)
```

Offline gates:
- EXP-38 single-run tensors cannot fully validate cross-rank disagreement; use them only for rank/curvature sizing.
- Pre-GPU requirement: define exact all-reduced sufficient stats and comm budget.
- Kill if proposed stats are per-rank divergent or collapse to Adam-style diagonal only.

### H. Curvature / Off-Diagonal Route

The only stale-gradient reuse route with a credible "surpass" story: estimate `H * delta_theta` in the active subspace to un-rotate stale gradients.

```python
dg = g[t] - g[t-age]
dtheta = theta[t] - theta[t-age]
fit B such that dg ~= B @ lowrank(dtheta)  # block/sketched/off-diagonal
g_hat = g_anchor + B @ lowrank(theta_live - theta_anchor)
```

Offline gates:
- Fit block/sketched `B` on EXP-38 `(theta, g_dense)` pairs.
- Require off-diagonal map to beat diagonal DC-ASGD-style baseline on held-out cosine.
- Kill if residual is mostly distribution/off-policy drift rather than parameter curvature, which EXP-38 H2 suggests may dominate.

## Overall GPU Gate

Run no GPU experiment for stale-gradient reuse until EXP-38 replay shows held-out cosine lift, task-separated, with diagonal-vs-off-diagonal attribution. The immediate engineering path is A+B; C+D are bounded safety levers; E-H are research routes with cheap offline kill switches.
