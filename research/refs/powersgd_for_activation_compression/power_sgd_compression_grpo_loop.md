# PowerSGD-Style PP Activation Compression for GRPO

## Source Evidence Used By This Plan

The activation-compression prior art comes from the supervised repository at `/Users/shamane/Documents/comm-eff-ft`, specifically:

- [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:1): the activation PowerSGD-style projector implementation.
- [ddp_train_randomproj.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/ddp_train_randomproj.py:235): the trainer path that installs `PowerSGDProjector` for `--projector-type powersgd`.
- [run_psgd_sql_smoke.sh](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/run_psgd_sql_smoke.sh:1): the supervised smoke setup for `r=128`.
- [run_psgd_sweep_low_r.sh](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/run_psgd_sweep_low_r.sh:1): the supervised low-rank sweep that treats `r=64/32/16/8` as shared-basis compression settings.

Supervised-repo handoff note, copied here so this plan does not depend on opening an external file outside `verl`:

```text
powersgd_projector.py | adaptive low-rank via power iteration.
Worked algorithmically but needs Q sync on fast wire.
```

Original provenance path: `/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/HANDOFF.md`, line 22.

The GRPO-side constraints come from the existing `verl` PRF masking path:

- [activation_mask.py](/Users/shamane/Documents/verl/verl/workers/comm_eff/activation_mask.py:15): current PP-boundary activation masking implementation.
- [state.py](/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py:91): train-only path eligibility, widened to old-logprob only for recompute.
- [engine_workers.py](/Users/shamane/Documents/verl/verl/workers/engine_workers.py:641): old-logprob recompute gating.
- [engine_workers.py](/Users/shamane/Documents/verl/verl/workers/engine_workers.py:828): actor update clean-step and train-path gating.
- [transformer_impl.py](/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:733): forward-hook lifecycle.

## 1. What PowerSGD Is In Gradient Descent

Original PowerSGD is a gradient-compression method for distributed optimization. It compresses a dense gradient matrix before communication.

Suppose one parameter gradient is:

```text
G in R^{m x n}
```

A rank-r PowerSGD approximation uses a skinny random basis:

```text
Q in R^{n x r}
r << min(m, n)
```

The usual sketch is:

```text
P      = G Q              # m x r
P_hat  = orth(P)          # m x r
Q_new  = G^T P_hat        # n x r
G_hat  = P_hat Q_new^T    # m x n
```

So the full gradient G is replaced by the low-rank approximation G_hat.

Communication cost:

```text
dense gradient:       m n
PowerSGD two-factor:  r(m + n)
```

The saved information is not free. The dropped part is:

```text
E = G - G_hat
```

Original gradient PowerSGD usually uses error feedback:

```text
E_t      = previous residual
G_input  = G_t + E_t
G_hat    = PowerSGD(G_input)
E_{t+1}  = G_input - G_hat
```

This works because the residual belongs to a stable semantic object: the same parameter gradient matrix across optimizer steps.

```text
step t:     model.layers.3.mlp.up_proj.weight.grad
step t+1:   model.layers.3.mlp.up_proj.weight.grad
```

The residual has somewhere meaningful to go.

## 2. Why Activation PowerSGD Is Different

For PP-boundary activations, the tensor is not a stable parameter slot. It is tied to:

```text
global step
microbatch
prompt/completion
token positions
current policy weights
boundary layer
```

So an activation residual:

```text
M - M_hat
```

does not cleanly belong to the next training step. The next step has different trajectories and different weights. Reusing activation residuals would mix stale information across samples, which is especially unsafe for GRPO where old logprobs, advantages, response masks, and grouped completions must keep their meaning.

Therefore this plan does not implement activation error feedback.

The repair mechanism is the existing GRPO clean cadence:

```text
compressed step
compressed step
compressed step
compressed step
dense clean step
```

A clean step replaces the compressed step with a true dense optimizer step.

## 3. Supervised Repo Activation Method

The supervised implementation is not full two-factor PowerSGD transport. It is a shared hidden-subspace activation projector.

Implementation reference: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:1).

The supervised method assumes that boundary activations live in a reusable low-dimensional hidden-space subspace. Instead of computing a fresh P_hat Q_new^T factorization for each activation matrix, it keeps a persistent basis:

```text
Q in R^{H x r}
```

For a boundary activation:

```text
M in R^{n x H}
n = tokens in this boundary call
H = hidden size
```

the forward is:

```text
Y     = M Q          # n x r
M_hat = Y Q^T        # n x H
      = M Q Q^T
```

This is exactly what the supervised repo does: flatten `(B, T, H)` to `(B*T, H)`, compute `y = A @ Q`, then reconstruct `A_recon = y @ Q.t()`: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:89).

The supervised trainer installs this path when `projector_type == "powersgd"` and passes `layer_indices`, `hidden_dim`, rank `r`, seed, dtype, fast process group, and `update_cadence` into `PowerSGDProjector`: [ddp_train_randomproj.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/ddp_train_randomproj.py:235).

## 4. Why Shared Basis Instead Of P_hat Q_new^T

This is the load-bearing theory.

Full fresh low-rank activation transport would cost:

```text
r(n + H)
```

because both factors must move.

The supervised method instead treats Q as a shared basis/codebook that both sides already have. Then only Y = M Q moves:

```text
shared-basis payload: n r
```

That is why the supervised repo comments:

```text
per step per layer = B*T*r values
one-time codebook = H*r
```

Reference: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:14).

The theoretical assumption is not simply "fine-tuning is low rank." The stronger and more precise assumption is:

```text
PP-boundary activations have a stable dominant hidden-space subspace across nearby batches/steps.
```

If:

```text
M = U S V^T
```

and Q has converged to the top r right singular vectors V_r, then:

```text
M Q Q^T = M V_r V_r^T = U_r S_r V_r^T
```

That is the best rank-r projection of M.

So the supervised method is a PCA-like hidden-subspace codec:

```text
learn/track Q
send coordinates M Q
reconstruct with Q^T
```

## 5. Supervised Repo PowerSGD Basis Mechanics

The supervised repo initializes one Q per compressed layer/boundary.

Reference: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:74).

Mathematically:

```text
X_L ~ N(0, 1)^{H x r}
seed_L = base_seed * 1_000_003 + layer_idx * 7919
Q_L = orth(X_L)
```

In code:

```python
g = torch.Generator(device=self.device)
g.manual_seed((self.base_seed * 1_000_003 + L * 7919) & 0x7FFFFFFF)
X = torch.randn(self.hidden_dim, self.r, generator=g, device=self.device, dtype=torch.float32)
Q_init, _ = torch.linalg.qr(X)
self.Q[L] = Q_init.to(self.dtype).contiguous()
```

So:

- Q is per boundary layer.
- Q is not per token.
- Q is deterministic from seed and layer index.
- Q is not learned by Adam.
- Q.requires_grad is not used.

The source evidence for these bullets is direct: the supervised projector stores `self.Q` as a dictionary keyed by layer index with each basis shaped `H x r`: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:68). It flattens activations to `A = h.reshape(B * T, H)` and applies the same `Q` to all token rows: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:89). The saved projector state contains only `step` and `Q`, not activation residuals or optimizer-owned parameters: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:141).

Then the repo updates Q under torch.no_grad():

Reference: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:97).

```text
Y       = M Q_t
V       = M^T Y
        = M^T M Q_t
Q_{t+1} = orth(V)
```

This is block power iteration on the activation covariance:

```text
C = M^T M
Q_{t+1} = orth(C Q_t)
```

Repeated updates push Q_t toward the dominant right-singular subspace of the activation stream.

The repo orthonormalizes Q, not P = M Q. That is intentional. If we orthonormalized P, we would destroy the coordinate payload and would need the matching hidden-side factor again:

```text
P_hat = orth(M Q)
Q_new = M^T P_hat
M_hat = P_hat Q_new^T
```

That returns to the full r(n + H) method and loses the shared-basis nr payload advantage.

The supervised-repo handoff note copied at the top of this file says PowerSGD worked algorithmically but needs Q sync on the fast wire. That matters for any real sender/receiver PP deployment.

## 6. Autograd Semantics

The supervised forward is in graph:

```text
M_hat = M Q Q^T
```

Q is treated as a fixed detached basis during the forward. Therefore the backward activation gradient is:

```text
dL/dM = dL/dM_hat Q Q^T
```

The gradient is projected into the same hidden subspace. There is no straight-through estimator.

The supervised repo states this explicitly: forward is `A_recon = (A Q) Q^T`, gradient flows naturally, and no STE is used: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:24).

## 7. Component-by-Component Theory Conversion

This section is the key bridge from original gradient PowerSGD to the supervised activation-compression implementation. The method should be described as **PowerSGD-style activation projection**, not as original DP-gradient PowerSGD with transferred convergence guarantees.

| Original gradient PowerSGD component | Supervised activation-compression equivalent | Theoretical reason for the conversion |
| --- | --- | --- |
| Parameter gradient matrix `G in R^{m x n}` | PP-boundary activation matrix `M in R^{n_tokens x H}` | The compressed object is the activation crossing a pipeline boundary, not a parameter gradient being all-reduced. This changes the semantics of residuals, optimizer state, and guarantees. |
| Random sketch basis `Q in R^{n x r}` used to form `P = GQ` | Persistent per-boundary hidden basis `Q_L in R^{H x r}` | The supervised repo assumes boundary activations have a stable dominant hidden-space subspace across nearby batches/steps. A persistent `Q_L` lets sender/receiver share a codebook and transmit only coordinates. |
| `P = GQ` | `Y = M Q` | `Y` is the actual logical payload: each token sends `r` coordinates instead of `H` activation values. This is why shared-basis cost is `nr`, not `r(n+H)`. |
| `P_hat = orth(P)` | Do **not** orthonormalize `Y = M Q` in the forward payload | Orthonormalizing `Y` destroys its coordinate meaning. To reconstruct after `P_hat = orth(MQ)`, the receiver would also need `Q_new = M^T P_hat`, returning to the full two-factor cost `r(n+H)`. |
| `Q_new = G^T P_hat` as a communicated/paired factor | `V = M^T(MQ)` as a no-grad basis-update statistic | In the supervised repo, `V = C Q` with `C = M^T M` is a block-power-iteration update for the shared basis, not a second factor for the current payload. |
| `G_hat = P_hat Q_new^T` | `M_hat = (M Q) Q^T = M Q Q^T` | The activation consumed downstream is the projection of `M` onto the column space of the shared hidden basis. If `Q = V_r` from `M = U S V^T`, this equals `U_r S_r V_r^T`, the best rank-`r` projection. |
| Error feedback residual `E_{t+1} = G_input - G_hat` | No activation error feedback; use clean dense cadence instead | Gradient residuals belong to stable parameter slots. Activation residuals belong to a particular step/microbatch/trajectory/token/current-weight instance and should not be added to future samples in GRPO. |
| Warm-start `Q` across gradient steps | Warm-start persistent activation basis `Q_L`, but freeze it across each GRPO global step | The supervised repo updates `Q` every compressed forward under `no_grad`; GRPO must adapt this so old-logprob recompute and actor-train forward for the same `comm_eff_global_step` see the same basis. |
| DP all-reduce of compressed gradient factors | Optional PP sender/receiver `Q` or `V` synchronization/accounting | The supervised code sums `V` across the fast subgroup before QR so all fast ranks share `Q`: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:103). A real PP deployment needs equivalent basis agreement or must account for basis sync bytes. |

There is intentionally no per-token `Q`. A per-token basis would destroy the matrix codec: the receiver would need a different `H x r` basis for every token row, so the payload would no longer be just `Y = M Q`. The supervised implementation's `Q[layer_idx]` is shared across the whole flattened activation matrix `A = h.reshape(B*T, H)`: [powersgd_projector.py](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/powersgd_projector.py:89). This row-shared basis is what makes the `nr` payload claim mathematically valid.

The central theoretical reason for keeping a shared basis is therefore:

```text
If Q is shared, communication is Y = M Q with cost nr.
If Q is not shared, or if P_hat is used, a second factor must move and cost returns to r(n + H).
```

The central theoretical reason for updating `Q` by `Q_{t+1} = orth(M^T M Q_t)` is:

```text
This is block power iteration on the activation covariance C = M^T M.
Repeated updates move Q toward the dominant right singular vectors of the activation stream.
```

The central theoretical reason for avoiding activation error feedback is:

```text
The dropped activation residual is sample/trajectory/token/weight specific.
It has no stable next-step semantic slot, unlike a parameter-gradient residual.
```

## 8. Rank And Byte Budgets

For this shared-basis method, payload cost is:

```text
dense:      nH
PRF mask:   q nH
PowerSGD:   nr
```

where:

```text
q = kept fraction = 1 - mask probability
```

So the PRF-matched rank is:

```text
r ~= qH
```

For H = 2048:

```text
90% masking: q=0.10 -> r=205 nearest match
95% masking: q=0.05 -> r=102 nearest match
r=64  -> 32x compression vs dense
r=32  -> 64x compression vs dense
r=16  -> 128x compression vs dense
```

The supervised sweep treats `r=64`, `r=32`, `r=16`, and `r=8` as increasingly aggressive shared-basis compression levels: [run_psgd_sweep_low_r.sh](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/run_psgd_sweep_low_r.sh:1). Its smoke/long scripts also used `r=128` for PowerSGD activation runs: [run_psgd_sql_smoke.sh](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/run_psgd_sql_smoke.sh:1), [run_psgd_sql_long.sh](/Users/shamane/Documents/comm-eff-ft/sim_v2/randomproj/run_psgd_sql_long.sh:1).

Use:

```text
r_p90 = round(0.10 * H)
r_p95 = round(0.05 * H)
```

For strict under-budget matching, use floor instead:

```text
H=2048: r_p90=204, r_p95=102
```

The main experiment ranks should be:

```text
r in {205, 102, 64, 32, 16}
```

with 205/102 as PRF-budget comparisons and 64/32/16 as aggressive shared-basis stress points.

## 9. GRPO-Specific Constraints Already Solved By PRF

verl already has the right GRPO confinement machinery for PRF masking.

PRF masking lives in [activation_mask.py](/Users/shamane/Documents/verl/verl/workers/comm_eff/activation_mask.py:15). It applies in-graph boundary hooks, and its hook asserts path eligibility before firing: [activation_mask.py](/Users/shamane/Documents/verl/verl/workers/comm_eff/activation_mask.py:289).

Important existing GRPO mechanisms:

- Disabled path is a strict no-op: [state.py](/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py:15).
- Default eligible path is only train: [state.py](/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py:91).
- Old-logprob recompute is eligible only when recompute compression is enabled: [state.py](/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py:100).
- Clean cadence disables compression for the whole step: [state.py](/Users/shamane/Documents/verl/verl/workers/comm_eff/state.py:348).
- The trainer stamps the same comm_eff_global_step into old-logprob recompute and actor update: [ray_trainer.py](/Users/shamane/Documents/verl/verl/trainer/ppo/ray_trainer.py:1265), [ray_trainer.py](/Users/shamane/Documents/verl/verl/trainer/ppo/ray_trainer.py:1310).
- Actor update sets mask_active = not clean_step: [engine_workers.py](/Users/shamane/Documents/verl/verl/workers/engine_workers.py:841).
- Old-logprob recompute suppresses compression on clean steps: [engine_workers.py](/Users/shamane/Documents/verl/verl/workers/engine_workers.py:674).
- Hooks register only for eligible forwards and unregister after the batch: [transformer_impl.py](/Users/shamane/Documents/verl/verl/workers/engine/fsdp/transformer_impl.py:733).

PowerSGD activation compression should reuse this lifecycle, generalized from "mask hooks" to "boundary compressor hooks."

## 10. GRPO PowerSGD Port

Add:

```text
verl/workers/comm_eff/powersgd_activation.py
```

Implement a GPU vast-ai (with vast ai machines) testable compressor with supervised-style shared-basis semantics:

```text
M      = flatten(h)                 # n x H
Q      = Q[layer_idx]               # H x r
Y      = M @ Q                      # n x r
M_hat  = Y @ Q.T                    # n x H
return reshape(M_hat, h.shape)
```

Add per-boundary state:

```text
Q[layer_idx]
pending_V[layer_idx]
basis_update_count
last_reconstruction_error[layer_idx]
basis_stats[layer_idx]
```

Initialize Q exactly like the supervised repo:

```text
seed_L = seed * 1_000_003 + layer_idx * 7919
Q_L = orth(randn(H, r; seed_L))
```

Use fp32 QR for initialization and updates; store Q in the activation dtype for forward.

## 11. GRPO Basis Update Rule

This is where GRPO must differ slightly from the supervised loop.

In the supervised repo, the hook updates Q inside each compressed forward. That is fine for supervised next-token training, but GRPO has a paired old-logprob recompute and actor-train forward for the same comm_eff_global_step.

For GRPO:

```text
Q must be frozen for the entire global step.
```

That means:

```text
old-logprob recompute uses Q_t
actor train forward uses Q_t
Q update happens only after the gradient-bearing actor train work for that global step
next global step uses Q_{t+1}
```

Do not update Q during old-logprob recompute.

Accumulate the supervised update statistic only on compressed train forwards:

```text
Y = M Q_t
V += M.detach().float().T @ Y.detach().float()
```

At the end of the actor update, if:

```text
not clean_step
global_step % powersgd.update_cadence == 0
```

then:

```text
Q_{t+1} = orth(V)
clear V
```

This preserves the supervised repo's block-power-iteration theory while preventing GRPO policy-ratio drift caused by old-logprob and train forward seeing different bases.

## 12. Config Changes

Add to [comm_eff.py](/Users/shamane/Documents/verl/verl/workers/config/comm_eff.py:246):

```text
comm_eff.enabled: false
comm_eff.compression_type: dense | prf_mask | powersgd
comm_eff.clean_cadence: 0 | 5 | 10

comm_eff.powersgd.rank: int
comm_eff.powersgd.seed: int = 0
comm_eff.powersgd.update_cadence: int = 1
comm_eff.powersgd.warm_start: true
comm_eff.powersgd.compress_recompute: bool = true
comm_eff.powersgd.sync_basis: false
comm_eff.powersgd.account_basis_sync_bytes: true
```

Recommended defaults for the prototype:

```text
compression_type = powersgd
rank = 102
warm_start = true
update_cadence = 1
compress_recompute = true
clean_cadence = 0 initially, then {5,10}
sync_basis = false for local in-graph simulation
```

If real PP sender/receiver factor agreement is required, sync_basis=true must transmit or broadcast Q or V at update cadence and count H*r basis-sync values. This is the exact concern noted in the supervised handoff: PowerSGD needs Q sync on the fast wire.

## 13. Hook Lifecycle

Refactor the current FSDP hook lifecycle from mask-specific names to compressor-specific names.

Current PRF lifecycle:

```text
_comm_eff_mask_active
_comm_eff_register_mask_hooks
_comm_eff_maybe_set_mask_context
masker.register(...)
masker.unregister()
```

New lifecycle:

```text
_comm_eff_compressor_active
_comm_eff_register_compressor_hooks
_comm_eff_maybe_set_compressor_context
state.boundary_compressor.register(...)
state.boundary_compressor.unregister()
```

ActivationMasker and PowerSGDActivationCompressor should share the same external contract:

```text
register(module)
unregister()
is_registered
set_context(...)
```

For PRF, context includes sample IDs and position IDs.

For PowerSGD, context includes:

```text
global_step
path_tag
forward_only
clean_step
allow_basis_accumulation
```

PowerSGD does not need per-token sample_id for its math, but it must still obey the same GRPO path gates.

## 14. Recompute Semantics

Use a clearer name than PRF's mask_recompute:

```text
powersgd.compress_recompute
```

When true:

```text
old-logprob recompute is compressed with the same Q_t used by train
```

When false:

```text
old-logprob recompute is dense
actor train forward is compressed
```

On clean steps:

```text
old-logprob recompute is dense
actor train forward is dense
no V accumulation
no Q update
clean_steps += 1
```

This mirrors the existing clean-step PRF behavior.

## 15. Metrics

Add PowerSGD metrics parallel to PRF counters:

```text
comm_eff/compression_type
comm_eff/powersgd_applications
comm_eff/powersgd_applications/train
comm_eff/powersgd_applications/old_logprob
comm_eff/powersgd_rank
comm_eff/powersgd_basis_updates
comm_eff/powersgd_update_cadence
comm_eff/powersgd_reconstruction_rel_error
comm_eff/powersgd_q_min_sv
comm_eff/powersgd_q_max_sv
comm_eff/powersgd_q_cond
comm_eff/logical_pp_bytes_dense
comm_eff/logical_pp_bytes_powersgd_y_only
comm_eff/logical_pp_bytes_powersgd_with_basis_sync
comm_eff/clean_steps
```

Reconstruction error:

```text
||M - M_hat||_F / max(||M||_F, eps)
```

Logical bytes:

```text
dense values: nH
powersgd y-only values: nr
basis sync values, if counted: Hr / update_cadence
```

## 16. Experiment Matrix

Run:

```text
dense baseline
PRF p=0.90
PRF p=0.95
PowerSGD r=205
PowerSGD r=102
PowerSGD r=64
PowerSGD r=32
PowerSGD r=16
PowerSGD r=102 clean_cadence=5
PowerSGD r=102 clean_cadence=10
PowerSGD r=16 clean_cadence=5
PowerSGD r=16 clean_cadence=10
```

Track:

```text
GRPO reward/task score
policy loss
KL
entropy
grad norm
update norm
PowerSGD reconstruction error
dense-vs-compressed update cosine
logical PP bytes
step time
memory
NaN counters
Q condition number
basis update count
clean step count
```

## 17. Scope Boundaries

This prototype does not modify:

```text
DP gradient synchronization
optimizer internals
rollout generation
reward computation
ref logprob path
validation path
checkpoint path
Anchor
Spectral
activation error feedback
```

It ports the supervised repo's shared hidden-subspace activation projection into the existing GRPO comm-eff hook/cadence machinery, with the one GRPO-required change that Q is frozen across the paired old-logprob and train forwards for each global step.
