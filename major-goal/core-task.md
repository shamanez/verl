# Communication-Efficient Pipeline Adaptation: Agent Implementation README

This README explains how to implement the paper's two-circuit, communication-efficient adaptation method inside a TRL-like post-training stack. It is written for an agent that needs to modify a trainer, a model wrapper, and distributed communication code without changing the training objective.

The core idea is simple: reduce pipeline-parallel activation traffic on the main training path, then use a slower uncompressed path only to estimate reliable gradient geometry.

## 1. What problem this solves

In pipeline-parallel training, adjacent model shards exchange hidden states in the forward pass and activation gradients in the backward pass. On low-bandwidth links, this activation traffic can dominate wall-clock time. Data-parallel gradient compression alone does not fix this, because it reduces replica synchronization but not the activation exchange between pipeline stages.

This method targets the pipeline boundary itself:

- The fast circuit sends sparse masked activations across pipeline boundaries.
- The backward pass reuses the same sparsity pattern because masking is in the autograd graph.
- A separate anchor circuit occasionally runs unmasked forward/backward passes and sends delayed clean gradients back.
- A spectral correction step uses the anchor gradients as geometry, not as direct updates.

The method is not a replacement for LoRA, QLoRA, PEFT, ZeRO, FSDP, or data-parallel compression. It addresses a different bottleneck: bytes crossing pipeline stage boundaries.

## 2. How this differs from LoRA and other efficiency methods

### LoRA / PEFT

LoRA reduces trainable parameter count and optimizer-state memory. It does not reduce the dense hidden-state tensor that crosses pipeline boundaries. Even if only adapters are trainable, every stage still needs the full forward activation and the full backward activation-gradient tensor.

This method can be combined with LoRA:

- LoRA gives fewer trainable weights and smaller optimizer state.
- Pipeline activation masking gives fewer bytes over pipeline links.
- Spectral correction can be applied to full fine-tuning matrices or only trainable adapter matrices.

### Quantized activation transport

Activation quantization compresses payloads off-graph and then reconstructs dense tensors. That can decouple forward and backward compression patterns. The method here applies the mask in the graph, so the backward pass naturally inherits the exact same mask.

### Top-k activation sparsification

Top-k is attractive for data-parallel gradient compression, but it is problematic at pipeline activation boundaries.

- The selected coordinates depend on the activation values, creating structured bias.
- The forward top-k coordinates are not necessarily the coordinates with important backward gradients.
- Error feedback is hard to reuse because activations are microbatch-local and step-local.
- The spectral filter is designed to contract approximately zero-mean random mask noise, not deterministic top-k bias.

### FSDP / ZeRO / tensor or sequence parallelism

These systems shard parameters, optimizer state, gradients, attention computation, or sequence dimensions. They do not by themselves define a sparse activation protocol across layer-wise pipeline boundaries. They may still be used under or around this design, but the activation masking hook must live at the pipeline boundary.

### Data-parallel compression

The paper composes pipeline activation masking with data-parallel compression. This is important: if pipeline communication is reduced but data-parallel all-reduce remains dense, the bottleneck simply moves. In the paper's setup, PowerSGD plus Streaming DiLoCo is used for the data-parallel axis, while masking plus anchor correction is used for the pipeline axis.

## 3. The algorithm in one page

Assume a 2D worker mesh:

- `X` = data-parallel dimension.
- `Y` = pipeline-parallel dimension.
- Fast circuit = `(X - Z) x Y`, with masked pipeline communication.
- Anchor circuit = `Z x Y`, with unmasked pipeline communication.
- The paper fixes `Z = 1` in experiments.

Training loop:

1. The fast circuit receives a normal training batch.
2. At configured pipeline boundaries, it applies a deterministic random mask to the hidden state.
3. Only retained hidden dimensions are transmitted to the next stage.
4. The same mask is used in backward automatically because the mask multiplication is part of autograd.
5. Data-parallel replicas synchronize gradients using the chosen data-parallel compressor.
6. If a delayed anchor gradient has arrived, update the anchor-gradient EMA.
7. For each targeted matrix, run spectral correction on the fast masked gradient.
8. AdamW steps on the corrected gradient.
9. Independently, the anchor circuit periodically pulls a fast weight snapshot, runs unmasked forward/backward, and pushes gradients back.

## 4. Theoretical assumptions that must be respected

The method is designed for post-pretraining adaptation, not training from random initialization. The analysis depends on the following assumptions.

### 4.1 Objective smoothness

The population objective is assumed to be smooth and bounded below. This gives the standard descent inequality used by nonconvex SGD analyses.

Implementation implication: avoid unstable learning-rate schedules, very large gradient spikes, or excessive clipping artifacts that violate the bounded-update regime.

### 4.2 Random mask noise is approximately unbiased

The masked gradient is modeled as:

```text
g_mask_t = grad F(w_t) + epsilon_t
E[epsilon_t | history] = 0
E[||epsilon_t||^2] <= sigma_m^2
```

This requires the mask to be generated independently of the model state and activation magnitude. A deterministic pseudorandom function is used so sender and receiver agree on the mask without transmitting mask indices.

Implementation implication: use PRF-based random masks, not top-k masks.

### 4.3 Anchor gradients are delayed but clean

The anchor gradient is modeled as:

```text
g_anchor_tau = grad F(w_tau) + zeta_tau
current_step - tau <= Delta
E[zeta_tau] = 0
E[||zeta_tau||^2] <= sigma_a^2
```

The anchor gradient is too stale to be used as the direct update. It is only trusted as a low-rank geometry estimate.

Implementation implication: never replace the fast gradient with the anchor gradient. Use the anchor gradient to update the spectral basis only.

### 4.4 Updates are bounded

The update direction has bounded second moment:

```text
E[||u_t||^2] <= G^2
```

Implementation implication: keep gradient clipping, mixed-precision scaling, and optimizer state numerically stable.

### 4.5 Signal preservation

The spectral filter must preserve most of the true gradient:

```text
E[||(I - P_t) grad F(w_t)||^2] <= kappa E[||grad F(w_t)||^2] + stale_error
```

This is expected when adaptation gradients lie near a low-dimensional subspace and the anchor basis does not rotate too quickly.

Implementation implication: target adaptation workloads with moderate learning rates and short-to-medium training horizons. Monitor whether anchor subspace alignment collapses.

### 4.6 Noise contraction

The spectral filter must contract random mask noise:

```text
E[||P_t epsilon_t||^2] <= rho sigma_m^2
```

For matrix-shaped gradients, a low-rank two-sided filter contracts isotropic noise strongly.

Implementation implication: use fresh PRF masks per step/microbatch/boundary, and log filter energy to confirm the basis is low-rank enough to denoise.

## 5. Boundary masking design

### 5.1 Where boundaries are set

Boundaries are the layer cuts between pipeline stages. For a decoder-only transformer with 16 blocks and `pp_size = 8`, a natural layout is two blocks per stage:

```text
stage 0: blocks 0-1
boundary after block 1
stage 1: blocks 2-3
boundary after block 3
stage 2: blocks 4-5
boundary after block 5
...
stage 6: blocks 12-13
boundary after block 13
stage 7: blocks 14-15
```

So the masked boundary layer ids are:

```python
boundary_layers = [1, 3, 5, 7, 9, 11, 13]
```

For other models, derive boundaries from:

```python
layers_per_stage = ceil(num_hidden_layers / pp_size)
boundary_after = last_layer_index_of_each_stage_except_final
```

Do not mask inside attention or MLP submodules. Mask only the tensor that is actually sent from one pipeline stage to the next.

### 5.2 Mask shape

For hidden states:

```text
h: [tokens, hidden_size] or [micro_batch, seq_len, hidden_size]
mask: same broadcastable shape over hidden dimension
```

The paper describes retaining `K = round((1 - p) * H)` hidden dimensions per token at mask probability `p`.

For `p = 0.95`, retain 5% of hidden dimensions. This is 20x forward and 20x backward activation compression, before metadata and transport overheads.

### 5.3 PRF seed contract

Sender and receiver must generate the same mask without communicating indices.

Use a seed tuple like:

```text
seed = hash64(
    run_seed,
    global_optimizer_step,
    gradient_accumulation_microstep,
    pipeline_boundary_id,
    sequence_shard_id,
    microbatch_id,
    hidden_size,
    mask_version
)
```

Rules:

- Never include activation values in the seed.
- Include enough batch metadata to avoid mask reuse collisions.
- Keep the seed stable across forward recomputation if gradient checkpointing is enabled.
- Save `mask_version` in checkpoints so future code changes do not silently alter masks.

### 5.4 In-graph masking

Conceptually:

```python
class MaskedBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, boundary_id, step, microstep, cfg):
        mask = prf_mask(h.shape, boundary_id, step, microstep, cfg)
        scale = 1.0 / (1.0 - cfg.mask_prob)
        ctx.save_for_backward(mask)
        return h * mask * scale

    @staticmethod
    def backward(ctx, grad_out):
        (mask,) = ctx.saved_tensors
        return grad_out * mask, None, None, None, None
```

For real distributed pipeline execution, the forward must send only retained values, and the receiver must reconstruct the dense masked tensor before continuing. The backward path must send only retained gradient values in the reverse direction.

### 5.5 Sparse payload format

Because the mask is deterministic, the payload can contain only values:

```text
payload = {
  dtype,
  original_shape,
  retained_values_contiguous
}
```

No indices are needed. The receiver regenerates the mask, scatters values into the retained positions, fills dropped positions with zero, and applies the scaling convention.

## 6. Anchor circuit design

The anchor circuit is a separate pipeline group using the same layer partitioning as the fast circuit.

### 6.1 Anchor lifecycle

Each anchor cycle:

1. Pull a recent weight snapshot for the local pipeline shard.
2. Run an unmasked forward/backward pass on a batch from the same training distribution.
3. Collect gradients for targeted matrices.
4. Push gradients to the fast circuit asynchronously.
5. Repeat without blocking the fast circuit.

### 6.2 What is transferred

Minimum viable transfer:

```text
from fast to anchor:
  model shard weights, or delta since last snapshot

from anchor to fast:
  parameter name
  gradient tensor or low-rank representation
  anchor source step
  batch metadata hash
  dtype
```

The paper uses the anchor gradient as a geometry signal. It is acceptable to compress anchor-gradient transport, but do not mask anchor activations.

### 6.3 Staleness contract

Log staleness in fast steps:

```python
staleness = fast_global_step_when_received - fast_global_step_of_snapshot
```

The paper's practical operating region is roughly 20-25 fast steps at 95% masking in a 200 Mbps setting. A staleness around 50 fast steps was outside the good region in the reported ablation.

### 6.4 Do not block fast training

Anchor updates should enter a nonblocking queue:

```python
anchor_queue.push({
    "source_step": source_step,
    "grads": anchor_grads_by_param,
})
```

The fast optimizer consumes whatever has arrived before the optimizer step. If no new anchor gradient has arrived, reuse the existing anchor EMA.

## 7. Spectral correction optimizer

For every targeted matrix `W`, maintain a separate anchor EMA:

```text
M_anchor = beta_anchor * M_anchor + (1 - beta_anchor) * G_anchor
```

Do not merge this with AdamW's first moment. AdamW's first moment is updated every fast step and is dominated by masked-gradient noise.

### 7.1 Filter construction

Given:

```text
M_anchor = U S V^T
```

Define:

```text
d_i = s_i / (s_i + tau_p)
D = diag(d_i)
```

For the masked fast gradient `G_mask`:

```text
X = U^T G_mask V
G_filt = U D X D V^T
G_proj = alpha * G_mask + (1 - alpha) * G_filt
```

Then AdamW consumes `G_proj`.

### 7.2 Practical SVD choices

Full SVD of every matrix every step is expensive. Use one of these:

- Update SVD only when a new anchor gradient arrives.
- Use truncated SVD with rank 16, 32, 64, or 128.
- Restrict correction to large linear matrices first: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head` if trainable.
- Store `U`, `S`, and `V` in BF16/FP16 for memory, but compute the SVD/filter in FP32 when possible.
- Fall back to uncorrected gradients if SVD fails or the anchor tensor contains non-finite values.

### 7.3 Optimizer wrapper sketch

```python
class SpectralCorrectionAdamW(torch.optim.Optimizer):
    def __init__(self, base_optimizer, cfg, named_parameters):
        self.base = base_optimizer
        self.cfg = cfg
        self.state_by_name = {
            name: AnchorState() for name, p in named_parameters if should_correct(name, p)
        }

    @torch.no_grad()
    def ingest_anchor_gradients(self, packet):
        for name, g_anchor in packet["grads"].items():
            st = self.state_by_name.get(name)
            if st is None:
                continue
            st.update_ema(g_anchor, beta=self.cfg.beta_anchor)
            st.update_svd(tau_p=self.cfg.tau_p)

    @torch.no_grad()
    def apply_spectral_correction(self):
        for group in self.base.param_groups:
            for p in group["params"]:
                name = self.param_to_name[p]
                st = self.state_by_name.get(name)
                if st is None or p.grad is None or not st.has_basis:
                    continue
                g = p.grad.data
                g_filt = st.filter(g)
                p.grad.data = self.cfg.alpha * g + (1.0 - self.cfg.alpha) * g_filt

    def step(self, closure=None):
        self.drain_anchor_queue()
        self.apply_spectral_correction()
        return self.base.step(closure=closure)
```

The optimizer wrapper is a stable integration point because it works across supervised, preference, and reward-driven trainers as long as gradients land in `param.grad` before `optimizer.step()`.

## 8. How to fit this into a TRL-like trainer

TRL trainers are thin task-specific layers around the Transformers/Accelerate training stack. The clean integration is to keep dataset formatting, chat templates, collators, metrics, and objective-specific loss exactly as they are, while inserting pipeline transport and spectral correction below the loss.

### 8.1 Files to add

Suggested module layout:

```text
trl/trainer/pipeline_anchor_config.py
trl/trainer/pipeline_masking.py
trl/trainer/pipeline_stage.py
trl/trainer/anchor_worker.py
trl/trainer/spectral_correction.py
trl/trainer/compressed_sft_trainer.py
examples/scripts/sft_with_pipeline_anchor.py
tests/test_pipeline_prf_mask.py
tests/test_spectral_correction.py
tests/test_anchor_queue.py
```

### 8.2 Config object

```python
@dataclass
class PipelineAnchorConfig:
    enabled: bool = False
    pp_size: int = 1
    dp_size: int = 1
    anchor_dp_replicas: int = 1

    mask_prob: float = 0.95
    boundary_layers: list[int] | None = None
    mask_seed: int = 1234

    alpha: float = 0.3
    tau_p: float = 1e-3
    beta_anchor: float = 0.99
    svd_rank: int = 64
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    )

    anchor_refresh_every: int = 20
    max_anchor_staleness: int = 30
    eval_disable_masking: bool = True

    dp_compression: str | None = "powersgd_streaming_local_sgd"
```

Add this config as an optional field on the trainer config. If absent or disabled, the trainer should behave exactly like upstream TRL.

### 8.3 Model wrapping

A normal `SFTTrainer` expects one process to call:

```python
outputs = model(**inputs)
loss = outputs.loss
```

True pipeline parallelism changes that assumption. Each process owns only a shard of layers, and hidden states are sent stage-to-stage. Therefore, `compute_loss` alone is not the right place to implement this method.

Use one of two approaches:

#### Approach A: trainer-compatible wrapper

Wrap the model in a `PipelineMaskedModel` whose `forward()` internally runs the pipeline stage protocol. The trainer still sees a callable `nn.Module`.

This is easier to integrate with TRL APIs but requires careful handling of microbatch scheduling, loss aggregation, and distributed autograd.

#### Approach B: custom inner training loop using TRL components

Reuse TRL's config, tokenizer processing, data collator, dataset packing, logging conventions, and save/push utilities, but replace the core training loop with a pipeline engine.

This is safer for real pipeline parallelism because the training loop can explicitly schedule:

```text
microbatch forward wave -> microbatch backward wave -> gradient sync -> spectral correction -> optimizer step
```

### 8.4 Trainer subclass sketch

```python
class CompressedPipelineSFTTrainer(SFTTrainer):
    def __init__(self, *args, pipeline_anchor_config=None, **kwargs):
        self.pipeline_anchor_config = pipeline_anchor_config or PipelineAnchorConfig(enabled=False)
        super().__init__(*args, **kwargs)

        if self.pipeline_anchor_config.enabled:
            self.pp_context = build_pp_dp_process_groups(self.pipeline_anchor_config)
            self.model = PipelineMaskedModel(
                model=self.model,
                pp_context=self.pp_context,
                cfg=self.pipeline_anchor_config,
            )
            self.anchor_worker = AnchorWorker(
                model_factory=self._anchor_model_factory,
                pp_context=self.pp_context,
                cfg=self.pipeline_anchor_config,
            )

    def create_optimizer(self):
        super().create_optimizer()
        if self.pipeline_anchor_config.enabled:
            self.optimizer = SpectralCorrectionAdamW(
                base_optimizer=self.optimizer,
                cfg=self.pipeline_anchor_config,
                named_parameters=self.model.named_parameters(),
            )

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.pipeline_anchor_config.enabled:
            model.set_masking_enabled(True)
            model.set_global_step(self.state.global_step)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.pipeline_anchor_config.enabled and self.pipeline_anchor_config.eval_disable_masking:
            model.set_masking_enabled(False)
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        finally:
            if self.pipeline_anchor_config.enabled:
                model.set_masking_enabled(True)
```

The exact method signatures may need adjustment for the installed Transformers/TRL version. Keep the integration point concept: mask during training forward/backward, disable during evaluation/generation, and correct gradients immediately before the base optimizer step.

### 8.5 What should not change in the trainer

Do not change:

- dataset format handling
- chat template application
- completion-only or assistant-only loss masking
- reward function APIs
- objective-specific loss formulas
- generation code used for evaluation or sample creation

The method is below the objective: it changes how activation tensors move between pipeline stages and how gradients are denoised before the optimizer step.

## 9. Compatibility with generation-and-reward training stacks

For a trainer that alternates generation, scoring, and update phases, integrate this method only in the update phase.

Recommended separation:

```text
generation workers:
  use dense inference
  no activation masking
  no anchor correction
  no spectral optimizer state

training workers:
  use pipeline activation masking on backward-enabled forward passes
  run anchor circuit asynchronously
  apply spectral correction before optimizer step
```

Why:

- Inference should be deterministic and quality-preserving.
- The communication bottleneck targeted here is training-time pipeline activation exchange.
- The anchor signal is based on training gradients, so it only exists during backward passes.

The output contract for a training library is simple:

```text
input: batch with token ids, masks, labels/rewards/advantages as required by the objective
loss: unchanged objective-specific scalar
gradients: produced through masked pipeline forward/backward
optimizer: receives spectrally corrected gradients
checkpoint: stores normal model weights plus correction state
```

## 10. Experiment boundaries to reproduce first

Start with the paper's stable region before trying more aggressive settings.

### 10.1 Default operating point

```yaml
mask_prob: 0.95
retained_fraction: 0.05
activation_compression: 20x forward, 20x backward
anchor_staleness_target: 20
spectral_tau_p: 1.0e-3
spectral_alpha: 0.3
anchor_dp_replicas: 1
```

### 10.2 Mesh

For a 16-layer decoder-only model:

```yaml
fast_mesh: 7 x 8
anchor_mesh: 1 x 8
pp_size: 8
fast_dp_replicas: 7
anchor_dp_replicas: 1
boundary_layers: [1, 3, 5, 7, 9, 11, 13]
```

For smaller experiments:

```yaml
fast_mesh: 3 x 4
anchor_mesh: 1 x 4
pp_size: 4
boundary_layers: [3, 7, 11]
```

Adjust boundary layers for the model depth.

### 10.3 Sweep knobs

Run these ablations:

```yaml
mask_prob: [0.90, 0.95, 0.99]
anchor_staleness: [10, 20, 30, 50]
tau_p: [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-1]
alpha: [0.0, 0.3, 0.5, 1.0]
```

Expected behavior from the paper:

- `p = 0.90` and `p = 0.95` are workable with anchor correction.
- `p = 0.99` is too sparse in the reported setting.
- staleness near 50 fast steps is too stale in the reported setting.
- `alpha = 1.0` is masked-only and removes spectral correction.
- `alpha = 0.0` overtrusts stale filtered geometry.
- `alpha = 0.3` was the default stable point.

## 11. Logging and validation

Add these metrics to the trainer logs.

### 11.1 Communication metrics

```text
pp_forward_bytes_per_token
pp_backward_bytes_per_token
dp_sync_bytes_per_token
anchor_weight_pull_bytes
anchor_gradient_push_bytes
mask_retained_fraction
```

### 11.2 Staleness and anchor metrics

```text
anchor_packets_received
anchor_mean_staleness_steps
anchor_p95_staleness_steps
anchor_dropped_for_staleness
anchor_ema_norm_by_module
```

### 11.3 Spectral metrics

```text
spectral_rank_used
spectral_energy_top_r
spectral_filter_mean_d
spectral_filter_min_d
spectral_filter_max_d
grad_norm_masked
grad_norm_filtered
grad_norm_projected
cosine_masked_projected
nonfinite_anchor_count
svd_failure_count
```

### 11.4 Quality metrics

Use the same evaluation harness and validation tasks as the dense baseline. Evaluation should run with masking disabled unless explicitly measuring masked inference behavior.

## 12. Failure modes and guardrails

### Failure: quality collapses with masking only

This is expected. Masking without anchor correction is noisy. Check that anchor packets arrive and correction is actually applied before optimizer step.

### Failure: anchor correction has no effect

Check:

- Are target module names matching real parameter names?
- Are anchor gradients mapped to the same parameter names as fast gradients?
- Is the SVD rank nonzero?
- Is `alpha < 1.0`?
- Is the correction applied after data-parallel gradient sync but before AdamW step?

### Failure: staleness too high

Check:

- bandwidth throttling
- anchor weight-pull time
- anchor microbatch size
- unmasked pipeline pass time
- whether multiple anchor cycles are queued concurrently

Drop anchor packets older than `max_anchor_staleness`.

### Failure: masks mismatch across stages

Check:

- global step agreement
- microbatch id agreement
- boundary id agreement
- gradient accumulation counter agreement
- model stage layout agreement
- seed hash implementation agreement

Add a debug mode that sends mask checksums for the first few steps.

### Failure: activation reconstruction overhead dominates

Use fused gather/scatter kernels or pack retained dimensions contiguously per token. Avoid Python loops over tokens.

### Failure: SVD overhead dominates

Compute SVD only on anchor arrival, use truncated SVD, and restrict to selected large matrices.

## 13. Checkpointing

A checkpoint must include:

```text
model weights
base optimizer state
scheduler state
pipeline_anchor_config
mask PRF version
anchor EMA tensors or low-rank states
last accepted anchor source step
spectral basis per corrected module
```

For portability, support loading a model without the spectral state. In that case, initialize anchor EMA to zero and warm up correction after the first anchor packets arrive.

## 14. Minimal implementation milestones

### Milestone 1: single-process correctness

- Implement PRF mask.
- Apply in-graph masking at configured layer outputs.
- Confirm backward gradient is zero at masked entries.
- Confirm dense eval path is unchanged.

### Milestone 2: two-stage pipeline prototype

- Split a small transformer into two stages.
- Send masked activations across one boundary.
- Reconstruct on receiver.
- Send masked activation gradients backward.
- Compare against dense training for a tiny task.

### Milestone 3: spectral optimizer without anchor process

- Feed precomputed dense gradients as fake anchors.
- Verify EMA, SVD, filter, blend, and AdamW step.
- Check `alpha = 1.0` exactly matches masked-only training.

### Milestone 4: asynchronous anchor worker

- Add separate anchor group.
- Pull weight snapshots.
- Run unmasked backward.
- Push gradients into a nonblocking queue.
- Log staleness.

### Milestone 5: full trainer integration

- Add config to trainer.
- Wrap model or replace inner loop with pipeline engine.
- Wrap optimizer.
- Disable masking during eval/generation.
- Save/load correction state.

### Milestone 6: reproduction sweep

- Run dense baseline.
- Run data-parallel compression only.
- Run mask-only.
- Run mask plus anchor correction.
- Sweep mask probability, anchor staleness, `tau_p`, and `alpha`.

## 15. Agent checklist

Before coding:

- Identify model layer list and layer count.
- Decide `pp_size` and derive boundary layers.
- Decide whether to use a trainer-compatible wrapper or custom pipeline loop.
- Confirm process groups for fast DP, fast PP, anchor DP, anchor PP, and cross-circuit exchange.

Before training:

- Run mask checksum test.
- Run one dense and one masked forward on the same batch.
- Verify loss finite.
- Verify backward finite.
- Verify anchor packets arrive.
- Verify correction modifies gradients when `alpha < 1`.

Before claiming a speedup:

- Report tokens/sec.
- Report PP bytes/token.
- Report DP bytes/token.
- Report anchor overhead.
- Report evaluation with masking disabled.
- Compare against a dense baseline under the same bandwidth throttle.

## 16. Key defaults

```yaml
mask_prob: 0.95
alpha: 0.3
tau_p: 1.0e-3
beta_anchor: 0.99
svd_rank: 64
anchor_dp_replicas: 1
max_anchor_staleness: 30
eval_disable_masking: true
correct_large_linear_modules_only: true
```

Treat these as starting points, not universal constants.

## 17. Bottom line

This method should be implemented as a distributed transport and optimizer layer beneath the trainer objective. In a TRL-like stack, keep the trainer's data formatting, loss computation, metrics, and adapter integration intact. Add sparse pipeline-boundary communication inside the model/pipeline engine, add an asynchronous unmasked anchor circuit, and wrap the optimizer so that masked gradients are spectrally corrected immediately before AdamW updates the parameters.
