#!/usr/bin/env python3
"""EXP-16 DEFINITIVE root-cause probe. Three questions, direct evidence:

(A) Is the mask applied to BOTH forward and backward in autograd? (the operator's
    core worry.) Elementary autograd: with upstream grad = ones, h.grad MUST equal
    the mask (no-rescale) / mask*gain (rescale). If backward were unmasked, h.grad
    would be all-ones. This is the ground truth for forward/backward consistency.

(B) WHY does rescale cut grad_norm so much, and WHY is dense still smaller? Measure
    the RMSNorm backward amplification as a function of input RMS (Qwen uses RMSNorm
    pre-norm). grad through RMSNorm ~ 1/RMS(input). no-rescale collapses the
    residual RMS -> 1/RMS blows up; rescale restores RMS -> normal; dense has the
    pretrained RMS scale AND no mask variance -> smallest.

(C) Boundary placement: which blocks are masked and where on the block.
"""
import torch
from verl.workers.comm_eff.activation_mask import prf_token_mask, decoder_boundary_indices

H, P = 1536, 0.9
torch.manual_seed(0)

print("==== (C) simulated pipeline-boundary placement ====")
for L, pp in [(28, 8)]:
    b = decoder_boundary_indices(L, pp)
    print(f"num_layers={L} pp_size={pp} -> {len(b)} boundaries at decoder-block OUTPUTS {b}")
print("hook target: layers[idx].register_forward_hook, modifies output[0] = the decoder-block")
print("hidden-state OUTPUT = the residual-stream tensor passed to the next block (== the")
print("inter-pipeline-stage activation). So masking is applied to the inter-stage activation.\n")

print("==== (A) autograd: mask applied to forward AND backward? ====")
N = 32
sid = torch.arange(N); pos = torch.arange(N)
mask = prf_token_mask(sid, pos, layer_idx=3, global_step=1, base_seed=0,
                      hidden_size=H, p=P, device=torch.device("cpu"), dtype=torch.float32)
print(f"mask.requires_grad={mask.requires_grad} (must be False)  kept_frac={mask.mean().item():.4f}")

# no-rescale: forward h*mask ; backward d(h*mask)/dh = mask
h = torch.randn(N, H, requires_grad=True)
(h * mask).backward(torch.ones(N, H))
print(f"[no-rescale] h.grad == mask ? {torch.equal(h.grad, mask)}   "
      f"(if backward were UNMASKED, h.grad would be all-ones; unmasked_frac={(h.grad==1).float().mean():.3f})")

# rescale: forward h*mask/(1-p) ; backward = mask/(1-p)
gain = 1.0 / (1.0 - P)
h2 = torch.randn(N, H, requires_grad=True)
(h2 * mask * gain).backward(torch.ones(N, H))
print(f"[rescale]    h2.grad == mask*{gain:.1f} ? {torch.allclose(h2.grad, mask*gain)}")

# through gradient checkpointing: same result?
import torch.utils.checkpoint as cp
h3 = torch.randn(N, H, requires_grad=True)
def seg(x): return x * mask
cp.checkpoint(seg, h3, use_reentrant=False).backward(torch.ones(N, H))
print(f"[ckpt]       h3.grad == mask ? {torch.equal(h3.grad, mask)}   "
      f"(recompute preserves the mask in the backward graph)\n")

print("==== (B) RMSNorm backward amplification ~ 1/RMS(input) ====")
def rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
w = torch.ones(H)
base = torch.randn(64, H)
print("  (sweeping input RMS to mimic dense / no-rescale-collapse / rescale-overshoot)")
for scale, label in [(0.02, "norescale-like collapse"), (1.0, "dense-like"), (58.0, "rescale-like overshoot")]:
    x = (base * scale).clone().requires_grad_(True)
    rmsnorm(x, w).backward(torch.ones(64, H))
    rms = x.detach().pow(2).mean().sqrt().item()
    print(f"  input_RMS={rms:10.4f} ({label:24s}) -> input_grad_norm={x.grad.norm().item():12.2f}")
print("  => grad through RMSNorm scales as 1/RMS: small RMS (no-rescale) BLOWS UP the")
print("     backward gradient; restoring RMS (rescale) removes the blow-up.")
print("\nGRAD_DIAG3_DONE")
