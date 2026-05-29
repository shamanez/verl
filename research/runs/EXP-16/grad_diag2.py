#!/usr/bin/env python3
"""EXP-16 grad-norm root cause — DIRECT test of the gradient-checkpoint x mask
hypothesis + the RMSNorm-collapse mechanism. CPU-only (CUDA_VISIBLE_DEVICES="")
so it does not touch the GPUs running the A/B comparison.

Resolves the operator's question: does the boundary mask hook fire on the
gradient-checkpoint RECOMPUTE (backward), or only on the original forward?
  - boundary_fires(ckpt ON) == 2 x boundary_fires(ckpt OFF)  => fires on recompute
    => the backward graph IS masked => NO forward/backward inconsistency.
  - grad_norm(ckpt ON) == grad_norm(ckpt OFF)                => consistent (no bug).
And demonstrates the real mechanism:
  - mean boundary-output RMS: dense vs mask-norescale vs mask-rescale.
    no-rescale collapses RMS -> downstream RMSNorm (Jacobian ~1/RMS) amplifies the
    backward gradient. rescale restores RMS -> normal gradient.
"""
import math, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from verl.workers.comm_eff.activation_mask import ActivationMasker, find_decoder_layers, decoder_boundary_indices

MODEL, P, PP, B, L, STEP = "Qwen/Qwen2.5-1.5B-Instruct", 0.9, 8, 2, 32, 1
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")
model.config.use_cache = False
layers = find_decoder_layers(model)
bidx = decoder_boundary_indices(len(layers), PP)
ids = torch.randint(0, model.config.vocab_size, (B, L))
sample_ids = torch.arange(B).repeat_interleave(L)
position_ids = torch.arange(L).repeat(B)

def gnorm():
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(s)

def run(tag, mask, rescale, ckpt):
    model.zero_grad(set_to_none=True)
    (model.gradient_checkpointing_enable if ckpt else model.gradient_checkpointing_disable)()
    fires = [0]; rms_rec = []
    masker = None
    if mask:
        masker = ActivationMasker(p=P, base_seed=0, pp_size=PP, rescale=rescale, state=None)
        masker.register(model)
        masker.set_context(global_step=STEP, sample_ids=sample_ids, position_ids=position_ids)
    # recorder registered AFTER masker -> sees the (masked) boundary output, and
    # counts every fire incl. the gradient-checkpoint recompute.
    def rec(m, i, o):
        fires[0] += 1
        h = o[0] if isinstance(o, tuple) else o
        rms_rec.append(float(h.detach().float().pow(2).mean().sqrt()))
    rh = [layers[i].register_forward_hook(rec) for i in bidx]
    out = model(input_ids=ids, use_cache=False).logits
    loss = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), ids[:, 1:].reshape(-1))
    loss.backward()
    gn = gnorm()
    nfire = fires[0]
    mean_rms = sum(rms_rec) / len(rms_rec) if rms_rec else float("nan")
    for h in rh:
        h.remove()
    if masker:
        masker.unregister()
    print(f"[{tag:16s}] ckpt={str(ckpt):5s} boundary_fires={nfire:3d}  mean_boundary_RMS={mean_rms:9.4f}  grad_norm={gn:12.3f}")
    return gn, nfire, mean_rms

print(f"boundaries={bidx}  (one full forward = {len(bidx)} boundary fires)\n")
dF = run("dense",          False, False, False)
dT = run("dense",          False, False, True)
nF = run("mask_norescale", True,  False, False)
nT = run("mask_norescale", True,  False, True)
rF = run("mask_rescale",   True,  True,  False)
rT = run("mask_rescale",   True,  True,  True)

print("\n==== VERDICT ====")
print(f"recompute fires the hook?  ckpt_ON/OFF boundary_fires = {nT[1]}/{nF[1]}  "
      f"(=> {'YES, ~2x: mask IS applied in the backward recompute' if nT[1] >= 2*nF[1] else 'NO'})")
print(f"forward/backward consistent? grad_norm ckpt_ON==OFF (norescale): {nT[0]:.3f} vs {nF[0]:.3f}  "
      f"(=> {'CONSISTENT, no checkpoint bug' if abs(nT[0]-nF[0])/max(nF[0],1e-9) < 1e-3 else 'INCONSISTENT'})")
print(f"RMS collapse?  dense={dF[2]:.4f}  norescale={nF[2]:.4f}  rescale={rF[2]:.4f}  "
      f"(norescale/dense={nF[2]/dF[2]:.3f}, rescale/dense={rF[2]/dF[2]:.3f})")
print(f"grad inflation: norescale/dense={nF[0]/dF[0]:.1f}x   rescale/dense={rF[0]/dF[0]:.1f}x")
print("GRAD_DIAG2_DONE")
