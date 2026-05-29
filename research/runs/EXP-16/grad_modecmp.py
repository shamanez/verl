"""EXP-16: validate the switchable mask.rescale_mode {none|constant|rms_match}.

Exercises the REAL ActivationMasker (not a reimplementation) on Qwen2.5-1.5B so
the test covers the actual hook code being shipped. Three claims to confirm:

  (1) SWITCH is non-destructive: mode=none reproduces raw h*mask (grad_norm
      blows up); mode=constant reproduces inverted-dropout h*mask/(1-p).
  (2) rms_match delivers EXACT norm: RMS(h_tilde)/RMS(h_dense) == 1 per token at
      a boundary (none collapses ~sqrt(1-p); constant overshoots ~sqrt(1/(1-p))).
  (3) rms_match FIXES the grad-norm blow-up (norm/dense ~ O(1), like/under
      constant) WITHOUT changing direction (cos_to_dense ~ same as constant).
"""
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from verl.workers.comm_eff.activation_mask import (
    ActivationMasker,
    find_decoder_layers,
    decoder_boundary_indices,
)

DEV = "cuda:0"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
B, L, STEP, PP, SEED = 4, 256, 1, 8, 7
torch.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).to(DEV)
model.gradient_checkpointing_disable()
model.train()
V = model.config.vocab_size
Hd = model.config.hidden_size
layers = find_decoder_layers(model)
bidx = sorted(decoder_boundary_indices(len(layers), PP))
print(f"boundaries={bidx} L={len(layers)} H={Hd}")

ids = torch.randint(0, V, (B, L), device=DEV)
sid = torch.arange(B, device=DEV).repeat_interleave(L)
pos = torch.arange(L, device=DEV).repeat(B)

# A capture hook on the FIRST boundary, registered AFTER the masker so it reads
# the post-mask tensor h_tilde. On the dense pass (no masker) it reads h_dense.
_cap = {}
def _capture(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    _cap["rms"] = h.detach().float().pow(2).mean(dim=-1).sqrt().reshape(-1)  # per-token RMS

def grads(masker):
    model.zero_grad(set_to_none=True)
    cap_handle = None
    if masker is not None:
        masker.register(model)
        masker.set_context(global_step=STEP, sample_ids=sid, position_ids=pos)
    cap_handle = layers[bidx[0]].register_forward_hook(_capture)  # fires after masker's
    out = model(input_ids=ids, use_cache=False).logits
    F.cross_entropy(out[:, :-1].reshape(-1, V).float(), ids[:, 1:].reshape(-1)).backward()
    cap_handle.remove()
    if masker is not None:
        masker.unregister()
    g = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return g, _cap["rms"].clone()

def cmp(gA, gB):
    dot = na = nb = 0.0
    for k in gA:
        a, b = gA[k], gB[k]
        dot += (a * b).sum().item(); na += (a * a).sum().item(); nb += (b * b).sum().item()
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12), math.sqrt(nb) / (math.sqrt(na) + 1e-12)

gd, rms_dense = grads(None)
print(f"\ndense ref: grad collected for {len(gd)} params, boundary RMS mean={rms_dense.mean():.4f}\n")
print(f"{'mode':>10} {'p':>5} | {'RMS(h~)/RMS(dense)':>22} | {'cos_to_dense':>12} | {'norm/dense':>10}")
print("-" * 72)
for p in [0.5, 0.9, 0.95]:
    for mode in ["none", "constant", "rms_match"]:
        m = ActivationMasker(p=p, base_seed=SEED, pp_size=PP, rescale_mode=mode, state=None)
        g, rms_t = grads(m)
        ratio = (rms_t / (rms_dense + 1e-8))
        c, nr = cmp(gd, g)
        print(f"{mode:>10} {p:>5} | mean={ratio.mean():.3f} [min {ratio.min():.3f}, max {ratio.max():.3f}] "
              f"| {c:>12.4f} | {nr:>10.2f}")
        del g
    print("-" * 72)
print("\nEXPECT: none -> RMS ratio ~sqrt(1-p) (collapse), norm/dense >> 1 (blow-up);")
print("        constant -> RMS ratio ~sqrt(1/(1-p)) (overshoot), norm/dense few x;")
print("        rms_match -> RMS ratio ~1.000 EXACT, norm/dense ~O(1) (<= constant),")
print("        cos_to_dense(rms_match) ~ cos_to_dense(constant) (norm fix, not direction).")
print("MODECMP_DONE")
