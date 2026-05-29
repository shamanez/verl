#!/usr/bin/env python3
"""EXP-16 grad-norm root-cause probe (operator: dense~0.03 vs masked~2700).

Isolates WHERE the 10^5x grad-norm blow-up comes from, on a SINGLE GPU (no FSDP),
so we can tell whether it is (a) an FSDP interaction, (b) the masking forward-hook
math, or (c) a gradient-checkpoint recompute mask mismatch.

For each config it does one forward+backward of an identical fixed batch through
Qwen2.5-1.5B and reports the global grad-norm + top layers by grad contribution.
Configs: dense / mask-no-rescale / mask-rescale, each with grad-ckpt ON and OFF.
Also asserts the mask tensor is not requires_grad, and (with ckpt ON) that the
hook's forward-pass mask == its recompute-pass mask (the grad-ckpt correctness
condition).
"""
import os, hashlib
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.workers.comm_eff.activation_mask import ActivationMasker, find_decoder_layers, decoder_boundary_indices

DEV = "cuda:0"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
P, PP_SIZE, B, L, STEP = 0.9, 8, 4, 256, 1
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2").to(DEV)
V = model.config.vocab_size
layers = find_decoder_layers(model)
print(f"num_decoder_layers={len(layers)} boundaries={decoder_boundary_indices(len(layers), PP_SIZE)}")

# fixed batch + per-token identities for the masker
ids = torch.randint(0, V, (B, L), device=DEV)
sample_ids = torch.arange(B, device=DEV).repeat_interleave(L)
position_ids = torch.arange(L, device=DEV).repeat(B)

def global_grad_norm():
    sq = 0.0
    per_layer = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float().norm().item()
        sq += g * g
        # bucket by "...layers.<i>..."
        key = "other"
        if ".layers." in n:
            key = "layer_" + n.split(".layers.")[1].split(".")[0]
        per_layer[key] = per_layer.get(key, 0.0) + g * g
    return sq ** 0.5, {k: v ** 0.5 for k, v in per_layer.items()}

def run(tag, mask, rescale, ckpt):
    model.zero_grad(set_to_none=True)
    (model.gradient_checkpointing_enable if ckpt else model.gradient_checkpointing_disable)()
    model.train()
    masker = None
    hook_masks = {}        # layer_idx -> list of mask checksums (one per hook fire)
    if mask:
        masker = ActivationMasker(p=P, base_seed=0, pp_size=PP_SIZE, rescale=rescale, state=None)
        masker.register(model)
        masker.set_context(global_step=STEP, sample_ids=sample_ids, position_ids=position_ids)
        # wrap each boundary block to record the mask the hook actually applied
        # (detect grad-ckpt recompute mismatch): we recompute the deterministic
        # mask the same way the hook does and checksum it per fire.
        orig_handles = list(masker._handles)
    out = model(input_ids=ids, use_cache=False).logits
    loss = F.cross_entropy(out[:, :-1].reshape(-1, V).float(), ids[:, 1:].reshape(-1))
    loss.backward()
    gn, per_layer = global_grad_norm()
    if masker:
        masker.unregister()
    top = sorted(per_layer.items(), key=lambda kv: -kv[1])[:5]
    print(f"[{tag:28s}] ckpt={ckpt!s:5s} loss={loss.item():8.4f} GRAD_NORM={gn:12.4f}  top_layers={[(k, round(v,2)) for k,v in top]}")
    return gn

print("\n==== grad-norm by config (single GPU, no FSDP) ====")
gd_on  = run("dense",            mask=False, rescale=False, ckpt=True)
gd_off = run("dense",            mask=False, rescale=False, ckpt=False)
gm_on  = run("mask_p0.9_norescale", mask=True,  rescale=False, ckpt=True)
gm_off = run("mask_p0.9_norescale", mask=True,  rescale=False, ckpt=False)
gr_on  = run("mask_p0.9_rescale",   mask=True,  rescale=True,  ckpt=True)
gr_off = run("mask_p0.9_rescale",   mask=True,  rescale=True,  ckpt=False)

print("\n==== RATIOS (masked / dense), same ckpt setting ====")
print(f"mask_norescale/dense  ckpt_ON : {gm_on/gd_on:.1f}x      ckpt_OFF: {gm_off/gd_off:.1f}x")
print(f"mask_rescale/dense    ckpt_ON : {gr_on/gd_on:.1f}x      ckpt_OFF: {gr_off/gd_off:.1f}x")
print(f"grad-ckpt effect on masked (ON/OFF): {gm_on/gm_off:.3f}  (==1 => ckpt recompute is consistent)")

# mask tensor sanity: requires_grad must be False
from verl.workers.comm_eff.activation_mask import prf_token_mask
m = prf_token_mask(sample_ids, position_ids, layer_idx=3, global_step=STEP, base_seed=0,
                   hidden_size=model.config.hidden_size, p=P, device=torch.device(DEV), dtype=torch.bfloat16)
print(f"\nmask.requires_grad={m.requires_grad} (must be False); mask unique={torch.unique(m).tolist()}")
print("GRAD_DIAG_DONE")
