#!/usr/bin/env python3
"""EXP-16 cross-pass mask consistency — RUNTIME verification (operator MUST-have).

Proves, on the REAL masker (verl.workers.comm_eff.activation_mask.prf_token_mask,
the exact function the in-graph hook calls), that for every (global_step, prompt,
pipeline-boundary) the {0,1} keep/zero mask is BIT-IDENTICAL across two forward
passes that pack the tokens DIFFERENTLY — i.e. the old-logprob recompute vs the
actor-train forward. This is the importance-sampling correctness backbone: if the
two passes saw different masks, exp(log_prob - old_log_prob) would be corrupt.

Runs on CPU (the PRF is device-independent), so it does not contend with the live
training cell on the GPUs.
"""
import sys
import torch
from verl.workers.comm_eff.activation_mask import prf_token_mask, decoder_boundary_indices

P, BASE_SEED, H, STEP = 0.9, 0, 1536, 7
NUM_LAYERS, PP_SIZE = 28, 8           # Qwen2.5-1.5B has 28 decoder blocks
BOUNDARIES = decoder_boundary_indices(NUM_LAYERS, PP_SIZE)

# A realistic batch: 12 prompts of varying length (each prompt = one sample_id,
# its tokens carry position_ids 0..L-1). Mirrors verl's per-row sample identity.
torch.manual_seed(0)
prompts = [(sid, int(L)) for sid, L in enumerate(torch.randint(40, 320, (12,)))]

def build_pass(order):
    """Flatten prompts into a packed token stream in the given prompt order.
    Returns (sample_ids[N], position_ids[N], owner[N]) — owner = prompt sid."""
    sids, poss, owner = [], [], []
    for sid in order:
        L = dict(prompts)[sid]
        sids += [sid] * L
        poss += list(range(L))
        owner += [sid] * L
    return (torch.tensor(sids), torch.tensor(poss), torch.tensor(owner))

# PASS A: prompts in natural order.  PASS B: a DIFFERENT packing (reversed +
# interleaved) — same tokens, different micro-batch layout. A truly packing-
# invariant mask must be identical per token regardless of this.
orderA = [sid for sid, _ in prompts]
orderB = list(reversed(orderA[::2])) + list(reversed(orderA[1::2]))

sidA, posA, ownA = build_pass(orderA)
sidB, posB, ownB = build_pass(orderB)

fails = 0
per_boundary_ok = {}
for layer in BOUNDARIES:
    mA = prf_token_mask(sidA, posA, layer_idx=layer, global_step=STEP, base_seed=BASE_SEED,
                        hidden_size=H, p=P, device=torch.device("cpu"), dtype=torch.float32)
    mB = prf_token_mask(sidB, posB, layer_idx=layer, global_step=STEP, base_seed=BASE_SEED,
                        hidden_size=H, p=P, device=torch.device("cpu"), dtype=torch.float32)
    # Build (sample_id, position_id) -> mask-row lookup for each pass and compare
    # PER PROMPT, PER POSITION (the matching the engine relies on).
    keyA = {(int(sidA[t]), int(posA[t])): mA[t] for t in range(sidA.numel())}
    keyB = {(int(sidB[t]), int(posB[t])): mB[t] for t in range(sidB.numel())}
    assert set(keyA) == set(keyB), f"layer {layer}: token-id sets differ between passes"
    layer_ok = True
    per_prompt_mismatch = {}
    for (sid, pos), rowA in keyA.items():
        if not torch.equal(rowA.to(torch.int64), keyB[(sid, pos)].to(torch.int64)):
            layer_ok = False; fails += 1
            per_prompt_mismatch[sid] = per_prompt_mismatch.get(sid, 0) + 1
    per_boundary_ok[layer] = layer_ok
    ratio = float(1.0 - mA.float().mean().item())
    tag = "OK " if layer_ok else "MISMATCH"
    print(f"[boundary layer={layer:2d}] cross-pass bit-identical per (prompt,pos): {tag} "
          f"(tokens={sidA.numel()}, prompts={len(prompts)}, masked_frac={ratio:.4f})"
          + ("" if layer_ok else f"  mismatched_prompts={per_prompt_mismatch}"))

# Independence + step-sensitivity sanity on the same realistic batch.
m_l3  = prf_token_mask(sidA, posA, layer_idx=BOUNDARIES[0], global_step=STEP, base_seed=BASE_SEED,
                       hidden_size=H, p=P, device=torch.device("cpu"), dtype=torch.float32)
m_l7  = prf_token_mask(sidA, posA, layer_idx=BOUNDARIES[1], global_step=STEP, base_seed=BASE_SEED,
                       hidden_size=H, p=P, device=torch.device("cpu"), dtype=torch.float32)
m_s8  = prf_token_mask(sidA, posA, layer_idx=BOUNDARIES[0], global_step=STEP+1, base_seed=BASE_SEED,
                       hidden_size=H, p=P, device=torch.device("cpu"), dtype=torch.float32)
indep = not torch.equal(m_l3.to(torch.int64), m_l7.to(torch.int64))
stepd = not torch.equal(m_l3.to(torch.int64), m_s8.to(torch.int64))
print(f"per-boundary independent (layer{BOUNDARIES[0]} != layer{BOUNDARIES[1]}): {indep}")
print(f"step-sensitive (step{STEP} != step{STEP+1}): {stepd}")

print(f"\nboundaries checked: {BOUNDARIES}")
ok = fails == 0 and indep and stepd
print("XPASS_MASK_VERIFY_PASS" if ok else f"XPASS_MASK_VERIFY_FAIL (fails={fails})")
sys.exit(0 if ok else 1)
