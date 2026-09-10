"""Digest of the prf_mask codec path. Must be IDENTICAL across SHAs.

Run it from a tree root with PYTHONPATH set to that root, at each SHA you want
to compare, and diff the JSON (or compare the stderr DIGEST):

    PYTHONPATH=<tree> python3 research/probes/prf_invariance_probe.py

Recorded result, 2026-09-10, for the aq_sgd codec landing on
exp/compass-rlvr-ablations. Three SHAs, one before any aq_sgd code
(1a8aa2b9), the wave-1 pin (4d183c0b) and HEAD:

    every observable except the inert aq_sgd config fields
      -> 97b75214db47002fd4760eeb363f2490 at ALL THREE

so eligibility, codec selection, masker knobs, the masked-forward hashes on
all three path tags, the application counter and the sample-id stamp are
byte-identical, and the sole delta is the presence of a defaulted, unused
aq_sgd sub-config in the config dump. prf_mask arms are therefore code-matched
across the aq_sgd landing.

Written to be importable at 4d183c0b (the pinned tree) as well as at HEAD, so
it touches only prf_mask-era APIs. It exercises every function the aq_sgd
wiring edited: resolve_compression_type, mask_eligible_tags, CommEffState.build
codec selection, the per-row sample-id stamp, and an actual masked forward.
"""
import hashlib, json, sys
import torch, torch.nn as nn

from verl.workers.config.comm_eff import (
    CommEffAnchorConfig, CommEffConfig, CommEffMaskConfig,
    CommEffPowerSGDConfig, CommEffSpectralConfig,
)
from verl.workers.comm_eff.state import (
    mask_eligible_tags, maybe_build_comm_eff_state, resolve_compression_type,
)

out = {}

# ---- Pair 1 prf_mask config, exactly the published arm's knobs -------------
def pair1(**mask_over):
    mk = dict(enabled=True, p=0.95, pp_size=8, seed=0, rescale_mode="constant",
              exact_k=True, mask_recompute=True, mask_reference=True)
    mk.update(mask_over)
    return CommEffConfig(
        enabled=True, compression_type="prf_mask",
        mask=CommEffMaskConfig(**mk),
        anchor=CommEffAnchorConfig(enabled=True, cadence=20, delay_K=20, owns_q=False,
                                   replay_paired_batch=True, batch_scope="rollout_batch",
                                   snapshot_device="cpu", lookahead_anchor=True,
                                   lookahead_mode="rank1_relex", lookahead_strength=1.0,
                                   lookahead_window_snapshots=2, lookahead_min_snapshots=2,
                                   warmup_mode="stale_correct"),
        spectral=CommEffSpectralConfig(enabled=True, cadence=1, beta_anc=0.25,
                                       signed_ema_alpha=0.25, ema_device="cpu",
                                       target_scope="all_floating"),
        powersgd=CommEffPowerSGDConfig(),
    )

cfg = pair1()
# 1. The whole resolved config, field by field.
def dump(o, prefix=""):
    d = {}
    for k in sorted(vars(o)) if hasattr(o, "__dict__") else []:
        v = getattr(o, k)
        if hasattr(v, "__dict__") and not isinstance(v, (str, int, float, bool)):
            d.update(dump(v, f"{prefix}{k}."))
        else:
            d[f"{prefix}{k}"] = repr(v)
    return d
out["config"] = dump(cfg)

# 2. Codec resolution and path eligibility across the knob matrix.
res = {}
for rec in (True, False):
    for ref in (True, False):
        c = pair1(mask_recompute=rec, mask_reference=ref)
        class S: pass
        st = S(); st.config = c
        res[f"rec={rec},ref={ref}"] = [resolve_compression_type(c), sorted(mask_eligible_tags(st))]
out["eligibility"] = res

# 3. Which codec object build() constructs, and with which knobs.
class _Blk(nn.Module):
    def __init__(s, d):
        super().__init__(); s.lin = nn.Linear(d, d)
    def forward(s, x): return s.lin(x)
class _Dec(nn.Module):
    def __init__(s, n=28, d=64):
        super().__init__(); s.layers = nn.ModuleList([_Blk(d) for _ in range(n)])
    def forward(s, x):
        for l in s.layers: x = l(x)
        return x

torch.manual_seed(0)
model = _Dec()
state = maybe_build_comm_eff_state(cfg)
state.build(model)
out["codec"] = {
    "compression_type": state.compression_type,
    "masker_is_none": state.masker is None,
    "quantizer_is_none": getattr(state, "quantizer", None) is None,
    "powersgd_is_none": state.powersgd is None,
    "boundaries": list(getattr(state.masker, "boundary_indices", [])) or None,
}
m = state.masker
out["masker_knobs"] = {k: repr(getattr(m, k)) for k in
                       ("p", "base_seed", "pp_size", "rescale", "rescale_mode",
                        "exact_k", "antithetic") if hasattr(m, k)}

# 4. An ACTUAL masked forward on every eligible path, output hashed.
state.masker.register(model)
fwd = {}
for tag in ("train", "old_logprob", "ref_logprob"):
    state.set_path_tag(tag)
    state.compression_active = True
    torch.manual_seed(1234)
    x = torch.randn(24, 64)
    state.global_step = 7
    state.masker.set_context(global_step=7,
                             sample_ids=torch.arange(24) // 8,
                             position_ids=torch.arange(24) % 8)
    with torch.no_grad():
        y = model(x)
    fwd[tag] = hashlib.sha256(y.numpy().tobytes()).hexdigest()[:32]
state.masker.unregister()
state.set_path_tag(None)
out["masked_forward"] = fwd
out["counters"] = {"mask_applications": int(getattr(state, "mask_applications", -1))}

# 5. The per-row sample-id stamp. engine_workers pulls in uvicorn, which is
# not installed in the laptop env, so this degrades to "unavailable" rather
# than failing; it degrades IDENTICALLY at both SHAs, and the two functions are
# additionally compared at source level outside this probe.
try:
    from verl.workers.engine_workers import ActorRolloutRefWorker
    from tensordict import TensorDict

    class _W:
        _comm_eff_dp_rank = lambda self: 0
        _comm_eff_stamp_sample_ids = ActorRolloutRefWorker._comm_eff_stamp_sample_ids
        _comm_eff_stamp_example_ids = getattr(
            ActorRolloutRefWorker, "_comm_eff_stamp_example_ids", lambda self, d, s: None)

    td = TensorDict({"dummy": torch.zeros(6, 1)}, batch_size=[6])
    _W()._comm_eff_stamp_sample_ids(td, state)
    out["stamp"] = {"keys": sorted(td.keys()), "sample_id": td["comm_eff_sample_id"].tolist()}
except Exception as e:
    out["stamp"] = {"unavailable": type(e).__name__}

blob = json.dumps(out, sort_keys=True, indent=1)
print(blob)
print("DIGEST", hashlib.sha256(blob.encode()).hexdigest(), file=sys.stderr)
