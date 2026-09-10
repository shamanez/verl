"""Digest of the prf_mask codec path. Must be IDENTICAL across SHAs.

Run it from a tree root with PYTHONPATH set to that root, at each SHA you want
to compare, and diff the JSON (or compare the stderr DIGEST):

    PYTHONPATH=<tree> python3 research/probes/prf_invariance_probe.py

Recorded result, 2026-09-10, for the aq_sgd codec landing on
exp/compass-rlvr-ablations. Three SHAs, one before any aq_sgd code
(1a8aa2b9), the wave-1 pin (4d183c0b) and HEAD:

    BEHAVIOUR digest -> IDENTICAL at all three

so codec resolution, path eligibility, which codec object is built and with
which knobs, the masked-forward hashes on all three path tags, the application
counter and what the sample-id stamp writes are byte-identical. The FULL digest
differs only by the ten inert, defaulted "aq_sgd.*" keys the config dump gains
once a fifth codec is registered. prf_mask arms are therefore code-matched
across the aq_sgd landing.

The stamp component is lifted out of engine_workers.py BY SOURCE and executed,
rather than imported: importing that module pulls in uvicorn, and an earlier
version of this probe let the resulting ModuleNotFoundError degrade to
"unavailable" in every run, which agrees trivially across SHAs and therefore
tested nothing.

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

# 5. The per-row sample-id stamp. Importing engine_workers pulls in uvicorn,
# which is absent from a laptop env, and an earlier version of this probe let
# that degrade to "unavailable" in every run. That agrees trivially across
# SHAs and therefore tested NOTHING, which is the one hole a reviewer of this
# invariance claim would find. So the two methods are lifted out of the file by
# source, with no import at all, and actually run. If extraction itself fails
# that is recorded as a hard error rather than a shrug.
import ast

try:
    _src = open("verl/workers/engine_workers.py").read()
    _tree = ast.parse(_src)
    _want = {"_comm_eff_stamp_sample_ids", "_comm_eff_stamp_example_ids"}
    _fns, _consts = {}, {}
    for _node in _tree.body:
        if (isinstance(_node, ast.Assign) and isinstance(_node.targets[0], ast.Name)
                and _node.targets[0].id == "_COMM_EFF_SAMPLE_ID_RANK_STRIDE"):
            _consts["_COMM_EFF_SAMPLE_ID_RANK_STRIDE"] = ast.literal_eval(_node.value)
        if isinstance(_node, ast.ClassDef):
            for _sub in _node.body:
                if isinstance(_sub, ast.FunctionDef) and _sub.name in _want:
                    _fns[_sub.name] = ast.unparse(_sub)
    _ns = {"torch": torch, "TensorDict": object, **_consts}
    for _code in _fns.values():
        exec(_code, _ns)

    class _W:
        _comm_eff_dp_rank = lambda self: 0
        _comm_eff_stamp_sample_ids = _ns["_comm_eff_stamp_sample_ids"]
        # Absent before the aq_sgd landing; a no-op stand-in keeps the older
        # tree runnable, and the caller is a no-op there anyway.
        _comm_eff_stamp_example_ids = _ns.get(
            "_comm_eff_stamp_example_ids", lambda self, d, s: None)

    from tensordict import TensorDict as _TD
    td = _TD({"dummy": torch.zeros(6, 1)}, batch_size=[6])
    _W()._comm_eff_stamp_sample_ids(td, state)
    out["stamp"] = {
        "extracted": sorted(_fns),
        "rank_stride": _consts.get("_COMM_EFF_SAMPLE_ID_RANK_STRIDE"),
        "keys_added": sorted(set(td.keys()) - {"dummy"}),
        "sample_id": td["comm_eff_sample_id"].tolist(),
    }
    # A dense-codec state must be left completely alone.
    td2 = _TD({"dummy": torch.zeros(4, 1)}, batch_size=[4])
    _W()._comm_eff_stamp_sample_ids(td2, None)
    out["stamp"]["no_state_is_noop"] = sorted(td2.keys()) == ["dummy"]
except Exception as e:
    out["stamp"] = {"ERROR": f"{type(e).__name__}: {e}"}

# Two digests, because they answer different questions and conflating them is
# how a probe misleads. BEHAVIOUR covers only what the compiled prf_mask path
# does: codec resolution, path eligibility, which codec object is built and
# with which knobs, the hashed masked forward on every eligible tag, the
# application counter, and what the stamp writes onto a batch. It deliberately
# EXCLUDES the config dump, which legitimately grows an inert defaulted
# sub-config when a new codec is registered, and the probe's own "extracted"
# list, which reports which functions the probe found rather than what they do.
# FULL covers everything, so a change can be located.
behaviour = {
    "eligibility": out["eligibility"],
    "codec": out["codec"],
    "masker_knobs": out["masker_knobs"],
    "masked_forward": out["masked_forward"],
    "counters": out["counters"],
    "stamp": {k: v for k, v in out["stamp"].items() if k != "extracted"},
}
out["probe_meta"] = {"stamp_extracted": out["stamp"].pop("extracted", None)}

blob = json.dumps(out, sort_keys=True, indent=1)
bdig = hashlib.sha256(json.dumps(behaviour, sort_keys=True).encode()).hexdigest()
print(blob)
print("BEHAVIOUR", bdig, file=sys.stderr)
print("FULL", hashlib.sha256(blob.encode()).hexdigest(), file=sys.stderr)
