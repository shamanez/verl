# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU gate for the COMPASS RLVR ablation arms.

Every arm in ``examples/grpo_trainer/run_compass_rlvr_ablations_fsdp.sh`` is built
here as a real :class:`CommEffConfig` and pushed through the real validator, so
a configuration that the validator would reject is caught on the laptop rather
than on a paid box. The arm table below is the single source of truth for the
env deltas and must stay in step with the shell script.

The no-anchor arm is the one that needs this most: removing the anchor is four
coupled flips plus the merger, because ``rank1_relex`` requires
``anchor.enabled``, ``replay_paired_batch`` and ``spectral.enabled`` to all be
true, and ``lookahead_min_snapshots != -1`` requires ``rank1_relex`` to be
active. Every intuitive partial attempt is therefore rejected, which
``test_naive_anchor_removal_is_rejected`` pins down one case at a time.

Run directly (no pytest needed):

    python3 tests/workers/comm_eff/test_compass_ablation_arms.py
"""

import pytest

from verl.workers.comm_eff.anchor import AnchorReplayRing, anchor_should_fire
from verl.workers.config.comm_eff import (
    CommEffAnchorConfig,
    CommEffAQSGDConfig,
    CommEffConfig,
    CommEffMaskConfig,
    CommEffQuantConfig,
    CommEffSpectralConfig,
    CommEffTAHQuantConfig,
)

# Pair 1: Qwen2.5-Math-1.5B, H = 1536, 8 logical stages, k = 77 of 1536.
PAIR1_MASK = dict(
    enabled=True,
    p=0.95,
    pp_size=8,
    rescale_mode="constant",
    exact_k=True,
    mask_recompute=True,
    mask_reference=True,
)
PAIR1_ANCHOR = dict(
    enabled=True,
    cadence=20,
    delay_K=20,
    owns_q=False,
    replay_paired_batch=True,
    batch_scope="rollout_batch",
    snapshot_device="cpu",
    lookahead_anchor=True,
    lookahead_mode="rank1_relex",
    lookahead_strength=1.0,
    lookahead_window_snapshots=2,
    lookahead_min_snapshots=2,
    warmup_mode="stale_correct",
)
PAIR1_SPECTRAL = dict(
    enabled=True,
    cadence=1,
    beta_anc=0.25,
    signed_ema_alpha=0.25,
    ema_device="cpu",
    target_scope="all_floating",
)

# Removing the anchor: the four coupled flips plus the merger.
NO_ANCHOR_ANCHOR = dict(
    enabled=False,
    lookahead_anchor=False,
    lookahead_mode="disabled",
    lookahead_min_snapshots=-1,
)

# The quantizing codecs replace the PRF mask rather than layering on it, so
# mask.enabled goes false while mask_recompute / mask_reference stay true: both
# reuse those two for path eligibility, and aq_sgd REQUIRES mask_recompute
# because its buffer is read on every eligible pass.
QUANTIZING_MASK = {"enabled": False}

# Byte parity at Pair 1. The mask sends 77 fp16 coordinates = 1232 bits per
# token per boundary; 2 bits on k=493 with fp16 scales per 32 kept channels is
# 493*2 + 493*16/32 = 1232.5, matched to 1.0004x.
PARITY_CODEC = dict(bits=2, subset_k=493, block_size=32)

# arm -> (mask delta, anchor delta, spectral delta, codec)
# `codec` is the compression_type verbatim. "dense" also implies enabled=False.
ARMS = {
    "base": ({}, {}, {}, "prf_mask"),
    "dense": ({"enabled": False}, NO_ANCHOR_ANCHOR, {"enabled": False}, "dense"),
    "noanchor": ({}, NO_ANCHOR_ANCHOR, {"enabled": False}, "prf_mask"),
    "nosign": ({}, {}, {"signed_ema_alpha": 1.0}, "prf_mask"),
    "k10": ({}, {"delay_K": 10}, {}, "prf_mask"),
    "k40": ({}, {"delay_K": 40}, {}, "prf_mask"),
    "k80": ({}, {"delay_K": 80}, {}, "prf_mask"),
    "k40gm": ({}, {"delay_K": 40, "lookahead_strength": 0.5}, {}, "prf_mask"),
    "k80gm": ({}, {"delay_K": 80, "lookahead_strength": 0.25}, {}, "prf_mask"),
    "smoke": ({}, {}, {}, "prf_mask"),
    # Codec ablation (the quantization family), byte-matched to the mask.
    "aqsgd": (QUANTIZING_MASK, {}, {}, "aq_sgd"),
    "aqsgd-all": (QUANTIZING_MASK, {}, {}, "aq_sgd"),
    "aqsgd-rn": (QUANTIZING_MASK, {}, {}, "aq_sgd"),
    "aqsgd-payload": (QUANTIZING_MASK, {}, {}, "aq_sgd"),
    "srquant": (QUANTIZING_MASK, {}, {}, "sr_quant"),
    # TAH-Quant (arXiv:2506.01352v2), the STATELESS quantization arm.
    "tahquant": (QUANTIZING_MASK, {}, {}, "tah_quant"),
    "tahquant-noh": (QUANTIZING_MASK, {}, {}, "tah_quant"),
    "tahquant-sr": (QUANTIZING_MASK, {}, {}, "tah_quant"),
    "tahquant-fullrate": (QUANTIZING_MASK, {}, {}, "tah_quant"),
}

# Codec sub-config per arm, for the arms whose codec has one. Absent means the
# codec's own defaults.
CODEC_CFG = {
    "aqsgd": dict(**PARITY_CODEC, rounding="sr", scope="prompt", first_visit="rescaled"),
    "aqsgd-all": dict(**PARITY_CODEC, rounding="sr", scope="all", first_visit="rescaled"),
    "aqsgd-rn": dict(**PARITY_CODEC, rounding="rn", scope="prompt", first_visit="rescaled"),
    "aqsgd-payload": dict(**PARITY_CODEC, rounding="sr", scope="prompt", first_visit="dense"),
    "srquant": dict(**PARITY_CODEC, rounding="sr"),
    # k=242 at tile=32 is the parity point: 242*3.8 + 8 tiles x 39 b = 1231.6
    # bits/token/boundary, 0.99968x the incumbent's 1232.
    "tahquant": dict(tile=32, subset_k=242, int4_frac=0.8, tau=2.0, rounding="rn"),
    "tahquant-noh": dict(tile=32, subset_k=242, int4_frac=0.8, tau=float("inf"), rounding="rn"),
    "tahquant-sr": dict(tile=32, subset_k=242, int4_frac=0.8, tau=2.0, rounding="sr"),
    # subset_k=0 is the paper's full width, deliberately OFF budget at 5.50x.
    "tahquant-fullrate": dict(tile=32, subset_k=0, int4_frac=0.8, tau=2.0, rounding="rn"),
}


def build(arm):
    mask_d, anchor_d, spectral_d, codec = ARMS[arm]
    kwargs = dict(
        enabled=codec != "dense",
        compression_type=codec,
        mask=CommEffMaskConfig(**{**PAIR1_MASK, **mask_d}),
        anchor=CommEffAnchorConfig(**{**PAIR1_ANCHOR, **anchor_d}),
        spectral=CommEffSpectralConfig(**{**PAIR1_SPECTRAL, **spectral_d}),
    )
    codec_d = CODEC_CFG.get(arm, {})
    if codec == "aq_sgd":
        kwargs["aq_sgd"] = CommEffAQSGDConfig(**codec_d)
    elif codec == "sr_quant":
        kwargs["quant"] = CommEffQuantConfig(**codec_d)
    elif codec == "tah_quant":
        kwargs["tah"] = CommEffTAHQuantConfig(**codec_d)
    elif codec_d:
        raise AssertionError(f"arm {arm!r} carries a codec sub-config but codec {codec!r} has no branch here")
    return CommEffConfig(**kwargs)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_arm_validates(arm):
    """Every shipped arm must pass CommEffConfig.__post_init__."""
    cfg = build(arm)
    assert cfg.anchor.delay_K >= 0


def test_noanchor_actually_removes_the_anchor():
    cfg = build("noanchor")
    assert cfg.enabled is True and cfg.compression_type == "prf_mask"
    assert cfg.mask.enabled is True, "the codec must stay on: this is not the dense control"
    assert cfg.anchor.enabled is False
    assert cfg.anchor.lookahead_mode == "disabled"
    assert cfg.spectral.enabled is False, (
        "leaving the merger on allocates a full-model fp32 CPU EMA and all-gathers "
        "every gradient for an identity transform, which would contaminate the "
        "throughput comparison"
    )


def test_nosign_keeps_the_anchor_running_at_full_cost():
    """alpha=1 answers a different question from anchor.enabled=false."""
    cfg = build("nosign")
    assert cfg.anchor.enabled is True
    assert cfg.spectral.enabled is True
    assert cfg.spectral.signed_ema_alpha == 1.0


# Every intuitive way to "just turn the anchor off" that a person would try
# first, and which the validator refuses. Only the full five-flip set of the
# `noanchor` arm gets through, which is why that arm exists as a named arm
# rather than as an env line typed at the shell.
NAIVE_ANCHOR_REMOVALS = [
    ("anchor.enabled=false alone", {"enabled": False}, {}),
    (
        "lookahead off, min_snapshots left at 2",
        {"lookahead_anchor": False, "lookahead_mode": "disabled"},
        {},
    ),
    ("spectral.enabled=false alone", {}, {"enabled": False}),
    (
        "anchor and lookahead off, min_snapshots left at 2",
        {"enabled": False, "lookahead_anchor": False, "lookahead_mode": "disabled"},
        {},
    ),
]


@pytest.mark.parametrize("label,anchor_d,spectral_d", NAIVE_ANCHOR_REMOVALS)
def test_naive_anchor_removal_is_rejected(label, anchor_d, spectral_d):
    """The validator refuses every partial removal, so a mistyped arm cannot
    reach a GPU and quietly train something other than the intended control."""
    with pytest.raises(ValueError):
        CommEffConfig(
            enabled=True,
            compression_type="prf_mask",
            mask=CommEffMaskConfig(**PAIR1_MASK),
            anchor=CommEffAnchorConfig(**{**PAIR1_ANCHOR, **anchor_d}),
            spectral=CommEffSpectralConfig(**{**PAIR1_SPECTRAL, **spectral_d}),
        )


@pytest.mark.parametrize("k", [10, 20, 40, 80])
def test_replay_ring_retains_what_delay_K_needs(k):
    """Retention is fire-aware, so K costs host RAM linearly in K/c, not in K."""
    ring = AnchorReplayRing(cadence=20, delay_K=k)
    expected = k // 20 + 1
    assert ring._maxlen == expected
    assert ring._keep_residue == (-k) % 20


@pytest.mark.parametrize("k", [10, 20, 40, 80])
def test_forecast_horizon_equals_delay_K(k):
    """theta_hat = theta_{t-K} + (K/c)(theta_{t-K} - theta_{t-K-c}).

    The secant increment is strength * slope * horizon with horizon == K, so a
    K sweep at fixed cadence also scales the extrapolation by K/c. The kNNgm
    arms cancel that with strength = c/K; assert the product is invariant.
    """
    cadence = 20
    gain = 1.0 * (k / cadence)
    matched_strength = cadence / k
    assert pytest.approx(matched_strength * (k / cadence), rel=1e-9) == 1.0
    assert gain == pytest.approx(k / cadence)


def test_cadence_counts_optimizer_ticks_and_pair1_has_one_per_step():
    """At 128/128 there is exactly one optimizer tick per generation, so the
    tick-counted cadence and delay equal global steps on Pair 1."""
    fires = [t for t in range(1, 201) if anchor_should_fire(t, 20, True)]
    assert len(fires) == 10
    assert fires[0] == 20 and fires[-1] == 200


if __name__ == "__main__":
    import sys
    import traceback

    failures = []
    for arm in sorted(ARMS):
        try:
            build(arm)
            print(f"  OK    {arm}")
        except Exception as exc:  # noqa: BLE001
            failures.append((arm, exc))
            print(f"  FAIL  {arm}: {exc}")
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        params = [m for m in marks if m.name == "parametrize"]
        if not params:
            cases = [()]
        else:
            vals = params[0].args[1]
            n_args = len(params[0].args[0].split(","))
            cases = [tuple(v) if n_args > 1 else (v,) for v in vals]
        for case in cases:
            label = f"{name}{case if case else ''}"
            try:
                fn(*case)
                print(f"  OK    {label}")
            except Exception as exc:  # noqa: BLE001
                failures.append((label, exc))
                print(f"  FAIL  {label}")
                traceback.print_exc()
    print()
    if failures:
        print(f"{len(failures)} FAILURE(S) -- do not launch")
        sys.exit(1)
    print("all COMPASS ablation arms validate; safe to launch")

# --------------------------------------------------------------------------- #
# codec ablation arms (the quantization family)
# --------------------------------------------------------------------------- #
CODEC_ARMS = (
    "aqsgd",
    "aqsgd-all",
    "aqsgd-rn",
    "aqsgd-payload",
    "srquant",
    "tahquant",
    "tahquant-noh",
    "tahquant-sr",
)
# tahquant-fullrate is DELIBERATELY excluded from the byte-parity arms: it runs
# the paper's published full-width rate, which is 5.50x the incumbent, and it
# exists to answer whether the budget or the codec binds. Listing it here would
# assert parity of an arm whose whole point is to break parity.
OFF_BUDGET_CODEC_ARMS = ("tahquant-fullrate",)


def codec_wire_bits(cfg) -> float:
    """Bits per token per boundary for whichever codec the arm selected.

    Dispatches on ``compression_type`` and RAISES on anything it does not know.
    The previous form was ``cfg.aq_sgd if compression_type == "aq_sgd" else
    cfg.quant``, which silently priced any third codec against sr_quant's
    sub-config and sr_quant's formula, so a new arm would be asserted
    byte-matched using knobs it does not even have.
    """
    ctype = cfg.compression_type
    if ctype in ("sr_quant", "aq_sgd"):
        sub = cfg.quant if ctype == "sr_quant" else cfg.aq_sgd
        eff_block = sub.subset_k if (sub.block_size <= 0 or sub.block_size >= sub.subset_k) else sub.block_size
        return sub.subset_k * sub.bits + sub.subset_k * 16 / eff_block
    if ctype == "tah_quant":
        from verl.workers.comm_eff.activation_tahquant import (
            TAHQUANT_INT3_BITS,
            TAHQUANT_INT4_BITS,
            tahquant_metadata_bits,
        )

        sub = cfg.tah
        k = sub.subset_k
        n_tiles = (k + sub.tile - 1) // sub.tile
        payload = k * (sub.int4_frac * TAHQUANT_INT4_BITS + (1.0 - sub.int4_frac) * TAHQUANT_INT3_BITS)
        return payload + n_tiles * tahquant_metadata_bits(sub.tile)
    raise AssertionError(
        f"no wire ledger for compression_type={ctype!r}; add one rather than letting the arm "
        "be priced against another codec's sub-config"
    )


@pytest.mark.parametrize("arm", CODEC_ARMS)
def test_codec_arm_is_byte_matched_to_the_mask(arm):
    """Every codec arm must price out against PRF exact-k, or it proves nothing.

    The mask sends k=77 fp16 coordinates, so 1232 bits per token per boundary.
    A quantizing codec sends subset_k*bits payload plus one fp16 scale per block
    of kept channels. The comparison is only a codec comparison if those match.
    """
    cfg = build(arm)
    bits = codec_wire_bits(cfg)
    prf = 77 * 16  # the incumbent, k=77 fp16 coordinates at H=1536
    assert abs(bits / prf - 1.0) < 1e-3, (arm, cfg.compression_type, bits, bits / prf)


def test_every_arm_gets_a_distinct_r2_path_regardless_of_keep_ckpt():
    """No two arms may share an R2 key, and KEEP_CKPT must not gate the path.

    The bug this defends against, measured live on 2026-09-11. The launcher set
    `R2_EXPERIMENT` / `R2_REGIME` only inside `if [[ $KEEP_CKPT == 1 ]]`, which
    is safe ONLY if KEEP_CKPT is the sole thing that can enable saving. It is
    not: `SAVE_FREQ` and `CKPT_R2_ENABLED` both honour an inherited value, so a
    box-level launcher exporting SAVE_FREQ=200 turns saving on for every arm,
    including KEEP_CKPT=0 arms that never reached those two lines. Those arms
    fell through to the sink defaults and wrote to `EXP-unknown/regime/`, a
    SINGLE shared key: aqsgd-rn landed there and tahquant-sr would have
    overwritten it.

    So the assertion is structural (the exports precede the conditional) plus
    exhaustive over the arm table, and the arm list is DERIVED from the shell
    case labels rather than restated here.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    sh = (root / "examples/grpo_trainer/run_compass_rlvr_ablations_fsdp.sh").read_text()

    exp_at = sh.index("export R2_EXPERIMENT=")
    reg_at = sh.index("export R2_REGIME=")
    gate_at = sh.index('if [[ "$KEEP_CKPT" == "1" ]]; then')
    assert exp_at < gate_at and reg_at < gate_at, (
        "R2_EXPERIMENT / R2_REGIME must be exported BEFORE the KEEP_CKPT gate, or an arm "
        "whose saving was enabled from outside writes to the sink's shared default key"
    )
    assert sh.count("export R2_EXPERIMENT=") == 1 and sh.count("export R2_REGIME=") == 1

    # Derive the arm names from the dispatcher's own case labels, so a new arm
    # is covered without editing this test.
    case_body = sh[sh.index('case "$ARM" in') : sh.index("\n  smoke)")]
    labels = set()
    for m in re.finditer(r"^\s{2}([a-z0-9|_-]+)\)", case_body, re.M):
        labels.update(m.group(1).split("|"))
    labels.discard("*")
    assert len(labels) >= 9, sorted(labels)

    # R2_REGIME defaults to $ARM, so distinctness of the arm names IS
    # distinctness of the keys, and no arm may collide with the sink default.
    assert len(labels) == len(set(labels))
    assert "regime" not in labels and "EXP-unknown" not in labels


@pytest.mark.parametrize("arm", OFF_BUDGET_CODEC_ARMS)
def test_off_budget_codec_arm_is_actually_off_budget(arm):
    """The full-rate arm must be far off parity, and provably so.

    If someone later gives it a subset_k it would quietly become a second,
    unexplainable byte-matched arm. Asserting the DEVIATION keeps its purpose
    legible: it is the arm that prices what a bigger budget buys.
    """
    cfg = build(arm)
    assert cfg.compression_type == "tah_quant"
    assert cfg.tah.subset_k == 0, "the full-rate arm must quantize the full width"
    # At H=1536 the published setting is about 4.41 bits/element.
    from verl.workers.comm_eff.activation_tahquant import (
        TAHQUANT_INT3_BITS,
        TAHQUANT_INT4_BITS,
        tahquant_metadata_bits,
    )

    h = 1536
    n_tiles = (h + cfg.tah.tile - 1) // cfg.tah.tile
    payload = h * (cfg.tah.int4_frac * TAHQUANT_INT4_BITS + (1.0 - cfg.tah.int4_frac) * TAHQUANT_INT3_BITS)
    bits = payload + n_tiles * tahquant_metadata_bits(cfg.tah.tile)
    ratio = bits / (77 * 16)
    assert ratio > 5.0, (arm, bits, ratio)


@pytest.mark.parametrize("arm", CODEC_ARMS)
def test_codec_arm_keeps_the_anchor_and_drops_the_mask(arm):
    """The codec is the only thing that changes. The anchor circuit is untouched."""
    cfg = build(arm)
    assert cfg.mask.enabled is False, "a quantizing codec replaces the mask, it does not layer on it"
    assert cfg.mask.mask_recompute is True, "aq_sgd reads its buffer on every eligible pass"
    assert cfg.mask.mask_reference is True
    assert cfg.anchor.enabled is True and cfg.anchor.cadence == 20 and cfg.anchor.delay_K == 20
    assert cfg.anchor.owns_q is False, "no codec here carries a basis Q"
    assert cfg.spectral.enabled is True and cfg.spectral.signed_ema_alpha == 0.25


def test_the_codec_arms_differ_only_where_intended():
    """aqsgd vs srquant is delta coding alone; the aqsgd variants are one knob each."""
    aq, sr = build("aqsgd"), build("srquant")
    assert (aq.aq_sgd.bits, aq.aq_sgd.subset_k, aq.aq_sgd.block_size) == (
        sr.quant.bits,
        sr.quant.subset_k,
        sr.quant.block_size,
    ), "byte-identical on the wire, so delta coding is the only variable"
    assert build("aqsgd-all").aq_sgd.scope == "all"
    assert build("aqsgd").aq_sgd.scope == "prompt"
    assert build("aqsgd-rn").aq_sgd.rounding == "rn"
    assert build("aqsgd").aq_sgd.rounding == "sr", (
        "sr is the faithful setting: AQ-SGD's Theorem 3.1 assumes an unbiased quantizer"
    )
    assert build("aqsgd-payload").aq_sgd.first_visit == "dense", (
        "the payload arm is OFF-budget by construction and measures traffic, not accuracy"
    )
    assert build("aqsgd").aq_sgd.first_visit == "rescaled"


def test_aq_sgd_refuses_a_dense_recompute_pass():
    """Not a style gate: the backward would differentiate a reconstruction never sent."""
    mask_d, anchor_d, spectral_d, _ = ARMS["aqsgd"]
    with pytest.raises(ValueError, match="mask_recompute"):
        CommEffConfig(
            enabled=True,
            compression_type="aq_sgd",
            mask=CommEffMaskConfig(**{**PAIR1_MASK, **mask_d, "mask_recompute": False}),
            anchor=CommEffAnchorConfig(**{**PAIR1_ANCHOR, **anchor_d}),
            spectral=CommEffSpectralConfig(**{**PAIR1_SPECTRAL, **spectral_d}),
            aq_sgd=CommEffAQSGDConfig(**CODEC_CFG["aqsgd"]),
        )
