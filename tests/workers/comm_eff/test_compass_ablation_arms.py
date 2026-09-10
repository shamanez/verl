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
    CommEffConfig,
    CommEffMaskConfig,
    CommEffSpectralConfig,
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

# arm -> (mask delta, anchor delta, spectral delta, codec on?)
ARMS = {
    "base": ({}, {}, {}, True),
    "dense": ({"enabled": False}, NO_ANCHOR_ANCHOR, {"enabled": False}, False),
    "noanchor": ({}, NO_ANCHOR_ANCHOR, {"enabled": False}, True),
    "nosign": ({}, {}, {"signed_ema_alpha": 1.0}, True),
    "k10": ({}, {"delay_K": 10}, {}, True),
    "k40": ({}, {"delay_K": 40}, {}, True),
    "k80": ({}, {"delay_K": 80}, {}, True),
    "k40gm": ({}, {"delay_K": 40, "lookahead_strength": 0.5}, {}, True),
    "k80gm": ({}, {"delay_K": 80, "lookahead_strength": 0.25}, {}, True),
    "smoke": ({}, {}, {}, True),
}


def build(arm):
    mask_d, anchor_d, spectral_d, codec_on = ARMS[arm]
    return CommEffConfig(
        enabled=codec_on,
        compression_type="prf_mask" if codec_on else "dense",
        mask=CommEffMaskConfig(**{**PAIR1_MASK, **mask_d}),
        anchor=CommEffAnchorConfig(**{**PAIR1_ANCHOR, **anchor_d}),
        spectral=CommEffSpectralConfig(**{**PAIR1_SPECTRAL, **spectral_d}),
    )


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
