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

"""CPU unit tests for the EXP-29 anchor on-policy replay ring.

Loads ``anchor.py`` by file path (same harness as ``test_anchor_queue.py``) so
the heavy ``verl.__init__`` chain is not required. Pins the plan's CPU-gate
invariants:

1. **Exact pairing post-warmup** — ``get_replay(t)`` returns the ``t - delay_K``
   batch AND the generator snapshot of that batch's global step
   (``batch_gs == snapshot_gs`` structurally); realized weight staleness
   alternates K / K+1 on a 2-tick-per-step schedule.
2. **Warmup fallback** — before ``t > delay_K`` the oldest batch is replayed,
   still exactly paired with ITS OWN generator snapshot.
3. **Bounded memory** — at most ``delay_K + 1`` batches; snapshots are evicted
   as soon as no retained batch references them.
4. **Flag-OFF parity** — ``maybe_build_replay_ring`` returns ``None`` and
   builds NOTHING when ``replay_paired_batch`` is false (the CPU-testable
   half of the off-path-parity hard gate).
5. **NJT round trip + mutation isolation** — ``clone_batch_for_replay`` deep
   clones jagged and dense leaves; mutating the live batch (in place or via
   key assignment) never touches the stored clone.
6. **Value-level canary** — push-time fp32-on-CPU (norm, sum) fingerprints
   match bitwise after a bf16 cpu round trip + module load, and a single-bit
   weight perturbation is caught.
"""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _stub_parent_packages():
    # Prefer the REAL package when the host has the full dep chain: collection-
    # order independence. (test_anchor_queue.py's stub relies on
    # test_activation_mask.py importing real verl first — alphabetical luck; a
    # cherry-picked run starting with THIS file must not poison sys.modules
    # with empty-__path__ stubs that break later `verl.utils.*` imports.)
    try:
        import verl.workers.comm_eff  # noqa: F401

        return
    except Exception:
        pass
    for pkg in ("verl", "verl.workers", "verl.workers.comm_eff"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_parent_packages()
_anchor = _load("verl.workers.comm_eff.anchor", "verl/workers/comm_eff/anchor.py")

AnchorReplayRing = _anchor.AnchorReplayRing
clone_batch_for_replay = _anchor.clone_batch_for_replay
maybe_build_replay_ring = _anchor.maybe_build_replay_ring
snapshot_canary = _anchor.snapshot_canary
verify_canary_on_module = _anchor.verify_canary_on_module
snapshot_named_params = _anchor.snapshot_named_params


class _State:
    """Bare attribute bag standing in for CommEffState (plain __dict__)."""


class _AnchorCfg:
    def __init__(self, replay_paired_batch=False):
        self.replay_paired_batch = replay_paired_batch


def _fill_ring(ring, n_ticks, ticks_per_gs=2):
    """Simulate the engine's per-tick push schedule.

    Tick tau belongs to global step gs = (tau - 1) // ticks_per_gs + 1; the
    generator snapshot for gs is pushed at its FIRST tick (idempotent), then
    the tick's batch. The 'snapshot' payload records the tick it was taken at
    so pairing is value-checkable.
    """
    for tau in range(1, n_ticks + 1):
        gs = (tau - 1) // ticks_per_gs + 1
        if not ring.has_snapshot(gs):
            ring.push_snapshot(gs, {"w": torch.tensor(float(tau))}, canary={}, tick=tau)
        ring.push_batch(tau, {"batch_tick": tau}, gs)


# =========================================================================== #
# 1. Exact pairing post-warmup (+ K / K+1 realized weight staleness)
# =========================================================================== #
def test_exact_pairing_post_warmup():
    K = 5
    ring = AnchorReplayRing(delay_K=K)
    _fill_ring(ring, n_ticks=10, ticks_per_gs=2)
    used, batch, gs, snap, _canary, snap_tick, fallback = ring.get_replay(10, K)
    assert not fallback
    assert used == 10 - K == 5
    assert batch["batch_tick"] == 5
    # tick 5 belongs to gs 3 (ticks 5,6); its generator snapshot was pushed at
    # tick 5 (the first tick of gs 3) => weight staleness == K exactly here.
    assert gs == 3
    assert snap_tick == 5
    assert snap["w"].item() == 5.0
    assert 10 - snap_tick == K


def test_realized_weight_staleness_alternates_k_and_k_plus_1():
    K = 5
    ring = AnchorReplayRing(delay_K=K)
    _fill_ring(ring, n_ticks=15, ticks_per_gs=2)
    # Fire at tick 15: used tick 10 is the SECOND tick of gs 5 (ticks 9,10), so
    # the generator snapshot sits at tick 9 => realized weight delay K+1.
    used, _b, gs, _s, _c, snap_tick, fallback = ring.get_replay(15, K)
    assert not fallback and used == 10 and gs == 5 and snap_tick == 9
    assert 15 - snap_tick == K + 1
    # Weights are never FRESHER than K (the engine's hard assert).
    assert 15 - snap_tick >= K


def test_warmup_falls_back_to_oldest_with_exact_pairing():
    K = 5
    ring = AnchorReplayRing(delay_K=K)
    _fill_ring(ring, n_ticks=5, ticks_per_gs=2)  # fire at tick 5: t-K = 0 absent
    used, batch, gs, snap, _c, snap_tick, fallback = ring.get_replay(5, K)
    assert fallback
    assert used == 1 and batch["batch_tick"] == 1
    # The oldest batch still pairs with ITS OWN generator snapshot (gs 1 @ tick 1).
    assert gs == 1 and snap_tick == 1 and snap["w"].item() == 1.0


def test_empty_ring_returns_none():
    assert AnchorReplayRing(delay_K=3).get_replay(1) is None


# =========================================================================== #
# 2. Bounded memory
# =========================================================================== #
def test_batches_bounded_and_snapshots_evicted():
    K = 5
    ring = AnchorReplayRing(delay_K=K)
    _fill_ring(ring, n_ticks=40, ticks_per_gs=2)
    assert len(ring) == K + 1
    assert ring.batch_ticks == list(range(35, 41))
    # Retained batches (ticks 35..40) span gs 18..20: every other snapshot is gone.
    assert set(ring.snapshot_steps) == {18, 19, 20}


def test_push_snapshot_is_idempotent_per_gs():
    ring = AnchorReplayRing(delay_K=2)
    assert ring.push_snapshot(1, {"w": torch.tensor(1.0)}, tick=1) is True
    # Second tick of the same gs must NOT overwrite the first-tick snapshot.
    assert ring.push_snapshot(1, {"w": torch.tensor(99.0)}, tick=2) is False
    ring.push_batch(1, {"batch_tick": 1}, 1)
    *_, snap, _c, snap_tick, _fb = ring.get_replay(1, 0)
    assert snap["w"].item() == 1.0 and snap_tick == 1


def test_push_batch_requires_snapshot():
    ring = AnchorReplayRing(delay_K=2)
    with pytest.raises(AssertionError):
        ring.push_batch(1, {"batch_tick": 1}, gs=7)


# =========================================================================== #
# 2b. Fire-aware retention (operator requirement: bound = f(cadence, staleness))
# =========================================================================== #
def test_cadence_filter_retains_only_replayable_ticks():
    """cadence=5, delay_K=5: fires at 5,10,15,... request ticks 0,5,10,... — only
    ticks ≡ 0 (mod 5) are stored; everything else is rejected at push time."""
    K, C = 5, 5
    ring = AnchorReplayRing(delay_K=K, cadence=C)
    assert ring._maxlen == 2  # delay_K // cadence + 1
    stored = []
    for tau in range(1, 16):
        gs = (tau - 1) // 2 + 1
        if not ring.has_snapshot(gs):
            ring.push_snapshot(gs, {"w": torch.tensor(float(tau))}, tick=tau)
        if ring.push_batch(tau, {"batch_tick": tau}, gs):
            stored.append(tau)
    assert stored == [5, 10, 15]
    assert ring.batch_ticks == [10, 15]  # maxlen 2 — tick 5 evicted after 15
    # Snapshots: only the gs of retained batches (gs 5 for tick 10, gs 8 for 15).
    assert set(ring.snapshot_steps) == {5, 8}


def test_cadence_filter_exact_pairing_at_fires():
    K, C = 5, 5
    ring = AnchorReplayRing(delay_K=K, cadence=C)
    for tau in range(1, 16):
        gs = (tau - 1) // 2 + 1
        if not ring.has_snapshot(gs):
            ring.push_snapshot(gs, {"w": torch.tensor(float(tau))}, tick=tau)
        ring.push_batch(tau, {"batch_tick": tau}, gs)
        if tau == 5:
            # Warmup fire: falls back to tick 5's own batch, exactly paired
            # with ITS generator snapshot (gs 3 @ tick 5).
            used, b, gs_got, _s, _c, snap_tick, fb = ring.get_replay(5, K)
            assert fb and used == 5 and gs_got == 3 and snap_tick == 5
        if tau == 10:
            used, b, gs_got, _s, _c, snap_tick, fb = ring.get_replay(10, K)
            assert not fb and used == 5 and b["batch_tick"] == 5
            assert gs_got == 3 and snap_tick == 5 and 10 - snap_tick == K
        if tau == 15:
            used, b, gs_got, _s, _c, snap_tick, fb = ring.get_replay(15, K)
            assert not fb and used == 10 and b["batch_tick"] == 10
            assert gs_got == 5 and snap_tick == 9 and 15 - snap_tick == K + 1


def test_cadence_filter_bounds_hold_over_long_run():
    """The 'nothing blows up' guard: bounds hold over hundreds of ticks for
    several (delay_K, cadence, ticks_per_gs) shapes."""
    for K, C, tpg in [(5, 5, 2), (5, 1, 2), (7, 5, 2), (2, 5, 1), (5, 2, 3)]:
        ring = AnchorReplayRing(delay_K=K, cadence=C)
        for tau in range(1, 301):
            gs = (tau - 1) // tpg + 1
            if not ring.has_snapshot(gs):
                ring.push_snapshot(gs, {"w": torch.tensor(float(tau))}, tick=tau)
            ring.push_batch(tau, {"batch_tick": tau}, gs)
            assert len(ring) <= K // C + 1
            assert len(ring.snapshot_steps) <= K // C + 2
            # Post-warmup fires must always find their exact pair.
            if tau % C == 0 and tau > K:
                used, _b, _g, _s, _c, snap_tick, fb = ring.get_replay(tau, K)
                assert not fb and used == tau - K, (K, C, tpg, tau, used, fb)
                assert tau - snap_tick >= K


def test_cadence_one_retains_every_tick():
    ring = AnchorReplayRing(delay_K=3, cadence=1)
    for tau in (1, 2, 3):
        ring.push_snapshot(tau, {"w": torch.tensor(float(tau))}, tick=tau)
        assert ring.tick_retained(tau)
        assert ring.push_batch(tau, {"batch_tick": tau}, tau) is True
    assert ring.batch_ticks == [1, 2, 3]


# =========================================================================== #
# 3. Flag-OFF parity (the CPU-testable half of the off-path-parity hard gate)
# =========================================================================== #
def test_flag_off_builds_no_ring():
    state = _State()
    out = maybe_build_replay_ring(state, _AnchorCfg(replay_paired_batch=False), delay_K=5)
    assert out is None
    assert not hasattr(state, "_anchor_replay_ring")


def test_flag_on_builds_and_caches_ring():
    state = _State()
    r1 = maybe_build_replay_ring(state, _AnchorCfg(replay_paired_batch=True), delay_K=5)
    r2 = maybe_build_replay_ring(state, _AnchorCfg(replay_paired_batch=True), delay_K=5)
    assert isinstance(r1, AnchorReplayRing) and r1 is r2
    assert state._anchor_replay_ring is r1


# =========================================================================== #
# 4. clone_batch_for_replay: deep clone, NJT round trip, mutation isolation
# =========================================================================== #
tensordict = pytest.importorskip("tensordict")
TensorDict = tensordict.TensorDict


def _njt(lengths, dim=4):
    return torch.nested.as_nested_tensor(
        [torch.randn(n, dim) for n in lengths], layout=torch.jagged
    )


def test_clone_dense_leaf_mutation_isolation():
    td = TensorDict({"adv": torch.randn(3, 7)}, batch_size=[3])
    clone = clone_batch_for_replay(td, device=torch.device("cpu"))
    before = clone["adv"].clone()
    td["adv"].add_(123.0)        # in-place mutation of the live leaf
    td["new_key"] = torch.zeros(3)  # key assignment on the live mapping
    assert torch.equal(clone["adv"], before)
    assert "new_key" not in clone.keys()


def test_clone_njt_round_trip_identity():
    nt = _njt([3, 5, 2])
    td = TensorDict({"input_ids": nt}, batch_size=[3])
    clone = clone_batch_for_replay(td, device=torch.device("cpu"))
    got = clone["input_ids"]
    assert got.is_nested
    assert torch.equal(got.values(), nt.values())
    assert torch.equal(got.offsets(), nt.offsets())
    assert got._ragged_idx == nt._ragged_idx
    # Deep clone: mutating the live NJT's values must not touch the stored copy.
    ref = got.values().clone()
    nt.values().add_(7.0)
    assert torch.equal(clone["input_ids"].values(), ref)


def test_clone_preserves_non_tensor_entries():
    td = TensorDict({"adv": torch.randn(2, 3)}, batch_size=[2])
    sys.path.insert(0, str(_REPO))
    try:
        from verl.utils import tensordict_utils as tu
    except Exception:
        pytest.skip("full verl import chain unavailable on this host")
    finally:
        sys.path.pop(0)
    tu.assign_non_tensor(td, dp_size=4)
    clone = clone_batch_for_replay(td, device=torch.device("cpu"))
    assert tu.get_non_tensor_data(clone, key="dp_size", default=None) == 4
    # Re-assigning on the LIVE batch (what _forward_backward_batch_inner does)
    # must not leak into the stored clone.
    tu.assign_non_tensor(td, dp_size=8)
    assert tu.get_non_tensor_data(clone, key="dp_size", default=None) == 4


# =========================================================================== #
# 5. Value-level canary
# =========================================================================== #
class _Tiny(torch.nn.Module):
    def __init__(self, d=8, dtype=torch.float32):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d, bias=False).to(dtype)
        self.o_proj = torch.nn.Linear(d, d, bias=False).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_canary_matches_after_snapshot_load(dtype):
    src = _Tiny(dtype=dtype)
    snap = snapshot_named_params(src.named_parameters(), device=torch.device("cpu"))
    canary = snapshot_canary(snap, target_substrs=("q_proj", "o_proj"))
    assert len(canary) == 2
    # Load the snapshot into a fresh module (the clone), then verify bitwise.
    dst = _Tiny(dtype=dtype)
    with torch.no_grad():
        for n, p in dst.named_parameters():
            p.copy_(snap[n].to(p.device, p.dtype))
    ok, results = verify_canary_on_module(dst, canary)
    assert ok, results


def test_canary_catches_perturbation():
    src = _Tiny(dtype=torch.bfloat16)
    snap = snapshot_named_params(src.named_parameters(), device=torch.device("cpu"))
    canary = snapshot_canary(snap, target_substrs=("q_proj", "o_proj"))
    dst = _Tiny(dtype=torch.bfloat16)
    with torch.no_grad():
        for n, p in dst.named_parameters():
            p.copy_(snap[n].to(p.device, p.dtype))
        # Single-element perturbation in ONE canary target.
        dst.q_proj.weight[0, 0] += 1.0
    ok, _results = verify_canary_on_module(dst, canary)
    assert not ok


def test_canary_target_choice_is_deterministic():
    src = _Tiny()
    snap = snapshot_named_params(src.named_parameters(), device=torch.device("cpu"))
    c1 = snapshot_canary(snap, target_substrs=("q_proj", "o_proj"))
    c2 = snapshot_canary(snap, target_substrs=("q_proj", "o_proj"))
    assert list(c1.keys()) == list(c2.keys()) == sorted(c1.keys())
    assert c1 == c2
