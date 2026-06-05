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

"""Unit tests for the PowerSGD activation-compression codec.

These are the CPU-runnable correctness invariants:

  * autograd / no-STE — backward equals the self-adjoint projector ``QQᵀ``;
  * r=H lossless limiting case — ``M_hat == M`` to fp tolerance;
  * determinism / multi-rank Q — identical basis from the same per-layer seed;
  * fp32 orthonormality / finite ``q_cond``;
  * frozen-Q-across-the-step — the basis advances ONLY at ``maybe_update_basis``
    (after the gradient-bearing work), never inside a forward;
  * sketch gating — V accumulates on the gradient-bearing train forward only,
    once per forward generation (grad-ckpt-recompute-safe), never on the
    no-grad old-logprob recompute.
"""

import unittest

import torch
import torch.nn as nn

from verl.workers.comm_eff.powersgd_activation import (
    PowerSGDActivationCompressor,
    init_basis,
    orthonormalize,
    powersgd_layer_seed,
)
from verl.workers.comm_eff.state import TRAIN_TAG


class _FakeState:
    """Minimal CommEffState stand-in carrying the path tag + op counters."""

    def __init__(self, path_tag=TRAIN_TAG):
        self.path_tag = path_tag
        self.powersgd_applications = 0
        self.powersgd_basis_updates = 0

    def note_powersgd_application(self):
        self.powersgd_applications += 1

    def note_powersgd_basis_update(self):
        self.powersgd_basis_updates += 1


class _TinyModel(nn.Module):
    """4 linear "decoder blocks" (H->H). pp_size=4 -> boundaries [0, 1, 2]."""

    def __init__(self, H):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in range(4)])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


class TestPowerSGDSeed(unittest.TestCase):
    def test_seed_formula_inf13(self):
        # seed_L = (base*1_000_003 + layer*7919) & 0x7FFFFFFF
        self.assertEqual(powersgd_layer_seed(0, 0), 0)
        self.assertEqual(powersgd_layer_seed(0, 1), 7919)
        self.assertEqual(powersgd_layer_seed(1, 0), 1_000_003)
        # masked into int31
        self.assertLess(powersgd_layer_seed(10_000, 10_000), 0x80000000)

    def test_determinism_multi_rank(self):
        # Same seed/layer => bit-identical basis (zero-comm bootstrap; identical
        # on every "rank" since it's a pure function of the seed).
        a = init_basis(hidden_size=128, rank=16, base_seed=0, layer_idx=3)
        b = init_basis(hidden_size=128, rank=16, base_seed=0, layer_idx=3)
        self.assertTrue(torch.equal(a, b))
        c = init_basis(hidden_size=128, rank=16, base_seed=0, layer_idx=5)
        self.assertFalse(torch.equal(a, c))


class TestPowerSGDBasisMath(unittest.TestCase):
    def test_orthonormal_fp32(self):
        Q = init_basis(hidden_size=256, rank=32, base_seed=0, layer_idx=1)
        self.assertEqual(Q.dtype, torch.float32)
        err = (Q.t() @ Q - torch.eye(32)).abs().max().item()
        self.assertLess(err, 1e-4)
        sv = torch.linalg.svdvals(Q)
        q_cond = (sv.max() / sv.min()).item()
        self.assertTrue(torch.isfinite(torch.tensor(q_cond)))
        self.assertLess(q_cond, 1.01)

    def test_orth_rank_deficient_repaired(self):
        H, r = 64, 16
        V = torch.zeros(H, r)
        V[:, 0] = torch.randn(H)  # rank-1 sketch
        Q = orthonormalize(V)
        self.assertTrue(torch.isfinite(Q).all())
        sv = torch.linalg.svdvals(Q)
        self.assertGreater(sv.min().item(), 0.5)  # full numerical rank

    def test_orth_nonfinite_sketch_does_not_propagate_nan(self):
        H, r = 64, 16
        V = torch.full((H, r), float("nan"))
        Q = orthonormalize(V)
        self.assertTrue(torch.isfinite(Q).all())


class TestPowerSGDProjector(unittest.TestCase):
    def test_autograd_no_ste(self):
        # backward of M_hat=(M@Q)@Qᵀ must equal the self-adjoint projector:
        # dL/dM = (dL/dM_hat) Q Qᵀ. No straight-through.
        H, r, N = 256, 32, 17
        Q = init_basis(hidden_size=H, rank=r, base_seed=0, layer_idx=2)
        M = torch.randn(N, H, requires_grad=True)
        Mhat = (M @ Q) @ Q.t()
        g = torch.randn(N, H)
        (Mhat * g).sum().backward()
        analytic = (g @ Q) @ Q.t()
        rel = (M.grad - analytic).norm() / analytic.norm()
        self.assertLess(rel.item(), 1e-5)

    def test_r_equals_H_lossless(self):
        # With rank == H the projector is the identity, M_hat == M.
        H, N = 128, 9
        Qfull = init_basis(hidden_size=H, rank=H, base_seed=0, layer_idx=7)
        M = torch.randn(N, H)
        Mhat = (M @ Qfull) @ Qfull.t()
        rel = ((M - Mhat).norm() / M.norm()).item()
        self.assertLess(rel, 1e-4)


class TestPowerSGDCompressorLifecycle(unittest.TestCase):
    def _build(self, **kw):
        torch.manual_seed(0)
        model = _TinyModel(kw.pop("H", 64))
        state = _FakeState()
        comp = PowerSGDActivationCompressor(
            rank=kw.pop("rank", 16),
            base_seed=0,
            pp_size=4,
            update_cadence=kw.pop("update_cadence", 1),
            warm_start=kw.pop("warm_start", True),
            compress_recompute=True,
            sync_basis=False,
            qr_dtype="fp32",
            reortho_eps=1e-6,
            state=state,
        )
        comp.register(model)
        return model, comp, state

    def test_boundaries_match_mask_selection(self):
        _, comp, _ = self._build()
        self.assertEqual(comp.boundary_indices, [0, 1, 2])

    def test_sketch_accumulates_on_train_forward(self):
        H = 64
        model, comp, state = self._build(H=H)
        comp.set_context(global_step=1)
        x = torch.randn(20, H, requires_grad=True)
        model(x).pow(2).sum().backward()
        self.assertEqual(state.powersgd_applications, 3)
        self.assertEqual(set(comp._sketch), {0, 1, 2})
        self.assertTrue(all(v == 1 for v in comp._sketch_count.values()))

    def test_grad_ckpt_recompute_not_double_counted(self):
        H = 64
        model, comp, _ = self._build(H=H)
        comp.set_context(global_step=1)
        x = torch.randn(20, H, requires_grad=True)
        model(x).pow(2).sum().backward()
        before = {k: v.clone() for k, v in comp._sketch.items()}
        # Re-run the forward in the SAME generation (the grad-ckpt recompute
        # reuses the context set_context stamped) — sketch must NOT change.
        model(x)
        for k in comp._sketch:
            self.assertTrue(torch.equal(comp._sketch[k], before[k]))

    def test_old_logprob_recompute_projects_but_no_sketch(self):
        H = 64
        model, comp, state = self._build(H=H)
        comp.set_context(global_step=1)
        model(torch.randn(20, H, requires_grad=True)).pow(2).sum().backward()
        n_app = state.powersgd_applications
        counts_before = dict(comp._sketch_count)
        # old-logprob recompute: no_grad + tag flipped => projects, no sketch.
        state.path_tag = "old_logprob"
        comp.set_context(global_step=1)
        with torch.no_grad():
            model(torch.randn(20, H))
        self.assertGreater(state.powersgd_applications, n_app)  # projected
        self.assertEqual(comp._sketch_count, counts_before)  # no new sketch

    def test_q_frozen_within_step_advances_only_on_update(self):
        H = 64
        model, comp, state = self._build(H=H)
        comp.set_context(global_step=1)
        Q0 = {k: comp._basis.get(k, None) for k in comp.boundary_indices}
        # First forward bootstraps the basis; capture AFTER it exists.
        model(torch.randn(20, H, requires_grad=True)).pow(2).sum().backward()
        Q_after_fwd = {k: comp._basis[k].clone() for k in comp._basis}
        # The basis is unchanged by additional forwards within the step.
        comp.set_context(global_step=1)
        model(torch.randn(20, H, requires_grad=True)).pow(2).sum().backward()
        for k in comp._basis:
            self.assertTrue(torch.equal(comp._basis[k], Q_after_fwd[k]))
        # Only maybe_update_basis advances it, and the result is orthonormal.
        updated = comp.maybe_update_basis(is_clean_step=False)
        self.assertTrue(updated)
        self.assertEqual(state.powersgd_basis_updates, 1)
        for k in comp._basis:
            Q = comp._basis[k]
            self.assertFalse(torch.equal(Q, Q_after_fwd[k]))
            err = (Q.t() @ Q - torch.eye(Q.shape[1])).abs().max().item()
            self.assertLess(err, 1e-4)
        self.assertFalse(comp._sketch)  # cleared

    def test_clean_step_does_not_update_basis(self):
        H = 64
        model, comp, _ = self._build(H=H)
        comp.set_context(global_step=2)
        model(torch.randn(20, H, requires_grad=True)).pow(2).sum().backward()
        self.assertFalse(comp.maybe_update_basis(is_clean_step=True))
        self.assertFalse(comp._sketch)

    def test_update_cadence_gates_basis_update(self):
        H = 64
        model, comp, _ = self._build(H=H, update_cadence=2)
        # step 1 (odd) — no update under cadence 2
        comp.set_context(global_step=1)
        model(torch.randn(20, H, requires_grad=True)).pow(2).sum().backward()
        self.assertFalse(comp.maybe_update_basis(is_clean_step=False))
        # step 2 (even) — updates
        comp.set_context(global_step=2)
        model(torch.randn(20, H, requires_grad=True)).pow(2).sum().backward()
        self.assertTrue(comp.maybe_update_basis(is_clean_step=False))

    def test_diagnostics_finite(self):
        H = 64
        model, comp, _ = self._build(H=H)
        comp.set_context(global_step=1)
        model(torch.randn(20, H, requires_grad=True)).pow(2).sum().backward()
        self.assertEqual(set(comp.last_q_cond), {0, 1, 2})
        for v in comp.last_q_cond.values():
            self.assertTrue(torch.isfinite(torch.tensor(v)))
            self.assertLess(v, 1.01)  # orthonormal basis => cond ≈ 1
        for v in comp.last_reconstruction_rel_error.values():
            self.assertLessEqual(v, 1.0 + 1e-6)
        self.assertEqual(comp.last_y_coords_per_token, 16)


class TestPowerSGDCollectiveHelpers(unittest.TestCase):
    """CPU-checkable helper invariants for the real DP sync path.

    Actual all-reduce/all-gather behavior and cross-rank Q agreement are
    multi-GPU probe requirements, not single-process unit-test claims. These
    tests only cover deterministic helper behavior used by that path."""

    def _build(self, **kw):
        torch.manual_seed(0)
        model = _TinyModel(64)
        comp = PowerSGDActivationCompressor(
            rank=16, base_seed=0, pp_size=4, update_cadence=1, warm_start=True,
            compress_recompute=True, sync_basis=kw.pop("sync_basis", True),
            qr_dtype="fp32", reortho_eps=1e-6, state=_FakeState(),
        )
        comp.register(model)
        return model, comp

    def test_boundary_for_update_is_fixed_sorted(self):
        # The deadlock guard: every rank iterates the FIXED sorted boundary set,
        # NOT its rank-local sketch keys.
        _, comp = self._build()
        self.assertEqual(comp._boundary_for_update(), [0, 1, 2])
        # Even if the local sketch is missing a boundary, the update set is fixed.
        comp.set_context(global_step=1)
        # simulate a sketch that only has boundary 0
        comp._sketch = {0: torch.randn(64, 16)}
        comp._sketch_count = {0: 1}
        self.assertEqual(comp._boundary_for_update(), [0, 1, 2])

    def test_basis_checksums_deterministic_and_sensitive(self):
        model, comp = self._build()
        comp.set_context(global_step=1)
        # bootstrap the per-boundary bases by firing the registered model.
        model(torch.randn(8, 64, requires_grad=True)).pow(2).sum().backward()
        s1 = comp.basis_checksums()
        s2 = comp.basis_checksums()
        self.assertEqual(s1, s2)  # deterministic
        self.assertEqual(set(s1), {0, 1, 2})  # one checksum per boundary
        # mutate one Q column -> checksum for that boundary must change
        k = sorted(s1)[0]
        comp._basis[k] = comp._basis[k].clone()
        comp._basis[k][:, 0] *= -1.0  # sign flip
        s3 = comp.basis_checksums()
        self.assertNotEqual(s1[k], s3[k])

    def test_set_dp_group_none_is_world(self):
        _, comp = self._build()
        self.assertIsNone(comp._dp_group())  # default => world
        sentinel = object()
        comp.set_dp_group(sentinel)
        self.assertIs(comp._dp_group(), sentinel)


class TestPowerSGDConfigSyncDefault(unittest.TestCase):
    def test_sync_basis_defaults_true(self):
        # Operator clarification: the shared codebook MUST be synced under DP.
        from verl.workers.config import CommEffPowerSGDConfig

        self.assertTrue(CommEffPowerSGDConfig().sync_basis)


if __name__ == "__main__":
    unittest.main()
