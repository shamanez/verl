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

"""Tests for the clean anchor policy-gradient loss.

The anchor must not reuse the fast-path PPO ratio/clip loss: masked
``old_log_probs`` against the anchor's unmasked forward can corrupt
``G_anchor``. ``anchor_pg_loss`` uses ratio == 1, no clip, and no
``old_log_probs``. These tests pin two invariants on a 2-token toy:

A. **Gradient == plain PG.** ``∂loss/∂θ == -(A · ∇logπ) / N`` (token-mean
   normalization), i.e. the clean unmasked policy gradient — no ratio, no clip.
B. **``old_log_probs`` is IGNORED.** The loss + gradient are byte-identical for
   wildly different ``old_log_probs`` (the corruption channel is gone), whereas
   the fast-path ``compute_policy_loss_vanilla`` produces a DIFFERENT gradient
   once ``old_log_probs != logπ`` (ratio != 1), proving the anchor and fast path
   intentionally differ here.

Unlike the file-path-isolated harness in ``test_anchor_queue.py``, these tests
import through the real ``verl`` package (the runner's env / the box both have
the full deps), so they exercise the actual ``anchor_pg_loss`` reduction. We
monkeypatch ``no_padding_2_padding`` to an identity extractor so the test isolates
the PG reduction + ratio==1 + no-old_log_probs from verl's nested
rmpad plumbing, which ``ppo_loss`` already shares and which is tested elsewhere.
"""

import pytest
import torch
from tensordict import TensorDict


# ---------------------------------------------------------------------------
# Minimal stand-ins so we do not need a real model / rollout to test the loss.
# ---------------------------------------------------------------------------
class _Cfg:
    """Just the attributes anchor_pg_loss reads off ActorConfig."""

    def __init__(self, loss_agg_mode="token-mean"):
        self.loss_agg_mode = loss_agg_mode
        self.loss_scale_factor = None
        self.global_batch_info = {}


def _make_batch(logits_param, advantages, response_mask, old_log_probs):
    """Build (model_output, data) for a 1-sequence, 2-token toy.

    ``logits_param`` is the single learnable scalar/tensor whose gradient we
    check; ``log_probs`` is a differentiable function of it so backward populates
    its ``.grad``. We bypass no_padding_2_padding (monkeypatched to identity) so
    ``model_output['log_probs']`` is already the padded (bsz, resp_len) tensor.
    """
    from verl.utils import tensordict_utils as tu

    bsz, resp_len = response_mask.shape
    model_output = {"log_probs": logits_param}
    data = TensorDict(
        {
            "response_mask": response_mask,
            "advantages": advantages,
            "old_log_probs": old_log_probs,
        },
        batch_size=[bsz],
    )
    # Non-tensor scalars anchor_pg_loss / agg_loss read off the batch (set via
    # the same util the engine uses, so they are stored as non-tensor metadata
    # and don't trip the TensorDict batch-dim check).
    tu.assign_non_tensor(data, dp_size=1)
    tu.assign_non_tensor(data, batch_num_tokens=None)  # token-mean -> mask.sum() at dp=1
    tu.assign_non_tensor(data, global_batch_size=None)
    return model_output, data


@pytest.fixture
def _identity_extract(monkeypatch):
    """Make no_padding_2_padding(tensor, data) -> tensor (identity).

    The response-log-prob extraction is IDENTICAL to ppo_loss's and tested in
    the engine path; here we isolate the C4 reduction. Patch it in BOTH the
    defining module and the anchor module's lazily-imported reference.
    """
    import verl.workers.utils.padding as padding_mod

    monkeypatch.setattr(padding_mod, "no_padding_2_padding", lambda tensor, data: tensor)
    yield


def test_anchor_pg_loss_gradient_equals_plain_pg(_identity_extract):
    """A: ∂loss/∂logπ == -(A · mask) / N — the clean policy gradient (token-mean)."""
    from verl.workers.comm_eff.anchor import anchor_pg_loss

    # 2 response tokens, both unmasked. logπ is the leaf we differentiate.
    log_probs = torch.tensor([[0.3, -0.7]], requires_grad=True)
    advantages = torch.tensor([[1.5, -2.0]])
    response_mask = torch.ones(1, 2)
    # Deliberately MASKED-path-style old_log_probs (very different from logπ);
    # the clean PG must IGNORE these entirely.
    old_log_probs = torch.tensor([[5.0, -5.0]])

    model_output, data = _make_batch(log_probs, advantages, response_mask, old_log_probs)
    cfg = _Cfg(loss_agg_mode="token-mean")

    loss, metrics = anchor_pg_loss(cfg, model_output, data)
    loss.backward()

    # Hand-computed clean PG gradient: loss = sum(-A·logπ·mask)/N, N=mask.sum().
    # ∂loss/∂logπ_i = -A_i · mask_i / N.
    n_tokens = response_mask.sum()
    expected_grad = -(advantages * response_mask) / n_tokens

    assert log_probs.grad is not None, "anchor_pg_loss did not populate logπ.grad"
    torch.testing.assert_close(log_probs.grad, expected_grad, rtol=1e-6, atol=1e-7)

    # Loss value equals the plain PG scalar too.
    expected_loss = (-(advantages * log_probs.detach() * response_mask)).sum() / n_tokens
    torch.testing.assert_close(loss.detach(), expected_loss, rtol=1e-6, atol=1e-7)

    # Ratio is identically 1.
    assert metrics["actor/anchor_ratio_mean"].values[0] == 1.0


def test_anchor_pg_loss_ignores_old_log_probs(_identity_extract):
    """B: loss + gradient are invariant to old_log_probs (ratio == 1, no clip)."""
    from verl.workers.comm_eff.anchor import anchor_pg_loss

    advantages = torch.tensor([[1.5, -2.0]])
    response_mask = torch.ones(1, 2)

    def _grad_for(old_lp):
        lp = torch.tensor([[0.3, -0.7]], requires_grad=True)
        mo, data = _make_batch(lp, advantages, response_mask, old_lp)
        loss, _ = anchor_pg_loss(_Cfg(), mo, data)
        loss.backward()
        return loss.detach().clone(), lp.grad.clone()

    loss_a, grad_a = _grad_for(torch.tensor([[0.3, -0.7]]))   # old == new (ratio 1 anyway)
    loss_b, grad_b = _grad_for(torch.tensor([[9.0, -9.0]]))   # old wildly off
    loss_c, grad_c = _grad_for(torch.tensor([[-4.0, 4.0]]))   # old wildly off, other sign

    # The clean PG must be byte-identical across all three: old_log_probs unused.
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    torch.testing.assert_close(loss_a, loss_c, rtol=0, atol=0)
    torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)
    torch.testing.assert_close(grad_a, grad_c, rtol=0, atol=0)


def test_fast_path_ppo_loss_DOES_depend_on_old_log_probs():
    """Contrast: compute_policy_loss_vanilla's grad changes with old_log_probs.

    This is the C1/C2/C3 corruption the anchor suffered. We compare ratio==1
    (old == new) vs ratio!=1 (old != new) on the SAME logπ leaf and assert the
    fast-path gradient DIFFERS — confirming the anchor genuinely needed C4 and
    that the fast path (untouched) still behaves as PPO.
    """
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    class _PPOCfg:
        # vanilla GRPO clip config; global_batch_info empty -> agg_loss uses
        # token-mean fallback (dp_size defaults to 1, batch_num_tokens=mask.sum).
        clip_ratio = 0.2
        clip_ratio_low = 0.2
        clip_ratio_high = 0.2
        global_batch_info = {}

        def get(self, k, default=None):
            return {"clip_ratio_c": 3.0}.get(k, default)

    advantages = torch.tensor([[1.5, -2.0]])
    response_mask = torch.ones(1, 2)

    def _grad_for(old_lp):
        lp = torch.tensor([[0.3, -0.7]], requires_grad=True)
        loss, _ = compute_policy_loss_vanilla(
            old_log_prob=old_lp,
            log_prob=lp,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=_PPOCfg(),
        )
        loss.backward()
        return lp.grad.clone()

    grad_ratio1 = _grad_for(torch.tensor([[0.3, -0.7]]))      # old == new => ratio 1
    grad_ratio_off = _grad_for(torch.tensor([[0.05, -0.9]]))  # small offset => ratio != 1

    # At ratio==1 the fast path reduces to the SAME clean PG the anchor computes.
    n_tokens = response_mask.sum()
    clean_pg = -(advantages * response_mask) / n_tokens
    torch.testing.assert_close(grad_ratio1, clean_pg, rtol=1e-5, atol=1e-6)

    # But with ratio != 1 the fast-path gradient is scaled by the ratio (and may
    # be clipped) => it DIFFERS from the clean PG. This is exactly the corruption
    # the anchor inherited by reusing ppo_loss; C4 removes it for the anchor.
    assert not torch.allclose(grad_ratio_off, clean_pg, rtol=1e-3, atol=1e-4), (
        "fast-path PPO grad did not change with old_log_probs — the ratio "
        "corruption premise (C1/C2/C3) is not reproduced; check the toy."
    )
