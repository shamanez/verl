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
``old_log_probs``. These tests pin three invariants on a 2-token toy:

A. **Gradient == plain PG when KL is disabled.** ``∂loss/∂θ ==
   -(A · ∇logπ) / N`` (token-mean normalization), i.e. the clean unmasked
   policy gradient — no ratio, no clip.
B. **``old_log_probs`` is IGNORED.** The loss + gradient are byte-identical for
   wildly different ``old_log_probs`` (the corruption channel is gone), whereas
   the fast-path ``compute_policy_loss_vanilla`` produces a DIFFERENT gradient
   once ``old_log_probs != logπ`` (ratio != 1), proving the anchor and fast path
   intentionally differ here.
C. **Configured KL is retained.** With ``use_kl_loss=true``, the anchor adds the
   same reference-policy KL penalty, coefficient, mask, and normalization as
   the locked fast objective without reintroducing the PPO ratio.

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

    def __init__(
        self,
        loss_agg_mode="token-mean",
        *,
        use_kl_loss=False,
        kl_loss_coef=0.001,
        kl_loss_type="low_var_kl",
    ):
        self.loss_agg_mode = loss_agg_mode
        self.loss_scale_factor = None
        self.global_batch_info = {}
        self.use_kl_loss = use_kl_loss
        self.kl_loss_coef = kl_loss_coef
        self.kl_loss_type = kl_loss_type


def _make_batch(logits_param, advantages, response_mask, old_log_probs, ref_log_prob=None):
    """Build (model_output, data) for a 1-sequence, 2-token toy.

    ``logits_param`` is the single learnable scalar/tensor whose gradient we
    check; ``log_probs`` is a differentiable function of it so backward populates
    its ``.grad``. We bypass no_padding_2_padding (monkeypatched to identity) so
    ``model_output['log_probs']`` is already the padded (bsz, resp_len) tensor.
    """
    from verl.utils import tensordict_utils as tu

    bsz, resp_len = response_mask.shape
    model_output = {"log_probs": logits_param}
    tensors = {
        "response_mask": response_mask,
        "advantages": advantages,
        "old_log_probs": old_log_probs,
    }
    if ref_log_prob is not None:
        tensors["ref_log_prob"] = ref_log_prob
    data = TensorDict(tensors, batch_size=[bsz])
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

    loss_a, grad_a = _grad_for(torch.tensor([[0.3, -0.7]]))  # old == new (ratio 1 anyway)
    loss_b, grad_b = _grad_for(torch.tensor([[9.0, -9.0]]))  # old wildly off
    loss_c, grad_c = _grad_for(torch.tensor([[-4.0, 4.0]]))  # old wildly off, other sign

    # The clean PG must be byte-identical across all three: old_log_probs unused.
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    torch.testing.assert_close(loss_a, loss_c, rtol=0, atol=0)
    torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)
    torch.testing.assert_close(grad_a, grad_c, rtol=0, atol=0)


def test_anchor_pg_loss_retains_configured_kl(_identity_extract):
    """C: ratio-one anchor objective includes the locked reference KL term."""
    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
    from verl.workers.comm_eff.anchor import anchor_pg_loss

    values = torch.tensor([[0.3, -0.7]])
    advantages = torch.tensor([[1.5, -2.0]])
    response_mask = torch.ones(1, 2)
    old_log_probs = torch.tensor([[9.0, -9.0]])
    ref_log_prob = torch.tensor([[0.1, -0.4]])
    cfg = _Cfg(use_kl_loss=True, kl_loss_coef=0.001, kl_loss_type="low_var_kl")

    anchor_log_probs = values.clone().requires_grad_(True)
    model_output, data = _make_batch(
        anchor_log_probs,
        advantages,
        response_mask,
        old_log_probs,
        ref_log_prob,
    )
    anchor_loss, metrics = anchor_pg_loss(cfg, model_output, data)
    anchor_loss.backward()

    manual_log_probs = values.clone().requires_grad_(True)
    global_batch_info = {
        "dp_size": 1,
        "batch_num_tokens": None,
        "global_batch_size": None,
        "loss_scale_factor": None,
    }
    manual_pg = agg_loss(
        loss_mat=-advantages * manual_log_probs,
        loss_mask=response_mask.bool(),
        loss_agg_mode="token-mean",
        **global_batch_info,
    )
    manual_kld = kl_penalty(
        logprob=manual_log_probs,
        ref_logprob=ref_log_prob,
        kl_penalty="low_var_kl",
    )
    manual_kl = agg_loss(
        loss_mat=manual_kld,
        loss_mask=response_mask.bool(),
        loss_agg_mode="token-mean",
        **global_batch_info,
    )
    manual_loss = manual_pg + 0.001 * manual_kl
    manual_loss.backward()

    torch.testing.assert_close(anchor_loss.detach(), manual_loss.detach(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(anchor_log_probs.grad, manual_log_probs.grad, rtol=1e-6, atol=1e-7)
    assert metrics["actor/anchor_kl_loss"].values[0] == pytest.approx(float(manual_kl.detach()), rel=1e-6, abs=1e-7)
    assert metrics["actor/anchor_kl_coef"] == 0.001
    assert metrics["actor/anchor_total_loss"].values[0] == pytest.approx(
        float(manual_loss.detach()), rel=1e-6, abs=1e-7
    )


def test_anchor_pg_loss_kl_depends_on_reference_not_old_policy(_identity_extract):
    """KL-enabled anchor remains old-policy invariant but reference-sensitive."""
    from verl.workers.comm_eff.anchor import anchor_pg_loss

    advantages = torch.tensor([[1.5, -2.0]])
    response_mask = torch.ones(1, 2)
    cfg = _Cfg(use_kl_loss=True)

    def _loss_grad(old_lp, ref_lp):
        lp = torch.tensor([[0.3, -0.7]], requires_grad=True)
        mo, data = _make_batch(lp, advantages, response_mask, old_lp, ref_lp)
        loss, _ = anchor_pg_loss(cfg, mo, data)
        loss.backward()
        return loss.detach().clone(), lp.grad.clone()

    loss_a, grad_a = _loss_grad(torch.tensor([[9.0, -9.0]]), torch.tensor([[0.1, -0.4]]))
    loss_b, grad_b = _loss_grad(torch.tensor([[-4.0, 4.0]]), torch.tensor([[0.1, -0.4]]))
    loss_c, grad_c = _loss_grad(torch.tensor([[9.0, -9.0]]), torch.tensor([[-0.8, 0.2]]))

    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)
    assert not torch.allclose(loss_a, loss_c, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(grad_a, grad_c, rtol=1e-6, atol=1e-7)


def test_anchor_pg_loss_kl_requires_reference_log_probs(_identity_extract):
    """KL-enabled anchor fails closed when its reference-policy term is absent."""
    from verl.workers.comm_eff.anchor import anchor_pg_loss

    log_probs = torch.tensor([[0.3, -0.7]], requires_grad=True)
    model_output, data = _make_batch(
        log_probs,
        torch.tensor([[1.5, -2.0]]),
        torch.ones(1, 2),
        torch.tensor([[9.0, -9.0]]),
    )

    with pytest.raises(KeyError, match="ref_log_prob"):
        anchor_pg_loss(_Cfg(use_kl_loss=True), model_output, data)


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

    grad_ratio1 = _grad_for(torch.tensor([[0.3, -0.7]]))  # old == new => ratio 1
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


def test_anchor_pg_loss_round_trips_through_replay_clone(_identity_extract):
    """paired replay: anchor_pg_loss on a replay-ring batch clone is BYTE-identical to
    the loss on the original batch.

    The replay ring stores ``clone_batch_for_replay(data, device=cpu)`` and the
    anchor later consumes the clone instead of the live batch. The clone must
    be value-transparent: same loss scalar, same gradient, bit for bit — the
    deep clone changes WHICH batch the anchor sees (the t-K one), never the
    numerics of how a given batch is consumed.
    """
    from verl.workers.comm_eff.anchor import anchor_pg_loss, clone_batch_for_replay

    advantages = torch.tensor([[1.5, -2.0]])
    response_mask = torch.ones(1, 2)
    old_log_probs = torch.tensor([[5.0, -5.0]])
    ref_log_prob = torch.tensor([[0.1, -0.4]])
    cfg = _Cfg(use_kl_loss=True)

    def _loss_grad(data):
        lp = torch.tensor([[0.3, -0.7]], requires_grad=True)
        mo = {"log_probs": lp}
        loss, _ = anchor_pg_loss(cfg, mo, data)
        loss.backward()
        return loss.detach().clone(), lp.grad.clone()

    _mo, data = _make_batch(
        torch.tensor([[0.3, -0.7]], requires_grad=True),
        advantages,
        response_mask,
        old_log_probs,
        ref_log_prob,
    )
    cloned = clone_batch_for_replay(data, device=torch.device("cpu"))

    loss_orig, grad_orig = _loss_grad(data)
    loss_clone, grad_clone = _loss_grad(cloned)
    torch.testing.assert_close(loss_orig, loss_clone, rtol=0, atol=0)
    torch.testing.assert_close(grad_orig, grad_clone, rtol=0, atol=0)

    # Mutating the ORIGINAL batch after cloning (what the fast path does in
    # place) must not change what the clone computes.
    data["advantages"].mul_(-3.0)
    loss_clone2, grad_clone2 = _loss_grad(cloned)
    torch.testing.assert_close(loss_clone, loss_clone2, rtol=0, atol=0)
    torch.testing.assert_close(grad_clone, grad_clone2, rtol=0, atol=0)
