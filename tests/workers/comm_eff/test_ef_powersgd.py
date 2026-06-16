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

"""comm-eff CPU unit tests for the new code: the direction-preserving ef_powersgd
merger, the Q-basis-family guard, and the diagnostic-capture writer.

These exercise the Correctness invariants the on-box pre-run probe also gates on,
on CPU with no distributed runtime, so a hard-gate regression is caught here
before any GPU-hour is spent:

* EF residual limiting-case identity (ef_decay=ef_clip=0 ⇒ G_corr == G_comp)
* EF merger has NO sign term (G_corr shares sign of G_comp on the re-injection)
* EF residual is shape-aware (reset on shape change), clipped, detached
* off-path parity of the capture config (capture defaults OFF, no side effect)
* config validation (ef_powersgd accepted; ef_decay/ef_clip/q_basis ranges)
"""

import os
import tempfile

import pytest
import torch

from verl.workers.comm_eff.spectral_filter import SpectralFilter, apply_spectral_correction_to_params


class _FakeState:
    """Minimal stand-in for CommEffState the FSDP-agnostic core reads."""

    def __init__(self):
        self.fsdp_grad_repr = {}
        self.spectral_rel_change = {}
        self.spectral_corrections = 0
        self.merger_coldM_fallbacks = 0
        self.residual_reset_on_shape_mismatch = 0
        self.global_step = 1
        self.spectral_step = 1
        self._capture_writer = None


def _warm_anchor(sf: SpectralFilter, name: str, g: torch.Tensor):
    """Warm M_anchor for ``name`` with a non-trivial anchor gradient."""
    sf.update_anchor(name, g)


# --------------------------------------------------------------------------- #
# EF residual limiting-case identity  (hard gate)
# --------------------------------------------------------------------------- #
def test_ef_powersgd_limiting_case_is_plain_powersgd():
    """ef_decay=0 AND ef_clip=0 ⇒ G_corr == G_comp bit-for-bit (the residual is a no-op)."""
    sf = SpectralFilter(correction_mode="ef_powersgd", ef_decay=0.0, ef_clip=0.0)
    torch.manual_seed(0)
    g_comp = torch.randn(8, 6)
    # Warm M with a DIFFERENT direction so a non-zero residual WOULD exist if applied.
    _warm_anchor(sf, "layer.q_proj.weight", torch.randn(8, 6))
    g_corr = sf.ef_powersgd_matrix("layer.q_proj.weight", g_comp)
    assert torch.equal(g_corr, g_comp), "ef_decay=ef_clip=0 must reduce ef_powersgd to plain PowerSGD"


def test_ef_powersgd_cold_M_returns_g_comp_unchanged():
    """An unwarmed (cold) anchor M ⇒ G_corr == G_comp (no silent grad change)."""
    sf = SpectralFilter(correction_mode="ef_powersgd", ef_decay=0.5, ef_clip=1.0)
    g_comp = torch.randn(4, 4)
    before = sf.merger_coldM_fallbacks
    g_corr = sf.ef_powersgd_matrix("layer.k_proj.weight", g_comp)  # M never warmed
    assert torch.equal(g_corr, g_comp)
    assert sf.merger_coldM_fallbacks == before + 1


# --------------------------------------------------------------------------- #
# EF merger has NO sign term  (hard gate)
# --------------------------------------------------------------------------- #
def test_ef_powersgd_preserves_sign_of_g_comp():
    """The re-injected residual must not flip G_comp's sign on the active entries.

    ef_powersgd adds the OFF-SUBSPACE component of M (orthogonal-ish to G_comp);
    with a modest clip the result must stay sign-correlated with G_comp (unlike
    signed_ema which REPLACES the sign). We assert the corrected gradient's
    cosine with G_comp is strongly positive — direction-preserving.
    """
    sf = SpectralFilter(correction_mode="ef_powersgd", ef_decay=0.0, ef_clip=0.5)
    torch.manual_seed(1)
    g_comp = torch.randn(16, 16)
    _warm_anchor(sf, "layer.o_proj.weight", torch.randn(16, 16))
    g_corr = sf.ef_powersgd_matrix("layer.o_proj.weight", g_comp)
    cos = torch.nn.functional.cosine_similarity(g_corr.flatten().unsqueeze(0), g_comp.flatten().unsqueeze(0)).item()
    # With ef_clip=0.5 the residual is capped at half ||G_comp|| so the corrected
    # direction must remain dominated by G_comp (strongly positive cosine).
    assert cos > 0.7, f"ef_powersgd flipped/destroyed the direction (cos={cos:.3f})"


def test_ef_powersgd_never_calls_sign():
    """Static guarantee: ef_powersgd_matrix contains no torch.sign() call."""
    import inspect

    src = inspect.getsource(SpectralFilter.ef_powersgd_matrix)
    assert "sign(" not in src and ".sign" not in src, (
        "ef_powersgd must be direction-preserving (NO sign term) — the signed_ema "
        "signed_ema failure mode must be structurally excluded"
    )


# --------------------------------------------------------------------------- #
# EF residual is shape-aware, clipped, detached  (hard gate)
# --------------------------------------------------------------------------- #
def test_ef_powersgd_resets_residual_on_shape_mismatch():
    """A shape change for the SAME name resets the accumulated residual (no carry)."""
    sf = SpectralFilter(correction_mode="ef_powersgd", ef_decay=0.9, ef_clip=2.0)
    name = "layer.gate_proj.weight"
    _warm_anchor(sf, name, torch.randn(8, 8))
    sf.ef_powersgd_matrix(name, torch.randn(8, 8))  # builds e_t at (8,8)
    assert tuple(sf._ef_residual[name].shape) == (8, 8)
    # Now feed a differently-shaped grad under the SAME name (after re-warming M
    # at the new shape so the cold-M guard does not short-circuit first).
    sf._anchor.pop(name, None)  # drop old-shape M
    _warm_anchor(sf, name, torch.randn(4, 4))
    before = sf.residual_reset_on_shape_mismatch
    sf.ef_powersgd_matrix(name, torch.randn(4, 4))
    assert sf.residual_reset_on_shape_mismatch == before + 1
    assert tuple(sf._ef_residual[name].shape) == (4, 4)


def test_ef_powersgd_residual_is_norm_clipped():
    """||e_t|| must be bounded by ef_clip * ||G_comp|| ⇒ ||G_corr - G_comp|| <= cap."""
    sf = SpectralFilter(correction_mode="ef_powersgd", ef_decay=0.0, ef_clip=0.3)
    torch.manual_seed(2)
    g_comp = torch.randn(12, 10)
    # A LARGE anchor so the raw residual would blow past the cap if unclipped.
    _warm_anchor(sf, "layer.up_proj.weight", 100.0 * torch.randn(12, 10))
    g_corr = sf.ef_powersgd_matrix("layer.up_proj.weight", g_comp)
    resid_norm = torch.linalg.norm(g_corr.float() - g_comp.float()).item()
    cap = 0.3 * torch.linalg.norm(g_comp.float()).item()
    assert resid_norm <= cap + 1e-4, f"residual {resid_norm:.4f} exceeded cap {cap:.4f}"


def test_ef_powersgd_residual_is_detached():
    """The stored residual carries no autograd graph (dump-safe, no leak)."""
    sf = SpectralFilter(correction_mode="ef_powersgd", ef_decay=0.5, ef_clip=1.0)
    name = "layer.down_proj.weight"
    _warm_anchor(sf, name, torch.randn(6, 6))
    g_comp = torch.randn(6, 6, requires_grad=True)
    sf.ef_powersgd_matrix(name, g_comp)
    assert not sf._ef_residual[name].requires_grad


# --------------------------------------------------------------------------- #
# ef_powersgd through the FSDP-agnostic correction loop
# --------------------------------------------------------------------------- #
def test_ef_powersgd_via_apply_loop_limiting_case():
    """The full correction loop with ef_decay=ef_clip=0 leaves every grad unchanged."""

    class _P:
        def __init__(self, g):
            self.grad = g

    sf = SpectralFilter(correction_mode="ef_powersgd", ef_decay=0.0, ef_clip=0.0)
    state = _FakeState()
    torch.manual_seed(3)
    params = [
        ("model.layers.0.self_attn.q_proj.weight", _P(torch.randn(8, 8))),
        ("model.layers.0.mlp.up_proj.weight", _P(torch.randn(8, 8))),
    ]
    originals = {n: p.grad.clone() for n, p in params}
    # Warm M for both so the cold-M guard is not what produces the identity.
    for n, _ in params:
        _warm_anchor(sf, n, torch.randn(8, 8))

    def full_grad_of(g):
        return g, {"grad_container_type": type(g).__name__}

    def writeback(g, gp):
        g.copy_(gp)

    n = apply_spectral_correction_to_params(
        params,
        spectral=sf,
        target_substrs=("q_proj", "up_proj"),
        max_targets=-1,
        state=state,
        discovery_meta={},
        full_grad_of=full_grad_of,
        writeback=writeback,
    )
    assert n == 2
    for name, p in params:
        assert torch.equal(p.grad, originals[name]), "limiting-case ef_powersgd changed a grad"
    assert state.residual_reset_on_shape_mismatch == 0


# --------------------------------------------------------------------------- #
# config validation + off-path parity
# --------------------------------------------------------------------------- #
def test_config_accepts_ef_powersgd_and_validates_ranges():
    # BaseConfig is frozen, so construct nested sub-configs via their ctors and
    # let CommEffConfig validate in __post_init__ at construction time.
    from verl.workers.config.comm_eff import (
        CommEffConfig,
        CommEffPowerSGDConfig,
        CommEffSpectralConfig,
    )

    # ef_powersgd is an accepted correction_mode (constructs cleanly).
    CommEffConfig(
        enabled=True,
        spectral=CommEffSpectralConfig(enabled=True, correction_mode="ef_powersgd", ef_decay=0.5, ef_clip=1.0),
    )

    # Bad ef_decay (>= 1) is loud.
    with pytest.raises(ValueError):
        CommEffConfig(enabled=True, spectral=CommEffSpectralConfig(ef_decay=1.0))

    # Bad ef_clip (< 0) is loud.
    with pytest.raises(ValueError):
        CommEffConfig(enabled=True, spectral=CommEffSpectralConfig(ef_clip=-0.1))

    # Bad q_basis is loud.
    with pytest.raises(ValueError):
        CommEffConfig(enabled=True, powersgd=CommEffPowerSGDConfig(q_basis="gradient"))

    # Bad correction_mode still loud.
    with pytest.raises(ValueError):
        CommEffConfig(enabled=True, spectral=CommEffSpectralConfig(correction_mode="ef_powergsd"))


def test_capture_config_defaults_off():
    """The capture sub-config defaults OFF (off-path parity — no side effect)."""
    from verl.workers.config.comm_eff import CommEffConfig

    c = CommEffConfig()
    assert c.capture.enabled is False
    assert c.capture.capture_g_dense is False
    assert c.capture.capture_fresh_anchor is False
    # Default-disabled CommEffConfig validates with no capture side effect.
    c.__post_init__()


# --------------------------------------------------------------------------- #
# capture writer: keyed dumps, detached, tick cap, stratification
# --------------------------------------------------------------------------- #
def test_capture_writer_dumps_keyed_fp32_and_caps_ticks():
    from verl.workers.comm_eff.capture import CaptureWriter

    with tempfile.TemporaryDirectory() as d:
        w = CaptureWriter(capture_dir=d, max_ticks=2, stratified_targets=0, rank=0)
        t = torch.randn(4, 4)
        assert w.dump(role="G_comp", target_name="m.q_proj.weight", tensor=t, global_step=1, optimizer_tick=1)
        # Reload + verify it is the real fp32 tensor.
        import json

        rows = [json.loads(line) for line in open(w.manifest_path)]
        assert len(rows) == 1
        assert rows[0]["role"] == "G_comp"
        assert rows[0]["shape"] == [4, 4]
        assert abs(rows[0]["norm"] - float(torch.linalg.norm(t).item())) < 1e-4
        loaded = torch.load(os.path.join(w.root, rows[0]["path"]))
        assert torch.allclose(loaded.float(), t, atol=1e-5)

        # Tick cap: ticks 1 + 2 open; tick 3 is refused.
        assert w.should_capture_tick(1, 1)  # already open
        assert w.dump(role="A", target_name="m.k_proj.weight", tensor=t, global_step=2, optimizer_tick=2)
        assert not w.should_capture_tick(3, 3)
        assert not w.dump(role="A", target_name="m.k_proj.weight", tensor=t, global_step=3, optimizer_tick=3)


def test_capture_writer_does_not_touch_input_grad():
    """The writer detaches/clones — the source tensor's graph is untouched."""
    from verl.workers.comm_eff.capture import CaptureWriter

    with tempfile.TemporaryDirectory() as d:
        w = CaptureWriter(capture_dir=d, rank=0)
        t = torch.randn(3, 3, requires_grad=True)
        w.dump(role="G_corr", target_name="x", tensor=t, global_step=1, optimizer_tick=1)
        # Source still requires grad; the dumped copy must not alias it.
        assert t.requires_grad


def test_capture_writer_stratified_subset():
    from verl.workers.comm_eff.capture import CaptureWriter

    with tempfile.TemporaryDirectory() as d:
        w = CaptureWriter(capture_dir=d, max_ticks=0, stratified_targets=1, rank=0)
        t = torch.randn(2, 2)
        # Two q_proj targets at the same (tick, role): only the FIRST is admitted.
        assert w.dump(role="A", target_name="l0.q_proj.weight", tensor=t, global_step=1, optimizer_tick=1)
        assert not w.dump(role="A", target_name="l1.q_proj.weight", tensor=t, global_step=1, optimizer_tick=1)
        # A different matrix-type (k_proj) is a separate bucket ⇒ admitted.
        assert w.dump(role="A", target_name="l0.k_proj.weight", tensor=t, global_step=1, optimizer_tick=1)


# --------------------------------------------------------------------------- #
# comm-eff hotfix: unified capture tick (all roles share ONE (gs, tick) key)
# --------------------------------------------------------------------------- #
def test_capture_tick_unification():
    """current_optimizer_tick()/capture_tick() give the per-train_batch tick.

    Regression guard for the geometry-probe bug where the powersgd activation hook keyed
    dumps by fwd_generation (hundreds/step) and starved the max_ticks budget so
    NO gradient role (G_comp/G_corr/G_dense/M) ever landed. The fix: every role
    reads ONE stamped tick via capture_tick().
    """
    from verl.workers.comm_eff.state import maybe_build_comm_eff_state
    from verl.workers.config.comm_eff import CommEffConfig, CommEffSpectralConfig

    st = maybe_build_comm_eff_state(CommEffConfig(enabled=True, spectral=CommEffSpectralConfig(enabled=True)))
    # During the fast forward of train_batch N, spectral_step holds N-1, so the
    # tick this batch's tensors belong to is N == spectral_step + 1.
    st.spectral_step = 0
    st.anchor_step = 0
    assert st.current_optimizer_tick() == 1
    # Unstamped capture_tick() falls back to current_optimizer_tick().
    assert st.capture_tick() == 1
    # The forward stamps _capture_tick once; every role then reads the SAME value
    # even after the grad-correction hook advances spectral_step (the merger runs
    # post-advance but reads the stamp, not the live counter).
    st._capture_tick = st.current_optimizer_tick()  # stamped at forward: tick=1
    st.spectral_step += 1  # grad-correction hook advances it (now 1)
    assert st.capture_tick() == 1, "merger must read the STAMPED tick, not the advanced counter"
    # Next batch: re-stamp.
    st._capture_tick = st.current_optimizer_tick()  # spectral_step=1 => tick=2
    assert st.capture_tick() == 2

    # comm-eff tick regression: a NO-MERGER arm (spectral disabled) never
    # advances spectral_step, so the tick MUST track anchor_step instead (else
    # every dump collapses to tick=1 and overwrites). anchor_step is incremented
    # at the TOP of the anchor refresh, so it already equals N during the batch.
    st2 = maybe_build_comm_eff_state(CommEffConfig(enabled=True, spectral=CommEffSpectralConfig(enabled=False)))
    st2.spectral_step = 0  # stays 0 all run (grad-correction early-returns: spectral None)
    st2.anchor_step = 1  # anchor refresh advanced it to 1 at the top of batch 1
    assert st2.current_optimizer_tick() == 1
    st2.anchor_step = 2  # batch 2
    assert st2.current_optimizer_tick() == 2, "no-merger arm must key the tick off anchor_step"


def test_capture_writer_canonicalizes_target_name():
    """G_comp (live-FSDP infixed name) and G_dense (clone non-infixed name) for the
    SAME logical matrix must key on the SAME canonical target_name so the audit
    pairs them. Regression guard for the cos(G_dense,G_comp) n=0 bug.
    """
    import json as _json

    from verl.workers.comm_eff.capture import CaptureWriter

    with tempfile.TemporaryDirectory() as d:
        w = CaptureWriter(capture_dir=d, rank=0)
        t = torch.randn(4, 4)
        # live-FSDP infixed name (as the merger dumps G_comp)
        w.dump(
            role="G_comp",
            target_name="model.layers.0._fsdp_wrapped_module.mlp.up_proj.weight",
            tensor=t,
            global_step=1,
            optimizer_tick=1,
        )
        # clone non-infixed name (as the G_dense probe dumps)
        w.dump(
            role="G_dense", target_name="model.layers.0.mlp.up_proj.weight", tensor=t, global_step=1, optimizer_tick=1
        )
        rows = [_json.loads(line) for line in open(w.manifest_path)]
        names = {r["role"]: r["target_name"] for r in rows}
        assert names["G_comp"] == names["G_dense"] == "model.layers.0.mlp.up_proj.weight", names
        # raw preserved
        raws = {r["role"]: r["target_name_raw"] for r in rows}
        assert "_fsdp_wrapped_module" in raws["G_comp"] and "_fsdp_wrapped_module" not in raws["G_dense"]


def test_capture_min_tick_skips_cold_ticks_before_budget():
    """comm-eff min_tick regression: min_tick must skip cold-Q ticks BEFORE the
    max_ticks budget is consumed, so the post-warm anchor-fire ticks (10/15) land.
    Previously COMM_EFF_CAPTURE_MIN_TICK was silently dropped (not wired) and the
    8-slot budget filled with cold ticks 1-8, losing the H1 inputs.
    """
    import tempfile as _tf

    import torch as _torch

    from verl.workers.comm_eff.capture import CaptureWriter

    with _tf.TemporaryDirectory() as d:
        w = CaptureWriter(capture_dir=d, max_ticks=8, min_tick=9, rank=0)
        t = _torch.randn(2, 2)
        # cold ticks 1..8 must NOT open (and must NOT consume budget slots)
        for tk in range(1, 9):
            assert not w.should_capture_tick(tk, tk), f"cold tick {tk} should be skipped"
            assert not w.dump(
                role="G_comp", target_name="m.q_proj.weight", tensor=t, global_step=tk, optimizer_tick=tk
            ), f"cold tick {tk} dumped"
        # post-warm ticks 9..16 must open (budget was untouched by the cold ticks)
        opened = 0
        for tk in range(9, 17):
            if w.dump(role="G_comp", target_name="m.q_proj.weight", tensor=t, global_step=tk, optimizer_tick=tk):
                opened += 1
        assert opened == 8, f"expected ticks 9-16 to open (8), got {opened}"
        # specifically the post-warm anchor-fire ticks 10 and 15 are present
        import json as _json

        ticks = {(_json.loads(line)["optimizer_tick"]) for line in open(w.manifest_path)}
        assert 10 in ticks and 15 in ticks, f"post-warm fires 10/15 missing: {sorted(ticks)}"
        assert 8 not in ticks and 1 not in ticks, f"cold ticks leaked: {sorted(ticks)}"
