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

"""Configuration for the communication-efficient pipeline-adaptation method.

This module defines the ``comm_eff`` config group: the two-circuit compression
method (pipeline activation masking + asynchronous unmasked anchor circuit +
spectral correction of masked gradients). The config defaults to **disabled**,
in which case the integration hooks in the actor train path are strict no-ops
(see ``verl.workers.comm_eff.state`` and the guards in ``engine_workers``,
``engine/base`` and ``engine/fsdp/transformer_impl``).

The nested sub-configs validate the Hydra schema up front, but carry no
behavior while ``enabled=false``.
"""

from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = [
    "CommEffMaskConfig",
    "CommEffAnchorConfig",
    "CommEffSpectralConfig",
    "CommEffPowerSGDConfig",
    "CommEffCaptureConfig",
    "CommEffConfig",
]

# The Q-basis families PowerSGD may use (EXP-26 Step C). ``act`` is the existing
# activation-energy basis (block power iteration on the boundary activations);
# the others are RLVR-native candidates that bias Q toward GRPO UPDATE energy.
# All share the SAME fixed rank ``r`` (the byte budget is invariant); only the
# CONTENT of the sketch fed to ``orth(V)`` changes. ``act`` is the byte-identical
# default (EXP-25 substrate) so a run that does not opt into a family is unchanged.
Q_BASIS_FAMILIES = ("act", "grad", "adv", "tail", "hybrid", "ticket")

# The compression codecs ``comm_eff.compression_type`` may select. Exactly one
# codec is active per run (mutually exclusive). ``dense`` is the byte-identical
# off-path (equivalent to ``comm_eff.enabled=false`` for the activation path);
# ``prf_mask`` is the per-(token, dim) PRF Bernoulli mask; ``powersgd`` is the
# shared frozen-basis PowerSGD-style projector
# ``M_hat = (M @ Q) @ Qᵀ`` at the same logical PP byte budget.
COMPRESSION_TYPES = ("dense", "prf_mask", "powersgd")


@dataclass
class CommEffMaskConfig(BaseConfig):
    """Per-(token, dimension) pipeline-boundary activation masking (inert while disabled).

    A deterministic PRF Bernoulli mask applied in-graph (``h_tilde = h * mask``)
    to the boundary decoder blocks, only on the actor-train forward (and the
    old-logprob recompute when ``mask_recompute``). Each token independently keeps
    ``round((1-p)*H)`` dims; the mask is keyed on each token's stable
    ``(sample_id, position_id)`` so it is packing-invariant across the
    differently-packed forwards. See ``verl.workers.comm_eff.activation_mask``.

    Args:
        enabled (bool): Whether masking runs (still gated by ``comm_eff.enabled``).
        p (float): Masked (zeroed) fraction in ``[0, 1]``; ``comm_eff/mask_ratio``
            tracks it. ``0.0`` means no masking.
        seed (int): Base seed folded into the mask PRF key.
        pp_size (int): Logical pipeline-shard count; the last block of every shard
            except the final one is masked (``L`` from ``model.config``).
            ``L=16, pp_size=8`` -> ``[1, 3, 5, 7, 9, 11, 13]``.
        mask_recompute (bool): When ``True`` the mask also fires on the
            old-logprob recompute, so both gradient-feeding forwards are masked.
            The stable key gives a token the identical mask in both, keeping the
            PPO importance ratio ≈1 at the first inner step. ``False`` (default)
            masks only the train forward.
        rescale (bool): ``False`` (default) writes the raw product ``h * mask``;
            ``True`` applies inverted-dropout ``h * mask / (1 - p)`` so
            ``E[h_tilde] = h`` (requires ``p < 1``). The theory wants the
            ``1/(1-p)`` rescale (unbiased), but the supervised reference omits it
            (the "dropped activations are zeros" reading, and ``1/(1-p)`` is
            bf16-risky at high ``p``); default matches the reference.
    """

    enabled: bool = True
    p: float = 0.95
    seed: int = 0
    pp_size: int = 8
    mask_recompute: bool = False
    rescale: bool = False
    rescale_mode: str = "auto"


@dataclass
class CommEffAnchorConfig(BaseConfig):
    """Asynchronous unmasked-anchor-circuit sub-config (inert while disabled).

    Every ``cadence`` trainer steps, the anchor circuit runs one unmasked
    GRPO-actor-loss forward/backward from a ``delay_K``-stale weight snapshot to
    produce a clean per-target gradient ``G_anchor``. ``G_anchor`` is read RAW
    into the spectral filter's EMA ``M_anchor``. The anchor takes NO optimizer
    step and generates NO rollouts / recomputes NO rewards.

    Args:
        enabled (bool): Whether the anchor circuit runs. Gated by the parent
            ``comm_eff.enabled`` regardless of this value. ``false`` (default) is
            a strict no-op — opt-in only.
        cadence (int): Anchor-refresh cadence in trainer steps. The anchor fires
            when ``(step % cadence) == 0``. Must be ``>= 1``.
        delay_K (int): Staleness of the weight snapshot the anchor forwards
            from, in trainer steps. ``0`` = current weights; ``1`` = the prior
            step's weights. Must be ``>= 0``.
        owns_q (bool): EXP-25 (R2) structural inversion. When ``true`` the ANCHOR
            owns the PowerSGD projection basis ``Q``: (a) the fast net's
            ``maybe_update_basis`` + sketch accumulation are gated OFF (the fast
            net is a pure read-only consumer of ``Q``), and (b) the anchor
            computes ``Q ← orth(V)`` from its OWN slow-net stale-weight forward
            ACTIVATIONS (the same ``V += Aᵀ(AQ)`` block-power-iteration the fast
            path used, relocated fast→slow) and ``dist.broadcast``s both ``Q`` and
            the gradient-EMA ``M`` to every DP rank each refresh. ``false``
            (default) keeps the EXP-20 fast-owns-Q behaviour byte-identical
            (Prime Directive). Gated by ``anchor.enabled`` + the powersgd codec.
        replay_paired_batch (bool): EXP-29 on-policy replay. When ``true`` the
            anchor pairs its ``delay_K``-stale weights with the BATCH those same
            weights generated: every ``train_batch`` tick deep-clones the batch
            into a ring (CPU-resident), ONE generator-weight snapshot is taken
            per global step (at its first tick — exactly the weights vLLM
            generated that step's rollouts from), and at fire time the anchor
            replays ``(batch[t-delay_K], gen_snapshot)`` instead of (current
            batch, t-delay_K weights). ``false`` (default) keeps the legacy
            behaviour byte-identical: the anchor forwards the CURRENT tick's
            batch at the stale weights, and no replay ring is constructed.
        snapshot_device (str): Where the anchor weight snapshots live between
            ticks — ``"gpu"`` (default, faithful: detached clones stay on each
            param's device, today's exact behaviour) or ``"cpu"`` (memory-lean:
            the ``delay_K+1`` full bf16 snapshots move off HBM; the clone load
            casts back via ``.to(p.device, p.dtype)``, a byte-preserving round
            trip — numerics-neutral). Applies to BOTH the legacy staleness
            queue and the EXP-29 replay ring. Validated against {gpu, cpu}.
    """

    enabled: bool = False
    cadence: int = 20
    delay_K: int = 20
    owns_q: bool = False
    replay_paired_batch: bool = False
    snapshot_device: str = "gpu"


@dataclass
class CommEffSpectralConfig(BaseConfig):
    """Anchor-guided gradient-correction sub-config (inert while disabled).

    Applies an anchor combiner to selected 2D decoder gradients after the actor
    backward and before ``optimizer.step()``, using the anchor-gradient EMA
    ``M_anchor`` (no SVD / no basis). See
    ``verl.workers.comm_eff.spectral_filter.SpectralFilter``.

    The live correction is the signed-EMA merger (per targeted 2D matrix
    ``G_noisy`` with anchor-EMA ``M_anchor``)::

        M_anchor = beta_anc * M_anchor + (1 - beta_anc) * G_anchor    # EMA
        G_corr   = alpha * G_noisy + (1 - alpha) * |G_noisy| * sign(M_anchor)

    Magnitude from ``G_noisy``, sign from ``M_anchor``. ``inject`` and ``blend``
    are alternate anchor combiners. A cold-M guard returns ``G_noisy`` unchanged
    whenever ``M_anchor`` is unwarmed, so the masked gradient is never silently
    zeroed.

    Args:
        enabled (bool): Whether anchor-guided correction of masked gradients
            runs. Gated by the parent ``comm_eff.enabled`` regardless of this
            value. ``false`` (default) ⇒ the grad-correction hook is a strict
            no-op and the actor path is identical to dense GRPO.
        beta_anc (float): EMA decay for the anchor-gradient running matrix
            ``M_anchor``. Default ``0.95``.
        target_substr (list[str]): Substrings used to SELECT which named 2D
            parameters receive correction. A parameter is targeted iff its name
            contains one of these substrings AND its logical shape is 2D.
            Defaults select the decoder attention/MLP projection matrices and
            skip norms, biases, embeddings and the lm head.
        max_targets (int): Cap on the number of target matrices corrected per
            step (keeps the discovery smoke cheap). ``-1`` ⇒ no cap (the EXP-25
            default — full coverage of ALL 196 = 28 layers × 7 projection
            matrices, the set the signed_ema merger corrects; ``max_targets``
            caps BOTH the anchor extraction AND the merger, so a residual cap
            silently drops merger targets). Set ``>= 0`` only as a diagnostic
            throttle, never in production.
        ema_device (str): Where the anchor-gradient EMA ``M_anchor`` is stored
            between refreshes — ``"gpu"`` (default, faithful: kept in HBM) or
            ``"cpu"`` (memory-lean: offloaded to pinned CPU, moved to GPU only
            inside the refresh/correct call and moved back). ``M_anchor`` is
            touched only at refresh, so CPU offload costs one H2D/D2H per refresh,
            not per mini-batch. Validated against {gpu, cpu}.
        correction_mode (str): The anchor combiner the fast-path grad uses —
            ``"signed_ema"`` (default, EXP-25/R3:
            ``alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M_anchor)``), ``"inject"``
            (additive scale-matched anchor complement) or ``"blend"``
            (``(1-eta)*G_mask + eta*scale*M_anchor``). Validated against
            {inject, blend, signed_ema}.
        inject_gamma (float): Injection strength for
            ``correction_mode="inject"``; unused otherwise. Must be ``>= 0``.
        blend_eta (float): Convex-blend weight for ``correction_mode="blend"``;
            validated to ``[0, 1]``. Unused otherwise.
        signed_ema_alpha (float): The signed_ema merger weight ``alpha`` in
            ``G_corr = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M_anchor)``.
            ``alpha=0`` is the SL-validated pure sign-merger; ``alpha=1`` returns
            ``G_noisy`` unchanged. THE swept axis. Validated to ``[0, 1]``.
            Unused unless ``correction_mode=signed_ema``.
        cadence (int): Correction cadence in optimizer steps.
            The grad-correction hook (``_maybe_comm_eff_grad_correction``) fires
            only when ``(spectral_step % cadence) == 0`` on the monotonic
            per-optimizer-step counter, MIRRORING the anchor cadence
            (``anchor_should_fire``) and the clean-step cadence
            (``CommEffState.is_clean_step``). ``1`` (default) fires EVERY step =
            the default behavior. Set ``> 1`` (e.g. ``2``) to align
            correction with a matching ``anchor.cadence`` so the
            correction always uses a freshly-refreshed anchor EMA instead of a
            stale one on the in-between steps. Must be ``>= 1``.
    """

    enabled: bool = False
    beta_anc: float = 0.95
    cadence: int = 1
    target_substr: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    # EXP-25: full coverage by default (-1 = no cap). The signed_ema merger must
    # correct ALL 196 weight matrices (the activation compression corrupts weight
    # gradients throughout the network); a residual cap re-creates the EXP-23
    # coverage bug. Caps BOTH the anchor extraction and the merger (one knob).
    max_targets: int = -1
    # EMA storage default: keep tensors on GPU.
    ema_device: str = "gpu"
    # Correction mode (anchor combiner). "signed_ema" (EXP-25/R3) uses the SL
    # signed-EMA merger G_corr=alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M).
    # "inject" adds a scale-matched anchor-EMA complement. "blend" uses
    # G_corr=(1-eta)*G_mask + eta*scale*M_anchor. "ef_powersgd" (EXP-26 Step B) is
    # the direction-PRESERVING error-feedback merger (NO sign term).
    correction_mode: str = "signed_ema"
    # Injection strength for correction_mode="inject"; unused otherwise.
    inject_gamma: float = 1.0
    # Convex-blend weight for correction_mode="blend"; validated to [0, 1].
    blend_eta: float = 0.5
    # EXP-25 (R3): the signed_ema merger weight alpha in
    # G_corr = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M_anchor). alpha=0 is the
    # SFT-validated pure sign-merger; alpha=1 returns G_noisy unchanged. THE swept
    # axis (id-2). Validated to [0, 1]. Unused unless correction_mode=signed_ema.
    signed_ema_alpha: float = 0.0
    # ---- EXP-26 Step B: error-feedback PowerSGD merger (direction-preserving) ----
    # ``correction_mode="ef_powersgd"`` re-injects the PowerSGD reconstruction
    # residual ``e_t = G_dense_proxy - G_comp`` (the off-subspace component the
    # rank-r projection dropped), accumulated with decay, ADDED to G_comp with NO
    # sign term — so the corrected gradient KEEPS G_comp's direction/sign and only
    # restores the dropped magnitude along the off-principal directions. Unlike
    # signed_ema (which REPLACES the sign with sign(M)), this is sign-preserving.
    #
    #   e_t   <- ef_decay * e_{t-1} + (M_anchor - P_Q(M_anchor))         # residual
    #   e_t   <- clip(e_t, ef_clip * ||G_comp||)                         # norm cap
    #   G_corr = G_comp + e_t                                            # NO sign
    #
    # The residual proxy uses the anchor EMA's OFF-subspace component (M_anchor
    # minus its projection onto the span of G_comp), which is exactly the
    # low-rank-compression bias the audit (Step A) measures. The merger reduces to
    # plain PowerSGD (G_corr == G_comp) at the LIMITING setting ef_decay=0 AND
    # ef_clip=0 (Correctness-invariant "EF residual limiting-case identity").
    ef_decay: float = 0.0
    # ef_clip: the residual norm cap as a FRACTION of ||G_comp|| (shape-aware,
    # per-matrix). 0.0 ⇒ the residual is fully zeroed ⇒ G_corr == G_comp (the
    # plain-PowerSGD limiting case). A typical live value is ~1.0 (the re-injected
    # residual may not exceed the compressed gradient's own norm). Must be >= 0.
    ef_clip: float = 0.0


@dataclass
class CommEffPowerSGDConfig(BaseConfig):
    """PowerSGD-style pipeline-boundary activation-compression sub-config.

    The codec replaces each boundary block's hidden-state output ``M`` (shape
    ``(N, H)`` — ``N`` packed tokens × ``H`` hidden dims) with its rank-``r``
    projection onto a **shared, frozen, per-layer orthonormal basis** ``Q``
    (shape ``(H, r)``)::

        M_hat = (M @ Q) @ Q.T            # forward; Q detached, M in-graph (NO STE)

    so the pipeline boundary transmits only the ``N·r`` projected coordinates
    ``Y = M @ Q`` (plus the shared, communication-free ``Q``) instead of ``N·H``
    — the **identical logical PP byte budget as the PRF mask at ``p = 1 − r/H``**
    (``r=102 ≡ p=0.95`` at ``H=2048``). Because ``Q`` is detached and ``M`` stays
    in-graph, the backward is the exact self-adjoint projector
    ``dL/dM = (dL/dM_hat) · Q Qᵀ`` with no straight-through estimator.

    The basis is bootstrapped with **zero communication** from a deterministic
    per-layer seed ``seed_L = (base_seed·1_000_003 + layer_idx·7919) & 0x7FFFFFFF``
    in fp32, identical on every rank, and refined by block power iteration: on
    compressed *train* forwards we
    accumulate, OFF the autograd graph, ``V += Mᵀ (M Q)`` (one sketch per
    boundary forward), then once at end-of-actor-update (when not a clean step
    and ``global_step % update_cadence == 0``) set ``Q ← orth(V)`` in fp32 and
    clear ``V``. ``Q`` is **frozen for the entire global step** — the
    old-logprob recompute and the actor-train forward both see ``Q_t``; the
    update to ``Q_{t+1}`` happens only after the gradient-bearing actor work, so
    the GRPO importance ratio starts near 1 with no weight change.

    Inert unless ``comm_eff.enabled=true`` AND ``comm_eff.compression_type ==
    "powersgd"``. Every key is registered regardless of ``compression_type`` so a
    ``prf_mask`` run that passes ``comm_eff.powersgd.rank=...`` still parses.

    Args:
        enabled (bool): Sub-switch; gated by the parent ``comm_eff.enabled`` AND
            by ``compression_type == "powersgd"``. ``True`` (default) so simply
            selecting ``compression_type=powersgd`` activates it; the parent
            gates remain the real master switches.
        rank (int): Retained projection rank ``r``. ``r=102`` matches the PRF
            mask at ``p=0.95`` (``q·H = 0.05·2048 = 102.4``). Must be ``>= 1``.
            ``r == H`` is the lossless limiting case ``M_hat = M``.
        seed (int): Base seed for the deterministic per-layer basis bootstrap
            folded with ``layer_idx``. Identical on every rank ⇒
            zero-communication codebook init.
        pp_size (int): Logical pipeline-shard count; the same boundary-block
            selection as the PRF mask (last block of every shard except the
            final). ``L=16, pp_size=8 -> [1,3,5,7,9,11,13]``. Must be ``>= 1``.
        update_cadence (int): Block-power-iteration basis-update cadence in
            optimizer steps. ``1`` (default) refreshes ``Q`` every (non-clean)
            step. ``> 1`` refreshes only when ``global_step % update_cadence ==
            0`` (a quieter, more stable basis). Must be ``>= 1``.
        warm_start (bool): ``True`` (default) carries ``Q`` across steps (warm
            block power iteration — the standard PowerSGD warm start). ``False``
            re-bootstraps ``Q`` from the per-layer seed every update (cold,
            diagnostic only).
        compress_recompute (bool): When ``True`` (default) the projector also
            fires on the old-logprob recompute, so BOTH gradient-feeding forwards
            see the same frozen ``Q_t`` ⇒ ``ρ ≈ 1`` (the analogue of the mask's
            ``mask_recompute``). ``False`` projects only the actor-train forward
            (old-logprob runs dense), which dense-anchors the importance ratio's
            denominator.
        sync_basis (bool): ``True`` (default — REQUIRED for a correct shared
            codebook under DP). Each DP rank builds its sketch ``V`` from its OWN
            data shard (the dispatch scatters a different shard per rank), so a
            per-rank ``orth(V)`` would DIVERGE the per-boundary ``Q`` across ranks
            after the first update. ``True`` all-reduces the raw sketches over the
            DP group before ``orth`` so every rank orthonormalizes the SAME pooled
            ``V_global = Σ_ranks V`` → a bit-identical consensus ``Q`` on every
            rank, differing only per boundary. The collective is made
            deadlock-safe by iterating the fixed
            ``boundary_indices`` on every rank. ``False`` (diagnostic only) keeps
            each rank's basis local — only correct if every rank sees identical
            data, which DP does not.
        qr_dtype (str): Dtype for the orthonormalization (``orth``/QR) and the
            stored basis math — ``"fp32"`` (default, REQUIRED for correctness:
            bf16-QR loses orthogonality, drifts ``QᵀQ`` from ``I``, and is a
            frequent NaN / ``q_cond`` source) or ``"bf16"`` (diagnostic only,
            expected to degrade). The projection itself runs in the
            activation dtype regardless; only the QR/orth + ``V`` accumulation are
            in ``qr_dtype``.
        reortho_eps (float): Floor added under the QR when forming the basis, and
            the singular-value floor used in the ``q_cond`` diagnostic, to keep a
            rank-deficient sketch from producing a non-finite condition number.
            Must be ``> 0``.
    """

    enabled: bool = True
    rank: int = 102
    seed: int = 0
    pp_size: int = 8
    update_cadence: int = 1
    warm_start: bool = True
    compress_recompute: bool = True
    sync_basis: bool = True
    qr_dtype: str = "fp32"
    reortho_eps: float = 1e-6
    # EXP-26 Step C: the Q-basis FAMILY (content of the sketch ``orth(V)`` consumes
    # at FIXED rank). "act" (default) = the EXP-25 activation-energy basis (V is
    # built from the boundary ACTIVATIONS, V += Mᵀ(MQ)) — byte-identical to the
    # locked substrate so a run that does not opt in is unchanged. The RLVR-native
    # families ("grad"/"adv"/"tail"/"hybrid"/"ticket") bias V toward GRPO UPDATE
    # energy and ONLY run if Step A finds Q_act under-captures off-principal update
    # energy (H2). Validated to Q_BASIS_FAMILIES. The byte budget (rank) is held
    # fixed across families — only WHICH directions Q spans changes.
    #
    # This is the LIVE basis the fast/training path consumes (anchor feeds V from
    # the family's statistic). "act" is byte-identical; a non-"act" value is the
    # C2/Step-B LIVE-family training path (the fast path stays a read-only
    # consumer — the owns_q invariant is untouched, only the anchor's V content
    # changes). The compressor fails LOUD on a family it does not implement.
    q_basis: str = "act"
    # EXP-26 Step C1 PASSIVE screen: the families to PASSIVELY accumulate inside the
    # anchor's stale-weight pass (in no_grad, off the live Q / fast path / optimizer)
    # so ONE short run builds candidate bases for ALL families at once. Each family's
    # candidate Q_f is orthonormalized + dumped at the anchor cadence; the judge
    # metrics (update-capture, off-principal preservation) are computed offline
    # against the SAME captured reference grads. Empty (default) ⇒ no passive
    # accumulation (byte-identical). The live ``q_basis`` above is INDEPENDENT — the
    # screen runs with q_basis="act" LIVE while these families accumulate passively.
    # Validated to a subset of Q_BASIS_FAMILIES (minus "act"-only redundancy is OK —
    # "act" passively re-derives the live basis as the control row in the dump).
    q_basis_passive: list = field(default_factory=list)
    # EXP-26 Step C1: the hybrid family column split at FIXED total rank r. ``Q_h =
    # orth([Q_act[:, :hybrid_act_cols], Q_grad_deflated[:, :hybrid_grad_cols]])``.
    # ``-1`` (default) ⇒ AUTO: ``hybrid_act_cols = ceil(r/2)``,
    # ``hybrid_grad_cols = r − act`` (39 + 38 = 77 at the locked r=77 from
    # STEP_C_SPEC.md). When BOTH are set explicitly (>= 0) they MUST sum to the rank
    # (validated in __post_init__ only when "hybrid" is requested, so a non-hybrid
    # run is unconstrained). The compressor reads the resolved values via
    # ``resolved_hybrid_cols(rank)``.
    hybrid_act_cols: int = -1
    hybrid_grad_cols: int = -1


@dataclass
class CommEffCaptureConfig(BaseConfig):
    """EXP-26 Step A: real-gradient geometry-audit tensor-capture sub-config.

    **OFF by default — a strict no-op so the EXP-25 / plain-PowerSGD path is
    byte-identical** (Correctness invariant "off-path parity"). When
    ``enabled=true`` the comm-eff hooks dump fp32 tensors (``A``, ``Â=(A@Q)Qᵀ``,
    ``Q``, projection stats; ``G_comp`` the merger input; ``G_corr`` post-merger
    pre-Adam; ``M``/``G_anchor``; the parallel uncompressed ``G_dense``; and the
    ``delay_K=0`` fresh-anchor measurement grad) keyed by
    ``(global_step, optimizer_tick, target_name, shape, dtype, norm)`` under
    ``capture_dir``. Every dumped tensor is detached / dump-only — the capture
    path adds NO numerical side effect, only I/O (Correctness invariant
    "measurement-only probes never feed the optimizer").

    The two measurement-only PROBES (``capture_g_dense`` = a second uncompressed
    fast backward; ``capture_fresh_anchor`` = a delay_K=0 fresh-anchor grad) are
    the expensive, highest-integration-risk dumps; they are independently gated so
    the audit can be staged. ``delay_K=0`` appears ONLY here as a removed probe —
    it is forbidden as a TRAINING config (Correctness invariant "mandatory anchor
    staleness").

    Args:
        enabled (bool): Master capture switch. ``false`` (default) ⇒ NO dump hook
            fires, NO probe backward runs, NO tensor is written; the training path
            is byte-identical to the EXP-25 substrate.
        capture_dir (str): Directory the fp32 dumps + the manifest are written to
            (on the box; rsynced to ``runs/EXP-26/captures/``). Empty ⇒
            ``./captures`` relative to cwd.
        max_ticks (int): Cap on the number of optimizer ticks captured (the audit
            needs only ~5-10). ``<= 0`` ⇒ no cap. Bounds disk + rsync volume.
        stratified_targets (int): If ``> 0``, dump only this many targets PER
            layer-type (the √2 disagreement was uniform across the 7 matrix types
            in EXP-25, so a stratified subset is defensible — see ## Notes for
            runner). ``0`` ⇒ dump every target (full 196-matrix coverage).
        capture_g_dense (bool): Run the parallel UNCOMPRESSED fast backward to
            capture ``G_dense`` alongside ``G_comp`` at the SAME step. Detached /
            dump-only — MUST NOT touch the optimizer. ``false`` (default) ⇒ no
            second backward (the highest-OOM-risk probe; gate it on first).
        capture_fresh_anchor (bool): Capture the ``delay_K=0`` fresh-anchor grad as
            a MEASUREMENT probe for the sign-agreement decomposition. Detached /
            dump-only. ``false`` (default).
        fresh_anchor_loss (str): Loss the ``delay_K=0`` fresh-anchor MEASUREMENT
            probe backward uses — ``"clean_pg"`` (default, ratio≡1 ``-A·logπ`` like
            the anchor refresh) or ``"ppo_clip"`` (the SAME PPO ratio/clip loss as
            the fast path, against the batch's ``old_log_probs``). EXP-26 Step-C/B
            should-have: ``ppo_clip`` removes the clean-PG-vs-PPO-clip loss-mismatch
            confound the Step-A audit flagged, giving a clean
            ``cos(G_fresh_ppo, G_corr)`` improvement test. Affects ONLY the dump-only
            fresh-anchor probe (never the optimizer, the EMA, or the K-stale
            ``G_anchor`` that feeds ``M``). Validated to {clean_pg, ppo_clip}.
        dump_dtype (str): Dump precision. ``"fp32"`` (default, REQUIRED for the
            fidelity invariant — the reconstruction_rel_error recomputed from the
            dump must match the logged ~0.024 scalar within 1e-3). ``"bf16"`` is a
            volume-saving diagnostic only.
    """

    enabled: bool = False
    capture_dir: str = ""
    max_ticks: int = 10
    stratified_targets: int = 0
    capture_g_dense: bool = False
    capture_fresh_anchor: bool = False
    # EXP-26 Step C/B should-have: loss for the delay_K=0 fresh-anchor probe.
    # "clean_pg" (default, ratio≡1, matches the anchor refresh) or "ppo_clip" (the
    # fast path's PPO ratio/clip loss vs old_log_probs — removes the loss-mismatch
    # confound). Affects ONLY the dump-only probe.
    fresh_anchor_loss: str = "clean_pg"
    dump_dtype: str = "fp32"
    # EXP-26 disk-volume guard: capture ONLY rank 0 (default True). The audit's
    # gradient roles are DP-reduced and Q/A/Â are sync_basis-consensus — identical
    # across ranks — so rank 0 suffices, and writing all 4 ranks blew the box disk
    # (76 GB for ONE arm => torch.save crashed mid-write on the full disk). Set
    # False only if per-rank shards are genuinely needed.
    rank0_only: bool = True
    # EXP-26: skip capture ticks below this (cold-Q warmup). The anchor warms Q at
    # cadence (tick 5/10 for cadence=5); capturing from tick 1 fills the budget with
    # PRE-warm ticks (recon ~0.97), making H2 (Q activation/update-capture)
    # untrustworthy. Set just above the first anchor refresh so captured ticks are
    # POST-warm. 0 (default) = capture from the start (back-compat).
    min_tick: int = 0


@dataclass
class CommEffConfig(BaseConfig):
    """Top-level config for the communication-efficient compression method.

    The inheritance from BaseConfig provides an omegaconf.DictConfig-like
    interface for a dataclass config and, because it is a structured config,
    OmegaConf rejects unknown ``comm_eff.*`` keys at merge time (typos in the
    mask/anchor/spectral namespaces fail fast instead of being silently
    ignored).

    **Disabled is a strict no-op.** When ``enabled=false`` the integration
    hooks short-circuit before importing any comm_eff machinery: no forward
    hooks are registered, no SVD/EMA buffers are allocated, no extra
    all-reduce is issued, and crucially **no RNG is drawn**. Constructing this
    config (which every actor does, since it defaults disabled) must therefore
    have zero numerical side effects, so a dense GRPO run with the scaffolding
    merged is bit-for-bit / rel-tol-1e-4 identical to one without it.

    Args:
        enabled (bool): Master switch. ``false`` (default) makes every comm_eff
            hook a no-op. Must be set ``true`` explicitly to activate any
            circuit.
        compression_type (str): Codec selector, one of
            ``{dense, prf_mask, powersgd}`` (mutually exclusive per run).
            ``dense`` (default) = no activation compression. ``prf_mask`` = the
            existing PRF mask. ``powersgd`` = the shared frozen-basis projector.
            For back-compat the legacy ``mask.enabled`` path still selects the
            mask when ``compression_type`` is left at its ``dense`` default, so
            older mask configs behave unchanged; an explicit
            ``compression_type=powersgd`` selects the PowerSGD codec.
        mask (CommEffMaskConfig): Pipeline activation-masking sub-config.
        anchor (CommEffAnchorConfig): Asynchronous anchor-circuit sub-config.
        spectral (CommEffSpectralConfig): Spectral-correction sub-config.
        powersgd (CommEffPowerSGDConfig): PowerSGD activation-compression
            sub-config.
        clean_cadence (int): Periodic clean (unmasked) optimizer-step cadence,
            in trainer steps. ``0`` (default) = off. When
            ``> 0``, the trainer step ``s`` runs **entirely unmasked** whenever
            ``(s % clean_cadence) == 0`` — both gradient-feeding forwards (the
            ``compute_log_prob`` old-logprob recompute AND the actor-train
            forward) have masking forced OFF for that step, and the normal
            single ``optimizer.step()`` runs on the true dense gradient so
            AdamW's ``m``/``v`` are periodically refreshed (the design's own
            "naive synchronous fix"). It does NOT add a second optimizer step:
            the clean step REPLACES the masked step at that index. The other
            ``clean_cadence - 1`` of every ``clean_cadence`` steps stay masked
            exactly as today (≈90% of the pipeline-boundary savings retained at
            ``clean_cadence=10``). Gated by the parent ``comm_eff.enabled``; on
            the disabled path the threading short-circuits on the ``None`` state
            so there is no numerical side effect. No anchor/spectral machinery is
            touched — the clean step rides the existing dense optimizer path.
    """

    enabled: bool = False
    # Codec selector. Exactly one boundary codec is active per run:
    #   "dense"    -> no activation compression (the boundary forward is
    #                 byte-identical to dense; equivalent to the activation path
    #                 of enabled=false). A legacy mask-circuit run is selected
    #                 by mask.enabled, NOT by this field, for back-compat.
    #   "prf_mask" -> the per-(token, dim) PRF Bernoulli mask.
    #   "powersgd" -> the shared frozen-basis projector M_hat=(M@Q)@Qᵀ.
    # Registered with the powersgd block regardless of value so a prf_mask run
    # that passes comm_eff.powersgd.rank=... still parses.
    compression_type: str = "dense"
    mask: CommEffMaskConfig = field(default_factory=CommEffMaskConfig)
    anchor: CommEffAnchorConfig = field(default_factory=CommEffAnchorConfig)
    spectral: CommEffSpectralConfig = field(default_factory=CommEffSpectralConfig)
    powersgd: CommEffPowerSGDConfig = field(default_factory=CommEffPowerSGDConfig)
    # EXP-26 Step A diagnostic tensor capture. OFF by default ⇒ no numerical side
    # effect (byte-identical to the EXP-25 substrate).
    capture: CommEffCaptureConfig = field(default_factory=CommEffCaptureConfig)
    clean_cadence: int = 0

    def __post_init__(self):
        """Validate comm_eff configuration parameters.

        Validation only — no allocation, no RNG. When ``enabled=false`` this
        must remain free of numerical side effects.
        """
        if not 0.0 <= self.mask.p <= 1.0:
            raise ValueError(f"comm_eff.mask.p must be in [0, 1]; got {self.mask.p}")
        if self.mask.pp_size < 1:
            raise ValueError(f"comm_eff.mask.pp_size must be >= 1; got {self.mask.pp_size}")
        # mask_recompute is a strict bool. Validation here turns a YAML
        # typo ("False" string) or a numeric override into a loud error instead
        # of a silent truthy/falsy surprise that would mis-route masking.
        if not isinstance(self.mask.mask_recompute, bool):
            raise ValueError(
                f"comm_eff.mask.mask_recompute must be a bool; got {type(self.mask.mask_recompute).__name__} "
                f"({self.mask.mask_recompute!r})"
            )
        # rescale is a strict bool (same rationale as the other mask flags). It
        # also requires p < 1 so the 1/(1-p) factor is finite; p==1 with rescale
        # would divide by zero (and p==1 masks everything anyway).
        if not isinstance(self.mask.rescale, bool):
            raise ValueError(
                f"comm_eff.mask.rescale must be a bool; got {type(self.mask.rescale).__name__} "
                f"({self.mask.rescale!r})"
            )
        if self.mask.rescale and self.mask.p >= 1.0:
            raise ValueError(
                "comm_eff.mask.rescale=true requires comm_eff.mask.p < 1.0 (the 1/(1-p) "
                f"magnitude-preservation factor is undefined at p>=1); got p={self.mask.p}"
            )
        if not 0.0 <= self.spectral.beta_anc <= 1.0:
            raise ValueError(f"comm_eff.spectral.beta_anc must be in [0, 1]; got {self.spectral.beta_anc}")
        # Spectral-correction cadence. 1 = fire every step. A value < 1 is a
        # config error, not a silent disable; mirrors anchor.cadence >= 1.
        if self.spectral.cadence < 1:
            raise ValueError(f"comm_eff.spectral.cadence must be >= 1; got {self.spectral.cadence}")
        # Anchor cadence/staleness.
        if self.anchor.cadence < 1:
            raise ValueError(f"comm_eff.anchor.cadence must be >= 1; got {self.anchor.cadence}")
        if self.anchor.delay_K < 0:
            raise ValueError(f"comm_eff.anchor.delay_K must be >= 0; got {self.anchor.delay_K}")
        # EXP-29: on-policy replay flag is a strict bool (a YAML "False" string or
        # a numeric override must be loud, not a silent truthy surprise that turns
        # the replay ring on/off by accident — same rationale as mask_recompute).
        if not isinstance(self.anchor.replay_paired_batch, bool):
            raise ValueError(
                f"comm_eff.anchor.replay_paired_batch must be a bool; got "
                f"{type(self.anchor.replay_paired_batch).__name__} ({self.anchor.replay_paired_batch!r})"
            )
        # EXP-29: snapshot storage enum (mirrors spectral.ema_device). "gpu" is
        # the byte-identical legacy default; "cpu" moves the delay_K+1 full
        # snapshots off HBM (numerics-neutral byte-preserving round trip).
        if self.anchor.snapshot_device not in ("gpu", "cpu"):
            raise ValueError(
                f"comm_eff.anchor.snapshot_device must be one of (gpu, cpu); "
                f"got {self.anchor.snapshot_device!r}"
            )
        # Storage-layer enum.
        if self.spectral.ema_device not in ("gpu", "cpu"):
            raise ValueError(f"comm_eff.spectral.ema_device must be one of (gpu, cpu); got {self.spectral.ema_device!r}")
        if self.spectral.correction_mode not in ("inject", "blend", "signed_ema", "ef_powersgd"):
            raise ValueError(
                f"comm_eff.spectral.correction_mode must be one of "
                f"(inject, blend, signed_ema, ef_powersgd); "
                f"got {self.spectral.correction_mode!r}"
            )
        if self.spectral.inject_gamma < 0.0:
            raise ValueError(f"comm_eff.spectral.inject_gamma must be >= 0; got {self.spectral.inject_gamma}")
        # Convex-blend weight. [0, 1]: 0 => pure G_mask, 1 =>
        # scale-matched stale true gradient. Unused unless correction_mode=blend.
        if not 0.0 <= self.spectral.blend_eta <= 1.0:
            raise ValueError(f"comm_eff.spectral.blend_eta must be in [0, 1]; got {self.spectral.blend_eta}")
        # EXP-25 (R3) signed_ema merger weight. [0, 1]: 0 => pure |G|*sign(M)
        # sign-merger, 1 => G_noisy unchanged. Unused unless
        # correction_mode=signed_ema.
        if not 0.0 <= self.spectral.signed_ema_alpha <= 1.0:
            raise ValueError(
                f"comm_eff.spectral.signed_ema_alpha must be in [0, 1]; got {self.spectral.signed_ema_alpha}"
            )
        # EXP-26 Step B: error-feedback residual knobs. decay in [0, 1) (an EMA
        # decay; 1.0 would never forget the residual). clip >= 0 (a norm-cap
        # fraction; 0 ⇒ residual fully zeroed ⇒ G_corr==G_comp, the plain-PowerSGD
        # limiting case). Both validated unconditionally so a typo is loud even on
        # a non-ef_powersgd run that forwards them.
        if not 0.0 <= self.spectral.ef_decay < 1.0:
            raise ValueError(
                f"comm_eff.spectral.ef_decay must be in [0, 1); got {self.spectral.ef_decay}"
            )
        if self.spectral.ef_clip < 0.0:
            raise ValueError(
                f"comm_eff.spectral.ef_clip must be >= 0; got {self.spectral.ef_clip}"
            )
        # Periodic clean-step cadence. 0 = off. A negative value is a config
        # error, not a silent disable.
        if self.clean_cadence < 0:
            raise ValueError(f"comm_eff.clean_cadence must be >= 0; got {self.clean_cadence}")
        # Codec selector. Validated to the closed enum so a typo
        # (compression_type=powerSGD / powergsd) is a loud error, not a silent
        # fall-through to dense.
        if self.compression_type not in COMPRESSION_TYPES:
            raise ValueError(
                f"comm_eff.compression_type must be one of {COMPRESSION_TYPES}; "
                f"got {self.compression_type!r}"
            )
        # PowerSGD block. Validated unconditionally (the keys are
        # registered regardless of compression_type) so a prf_mask run that
        # forwards comm_eff.powersgd.* args still fails fast on a bad value.
        if self.powersgd.rank < 1:
            raise ValueError(f"comm_eff.powersgd.rank must be >= 1; got {self.powersgd.rank}")
        if self.powersgd.pp_size < 1:
            raise ValueError(f"comm_eff.powersgd.pp_size must be >= 1; got {self.powersgd.pp_size}")
        if self.powersgd.update_cadence < 1:
            raise ValueError(
                f"comm_eff.powersgd.update_cadence must be >= 1; got {self.powersgd.update_cadence}"
            )
        if not isinstance(self.powersgd.warm_start, bool):
            raise ValueError(
                f"comm_eff.powersgd.warm_start must be a bool; got "
                f"{type(self.powersgd.warm_start).__name__} ({self.powersgd.warm_start!r})"
            )
        if not isinstance(self.powersgd.compress_recompute, bool):
            raise ValueError(
                f"comm_eff.powersgd.compress_recompute must be a bool; got "
                f"{type(self.powersgd.compress_recompute).__name__} ({self.powersgd.compress_recompute!r})"
            )
        if not isinstance(self.powersgd.sync_basis, bool):
            raise ValueError(
                f"comm_eff.powersgd.sync_basis must be a bool; got "
                f"{type(self.powersgd.sync_basis).__name__} ({self.powersgd.sync_basis!r})"
            )
        if self.powersgd.qr_dtype not in ("fp32", "bf16"):
            raise ValueError(
                f"comm_eff.powersgd.qr_dtype must be one of (fp32, bf16); got {self.powersgd.qr_dtype!r}"
            )
        if self.powersgd.reortho_eps <= 0.0:
            raise ValueError(f"comm_eff.powersgd.reortho_eps must be > 0; got {self.powersgd.reortho_eps}")
        # EXP-26 Step C: Q-basis family. Validated to the closed enum so a typo
        # (q_basis=gradient) is a loud error, not a silent fall-through to "act".
        if self.powersgd.q_basis not in Q_BASIS_FAMILIES:
            raise ValueError(
                f"comm_eff.powersgd.q_basis must be one of {Q_BASIS_FAMILIES}; "
                f"got {self.powersgd.q_basis!r}"
            )
        # EXP-26 Step C1: the PASSIVE screen family list. Every entry must be a
        # known family; a typo is a loud error (a silently-dropped family would make
        # the screen miss an arm). OmegaConf may pass a ListConfig — iterate it.
        for _fam in list(self.powersgd.q_basis_passive):
            if _fam not in Q_BASIS_FAMILIES:
                raise ValueError(
                    f"comm_eff.powersgd.q_basis_passive entries must each be one of "
                    f"{Q_BASIS_FAMILIES}; got {_fam!r}"
                )
        # EXP-26 Step C1: the hybrid column split. Only meaningful when the hybrid
        # family is requested (live or passive); otherwise the (default -1/-1 AUTO)
        # values are inert. When BOTH are set EXPLICITLY (>= 0) they MUST sum to the
        # rank; the -1 sentinel means AUTO (resolved to ceil(r/2) + (r-act) by the
        # compressor) and is unconstrained. A single explicit value with the other
        # at -1 is a config error (ambiguous).
        _hybrid_used = (self.powersgd.q_basis == "hybrid") or ("hybrid" in list(self.powersgd.q_basis_passive))
        if _hybrid_used:
            _a, _g = self.powersgd.hybrid_act_cols, self.powersgd.hybrid_grad_cols
            if (_a < 0) != (_g < 0):
                raise ValueError(
                    "comm_eff.powersgd.hybrid_act_cols / hybrid_grad_cols must BOTH be -1 "
                    f"(AUTO) or BOTH be >= 0 (explicit); got {_a} / {_g}"
                )
            if _a >= 0 and _g >= 0 and _a + _g != self.powersgd.rank:
                raise ValueError(
                    "comm_eff.powersgd.hybrid_act_cols + hybrid_grad_cols must equal "
                    f"powersgd.rank ({self.powersgd.rank}) when set explicitly + the hybrid "
                    f"family is used; got {_a} + {_g} = {_a + _g}. (Use -1/-1 for AUTO.)"
                )
        # EXP-26 Step A: diagnostic-capture block. Validated unconditionally (the
        # keys are registered regardless of capture.enabled) so a bad dump_dtype /
        # negative cap fails fast even on a non-capture run that forwards them.
        if self.capture.dump_dtype not in ("fp32", "bf16"):
            raise ValueError(
                f"comm_eff.capture.dump_dtype must be one of (fp32, bf16); got {self.capture.dump_dtype!r}"
            )
        if self.capture.fresh_anchor_loss not in ("clean_pg", "ppo_clip"):
            raise ValueError(
                "comm_eff.capture.fresh_anchor_loss must be one of (clean_pg, ppo_clip); "
                f"got {self.capture.fresh_anchor_loss!r}"
            )
        if self.capture.stratified_targets < 0:
            raise ValueError(
                f"comm_eff.capture.stratified_targets must be >= 0; got {self.capture.stratified_targets}"
            )
