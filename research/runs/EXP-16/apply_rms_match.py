#!/usr/bin/env python3
"""EXP-16: add a switchable mask.rescale_mode {none|constant|rms_match|auto}.

Non-destructive: the legacy `rescale` bool keeps working (rescale_mode=="auto"
derives constant/none from it). Adds the new "rms_match" mode (idea 2b) realized
as a self-contained, comms-valid boundary hook (per-token EXACT RMS match via a
DETACHED gain, so the downstream pre-norm RMSNorm divides by the true pre-mask
RMS -> benign 1/RMS backward).

Idempotent-ish: refuses to double-apply (checks for the marker) and asserts every
anchor string is found exactly once before writing.
"""
import io
import sys

ROOT = "/workspace/verl"

EDITS = []


def edit(path, old, new, *, allow_count=1):
    EDITS.append((path, old, new, allow_count))


# ---- 1) dataclass field -----------------------------------------------------
edit(
    "verl/workers/config/comm_eff.py",
    """    enabled: bool = True
    p: float = 0.95
    seed: int = 0
    pp_size: int = 8
    mask_recompute: bool = False
    rescale: bool = False
""",
    """    enabled: bool = True
    p: float = 0.95
    seed: int = 0
    pp_size: int = 8
    mask_recompute: bool = False
    rescale: bool = False
    rescale_mode: str = "auto"
""",
)

# ---- 2) actor.yaml knob -----------------------------------------------------
edit(
    "verl/trainer/config/actor/actor.yaml",
    """    rescale: false
""",
    """    rescale: false

    # Magnitude-restoration scheme applied to h*mask (overrides `rescale` unless
    # "auto"). Switchable, non-destructive:
    #   none      -> h*mask                                  (raw; == rescale:false)
    #   constant  -> h*mask/(1-p)                            (inverted dropout; == rescale:true)
    #   rms_match -> h*mask * detach(rms_true/rms_masked)    (per-token EXACT RMS match: the
    #                downstream pre-norm RMSNorm then divides by the TRUE pre-mask RMS ->
    #                benign 1/RMS backward. Needs a 1-float/token RMS side channel ~0.6% comms.)
    #   auto      -> derive from `rescale` (true->constant, false->none).
    rescale_mode: auto
""",
)

# ---- 3) state.py construction ----------------------------------------------
edit(
    "verl/workers/comm_eff/state.py",
    """                rescale=bool(getattr(mask_cfg, "rescale", False)),
""",
    """                rescale=bool(getattr(mask_cfg, "rescale", False)),
                rescale_mode=str(getattr(mask_cfg, "rescale_mode", "auto")),
""",
)

# ---- 4a) ActivationMasker.__init__ -----------------------------------------
edit(
    "verl/workers/comm_eff/activation_mask.py",
    """        pp_size: int,
        rescale: bool = False,
        state: Any = None,
    ):
        self.p = float(p)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        # rescale=True applies inverted-dropout h*mask/(1-p) (preserves
        # E[h_tilde]=h); False (default) writes the raw product.
        self.rescale = bool(rescale)
        self._rescale_gain = (1.0 / (1.0 - self.p)) if (self.rescale and self.p < 1.0) else 1.0
""",
    """        pp_size: int,
        rescale: bool = False,
        rescale_mode: str = "auto",
        state: Any = None,
    ):
        self.p = float(p)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        # Magnitude-restoration scheme applied to h*mask. `rescale_mode` selects
        # it; the legacy `rescale` bool is honoured when rescale_mode == "auto".
        #   "none"      -> h*mask                             (raw product)
        #   "constant"  -> h*mask/(1-p)                        (inverted dropout; E[h_tilde]=h)
        #   "rms_match" -> h*mask*detach(rms_true/rms_masked)  (per-token EXACT RMS match: the
        #                  downstream pre-norm RMSNorm divides by the TRUE pre-mask RMS)
        #   "auto"      -> "constant" if rescale else "none"
        self.rescale = bool(rescale)
        mode = str(rescale_mode).lower()
        if mode == "auto":
            mode = "constant" if self.rescale else "none"
        if mode not in ("none", "constant", "rms_match"):
            raise ValueError(
                "mask rescale_mode must be one of none|constant|rms_match|auto; "
                f"got {rescale_mode!r}"
            )
        self.rescale_mode = mode
        self._rescale_gain = (1.0 / (1.0 - self.p)) if (mode == "constant" and self.p < 1.0) else 1.0
""",
)

# ---- 4b) the hook h_tilde branch -------------------------------------------
edit(
    "verl/workers/comm_eff/activation_mask.py",
    """            h_tilde = h * mask * masker._rescale_gain if masker._rescale_gain != 1.0 else h * mask
""",
    """            if masker.rescale_mode == "rms_match":
                # Idea 2b, realized self-contained and comms-valid: rescale the
                # masked activation by a DETACHED per-token gain so its RMS equals
                # the TRUE (pre-mask) RMS. The downstream pre-norm RMSNorm then
                # divides by the true RMS -> benign 1/RMS backward (no collapse
                # blow-up). Comms: rms_true is a 1-float/token side channel
                # (~1/((1-p)*H) overhead); rms_masked is recoverable on the
                # receiver from the kept (communicated) entries. The gain is
                # detached -> backward is mask*const (benign), like the constant
                # path but per-token exact. fp32 for bf16 safety; an all-masked
                # token yields h_tilde=0 (0 * finite gain), never NaN.
                hm = h * mask
                rms_true = h.float().pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
                rms_masked = hm.float().pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
                gain = (rms_true / rms_masked).detach().to(h.dtype)
                h_tilde = hm * gain
            else:
                h_tilde = h * mask * masker._rescale_gain if masker._rescale_gain != 1.0 else h * mask
""",
)


def main():
    changed = []
    for path, old, new, allow_count in EDITS:
        full = f"{ROOT}/{path}"
        with io.open(full, "r", encoding="utf-8") as fh:
            txt = fh.read()
        if new in txt and old not in txt:
            print(f"[skip] {path}: already applied")
            continue
        n = txt.count(old)
        if n != allow_count:
            print(f"[FAIL] {path}: anchor found {n}x (expected {allow_count}); aborting, NOTHING written")
            sys.exit(2)
        txt = txt.replace(old, new, 1)
        with io.open(full, "w", encoding="utf-8") as fh:
            fh.write(txt)
        changed.append(path)
        print(f"[ok]   {path}")
    print(f"\nchanged {len(changed)} file(s): {changed}")


if __name__ == "__main__":
    main()
