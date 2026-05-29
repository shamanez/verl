#!/usr/bin/env python3
"""EXP-16 in-branch hotfix: T5 added comm_eff.spectral.cadence to the dataclass
(CommEffSpectralConfig.cadence) + launcher + engine, but missed the Hydra YAML
schema node verl/trainer/config/actor/actor.yaml. Hydra struct-mode therefore
rejects `actor_rollout_ref.actor.comm_eff.spectral.cadence=...` ("Key 'cadence'
is not in struct"), crashing EVERY cell at config parse. This inserts the
missing `cadence: 1` key into the spectral: block, mirroring anchor.cadence.

Idempotent: no-op if spectral already has a cadence key.
"""
import sys

PATH = "verl/trainer/config/actor/actor.yaml"
lines = open(PATH).read().splitlines()

# locate the `  spectral:` block (2-space indent, top-level under comm_eff)
spec_i = next((i for i, ln in enumerate(lines) if ln.rstrip() == "  spectral:"), None)
if spec_i is None:
    print("FIX_FAIL: no '  spectral:' block found"); sys.exit(1)

# block runs until the next line indented <= 2 spaces that is a key (e.g. next sibling)
end = len(lines)
for j in range(spec_i + 1, len(lines)):
    ln = lines[j]
    if ln.strip() and not ln.startswith("    "):  # dedent out of the 4-space block
        end = j; break
block = lines[spec_i:end]

if any(l.strip().startswith("cadence:") for l in block):
    print("FIX_NOOP: spectral.cadence already present"); sys.exit(0)

# insert right after the spectral `enabled:` line (mirrors anchor block ordering)
ins_rel = next((k for k, l in enumerate(block) if l.strip().startswith("enabled:")), 0)
ins_abs = spec_i + ins_rel + 1
new = [
    "    # EXP-16 spectral-correction cadence in optimizer steps. 1 = fire every",
    "    # step (pre-EXP-16 behavior; strict no-op default). Validated >= 1.",
    "    # Mirrors anchor.cadence; set == anchor.cadence so corrections use a fresh basis.",
    "    cadence: 1",
]
lines[ins_abs:ins_abs] = new
open(PATH, "w").write("\n".join(lines) + "\n")
print(f"FIX_OK: inserted spectral.cadence at line {ins_abs + 1} of {PATH}")
