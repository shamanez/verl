#!/usr/bin/env python3
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

"""Extract the GROUND-TRUTH resolved parameters of a training run from its log.

After many experiments it is easy to lose track of what settings an agent
*actually* used — hand-written manifests drift from reality (the comm-eff
baseline manifest said anchor.cadence=5 while the run truly used 4). This
script recovers the real values from the launcher's `set -x` trace in
train.log, which prints the fully shell-expanded
`python3 -m verl.trainer.main_ppo <args>` command. Hydra is last-write-wins,
so a key passed twice resolves to its final value — we apply that here so the
output is unambiguous.

Outputs (into the run dir, next to verdict.md):
  resolved_cmd.txt     — the raw expanded main_ppo command line(s), verbatim
  resolved_params.txt  — one `key=value` per line, last-wins, sorted; the
                         single source of truth for "what actually ran"

Usage:
  python research/scripts/capture_resolved_config.py runs/<run-id>
  python research/scripts/capture_resolved_config.py runs/<run-id>/train.log
"""

import shlex
import sys
from pathlib import Path

MARKER = "python3 -m verl.trainer.main_ppo"
# Knobs worth surfacing first in the summary — the load-bearing ones a reader
# checks at a glance. Everything still lands in resolved_params.txt regardless.
HEADLINE_PREFIXES = (
    "actor_rollout_ref.actor.comm_eff",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.entropy_coeff",
    "algorithm.use_kl_in_reward",
    "data.train_batch_size",
    "data.max_prompt_length",
    "data.max_response_length",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.actor.fsdp_config.use_orig_params",
    "trainer.total_training_steps",
    "trainer.experiment_name",
)


def find_commands(log_text: str):
    """Return each fully-expanded main_ppo command found in the log."""
    cmds = []
    for line in log_text.splitlines():
        idx = line.find(MARKER)
        if idx == -1:
            continue
        # Drop the `set -x` prefix ("+ ", "++ ") and any Ray "(TaskRunner …) "
        # banner by slicing from "python3" onward.
        py = line.find("python3", 0, idx + len("python3"))
        cmds.append(line[py if py != -1 else idx :].rstrip())
    return cmds


def parse_params(cmd: str) -> dict:
    """key=value tokens, last-write-wins (Hydra semantics)."""
    params = {}
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        # Unbalanced quote in the trace — fall back to whitespace split.
        tokens = cmd.split()
    for tok in tokens:
        if "=" not in tok or tok.startswith("-"):
            continue
        key, _, val = tok.partition("=")
        if not key or "." not in key and key not in ("nproc_per_node",):
            # Hydra keys are dotted; skip bare `python3`/flags/etc.
            if "." not in key:
                continue
        params[key] = val
    return params


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    target = Path(argv[1])
    run_dir = target if target.is_dir() else target.parent
    log = target if target.is_file() else run_dir / "train.log"
    if not log.is_file():
        print(f"capture_resolved_config: no train.log at {log}", file=sys.stderr)
        return 1

    cmds = find_commands(log.read_text(errors="replace"))
    if not cmds:
        print(
            f"capture_resolved_config: no '{MARKER}' invocation found in {log} (was the launcher run under `set -x`?)",
            file=sys.stderr,
        )
        return 1

    (run_dir / "resolved_cmd.txt").write_text("\n\n".join(cmds) + "\n")

    # If a run launched multiple cells, the last invocation wins the flat file;
    # note the count so the reader knows to consult resolved_cmd.txt for the rest.
    params = parse_params(cmds[-1])
    lines = [f"{k}={params[k]}" for k in sorted(params)]
    header = [
        "# Resolved parameters — GROUND TRUTH, extracted from train.log set -x trace.",
        "# last-write-wins applied (Hydra semantics). Do NOT hand-edit; regenerate via",
        "#   python research/scripts/capture_resolved_config.py <run_dir>",
    ]
    if len(cmds) > 1:
        header.append(
            f"# NOTE: {len(cmds)} main_ppo invocations in this run; this file reflects "
            "the LAST. See resolved_cmd.txt for all."
        )
    (run_dir / "resolved_params.txt").write_text("\n".join(header) + "\n" + "\n".join(lines) + "\n")

    # Human summary to stdout.
    print(
        f"captured {len(params)} resolved params from {log} "
        f"({len(cmds)} main_ppo invocation(s)) -> {run_dir}/resolved_params.txt"
    )
    headline = [(k, params[k]) for k in sorted(params) if any(k.startswith(p) for p in HEADLINE_PREFIXES)]
    if headline:
        print("headline knobs (the ones that drift):")
        for k, v in headline:
            print(f"  {k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
