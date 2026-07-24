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

"""Launcher lint for the issue #93 run matrix (CPU-only, no network).

``run_93_cell.sh`` resolves its arm and echoes the full config BEFORE any
bring-up, and ``DRY_RUN=1`` stops right after that echo; these tests drive the
script through that path with a clean env. Also ``bash -n``-lints every
launcher in the #93 chain and asserts the engine wires the new knobs
(quant.subset_k, rollout_is) through to Hydra.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GRPO_DIR = REPO_ROOT / "examples" / "grpo_trainer"
CELL = GRPO_DIR / "run_93_cell.sh"
ENGINE = GRPO_DIR / "vast_comm_eff_engine_grpo.sh"
TABLE = "0:0.0005,300:0.01,600:0.02"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _run_cell(arm=None, dry_run=True, **extra_env):
    """Run run_93_cell.sh with a minimal env; return CompletedProcess."""
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    if arm is not None:
        env["ARM"] = arm
    if dry_run:
        env["DRY_RUN"] = "1"
    env.update({k: str(v) for k, v in extra_env.items()})
    return subprocess.run(["bash", str(CELL)], env=env, capture_output=True, text=True, timeout=60)


def test_bash_syntax_lint_all_launchers():
    for script in (
        CELL,
        ENGINE,
        GRPO_DIR / "run_qwen25_math_1p5b_rank1_relex_fsdp.sh",
        GRPO_DIR / "run_prf_exactk_600.sh",
    ):
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, f"bash -n failed for {script.name}: {proc.stderr}"


def test_unknown_and_missing_arm_fail_loud():
    proc = _run_cell("bogus")
    assert proc.returncode != 0
    assert "FATAL" in proc.stderr and "unknown ARM" in proc.stderr
    proc = _run_cell(None)
    assert proc.returncode != 0
    assert "FATAL" in proc.stderr


def test_a3_parity_arm_resolves_subset_quant():
    proc = _run_cell("a3")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "a3-srq-parity-k493" in out
    assert "93-long-horizon-stability" in out
    assert "sr_quant" in out
    assert "bits=2" in out and "subset_k=493" in out
    assert "steps=120 test_freq=-1 val_before_train=False save_freq=-1" in out


def test_a1_a2_rounding_pair():
    out_a1 = _run_cell("a1").stdout
    out_a2 = _run_cell("a2").stdout
    assert "bits=1" in out_a1 and "rounding=sr" in out_a1 and "subset_k=0" in out_a1
    assert "bits=1" in out_a2 and "rounding=rn" in out_a2


def test_a4_prf_exactk_plus_cvc():
    out = _run_cell("a4").stdout
    assert "prf_mask" in out and "exact_k=true" in out
    assert "ce_lambda=0.003" in out and "warmup=20" in out
    assert "rollout_is:          null" in out


def test_a5_frlr_plus_token_is():
    out = _run_cell("a5").stdout
    assert "frlr:                true rank=48 k=28" in out
    assert "rollout_is:          token threshold=2.0" in out


def test_b1_requires_codec_arm_and_table():
    proc = _run_cell("b1")
    assert proc.returncode != 0 and "CODEC_ARM" in proc.stderr
    proc = _run_cell("b1", CODEC_ARM="a1")
    assert proc.returncode != 0 and "COMM_EFF_PROBE_KL_TARGET_TABLE" in proc.stderr
    proc = _run_cell("b1", CODEC_ARM="a4", COMM_EFF_PROBE_KL_TARGET_TABLE=TABLE)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "b1-a4-prf-exactk-cvc-ce-ctrl" in out
    assert "every=25 ctrl=true" in out and TABLE in out
    assert "steps=200 test_freq=-1 val_before_train=False save_freq=-1" in out


def test_b1_rejects_unknown_codec_arm():
    proc = _run_cell("b1", CODEC_ARM="a9", COMM_EFF_PROBE_KL_TARGET_TABLE=TABLE)
    assert proc.returncode != 0 and "unknown CODEC_ARM" in proc.stderr


def test_c_winner_cell_val_and_r2():
    proc = _run_cell("c", CODEC_ARM="a3", COMM_EFF_PROBE_KL_TARGET_TABLE=TABLE)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "c-a3-srq-parity-k493-val600" in out
    assert "steps=600 test_freq=150 val_before_train=True save_freq=100" in out
    assert "r2 ckpt sink:        true" in out
    assert "every=25 ctrl=true" in out


def test_engine_wires_new_knobs_to_hydra():
    """Wiring lint: the engine passes subset_k and rollout_is through, and its
    sr_quant boot gate carries the subset bit accounting."""
    text = ENGINE.read_text()
    assert 'actor_rollout_ref.actor.comm_eff.quant.subset_k="$COMM_EFF_QUANT_SUBSET_K"' in text
    assert 'algorithm.rollout_correction.rollout_is="$ROLLOUT_IS"' in text
    assert 'algorithm.rollout_correction.rollout_is_threshold="$ROLLOUT_IS_THRESHOLD"' in text
    assert "sr_quant subset accounting" in text
    assert "incumbent prf exact-k 77x16 = 1232" in text
