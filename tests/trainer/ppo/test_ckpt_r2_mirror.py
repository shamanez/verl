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

"""CPU unit tests for the EXP-58 checkpoint -> R2 on-the-go mirror helpers in
``RayPPOTrainer`` (``_get_ckpt_r2_sink`` / ``_maybe_upload_checkpoint_to_r2`` /
``_maybe_upload_ckpt_tracker_to_r2`` / ``_close_ckpt_r2_sink``).

The full ``ray_trainer`` module pulls in Ray + the vLLM serving stack, which is
not installed on a laptop. So instead of importing it, we lift the exact SOURCE
of the four helper methods out of ``ray_trainer.py`` with ``ast`` and exec them
into a throwaway host class. This tests the real bytes on disk — the ``os.walk``
+ ``relpath`` key_suffix, the OFF-path no-op (imports nothing / spawns nothing),
the single-node assert, the tracker COPY-then-upload (real tracker survives), and
the run-end drain — without a GPU or the serving deps. The R2 sink itself is the
real ``R2ArtifactSink`` with only its ``aws`` subprocess monkeypatched (same
pattern as tests/workers/comm_eff/test_r2_sink.py), so the verify -> manifest ->
delete-local flow is exercised for real.

These map 1:1 to the plan's hard invariants:
  * method-OFF byte-parity            -> test_off_path_is_strict_noop_*
  * on-the-go upload-then-delete      -> test_on_uploads_all_files_and_deletes_local
  * resume completeness (key_suffix)  -> test_key_suffix_is_relpath_from_default_local_dir
  * resume completeness (tracker)     -> test_tracker_uploaded_under_root_key_and_kept
  * drain barrier + single-node guard -> test_close_drains_* / test_multinode_asserts
"""

import ast
import json
import os
import textwrap
import threading

import pytest
from omegaconf import OmegaConf

from verl.workers.comm_eff import r2_sink as r2mod
from verl.workers.comm_eff.r2_sink import R2_REQUIRED_BUCKET

_RAY_TRAINER_PY = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "verl", "trainer", "ppo", "ray_trainer.py"
)

_HELPER_NAMES = (
    "_get_ckpt_r2_sink",
    "_maybe_upload_checkpoint_to_r2",
    "_maybe_upload_ckpt_tracker_to_r2",
    "_close_ckpt_r2_sink",
)


def _load_helper_host():
    """Return a host class carrying the four ckpt-R2 helper methods, lifted from
    the real ray_trainer.py source via ast (no import of the heavy module)."""
    src = open(os.path.abspath(_RAY_TRAINER_PY)).read()
    tree = ast.parse(src)
    # Find the RayPPOTrainer class, then the four method defs inside it.
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer"
    )
    wanted = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in _HELPER_NAMES:
            wanted[node.name] = node
    missing = set(_HELPER_NAMES) - set(wanted)
    assert not missing, f"helper methods not found in ray_trainer.py: {missing}"

    # Re-emit each method as a top-level function source, dedented, then exec.
    ns = {"os": os}
    for name in _HELPER_NAMES:
        func_src = textwrap.dedent(ast.get_source_segment(src, wanted[name]))
        exec(compile(func_src, _RAY_TRAINER_PY, "exec"), ns)

    class Host:
        pass

    for name in _HELPER_NAMES:
        setattr(Host, name, ns[name])
    return Host


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _install_fake_aws(monkeypatch, *, cp_rc=0, head_rc=0):
    """Patch r2_sink.subprocess.run so cp records the source size and head-object
    verifies against it (size match). Concurrency-safe for the async worker pool."""
    calls = []
    sizes = {}
    lock = threading.Lock()

    def _key_of(cmd):
        if "cp" in cmd:
            return cmd[4].split("/", 3)[-1]
        i = cmd.index("--key")
        return cmd[i + 1]

    def fake_run(cmd, **kwargs):
        with lock:
            calls.append(list(cmd))
        if "cp" in cmd:
            if cp_rc == 0:
                with lock:
                    sizes[_key_of(cmd)] = os.path.getsize(cmd[3])
            return _FakeProc(returncode=cp_rc, stderr="cp boom" if cp_rc else "")
        if "head-object" in cmd:
            if head_rc:
                return _FakeProc(returncode=head_rc, stderr="head boom")
            with lock:
                size = sizes[_key_of(cmd)]
            return _FakeProc(returncode=0, stdout=json.dumps({"ContentLength": size}))
        raise AssertionError(f"unexpected aws cmd: {cmd}")

    monkeypatch.setattr(r2mod.subprocess, "run", fake_run)
    return calls


def _r2_env(monkeypatch):
    monkeypatch.setenv("R2_BUCKET", R2_REQUIRED_BUCKET)
    monkeypatch.setenv("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKIA_test")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret_test")
    monkeypatch.setenv("R2_EXPERIMENT", "EXP-58")
    monkeypatch.setenv("R2_REGIME", "regimeA")


def _make_host(tmp_path, *, enabled, nnodes=1, world_size=2, global_steps=1):
    """Build a Host with a fake config + a real global_step_<N>/ tree on disk.

    The tree mirrors the FSDP1 local-save layout the plan asserts:
      global_step_<N>/actor/model_/optim_/extra_state_world_size_<W>_rank_<R>.pt (R in 0..W-1)
      global_step_<N>/actor/huggingface/{config.json,tokenizer.json}
      global_step_<N>/actor/fsdp_config.json
      global_step_<N>/data.pt
    """
    host = _load_helper_host()()
    default_local_dir = str(tmp_path / "ckpts" / "proj" / "exp")
    cfg = OmegaConf.create(
        {"trainer": {"checkpoint_r2_enabled": enabled, "nnodes": nnodes, "default_local_dir": default_local_dir}}
    )
    host.config = cfg
    host.global_steps = global_steps

    step_dir = os.path.join(default_local_dir, f"global_step_{global_steps}")
    actor = os.path.join(step_dir, "actor")
    hf = os.path.join(actor, "huggingface")
    os.makedirs(hf, exist_ok=True)
    expected_suffixes = set()

    def _touch(path, payload):
        with open(path, "wb") as fh:
            fh.write(payload)
        expected_suffixes.add(os.path.relpath(os.path.abspath(path), os.path.abspath(default_local_dir)))

    for r in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            _touch(os.path.join(actor, f"{kind}_world_size_{world_size}_rank_{r}.pt"), b"x" * (10 + r))
    _touch(os.path.join(actor, "fsdp_config.json"), b'{"fsdp":1}')
    _touch(os.path.join(hf, "config.json"), b'{"cfg":1}')
    _touch(os.path.join(hf, "tokenizer.json"), b'{"tok":1}')
    _touch(os.path.join(step_dir, "data.pt"), b"dataloader-state")

    return host, default_local_dir, step_dir, expected_suffixes


# --------------------------------------------------------------------------- #
# method-OFF byte-parity (invariant #1)
# --------------------------------------------------------------------------- #
def test_off_path_returns_none_and_imports_nothing(tmp_path, monkeypatch):
    # Poison the import so ANY attempt to import r2_sink on the OFF save path fails
    # the test loudly (proves the deferred-import no-op).
    import builtins

    real_import = builtins.__import__

    def poisoned(name, *a, **k):
        if "r2_sink" in name:
            raise AssertionError(f"OFF path must not import r2_sink; tried to import {name!r}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", poisoned)

    host, dld, step_dir, _ = _make_host(tmp_path, enabled=False)
    assert host._get_ckpt_r2_sink() is None
    # No-ops must not touch the tree or spawn a sink.
    host._maybe_upload_checkpoint_to_r2(step_dir)
    tracker = os.path.join(dld, "latest_checkpointed_iteration.txt")
    with open(tracker, "w") as f:
        f.write("1")
    host._maybe_upload_ckpt_tracker_to_r2(tracker)
    host._close_ckpt_r2_sink()  # no sink built -> no-op, no raise
    assert getattr(host, "_ckpt_r2_sink", None) is None


def test_off_path_leaves_all_local_files_untouched(tmp_path, monkeypatch):
    host, dld, step_dir, expected = _make_host(tmp_path, enabled=False)
    before = {p for p in _walk_files(step_dir)}
    host._maybe_upload_checkpoint_to_r2(step_dir)
    after = {p for p in _walk_files(step_dir)}
    assert before == after and len(after) == len(expected)


# --------------------------------------------------------------------------- #
# on-the-go upload-then-delete + resume completeness (invariants #2, #3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("async_mode", ["true", "false"])
def test_on_uploads_all_files_and_deletes_local(tmp_path, monkeypatch, async_mode):
    _r2_env(monkeypatch)
    monkeypatch.setenv("CKPT_R2_ASYNC", async_mode)
    monkeypatch.setenv("CKPT_R2_DELETE_LOCAL", "true")
    monkeypatch.setenv("CKPT_R2_WORKERS", "3")
    calls = _install_fake_aws(monkeypatch)

    host, dld, step_dir, expected = _make_host(tmp_path, enabled=True, world_size=2)
    host._maybe_upload_checkpoint_to_r2(step_dir)
    sink = host._ckpt_r2_sink
    sink.close()  # drain (async) / no-op (sync); fail-loud if any upload unverified

    # (a) every file was uploaded (cp calls == #files), (b) all verified in manifest,
    # (c) local staging files deleted after verify.
    cp_keys = [c[4].split("/", 3)[-1] for c in calls if "cp" in c]
    assert len(cp_keys) == len(expected)
    manifest = [json.loads(x) for x in open(sink.manifest_path)]
    assert all(row["verified"] is True for row in manifest)
    assert sink.n_errors == 0
    assert sink.n_uploaded == len(expected)
    # local shards gone (delete-after-verify) => disk bounded, not keep-all.
    assert _walk_files(step_dir) == []


def test_key_suffix_is_relpath_from_default_local_dir(tmp_path, monkeypatch):
    _r2_env(monkeypatch)
    monkeypatch.setenv("CKPT_R2_ASYNC", "false")  # sync => deterministic call order
    calls = _install_fake_aws(monkeypatch)

    host, dld, step_dir, expected = _make_host(tmp_path, enabled=True, world_size=2, global_steps=1)
    host._maybe_upload_checkpoint_to_r2(step_dir)

    key_prefix = host._ckpt_r2_sink.key_prefix  # autonomous-harness-rlvr-compression/EXP-58/regimeA/checkpoints
    uploaded_suffixes = set()
    for c in calls:
        if "cp" in c:
            key = c[4].split("/", 3)[-1]  # <key_prefix>/<suffix>
            assert key.startswith(key_prefix + "/"), key
            uploaded_suffixes.add(key[len(key_prefix) + 1 :])
    # key_suffix == relpath(file, default_local_dir) for every object -> R2 byte-mirrors local.
    assert uploaded_suffixes == expected
    # spot-check the exact FSDP1 shard name mirror the plan calls out.
    assert "global_step_1/actor/model_world_size_2_rank_0.pt" in uploaded_suffixes
    assert "global_step_1/actor/model_world_size_2_rank_1.pt" in uploaded_suffixes
    assert "global_step_1/data.pt" in uploaded_suffixes
    assert "global_step_1/actor/huggingface/config.json" in uploaded_suffixes
    assert "global_step_1/actor/fsdp_config.json" in uploaded_suffixes


def test_key_suffix_correct_even_with_relative_default_local_dir(tmp_path, monkeypatch):
    """The launcher leaves default_local_dir RELATIVE (checkpoints/<proj>/<exp>).
    The helper abspaths both the walk root and the relpath base, so the suffix is
    still global_step_<N>/... — the exact bug the plan warns about."""
    _r2_env(monkeypatch)
    monkeypatch.setenv("CKPT_R2_ASYNC", "false")
    calls = _install_fake_aws(monkeypatch)

    # Build the tree under a RELATIVE default_local_dir by chdir-ing into tmp_path.
    monkeypatch.chdir(tmp_path)
    host = _load_helper_host()()
    rel_dld = os.path.join("checkpoints", "proj", "exp")
    host.config = OmegaConf.create(
        {"trainer": {"checkpoint_r2_enabled": True, "nnodes": 1, "default_local_dir": rel_dld}}
    )
    host.global_steps = 20
    actor = os.path.join(rel_dld, "global_step_20", "actor")
    os.makedirs(actor, exist_ok=True)
    with open(os.path.join(actor, "model_world_size_1_rank_0.pt"), "wb") as f:
        f.write(b"weights")

    host._maybe_upload_checkpoint_to_r2(os.path.join(rel_dld, "global_step_20"))
    key_prefix = host._ckpt_r2_sink.key_prefix
    keys = [c[4].split("/", 3)[-1] for c in calls if "cp" in c]
    assert keys == [f"{key_prefix}/global_step_20/actor/model_world_size_1_rank_0.pt"]


# --------------------------------------------------------------------------- #
# resume completeness — root tracker (invariant #3)
# --------------------------------------------------------------------------- #
def test_tracker_uploaded_under_root_key_and_local_kept(tmp_path, monkeypatch):
    _r2_env(monkeypatch)
    monkeypatch.setenv("CKPT_R2_ASYNC", "false")
    calls = _install_fake_aws(monkeypatch)

    host, dld, step_dir, _ = _make_host(tmp_path, enabled=True, global_steps=40)
    tracker = os.path.join(dld, "latest_checkpointed_iteration.txt")
    with open(tracker, "w") as f:
        f.write("40")

    host._maybe_upload_ckpt_tracker_to_r2(tracker)

    key_prefix = host._ckpt_r2_sink.key_prefix
    cp_keys = [c[4].split("/", 3)[-1] for c in calls if "cp" in c]
    # Uploaded under the fixed ROOT key (overwrites each save) so find_latest_ckpt_path
    # resolves from R2 alone.
    assert cp_keys == [f"{key_prefix}/latest_checkpointed_iteration.txt"]
    # CRITICAL: the real local tracker must SURVIVE (delete-after-verify hit the temp
    # copy, not the live root marker _load_checkpoint reads on an in-place resume).
    assert os.path.exists(tracker)
    assert open(tracker).read() == "40"
    manifest = [json.loads(x) for x in open(host._ckpt_r2_sink.manifest_path)]
    assert manifest[-1]["verified"] is True and manifest[-1].get("tracker") is True


# --------------------------------------------------------------------------- #
# drain barrier fails loud (invariant #4)
# --------------------------------------------------------------------------- #
def test_close_drains_clean_run_no_errors(tmp_path, monkeypatch):
    _r2_env(monkeypatch)
    monkeypatch.setenv("CKPT_R2_ASYNC", "true")
    _install_fake_aws(monkeypatch)
    host, dld, step_dir, expected = _make_host(tmp_path, enabled=True, world_size=2)
    host._maybe_upload_checkpoint_to_r2(step_dir)
    host._close_ckpt_r2_sink()  # must not raise on a clean run
    assert host._ckpt_r2_sink.n_errors == 0
    assert host._ckpt_r2_sink.n_uploaded == len(expected)


def test_close_raises_loud_on_upload_failure(tmp_path, monkeypatch):
    _r2_env(monkeypatch)
    monkeypatch.setenv("CKPT_R2_ASYNC", "true")
    _install_fake_aws(monkeypatch, cp_rc=1)  # every upload fails
    host, dld, step_dir, _ = _make_host(tmp_path, enabled=True, world_size=2)
    host._maybe_upload_checkpoint_to_r2(step_dir)
    with pytest.raises(Exception):
        host._close_ckpt_r2_sink()  # fail-loud: unverified uploads must surface


def test_close_is_noop_when_sink_never_built(tmp_path, monkeypatch):
    host, dld, step_dir, _ = _make_host(tmp_path, enabled=False)
    host._close_ckpt_r2_sink()  # no save happened -> no sink -> no raise


# --------------------------------------------------------------------------- #
# single-node guard (invariant #5 — FSDP1 rank-0 walk only valid on 1 node)
# --------------------------------------------------------------------------- #
def test_multinode_asserts_loud(tmp_path, monkeypatch):
    _r2_env(monkeypatch)
    host, dld, step_dir, _ = _make_host(tmp_path, enabled=True, nnodes=2)
    with pytest.raises(AssertionError, match="single-node"):
        host._get_ckpt_r2_sink()


def test_sink_is_cached_across_saves(tmp_path, monkeypatch):
    _r2_env(monkeypatch)
    monkeypatch.setenv("CKPT_R2_ASYNC", "false")
    _install_fake_aws(monkeypatch)
    host, dld, step_dir, _ = _make_host(tmp_path, enabled=True)
    s1 = host._get_ckpt_r2_sink()
    s2 = host._get_ckpt_r2_sink()
    assert s1 is s2  # one sink, one manifest, drained once at run-end


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _walk_files(root):
    out = []
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            out.append(os.path.join(dp, fn))
    return out
