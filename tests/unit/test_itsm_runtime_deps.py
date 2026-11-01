"""Runtime dependencies outside the code-sync set (configs/itsm/runtime_dependencies.json).

A simulated deployment (worktree minus the roots projtool's code sync ignores, so no /docs)
must fail task selection / usage validation until the listed files are pushed, succeed after
the push, and fail again -- before any sampling -- when a dependency is missing or modified.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_rl.data import itsm_runtime_deps, itsm_task_sets
from agent_rl.data.itsm_runtime_deps import RuntimeDependencyError, load_manifest, sha256_file
from agent_rl.data.itsm_usage import S0_AUDIT_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = "scripts/itsm/runtime_deps.py"
AUDIT = "docs/itsm_stepS0/usage_audit.json"


def _env():
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MICU_API_KEY")}
    return {**env, "HF_HUB_OFFLINE": "1", "PYTHONHASHSEED": "0"}


def _tool(cwd: Path, *args):
    result = subprocess.run([sys.executable, TOOL, *args], cwd=cwd, env=_env(), capture_output=True, text=True, timeout=900)
    return result.returncode, json.loads(result.stdout.strip().splitlines()[-1])


def test_manifest_matches_the_frozen_registry_and_the_sync_rules():
    manifest = load_manifest(REPO_ROOT)
    entries = {e["path"]: e for e in manifest["files"]}
    assert S0_AUDIT_PATH.relative_to(REPO_ROOT).as_posix() == AUDIT in entries  # original path kept
    frozen = json.loads((REPO_ROOT / "configs/data/itsm_usage_v1.json").read_text())["base"]["s0_audit"]
    assert frozen["path"] == AUDIT and frozen["sha256"] == entries[AUDIT]["sha256"] == sha256_file(REPO_ROOT / AUDIT)
    roots = manifest["sync_ignored_roots"]["roots"]
    for path in entries:  # listed only because the code sync does not deliver it
        assert any(("/" + path).startswith(root.rstrip("/") + "/") for root in roots)


def test_clean_deployment_needs_the_push_and_rejects_missing_or_modified(tmp_path):
    deploy = tmp_path / "workspace"
    code, report = _tool(REPO_ROOT, "simulate-remote", "--out", str(deploy))
    assert code == 0 and "docs" in report["excluded"] and not (deploy / "docs").exists()

    code, report = _tool(deploy, "preflight")  # what the remote holds after a code sync only
    assert code == 1 and "missing" in report["error"]

    code, report = _tool(REPO_ROOT, "push", "--target-dir", str(deploy))
    assert code == 0 and report["ok"]
    code, report = _tool(deploy, "preflight")
    assert code == 0, report
    assert Path(report["agent_rl_from"]).is_relative_to(deploy)  # the deployed copy ran, not the worktree
    assert report["grpo_trainable"] == 26 and report["dev"] == 13
    assert not (deploy / "docs/itsm_step6d/candidate_screening.json").exists()  # not a runtime dependency

    target = deploy / AUDIT
    target.write_text(target.read_text().replace('"version"', '"version" ', 1))  # one byte: still valid JSON
    code, report = _tool(deploy, "preflight")
    assert code == 1 and "!= manifest" in report["error"]
    code, report = _tool(deploy, "push", "--target-dir", str(tmp_path / "elsewhere"))  # a modified source is never pushed
    assert code == 1 and report["files"] == [AUDIT]
    target.unlink()
    code, report = _tool(deploy, "preflight")
    assert code == 1 and "missing" in report["error"]
    assert sha256_file(REPO_ROOT / AUDIT) == load_manifest(REPO_ROOT)["files"][0]["sha256"]  # worktree untouched


@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_rollout_fails_before_sampling_when_a_dependency_is_bad(tmp_path, monkeypatch, damage):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_itsm_agent_loop import GOLD_SCRIPT, StubSampler, make_loop, run

    root = tmp_path / "deploy"
    (root / "configs/itsm").mkdir(parents=True)
    (root / "configs/itsm/runtime_dependencies.json").write_bytes((REPO_ROOT / "configs/itsm/runtime_dependencies.json").read_bytes())
    if damage == "modified":
        (root / AUDIT).parent.mkdir(parents=True)
        (root / AUDIT).write_text((REPO_ROOT / AUDIT).read_text() + " ")
    monkeypatch.setattr(itsm_runtime_deps, "REPO_ROOT", root)
    itsm_task_sets._usage.cache_clear()
    try:
        sampler = StubSampler(GOLD_SCRIPT)
        loop = make_loop(sampler)
        with pytest.raises(RuntimeDependencyError):
            run(loop)
        assert sampler.calls == [] and loop.envs == []  # no env, no sample, no reward
    finally:
        itsm_task_sets._usage.cache_clear()
