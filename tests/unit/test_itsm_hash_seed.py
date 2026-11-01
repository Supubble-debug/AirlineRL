"""Protocol §13.1: the worker child inherits PYTHONHASHSEED=0 and list-observation order is
reproducible across INDEPENDENT, freshly started processes (no LLM, no GPU).

Fixing the hash seed makes the order of Bench2 list results reproducible; it says nothing about
model sampling or simulator determinism.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_rl.envs.itsm_scoring import ItsmInfrastructureError

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "itsm" / "itsm_python.sh"
CHECK = REPO_ROOT / "scripts" / "itsm" / "check_hash_seed.py"


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHASHSEED", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MICU_API_KEY")}
    return {**env, "ITSM_PYTHON": sys.executable, "HF_HUB_OFFLINE": "1", **extra}


def _run(argv, env):
    result = subprocess.run(argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    return result.returncode, json.loads(result.stdout.strip().splitlines()[-1])


def test_launcher_children_inherit_the_seed_and_list_order_is_reproducible():
    """Two independent processes started through the launcher, with no seed in the caller's env."""
    runs = [_run([str(LAUNCHER), str(CHECK), "--require-fixed"], _env()) for _ in range(2)]
    for code, report in runs:
        assert code == 0, report
        assert report["parent"]["fixed"] and report["parent"]["pythonhashseed_env"] == "0"
        assert report["child"]["fixed"] and report["child"]["hash_randomization"] == 0  # the worker really inherited it
        assert report["child"]["pid"] != report["parent"]["pid"]
        assert report["observations"]["list_incidents"]["records"] > 10
    (_, first), (_, second) = runs
    assert first["child"]["pid"] != second["child"]["pid"]
    assert first["observations"] == second["observations"]  # same order, same bytes


def test_different_seeds_reorder_the_same_list_negative_control():
    """Without the fix the order differs between processes: the check above is not vacuous."""
    runs = [_run([sys.executable, str(CHECK)], _env(PYTHONHASHSEED=seed)) for seed in ("1", "2")]
    assert all(code == 0 and not report["child"]["fixed"] for code, report in runs)
    one, two = (report["observations"]["list_incidents"] for _, report in runs)
    assert one["records"] == two["records"] and one["sha256"] != two["sha256"]


def test_unfixed_child_is_refused_at_startup():
    code, report = _run([sys.executable, str(CHECK), "--require-fixed"], _env(PYTHONHASHSEED="1"))
    assert code == 2 and report["stage"] == "startup" and "PYTHONHASHSEED=0" in report["startup_error"]
    assert "observations" not in report  # nothing ran: no reset, no tool call


def test_agent_loop_refuses_an_unfixed_worker_before_sampling(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_itsm_agent_loop import GOLD_SCRIPT, StubSampler, make_loop, run

    monkeypatch.setenv("PYTHONHASHSEED", "1")  # the spawned child inherits this
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler, worker=True)
    with pytest.raises(ItsmInfrastructureError) as raised:
        run(loop)
    assert raised.value.stage == "startup"
    assert sampler.calls == []  # failed before any sampling, so no reward
    assert not loop.envs[0].is_alive and multiprocessing.active_children() == []
