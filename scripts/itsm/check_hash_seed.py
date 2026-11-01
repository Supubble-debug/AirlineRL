"""Check, in THIS freshly started process, that the ITSM worker child inherits a fixed hash seed
and that a list observation has a reproducible order (protocol §13.1). No LLM, no GPU.

Usage:
    scripts/itsm/itsm_python.sh scripts/itsm/check_hash_seed.py [--task ID] [--require-fixed]

Prints one JSON line: the parent's and the worker child's startup hash flags and the SHA256
plus the record-id order of unfiltered list observations. Run it twice (two independent
processes) and compare the digests; the unit test does exactly that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_rl.envs.itsm_scripted_user import forbid_external_llm  # noqa: E402

forbid_external_llm()

LIST_TOOLS = ("list_incidents", "list_users")
_ID = re.compile(r'\{"[a-z_]*id":"([^"]+)"')  # the first (id) field of each listed record


def main() -> int:
    import functools

    from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig

    from agent_rl.envs.itsm_env import ItsmAction, ItsmEnvConfig
    from agent_rl.envs.itsm_scripted_user import PersonaScriptedUser
    from agent_rl.envs.itsm_worker import ItsmEnvWorker
    from agent_rl.utils.hash_seed import hash_seed_report

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="itsmbench_msp_013")
    parser.add_argument("--split-version", default="itsm_split_v3")
    parser.add_argument("--require-fixed", action="store_true")
    args = parser.parse_args()
    report = {"task_id": args.task, "parent": hash_seed_report()}
    config = ItsmEnvConfig(task_id=args.task, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=40, split_version=args.split_version)
    try:
        worker = ItsmEnvWorker(config, call_timeout_s=120.0, user_factory=functools.partial(PersonaScriptedUser, args.task, args.split_version),
                               initializer=forbid_external_llm, require_fixed_hash_seed=args.require_fixed)
    except Exception as error:  # noqa: BLE001 - reported; non-zero exit
        report["startup_error"] = f"{type(error).__name__}: {error}"
        report["stage"] = getattr(error, "stage", None)
        print(json.dumps(report))
        return 2
    try:
        report["child"] = worker.child_hash_seed
        worker.reset()
        report["observations"] = {}
        for tool in LIST_TOOLS:
            observation = worker.step(ItsmAction.tool_call(tool, {})).observation
            ids = _ID.findall(observation)
            report["observations"][tool] = {"sha256": hashlib.sha256(observation.encode()).hexdigest(), "records": len(ids), "first_ids": ids[:8]}
    finally:
        worker.close()
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
