"""Runtime dependencies outside the code-sync set: push, verify, preflight, simulate a deployment.

The manifest is configs/itsm/runtime_dependencies.json (files at their ORIGINAL paths, with
SHA256). projtool's code-sync session does not copy /docs etc., so these files must be pushed
explicitly before a remote run.

    # local, before run_script: copy every listed file to the remote workspace and verify there
    python scripts/itsm/runtime_deps.py push --ssh-alias autodl-exp-itsm-fc77b761 --remote-root /workspace
    # remote, first thing in a launcher (no sampling yet): deps + usage validation + task selection
    scripts/itsm/itsm_python.sh scripts/itsm/runtime_deps.py preflight
    # offline: build what the remote would hold after a code sync (no /docs), then push into it
    python scripts/itsm/runtime_deps.py simulate-remote --out DIR
    python scripts/itsm/runtime_deps.py push --target-dir DIR

``push`` refuses to send a local file that differs from the manifest, and fails unless every
pushed file's SHA256 at the destination equals the manifest. ``verify`` / ``preflight`` exit 1
on a missing or modified dependency. Nothing is skipped, repaired or deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_rl.data.itsm_runtime_deps import (  # noqa: E402
    RuntimeDependencyError,
    check_runtime_dependencies,
    load_manifest,
    sha256_file,
    verify_runtime_dependencies,
)

#: copied into a simulated deployment (the code under test); other synced top-level entries are
#: symlinked (read only in the simulation: never modified).
COPIED_TREES = ("src", "configs", "scripts")
SYNC_NOISE = ("__pycache__", "*.pyc", ".pytest_cache", ".env", ".venv")


def _ssh(alias: str, ssh_config: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-F", ssh_config, "-o", "BatchMode=yes", alias, command], capture_output=True, text=True, timeout=120)


def cmd_push(args) -> int:
    local_rows = check_runtime_dependencies(REPO_ROOT)
    bad = [r["path"] for r in local_rows if not r["ok"]]
    if bad:
        print(json.dumps({"ok": False, "error": "local files differ from the manifest; refusing to push", "files": bad}))
        return 1
    files = [e["path"] for e in load_manifest(REPO_ROOT)["files"]]
    report = {"destination": args.target_dir or f"{args.ssh_alias}:{args.remote_root}", "files": []}
    if args.target_dir:
        root = Path(args.target_dir)
        for rel in files:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / rel, root / rel)
        rows = check_runtime_dependencies(root) if (root / "configs/itsm/runtime_dependencies.json").exists() else None
        if rows is None:  # destination without the manifest: compare against ours
            rows = [{"path": rel, "ok": sha256_file(root / rel) == e["sha256"]} for rel, e in ((e["path"], e) for e in load_manifest(REPO_ROOT)["files"])]
    else:
        quoted = " ".join(shlex.quote(str(Path(args.remote_root) / Path(rel).parent)) for rel in files)
        made = _ssh(args.ssh_alias, args.ssh_config, f"mkdir -p {quoted}")
        if made.returncode != 0:
            print(json.dumps({"ok": False, "error": f"ssh mkdir failed: {made.stderr[-300:]}"}))
            return 1
        for rel in files:
            sent = subprocess.run(["scp", "-F", args.ssh_config, "-q", str(REPO_ROOT / rel), f"{args.ssh_alias}:{Path(args.remote_root) / rel}"],
                                  capture_output=True, text=True, timeout=300)
            if sent.returncode != 0:
                print(json.dumps({"ok": False, "error": f"scp {rel} failed: {sent.stderr[-300:]}"}))
                return 1
        listed = " ".join(shlex.quote(str(Path(args.remote_root) / rel)) for rel in files)
        remote = _ssh(args.ssh_alias, args.ssh_config, f"sha256sum {listed}")
        remote_sha = {Path(line.split()[1]).as_posix(): line.split()[0] for line in remote.stdout.splitlines() if len(line.split()) == 2}
        expected = {e["path"]: e["sha256"] for e in load_manifest(REPO_ROOT)["files"]}
        rows = [{"path": rel, "ok": remote_sha.get((Path(args.remote_root) / rel).as_posix()) == expected[rel]} for rel in files]
    report["files"] = rows
    report["ok"] = all(r["ok"] for r in rows)
    print(json.dumps(report))
    return 0 if report["ok"] else 1


def cmd_verify(args) -> int:
    root = Path(args.root) if args.root else REPO_ROOT
    rows = check_runtime_dependencies(root)
    print(json.dumps({"root": str(root), "ok": all(r["ok"] for r in rows), "files": rows}))
    return 0 if all(r["ok"] for r in rows) else 1


def cmd_preflight(args) -> int:
    """Everything a rollout needs before its first sample: deps, frozen data, usage validation, task gates."""
    report = {"repo_root": str(REPO_ROOT)}
    try:
        report["dependencies"] = verify_runtime_dependencies()
        import agent_rl

        from agent_rl.data import itsm_task_sets
        from agent_rl.rollout.itsm_agent_loop import ItsmAgentLoopSettings, allowed_task_ids

        report["agent_rl_from"] = agent_rl.__file__
        grpo = sorted(allowed_task_ids(ItsmAgentLoopSettings(purpose="grpo")))
        dev = sorted(allowed_task_ids(ItsmAgentLoopSettings(purpose="dev")))
        for task_id in grpo + dev:
            itsm_task_sets.resolve_task(task_id, split_version="itsm_split_v3")
        report |= {"ok": True, "grpo_trainable": len(grpo), "dev": len(dev)}
    except RuntimeDependencyError as error:
        report |= {"ok": False, "error": f"RuntimeDependencyError: {error}"}
    print(json.dumps(report, default=str))
    return 0 if report["ok"] else 1


def cmd_simulate_remote(args) -> int:
    """What a projtool code sync leaves on the remote: the worktree minus the sync-ignored roots."""
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    ignored = {r.lstrip("/") for r in load_manifest(REPO_ROOT)["sync_ignored_roots"]["roots"]}
    created = []
    for entry in sorted(REPO_ROOT.iterdir()):
        if entry.name == ".git" or entry.name in ignored or entry.name in SYNC_NOISE:
            continue
        if entry.name in COPIED_TREES:
            shutil.copytree(entry, out / entry.name, ignore=shutil.ignore_patterns(*SYNC_NOISE), symlinks=True)
        else:
            os.symlink(entry, out / entry.name)
        created.append(entry.name)
    print(json.dumps({"out": str(out), "entries": created, "excluded": sorted(ignored)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    push = sub.add_parser("push")
    target = push.add_mutually_exclusive_group(required=True)
    target.add_argument("--ssh-alias")
    target.add_argument("--target-dir")
    push.add_argument("--remote-root", default="/workspace")
    push.add_argument("--ssh-config", default=str(Path.home() / ".ssh" / "proj-tool.config"))
    verify = sub.add_parser("verify")
    verify.add_argument("--root")
    sub.add_parser("preflight")
    simulate = sub.add_parser("simulate-remote")
    simulate.add_argument("--out", required=True)
    args = parser.parse_args()
    return {"push": cmd_push, "verify": cmd_verify, "preflight": cmd_preflight, "simulate-remote": cmd_simulate_remote}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
