"""Explicit runtime dependencies outside the code-sync set (deployment manifest, after step 7B).

projtool's code-sync session does not copy ``/docs`` (and a few other roots) to the remote,
but usage-registry validation -- run on every usage load, i.e. before any sampling -- reads
``docs/itsm_stepS0/usage_audit.json``. ``configs/itsm/runtime_dependencies.json`` lists every
such file with its SHA256 at its ORIGINAL path; ``scripts/itsm/runtime_deps.py push`` copies
them to a deployment and verifies them there.

Before a listed file is used, ``require_dependency`` / ``verify_runtime_dependencies`` check
it against the manifest: a missing or modified file raises ``RuntimeDependencyError``. The
check never falls back, skips or repairs anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_RELPATH = "configs/itsm/runtime_dependencies.json"


class RuntimeDependencyError(RuntimeError):
    """A runtime dependency is missing or differs from the manifest; nothing may run on it."""


def load_manifest(root: Path | None = None) -> dict[str, Any]:
    path = (root or REPO_ROOT) / MANIFEST_RELPATH
    if not path.exists():
        raise RuntimeDependencyError(f"runtime dependency manifest {path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def check_runtime_dependencies(root: Path | None = None) -> list[dict[str, Any]]:
    """One row per manifest file: path, expected, actual (None if missing), ok."""
    root = root or REPO_ROOT
    rows = []
    for entry in load_manifest(root)["files"]:
        path = root / entry["path"]
        actual = sha256_file(path) if path.is_file() else None
        rows.append({"path": entry["path"], "expected_sha256": entry["sha256"], "actual_sha256": actual,
                     "ok": actual == entry["sha256"]})
    return rows


def verify_runtime_dependencies(root: Path | None = None) -> list[dict[str, Any]]:
    rows = check_runtime_dependencies(root)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        detail = "; ".join(f"{r['path']}: {'missing' if r['actual_sha256'] is None else 'sha256 ' + r['actual_sha256'][:12] + '… != manifest'}" for r in bad)
        raise RuntimeDependencyError(
            f"runtime dependencies not deployed as committed ({detail}); push them with "
            "scripts/itsm/runtime_deps.py push before launching"
        )
    return rows


def require_dependency(relpath: str, root: Path | None = None) -> Path:
    """The verified path of one listed dependency (raises if unlisted, missing or modified)."""
    root = root or REPO_ROOT
    entries = {e["path"]: e for e in load_manifest(root)["files"]}
    if relpath not in entries:
        raise RuntimeDependencyError(f"{relpath} is read at run time but not listed in {MANIFEST_RELPATH}")
    path = root / relpath
    if not path.is_file():
        raise RuntimeDependencyError(f"runtime dependency {relpath} is missing (push it with scripts/itsm/runtime_deps.py push)")
    actual = sha256_file(path)
    if actual != entries[relpath]["sha256"]:
        raise RuntimeDependencyError(f"runtime dependency {relpath} differs from the manifest (sha256 {actual[:12]}…)")
    return path
