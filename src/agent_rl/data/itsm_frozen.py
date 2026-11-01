"""Frozen data artifacts and their full SHA-256 (freeze accepted 2026-10-31).

The frozen files are not rewritten to change their ``status`` text: the record
of the freeze lives here and in docs/itsm_migration_progress.md. Any later
change of a task's usage eligibility is a new usage version; trajectory reviews
are stored in their own files and cite ``FROZEN[...]["sha256"]``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FREEZE_COMMIT = "70e6a3b451149bb24ae0eba18fc544705d4c323a"
FREEZE_DECISION = {
    "date": "2026-10-31",
    "user": "v3 manifest 与 itsm_usage_v1 冻结验收通过。冻结依据为提交 70e6a3b 的文件内容及完整 SHA256。",
}
FROZEN = {
    "itsm_split_v3": {
        "path": "configs/data/itsm_split_v3.json",
        "sha256": "9326167dfee696a809f685f178284e23e6e8f3402559d4b47650c4fcf85b5bc0",
    },
    "itsm_usage_v1": {
        "path": "configs/data/itsm_usage_v1.json",
        "sha256": "87f15baf96a5dbeec6a8c07ca0952edafd86667d18dfa01d63f066d35934fba7",
    },
}


def verify_frozen() -> None:
    """Raise if a frozen file differs from the bytes accepted at the freeze."""
    for name, entry in FROZEN.items():
        actual = hashlib.sha256((REPO_ROOT / entry["path"]).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"{name} differs from its frozen content ({actual} != {entry['sha256']})")
