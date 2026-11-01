"""Step 7B: space check, single-copy download and hash verification of the pinned Qwen3-8B.

Usage (cloud):
    python scripts/itsm/model_files_7b.py check-space --dir DIR [--margin-gib 2]
    python scripts/itsm/model_files_7b.py download    --dir DIR
    python scripts/itsm/model_files_7b.py verify      --dir DIR

Expected files, sizes and hashes: configs/itsm/step7b/model_files.json (HF model
info of Qwen/Qwen3-8B @ b968826d). LFS files are checked by SHA256, the others
by git blob SHA1; the small tokenizer/config files must also equal the locally
pinned tokenizer's files. ``download`` uses ``snapshot_download(local_dir=DIR)``
(no second copy in an HF cache) and only the listed files. Exit codes: 0 ok,
1 failure, 4 not enough space.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "configs" / "itsm" / "step7b" / "model_files.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def file_hashes(path: Path) -> tuple[str, str]:
    sha256, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 24):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def missing_bytes(directory: Path, manifest: dict) -> int:
    total = 0
    for name, entry in manifest["files"].items():
        path = directory / name
        if not path.exists() or path.stat().st_size != entry["size"]:
            total += entry["size"]
    return total


def cmd_check_space(args) -> int:
    manifest = load_manifest()
    directory = Path(args.dir)
    probe = directory if directory.exists() else directory.parent
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    need = missing_bytes(directory, manifest)
    margin = int(args.margin_gib * 2**30)
    report = {"dir": str(directory), "free_bytes": free, "missing_bytes": need, "margin_bytes": margin,
              "enough": free >= need + margin}
    print(json.dumps(report))
    return 0 if report["enough"] else 4


def cmd_download(args) -> int:
    from huggingface_hub import snapshot_download

    manifest = load_manifest()
    directory = Path(args.dir)
    started = time.monotonic()
    snapshot_download(repo_id=manifest["repo"], revision=manifest["revision"], local_dir=str(directory),
                      allow_patterns=sorted(manifest["files"]), max_workers=8)
    elapsed = time.monotonic() - started
    print(json.dumps({"download_s": round(elapsed, 1), "bytes": manifest["total_bytes"],
                      "mb_per_s": round(manifest["total_bytes"] / 1e6 / max(elapsed, 1e-9), 1)}))
    return 0


def cmd_verify(args) -> int:
    manifest = load_manifest()
    directory = Path(args.dir)
    names = sorted(manifest["files"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        hashes = dict(zip(names, pool.map(lambda n: file_hashes(directory / n) if (directory / n).exists() else (None, None), names)))
    problems, rows = [], {}
    for name in names:
        entry, (sha256, blob) = manifest["files"][name], hashes[name]
        row = {"sha256": sha256}
        if sha256 is None:
            problems.append(f"{name}: missing")
        elif (directory / name).stat().st_size != entry["size"]:
            problems.append(f"{name}: size differs")
        elif "sha256" in entry and sha256 != entry["sha256"]:
            problems.append(f"{name}: sha256 differs")
        elif "git_blob_sha1" in entry and blob != entry["git_blob_sha1"]:
            problems.append(f"{name}: git blob sha1 differs")
        elif "sha256_local_pinned" in entry and sha256 != entry["sha256_local_pinned"]:
            problems.append(f"{name}: differs from the locally pinned tokenizer file")
        rows[name] = row
    template = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8")).get("chat_template", "")
    template_sha = hashlib.sha256(template.encode()).hexdigest()
    if template_sha != manifest["chat_template_sha256"]:
        problems.append(f"chat template sha256 {template_sha} != pinned")
    extra = sorted(p.name for p in directory.iterdir() if p.is_file() and p.name not in manifest["files"])
    print(json.dumps({"ok": not problems, "problems": problems, "chat_template_sha256": template_sha,
                      "extra_files": extra, "files": rows}, indent=1))
    return 0 if not problems else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check-space", "download", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--dir", required=True)
        if name == "check-space":
            p.add_argument("--margin-gib", type=float, default=2.0)
    args = parser.parse_args()
    return {"check-space": cmd_check_space, "download": cmd_download, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
