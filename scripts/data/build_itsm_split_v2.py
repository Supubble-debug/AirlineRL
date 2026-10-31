"""Build the ITSM split v2 manifest (step 6C).

Usage (from the repo root):
    python scripts/data/build_itsm_split_v2.py [--check]

Writes configs/data/itsm_split_v2.json from the frozen v1 manifest, Bench2 data
and the three accepted tasks of configs/data/itsm_eops_pilot_v1.json. v1 is not
touched. ``--check`` rebuilds in memory and fails if the committed file differs.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent_rl.data.itsm_split_v2 import DEFAULT_V2_PATH, build_manifest_v2, validate_manifest_v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest_v2()
    validate_manifest_v2(manifest)
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = DEFAULT_V2_PATH.exists() and DEFAULT_V2_PATH.read_text(encoding="utf-8") == text
        print("split v2 manifest is up to date" if same else f"out of date: {DEFAULT_V2_PATH}")
        return 0 if same else 1
    DEFAULT_V2_PATH.write_text(text, encoding="utf-8")
    print(json.dumps({"sizes": manifest["sizes"], "family_changes": manifest["family_changes"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
