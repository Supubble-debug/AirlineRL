"""Build the ITSM split v3 manifest.

Usage (from the repo root):
    python scripts/data/build_itsm_split_v3.py [--check]

Writes configs/data/itsm_split_v3.json from the frozen v2 manifest, Bench2 data
and the accepted 6D-2 / 6D-3 / S0-R1 tasks. v1 and v2 are not touched.
``--check`` rebuilds in memory and fails if the committed file differs.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent_rl.data.itsm_split_v3 import DEFAULT_V3_PATH, build_manifest_v3, validate_manifest_v3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest_v3()
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = DEFAULT_V3_PATH.exists() and DEFAULT_V3_PATH.read_text(encoding="utf-8") == text
        print("split v3 manifest is up to date" if same else f"out of date: {DEFAULT_V3_PATH}")
        return 0 if same else 1
    DEFAULT_V3_PATH.write_text(text, encoding="utf-8")
    validate_manifest_v3(json.loads(text))
    print(json.dumps({"sizes": manifest["sizes"], "family_changes": manifest["family_changes"], "isolation": manifest["isolation"]["checks"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
