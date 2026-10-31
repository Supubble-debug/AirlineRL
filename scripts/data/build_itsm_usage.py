"""Build the ITSM usage registry v1 (SFT/GRPO use of each v3 task).

Usage (from the repo root):
    python scripts/data/build_itsm_usage.py [--check]

Writes configs/data/itsm_usage_v1.json from the v3 manifest and the accepted S0
audit. ``--check`` rebuilds in memory and fails if the committed file differs.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent_rl.data.itsm_usage import DEFAULT_USAGE_PATH, build_usage_registry, validate_usage_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    registry = build_usage_registry()
    text = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = DEFAULT_USAGE_PATH.exists() and DEFAULT_USAGE_PATH.read_text(encoding="utf-8") == text
        print("usage registry is up to date" if same else f"out of date: {DEFAULT_USAGE_PATH}")
        return 0 if same else 1
    DEFAULT_USAGE_PATH.write_text(text, encoding="utf-8")
    validate_usage_registry(json.loads(text))
    print(json.dumps(registry["counts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
