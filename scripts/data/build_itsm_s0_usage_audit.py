"""Build the step S0 joint SFT/GRPO usage audit (offline).

Usage (from the repo root):
    python scripts/data/build_itsm_s0_usage_audit.py [--check]

Writes docs/itsm_stepS0/usage_audit.json: core requirements and key-fact
negatives of the seven derived train tasks (6D-3-R1), proposed GRPO/SFT use per
task, the SFT u GRPO isolation audit against dev/held-out, and the excluded and
deferred sources. Gold and edited replays only; no tokenizer, model, LLM or GPU.
``--check`` rebuilds in memory and fails if the committed file differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_rl.data.itsm_s0_usage import build_s0_audit

OUT = Path(__file__).resolve().parents[2] / "docs" / "itsm_stepS0" / "usage_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build_s0_audit(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print("S0 usage audit is up to date" if same else f"out of date: {OUT}")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
