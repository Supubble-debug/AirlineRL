"""Lossless alternative serializations of the tool-schema block (step 6 measurement).

Each variant changes only formatting: every tool, parameter, constraint and
description is kept, and ``parse`` of the rendered block must return a list
equal to the original schemas. Used for an offline token comparison only; no
variant is wired into the rollout prompt.

``template_default`` is exactly what Qwen3's chat template emits for
``tools=...`` (one ``tojson`` line per tool, i.e. ``json.dumps`` with default
separators); the measurement script checks that byte for byte.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class Serialization:
    name: str
    description: str
    render: Callable[[list[dict[str, Any]]], str]
    parse: Callable[[str], list[dict[str, Any]]]


def _json_lines(separators: tuple[str, str] | None) -> Callable[[list[dict[str, Any]]], str]:
    def render(tools: list[dict[str, Any]]) -> str:
        return "\n".join(json.dumps(tool, ensure_ascii=False, separators=separators) for tool in tools)

    return render


def _parse_json_lines(block: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in block.split("\n")]


def _render_yaml(tools: list[dict[str, Any]]) -> str:
    return "---\n".join(
        yaml.safe_dump(tool, sort_keys=False, allow_unicode=True, width=1_000_000) for tool in tools
    ).rstrip("\n")


def _parse_yaml(block: str) -> list[dict[str, Any]]:
    return list(yaml.safe_load_all(block))


SERIALIZATIONS = {
    s.name: s
    for s in (
        Serialization(
            "template_default",
            "Qwen3 chat template tojson: json.dumps, separators ', ' and ': ' (current)",
            _json_lines(None),
            _parse_json_lines,
        ),
        Serialization(
            "json_compact",
            "json.dumps with separators ',' and ':' (whitespace between tokens removed)",
            _json_lines((",", ":")),
            _parse_json_lines,
        ),
        Serialization(
            "yaml_block",
            "YAML block style, one document per tool (not the JSON format Qwen3 tool use is trained on)",
            _render_yaml,
            _parse_yaml,
        ),
    )
}


def roundtrip_equal(name: str, tools: list[dict[str, Any]]) -> bool:
    serialization = SERIALIZATIONS[name]
    return serialization.parse(serialization.render(tools)) == tools
