#!/usr/bin/env python3
"""Validate all benchmark JSON specs in eval/benchmarks/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
PLUGINS_DIR = BENCHMARKS_DIR / "plugins"

REQUIRED_FIELDS = ["id", "description", "prompt.template", "scoring.type", "default_limit"]


def _nested_get(d: dict, dotted: str):
    """Get a nested dict value by dotted key path, e.g. 'prompt.template'."""
    parts = dotted.split(".")
    for p in parts:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def validate_spec(spec: dict, path: str) -> list[str]:
    """Return a list of error strings for a single spec. Empty list = valid."""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if _nested_get(spec, field) is None:
            errors.append(f"missing required field '{field}'")

    if "id" in spec:
        if not isinstance(spec["id"], str) or not spec["id"].strip():
            errors.append("'id' must be a non-empty string")

    if "default_limit" in spec:
        if not isinstance(spec["default_limit"], int) or spec["default_limit"] < 1:
            errors.append("'default_limit' must be a positive integer")

    if "scoring" in spec and isinstance(spec["scoring"], dict):
        scoring_type = spec["scoring"].get("type", "")
        valid_types = {"multiple_choice", "label_match", "translation"}
        if scoring_type not in valid_types:
            errors.append(
                f"scoring.type '{scoring_type}' not one of {sorted(valid_types)}"
            )

    if "plugin" in spec:
        plugin_name = spec["plugin"]
        if not isinstance(plugin_name, str) or not plugin_name.strip():
            errors.append("'plugin' must be a non-empty string if present")
        else:
            plugin_path = PLUGINS_DIR / f"_{plugin_name}.py"
            if not plugin_path.is_file():
                errors.append(
                    f"plugin '_{plugin_name}.py' does not exist at {plugin_path}"
                )

    return errors


def main() -> int:
    json_files = sorted(BENCHMARKS_DIR.glob("*.json"))
    if not json_files:
        print("No benchmark JSON files found in", BENCHMARKS_DIR)
        return 1

    any_error = False
    for f in json_files:
        try:
            spec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"FAIL {f.name}: invalid JSON — {e}")
            any_error = True
            continue

        errors = validate_spec(spec, str(f))
        if errors:
            any_error = True
            print(f"FAIL {f.name}:")
            for err in errors:
                print(f"  - {err}")
        else:
            plugin_note = f" [plugin: {spec['plugin']}]" if spec.get("plugin") else ""
            print(f"OK   {f.name}: {spec['id']} (limit={spec['default_limit']}){plugin_note}")

    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
