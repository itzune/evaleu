#!/usr/bin/env python3
"""Scoring plugin for BasqueGLUE Intent (12-class classification).

Prompt asks for numbers 0-11. Model may return a number or a label string.
"""
from __future__ import annotations

import re
from typing import Any


def score_item(item: dict[str, Any], answer: str, label_names: list[str]) -> dict:
    t = (answer or "").strip()

    # Try number extraction first (preferred, prompt asks for 0-11).
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(label_names):
            ok = idx == int(item["gold"])
            return {"pred_label": idx, "accuracy": 1.0 if ok else 0.0, "coverage": 1.0}

    # Fallback: signal to use generic scorer
    return {}
