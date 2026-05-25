#!/usr/bin/env python3
"""Scoring plugin for BasqueGLUE WiC (word-in-context).

Labels are false / true.
Model answers may be Basque words (berdin, desberdin, bai, ez).
"""
from __future__ import annotations

import re
from typing import Any


def score_item(item: dict[str, Any], answer: str, label_names: list[str]) -> tuple[int | None, bool]:
    t = (answer or "").strip().lower()
    t = re.sub(r"\s+", " ", t)

    if re.search(r"\b(true|berdin(ak?)?|bai|same)\b", t):
        pred = 1
    elif re.search(r"\b(false|desberdin(ak?)?|ezberdin(ak?)?|different)\b", t):
        pred = 0
    else:
        return None, False  # signal: use generic scorer

    ok = pred == int(item["gold"])
    return pred, ok
