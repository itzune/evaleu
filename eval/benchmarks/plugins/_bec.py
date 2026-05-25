#!/usr/bin/env python3
"""Scoring plugin for BasqueGLUE BEC (sentiment classification).

Labels are N (negative), NEU (neutral), P (positive).
Model answers may be full words (negatiboa, positiboa, neutrala).
"""
from __future__ import annotations

import re
from typing import Any


def score_item(item: dict[str, Any], answer: str, label_names: list[str]) -> dict:
    t = (answer or "").strip().lower()
    t = re.sub(r"\s+", " ", t)

    pred = None
    if re.search(r"\bneu(tral(ak?)?)?\b", t):
        pred = 1
    elif re.search(r"\bn(egatibo(ak?)?|egative)?\b", t):
        pred = 0
    elif re.search(r"\bp(ositibo(ak?)?|ositive)?\b", t):
        pred = 2

    if pred is not None:
        ok = pred == int(item["gold"])
        return {"pred_label": pred, "accuracy": 1.0 if ok else 0.0, "coverage": 1.0}

    # Fallback: signal to use generic scorer
    return {}
