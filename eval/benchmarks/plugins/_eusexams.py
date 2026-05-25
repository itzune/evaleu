#!/usr/bin/env python3
"""Builder plugin for LatxaEval EusExams.

Merges all EU configs from HiTZ/EusExams, sanitizes rows, caches locally.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, load_dataset

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


def _bank_path() -> Path:
    return _CACHE_DIR / "eusexams_eu_bank.json"


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    cands = row.get("candidates") or []
    if not cands:
        return None
    ans = row.get("answer")
    if ans is None:
        return None
    try:
        ans_i = int(ans)
    except (ValueError, TypeError):
        return None
    if not (0 <= ans_i < len(cands)):
        return None
    return {"cfg": row["cfg"], "id": str(row.get("id")), "question": row.get("question", ""),
            "candidates": cands, "answer": ans_i}


def _load_or_build_bank() -> list[dict[str, Any]]:
    bp = _bank_path()
    if bp.exists():
        cached = json.loads(bp.read_text(encoding="utf-8"))
        return [x for x in (_sanitize_row(r) for r in cached) if x is not None]

    cfgs = [c for c in get_dataset_config_names("HiTZ/EusExams") if c.startswith("eu_")]
    bank: list[dict[str, Any]] = []
    for cfg in cfgs:
        ds = load_dataset("HiTZ/EusExams", cfg, split="test")
        for row in ds:
            clean = _sanitize_row({
                "cfg": cfg,
                "id": str(row.get("id")),
                "question": row.get("question", ""),
                "candidates": row.get("candidates", []),
                "answer": row.get("answer"),
            })
            if clean is not None:
                bank.append(clean)

    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    return bank


def build_items(limit: int, seed: int) -> list[dict[str, Any]]:
    bank = _load_or_build_bank()
    rng = random.Random(seed)
    idxs = list(range(len(bank)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items: list[dict[str, Any]] = []
    for i in idxs:
        row = bank[int(i)]
        cands = row.get("candidates", [])
        if not cands:
            continue
        letters = [chr(ord("A") + j) for j in range(len(cands))]
        opts = "\n".join(f"{letters[j]}) {cands[j]}" for j in range(len(cands)))
        prompt = (
            "Aukeratu aukera zuzena (A/B/C/D... edo 1/2/3/4... bakarrik).\n"
            f"Galdera: {row['question']}\n{opts}\n"
            "Erantzuna:"
        )
        items.append({
            "bench": "LatxaEval_eusexams",
            "id": f"latxa_eusexams_{row.get('cfg')}_{row.get('id')}",
            "prompt": prompt,
            "gold": int(row.get("answer", 0)),
            "label_names": letters,
            "meta": {"candidates": cands, "config": row.get("cfg")},
        })
    return items
