#!/usr/bin/env python3
"""Builder plugin for Belebele EU — Basque reading comprehension.

Loads facebook/belebele (config eus_Latn), assembles candidates from the four
individual mc_answer fields, and converts the 1-indexed correct_answer_num to
0-indexed gold.  No score_item override needed — the generic MC scorer handles
letter/index extraction and candidate text matching.
"""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset


def build_items(limit: int, seed: int, spec: dict | None = None) -> list[dict[str, Any]]:
    bench_id = spec["id"] if (spec and "id" in spec) else "Belebele_eu"
    prompt_template = (
        spec["prompt"]["template"]
        if (spec and isinstance(spec.get("prompt"), dict) and "template" in spec["prompt"])
        else "Aukeratu aukera zuzena (A/B/C/D bakarrik).\nTestua: {context}\nGaldera: {question}\n{options}\nErantzuna:"
    )

    ds = load_dataset("facebook/belebele", "eus_Latn", split="test")

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items: list[dict[str, Any]] = []
    for i in idxs:
        row = ds[int(i)]

        candidates = [
            row["mc_answer1"],
            row["mc_answer2"],
            row["mc_answer3"],
            row["mc_answer4"],
        ]
        if not any(candidates):
            continue

        # correct_answer_num is 1-indexed string ("1".."4") → 0-indexed int
        gold = int(row["correct_answer_num"]) - 1
        if not (0 <= gold < len(candidates)):
            continue

        letters = [chr(ord("A") + j) for j in range(len(candidates))]
        opts = "\n".join(f"{letters[j]}) {candidates[j]}" for j in range(len(candidates)))
        context = row.get("flores_passage", "")
        prompt = prompt_template.format(
            question=row["question"], options=opts, context=context
        )

        items.append({
            "bench": bench_id,
            "id": f"{bench_id.lower()}_{i}",
            "prompt": prompt,
            "gold": gold,
            "label_names": letters,
            "meta": {"candidates": candidates},
        })

    return items
