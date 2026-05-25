#!/usr/bin/env python3
"""Plugin for FLORES-200 translation benchmarks.

Handles building prompts and computing chrF / BLEU scores via sacrebleu.
"""
from __future__ import annotations

import random
from typing import Any

import sacrebleu
from datasets import load_dataset


def build_items(limit: int, seed: int, spec: dict | None = None) -> list[dict[str, Any]]:
    """Build translation evaluation items.

    Receives the benchmark spec via the optional `spec` kwarg, which
    the engine passes since the metric-agnostic refactor.
    """
    if spec is None:
        raise RuntimeError("build_items() requires spec= to be passed by the engine")

    bench_id = spec["id"]
    ds_cfg = spec["dataset"]
    prompt_template = spec["prompt"]["template"]
    swap = ds_cfg.get("swap_source_target", False)

    # Resolve source/target configs
    source_cfg_name = ds_cfg["config"]
    target_cfg_name = ds_cfg["target_config"]
    if swap:
        source_cfg_name, target_cfg_name = target_cfg_name, source_cfg_name

    # Load datasets
    ds_source = load_dataset(
        ds_cfg["path"], source_cfg_name, trust_remote_code=True, split=ds_cfg["split"]
    )
    ds_target = load_dataset(
        ds_cfg["path"], target_cfg_name, trust_remote_code=True, split=ds_cfg["split"]
    )

    # Shuffle and select
    rng = random.Random(seed)
    n_total = len(ds_source)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    id_field = ds_cfg.get("id_field", "id")
    items: list[dict[str, Any]] = []
    for i in idxs:
        source_text = ds_source[int(i)][ds_cfg["source_field"]]
        target_text = ds_target[int(i)][ds_cfg["target_field"]]
        item_id = str(ds_source[int(i)].get(id_field, i))

        prompt = prompt_template.format(source=source_text)

        items.append({
            "bench": bench_id,
            "id": f"{bench_id.lower()}_{item_id}",
            "prompt": prompt,
            "gold": target_text,  # string reference translation
            "label_names": [],
            "meta": {
                "source": source_text,
                "reference": target_text,
            },
        })

    return items


def score_item(item: dict[str, Any], answer: str, label_names: list[str]) -> dict:
    """Compute chrF and BLEU scores for a translation item.

    Uses sacrebleu sentence-level scoring for per-item metrics.
    """
    reference = item.get("gold") or item.get("meta", {}).get("reference", "")
    hypothesis = (answer or "").strip()

    if not hypothesis or not reference:
        return {"chrf": 0.0, "bleu": 0.0}

    # Sentence-level chrF (chrF++ with word_order=2)
    chrf_score = sacrebleu.sentence_chrf(hypothesis=hypothesis, references=[reference])

    # Sentence-level BLEU  
    bleu_score = sacrebleu.sentence_bleu(hypothesis=hypothesis, references=[reference])

    return {
        "chrf": chrf_score.score,
        "bleu": bleu_score.score,
    }
