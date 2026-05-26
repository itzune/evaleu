#!/usr/bin/env python3
"""Plugin for MGSM EU — grade school math reasoning in Basque.

Generative benchmark: models produce free-form chain-of-thought reasoning
followed by a numeric answer.  Answer extraction uses a two-stage regex
pipeline (primary: "Erantzuna <N> da", fallback: last integer in response),
consistent with the EleutherAI lm-evaluation-harness mgsm_native_cot_eu task.

Design note — system prompt conflict:
  The evaleu engine sends the system message "Erantzun bakarra eman. Ez
  azaldu arrazoiketa." to all models, which suppresses chain-of-thought.
  This hurts math accuracy.  To work around this without modifying the
  engine, we bake 3 few-shot CoT examples directly into the user prompt.
  When the model sees the in-context format it will follow it even under
  the restrictive system prompt.  These examples come from the dataset's
  train split (first 3 of 8 available demonstrations).
"""

from __future__ import annotations

import random
import re
from typing import Any

from datasets import load_dataset

# ── Few-shot prefix ─────────────────────────────────────────────────────────
# First 3 train-split items from HiTZ/MGSM-eu (config "mgsm"), used in-order
# so the model learns the expected CoT answer format.  The questions here
# already include the "Galdera: " prefix as stored in the dataset so we
# render them verbatim.
_FEW_SHOT_PREFIX = (
    "Galdera: Rogerrek 5 teniseko pilota ditu. Teniseko piloten 2 pote gehiago erosi ditu. Pote bakoitzak 3 teniseko pilota ditu. Zenbat teniseko pilota ditu orain?\n"
    "Erantzuna urratsez urrats: Roger 5 pilotarekin hasi zen. 2 pote, bakoitza 3 teniseko pilotakoa, 6 teniseko pilota dira. 5 + 6 = 11. Erantzuna 11 da.\n"
    "\n"
    "Galdera: Bederatzi ordenagailu zeuden zerbitzari-gelan. Beste bost ordenagailu instalatu ziren egun bakoitzean, astelehenetik ostegunera. Zenbat ordenagailu daude orain zerbitzari-gelan?\n"
    "Erantzuna urratsez urrats: 4 egun daude astelehenetik ostegunera. 5 ordenagailu gehitu ziren egunero. Horrek esan nahi du guztira 4 * 5 = 20 ordenagailu gehitu zirela. Hasieran 9 ordenagailu zeuden, beraz orain 9 + 20 = 29 ordenagailu daude. Erantzuna 29 da.\n"
    "\n"
    "Galdera: Leahk 32 txokolate zituen eta bere ahizpak 42 zituen. 35 jan bazituzten, zenbat txokolate dituzte orain guztira?\n"
    "Erantzuna urratsez urrats: Leahk 32 txokolate zituen eta Leahren ahizpak 42 zituen. Horrek esan nahi du hasieran 32 + 42 = 74 txokolate zeudela. 35 jan zituzten. Beraz, guztira oraindik 74 - 35 = 39 txokolate dituzte. Erantzuna 39 da.\n"
    "\n"
    "Galdera: {question}\n"
    "Erantzuna urratsez urrats:"
)

# ── Answer extraction ──────────────────────────────────────────────────────
# Stage 1: look for "Erantzuna <number> da" pattern (lm-harness approach).
_PRIMARY_RE = re.compile(
    r'Erantzuna\s+[\$%]?\s*(-?[0-9]+(?:[ .,][0-9.,]+)?)\s*[\$%]?\s*da',
    re.IGNORECASE,
)

# Stage 2: fallback — grab any integer-like token; take the last one.
_FALLBACK_RE = re.compile(r'-?[0-9]+(?:[.,][0-9]+)*')


def _parse_number(s: str) -> int | None:
    """Normalise a numeric string by stripping thousands separators.

    Handles European-style thousands (276.000), English-style (276,000),
    and plain integers.  Returns None when the cleaned string is not a
    valid integer.
    """
    s = s.strip().replace(" ", "").replace(".", "").replace(",", "")
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


# ── Plugin API ─────────────────────────────────────────────────────────────

def build_items(limit: int, seed: int, spec: dict | None = None) -> list[dict[str, Any]]:
    """Build MGSM EU evaluation items.

    Loads the test split of HiTZ/MGSM-eu (config "mgsm"), shuffles with the
    given seed, and embeds each question into the 3-shot CoT prompt template.
    Each item's ``gold`` is the integer ``answer_number``.
    """
    bench_id = spec["id"] if spec else "MGSM_eu"
    ds = load_dataset("HiTZ/MGSM-eu", "mgsm", split="test")

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items: list[dict[str, Any]] = []
    for i in idxs:
        row = ds[int(i)]
        q = row["question"]
        # The dataset stores questions already prefixed with "Galdera: ".
        # Strip that prefix so the template can supply its own consistent one.
        q_clean = q.removeprefix("Galdera: ").removeprefix("Galdera:").strip()

        prompt = _FEW_SHOT_PREFIX.format(question=q_clean)
        items.append({
            "bench": bench_id,
            "id": f"mgsm_eu_{i}",
            "prompt": prompt,
            "gold": int(row["answer_number"]),
            "label_names": [],  # not a classification task
            "meta": {"answer_number": int(row["answer_number"])},
        })
    return items


def score_item(item: dict[str, Any], answer: str, label_names: list[str]) -> dict:
    """Score an MGSM item by extracting and comparing the numeric answer.

    Uses a two-stage extraction consistent with the lm-harness
    mgsm_native_cot_eu task:
      1. Attempt to match "Erantzuna <N> da" via regex.
      2. Fall back to the last integer-like token in the response.

    Returns a dict with ``pred_number``, ``accuracy`` and ``coverage``
    so the generic aggregator in run_eval.py can include it in the
    overall accuracy mean.
    """
    gold = int(item["gold"])

    # Stage 1 — primary regex
    m = _PRIMARY_RE.search(answer)
    if m:
        pred = _parse_number(m.group(1))
    else:
        # Stage 2 — fallback: last integer-like token in response
        matches = _FALLBACK_RE.findall(answer)
        pred = _parse_number(matches[-1]) if matches else None

    ok = (pred == gold) if pred is not None else False
    return {
        "pred_number": pred,
        "accuracy": 1.0 if ok else 0.0,
        "coverage": 1.0 if pred is not None else 0.0,
    }
