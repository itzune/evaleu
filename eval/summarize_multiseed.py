#!/usr/bin/env python3
"""Summarize multi-seed eval results.

Read-modify-write: loads the existing eval/summary.json (if present),
patches only model entries that have local *_seed*.json files, and writes
back atomically via a temp file + os.replace().  All other model entries
are preserved verbatim.

Usage:
    python eval/summarize_multiseed.py [--input-dir eval] [--out eval/summary.json]
"""

import argparse
import json
import logging
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = str(EVAL_DIR)
DEFAULT_OUT_PATH = str(EVAL_DIR / "summary.json")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("summarize_multiseed")
_LOGGER_CONFIGURED = False


def _init_logging() -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return
    _LOGGER_CONFIGURED = True
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list) -> float:
    """Sample standard deviation (ddof=1)."""
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
_SEED_RE = re.compile(r"_seed(\d+)\.json$")


def _parse_seed(name: str) -> int | None:
    m = _SEED_RE.search(name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Seed-file loading & validation
# ---------------------------------------------------------------------------
def _load_seed_rows(input_dir: Path) -> list[dict]:
    """Scan *json files under *input_dir*, keep only valid seed files.

    A file is valid if:
      - Its filename matches *_seed<integer>.json
      - It contains a ``model`` field (str)
      - It contains an ``overall_accuracy`` field (numeric)
    Malformed files are skipped with a warning.
    """
    rows: list[dict] = []
    for p in sorted(input_dir.glob("*.json")):
        seed = _parse_seed(p.name)
        if seed is None:
            continue  # not a seed file (e.g. summary.json itself)

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable seed file %s: %s", p.name, exc)
            continue

        model = data.get("model")
        overall = data.get("overall_accuracy")
        if not isinstance(model, str) or not isinstance(overall, (int, float)):
            logger.warning(
                "Skipping %s: missing or invalid 'model' (str) / 'overall_accuracy' (numeric)",
                p.name,
            )
            continue

        rows.append({
            "file": str(p),
            "model": model,
            "seed": seed,
            "overall": float(overall),
            "by_benchmark": data.get("by_benchmark", {}),
        })

    logger.info("Loaded %d valid seed rows from %s", len(rows), input_dir)
    return rows


# ---------------------------------------------------------------------------
# Summary load / save
# ---------------------------------------------------------------------------
def _load_existing_summary(out_path: Path) -> dict:
    """Load existing summary.json, returning {'models': {}, 'n_runs': 0} on failure."""
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        logger.info("No existing summary at %s — starting fresh.", out_path)
        return {"models": {}, "n_runs": 0}
    except json.JSONDecodeError as exc:
        logger.warning("Corrupt summary at %s (%s) — starting fresh.", out_path, exc)
        return {"models": {}, "n_runs": 0}

    if not isinstance(existing, dict) or not isinstance(existing.get("models"), dict):
        logger.warning("Unexpected summary schema in %s — starting fresh.", out_path)
        return {"models": {}, "n_runs": 0}

    n_models = len(existing["models"])
    logger.info(
        "Loaded existing summary with %d model(s) and %d total runs from %s.",
        n_models,
        existing.get("n_runs", 0),
        out_path,
    )
    return existing


def _atomic_write_summary(data: dict, out_path: Path) -> None:
    """Write *data* atomically via a temp file + os.replace."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # tempfile.NamedTemporaryFile with delete=False so we can call os.replace
    fd, tmp_name = tempfile.mkstemp(
        suffix=".json", prefix=".summary_", dir=str(out_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, str(out_path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Per-model aggregation
# ---------------------------------------------------------------------------
def _aggregate_model(rs: list[dict]) -> dict:
    """Aggregate a list of seed rows for a single model.

    Returns a dict with keys:
      n_seeds, seeds, overall_mean, overall_std, runs, benchmarks.
    """
    rs = sorted(rs, key=lambda x: (x["seed"] if x["seed"] is not None else -1))
    ov = [x["overall"] for x in rs]

    # Per-benchmark metric collection
    bench_metric_vals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for x in rs:
        for b, bm in x.get("by_benchmark", {}).items():
            for k, v in bm.items():
                if k == "n":
                    continue
                try:
                    bench_metric_vals[b][k].append(float(v))
                except (TypeError, ValueError):
                    logger.warning(
                        "Non-numeric metric %s=%r in benchmark %s for model %s — skipping.",
                        k, v, b, rs[0]["model"],
                    )

    benchmarks = {}
    for b, mk in bench_metric_vals.items():
        entry: dict = {}
        for k, vals in mk.items():
            entry[f"{k}_mean"] = _mean(vals)
            entry[f"{k}_std"] = _std(vals)
        first_vals = next(iter(mk.values()), [])
        entry["n"] = len(first_vals)
        benchmarks[b] = entry

    seeds = sorted(x["seed"] for x in rs if x["seed"] is not None)

    return {
        "n_seeds": len(rs),
        "seeds": seeds,
        "overall_mean": _mean(ov),
        "overall_std": _std(ov),
        "runs": [
            {
                "seed": x["seed"],
                "overall": x["overall"],
                "by_benchmark": x["by_benchmark"],
                "file": x["file"],
            }
            for x in rs
        ],
        "benchmarks": benchmarks,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def _process(input_dir: Path, out_path: Path) -> None:
    _init_logging()

    # 1. Load existing shared state
    existing = _load_existing_summary(out_path)
    models: dict = existing.get("models", {}) or {}

    # 2. Load local seed rows and group by model
    seed_rows = _load_seed_rows(input_dir)
    local_grouped: dict[str, list[dict]] = defaultdict(list)
    for r in seed_rows:
        local_grouped[r["model"]].append(r)

    # 3. Merge: replace/insert only models with local seed files
    added = []
    updated = []
    preserved = []
    for model_name, rs in local_grouped.items():
        existed = model_name in models
        models[model_name] = _aggregate_model(rs)
        if existed:
            updated.append(model_name)
        else:
            added.append(model_name)

    for model_name in models:
        if model_name not in local_grouped:
            preserved.append(model_name)

    logger.info(
        "Added %d model(s): %s", len(added), ", ".join(added) if added else "—"
    )
    logger.info(
        "Updated %d model(s): %s", len(updated), ", ".join(updated) if updated else "—"
    )
    logger.info(
        "Preserved %d model(s): %s",
        len(preserved),
        ", ".join(preserved) if preserved else "—",
    )

    # 4. Recalculate n_runs across all models
    total_seeds = sum(m.get("n_seeds", len(m.get("runs", []))) for m in models.values())
    summary = {"models": models, "n_runs": total_seeds}

    # 5. Atomic write
    _atomic_write_summary(summary, out_path)

    # 6. Console report (via logging for consistency, but also print the
    #    human-readable table that existing callers expect)
    logger.info("Final totals: %d model(s), %d total seed runs.", len(models), total_seeds)

    # Human-readable table (preserve existing CLI behaviour)
    print()  # blank line before table
    print(f"Runs: {total_seeds}")
    for model, m in sorted(
        models.items(), key=lambda kv: kv[1]["overall_mean"], reverse=True
    ):
        print(
            f"{model:18s} "
            f"overall={m['overall_mean'] * 100:5.2f}% ± {m['overall_std'] * 100:4.2f}"
        )
        for b, bm in m.get("benchmarks", {}).items():
            parts = []
            if "accuracy_mean" in bm:
                parts.append(
                    f"acc={bm['accuracy_mean'] * 100:5.2f}% ± {bm['accuracy_std'] * 100:4.2f}"
                )
            if "chrf_mean" in bm:
                parts.append(
                    f"chrF={bm['chrf_mean']:.1f} ± {bm['chrf_std']:.1f}"
                )
            if "bleu_mean" in bm:
                parts.append(
                    f"BLEU={bm['bleu_mean']:.1f} ± {bm['bleu_std']:.1f}"
                )
            if not parts:
                parts.append("—")
            print(f"  - {b:16s} {' '.join(parts)} (n={bm['n']})")

    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Summarize multi-seed eval results")
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.out)
    _process(in_dir, out_path)


if __name__ == "__main__":
    main()
