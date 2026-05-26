#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def load_model_cards(root: Path):
    model_cards_path = root / "site" / "model_cards.json"
    return json.loads(model_cards_path.read_text(encoding="utf-8"))


ALL_BENCHMARKS = [
    {
        "id": "EusTrivia",
        "family": "Core",
        "task": "Multiple-choice factual and cultural knowledge in Basque",
        "metric": "Accuracy",
        "labels": "4 options (A/B/C/D)",
    },
    {
        "id": "XNLIeu",
        "family": "Core",
        "task": "Natural language inference in Basque (premise-hypothesis)",
        "metric": "Accuracy",
        "labels": "entailment / neutral / contradiction",
    },
    {
        "id": "BasqueGLUE_qnli",
        "family": "BasqueGLUE",
        "task": "Question-Answer NLI from BasqueGLUE (sentence pair classification)",
        "metric": "Accuracy",
        "labels": "entailment / not_entailment",
    },
    {
        "id": "BasqueGLUE_bec",
        "family": "BasqueGLUE",
        "task": "Sentiment classification from BasqueGLUE BEC",
        "metric": "Accuracy",
        "labels": "N / NEU / P",
    },
    {
        "id": "BasqueGLUE_wic",
        "family": "BasqueGLUE",
        "task": "Word-in-Context disambiguation from BasqueGLUE WiC",
        "metric": "Accuracy",
        "labels": "false / true",
    },
    {
        "id": "BasqueGLUE_intent",
        "family": "BasqueGLUE",
        "task": "Intent classification from BasqueGLUE Intent",
        "metric": "Accuracy",
        "labels": "12 intent classes",
    },
    {
        "id": "LatxaEval_eusexams",
        "family": "LatxaEvalSuite",
        "task": "Professional and domain exams (multiple choice)",
        "metric": "Accuracy",
        "labels": "index of correct choice",
    },
    {
        "id": "LatxaEval_eusproficiency",
        "family": "LatxaEvalSuite",
        "task": "Basque language proficiency questions (multiple choice)",
        "metric": "Accuracy",
        "labels": "index of correct choice",
    },
    {
        "id": "LatxaEval_eusreading",
        "family": "LatxaEvalSuite",
        "task": "Basque reading comprehension questions (multiple choice)",
        "metric": "Accuracy",
        "labels": "index of correct choice",
    },
    {
        "id": "FloresTranslation_eu_en",
        "family": "FLORES",
        "task": "Translation Basque → English",
        "metric": "chrF / BLEU",
        "labels": "free-text translation",
    },
    {
        "id": "FloresTranslation_en_eu",
        "family": "FLORES",
        "task": "Translation English → Basque",
        "metric": "chrF / BLEU",
        "labels": "free-text translation",
    },
    {
        "id": "FloresTranslation_eu_es",
        "family": "FLORES",
        "task": "Translation Basque → Spanish",
        "metric": "chrF / BLEU",
        "labels": "free-text translation",
    },
    {
        "id": "FloresTranslation_es_eu",
        "family": "FLORES",
        "task": "Translation Spanish → Basque",
        "metric": "chrF / BLEU",
        "labels": "free-text translation",
    },
    {
        "id": "MMLU_eu",
        "family": "MMLU",
        "task": "Academic multiple-choice knowledge in Basque (MMLU-style)",
        "metric": "Accuracy",
        "labels": "4 options (A/B/C/D)",
    },
    {
        "id": "BertaQA_eu",
        "family": "BertaQA",
        "task": "Basque cultural and general knowledge trivia (local + global topics)",
        "metric": "Accuracy",
        "labels": "3 options (A/B/C)",
    },
]

BENCH_LABELS = {
    "BasqueGLUE_qnli": "BasqueGLUE-QNLI",
    "BasqueGLUE_bec": "BasqueGLUE-BEC",
    "BasqueGLUE_wic": "BasqueGLUE-WiC",
    "BasqueGLUE_intent": "BasqueGLUE-Intent",
    "LatxaEval_eusexams": "LatxaEval-EusExams",
    "LatxaEval_eusproficiency": "LatxaEval-EusProficiency",
    "LatxaEval_eusreading": "LatxaEval-EusReading",
    "FloresTranslation_eu_en": "FLORES EU→EN",
    "FloresTranslation_en_eu": "FLORES EN→EU",
    "FloresTranslation_eu_es": "FLORES EU→ES",
    "FloresTranslation_es_eu": "FLORES ES→EU",
    "MMLU_eu": "MMLU EU",
    "BertaQA_eu": "BertaQA EU",
}


def mean(xs):
    vals = [float(x) for x in xs if x is not None]
    return (sum(vals) / len(vals)) if vals else 0.0


def family_from_benchmark_defs(benchmark_defs):
    fam = {}
    for b in benchmark_defs:
        fam.setdefault(b["family"], []).append(b["id"])
    return fam


def main():
    ap = argparse.ArgumentParser(description="Build site/data.json from eval summary")
    ap.add_argument(
        "--summary",
        default="eval/summary.json",
        help="Path (relative to repo root or absolute) to summary.json",
    )
    ap.add_argument(
        "--out",
        default="site/data.json",
        help="Output path (relative to repo root or absolute)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = root / summary_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_cards = load_model_cards(root)

    models_map = summary.get("models", {})
    present_benchmark_ids = {
        b
        for s in models_map.values()
        for b in (s.get("benchmarks", {}) or {}).keys()
    }
    benchmark_defs = [b for b in ALL_BENCHMARKS if b["id"] in present_benchmark_ids]
    benchmark_ids = [b["id"] for b in benchmark_defs]
    family_defs = family_from_benchmark_defs(benchmark_defs)
    family_names = list(family_defs.keys())

    rows = []
    all_seed_ids = set()

    for model_id, s in models_map.items():
        meta = model_cards.get(model_id, {
            "display_name": model_id,
            "family": "unknown",
            "params": "unknown",
            "weights_quant": "unknown",
            "kv_cache": "unknown",
            "upstream_model_id": "unknown",
            "release_date_utc": None,
            "release_source_url": None,
            "site_visibility": "published",
        })

        if meta.get("site_visibility", "published") != "published":
            continue

        runs = s.get("runs", [])
        for r in runs:
            if r.get("seed") is not None:
                all_seed_ids.add(int(r["seed"]))

        bench_means = s.get("benchmarks", {})
        by_benchmark = {}
        for b in benchmark_ids:
            bm = bench_means.get(b, {})
            entry = {
                "n": bm.get("n", 80),
                "accuracy": bm.get("accuracy_mean", 0.0),
                "accuracy_std": bm.get("accuracy_std", 0.0),
                "coverage": 1.0,
                "family": next((x["family"] for x in benchmark_defs if x["id"] == b), "Other"),
            }
            # Plumb through any additional metric keys (chrF, BLEU, etc.)
            for key in bm:
                if key.endswith("_mean") and key not in ("accuracy_mean",):
                    base = key[:-5]  # strip _mean
                    entry[base] = bm[key]
                    std_key = f"{base}_std"
                    if std_key in bm:
                        entry[std_key] = bm[std_key]
            by_benchmark[b] = entry

        family_scores = {}
        for fam, fam_bench_ids in family_defs.items():
            fam_vals = [by_benchmark[b]["accuracy"] for b in fam_bench_ids if b in by_benchmark]
            fam_stds = [by_benchmark[b]["accuracy_std"] for b in fam_bench_ids if b in by_benchmark]
            family_scores[fam] = {
                "accuracy": mean(fam_vals),
                "accuracy_std": mean(fam_stds),
                "benchmarks": fam_bench_ids,
            }

        rows.append({
            "model_id": model_id,
            "display_name": meta["display_name"],
            "family": meta["family"],
            "params": meta["params"],
            "weights_quant": meta["weights_quant"],
            "kv_cache": meta["kv_cache"],
            "upstream_model_id": meta.get("upstream_model_id"),
            "release_date_utc": meta.get("release_date_utc"),
            "release_source_url": meta.get("release_source_url"),
            "overall_accuracy": s.get("overall_mean", 0.0),
            "overall_accuracy_std": s.get("overall_std", 0.0),
            "n_items": 80 * len(by_benchmark),
            "n_seeds": len(runs),
            "seed_ids": sorted([r.get("seed") for r in runs if r.get("seed") is not None]),
            "by_benchmark": by_benchmark,
            "by_family": family_scores,
            "source_file": str(summary_path.relative_to(root)),
        })

    rows.sort(key=lambda x: x["overall_accuracy"], reverse=True)

    benchmark_label_list = ", ".join([BENCH_LABELS.get(b["id"], b["id"]) for b in benchmark_defs])
    n_items_per_model = 80 * len(benchmark_defs)

    out = {
        "title": "Basque LLM Evaluation",
        "subtitle": f"Comparative evaluation on {benchmark_label_list} (multi-seed robust view)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "${OPENAI_API_BASE}",
        "evaluation_protocol": {
            "suite": "Official benchmark subset (multi-seed)",
            "metric": "Accuracy",
            "sampling": f"80 items per benchmark ({n_items_per_model} total per model)",
            "decoding": "temperature=0",
            "seeds": sorted(all_seed_ids),
            "benchmark_families": [
                {
                    "id": fam,
                    "benchmarks": [{"id": bid, "label": BENCH_LABELS.get(bid, bid)} for bid in bids],
                }
                for fam, bids in family_defs.items()
            ],
            "benchmarks": [
                {
                    **b,
                    "label": BENCH_LABELS.get(b["id"], b["id"]),
                }
                for b in benchmark_defs
            ],
        },
        "results": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
