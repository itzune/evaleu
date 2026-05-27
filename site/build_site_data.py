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
    {
        "id": "MGSM_eu",
        "family": "MathReasoning",
        "task": "Grade school math word problems in Basque (free-form generation, exact numeric match)",
        "metric": "Accuracy",
        "labels": "free-form — numeric answer extracted from CoT output",
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
    "MGSM_eu": "MGSM EU",
}


SKILL_DEFS = [
    {
        "id": "knowledge",
        "label": "KNOWLEDGE",
        "description": "What the model knows — facts, culture, exams",
        "benchmarks": ["EusTrivia", "MMLU_eu", "BertaQA_eu", "LatxaEval_eusexams"],
    },
    {
        "id": "reading",
        "label": "READING",
        "description": "How well the model understands text",
        "benchmarks": ["XNLIeu", "LatxaEval_eusreading", "BasqueGLUE_qnli"],
    },
    {
        "id": "writing",
        "label": "WRITING",
        "description": "Basque proficiency and usage",
        "benchmarks": ["LatxaEval_eusproficiency"],
    },
    {
        "id": "reasoning",
        "label": "REASONING",
        "description": "Logic, math, and disambiguation",
        "benchmarks": ["MGSM_eu", "BasqueGLUE_wic", "BasqueGLUE_bec"],
    },
    {
        "id": "pragmatics",
        "label": "PRAGMATICS",
        "description": "Intent and real-world language use",
        "benchmarks": ["BasqueGLUE_intent"],
    },
]


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

    skills = []
    rows_by_model = {r["model_id"]: r for r in rows}
    for sd in SKILL_DEFS:
        skill_benchmarks = [b for b in sd["benchmarks"] if b in benchmark_ids]
        ranking = []
        for r in rows:
            by_b = r.get("by_benchmark", {})
            vals = []
            for bid in skill_benchmarks:
                acc = ((by_b.get(bid) or {}).get("accuracy"))
                if acc is not None and acc > 0:
                    vals.append(float(acc) * 100.0)
            if vals:
                ranking.append({
                    "model_id": r["model_id"],
                    "display_name": r["display_name"],
                    "score": mean(vals),
                    "n_benchmarks": len(vals),
                })

        ranking.sort(key=lambda x: x["score"], reverse=True)
        winner = ranking[0] if ranking else None
        runner_up = ranking[1] if len(ranking) > 1 else None
        margin = ((winner["score"] - runner_up["score"]) if (winner and runner_up) else 0.0)

        skills.append({
            "id": sd["id"],
            "label": sd["label"],
            "description": sd["description"],
            "benchmarks": skill_benchmarks,
            "winner": winner,
            "runner_up": runner_up,
            "margin": margin,
            "ranking": ranking,
        })

    benchmark_label_list = ", ".join([BENCH_LABELS.get(b["id"], b["id"]) for b in benchmark_defs])
    n_items_per_model = 80 * len(benchmark_defs)

    # Build skill_categories for the frontend toggle (benchmark → skill mapping)
    skill_categories = []
    for sd in SKILL_DEFS:
        skill_benchmarks = [b for b in sd["benchmarks"] if b in benchmark_ids]
        if skill_benchmarks:
            skill_categories.append({
                "id": sd["id"],
                "label": sd["label"],
                "benchmarks": [{"id": bid, "label": BENCH_LABELS.get(bid, bid)} for bid in skill_benchmarks],
            })

    out = {
        "title": "Basque LLM Evaluation",
        "subtitle": f"Comparative evaluation on {benchmark_label_list} (multi-seed robust view)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "${OPENAI_API_BASE}",
        "skill_categories": skill_categories,
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
        "skills": skills,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
