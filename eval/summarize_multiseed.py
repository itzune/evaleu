#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def parse_seed(name: str):
    m = re.search(r"_seed(\d+)\.json$", name)
    return int(m.group(1)) if m else None


def _make_benchmark_entry(metric_vals: dict) -> dict:
    """Build a benchmark summary entry from {metric_name: [values_across_seeds]}."""
    entry = {}
    for k, vals in metric_vals.items():
        entry[f"{k}_mean"] = mean(vals)
        entry[f"{k}_std"] = std(vals)
    entry["n"] = len(next(iter(metric_vals.values()), []))
    return entry


def main():
    ap = argparse.ArgumentParser(description="Summarize multi-seed eval results")
    ap.add_argument("--input-dir", default="eval")
    ap.add_argument("--out", default="eval/summary.json")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    rows = []
    for p in sorted(in_dir.glob("*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        if "model" not in j:
            continue
        seed = parse_seed(p.name)
        rows.append({
            "file": str(p),
            "model": j["model"],
            "seed": seed,
            "overall": j.get("overall_accuracy", 0.0),
            "by_benchmark": j.get("by_benchmark", {}),
        })

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["model"]].append(r)

    summary = {"models": {}, "n_runs": len(rows)}

    for model, rs in grouped.items():
        rs = sorted(rs, key=lambda x: (x["seed"] is None, x["seed"]))
        ov = [x["overall"] for x in rs]

        # Collect per-benchmark metrics across seeds.
        # metric_keys[bench_id] = set of metric names found.
        bench_metric_keys: dict = defaultdict(set)
        bench_metric_vals: dict = defaultdict(lambda: defaultdict(list))
        for x in rs:
            for b, bm in x.get("by_benchmark", {}).items():
                for k, v in bm.items():
                    if k == "n":
                        continue
                    bench_metric_keys[b].add(k)
                    bench_metric_vals[b][k].append(v)

        summary["models"][model] = {
            "runs": [
                {
                    "seed": x["seed"],
                    "overall": x["overall"],
                    "by_benchmark": x["by_benchmark"],
                    "file": x["file"],
                }
                for x in rs
            ],
            "overall_mean": mean(ov),
            "overall_std": std(ov),
            "benchmarks": {
                b: _make_benchmark_entry(mk)
                for b, mk in bench_metric_vals.items()
            },
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable console summary
    print(f"Runs: {len(rows)}")
    for model, m in sorted(summary["models"].items(), key=lambda kv: kv[1]["overall_mean"], reverse=True):
        print(f"{model:18s} overall={m['overall_mean']*100:5.2f}% ± {m['overall_std']*100:4.2f}")
        for b, bm in m["benchmarks"].items():
            parts = []
            if "accuracy_mean" in bm:
                parts.append(f"acc={bm['accuracy_mean']*100:5.2f}% ± {bm['accuracy_std']*100:4.2f}")
            if "chrf_mean" in bm:
                parts.append(f"chrF={bm['chrf_mean']:.1f} ± {bm['chrf_std']:.1f}")
            if "bleu_mean" in bm:
                parts.append(f"BLEU={bm['bleu_mean']:.1f} ± {bm['bleu_std']:.1f}")
            if not parts:
                parts.append("—")
            print(f"  - {b:16s} {' '.join(parts)} (n={bm['n']})")

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
