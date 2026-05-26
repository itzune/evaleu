#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple


# Suffix for backup files created before --force/--merge operations.
BACKUP_SUFFIX = "~"


def _backup_file(path: Path) -> Path | None:
    """Backup a file by copying it to <path>.bak.<iso-timestamp>~.

    Returns the backup path if a backup was created, None otherwise.
    """
    if not path.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_suffix(f"{path.suffix}.bak.{ts}{BACKUP_SUFFIX}")
    shutil.copy2(path, backup)
    print(f"[backup] {path} → {backup}")
    return backup

import requests
from datasets import load_dataset, get_dataset_config_names

# Make eval/benchmarks/plugins importable when run as a subprocess.
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))


# ── env & api plumbing (unchanged) ─────────────────────────────────────────

def _load_dotenv(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _resolve_base_url(cli_base_url: str | None) -> str:
    if cli_base_url:
        return cli_base_url
    return os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8080")


def _resolve_api_key(cli_api_key: str | None) -> str:
    if cli_api_key is not None:
        return cli_api_key
    return os.environ.get("OPENAI_API_KEY", "")


def _max_tokens_for_model(cli_max_tokens: int | None) -> int:
    if cli_max_tokens is not None:
        return cli_max_tokens
    return 256


def _timeout_for_model(cli_timeout: int | None) -> int:
    if cli_timeout is not None:
        return cli_timeout
    return 120


def _extract_answer(msg: Dict[str, Any]) -> str:
    content = (msg.get("content") or "").strip()
    if content:
        return content
    reasoning = (msg.get("reasoning_content") or "").strip()
    if not reasoning:
        return ""

    lines = [ln.strip() for ln in reasoning.splitlines() if ln.strip()]
    for ln in lines:
        if ln.lower().startswith("final:"):
            return ln.split(":", 1)[1].strip()
    markers = ["final answer", "answer:", "therefore", "thus", "so the answer is"]
    for ln in reversed(lines):
        low = ln.lower()
        if any(m in low for m in markers):
            return ln.split(":", 1)[1].strip() if ":" in ln else ln
    return lines[-1] if lines else ""


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    disable_thinking: bool = False,
    retries: int = 2,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Erantzun bakarra eman. Ez azaldu arrazoiketa."
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    last_err = None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            finish = data["choices"][0].get("finish_reason")

            if (not (msg.get("content") or "").strip()) and finish == "length" and max_tokens < 8192:
                payload2 = dict(payload)
                payload2["max_tokens"] = min(8192, max_tokens * 2)
                r2 = requests.post(url, json=payload2, headers=headers, timeout=timeout)
                r2.raise_for_status()
                data2 = r2.json()
                msg2 = data2["choices"][0]["message"]
                return _extract_answer(msg2).strip()

            return _extract_answer(msg).strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_err


# ── generic scoring helpers ────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_choice_letter(text: str) -> str | None:
    t = _normalize(text)
    m = re.search(r"\b([a-z])\b", t)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        idx = int(m.group(1))
        if idx >= 1:
            return chr(ord('A') + idx - 1)
    return None


def _extract_choice_index(answer: str, n_choices: int) -> int | None:
    if n_choices <= 0:
        return None
    letter = _extract_choice_letter(answer)
    if letter:
        idx = ord(letter) - ord("A")
        if 0 <= idx < n_choices:
            return idx

    t = _normalize(answer)
    if t.isdigit():
        idx = int(t)
        if 0 <= idx < n_choices:
            return idx
        if 1 <= idx <= n_choices:
            return idx - 1
    return None


def _label_from_text(answer: str, names: List[str]) -> int | None:
    t = _normalize(answer)
    if t.isdigit():
        idx = int(t)
        if 0 <= idx < len(names):
            return idx

    normalized = [(_normalize(n), i) for i, n in enumerate(names)]
    normalized.sort(key=lambda x: len(x[0]), reverse=True)

    for name, idx in normalized:
        if t == name:
            return idx

    t_soft = re.sub(r"[_\-]", " ", t)
    for name, idx in normalized:
        n_soft = re.sub(r"[_\-]", " ", name)
        if re.search(rf"\b{re.escape(n_soft)}\b", t_soft):
            return idx

    return None


# ── benchmark registry (json + plugins) ────────────────────────────────────

_BENCHMARKS_DIR = Path(__file__).resolve().parent / "benchmarks"
_DEFAULT_BENCHMARKS = ["EusTrivia", "XNLIeu", "BasqueGLUE_qnli"]


def _load_benchmark_specs() -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    for f in sorted(_BENCHMARKS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        bench_id = data["id"]
        specs[bench_id] = data
    return specs


def _load_plugin(name: str):
    """Lazy-import a plugin module from eval/benchmarks/plugins/_<name>.py."""
    import importlib
    return importlib.import_module(f"benchmarks.plugins._{name}")


# ── generic item builder (json-driven) ─────────────────────────────────────

def _build_mc_prompt(template: str, row: dict, candidates: list, context: str | None = None) -> str:
    letters = [chr(ord("A") + i) for i in range(len(candidates))]
    opts = "\n".join(f"{letters[j]}) {candidates[j]}" for j in range(len(candidates)))
    ctx = f"Testuingurua: {context}\n" if context else ""
    # Build format kwargs, avoiding duplicates with named args
    fmt_kwargs = {**row, "question": row.get("question", ""), "options": opts,
                  "context": context or ""}
    prompt = template.format(**fmt_kwargs)
    # If template had {context} but no context, remove the resulting blank line.
    prompt = re.sub(r"Testuingurua: \s*\n", "", prompt)
    return prompt


def _build_generic_items(spec: dict, limit: int, seed: int) -> list[dict]:
    ds_cfg = spec["dataset"]
    prompt_cfg = spec["prompt"]

    # Load dataset
    load_kwargs: dict = {}
    if ds_cfg.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True
    if "config" in ds_cfg:
        ds = load_dataset(ds_cfg["path"], ds_cfg["config"], split=ds_cfg["split"], **load_kwargs)
    else:
        ds = load_dataset(ds_cfg["path"], split=ds_cfg["split"], **load_kwargs)

    # Shuffle
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    bench_id = spec["id"]
    id_field = ds_cfg.get("id_field", "id")
    question_field = ds_cfg.get("question_field", "question")
    candidates_field = ds_cfg.get("candidates_field", "candidates")
    answer_field = ds_cfg.get("answer_field", "answer")
    label_field = ds_cfg.get("label_field", "label")
    context_field = ds_cfg.get("context_field")
    meta_fields = ds_cfg.get("meta_fields", []) or []
    label_names_from = ds_cfg.get("label_names_from")
    label_mapping = ds_cfg.get("label_mapping")

    # Resolve label names
    if label_names_from == "dataset_features":
        label_names = list(ds.features[label_field].names)
    else:
        label_names = [chr(ord("A") + j) for j in range(4)]  # fallback for MC

    if label_mapping:
        label_names = [label_mapping.get(n, n) for n in label_names]

    # Determine whether this is MC (has candidates) or text classification
    is_mc = candidates_field in (ds[0] if len(ds) > 0 else {})

    items: list[dict] = []
    for i in idxs:
        row = ds[int(i)]

        if is_mc:
            cands = row.get(candidates_field, [])
            if not cands:
                continue
            gold = int(row.get(answer_field, 0))
            letters = [chr(ord("A") + j) for j in range(len(cands))]
            item_label_names = letters
            context = row.get(context_field) if context_field else None
            prompt = _build_mc_prompt(prompt_cfg["template"], row, cands, context)
            meta = {"candidates": cands}
            for mf in meta_fields:
                meta[mf] = row.get(mf)
        else:
            gold = int(row[label_field])
            item_label_names = list(label_names)
            # Build format kwargs; handle {label_block} if template uses it
            fmt_kwargs = dict(row)
            if "{label_block}" in prompt_cfg["template"]:
                label_block = "\n".join(f"{j}: {name}" for j, name in enumerate(label_names))
                fmt_kwargs["label_block"] = label_block
            prompt = prompt_cfg["template"].format(**fmt_kwargs)
            meta = {}
            for mf in meta_fields:
                meta[mf] = row.get(mf)

        item_id = str(row.get(id_field, i))
        items.append({
            "bench": bench_id,
            "id": f"{bench_id.lower()}_{item_id}",
            "prompt": prompt,
            "gold": gold,
            "label_names": item_label_names,
            "meta": meta,
        })

    return items


# ── generic scorer (json-driven + plugin override) ─────────────────────────

def _generic_score_mc(item: dict, answer: str) -> dict:
    label_names = item["label_names"]
    pred = _extract_choice_index(answer, len(label_names))
    if pred is None:
        t = _normalize(answer)
        for i, cand in enumerate(item.get("meta", {}).get("candidates", []) or []):
            if _normalize(cand) in t:
                pred = i
                break
    ok = pred == int(item["gold"]) if pred is not None else False
    return {"pred_label": pred, "accuracy": 1.0 if ok else 0.0, "coverage": 1.0 if pred is not None else 0.0}


def _generic_score_label(item: dict, answer: str) -> dict:
    pred = _label_from_text(answer, item["label_names"])
    ok = pred == int(item["gold"]) if pred is not None else False
    return {"pred_label": pred, "accuracy": 1.0 if ok else 0.0, "coverage": 1.0 if pred is not None else 0.0}


# ── public API ─────────────────────────────────────────────────────────────

def score_item(bench_spec: dict, item: dict, answer: str) -> dict:
    """Score an item using the benchmark spec + optional plugin.

    Returns a dict of metric_name -> value. Classification benchmarks return
    {"pred_label": int|None, "accuracy": 0.0|1.0, "coverage": 0.0|1.0}.
    Translation benchmarks return {"chrf": float, "bleu": float, ...}.
    Per-item diagnostic fields (e.g. pred_label, pred_number) should use the
    "pred_" prefix; they are excluded from summary aggregation automatically.
    """
    # Try plugin scorer first
    plugin_name = bench_spec.get("plugin")
    if plugin_name:
        try:
            plugin = _load_plugin(plugin_name)
            if hasattr(plugin, "score_item"):
                result = plugin.score_item(item, answer, item["label_names"])
                if isinstance(result, dict):
                    return result
                # Back-compat: old plugins returning (pred, ok) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    pred, ok = result
                    if pred is not None:
                        return {"pred_label": pred, "accuracy": 1.0 if ok else 0.0, "coverage": 1.0}
                # plugin returned None → fall through to generic
        except Exception:
            pass

    # Generic scoring based on spec
    scoring_type = bench_spec["scoring"]["type"]
    if scoring_type == "multiple_choice":
        return _generic_score_mc(item, answer)
    else:
        return _generic_score_label(item, answer)


@dataclass
class Pred:
    bench: str
    item_id: str
    answer: str
    gold: Any
    metrics: dict  # {"accuracy": 0.0|1.0, "chrf": 54.2, "pred_label": 3, ...}
    # Per-item diagnostics must use the "pred_" or "debug_" prefix;
    # those keys are excluded from summary aggregation.


def aggregate(preds: List[Pred]) -> Dict[str, Any]:
    by_bench: Dict[str, List[Pred]] = {}
    for p in preds:
        by_bench.setdefault(p.bench, []).append(p)

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    metrics = {}
    total = len(preds)
    for b, ps in by_bench.items():
        # Discover metric keys from the first pred
        metric_keys = set()
        for p in ps:
            metric_keys.update(p.metrics.keys())
        # Exclude per-item diagnostic fields from aggregation.
        # Keys starting with "pred_" or "debug_" are per-item values
        # (e.g. pred_label, pred_number) — averaging them is meaningless.
        metric_keys = {k for k in metric_keys
                       if not (k.startswith("pred_") or k.startswith("debug_"))}

        bench_metrics = {"n": len(ps)}
        for key in sorted(metric_keys):
            vals = [p.metrics[key] for p in ps if key in p.metrics]
            bench_metrics[key] = _mean(vals)
        metrics[b] = bench_metrics

    # Overall accuracy: mean of per-benchmark accuracy (only classification)
    cls_accs = [m["accuracy"] for m in metrics.values() if "accuracy" in m]
    overall_accuracy = _mean(cls_accs) if cls_accs else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "n_items": total,
        "by_benchmark": metrics,
    }


# ── CLI argument parsing & item building ───────────────────────────────────

def _parse_eval_arg(raw: str) -> Tuple[str, int]:
    """Parse a --benchmark value: 'NAME', 'NAME/LIMIT', 'FAMILY/NAME', or 'FAMILY/NAME/LIMIT'."""
    parts = raw.rsplit("/", 2)
    if len(parts) == 3:
        bench_id = f"{parts[0].strip()}/{parts[1].strip()}"
        limit = int(parts[2].strip())
    elif len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        if b.isdigit():
            bench_id = a
            limit = int(b)
        else:
            bench_id = f"{a}/{b}"
            limit = 0
    else:
        bench_id = raw.strip()
        limit = 0
    return bench_id, limit


def _build_selected(eval_args: list[str] | None, all_specs: dict) -> Dict[str, int]:
    if not eval_args:
        return {b: 0 for b in _DEFAULT_BENCHMARKS}

    selected: Dict[str, int] = {}
    for raw in eval_args:
        for segment in raw.split(","):
            segment = segment.strip()
            if not segment:
                continue
            bench_id, limit = _parse_eval_arg(segment)
            if bench_id not in all_specs:
                valid = ", ".join(all_specs.keys())
                raise SystemExit(f"Unknown benchmark '{bench_id}'. Valid options: {valid}")
            selected[bench_id] = limit
    return selected


def build_items(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, dict]]:
    """Build eval items from selected benchmarks. Returns (items, limits, bench_specs_map)."""
    all_specs = _load_benchmark_specs()
    selected = _build_selected(args.benchmark, all_specs)

    items: List[Dict[str, Any]] = []
    limits: Dict[str, int] = {}
    bench_specs: Dict[str, dict] = {}

    for bench_id, limit in selected.items():
        spec = all_specs[bench_id]
        if limit == 0:
            limit = spec.get("default_limit", 100)
        limits[bench_id] = limit
        bench_specs[bench_id] = spec

        if spec.get("plugin"):
            plugin = _load_plugin(spec["plugin"])
            if hasattr(plugin, "build_items"):
                # Pass spec to plugin so it can read dataset configs
                try:
                    built = plugin.build_items(limit, args.seed, spec=spec)
                except TypeError:
                    # Backward compat: old plugins only take (limit, seed)
                    built = plugin.build_items(limit, args.seed)
            else:
                built = _build_generic_items(spec, limit, args.seed)
        else:
            built = _build_generic_items(spec, limit, args.seed)

        items.extend(built)

    return items, limits, bench_specs


# ── main ───────────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root)

    ap = argparse.ArgumentParser(description="Official Phase-1 Basque benchmark runner")
    ap.add_argument("--base-url", default=None, help="API base URL (if omitted uses OPENAI_API_BASE from .env)")
    ap.add_argument("--api-key", default=None, help="API key (if omitted uses OPENAI_API_KEY from .env)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--disable-thinking", action="store_true", help="Set chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--timeout", type=int, default=None, help="request timeout seconds (per call)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--benchmark", action="append", default=None, metavar="FAMILY/NAME[/LIMIT]",
                    help="Benchmarks to evaluate. Format: FAMILY/NAME or FAMILY/NAME/LIMIT. "
                         "Comma-separate multiple: --benchmark EusTrivia/200,BasqueGLUE_bec/50. "
                         "Repeat flag for more: --benchmark A/B --benchmark C/D/30. "
                         "Defaults: EusTrivia,XNLIeu,BasqueGLUE_qnli (limit=100 each)")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--merge", action="store_true", help="Merge into existing --out file instead of overwriting (skips benchmarks already present)")
    args = ap.parse_args()

    base_url = _resolve_base_url(args.base_url)
    api_key = _resolve_api_key(args.api_key)
    max_tokens = _max_tokens_for_model(args.max_tokens)
    timeout = _timeout_for_model(args.timeout)

    items, limits, bench_specs = build_items(args)

    preds: List[Pred] = []
    for i, it in enumerate(items, 1):
        ans = chat_completion(
            base_url,
            api_key,
            args.model,
            it["prompt"],
            args.temperature,
            max_tokens,
            timeout=timeout,
            disable_thinking=args.disable_thinking,
        )
        bench_spec = bench_specs[it["bench"]]
        m = score_item(bench_spec, it, ans)
        preds.append(Pred(it["bench"], it["id"], ans, it.get("gold"), m))

        # Human-friendly per-item line
        if m.get("accuracy") is not None:
            status = "OK" if m["accuracy"] >= 1.0 else "FAIL"
        elif m.get("chrf") is not None:
            status = f"chrF={m['chrf']:.1f}"
        else:
            status = ", ".join(f"{k}={v:.1f}" for k, v in m.items() if k != "pred_label")
        print(f"[{i}/{len(items)}] {it['bench']} {it['id']}: {status} | ans={ans!r}")

    summary = aggregate(preds)
    out = {
        "base_url": "${OPENAI_API_BASE}",
        "model": args.model,
        "suite": "evaleu",
        "max_tokens": max_tokens,
        "timeout": timeout,
        "limits": limits,
        **summary,
        "items": [
            {
                "bench": p.bench,
                "id": p.item_id,
                "answer": p.answer,
                "gold": p.gold,
                "metrics": p.metrics,
            }
            for p in preds
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge into existing results if --merge and file exists
    if args.merge and out_path.exists():
        _backup_file(out_path)
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        existing_bms = set(item["bench"] for item in existing.get("items", []))
        new_bms = set(item["bench"] for item in out["items"])
        overlap = existing_bms & new_bms
        if overlap:
            print(f"[merge] Skipping {len(overlap)} already-present benchmark(s): {sorted(overlap)}")
        # Only keep new benchmarks
        new_items = [item for item in out["items"] if item["bench"] not in existing_bms]
        if not new_items:
            print(f"[merge] All benchmarks already present; nothing new to add.")
            return
        # Merge items and re-aggregate
        all_items = existing["items"] + new_items
        preds_merged = [
            Pred(p["bench"], p["id"], p["answer"], p.get("gold"), p.get("metrics", {}))
            for p in all_items
        ]
        summary_merged = aggregate(preds_merged)
        merged_limits = {**existing.get("limits", {}), **out.get("limits", {})}
        out = {
            **existing,
            **summary_merged,
            "limits": merged_limits,
            "items": [
                {
                    "bench": p.bench,
                    "id": p.item_id,
                    "answer": p.answer,
                    "gold": p.gold,
                    "metrics": p.metrics,
                }
                for p in preds_merged
            ],
        }
        print(f"[merge] Added {len(new_items)} items from {len(new_bms - overlap)} new benchmark(s).")
        print(f"[merge] Total: {out['n_items']} items across {len(out['by_benchmark'])} benchmarks.")

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print("Model:", args.model)
    print("Items:", out["n_items"])
    print("Overall accuracy:", f"{out['overall_accuracy']:.3f}")
    for b, m in out["by_benchmark"].items():
        parts = []
        if "accuracy" in m:
            parts.append(f"acc={m['accuracy']:.3f}")
        if "coverage" in m:
            parts.append(f"cov={m['coverage']:.3f}")
        if "chrf" in m:
            parts.append(f"chrF={m['chrf']:.1f}")
        if "bleu" in m:
            parts.append(f"BLEU={m['bleu']:.1f}")
        if not parts:
            parts.append(f"n={m['n']}")
        print(f"- {b}: {' '.join(parts)} n={m['n']}")
    print("Saved:", str(out_path))


if __name__ == "__main__":
    main()
