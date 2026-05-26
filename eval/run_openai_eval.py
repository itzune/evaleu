#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import requests


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_accents(s: str) -> str:
    repl = str.maketrans({
        "á": "a", "à": "a", "ä": "a", "â": "a",
        "é": "e", "è": "e", "ë": "e", "ê": "e",
        "í": "i", "ì": "i", "ï": "i", "î": "i",
        "ó": "o", "ò": "o", "ö": "o", "ô": "o",
        "ú": "u", "ù": "u", "ü": "u", "û": "u",
        "ñ": "n",
    })
    return s.translate(repl)


def contains_any(answer: str, candidates: List[str]) -> bool:
    a = strip_accents(norm(answer))
    for c in candidates:
        if strip_accents(norm(c)) in a:
            return True
    return False


def keyword_hits(answer: str, keywords: List[str]) -> int:
    a = strip_accents(norm(answer))
    return sum(1 for k in keywords if strip_accents(norm(k)) in a)


@dataclass
class ItemResult:
    item_id: str
    task_type: str
    ok: bool
    score: float
    answer: str
    expected: Dict[str, Any]


def evaluate_item(item: Dict[str, Any], answer: str) -> ItemResult:
    task = item.get("task_type", "unknown")
    item_id = item.get("id", "unknown")

    ok = False
    score = 0.0

    if "expected_any" in item:
        ok = contains_any(answer, item["expected_any"])
        score = 1.0 if ok else 0.0
    elif "expected_keywords" in item:
        hits = keyword_hits(answer, item["expected_keywords"])
        needed = int(item.get("min_keyword_hits", len(item["expected_keywords"])))
        ok = hits >= needed
        score = min(1.0, hits / max(1, needed))
    elif "expected_option" in item:
        option_ok = contains_any(answer, [item["expected_option"]])
        kw_ok = True
        if "expected_keywords" in item:
            kw_ok = keyword_hits(answer, item["expected_keywords"]) >= 1
        ok = option_ok or kw_ok
        score = 1.0 if ok else 0.0

    expected = {k: v for k, v in item.items() if k.startswith("expected") or k.startswith("min_")}
    return ItemResult(item_id=item_id, task_type=task, ok=ok, score=score, answer=answer, expected=expected)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_answer(msg: Dict[str, Any]) -> str:
    content = (msg.get("content") or "").strip()
    if content:
        return content
    reasoning = (msg.get("reasoning_content") or "").strip()
    if not reasoning:
        return ""

    # Try to find explicit final answer markers in reasoning traces
    lines = [ln.strip() for ln in reasoning.splitlines() if ln.strip()]
    for ln in lines:
        low = ln.lower()
        if low.startswith("final:"):
            return ln.split(":", 1)[1].strip()

    markers = ["final answer", "answer:", "therefore", "thus", "so the answer is"]
    for ln in reversed(lines):
        low = ln.lower()
        if any(m in low for m in markers):
            if ":" in ln:
                return ln.split(":", 1)[1].strip()
            return ln

    # Fallback: last non-empty line
    return lines[-1] if lines else ""


def chat_completion(base_url: str, api_key: str, model: str, prompt: str, temperature: float, max_tokens: int, timeout: int = 90, retries: int = 2) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eman erantzun zuzena eta laburra euskaraz, salbu eta itzulpena eskatzen denean. "
                    "Ez erakutsi arrazoiketa. Eman azken erantzuna bakarrik."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            ans = _extract_answer(msg)
            return (ans or "").strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_err


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


def main():
    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root)

    ap = argparse.ArgumentParser(description="MVP Basque eval runner for OpenAI-compatible endpoints")
    ap.add_argument("--base-url", default=None, help="API base URL (if omitted uses OPENAI_API_BASE from .env)")
    ap.add_argument("--api-key", default=None, help="API key (if omitted uses OPENAI_API_KEY from .env)")
    ap.add_argument("--model", default="kimu-9b", help="model id")
    ap.add_argument("--dataset", default="eval/basque_mvp_dataset.jsonl", help="jsonl dataset path")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None, help="override max tokens; if omitted uses per-model defaults")
    ap.add_argument("--out", default="eval/results_kimu9b_mvp.json", help="output results JSON")
    args = ap.parse_args()

    base_url = _resolve_base_url(args.base_url)
    api_key = _resolve_api_key(args.api_key)
    max_tokens = _max_tokens_for_model(args.max_tokens)

    items = load_jsonl(args.dataset)
    results: List[ItemResult] = []

    for i, item in enumerate(items, 1):
        ans = chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            prompt=item["prompt_eu"],
            temperature=args.temperature,
            max_tokens=max_tokens,
        )
        r = evaluate_item(item, ans)
        results.append(r)
        print(f"[{i}/{len(items)}] {r.item_id}: {'OK' if r.ok else 'FAIL'} | answer={ans!r}")

    overall = sum(r.score for r in results) / max(1, len(results))
    by_task: Dict[str, List[ItemResult]] = {}
    for r in results:
        by_task.setdefault(r.task_type, []).append(r)

    by_task_metrics = {}
    for t, rs in by_task.items():
        by_task_metrics[t] = {
            "n": len(rs),
            "accuracy": sum(1 for x in rs if x.ok) / len(rs),
            "avg_score": sum(x.score for x in rs) / len(rs),
        }

    report = {
        "base_url": base_url,
        "model": args.model,
        "dataset": args.dataset,
        "max_tokens": max_tokens,
        "n_items": len(results),
        "overall_avg_score": overall,
        "overall_accuracy": sum(1 for r in results if r.ok) / max(1, len(results)),
        "by_task": by_task_metrics,
        "items": [
            {
                "id": r.item_id,
                "task_type": r.task_type,
                "ok": r.ok,
                "score": r.score,
                "answer": r.answer,
                "expected": r.expected,
            }
            for r in results
        ],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Model: {args.model}")
    print(f"Items: {len(results)}")
    print(f"Accuracy: {report['overall_accuracy']:.3f}")
    print(f"Avg score: {report['overall_avg_score']:.3f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
