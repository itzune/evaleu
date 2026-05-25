#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests
from datasets import load_dataset, get_dataset_config_names


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


def _max_tokens_for_model(model: str, cli_max_tokens: int | None) -> int:
    if cli_max_tokens is not None:
        return cli_max_tokens
    return 3072 if model == "qwen3.5-27b" else 256


def _timeout_for_model(model: str, cli_timeout: int | None) -> int:
    if cli_timeout is not None:
        return cli_timeout
    return 240 if model == "qwen3.5-27b" else 120


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

    # Reasoning control for llama-server chat template.
    # Keep legacy qwen default (thinking off) and allow explicit override for any model.
    if disable_thinking or model.startswith("qwen"):
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

            # Qwen deepseek reasoning can return empty content on length stop.
            # Retry once with higher max_tokens before falling back to reasoning extraction.
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

    # Prefer exact/word-boundary matches and check longer labels first
    # (e.g., "not_entailment" contains "entailment").
    normalized = [(_normalize(n), i) for i, n in enumerate(names)]
    normalized.sort(key=lambda x: len(x[0]), reverse=True)

    # exact whole answer
    for name, idx in normalized:
        if t == name:
            return idx

    # token/word-boundary variants (underscore/hyphen/space interchangeable)
    t_soft = re.sub(r"[_\-]", " ", t)
    for name, idx in normalized:
        n_soft = re.sub(r"[_\-]", " ", name)
        if re.search(rf"\b{re.escape(n_soft)}\b", t_soft):
            return idx

    return None


def build_eustrivia_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("HiTZ/EusTrivia")["test"]
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        cands = row["candidates"]
        letters = ["A", "B", "C", "D"]
        opts = "\n".join(f"{letters[j]}) {cands[j]}" for j in range(len(cands)))
        prompt = (
            "Aukeratu aukera zuzena (A/B/C/D bakarrik).\n"
            f"Galdera: {row['question']}\n{opts}\n"
            "Erantzuna:"
        )
        items.append({
            "bench": "EusTrivia",
            "id": f"eustrivia_{row['id']}",
            "prompt": prompt,
            "gold": int(row["answer"]),
            "label_names": letters,
            "meta": {
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
            },
        })
    return items


def build_xnli_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("HiTZ/xnli-eu", "eu_native")["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    label_map = {
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
    }

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Sailkatu hipotesiaren eta premisaren arteko erlazioa. "
            "Erantzun hitz bakarrarekin: entailment, neutral, edo contradiction.\n"
            f"Premisa: {row['premise']}\n"
            f"Hipotesia: {row['hypothesis']}\n"
            "Erantzuna:"
        )
        items.append({
            "bench": "XNLIeu",
            "id": f"xnli_{i}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": [label_map[n] for n in names],
            "meta": {},
        })
    return items


def build_basqueglue_qnli_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("orai-nlp/basqueGLUE", "qnli", trust_remote_code=True)["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Sailkatu ondorengo bikotea. Erantzun ETIKETA BAKARRAREKIN, besterik ez: entailment edo not_entailment.\n"
            "Ez eman azalpenik. Ez erabili beste hitzik.\n"
            f"Galdera: {row['question']}\n"
            f"Esaldia: {row['sentence']}\n"
            "Erantzuna (entailment/not_entailment):"
        )
        items.append({
            "bench": "BasqueGLUE_qnli",
            "id": f"bg_qnli_{row['idx']}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": list(names),
            "meta": {},
        })
    return items


def build_benchmark4_template_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    """
    Benchmark-4 implementation (sentiment-style): BasqueGLUE BEC
    Labels: N (negative), NEU (neutral), P (positive)
    """
    ds = load_dataset("orai-nlp/basqueGLUE", "bec", trust_remote_code=True)["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Sailkatu testuaren sentimendua. Erantzun ETIKETA BAKARRAREKIN, besterik ez: N, NEU, edo P.\n"
            "Ez eman azalpenik. Ez erabili beste hitzik.\n"
            f"Testua: {row['text']}\n"
            "Erantzuna (N/NEU/P):"
        )
        items.append({
            "bench": "BasqueGLUE_bec",
            "id": f"bg_bec_{row['idx']}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": list(names),
            "meta": {},
        })
    return items


def build_benchmark5_template_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    """
    Benchmark-5 implementation (lexical semantics): BasqueGLUE WiC
    Labels: false / true
    """
    ds = load_dataset("orai-nlp/basqueGLUE", "wic", trust_remote_code=True)["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Esan hitzak bi esaldietan esanahi bera duen ala ez. "
            "Erantzun ETIKETA BAKARRAREKIN: true edo false.\n"
            "Ez eman azalpenik. Ez erabili beste hitzik.\n"
            f"Hitza: {row['word']}\n"
            f"Esaldia 1: {row['sentence1']}\n"
            f"Esaldia 2: {row['sentence2']}\n"
            "Erantzuna (true/false):"
        )
        items.append({
            "bench": "BasqueGLUE_wic",
            "id": f"bg_wic_{row['idx']}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": list(names),
            "meta": {"word": row.get("word")},
        })
    return items


def build_benchmark6_template_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    """
    Benchmark-6 implementation (intent classification): BasqueGLUE Intent
    Labels: 12 intent classes
    """
    ds = load_dataset("orai-nlp/basqueGLUE", "intent", trust_remote_code=True)["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    label_block = "\n".join([f"{i}: {name}" for i, name in enumerate(names)])

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Sailkatu erabiltzailearen asmoa (intent).\n"
            "Aukeratu zerrendako etiketa zuzena eta erantzun ZENBAKI BAKARRAREKIN (0-11), besterik ez.\n"
            "Ez eman azalpenik. Ez erabili etiketa testurik.\n"
            f"Etiketak:\n{label_block}\n\n"
            f"Testua: {row['text']}\n"
            "Erantzuna (0-11):"
        )
        items.append({
            "bench": "BasqueGLUE_intent",
            "id": f"bg_intent_{row['idx']}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": list(names),
            "meta": {},
        })
    return items


def _latxa_mc_prompt(question: str, candidates: List[str], context: str | None = None) -> str:
    letters = [chr(ord("A") + i) for i in range(len(candidates))]
    opts = "\n".join(f"{letters[i]}) {candidates[i]}" for i in range(len(candidates)))
    ctx = f"Testuingurua: {context}\n" if context else ""
    return (
        "Aukeratu aukera zuzena (A/B/C/D... edo 1/2/3/4... bakarrik).\n"
        f"{ctx}"
        f"Galdera: {question}\n{opts}\n"
        "Erantzuna:"
    )


def _eusexams_bank_path() -> Path:
    return Path(__file__).resolve().parents[1] / "eval" / ".cache" / "eusexams_eu_bank.json"


def _sanitize_latxa_mc_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    cands = row.get("candidates", []) or []
    if not cands:
        return None
    ans = row.get("answer")
    if ans is None:
        return None
    try:
        ans_i = int(ans)
    except Exception:
        return None
    if not (0 <= ans_i < len(cands)):
        return None
    out = dict(row)
    out["candidates"] = cands
    out["answer"] = ans_i
    return out


def _load_or_build_eusexams_eu_bank() -> List[Dict[str, Any]]:
    cache_path = _eusexams_bank_path()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return [x for x in (_sanitize_latxa_mc_row(r) for r in cached) if x is not None]

    cfgs = [c for c in get_dataset_config_names("HiTZ/EusExams") if c.startswith("eu_")]
    bank: List[Dict[str, Any]] = []
    for cfg in cfgs:
        ds = load_dataset("HiTZ/EusExams", cfg, split="test")
        for row in ds:
            clean = _sanitize_latxa_mc_row({
                "cfg": cfg,
                "id": str(row.get("id")),
                "question": row.get("question", ""),
                "candidates": row.get("candidates", []),
                "answer": row.get("answer"),
            })
            if clean is not None:
                bank.append(clean)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    return bank


def build_latxa_eusexams_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    bank = _load_or_build_eusexams_eu_bank()
    rng = random.Random(seed)
    idxs = list(range(len(bank)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = bank[int(i)]
        cands = row.get("candidates", [])
        if not cands:
            continue
        items.append({
            "bench": "LatxaEval_eusexams",
            "id": f"latxa_eusexams_{row.get('cfg')}_{row.get('id')}",
            "prompt": _latxa_mc_prompt(row.get("question", ""), cands),
            "gold": int(row.get("answer", 0)),
            "label_names": [chr(ord("A") + j) for j in range(len(cands))],
            "meta": {"candidates": cands, "config": row.get("cfg")},
        })
    return items


def build_latxa_eusproficiency_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("HiTZ/EusProficiency", split="test")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        cands = row.get("candidates", [])
        if not cands:
            continue
        items.append({
            "bench": "LatxaEval_eusproficiency",
            "id": f"latxa_eusproficiency_{row['id']}",
            "prompt": _latxa_mc_prompt(row["question"], cands),
            "gold": int(row["answer"]),
            "label_names": [chr(ord("A") + j) for j in range(len(cands))],
            "meta": {"candidates": cands},
        })
    return items


def build_latxa_eusreading_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("HiTZ/EusReading", split="test")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        cands = row.get("candidates", [])
        if not cands:
            continue
        items.append({
            "bench": "LatxaEval_eusreading",
            "id": f"latxa_eusreading_{row['id']}",
            "prompt": _latxa_mc_prompt(row["question"], cands, context=row.get("context", "")),
            "gold": int(row["answer"]),
            "label_names": [chr(ord("A") + j) for j in range(len(cands))],
            "meta": {"candidates": cands},
        })
    return items


def _postprocess_eustrivia_candidates(items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    ds_eu = load_dataset("HiTZ/EusTrivia")["test"]
    eu_by_id = {f"eustrivia_{r['id']}": r for r in ds_eu}
    for it in items:
        row = eu_by_id.get(it["id"])
        if row:
            it.setdefault("meta", {})["candidates"] = row["candidates"]


def _parse_eval_arg(raw: str) -> Tuple[str, int]:
    """Parse a --benchmark value: 'FAMILY/NAME', 'NAME', 'FAMILY/NAME/LIMIT', or 'NAME/LIMIT'."""
    parts = raw.rsplit("/", 2)
    if len(parts) == 3:
        # FAMILY/NAME/LIMIT or NAME/LIMIT (3 parts: treat first two as id)
        bench_id = f"{parts[0].strip()}/{parts[1].strip()}"
        limit = int(parts[2].strip())
    elif len(parts) == 2:
        # NAME/LIMIT or FAMILY/NAME
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


def _build_selected(eval_args: list[str] | None) -> Dict[str, int]:
    """Build selected benchmarks dict from --benchmark args."""
    if not eval_args:
        return {b: 0 for b in _DEFAULT_BENCHMARKS}

    selected: Dict[str, int] = {}
    for raw in eval_args:
        for segment in raw.split(","):
            segment = segment.strip()
            if not segment:
                continue
            bench_id, limit = _parse_eval_arg(segment)
            if bench_id not in _BENCHMARK_DEFS:
                valid = ", ".join(_BENCHMARK_DEFS.keys())
                raise SystemExit(f"Unknown benchmark '{bench_id}'. Valid options: {valid}")
            selected[bench_id] = limit
    return selected


# All known benchmarks with their builders and optional postprocessors.
_BENCHMARK_DEFS: Dict[str, Dict[str, Any]] = {
    "EusTrivia": {
        "builder": build_eustrivia_items,
        "postprocess": _postprocess_eustrivia_candidates,
    },
    "XNLIeu": {
        "builder": build_xnli_items,
    },
    "BasqueGLUE_qnli": {
        "builder": build_basqueglue_qnli_items,
    },
    "BasqueGLUE_bec": {
        "builder": build_benchmark4_template_items,
    },
    "BasqueGLUE_wic": {
        "builder": build_benchmark5_template_items,
    },
    "BasqueGLUE_intent": {
        "builder": build_benchmark6_template_items,
    },
    "LatxaEval_eusexams": {
        "builder": build_latxa_eusexams_items,
    },
    "LatxaEval_eusproficiency": {
        "builder": build_latxa_eusproficiency_items,
    },
    "LatxaEval_eusreading": {
        "builder": build_latxa_eusreading_items,
    },
}

_DEFAULT_BENCHMARKS = ["EusTrivia", "XNLIeu", "BasqueGLUE_qnli"]


def build_benchmark_registry(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = _build_selected(args.benchmark)

    specs: List[Dict[str, Any]] = []
    for bench_id, limit in selected.items():
        if limit == 0:
            limit = 100  # default limit when not specified
        spec = {
            "id": bench_id,
            "limit": limit,
            "builder": _BENCHMARK_DEFS[bench_id]["builder"],
        }
        postprocess = _BENCHMARK_DEFS[bench_id].get("postprocess")
        if postprocess:
            spec["postprocess"] = postprocess
        specs.append(spec)

    return specs


def build_items_from_registry(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    items: List[Dict[str, Any]] = []
    limits: Dict[str, int] = {}

    for spec in build_benchmark_registry(args):
        limit = int(spec.get("limit", 0) or 0)
        bench_id = spec["id"]
        limits[bench_id] = limit
        if limit <= 0:
            continue

        built = spec["builder"](limit, args.seed)
        post = spec.get("postprocess")
        if post:
            post(built)
        items.extend(built)

    return items, limits


@dataclass
class Pred:
    bench: str
    item_id: str
    answer: str
    pred_label: int | None
    gold_label: int
    ok: bool


def score_item(item: Dict[str, Any], answer: str) -> Tuple[int | None, bool]:
    label_names = item["label_names"]
    pred = None

    if item["bench"] in {"EusTrivia", "LatxaEval_eusexams", "LatxaEval_eusproficiency", "LatxaEval_eusreading"}:
        pred = _extract_choice_index(answer, len(label_names))
        if pred is None:
            t = _normalize(answer)
            for i, cand in enumerate(item.get("meta", {}).get("candidates", []) or []):
                if _normalize(cand) in t:
                    pred = i
                    break
    elif item["bench"] == "BasqueGLUE_bec":
        t = _normalize(answer)
        if re.search(r"\bneu(tral)?\b", t):
            pred = 1
        elif re.search(r"\bn(egatiboa|egative)?\b", t):
            pred = 0
        elif re.search(r"\bp(ositiboa|ositive)?\b", t):
            pred = 2
        else:
            pred = _label_from_text(answer, label_names)
    elif item["bench"] == "BasqueGLUE_wic":
        t = _normalize(answer)
        if re.search(r"\b(true|berdin(a)?|bai|same)\b", t):
            pred = 1
        elif re.search(r"\b(false|desberdin(a)?|ezberdin(a)?|different)\b", t):
            pred = 0
        else:
            pred = _label_from_text(answer, label_names)
    elif item["bench"] == "BasqueGLUE_intent":
        m = re.search(r"\b(\d{1,2})\b", answer)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(label_names):
                pred = idx
        if pred is None:
            pred = _label_from_text(answer, label_names)
    else:
        pred = _label_from_text(answer, label_names)

    ok = pred == int(item["gold"]) if pred is not None else False
    return pred, ok


def aggregate(preds: List[Pred]) -> Dict[str, Any]:
    by_bench: Dict[str, List[Pred]] = {}
    for p in preds:
        by_bench.setdefault(p.bench, []).append(p)

    metrics = {}
    total = len(preds)
    total_ok = sum(1 for p in preds if p.ok)
    for b, ps in by_bench.items():
        metrics[b] = {
            "n": len(ps),
            "accuracy": sum(1 for p in ps if p.ok) / max(1, len(ps)),
            "coverage": sum(1 for p in ps if p.pred_label is not None) / max(1, len(ps)),
        }

    return {
        "overall_accuracy": total_ok / max(1, total),
        "n_items": total,
        "by_benchmark": metrics,
    }


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
    args = ap.parse_args()

    base_url = _resolve_base_url(args.base_url)
    api_key = _resolve_api_key(args.api_key)
    max_tokens = _max_tokens_for_model(args.model, args.max_tokens)
    timeout = _timeout_for_model(args.model, args.timeout)

    items, limits = build_items_from_registry(args)

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
        pred_label, ok = score_item(it, ans)
        preds.append(Pred(it["bench"], it["id"], ans, pred_label, int(it["gold"]), ok))
        print(f"[{i}/{len(items)}] {it['bench']} {it['id']}: {'OK' if ok else 'FAIL'} | ans={ans!r}")

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
                "pred_label": p.pred_label,
                "gold_label": p.gold_label,
                "ok": p.ok,
            }
            for p in preds
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print("Model:", args.model)
    print("Items:", out["n_items"])
    print("Overall accuracy:", f"{out['overall_accuracy']:.3f}")
    for b, m in out["by_benchmark"].items():
        print(f"- {b}: acc={m['accuracy']:.3f} cov={m['coverage']:.3f} n={m['n']}")
    print("Saved:", str(out_path))


if __name__ == "__main__":
    main()
