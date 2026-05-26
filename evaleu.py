#!/usr/bin/env python3
"""Unified CLI for Basque LLM evaluation workflow.

Examples:
  uv run evaleu.py eval --model latxa-qwen3-vl-4b
  uv run evaleu.py eval --model latxa-qwen3-vl-4b --benchmark BasqueGLUE_bec/50
  uv run evaleu.py eval --all
  uv run evaleu.py summarize
  uv run evaleu.py build
  uv run evaleu.py model --id my-model --display-name "My Model" --family "Llama" --params "8B" \
    --upstream-model-id org/model --release-date-utc 2026-01-01T00:00:00Z --release-source-url https://huggingface.co/org/model
  uv run evaleu.py status
  uv run evaleu.py server --dir site --host 127.0.0.1 --port 8000
  uv run evaleu.py clean --apply

Backwards compatibility:
  uv run evaleu.py --model latxa-qwen3-vl-4b
  (interpreted as: uv run evaleu.py eval --model ...)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_CARDS_PATH = ROOT / "site" / "model_cards.json"

CLEAN_PATHS = [
    "eval/.run_status",
    "eval/.cache",
    "eval/run.log",
]


def run_cmd(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    # Avoid inheriting uv-run isolation values into child Python executions.
    for key in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def default_python() -> str:
    return "/usr/bin/python3" if Path("/usr/bin/python3").exists() else "python3"


def parse_csv(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def load_model_cards() -> dict:
    return json.loads(MODEL_CARDS_PATH.read_text(encoding="utf-8"))


def save_model_cards(cards: dict) -> None:
    MODEL_CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_ids_from_cards() -> list[str]:
    return list(load_model_cards().keys())


def summary_from_out_dir(out_dir: Path, explicit_summary: str | None) -> Path:
    if explicit_summary:
        p = Path(explicit_summary)
        return p if p.is_absolute() else (ROOT / p).resolve()
    return (out_dir / "summary.json").resolve()


def add_common_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seeds", default="42,123,777", help="Comma-separated seeds")
    p.add_argument("--out-dir", default="eval", help="Directory for per-seed outputs + summary.json")
    p.add_argument("--summary", default=None, help="Summary output path (default: <out-dir>/summary.json)")
    p.add_argument("--site-data", default="site/data.json", help="Site payload output path")
    p.add_argument("--force", action="store_true", help="Re-run even if output json already exists; merges new benchmarks instead of overwriting")
    p.add_argument("--python", default=default_python(), help="Python interpreter for underlying scripts")
    p.add_argument("--no-summarize", action="store_true", help="Do not run summarize step")
    p.add_argument("--no-build", action="store_true", help="Do not run site build step")
    p.add_argument("--disable-thinking", action="store_true", help="Forward --disable-thinking to runner (chat_template_kwargs.enable_thinking=false)")

    p.add_argument("--benchmark", action="append", default=None, metavar="FAMILY/NAME[/LIMIT]",
                   help="Benchmarks to evaluate. Format: FAMILY/NAME or FAMILY/NAME/LIMIT. "
                        "Comma-separate multiple: --benchmark EusTrivia/200,BasqueGLUE_bec/50. "
                        "Repeat flag for more: --benchmark A --benchmark B/30. "
                        "Defaults: EusTrivia,XNLIeu,BasqueGLUE_qnli (limit=80 each)")


def _model_slug(model_id: str) -> str:
    """Sanitize model ID for use in filenames. Replaces '/' with '_' to prevent
    subdirectory creation when model IDs contain slashes (e.g. org/model)."""
    return model_id.replace("/", "_")


def run_one_model_eval(args: argparse.Namespace, model_id: str) -> None:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in parse_csv(args.seeds):
        out_file = out_dir / f"{_model_slug(model_id)}_seed{seed}.json"
        if out_file.exists() and not args.force:
            print(f"[skip] exists: {out_file}")
            continue

        cmd = [
            args.python,
            "eval/run_eval.py",
            "--model",
            model_id,
            "--seed",
            seed,
            "--out",
            str(out_file),
        ]

        # When --force and file exists with --benchmark flags, merge instead of overwrite
        if out_file.exists() and args.force and args.benchmark:
            cmd.append("--merge")

        if args.benchmark:
            for bm_spec in args.benchmark:
                cmd += ["--benchmark", bm_spec]

        if model_id == "qwen3.5-27b":
            cmd += ["--max-tokens", "4096", "--timeout", "300"]

        if args.disable_thinking:
            cmd.append("--disable-thinking")

        run_cmd(cmd)


def run_summarize(python: str, out_dir: str, summary: str | None) -> Path:
    out = Path(out_dir)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    summ = summary_from_out_dir(out, summary)
    run_cmd([python, "eval/summarize_multiseed.py", "--input-dir", str(out), "--out", str(summ)])
    return summ


def run_build(python: str, summary: str, site_data: str) -> None:
    summary_p = Path(summary)
    if not summary_p.is_absolute():
        summary_p = (ROOT / summary).resolve()
    site_data_p = Path(site_data)
    if not site_data_p.is_absolute():
        site_data_p = (ROOT / site_data).resolve()
    run_cmd([python, "site/build_site_data.py", "--summary", str(summary_p), "--out", str(site_data_p)])


def cmd_eval(args: argparse.Namespace) -> int:
    if args.all and args.model:
        raise SystemExit("Use either --model <id> or --all (not both)")

    if args.all:
        models = parse_csv(args.models_csv) if args.models_csv else model_ids_from_cards()
        if not models:
            raise SystemExit("No models provided and no model_cards entries found")
        for model_id in models:
            run_one_model_eval(args, model_id)
    else:
        if not args.model:
            raise SystemExit("Provide --model <id> or use --all")
        run_one_model_eval(args, args.model)

    summary_p = summary_from_out_dir((ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir), args.summary)
    summarized = False
    built = False
    if not args.no_summarize:
        summary_p = run_summarize(args.python, args.out_dir, args.summary)
        summarized = True
    if not args.no_build:
        run_build(args.python, str(summary_p), args.site_data)
        built = True

    if summarized:
        print(f"Summary refreshed: {summary_p}")
    else:
        print(f"Summary skipped. Current path: {summary_p}")

    if built:
        print(f"Site data refreshed: {args.site_data}")
    else:
        print("Site build skipped.")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    summary = run_summarize(args.python, args.out_dir, args.summary)
    print(f"Summary refreshed: {summary}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    run_build(args.python, args.summary, args.site_data)
    print(f"Site data refreshed: {args.site_data}")
    return 0


def cmd_server(args: argparse.Namespace) -> int:
    serve_dir = Path(args.dir)
    if not serve_dir.is_absolute():
        serve_dir = (ROOT / serve_dir).resolve()
    if not serve_dir.exists() or not serve_dir.is_dir():
        raise SystemExit(f"Server directory does not exist: {serve_dir}")

    run_cmd(
        [
            args.python,
            "-m",
            "http.server",
            str(args.port),
            "--bind",
            args.host,
        ],
        cwd=serve_dir,
    )
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    cards = load_model_cards()
    existing = cards.get(args.id, {})
    if args.id in cards and not args.force:
        raise SystemExit(f"Model '{args.id}' already exists. Use --force to overwrite.")

    def pick(field: str, arg_value: str | None, default: str | None = None) -> str:
        if arg_value is not None:
            return arg_value
        if field in existing and existing[field] is not None:
            return existing[field]
        if default is not None:
            return default
        raise SystemExit(f"Missing required field for model '{args.id}': --{field.replace('_', '-')}")

    display_name = pick("display_name", args.display_name)
    suffix = "(no-thinking)"
    if args.no_thinking and suffix not in display_name:
        display_name = f"{display_name} {suffix}".strip()

    site_visibility = existing.get("site_visibility", "published")
    if args.site_visibility is not None:
        site_visibility = args.site_visibility
    elif args.hide or args.draft:
        site_visibility = "draft"
    elif args.unhide or args.published:
        site_visibility = "published"

    cards[args.id] = {
        "display_name": display_name,
        "family": pick("family", args.family),
        "params": pick("params", args.params),
        "weights_quant": pick("weights_quant", args.weights_quant, default="F16"),
        "kv_cache": pick("kv_cache", args.kv_cache, default="f16/f16"),
        "upstream_model_id": pick("upstream_model_id", args.upstream_model_id),
        "release_date_utc": pick("release_date_utc", args.release_date_utc),
        "release_source_url": pick("release_source_url", args.release_source_url),
        "site_visibility": site_visibility,
    }
    save_model_cards(cards)
    print(f"Added/updated model card: {args.id}")
    return 0


def _extract_model_from_tokens(tokens: list[str]) -> str | None:
    for i, tok in enumerate(tokens):
        if tok == "--model" and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith("--model="):
            return tok.split("=", 1)[1]
    return None


def detect_running_models(expected_models: list[str]) -> set[str]:
    model_set = set(expected_models)
    running: set[str] = set()
    try:
        ps_out = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except Exception:
        return running

    for line in ps_out.splitlines():
        if ("evaleu.py eval" not in line) and ("eval/run_eval.py" not in line):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()

        model_id = _extract_model_from_tokens(tokens)
        if model_id and model_id in model_set:
            running.add(model_id)
    return running


def format_status_table(rows: list[dict], total_done: int, total_expected: int) -> str:
    headers = ["Model", "Runs", "Progress", "Status", "Missing seeds", "Last update (UTC)"]
    rendered_rows = []
    for row in rows:
        rendered_rows.append([
            row["model"],
            f"{row['done']}/{row['expected']}",
            f"{row['progress_percent']:.1f}%",
            row["status"],
            ",".join(row["missing_seeds"]) if row["missing_seeds"] else "-",
            row["latest_result_mtime_utc"] or "-",
        ])

    widths = [len(h) for h in headers]
    for r in rendered_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)
    total_pct = round((100.0 * total_done / total_expected), 1) if total_expected else 0.0
    lines = [
        f"Overall: {total_done}/{total_expected} ({total_pct:.1f}%)",
        fmt_row(headers),
        sep,
    ]
    lines.extend(fmt_row(r) for r in rendered_rows)
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    models = parse_csv(args.models_csv) if args.models_csv else model_ids_from_cards()
    seeds = parse_csv(args.seeds)
    expected = len(models) * len(seeds)

    # Build slug → real model_id mapping so we can match sanitized filenames.
    slug_to_model = {_model_slug(m): m for m in models}

    run_files = sorted(p for p in out_dir.glob("*_seed*.json") if p.name != "summary.json")

    results_by_model: dict[str, set[str]] = {}
    latest_by_model: dict[str, datetime] = {}
    for p in run_files:
        stem = p.stem
        if "_seed" not in stem:
            continue
        file_model_id, seed = stem.rsplit("_seed", 1)
        # Resolve sanitized slug back to real model ID if possible.
        model_id = slug_to_model.get(file_model_id, file_model_id)
        results_by_model.setdefault(model_id, set()).add(seed)

        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        prev = latest_by_model.get(model_id)
        if prev is None or mtime > prev:
            latest_by_model[model_id] = mtime

    model_rows: list[dict] = []
    done = 0
    running_models = detect_running_models(models)
    for model_id in models:
        seeds_done = results_by_model.get(model_id, set())
        done_for_model = sum(1 for s in seeds if s in seeds_done)
        done += done_for_model
        missing = [s for s in seeds if s not in seeds_done]
        if model_id in running_models:
            status = "running"
        elif done_for_model == len(seeds) and len(seeds) > 0:
            status = "complete"
        elif done_for_model > 0:
            status = "partial"
        else:
            status = "pending"

        latest = latest_by_model.get(model_id)
        model_rows.append({
            "model": model_id,
            "done": done_for_model,
            "expected": len(seeds),
            "progress_percent": round((100.0 * done_for_model / len(seeds)), 1) if seeds else 0.0,
            "status": status,
            "is_running": model_id in running_models,
            "missing_seeds": missing,
            "latest_result_mtime_utc": latest.isoformat() if latest else None,
        })

    pct = round((100.0 * done / expected), 1) if expected else 0.0

    latest_mtime = None
    if run_files:
        latest = max(run_files, key=lambda p: p.stat().st_mtime)
        latest_mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).isoformat()

    expected_set = set(models)
    unexpected_models = sorted(m for m in results_by_model.keys() if m not in expected_set)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "expected_runs": expected,
        "completed_runs": done,
        "progress_percent": pct,
        "summary_exists": (out_dir / "summary.json").exists(),
        "latest_result_mtime_utc": latest_mtime,
        "running_models": sorted(running_models),
        "models": model_rows,
        "unexpected_models_with_results": unexpected_models,
    }

    table = format_status_table(model_rows, done, expected)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(table)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    root = ROOT
    removed: list[str] = []
    for rel in CLEAN_PATHS:
        p = root / rel
        if not p.exists():
            continue
        if args.apply:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        removed.append(rel)

    print(json.dumps({
        "mode": "apply" if args.apply else "dry_run",
        "removed_or_planned": removed,
        "count": len(removed),
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Unified CLI for eval, summarize, build, model registry, and cleanup")
    sub = ap.add_subparsers(dest="cmd")

    p_eval = sub.add_parser("eval", help="Evaluate one model (--model) or all models (--all), then summarize/build by default")
    p_eval.add_argument("--model", required=False)
    p_eval.add_argument("--all", action="store_true", help="Evaluate all models from model_cards (or --models-csv)")
    p_eval.add_argument("--models-csv", default=None, help="Comma-separated model IDs when using --all")
    add_common_eval_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_sum = sub.add_parser("summarize", help="Generate eval/summary.json from per-seed eval/*.json")
    p_sum.add_argument("--out-dir", default="eval")
    p_sum.add_argument("--summary", default=None, help="Default: <out-dir>/summary.json")
    p_sum.add_argument("--python", default=default_python())
    p_sum.set_defaults(func=cmd_summarize)

    p_build = sub.add_parser("build", help="Build site/data.json from summary")
    p_build.add_argument("--summary", default="eval/summary.json")
    p_build.add_argument("--site-data", default="site/data.json")
    p_build.add_argument("--python", default=default_python())
    p_build.set_defaults(func=cmd_build)

    p_server = sub.add_parser("server", help="Serve static files (default: ./site) using Python http.server")
    p_server.add_argument("--dir", default="site", help="Directory to serve")
    p_server.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_server.add_argument("--port", type=int, default=8000, help="Bind port")
    p_server.add_argument("--python", default=default_python())
    p_server.set_defaults(func=cmd_server)

    p_model = sub.add_parser("model", help="Add or update a model card entry")
    p_model.add_argument("--id", required=True)
    p_model.add_argument("--display-name", default=None)
    p_model.add_argument("--family", default=None)
    p_model.add_argument("--params", default=None)
    p_model.add_argument("--upstream-model-id", default=None)
    p_model.add_argument("--release-date-utc", default=None)
    p_model.add_argument("--release-source-url", default=None)
    p_model.add_argument("--weights-quant", default=None)
    p_model.add_argument("--kv-cache", default=None)
    p_model.add_argument("--no-thinking", action="store_true", help="Append '(no-thinking)' to display name")
    visibility_group = p_model.add_mutually_exclusive_group()
    visibility_group.add_argument(
        "--site-visibility",
        choices=["published", "draft"],
        default=None,
        help="Website visibility state (published or draft)",
    )
    visibility_group.add_argument("--hide", action="store_true", help="Alias for --site-visibility draft")
    visibility_group.add_argument("--unhide", action="store_true", help="Alias for --site-visibility published")
    visibility_group.add_argument("--draft", action="store_true", help="Alias for --site-visibility draft")
    visibility_group.add_argument("--published", action="store_true", help="Alias for --site-visibility published")
    p_model.add_argument("--force", action="store_true")
    p_model.set_defaults(func=cmd_model)

    p_status = sub.add_parser("status", help="Show per-model progress table by default; use --json for machine-readable output")
    p_status.add_argument("--out-dir", default="eval")
    p_status.add_argument("--models-csv", default=None, help="Comma-separated expected models (default: model_cards)")
    p_status.add_argument("--seeds", default="42,123,777")
    p_status.add_argument("--json", action="store_true", help="Output JSON instead of the default ASCII table")
    p_status.set_defaults(func=cmd_status)

    p_clean = sub.add_parser("clean", help="Remove local transient eval runtime artifacts")
    p_clean.add_argument("--apply", action="store_true", help="Actually delete files; default is dry-run")
    p_clean.set_defaults(func=cmd_clean)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or os.sys.argv[1:])

    # Back-compat: if user calls `uv run evaleu.py --model ...`, treat as `eval`.
    # Keep top-level help/version-like flags as top-level behavior.
    if argv and argv[0] in {"--model", "-m"}:
        argv = ["eval", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
