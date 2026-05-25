# Benchmark-4 Onboarding Template (Unified Runner)

This project now includes a benchmark-registry hook in:
- `eval/run_eval.py`

## Current Benchmark-4 implementation

- **Implemented benchmark**: `BasqueGLUE_bec` (sentiment-style classification)
- Labels: `N` (negative), `NEU` (neutral), `P` (positive)
- Dataset source: `orai-nlp/basqueGLUE`, config `bec`, split `test`

CLI enablement:
- `--benchmark BasqueGLUE_bec` (default limit=100)
- `--benchmark BasqueGLUE_bec/N` (custom limit)

When enabled, output JSON `limits` includes `BasqueGLUE_bec`.

## How to run with BEC (smoke)

```bash
python3 eval/run_eval.py \
  --model kimu-9b \
  --benchmark EusTrivia/20,XNLIeu/20,BasqueGLUE_qnli/20,BasqueGLUE_bec/20 \
  --out eval/with_bec_smoke.json
```

## How to swap Benchmark-4 later

If you want a different Benchmark-4 in the future:
1) Edit `build_benchmark4_template_items()`
2) Keep output item schema:

```python
{
  "bench": "<benchmark_id>",
  "id": "unique_id",
  "prompt": "...",
  "gold": 0,
  "label_names": ["label_a", "label_b"],
  "meta": {}
}
```

3) Register the new benchmark in `_BENCHMARK_DEFS` dict in `eval/run_eval.py`
4) Add bench-specific parsing branch in `score_item()` if needed

## Notes

- Keep endpoint private (`OPENAI_API_BASE` via `.env`).
- qwen no-thinking control remains active in chat payload.
- Registry-based `limits` auto-reflect enabled benchmarks.
