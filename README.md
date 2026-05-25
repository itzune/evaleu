# Basque LLM Evaluation (single-CLI workflow)

This repo benchmarks Basque-capable local LLMs served through a private OpenAI-compatible endpoint and publishes a static comparison website.

The operational interface is **one CLI**:

```bash
uv run evaleu.py <command> [options]
```

---

## Commands

### Evaluate one model (default day-to-day)
```bash
uv run evaleu.py eval --model latxa-qwen3-vl-4b
```
Runs the default benchmarks (EusTrivia, XNLIeu, BasqueGLUE_qnli) for seeds `42,123,777`, writes per-seed JSONs into `eval/`, refreshes `eval/summary.json`, and rebuilds `site/data.json`.

Select specific benchmarks with `--benchmark`:
```bash
# Run only EusTrivia
uv run evaleu.py eval --model latxa-qwen3-vl-4b --benchmark EusTrivia

# Custom limits per benchmark (comma-separated)
uv run evaleu.py eval --model latxa-qwen3-vl-4b --benchmark EusTrivia/200,BasqueGLUE_bec/50

# Repeat the flag for multiple benchmarks
uv run evaleu.py eval --model latxa-qwen3-vl-4b --benchmark EusTrivia --benchmark LatxaEval_eusexams/30
```

Available benchmarks: `EusTrivia`, `XNLIeu`, `BasqueGLUE_qnli`, `BasqueGLUE_bec`, `BasqueGLUE_wic`, `BasqueGLUE_intent`, `LatxaEval_eusexams`, `LatxaEval_eusproficiency`, `LatxaEval_eusreading`.

Back-compat shortcut also works:
```bash
uv run evaleu.py --model latxa-qwen3-vl-4b
```

### Evaluate all models in `site/model_cards.json`
```bash
uv run evaleu.py eval --all
```

### Summarize existing eval JSONs
```bash
uv run evaleu.py summarize
```

### Build website data from summary
```bash
uv run evaleu.py build
```

### Add or update a model
```bash
uv run evaleu.py model \
  --id my-model \
  --display-name "My Model" \
  --family "Llama" \
  --params "8B" \
  --upstream-model-id org/model \
  --release-date-utc 2026-01-01T00:00:00Z \
  --release-source-url https://huggingface.co/org/model
```

### Check progress
```bash
uv run evaleu.py status
```

### Clean legacy wrappers/artifacts
```bash
uv run evaleu.py clean --apply
```

---

## Benchmarks (all-bench suite)
- Core: EusTrivia, XNLIeu
- BasqueGLUE: QNLI, BEC, WiC, Intent
- LatxaEvalSuite: EusExams, EusProficiency, EusReading

## Methodology
- `temperature=0`
- Multi-seed robust view (`42,123,777` by default)
- Equal sampling budget per benchmark (`80` default)
- Ranking by mean overall accuracy
- UI shows rounded values, with `mean ± std` on hover
- Best value per benchmark highlighted in bold

For `qwen3.5-27b`, eval uses no-thinking mode (`--max-tokens 4096 --timeout 300`).

---

## Repository structure
- `evaleu.py` — single operational CLI
- `eval/run_eval.py` — evaluator engine (benchmark registry + scoring)
- `eval/summarize_multiseed.py` — summary builder (`eval/summary.json`)
- `site/model_cards.json` — model registry + metadata
- `site/build_site_data.py` — builds `site/data.json`
- `site/index.html` — static report UI

---

## Publish
Commit + push to `main`. GitHub Actions auto-deploys `site/` to `gh-pages`.

## Privacy
- Keep endpoint in local `.env` (`OPENAI_API_BASE=...`)
- Keep auth token in local `.env` (`OPENAI_API_KEY=`; empty for local/no-auth endpoints)
- Never commit private endpoint URLs
- Tracked artifacts must use placeholders where needed
