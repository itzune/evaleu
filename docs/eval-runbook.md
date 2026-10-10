# Eval Runbook (single CLI)

This project is operated through `uv run evaleu.py`.

## 1) Preflight
```bash
# The eval environment is fully defined by pyproject.toml + uv.lock
# (datasets and huggingface-hub are pinned exactly: newer `datasets` (>=4)
# drops loading-script support (basqueGLUE) and its cache format breaks 2.21;
# newer `huggingface-hub` is incompatible with `datasets` 2.21).
uv sync
python3 -m py_compile evaleu.py eval/run_eval.py eval/summarize_multiseed.py site/build_site_data.py
python3 -m json.tool site/model_cards.json >/dev/null
```
`uv run` (and `evaleu.py`) automatically use the lock-managed `.venv` for the evaluator subprocesses when present.

## 2) Add model metadata (if new model)
```bash
uv run evaleu.py model \
  --id <model-id> \
  --display-name "<Display Name>" \
  --family "<Family>" \
  --params "<Params>" \
  --upstream-model-id <org/model> \
  --release-date-utc <ISO8601> \
  --release-source-url <URL>
```

## 3) Evaluate one model (default)
```bash
uv run evaleu.py eval --model <model-id>
```
Outputs:
- per-seed json: `eval/<model-id>_seed<seed>.json`
- summary: `eval/summary.json`
- website data: `site/data.json`

### llama.cpp inference VM

The GGUF models are served by `llama-server` (llama.cpp prebuilt
`b11541`, CUDA) on the GPU inference VM (NVIDIA L40, 46 GB — host in
local `.env`/team secrets, never in tracked files):

| Model | Endpoint | systemd unit |
|---|---|---|
| `latxa-qwen3.5-4b` (Q8_0 + mmproj Q8_0) | `http://${EVAL_VM_HOST}:8080` | `llama-latxa-qwen35-4b.service` |
| `latxa-qwen3.5-2b` (Q8_0 + mmproj Q8_0) | `http://${EVAL_VM_HOST}:8081` | `llama-latxa-qwen35-2b.service` |

Both run `--jinja -ngl 99 -c 16384 --parallel 4` and are enabled at boot.
`.env` points `OPENAI_API_BASE` at the 4B port; evaluate the 2B with
`--base-url http://${EVAL_VM_HOST}:8081`.

Management (replace `${EVAL_VM_HOST}` with the VM address from `.env`):
```bash
ssh root@${EVAL_VM_HOST} 'systemctl status llama-latxa-qwen35-4b'
ssh root@${EVAL_VM_HOST} 'journalctl -u llama-latxa-qwen35-2b -f'
```
Binaries live in `/opt/llama-b11541`, weights in `/opt/models/latxa-qwen3.5`.

## 4) Status while running
```bash
uv run evaleu.py status
```

## 5) Rebuild only
```bash
uv run evaleu.py summarize
uv run evaleu.py build
```

## 6) Cleanup legacy artifacts
```bash
uv run evaleu.py clean --apply
```

## 7) Publish
```bash
git add -A
git commit -m "data: refresh eval summary and site data"
git push origin main
```
GitHub Actions deploys `site/` automatically on push.

## Privacy
- Keep endpoint in local `.env`
- Never commit private endpoint URLs
