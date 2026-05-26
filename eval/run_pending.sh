#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Load .env
set -a; source .env; set +a

# Common benchmark flags (13 benchmarks, 80 items each, matching existing evals)
BENCH_FLAGS=(
  --benchmark EusTrivia/80
  --benchmark XNLIeu/80
  --benchmark BasqueGLUE_qnli/80
  --benchmark BasqueGLUE_bec/80
  --benchmark BasqueGLUE_wic/80
  --benchmark BasqueGLUE_intent/80
  --benchmark LatxaEval_eusexams/80
  --benchmark LatxaEval_eusproficiency/80
  --benchmark LatxaEval_eusreading/80
  --benchmark FloresTranslation_eu_en/80
  --benchmark FloresTranslation_en_eu/80
  --benchmark FloresTranslation_eu_es/80
  --benchmark FloresTranslation_es_eu/80
)

SEEDS=(42 123 777)

# Models from smallest to largest (excluding kimu-2b which is running)
MODELS=(
  "qwen3vl-8b"
  "gpt-oss-20b"
  "qwen3.5-27b-eval"
)

for model in "${MODELS[@]}"; do
  echo "============================================"
  echo "  MODEL: $model"
  echo "============================================"

  out_prefix="eval/${model}"
  for seed in "${SEEDS[@]}"; do
    out_file="${out_prefix}_seed${seed}.json"
    echo ""
    echo "--- Seed $seed -> $out_file ---"

    python3 eval/run_eval.py \
      --model "$model" \
      --seed "$seed" \
      --out "$out_file" \
      "${BENCH_FLAGS[@]}"
  done

  echo ""
  echo "Done: $model"
  echo ""
done

echo "All pending evals complete."
