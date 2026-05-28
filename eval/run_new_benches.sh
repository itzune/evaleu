#!/usr/bin/env bash
# Run the 3 newest benchmarks (MMLU_eu, BertaQA_eu, MGSM_eu) for models
# that already have results on the original 13 benchmarks.
# Sorted smallest to largest. Uses --merge to add to existing files.
# Excludes: kimu-2b (running), deepseek/latxa-70b (already done),
#           qwen3vl-8b/gpt-oss-20b/qwen3.5-27b-eval (never evaluated).

set -euo pipefail
cd "$(dirname "$0")/.."

NEW_BENCHMARKS="MMLU_eu,BertaQA_eu,MGSM_eu"
SEEDS="42 123 777"

# Map: "model_api_id seed_override"
# seed_override can be "42,123,777" (all) or "42,123" (partial)
MODELS=(
  # Pending after crash:
  "qwen3.5-27b          123,777"
  "qwen3.6-27b          42,123,777"
)

echo "================================================"
echo "Running new benchmarks: ${NEW_BENCHMARKS}"
echo "Models to eval: ${#MODELS[@]}"
echo "================================================"

total=0
for entry in "${MODELS[@]}"; do
  read -r model seeds_str <<< "$entry"
  IFS=',' read -ra seeds <<< "$seeds_str"

  echo ""
  echo "=== ${model} (seeds: ${seeds_str}) ==="

  for seed in "${seeds[@]}"; do
    out_file="eval/${model}_seed${seed}.json"
    echo "  seed=${seed}  --out ${out_file}"

    python eval/run_eval.py \
      --model "${model}" \
      --seed "${seed}" \
      --out "${out_file}" \
      --merge \
      --checkpoint \
      --benchmark "${NEW_BENCHMARKS}"

    total=$((total + 1))
    echo "  done (${total} total runs so far)"
  done
done

echo ""
echo "================================================"
echo "All done. ${total} eval runs completed."
echo "Run the summarizer and site builder next:"
echo "  python eval/summarize_multiseed.py"
echo "  python site/build_site_data.py"
echo "================================================"
