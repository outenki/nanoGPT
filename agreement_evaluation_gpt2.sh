#!/bin/bash
model_path="gpt2"
echo "========== Evaluating model: $model_path ============="
uv run python agreement_evaluation.py \
    --model-path "openai-community/gpt2" \
    --model-type hf \
    --val-data data/evaluate_data/agreement_evaluate_data.json \
    -o data/"$model_path"