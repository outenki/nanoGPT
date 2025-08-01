#!/bin/bash
model_path="openwebtext"
echo "========== Evaluating model: $model_path ============="
uv run python agreement_evaluation.py \
    --model-path data/"$model_path"/nanogpt-openwebtext.safetensors \
    --model-type st \
    --val-data data/evaluate_data/agreement_evaluate_data.json \
    -o data/"$model_path"