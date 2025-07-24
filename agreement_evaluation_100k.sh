#!/bin/bash
for model_path in nonce_100k wikitext_100k; do
    echo "========== Evaluating model: $model_path ============="
    uv run python agreement_evaluation.py \
        --model-path data/"$model_path"/ckpt.pt \
        --val-data data/evaluate_data/agreement_evaluate_data.json \
        -o data/"$model_path"
done