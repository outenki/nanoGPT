#!/bin/bash
for model_path in nonce_10k wikitext_10k; do
    echo "========== Evaluating model: $model_path ============="
    python agreement_evaluation.py \
        --model-path data/"$model_path"/ckpt.pt \
        --val-data data/evaluate_data/agreement_evaluate_data.json \
        -o data/"$model_path"
done