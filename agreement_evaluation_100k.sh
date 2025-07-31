#!/bin/bash
for model_path in nonce_100k wikitext_100k; do
    for cfg in 6-6-384 6-6-768; do
        echo "========== Evaluating model: $model_path/$cfg ============="
        uv run python agreement_evaluation.py \
            --model-path data/"$model_path"/$cfg/ckpt.pt \
            --model-type pt \
            --val-data data/evaluate_data/agreement_evaluate_data.json \
            -o data/"$model_path"/$cfg
    done
done