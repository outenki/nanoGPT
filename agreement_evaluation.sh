#!/bin/bash
for model_path in nonce_10k nonce_100k wikitext_10k wikitext_100k; do
    for cfg in 6-6-384 12-12-768; do
        echo "========== Evaluating model: $model_path/$cfg ============="
        uv run python agreement_evaluation.py \
            --model-path data/"$model_path"/$cfg/ckpt.pt \
            --model-type pt \
            --val-data data/evaluate_data/agreement_evaluate_data.json \
            -o data/"$model_path"/$cfg
    done
done


model_path="openwebtext"
echo "========== Evaluating model: $model_path ============="
uv run python agreement_evaluation.py \
    --model-path data/"$model_path"/nanogpt-openwebtext.safetensors \
    --model-type st \
    --val-data data/evaluate_data/agreement_evaluate_data.json \
    -o data/"$model_path"


model_path="gpt2"
echo "========== Evaluating model: $model_path ============="
uv run python agreement_evaluation.py \
    --model-path "openai-community/gpt2" \
    --model-type hf \
    --val-data data/evaluate_data/agreement_evaluate_data.json \
    -o data/"$model_path"
