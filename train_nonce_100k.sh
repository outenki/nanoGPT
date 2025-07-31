#!/bin/bash
for cfg in 6-6-384 6-6-768; do
    echo "Training with config: $cfg"
    uv run python train.py data/nonce_100k/$cfg/train_config.py --device=cuda > data/nonce_100k/$cfg/train.loss
done