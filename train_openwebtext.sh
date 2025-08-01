#!/bin/bash
for cfg in 6-6-384 12-12-768; do
    echo "Training with config: $cfg"
    /home/pj25000107/ku50001566/.local/bin/uv run python train.py data/openwebtext/$cfg/train_config.py --device=cuda > data/openwebtext/$cfg/train.loss
done