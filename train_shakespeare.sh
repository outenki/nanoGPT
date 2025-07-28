#!/bin/bash
uv run python train.py data/shakespeare/train_config.py --device=cuda > data/shakespeare/train.loss