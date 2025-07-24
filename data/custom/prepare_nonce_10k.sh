#!/bin/bash
uv run python prepare.py \
    -dp data/input/wikitext_with_nonce_10k  \
    -dn train\
    -cn nonce \
    -lf local \
    -o data/output/nonce_10k
