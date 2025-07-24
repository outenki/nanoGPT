#!/bin/bash
uv run python prepare.py \
    -dp data/input/wikitext_with_nonce_100k  \
    -dn train\
    -cn text \
    -lf local \
    -o data/output/wikitext_100k
