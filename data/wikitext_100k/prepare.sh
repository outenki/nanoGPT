#!/bin/bash
uv run python ../custom/prepare.py \
    -dp ../input/wikitext_with_nonce_100k  \
    -dn train\
    -cn text \
    -lf local \
    -o .
