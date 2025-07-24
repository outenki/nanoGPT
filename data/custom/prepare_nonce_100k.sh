#!/bin/bash
uv ruhn python prepare.py \
    -dp data/input/wikitext_with_nonce_100k  \
    -dn train\
    -cn nonce \
    -lf local \
    -o data/output/nonce_100k
