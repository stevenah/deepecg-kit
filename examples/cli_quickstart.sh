#!/usr/bin/env bash
set -euo pipefail

deepecg list-models
echo ""
deepecg list-datasets
echo ""

deepecg info -m kanres

deepecg train -m kanres -d af-classification \
    --epochs 5 \
    --batch-size 64 \
    --download

CHECKPOINT=$(find runs/ -name "*.ckpt" -type f | sort | tail -1)
echo "Using checkpoint: $CHECKPOINT"

deepecg evaluate \
    --checkpoint "$CHECKPOINT" \
    -d af-classification \
    --split test

deepecg resume \
    --checkpoint "$CHECKPOINT" \
    --epochs 3
