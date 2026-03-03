#!/usr/bin/env bash
set -euo pipefail

deepecg train -m kanres -d af-classification \
    --weights kanres-af-30s \
    --epochs 20 \
    --lr 1e-4 \
    --batch-size 64 \
    --download

deepecg train -m resnet -d af-classification \
    --epochs 30 \
    --output-dir runs/resnet-experiment \
    --download

deepecg -v train -m lstm -d af-classification --epochs 5 --download

for model in simple-cnn kanres resnet lstm; do
    echo "=== Training $model ==="
    deepecg train -m "$model" -d af-classification \
        --epochs 10 \
        --seed 42 \
        --download
done

CHECKPOINT=$(find runs/ -name "*.ckpt" -type f | sort | tail -1)

deepecg evaluate \
    --checkpoint "$CHECKPOINT" \
    -d af-classification \
    --split val

deepecg evaluate \
    --checkpoint "$CHECKPOINT" \
    -d af-classification \
    --split test
