#!/bin/bash

set -e

export DISABLE_JAX_TYPING=1

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs/inet

DIT_SIZE="DiT-B/2"
DIT_SIZE_NAME=$(echo ${DIT_SIZE} | tr '/' '-')

LOG_FILE="logs/inet/$(date +%Y-%m-%d_%H-%M-%S)-${DIT_SIZE_NAME}.log"

echo "Logging to ${LOG_FILE}"

nohup uv run python "examples/train_imagenet.py" \
  --epochs 150 \
  --batch-size 256 \
  --inet-data examples/data/imagenet_int8/inet.npy \
  --inet-labels examples/data/imagenet_int8/inet.json \
  --dit-size ${DIT_SIZE} \
  --interval-unit steps \
  --log-images-every 1000 \
  --log-fid-every -1 \
  --save-every 1000 \
  --start-block 2 \
  --end-block 8 \
  --grid-n 9 \
  --wandb-name "${DIT_SIZE_NAME}" "$@" \
  > ${LOG_FILE} 2>&1 &

disown