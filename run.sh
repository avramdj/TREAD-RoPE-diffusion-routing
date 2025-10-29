#!/bin/bash

set -e

export DISABLE_JAX_TYPING=1

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs/inet

SIT_SIZE="SiT-B/2"
SIT_SIZE_NAME=$(echo ${SIT_SIZE} | tr '/' '-')

LOG_FILE="logs/inet/$(date +%Y-%m-%d_%H-%M-%S)-${SIT_SIZE_NAME}.log"

echo "Logging to ${LOG_FILE}"

nohup uv run python "examples/train_imagenet.py" \
  --epochs 150 \
  --batch-size 256 \
  --inet-data examples/data/imagenet_int8/inet.npy \
  --inet-labels examples/data/imagenet_int8/inet.json \
  --sit-size ${SIT_SIZE} \
  --interval-unit steps \
  --log-images-every 1000 \
  --log-fid-every -1 \
  --save-every 1000 \
  --start-block 2 \
  --end-block 8 \
  --grid-n 16 \
  --wandb-name "${SIT_SIZE_NAME}" "$@" \
  --save-name "sit-b2-tread" \
  > ${LOG_FILE} 2>&1 &

disown