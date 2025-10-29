#!/bin/bash

set -e

export DISABLE_JAX_TYPING=1

export CUDA_VISIBLE_DEVICES=1

mkdir -p logs/mnist

SIT_SIZE="SiT-S/2"
SIT_SIZE_NAME=$(echo ${SIT_SIZE} | tr '/' '-')

LOG_FILE="logs/mnist/$(date +%Y-%m-%d_%H-%M-%S)-${SIT_SIZE_NAME}.log"

echo "Logging to ${LOG_FILE}"

nohup uv run python "examples/train_mnist.py" \
  --epochs 150 \
  --batch-size 256 \
  --sit-size ${SIT_SIZE} \
  --interval-unit steps \
  --log-images-every 1000 \
  --log-fid-every -1 \
  --save-every 1000 \
  --start-block 2 \
  --end-block 8 \
  --grid-n 9 \
  --wandb-name "mnist-${SIT_SIZE_NAME}" "$@" \
  > ${LOG_FILE} 2>&1 &
  # --lr 2e-5 \

disown