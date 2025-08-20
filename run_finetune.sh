#!/bin/bash

set -e

export DISABLE_JAX_TYPING=1

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs/inet

DIT_SIZE="DiT-B/2"
DIT_SIZE_NAME=$(echo ${DIT_SIZE} | tr '/' '-')

LOG_FILE="logs/inet/$(date +%Y-%m-%d_%H-%M-%S)-${DIT_SIZE_NAME}-finetune.log"

echo "Logging to ${LOG_FILE}"

nohup uv run python "examples/train_imagenet.py" \
  --epochs 50 \
  --batch-size 128 \
  --grad-accum 2 \
  --lr 1e-5 \
  --inet-data examples/data/imagenet_int8/inet.npy \
  --inet-labels examples/data/imagenet_int8/inet.json \
  --dit-size ${DIT_SIZE} \
  --interval-unit steps \
  --log-images-every 1000 \
  --log-fid-every -1 \
  --save-every 1000 \
  --start-block -1 \
  --end-block -1 \
  --grid-n 16 \
  --wandb-name "${DIT_SIZE_NAME}-finetune" "$@" \
  --save-name "dit-b2-tread-ft" \
  --ckpt examples/checkpoints/dit-b2-tread.pt \
  > ${LOG_FILE} 2>&1 &

disown