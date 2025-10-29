#!/bin/bash

set -e

export DISABLE_JAX_TYPING=1

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs/inet

SIT_SIZE="SiT-L/2"
SIT_SIZE_NAME=$(echo ${SIT_SIZE} | tr '/' '-')

LOG_FILE="logs/inet/$(date +%Y-%m-%d_%H-%M-%S)-${SIT_SIZE_NAME}-ft.log"

echo "Logging to ${LOG_FILE}"

nohup uv run python "examples/train_imagenet.py" \
  --epochs 50 \
  --batch-size 64 \
  --grad-accum 2 \
  --lr 1e-5 \
  --inet-data examples/data/imagenet_int8/inet.npy \
  --inet-labels examples/data/imagenet_int8/inet.json \
  --sit-size ${SIT_SIZE} \
  --interval-unit steps \
  --log-images-every 1000 \
  --log-fid-every -1 \
  --save-every 1000 \
  --start-block -1 \
  --end-block -1 \
  --grid-n 16 \
  --wandb-name "${SIT_SIZE_NAME}-ft" "$@" \
  --save-name "SiT-L-2-tread-ft" \
  --ckpt examples/checkpoints/SiT-L-2-tread.pt \
  > ${LOG_FILE} 2>&1 &

disown