#!/bin/bash

set -e

mkdir -p logs/inet
export DISABLE_JAX_TYPING=1
export NUM_GPUS=$(nvidia-smi -L | wc -l)

SIT_SIZE="SiT-L/2"
SIT_SIZE_NAME=$(echo ${SIT_SIZE} | tr '/' '-')
SAVE_NAME=${SIT_SIZE_NAME}-tread-ggrope

# tread rope
# export CUDA_VISIBLE_DEVICES=0

LOG_FILE="logs/inet/$(date +%Y-%m-%d_%H-%M-%S)-${SAVE_NAME}.log"

echo "Logging to ${LOG_FILE}"

nohup uv run torchrun --nproc_per_node=$NUM_GPUS "examples/train_imagenet_ddp.py" \
  --epochs 150 \
  --batch-size 64 \
  --grad-accum 4 \
  --lr 1e-4 \
  --inet-data examples/data/imagenet_int8/inet.npy \
  --inet-labels examples/data/imagenet_int8/inet.json \
  --sit-size ${SIT_SIZE} \
  --interval-unit steps \
  --log-images-every 2000 \
  --log-fid-every -1 \
  --save-every 2000 \
  --start-block 2 \
  --end-block 20 \
  --grid-n 16 \
  --wandb-name ${SAVE_NAME} \
  --save-name ${SAVE_NAME} \
  --rope golden_gate \
  "$@" \
  > ${LOG_FILE} 2>&1 &

disown


# # tread no rope
# export CUDA_VISIBLE_DEVICES=1

# LOG_FILE="logs/inet/$(date +%Y-%m-%d_%H-%M-%S)-${SIT_SIZE_NAME}-norope.log"

# echo "Logging to ${LOG_FILE}"

# nohup uv run python "examples/train_imagenet.py" \
#   --epochs 150 \
#   --batch-size 128 \
#   --grad-accum 2 \
#   --inet-data examples/data/imagenet_int8/inet.npy \
#   --inet-labels examples/data/imagenet_int8/inet.json \
#   --sit-size ${SIT_SIZE} \
#   --interval-unit steps \
#   --log-images-every 2000 \
#   --log-fid-every -1 \
#   --save-every 2000 \
#   --start-block 2 \
#   --end-block 8 \
#   --grid-n 16 \
#   --wandb-name "${SIT_SIZE_NAME}-tread" "$@" \
#   --save-name "sit-b2-tread" \
#   > ${LOG_FILE} 2>&1 &

# disown

# # baseline
# export CUDA_VISIBLE_DEVICES=1

# LOG_FILE="logs/inet/$(date +%Y-%m-%d_%H-%M-%S)-${SIT_SIZE_NAME}-base.log"

# echo "Logging to ${LOG_FILE}"

# nohup uv run python "examples/train_imagenet.py" \
#   --epochs 150 \
#   --batch-size 128 \
#   --grad-accum 2 \
#   --inet-data examples/data/imagenet_int8/inet.npy \
#   --inet-labels examples/data/imagenet_int8/inet.json \
#   --sit-size ${SIT_SIZE} \
#   --interval-unit steps \
#   --log-images-every 2000 \
#   --log-fid-every -1 \
#   --save-every 2000 \
#   --start-block -1 \
#   --end-block -1 \
#   --grid-n 16 \
#   --wandb-name "${SIT_SIZE_NAME}-base-rope" "$@" \
#   --save-name "sit-b2-base-rope" \
#   > ${LOG_FILE} 2>&1 &

# disown