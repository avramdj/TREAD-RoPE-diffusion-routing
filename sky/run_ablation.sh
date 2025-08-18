#!/bin/bash

set -e
# set -x 

SCRIPT_DIR=$(dirname "$(realpath "$0")")

source "$SCRIPT_DIR/../.env"

if [ -z "$WANDB_API_KEY" ]; then
  echo "WANDB_API_KEY is not set"
  exit 1
fi

export WANDB_API_KEY=$WANDB_API_KEY

for conf in imagenet-int8-b2-train ; do
    echo "Launching $conf"
    sky launch -c $conf --secret WANDB_API_KEY $SCRIPT_DIR/$conf.yaml -y --down -d "$@" &
done