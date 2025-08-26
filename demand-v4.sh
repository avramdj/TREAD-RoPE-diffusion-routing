#!/bin/bash

gcloud compute tpus queued-resources create tpu-v4-8-queued-resource \
    --node-id tpu-v4-8-node \
    --project ml-work-469710 \
    --zone us-central2-b \
    --accelerator-type v4-8 \
    --runtime-version tpu-ubuntu2204-base \
    --network my-network \
    --subnetwork my-subnet


# gcloud workstations start-tcp-tunnel \
#     --project=ml-work-469710 \
#     --region=us-central2-b \
#     --cluster=tpu-v4-8-node \
#     --config=tpu-v4-8-node \
#     --local-host-port=:1024 \
#     tpu-v4-8-node
