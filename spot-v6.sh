#!/bin/bash

gcloud compute tpus queued-resources create tpu-v6e-8-queued-resource \
    --node-id tpu-v6e-8-node \
    --project ml-work-469710 \
    --zone europe-west4-a \
    --accelerator-type v6e-8 \
    --runtime-version v2-alpha-tpuv6e \
    --network my-network \
    --subnetwork my-subnet \
    --spot


# gcloud workstations start-tcp-tunnel \
#     --project=ml-work-469710 \
#     --region=europe-west4-a \
#     --cluster=tpu-v6e-8-node \
#     --config=tpu-v6e-8-node \
#     --local-host-port=:1026 \
#     tpu-v6e-8-node
