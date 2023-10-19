#!/usr/bin/env bash
  
export ONEFLOW_FUSE_OPTIMIZER_UPDATE_CAST=true

python3 -m oneflow.distributed.launch \
        --nproc_per_node 1 \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1\
        --master_port 12345 \
        train.py \
        --config-file T5/t5_large_pretrain.py \

