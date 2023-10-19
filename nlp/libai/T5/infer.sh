#!/bin/bash

FILE=$1

export ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT=1
export CUDA_VISIBLE_DEVICES=0

# 修改配置文件


python3 -m oneflow.distributed.launch \
        --nproc_per_node 1 \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 12345 \
        text_generation.py --text "summarize: She is a student, She is smart, She loves test" ${@:3}
exit 0
