#!/bin/bash

export ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT=1
export CUDA_VISIBLE_DEVICES=0

# 修改配置文件

TEXT=${TEXT:-"summarize: She is a teacher, She loves test"}
SCRIPT=${SCRIPT:-"./T5/infer.py"}
MODEL_PATH=${MODEL_PATH:-"/data/hf_models/t5-base"}
python3 -m oneflow.distributed.launch \
        --nproc_per_node 1 \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 12345 \
        text_generation.py --text "$TEXT" \
                	--t5_script "$SCRIPT" \
		       	--model_path "$MODEL_PATH" \
