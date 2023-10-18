#!/bin/bash

FILE=$1

export ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT=1

python3 -m oneflow.distributed.launch \
        --nproc_per_node 1 \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 12345 \
        text_generation.py ${@:3}


export CUDA_VISIBLE_DEVICES=0

# 修改配置文件
new_text='["summarize: She is a student, She is short, She loves play football"]'

sed -i "s/text = \[\".*\"\]/text = $new_text/" inference/text_generation.py


exit 0
~                  
