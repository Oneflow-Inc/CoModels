#!/usr/bin/env bash

CONFIG=${2:-"configs/t5_large_pretrain.py"}

sed -i 's#vocab_file = ".*"#vocab_file = "/data/dataset/bert_data/bert-base-chinese-vocab.txt"#' configs/t5_large_pretrain.py
sed -i 's#data_prefix = ".*"#data_prefix = "/data/dataset/bert_data/loss_compara_content_sentence"#' configs/t5_large_pretrain.py

export ONEFLOW_FUSE_OPTIMIZER_UPDATE_CAST=true

python3 -m oneflow.distributed.launch \
        --nproc_per_node 1 \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1\
        --master_port 12345 \
        train_net.py \
	--config-file $CONFIG  ${@:4}

