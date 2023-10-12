#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=6

# 修改配置文件
sed -i 's#vocab_file = ".*"#vocab_file = "/data/home/lifei/data/bert_data/bert-base-chinese-vocab.txt"#' configs/t5_large_pretrain.py
sed -i 's#data_prefix = ".*"#data_prefix = "/data/home/lifei/data/bert_data/loss_compara_content_sentence"#' configs/t5_large_pretrain.py

# 执行训练命令
bash tools/train.sh tools/train_net.py configs/t5_large_pretrain.py 1

exit 0
