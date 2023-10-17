#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 修改配置文件
config_file="inference/text_generation.py"
new_text='["summarize: She is a student, She is short, She loves play football"]'

new_data_parallel=1,
new_tensor_parallel=1,
new_pipeline_parallel=1,
new_pipeline_stage_id='[0] * 12 + [1] * 12',
new_pipeline_num_layers='12 * 2',
new_mode='huggingface'
new_model_path='/data/hf_models/t5-base'

sed -i 's#model_path = ".*"#model_path  = "/data/hf_models/t5-base"#' $config_file

#sed -i "s/data_parallel=[0-9]\+/data_parallel=$new_data_parallel/" $config_file
#sed -i "s/tensor_parallel=[0-9]\+/tensor_parallel=$new_tensor_parallel/" $config_file
#sed -i "s/pipeline_parallel=[0-9]\+/pipeline_parallel=$new_pipeline_parallel/" $config_file
#sed -i "s/pipeline_stage_id=[0-9]\+/pipeline_stage_id=$new_pipeline_stage_id/" $config_file
#sed -i "s/pipeline_num_layers=[0-9]\+/pipeline_num_layers=$new_pipeline_num_layers/" $config_file
#sed -i "s/model_path= \[\".*\"\]/model_path=$new_model_path/" $config_file
sed -i "s/mode=\"[^\"]*\"/mode=\"$new_mode\"/" $config_file
sed -i "s/text = \[\".*\"\]/text = $new_text/" $config_file





#执行训练命令
bash tools/infer.sh inference/text_generation.py 1

exit 0
