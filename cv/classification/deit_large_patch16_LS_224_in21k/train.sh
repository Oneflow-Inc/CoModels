export PYTHONPATH=$PWD:$PYTHONPATH
set -aux
CUDA_VISIBLE_DEVICES=2,3,5,7
GPU_NUMS=4
PORT=12346
MODEL_ARCH="deit_large_patch16_LS_224_in21k"

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/default_settings.yaml \
        --model_arch $MODEL_ARCH    