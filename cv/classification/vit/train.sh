export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=8
PORT=12346
MODEL_ARCH="vit_base_patch16_224"

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/vit_settings.yaml \
        --model_arch $MODEL_ARCH 

