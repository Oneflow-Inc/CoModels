export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=8
PORT=12346
MODEL_ARCH="efficientnet_b2"

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/default_settings.yaml \
        --model_arch $MODEL_ARCH 

