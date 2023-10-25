export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=2
PORT=12379
MODEL_ARCH="googlenet"

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/googlenet_default_settings.yaml \
        --model_arch $MODEL_ARCH
