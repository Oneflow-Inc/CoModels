export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=1
PORT=12357
MODEL_ARCH="resnest101"
IMAGE_SIZE=256
BATCH_SIZE=32

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/resnest_default_settings.yaml \
        --model_arch $MODEL_ARCH \
        --image_size $IMAGE_SIZE \
        --batch_size $BATCH_SIZE \
        --throughput
