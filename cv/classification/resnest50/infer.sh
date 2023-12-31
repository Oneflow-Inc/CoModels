export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=1
PORT=12347
MODEL_ARCH="resnest50"
IMAGE_SIZE=224

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/resnest_default_settings.yaml \
        --model_arch $MODEL_ARCH \
        --image-size $IMAGE_SIZE \
        --throughput
