export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=2
PORT=12388
MODEL_ARCH="resnest101"
IMAGE_SIZE=256
BATCH_SIZE=16

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/resnest_default_settings.yaml \
        --model_arch $MODEL_ARCH \
        --batch_size $BATCH_SIZE \
        --image_size $IMAGE_SIZE
