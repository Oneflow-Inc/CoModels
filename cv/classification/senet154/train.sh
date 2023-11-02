export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=4
PORT=12377
MODEL_ARCH="senet154"
IMAGE_SIZE=224
BATCH_SIZE=32
LEARNING_RATE=0.005

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/senet_default_settings.yaml \
        --model_arch $MODEL_ARCH \
        --batch-size $BATCH_SIZE \
        --image-size $IMAGE_SIZE \
        --lr $LEARNING_RATE
