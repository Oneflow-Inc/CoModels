export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=4
PORT=12346
MODEL_ARCH="deit_tiny_patch16_224"
BATCH_SIZE=256
LEARNING_RATE=5e-5

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/default_settings.yaml \
        --model_arch $MODEL_ARCH \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE  