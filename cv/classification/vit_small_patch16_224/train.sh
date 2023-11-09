export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=2
PORT=12346
MODEL_ARCH="vit_small_patch16_224"
BATCH_SIZE=256
LEARNING_RATE=3e-4

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/vit_settings.yaml \
        --model_arch $MODEL_ARCH \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE 

