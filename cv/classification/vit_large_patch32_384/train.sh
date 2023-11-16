export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=2
PORT=12346
MODEL_ARCH="vit_large_patch32_384"
IMAGE_SIZE=384
BATCH_SIZE=64
LEARNING_RATE=3e-5
export CUDA_VISIBLE_DEVICES=6,7
python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/vit_settings.yaml \
        --model_arch $MODEL_ARCH \
        --batch-size $BATCH_SIZE \
        --image-size $IMAGE_SIZE \
        --lr $LEARNING_RATE 

