export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=1
PORT=12346
MODEL_ARCH="fan_large_16_p4_hybrid_in22k_1k_384"
IMAGE_SIZE=384
BATCH_SIZE=8
LEARNING_RATE=1e-4
export CUDA_VISIBLE_DEVICES=3
python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/default_settings.yaml \
        --model_arch $MODEL_ARCH \
        --batch-size $BATCH_SIZE \
        --image-size $IMAGE_SIZE \
        --lr $LEARNING_RATE  

