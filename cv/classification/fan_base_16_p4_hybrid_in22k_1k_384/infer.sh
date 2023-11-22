export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=1
PORT=12346
MODEL_ARCH="fan_base_16_p4_hybrid_in22k_1k_384"

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/default_settings.yaml \
        --model_arch $MODEL_ARCH \
        --lr 1e-2 \
        --throughput

