export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

#GPU_NUMS = 1, 2
GPU_NUMS=1
PORT=12346
MODEL_ARCH="resnet50" 

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT  \
        train_graph.py \
        --cfg configs/cnn_inference_settings.yaml \
        --model_arch $MODEL_ARCH
