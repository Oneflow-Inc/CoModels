 python train.py \
    -b 24 \
    --dataset coco \
    --data-path /dataset/coco \
    --model deeplabv3_mobilenet_v3_large_coco \
    --aux-loss \
    --lr 0.12 \
    --pretrained \
    --test-only
    
