 python train.py \
    -b 24 \
    --dataset coco \
    --data-path /dataset/coco \
    --model fcn_resnet101_coco \
    --aux-loss \
    --lr 0.12 \
    --pretrained \
    --test-only
