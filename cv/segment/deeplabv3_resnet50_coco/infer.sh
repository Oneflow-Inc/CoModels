 python train.py \
    --dataset coco \
    -b 24 \
    --model deeplabv3_resnet50_coco \
    --aux-loss \
    --data-path /dataset/coco \
    --lr 0.12 \
    --pretrained \
    --test-only
    
