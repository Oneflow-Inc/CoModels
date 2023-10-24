python train.py \
    -b 24 \
    --dataset coco \
    --data-path /dataset/coco \
    --model deeplabv3_resnet50_coco \
    --aux-loss \
    --lr 0.1 \
    --output-dir model/deeplabv3_resnet50_coco
    
