python train.py \
  --data-path /dataset/mscoco_2017/ \
  --dataset coco \
  --model fasterrcnn_mobilenet_v3_large_320_fpn \
  --batch-size 4 \
  --lr 0.005 \
  --epochs 12 \
  --lr-steps 8 11
