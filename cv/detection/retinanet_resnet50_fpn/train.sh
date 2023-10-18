python train.py \
  --data-path /dataset/mscoco_2017/ \
  --dataset coco \
  --model retinanet_resnet50_fpn \
  --batch-size 16 \
  --lr 0.01 \
  --epochs 12 \
  --lr-steps 8 11 \
  --weight-decay 0.0001 \
  --num_classes 81

