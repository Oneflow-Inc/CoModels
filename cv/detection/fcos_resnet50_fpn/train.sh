python train.py \
  --data-path /dataset/mscoco_2017/ \
  --dataset coco \
  --model fcos_resnet50_fpn_ \
  --batch-size 8 \
  --lr 0.04 \
  --epochs 12 \
  --lr-steps 8 11 \
  --workers 32 \
  
