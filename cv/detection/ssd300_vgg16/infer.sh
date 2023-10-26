# Using our provided weight
python train.py \
  --data-path /dataset/mscoco_2017/ \
  --dataset coco \
  --model ssd300_vgg16 \
  --batch-size 32 \
  --lr 0.002 \
  --weight-decay 0.0005 \
  --pretrained \
  --test-only

# Using your trained weight
#python train.py \
#   --data-path /dataset/mscoco_2017/ \
#   --dataset coco \
#   --model ssd300_vgg16 \
#   --batch-size 32 \
#   --lr 0.002 \
#   --weight-decay 0.0005 \
#   --load /path/weight \
#   --test-only

