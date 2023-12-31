# Using our provided weight
python train.py \
  --data-path /dataset/mscoco_2017/ \
  --dataset coco \
  --model fasterrcnn_resnet50_fpn \
  --batch-size 4 \
  --pretrained \
  --test-only

# Using your trained weight
#python train.py \
#   --data-path /dataset/mscoco_2017/ \
#   --dataset coco \
#   --model fasterrcnn_resnet50_fpn \
#   --batch-size 4 \
#   --load /path/weight \
#   --test-only

