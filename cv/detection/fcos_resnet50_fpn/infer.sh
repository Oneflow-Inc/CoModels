# Using our provided weight
python train.py \
 --data-path /dataset/mscoco_2017/ \
 --dataset coco \
 --model fcos_resnet50_fpn_ \
 --batch-size 4 \
 --pretrained \
 --test-only

# Using your trained weight
# python train.py \
#    --data-path /dataset/mscoco_2017/ \
#    --dataset coco \
#    --model fcos_resnet50_fpn_ \
#    --batch-size 32 \
#    --load /path/weight \
#    --test-only

