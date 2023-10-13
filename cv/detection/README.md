# Installation
First install OneFlow, please refer to [install-oneflow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) for more details.

Then install the dependency
```Shell
pip install -r requirement.txt
```

## Model Zoo
| Architecture| Backbone |Dataset | Box AP |  Download  |
|-------------|----------|--------|--------|--------------|
| SSD      | VGG16 | COCO |25.1 | [model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/detection/ssd/ssd300_vgg16.tar.gz)|
| Faster-RCNN      | ResNet50 | COCO |36.9 | [model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/detection/faster_rcnn/fasterrcnn_resnet50_fpn_coco.tar.gz)|
| Faster-RCNN      | MobileNetV3_Large | COCO |32.8 | [model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/detection/faster_rcnn/fasterrcnn_mobilenet_v3_large_fpn.tar.gz)|

## Run
<details>
<summary>Training code</summary>
We take the ssd as an example to show how to train the model.

```Shell
cd cv/detection
bash ssd300_vgg16/train.sh
```
</details>

<details>
<summary>Inference code</summary>
We take the ssd as an example to show how to test the model.

```Shell
cd cv/detection
bash ssd300_vgg16/infer.sh
```
</details>
