# Installation
First install OneFlow, please refer to [install-oneflow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) for more details.

Then install the dependency
```Shell
pip install -r requirement.txt
```

## Model Zoo
| Architecture| Backbone |Dataset | mIOU |  Download  |
|-------------|----------|--------|--------|--------------|
| FCN      | ResNet50 | COCO |60.5 | [model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/segmentation/FCN/fcn_resnet50_coco.zip)|
| FCN      | ResNet101 | COCO |63.7 | [model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/segmentation/FCN/fcn_resnet101_coco.zip)|


## Run
<details>
<summary>Training code</summary>
We take the fcn as an example to show how to train the model.

```Shell
cd cv/detection
bash fcn_resnet50_coco/train.sh
```
</details>

<details>
<summary>Inference code</summary>
We take the fcn as an example to show how to test the model.

```Shell
cd cv/detection
bash fcn_resnet50_coco/infer.sh
```
</details>
