# SSD

> [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

<!-- [ALGORITHM] -->

## Abstract
We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has comparable accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. Compared to other single stage methods, SSD has much better accuracy, even with a smaller input image size. For 300×300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on a Nvidia Titan X and for 500×500 input, SSD achieves 75.1% mAP, outperforming a comparable state of the art Faster R-CNN model.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143998553-4e12f681-6025-46b4-8410-9e2e1e53a8ec.png"/>
</div>

## Model List
| Architecture| Backbone |Dataset | Box AP |  Download  |
|-------------|----------|--------|--------|--------------|
| SSD      | VGG16 | COCO |25.1 | [model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/detection/ssd/ssd300_vgg16.tar.gz)|

## Run
<details>
<summary>Training code</summary>

```Shell
cd cv/detection
bash ssd300_vgg16/train.sh
```
</details>

<details>
<summary>Inference code</summary>

```Shell
cd cv/detection
bash ssd300_vgg16/infer.sh
```
</details>

## Citation

```latex
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
}
```