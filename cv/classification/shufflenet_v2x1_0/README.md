## Introduction
ShuffleNet is a lightweight network structure based on the channel shuffle operation, proposed by the Chinese Academy of Sciences and other units. It is designed to provide efficient deep learning models for mobile and embedded devices.

The core of ShuffleNet lies in the use of grouped convolutions and channel shuffle operations to reduce parameters and computational costs, making the model smaller and faster in computation. Grouped convolutions reduce parameters and computational costs by dividing the input channels into several small groups and then performing convolution operations on each group separately. The channel shuffle operation, on the other hand, rearranges the order of the channels so that information can flow between different groups, thereby improving the expressive power of the model.

ShuffleNet achieves good results with its design. It has a small model size, low computational cost, and maintains high accuracy. It is very suitable for running on resource-constrained devices. ShuffleNet has been applied in various fields such as image classification and object detection and has achieved good results.

## Flowvision Image Classification Project
This folder contains the classification project based on `flowvision`

## Usage
### Installation
- Clone flowvision
```bash
git clone https://github.com/Oneflow-Inc/vision.git
cd vision/projects/classification
```

- Create a conda virtual environment and activate it
```bash
conda create -n oneflow python=3.7 -y
conda activate oneflow
```

- Install the latest version of OneFlow
```bash
python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/[PLATFORM]
```

- Install other requirements
```bash
pip install -r requirements.txt
```



### Data preparation
#### ImageNet
For ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```



### Training

You can use bash script `train.sh` to train this model.

```bash
sh train.sh
```

### inference

Bash script `infer.sh` is used to infer the trained model.

```bash
sh infer.sh
```



