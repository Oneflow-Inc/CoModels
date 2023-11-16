## Introduction
MNASNet (Mobile Neural Architecture Search Network) is a convolutional neural network architecture developed by Google researchers, specifically designed for efficient image recognition on mobile devices. MNASNet is an automated neural network architecture search (NAS) method that uses reinforcement learning to automatically search for the optimal network architecture, achieving the best image recognition performance while maintaining efficient use of computational resources.

MNASNet utilizes a multi-objective optimization approach, considering both the accuracy of the model and the latency, thereby striking a balance between model accuracy and execution speed. This enables MNASNet to run efficiently on mobile devices while maintaining high image recognition accuracy. The design philosophy and methodology of MNASNet provide an effective solution for image recognition and other computation-intensive tasks on mobile devices.
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
python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/cu102
```

- Install other requirements
```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
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


