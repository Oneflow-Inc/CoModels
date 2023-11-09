## MNAsNet
convolutional neural network architecture developed by Google researchers, specifically designed for efficient image recognition on mobile devices. MNASNet is an automated neural network architecture search (NAS) method that uses reinforcement learning to automatically search for the optimal network architecture, achieving the best image recognition performance while maintaining efficient use of computational resources.

MNASNet utilizes a multi-objective optimization approach, considering both the accuracy of the model and the latency, thereby striking a balance between model accuracy and execution speed. This enables MNASNet to run efficiently on mobile devices while maintaining high image recognition accuracy. The design philosophy and methodology of MNASNet provide an effective solution for image recognition and other computation-intensive tasks on mobile devices.

### Installation
- Install the latest version of OneFlow
```bash
python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]
```
Find more information on [install oneflow](https://github.com/Oneflow-Inc/oneflow#install-oneflow)

- Install flowvision

Then install the latest stable release of flowvision

```bash
pip install flowvision==0.2.1
```

- Install other requirements
```bash
python3 -m pip install -r requirements.txt
```

### Dataset
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