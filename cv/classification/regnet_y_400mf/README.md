## RegNet
RegNet is a neural network architecture designed for deep learning tasks, particularly in the field of computer vision. It was introduced in the paper titled "Designing Network Design Spaces" by Touvron et al. in 2020. RegNet is known for its efficiency, scalability, and strong performance on various computer vision tasks, making it a popular choice for researchers and practitioners in the field.
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

