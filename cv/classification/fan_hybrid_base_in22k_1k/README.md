## fan_hybrid_base_in22k_1k

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

