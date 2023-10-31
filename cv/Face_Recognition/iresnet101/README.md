
# InsightFace in OneFlow

[English](README.md) **|** [简体中文](README_CH.md)

It introduces how to train InsightFace in OneFlow, and do verification over the validation datasets via the well-toned networks.

## Contents

\- [InsightFace in OneFlow](#insightface-in-oneflow)

 \- [Contents](#contents)

 \- [Background](#background)

  \- [InsightFace opensource project](#insightface-opensource-project)

  \- [Implementation in OneFlow](#implementation-in-oneflow)

 \- [Preparations](#preparations)

  \- [Install OneFlow](#install-oneflow)

  \- [Data preparations](#data-preparations)

   \- [1. Download datasets](#1-download-datasets)


 \- [Training and verification](#training-and-verification)

  \- [Training](#training)

  \- [OneFLow2ONNX](#OneFLow2ONNX)

## Background

### InsightFace opensource project

[InsightFace](https://github.com/deepinsight/insightface) is an open-source 2D&3D deep face analysis toolbox, mainly based on MXNet.

In InsightFace, it supports:



- Datasets typically used for face recognition, such as CASIA-Webface、MS1M、VGG2(Provided with the form of a binary file which could run in MXNet, [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) is more details about the datasets and how to download.



* Backbones of ResNet, MobilefaceNet, InceptionResNet_v2, and other deep-learning networks to apply in facial recognition. 

* Implementation of different loss functions, including SphereFace Loss、Softmax Loss、SphereFace Loss, etc.

  

### Implementation in OneFlow

Based upon the currently existing work of Insightface, OneFlow ported basic models from it, and now OneFlow supports:



- Training datasets of MS1M、Glint360k, and validation datasets of Lfw、Cfp_fp and Agedb_30, scripts for training and validating.

- Backbones of ResNet100 and MobileFaceNet to recognize faces.

- Loss function, e.g. Softmax Loss and Margin Softmax Loss（including Arcface、Cosface and Combined Loss）.

- Model parallelism and [Partial FC](https://github.com/deepinsight/insightface/tree/760d6de043d7f654c5963391271f215dab461547/recognition/partial_fc#partial-fc) optimization.

- Model transformation via MXNet.



To be coming further:

- Additional datasets transformation.

- Plentiful backbones.

- Full-scale loss functions implementation.

- Incremental tutorial on the distributed configuration.



This project is open for every developer to PR, new implementation and animated discussion will be most welcome.



## Preparations

First of all, before execution, please make sure that:

1. Install OneFlow

2. Prepare training and validation datasets in form of OFRecord.



### Install OneFlow



According to steps in [Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) install the newest release master whl packages.

```
python3 -m pip install oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu102/6aa719d70119b65837b25cc5f186eb19ef2b7891/index.html --user
```



### Data preparations

According to [Load and Prepare OFRecord Datasets](https://docs.oneflow.org/en/extended_topics/how_to_make_ofdataset.html), datasets should be converted into the form of OFREcord, to test InsightFace.



It has provided a set of datasets related to face recognition tasks, which have been pre-processed via face alignment or other processions already in [InsightFace](https://github.com/deepinsight/insightface). The corresponding datasets could be downloaded from [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) and should be converted into OFRecord, which performs better in OneFlow. Considering the cumbersome steps, it is suggested to download converted OFrecord datasets:


[MS1MV3](https://oneflow-public.oss-cn-beijing.aliyuncs.com/facedata/MS1V3/oneflow/ms1m-retinaface-t1.zip)




## Training and verification



### Training

To reduce the usage cost of user, OneFlow draws close the scripts to Torch style, you can directly modify parameters via configs/*.py

#### eager 
```
bash train.sh
```


### Varification

Moreover, OneFlow offers a validation script to do verification separately, val.py, which facilitates you to check the precision of the pre-training model saved.

```
bash val.sh
```
## OneFLow2ONNX

```
pip install oneflow-onnx==0.5.1
./convert.sh
```
