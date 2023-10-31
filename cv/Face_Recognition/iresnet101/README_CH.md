# InsightFace 在 OneFlow 中的实现


[English](README.md) **|** [简体中文](README_CH.md)

本文介绍如何在 OneFlow 中训练 InsightFace，并在验证数据集上对训练好的网络进行验证。

## 目录
- [InsightFace 在 OneFlow 中的实现](#insightface-在-oneflow-中的实现)
  - [目录](#目录)
  - [背景介绍](#背景介绍)
    - [InsightFace 开源项目](#insightface-开源项目)
    - [InsightFace 在 OneFlow 中的实现](#insightface-在-oneflow-中的实现-1)
  - [准备工作](#准备工作)
    - [安装 OneFlow](#安装-oneflow)
    - [准备数据集](#准备数据集)
      - [1. 下载数据集](#1-下载数据集)
      - [2. 将训练数据集 MS1M 从 recordio 格式转换为 OFRecord 格式](#2-将训练数据集-ms1m-从-recordio-格式转换为-ofrecord-格式)
    
  - [训练和验证](#训练和验证)
    - [训练](#训练)
    - [验证](#验证)
    - [OneFLow2ONNX](#OneFLow2ONNX)
    

## 背景介绍

###  InsightFace 开源项目

[InsightFace 原仓库](https://github.com/deepinsight/insightface)是基于 MXNet 实现的人脸识别研究开源项目。

在该项目中，集成了：

* CASIA-Webface、MS1M、VGG2 等用于人脸识别研究常用的数据集（以 MXNet 支持的二进制形式提供，可以从[这里](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)查看数据集的详细说明以及下载链接）。

* 以 ResNet、MobileFaceNet、InceptionResNet_v2 等深度学习网络作为 Backbone 的人脸识别模型。
* 涵盖 SphereFace Loss、Softmax Loss、SphereFace Loss 等多种损失函数的实现。



### InsightFace 在 OneFlow 中的实现

在 InsightFace 开源项目已有的工作基础上，OneFlow 对 InsightFace 基本的人脸识别模型进行了移植，目前已实现的功能包括：

* 支持了使用 MS1M、Glint360k 作为训练数据集，Lfw、Cfp_fp 以及 Agedb_30 作为验证数据集，提供了对网络进行训练和验证的脚本。
* 支持 ResNet100 和 MobileFaceNet 作为人脸识别模型的 Backbone 网络。
* 实现了 Softmax Loss 以及 Margin Softmax Loss（包括 Nsoftmax、Arcface、Cosface 和 Combined Loss 等）。
* 实现了模型并行和 Partial FC 优化。
* 实现了 MXNet 的模型转换。


未来将计划逐步完善：

* 更多的数据集转换。
* 更丰富的 Backbone 网络。
* 更全面的损失函数实现。
* 增加分布式运行的说明。



我们对所有的开发者开放 PR，非常欢迎您加入新的实现以及参与讨论。

## 准备工作

在开始运行前，请先确定：

1. 安装 OneFlow。
2. 准备训练和验证的 OFRecord 数据集。



###  安装 OneFlow

根据 [Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) 的步骤进行安装最新 master whl 包即可。

```
python3 -m pip install oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu102/6aa719d70119b65837b25cc5f186eb19ef2b7891/index.html --user
```

### 准备数据集

根据 [加载与准备 OFRecord 数据集](https://docs.oneflow.org/extended_topics/how_to_make_ofdataset.html) 准备 ImageNet 的 OFReocord 数据集，用以进行 InsightFace 的测试。

[InsightFace 原仓库](https://github.com/deepinsight/insightface)中提供了一系列人脸识别任务相关的数据集，已经完成了人脸对齐等预处理过程。请从[这里](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)下载相应的数据集，并且转换成 OneFlow 可以识别的 OFRecord 格式。考虑到步骤繁琐，也可以直接下载已经转好的 OFRecord 数据集：


[MS1MV3](https://oneflow-public.oss-cn-beijing.aliyuncs.com/facedata/MS1V3/oneflow/ms1m-retinaface-t1.zip)




## 训练和验证

### 训练

为了减小用户使用的迁移成本，OneFlow 的脚本已经调整为 Torch 实现的风格，用户可以使用 configs/*.py 直接修改参数。


运行脚本：

#### eager 
```
bash train.sh
```

### 验证

另外，为了方便查看保存下来的预训练模型精度，我们提供了一个仅在验证数据集上单独执行验证过程的脚本。

运行

```
bash val.sh
```

## OneFLow2ONNX

```
pip install oneflow-onnx==0.5.1
./convert.sh
```