<!-- 配图 -->

<h2 align="center">LiBai</h2>
<p align="center">
    <a href="https://libai.readthedocs.io/en/latest/index.html">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/Oneflow-Inc/libai.svg?color=blue">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/Oneflow-Inc/libai.svg">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/issues">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/issues">
        <img alt="Python Checks" src="https://github.com/Oneflow-Inc/libai/workflows/Python checks/badge.svg">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/issues">
        <img alt="Docs Release Status" src="https://github.com/Oneflow-Inc/libai/workflows/Document Release/badge.svg">
    </a>
</p>


## Introduction

**English** | [简体中文](/README_zh-CN.md)

LiBai is a large-scale open-source model training toolbox based on OneFlow. The main branch works with OneFlow 0.7.0.

<details open>
<summary> <b> Highlights </b> </summary>

- **Support a collection of parallel training components**

    LiBai provides multiple parallelisms such as Data Parallelism, Tensor Parallelism, and Pipeline Parallelism. It's also extensible for other new parallelisms.

- **Varied training techniques**

    LiBai provides many out-of-the-box training techniques such as Distributed Training, Mixed Precision Training, Activation Checkpointing, Recomputation, Gradient Accumulation, and Zero Redundancy Optimizer(ZeRO).

- **Support for both CV and NLP tasks**

    LiBai has predefined data process for both CV and NLP datasets such as CIFAR, ImageNet, and BERT Dataset.

- **Easy to use**

    LiBai's components are designed to be modular for easier usage as follows:
    - LazyConfig system for more flexible syntax and no predefined structures 
    - Friendly trainer and engine
    - Used as a library to support building research projects on it. See [projects/](/projects) for some projects that are built based on LiBai

- **High Efficiency**

</details>

## Installation

See [Installation instructions](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html).

## Getting Started
### Prepare the Data and the Vocab

- We have prepared relevant datasets, which can be downloaded from the following links:

1. [VOCAB_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt)
2. [BIN_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin)
3. [IDX_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx)

- Download the dataset and move the data file to the folder. The file structure should be like:
```bash
$ tree data
path/to/bert_data
├── bert-base-chinese-vocab.txt
├── loss_compara_content_sentence.bin
└── loss_compara_content_sentence.idx
```
### How to Train Bert_large Model with Parallelism

We provide `train.sh` for execute training. Before invoking the script, perform the following steps.

**Step 1. Set data path and vocab path**

- Update the data path and vocab path in [T5/train.sh]

```shell
sed -i 's#vocab_file = ".*"#vocab_file = "/data/dataset/bert_data/bert-base-chinese-vocab.txt"#' configs/t5_large_pretrain.py
sed -i 's#data_prefix = ".*"#data_prefix = "/data/dataset/bert_data/loss_compara_content_sentence"#' configs/t5_large_pretrain.py
```

<details>
<summary>Training code</summary>
We take the T5 as an example to show how to train the model.

```Shell
bash T5/train.sh
```
</details>

