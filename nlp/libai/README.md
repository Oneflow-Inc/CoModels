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
### How to Train T5 Model with Parallelism

We provide `train.sh` for execute training. Before invoking the script, perform the following steps.
<details>
<summary>Training code</summary>
We take the T5 as an example to show how to train the model.

```Shell
bash T5/train.sh
```
</details>
<details>
<summary>Inference code</summary>
We take the T5 as an example to show how to test the model.

```Shell
bash T5/infer.sh
```
</details>

